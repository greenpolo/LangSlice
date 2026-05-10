"""Apply processor.apply_chat_template and build labels with -100 outside assistant turns."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import torch

from .render import RenderedExample

if TYPE_CHECKING:
    # Import-only for type hints; keeping the runtime path import-free means
    # the unit tests for collate.py do not require the embeddings package
    # (and its transitive rlvr.atlas_grid import) to be on sys.path.
    from embeddings.cache import AtlasEmbeddingCache


class LangSliceCollator:
    """Builds a TRL-compatible batch from RenderedExample objects.

    The processor's chat template is the source of truth for the assistant-token
    mask. Labels are constructed by cloning input_ids and zeroing (with -100)
    every position where the assistant_mask is False.

    Atlas-embedding cache integration
    ---------------------------------

    When ``atlas_cache`` is provided, the collator inspects every tool-result
    image path on each example and counts how many of them snap to a cached
    embedding. The hit/miss ratio is exposed via :meth:`cache_hit_rate` and
    drives the Phase 1 measurement: ship the splice only if the hit rate
    clears 50% on the real corpus.

    The Phase 2 splice itself is gated by ``enable_splice`` — when False (the
    default) the cache is used purely for measurement. When True, cached
    images are still rendered through the processor (so the chat template
    sees the right image-token count) but the per-image precomputed
    embeddings are returned alongside the batch under
    ``precomputed_image_embeddings`` and a positional mask
    ``precomputed_image_mask`` so a vision-tower forward pre-hook can splice
    them in without re-running SigLIP. The trainer is responsible for wiring
    the hook (see ``embeddings.splice_hook``); the collator only emits the
    sidecar.
    """

    def __init__(
        self,
        *,
        processor: Any,
        max_seq_length: int,
        atlas_cache: AtlasEmbeddingCache | None = None,
        enable_splice: bool = False,
    ) -> None:
        if processor.tokenizer.pad_token_id is None:
            raise RuntimeError(
                "processor.tokenizer.pad_token_id is None — set it (e.g. to "
                "eos_token_id) before constructing LangSliceCollator. Padding a "
                "batch of variable-length examples requires a pad token."
            )
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.atlas_cache = atlas_cache
        self.enable_splice = enable_splice
        if enable_splice and atlas_cache is None:
            raise ValueError(
                "enable_splice=True requires atlas_cache; passing splice without a "
                "cache would mark every image as a miss and the forward hook "
                "would never fire."
            )
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def cache_hit_rate(self) -> float:
        """Return cumulative cache hit fraction since construction.

        Returns 0.0 if the collator has not yet seen any atlas-grid image
        paths (e.g. no batches consumed, or no atlas_cache provided).
        """
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return 0.0
        return self._cache_hits / total

    def cache_counters(self) -> dict[str, int]:
        """Return current ``{'hits': X, 'misses': Y}`` snapshot for logging."""
        return {"hits": self._cache_hits, "misses": self._cache_misses}

    def _account_atlas_paths(
        self, example: RenderedExample
    ) -> list[torch.Tensor | None]:
        """Bump hit/miss counters from ``example.image_paths`` and return per-image embeddings.

        The returned list is aligned with the per-example pixel_values dim-0
        ordering: query images first, then tool-result images in trace
        order. Entry ``i`` is the cached embedding tensor when the path at
        index ``i`` snaps to a cache entry, else None.

        Query images never match the atlas-grid pattern (they live under
        ``queries/``), so they always contribute a None. We still walk them
        to keep alignment with the pixel_values tensor.
        """
        if self.atlas_cache is None:
            return [None] * len(example.image_paths)
        per_image: list[torch.Tensor | None] = []
        for path in example.image_paths:
            emb = self.atlas_cache.lookup_by_path(path)
            if emb is None:
                self._cache_misses += 1
                per_image.append(None)
            else:
                # Cache files store per-image SigLIP output as
                # (1, num_patches, hidden) — the leading "1" is the
                # processor's batch dim from a single-image render. Peel it
                # so per-image tensors stack cleanly into the
                # (N_cached, num_patches, hidden) layout the splice wants.
                if emb.dim() > 2 and emb.shape[0] == 1:
                    emb = emb.squeeze(0)
                self._cache_hits += 1
                per_image.append(emb)
        return per_image

    def __call__(self, examples: list[RenderedExample | dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Apply chat template per-example (not as a batch) so the per-example
        # assistant_mask aligns 1:1 with that example's input_ids.
        per_example: list[dict[str, torch.Tensor]] = []
        # Per-image cache lookups, batched across examples for the optional
        # splice sidecar. Each entry is (example_index, image_index_in_example,
        # cached_embedding_or_None). Order matches the eventual flattened
        # pixel_values dim-0 in _pad_batch.
        precomputed: list[torch.Tensor | None] = []
        for ex_idx, raw_ex in enumerate(examples):
            ex = raw_ex["rendered"] if isinstance(raw_ex, dict) else raw_ex
            # Account hits/misses BEFORE the heavy chat-template render so
            # measurement runs aren't burdened with re-tokenization on dataset
            # iteration alone. (No-op if atlas_cache is None.)
            per_image_emb = self._account_atlas_paths(ex)
            precomputed.extend(per_image_emb)
            out = self.processor.apply_chat_template(
                ex.messages,
                tools=ex.tools,
                chat_template_kwargs={"enable_thinking": False},
                add_generation_prompt=False,
                tokenize=True,
                return_assistant_tokens_mask=True,
                return_dict=True,
                return_tensors="pt",
            )
            ids = out["input_ids"]
            if ids.shape[1] > self.max_seq_length:
                raise ValueError(
                    f"rendered example exceeds max_seq_length="
                    f"{self.max_seq_length} (got {ids.shape[1]} tokens). "
                    f"subject_id={ex.metadata.subject_id!r}"
                )
            if "assistant_masks" in out and out["assistant_masks"].sum().item() > 0:
                assistant_mask = out["assistant_masks"]  # 1 where assistant, 0 elsewhere
            else:
                # Fallback: re-tokenize each assistant turn separately and find their token
                # spans in the full sequence. Triggers when the chat template lacks
                # {% generation %} markers OR emits an all-zero mask (template bug).
                assistant_mask = self._manual_span_mask(ex, ids[0])
            labels = ids.clone()
            labels[assistant_mask == 0] = -100

            # Sanity check: assistant tokens never overlap image tokens.
            self._sanity_check_no_image_tokens_in_labels(ids[0], labels[0])

            per_example.append({
                "input_ids": ids[0],
                "attention_mask": out["attention_mask"][0],
                "labels": labels[0],
                # Pixel values + image grid passed through verbatim
                **{k: v for k, v in out.items()
                   if k not in ("input_ids", "attention_mask", "assistant_masks")},
            })
            del ex_idx  # tracked only via per_example/precomputed ordering

        # Pad to the longest example in the batch
        batch = _pad_batch(per_example, pad_token_id=self.processor.tokenizer.pad_token_id)

        # Phase 2 splice (gated): emit the per-image embedding sidecar so a
        # vision-tower forward pre-hook can replace the cached slots without
        # re-running SigLIP. Per-image patch counts vary across atlas slices
        # (different positions render at different resolutions, query images
        # differ from atlas images), so the sidecar is a flat concatenated
        # tensor + a per-image patch-count list rather than a stacked
        # uniform-shape tensor.
        if self.enable_splice and any(e is not None for e in precomputed):
            batch["precomputed_image_mask"] = torch.tensor(
                [e is not None for e in precomputed], dtype=torch.bool,
            )
            cached_tensors = [e for e in precomputed if e is not None]
            # Each cached entry has shape ``(P_i, hidden)`` with variable
            # ``P_i``. Concatenating along dim=0 gives one flat
            # ``(sum_P, hidden)`` tensor; the splice partitions it back into
            # per-image chunks via the patch-count list.
            batch["precomputed_cached_flat"] = torch.cat(cached_tensors, dim=0)
            batch["precomputed_cached_patch_counts"] = torch.tensor(
                [t.shape[0] for t in cached_tensors], dtype=torch.long,
            )

        return batch

    def _manual_span_mask(self, example: RenderedExample, input_ids: torch.Tensor) -> torch.Tensor:
        """Build a per-token assistant mask via incremental-render diffs.

        Strategy: for each assistant turn at position i in example.messages, render
        messages[:i] and messages[:i+1]; the suffix that appears in the second but
        not the first is the assistant turn's token span. This is robust to BOS
        markers and chat-template wrappers that single-message rendering would
        otherwise produce.

        Raises RuntimeError if the diff is empty for any assistant turn (would
        indicate a chat-template bug or mismatched tokenization).
        """
        mask = torch.zeros_like(input_ids)
        cursor = 0
        for i, msg in enumerate(example.messages):
            if msg["role"] != "assistant":
                continue
            before = self.processor.apply_chat_template(
                example.messages[:i],
                tools=example.tools,
                chat_template_kwargs={"enable_thinking": False},
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )["input_ids"][0]
            through = self.processor.apply_chat_template(
                example.messages[: i + 1],
                tools=example.tools,
                chat_template_kwargs={"enable_thinking": False},
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )["input_ids"][0]
            if through.shape[0] <= before.shape[0]:
                raise RuntimeError(
                    f"manual-span mask: assistant turn {i} produced no new tokens "
                    f"(before={before.shape[0]}, through={through.shape[0]}); "
                    "chat template appears to drop assistant content."
                )
            # The new tokens added by including msg i are the suffix of `through`
            # beyond `before`. Locate that suffix in the full input_ids by searching
            # forward from `cursor`.
            new_tokens = through[before.shape[0]:]
            n = new_tokens.shape[0]
            found = False
            for start in range(cursor, input_ids.shape[0] - n + 1):
                if torch.equal(input_ids[start:start + n], new_tokens):
                    mask[start:start + n] = 1
                    cursor = start + n
                    found = True
                    break
            if not found:
                raise RuntimeError(
                    f"manual-span mask: could not locate assistant turn {i} "
                    f"({n} tokens) in full input_ids beyond cursor={cursor}. "
                    "Chat-template tokenization may differ between incremental and "
                    "full renders."
                )
        if mask.sum().item() == 0:
            raise RuntimeError(
                "manual-span mask: no assistant tokens marked. "
                f"messages had {sum(1 for m in example.messages if m['role'] == 'assistant')} "
                "assistant turns but none could be located in input_ids."
            )
        return mask.unsqueeze(0)

    def _sanity_check_no_image_tokens_in_labels(
        self,
        ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Safety net: assistant tokens must never overlap image-placeholder tokens.

        If none of the candidate names resolve to a real token id, the check is
        disabled — this is logged as a warning so users know the safety net is off.
        """
        candidate_names = ("<image_soft_token>", "<image>", "<|image|>")
        unk = self.processor.tokenizer.unk_token_id
        resolved: set[int] = set()
        for name in candidate_names:
            tok_id = self.processor.tokenizer.convert_tokens_to_ids(name)
            if tok_id is None or tok_id < 0:
                continue
            if unk is not None and tok_id == unk:
                continue
            resolved.add(tok_id)
        if not resolved:
            warnings.warn(
                "image-token sanity check disabled: none of "
                f"{candidate_names} resolved to a real token id. "
                "Verify your processor's image-placeholder token name.",
                stacklevel=2,
            )
            return
        image_ids = torch.tensor(sorted(resolved), dtype=ids.dtype, device=ids.device)
        bad = ((labels != -100) & torch.isin(ids, image_ids)).any().item()
        if bad:
            raise RuntimeError(
                f"labels mask is keeping image-token positions (token ids={sorted(resolved)}); "
                "the chat template's assistant-mask logic is wrong for this trace shape — "
                "investigate before training."
            )


def _pad_batch(
    per_example: list[dict[str, Any]],
    *,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    max_len = max(ex["input_ids"].shape[0] for ex in per_example)
    out: dict[str, torch.Tensor] = {}
    keys_with_seq_dim = ("input_ids", "attention_mask", "labels")
    for k in keys_with_seq_dim:
        padded = []
        for ex in per_example:
            t = ex[k]
            pad_len = max_len - t.shape[0]
            if pad_len > 0:
                pad_value = (
                    pad_token_id if k == "input_ids"
                    else 0 if k == "attention_mask"
                    else -100
                )
                padding = torch.full((pad_len,), pad_value, dtype=t.dtype)
                t = torch.cat([t, padding], dim=0)
            padded.append(t)
        out[k] = torch.stack(padded, dim=0)

    # Image-related tensors (e.g. pixel_values, image_position_ids) have a leading
    # "image-count" dimension per example, not a batch dim. Gemma 4's vision tower
    # expects shape (total_images_in_batch, max_patches, ...); the language model
    # then routes each image to its <image_soft_token> position via the prepared
    # input_ids. Concatenate along dim=0 so we flatten across (example, image),
    # rather than stack which would introduce a spurious extra dim.
    image_keys: set[str] = set()
    for ex in per_example:
        image_keys.update(
            k for k in ex if k not in keys_with_seq_dim and isinstance(ex[k], torch.Tensor)
        )
    for k in image_keys:
        try:
            out[k] = torch.cat([ex[k] for ex in per_example], dim=0)
        except RuntimeError as e:
            raise RuntimeError(
                f"failed to concat image-tensor key {k!r} across batch: {e}. "
                "Examples in the same batch must produce concatenatable image-tensor "
                "shapes (matching dims after the leading image-count dim); check the "
                "processor configuration."
            ) from e
    return out

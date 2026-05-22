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
    from langslice_training.embeddings.cache import AtlasEmbeddingCache
    from langslice_training.embeddings.query_cache import QueryEmbeddingCache


# --- Gemma 4 token-level assistant-mask boundary IDs ---------------------
#
# Gemma 4's chat template does not emit ``{% generation %}`` markers, so
# ``apply_chat_template(..., return_assistant_tokens_mask=True)`` returns
# an all-zero mask. Earlier (commit history before 2026-05-12) we patched
# the template at runtime to add the markers — that approach was brittle
# under multi-turn agent traces whose tool responses contain image-soft
# tokens (id 258880) and was abandoned.
#
# Instead we walk ``input_ids`` directly and toggle the mask on three
# kinds of boundaries:
#
#   1. ``<|turn>model\n``        — opens an assistant span
#   2. ``<|turn>user\n`` / ``<|turn>system\n``    — closes any assistant span
#   3. ``<|tool_response>`` … ``<tool_response|>`` — environment-injected
#      tool output inlined INSIDE a model turn (Gemma 4's OpenAI
#      Chat Completions compat path). These wrap image-soft tokens when
#      the tool returns images, so they must be masked OFF while the
#      surrounding model turn stays ON.
#
# All boundary tokens are stable single-token IDs on Gemma 4 (probed
# 2026-05-12); we look them up once at collator construction and bail
# out cleanly if any are missing.

class _BoundaryTokens:
    """Resolved Gemma 4 boundary token IDs for token-level mask construction."""

    __slots__ = (
        "sot", "model_role", "user_role", "system_role", "newline",
        "eot", "tool_resp_open", "tool_resp_close", "image",
    )

    def __init__(
        self,
        sot: int,
        model_role: int,
        user_role: int,
        system_role: int,
        newline: int,
        eot: int,
        tool_resp_open: int,
        tool_resp_close: int,
        image: int,
    ) -> None:
        self.sot = sot
        self.model_role = model_role
        self.user_role = user_role
        self.system_role = system_role
        self.newline = newline
        self.eot = eot
        self.tool_resp_open = tool_resp_open
        self.tool_resp_close = tool_resp_close
        self.image = image


def _resolve_gemma4_boundary_tokens(tokenizer: Any) -> _BoundaryTokens | None:
    """Look up boundary token IDs once. Returns None if any are missing.

    The token-level mask walker is the primary fast path. When the tokenizer
    doesn't ship the expected Gemma 4 special tokens (e.g. testing with a
    stub processor) this resolver returns None and the collator falls back
    to ``_manual_span_mask``.
    """
    unk = tokenizer.unk_token_id
    resolved: dict[str, int] = {}
    # We tokenize role-name words inside turn headers (e.g. ``model``,
    # ``user``) as raw input ids because they are NOT named special tokens
    # but the template emits them as a bare word between sot and newline.
    # Their single-token form was verified on Gemma 4 (2026-05-12): user=2364,
    # model=4368, system=9731. We resolve them via ``tokenizer(...)`` rather
    # than ``convert_tokens_to_ids`` so any tokenizer-version drift is caught.
    for name in ("<|turn>", "<turn|>", "<|tool_response>", "<tool_response|>", "<|image|>"):
        tid = tokenizer.convert_tokens_to_ids(name)
        if tid is None or tid < 0 or (unk is not None and tid == unk):
            return None
        resolved[name] = tid
    # Bare role-name words: tokenize with no special-token wrapping.
    for role in ("model", "user", "system"):
        toks = tokenizer(role, add_special_tokens=False).input_ids
        if len(toks) != 1:
            return None
        resolved[role] = toks[0]
    # Bare newline (the third position in ``<|turn>model\n``).
    nl = tokenizer("\n", add_special_tokens=False).input_ids
    if len(nl) != 1:
        return None
    resolved["\n"] = nl[0]
    return _BoundaryTokens(
        sot=resolved["<|turn>"],
        model_role=resolved["model"],
        user_role=resolved["user"],
        system_role=resolved["system"],
        newline=resolved["\n"],
        eot=resolved["<turn|>"],
        tool_resp_open=resolved["<|tool_response>"],
        tool_resp_close=resolved["<tool_response|>"],
        image=resolved["<|image|>"],
    )


def _token_level_assistant_mask(
    ids: torch.Tensor,
    tokens: _BoundaryTokens,
) -> torch.Tensor:
    """Build the per-token assistant mask by walking ``ids`` with a 2-state machine.

    State A: ``in_model`` — are we inside ``<|turn>model\\n`` ... ``<turn|>``?
    State B: ``in_tool_resp`` — are we inside ``<|tool_response> ... <tool_response|>``?

    A position contributes to the mask iff ``in_model and not in_tool_resp``
    AND the token isn't an image-soft token (defensive — image tokens never
    appear inside model output in our pipeline, but the template's tool
    response handling and possible future templates make this assertion
    worth enforcing locally).

    Returns a ``[1, seq_len]`` long tensor (same shape contract as the
    processor's ``assistant_masks``).
    """
    n = ids.shape[0]
    mask = torch.zeros(n, dtype=ids.dtype, device=ids.device)
    in_model = False
    in_tool_resp = False
    i = 0
    while i < n:
        tok = int(ids[i].item())
        # Turn header: <|turn> + (model|user|system) + \n
        if (
            tok == tokens.sot
            and i + 2 < n
            and int(ids[i + 2].item()) == tokens.newline
        ):
            role_tok = int(ids[i + 1].item())
            if role_tok == tokens.model_role:
                in_model = True
            elif role_tok in (tokens.user_role, tokens.system_role):
                in_model = False
            # Other role names (e.g. ``tool``) — Gemma 4's template doesn't
            # currently emit a ``<|turn>tool`` header, but if it ever does
            # we conservatively close the model span.
            else:
                in_model = False
            in_tool_resp = False
            # Header tokens themselves are template-emitted (not model
            # output) so we leave mask[i:i+3]==0 and skip past them.
            i += 3
            continue
        # End-of-turn closer: <turn|>\n. The model DOES emit <turn|> in a
        # real rollout (it's the eos), so keep it in the mask when we're
        # inside a model turn. The following \n is also part of the
        # template's terminator; including it makes the mask align cleanly
        # with the boundary used by the chat template.
        if tok == tokens.eot:
            if in_model and not in_tool_resp:
                mask[i] = 1
                if i + 1 < n and int(ids[i + 1].item()) == tokens.newline:
                    mask[i + 1] = 1
                    i += 2
                    in_model = False
                    in_tool_resp = False
                    continue
            in_model = False
            in_tool_resp = False
            i += 1
            continue
        # Tool-response inline boundaries.
        if tok == tokens.tool_resp_open:
            in_tool_resp = True
            i += 1
            continue
        if tok == tokens.tool_resp_close:
            in_tool_resp = False
            i += 1
            continue
        # Image-soft tokens are pixel-values slots, never model output.
        if tok == tokens.image:
            i += 1
            continue
        if in_model and not in_tool_resp:
            mask[i] = 1
        i += 1
    return mask.unsqueeze(0)


class LangSliceCollator:
    """Builds a TRL-compatible batch from RenderedExample objects.

    The processor's chat template is the source of truth for the assistant-token
    mask. Labels are constructed by cloning input_ids and zeroing (with -100)
    every position where the assistant_mask is False.

    Embedding cache integration (atlas + query)
    -------------------------------------------

    The collator can consult two caches in a fixed precedence chain: ``atlas_cache``
    (canonical atlas reference images, keyed by ``atlas/<atlas>/<plane>/<basename>``)
    and ``query_cache`` (per-row slice images, keyed by the manifest-relative
    ``data/datasets/<plane>/<dataset>/...`` path). The atlas cache wins on path
    collision (its layout is bit-exact-verified). When either cache supplies an
    embedding for an image, that image is counted as a hit and gets a sidecar slot
    so the splice forward can skip SigLIP for that position.

    The Phase 2 splice itself is gated by ``enable_splice`` — when False (the
    default) the caches are used purely for measurement. When True, cached
    images are still rendered through the processor (so the chat template
    sees the right image-token count) but the per-image precomputed
    embeddings are returned alongside the batch under
    ``precomputed_cached_flat`` + ``precomputed_image_mask`` +
    ``precomputed_cached_patch_counts`` so a vision-tower forward pre-hook can
    splice them in without re-running SigLIP. The trainer is responsible for
    wiring the hook (see ``embeddings.splice``); the collator only emits the
    sidecar.
    """

    def __init__(
        self,
        *,
        processor: Any,
        max_seq_length: int,
        atlas_cache: AtlasEmbeddingCache | None = None,
        query_cache: QueryEmbeddingCache | None = None,
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
        self.query_cache = query_cache
        self.enable_splice = enable_splice
        # Resolve boundary token IDs for the token-level masker. Returns None
        # on non-Gemma-4 / stub processors; in that case ``_call__`` falls
        # back to ``_manual_span_mask``.
        self._boundary_tokens = _resolve_gemma4_boundary_tokens(processor.tokenizer)
        if enable_splice and atlas_cache is None and query_cache is None:
            raise ValueError(
                "enable_splice=True requires at least one of atlas_cache / query_cache; "
                "passing splice without a cache would mark every image as a miss and "
                "the forward hook would never fire."
            )
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        # Per-source hit counters for diagnostics — the parent splice hook
        # doesn't care which cache the embedding came from, but operators
        # tuning coverage do.
        self._atlas_hits: int = 0
        self._query_hits: int = 0

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
        """Return the current cache counter snapshot for logging."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "atlas_hits": self._atlas_hits,
            "query_hits": self._query_hits,
        }

    def _account_image_paths(
        self, example: RenderedExample
    ) -> list[torch.Tensor | None]:
        """Bump hit/miss counters from ``example.image_paths`` and return per-image embeddings.

        The returned list is aligned with the per-example pixel_values dim-0
        ordering: query images first, then tool-result images in trace
        order. Entry ``i`` is the cached embedding tensor when the path at
        index ``i`` snaps to either the atlas or query cache, else ``None``.

        Atlas cache wins on path collision — its bit-exact verification
        history (``embeddings._verify_cache``) makes it canonical. Query
        cache fills slots the atlas cache doesn't cover (the slice/per-row
        images, which live under ``data/datasets/<plane>/<dataset>/...``
        rather than ``atlas/<atlas>/<plane>/...``).
        """
        if self.atlas_cache is None and self.query_cache is None:
            return [None] * len(example.image_paths)
        # CLAHE-augmented slice pixels must reach the live SigLIP forward, so
        # the per-row query images bypass the query embedding cache (which was
        # precomputed on the un-augmented JPEGs). Atlas cache lookups still run
        # for any atlas/* path in image_paths regardless of this flag.
        skip_query_cache = bool(getattr(example, "apply_clahe", False))
        n_query = int(getattr(example, "n_query_images", 0))
        per_image: list[torch.Tensor | None] = []
        for idx, path in enumerate(example.image_paths):
            is_query_slot = idx < n_query
            emb: torch.Tensor | None = None
            hit_source: str | None = None
            if self.atlas_cache is not None:
                emb = self.atlas_cache.lookup_by_path(path)
                if emb is not None:
                    hit_source = "atlas"
            if (
                emb is None
                and self.query_cache is not None
                and not (skip_query_cache and is_query_slot)
            ):
                emb = self.query_cache.lookup_by_path(path)
                if emb is not None:
                    hit_source = "query"
            if emb is None:
                self._cache_misses += 1
                per_image.append(None)
                continue
            # Cache files store per-image SigLIP output as
            # (1, num_patches, hidden) — the leading "1" is the
            # processor's batch dim from a single-image render. Peel it
            # so per-image tensors stack cleanly into the
            # (N_cached, num_patches, hidden) layout the splice wants.
            if emb.dim() > 2 and emb.shape[0] == 1:
                emb = emb.squeeze(0)
            self._cache_hits += 1
            if hit_source == "atlas":
                self._atlas_hits += 1
            elif hit_source == "query":
                self._query_hits += 1
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
            # iteration alone. (No-op if neither cache is provided.)
            per_image_emb = self._account_image_paths(ex)
            precomputed.extend(per_image_emb)
            # Note: ``chat_template_kwargs={"enable_thinking": False}`` was
            # removed in 2026-05-12 — the modern transformers (5.x)
            # ``apply_chat_template`` signature has no such kwarg; it was
            # captured by ``**kwargs`` and silently dropped (logged as
            # "Keyword argument `chat_template_kwargs` is not a valid argument
            # for this processor"). Gemma 4's template defaults to
            # ``enable_thinking`` being undefined, which equals "thinking off"
            # in the template's ``is defined and enable_thinking`` guard, so
            # the original intent is preserved.
            # We don't request ``return_assistant_tokens_mask`` — Gemma 4's
            # template has no ``{% generation %}`` markers and our prior
            # runtime template-patch attempt couldn't correctly handle
            # tool-response inlines without leaking image tokens into the
            # mask. ``_token_level_assistant_mask`` walks ``input_ids``
            # directly using known boundary token IDs and produces the
            # correct mask in one O(N) pass; ``_manual_span_mask`` remains
            # a safety net for non-Gemma-4 / stub processors where the
            # boundary tokens don't resolve.
            out = self.processor.apply_chat_template(
                ex.messages,
                tools=ex.tools,
                add_generation_prompt=False,
                tokenize=True,
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
            template_mask = out.get("assistant_masks")
            if self._boundary_tokens is not None:
                assistant_mask = _token_level_assistant_mask(ids[0], self._boundary_tokens)
                if assistant_mask.sum().item() == 0:
                    # Boundary tokens resolved but no assistant span found
                    # — likely an example with zero assistant turns OR a
                    # template change that breaks our walker. Defer to
                    # processor-provided assistant masks when available, then
                    # the incremental-render safety net.
                    if template_mask is not None and template_mask.sum().item() > 0:
                        assistant_mask = template_mask
                    else:
                        assistant_mask = self._manual_span_mask(ex, ids[0])
            else:
                # Non-Gemma-4 / stub processor: prefer explicit
                # assistant_masks if provided; otherwise fall back to
                # incremental-render diff.
                if template_mask is not None and template_mask.sum().item() > 0:
                    assistant_mask = template_mask
                else:
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
            # See header comment in __call__ on why chat_template_kwargs is
            # absent here too.
            before = self.processor.apply_chat_template(
                example.messages[:i],
                tools=example.tools,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )["input_ids"][0]
            through = self.processor.apply_chat_template(
                example.messages[: i + 1],
                tools=example.tools,
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

        We check ``<|image|>`` first (the canonical Gemma 4 name; resolves to
        id 258880) then fall back to legacy names for older models. Note that
        ``<image_soft_token>`` on Gemma 4 tokenizes to the UNK id rather than
        being a named special token — historical comments referenced that
        name but resolving it would silently disable the check, so the
        canonical name now takes precedence.

        If none of the candidate names resolve to a real token id, the check is
        disabled — logged as a warning so users know the safety net is off.
        """
        candidate_names = ("<|image|>", "<image_soft_token>", "<image>")
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
    sample = per_example[0]
    sample_seq_len = sample["input_ids"].shape[0]
    max_len = max(ex["input_ids"].shape[0] for ex in per_example)
    out: dict[str, torch.Tensor] = {}

    # Per-token tensors share input_ids' sequence dim. ``input_ids`` /
    # ``attention_mask`` / ``labels`` are pre-stripped of the processor's batch
    # dim and arrive as ``[seq_len]``; other per-token outputs like Gemma 4's
    # ``mm_token_type_ids`` are passed through verbatim and arrive as
    # ``[1, seq_len]``. Detect both by checking whether the trailing dim equals
    # the example's seq length — image tensors (``pixel_values``,
    # ``image_grid_thw``, etc.) never have seq_len in their shape.
    keys_with_seq_dim: set[str] = set()
    image_keys: set[str] = set()
    for k, v in sample.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.shape[-1] == sample_seq_len:
            keys_with_seq_dim.add(k)
        else:
            image_keys.add(k)

    pad_values: dict[str, int] = {"input_ids": pad_token_id, "labels": -100}

    for k in keys_with_seq_dim:
        padded: list[torch.Tensor] = []
        for ex in per_example:
            t = ex[k]
            # Squeeze the processor's batch dim so all seq tensors stack
            # uniformly to ``[batch, max_len]`` below.
            if t.dim() == 2 and t.shape[0] == 1:
                t = t[0]
            pad_len = max_len - t.shape[0]
            if pad_len > 0:
                pad_val = pad_values.get(k, 0)
                padding = torch.full((pad_len,), pad_val, dtype=t.dtype)
                t = torch.cat([t, padding], dim=0)
            padded.append(t)
        out[k] = torch.stack(padded, dim=0)

    # Image-related tensors (e.g. pixel_values, image_position_ids) have a leading
    # "image-count" dimension per example, not a batch dim. Gemma 4's vision tower
    # expects shape (total_images_in_batch, max_patches, ...); the language model
    # then routes each image to its <image_soft_token> position via the prepared
    # input_ids. Concatenate along dim=0 so we flatten across (example, image),
    # rather than stack which would introduce a spurious extra dim.
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

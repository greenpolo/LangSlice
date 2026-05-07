"""Apply processor.apply_chat_template and build labels with -100 outside assistant turns."""

from __future__ import annotations

from typing import Any

import torch

from .render import RenderedExample


class LangSliceCollator:
    """Builds a TRL-compatible batch from RenderedExample objects.

    The processor's chat template is the source of truth for the assistant-token
    mask. Labels are constructed by cloning input_ids and zeroing (with -100)
    every position where the assistant_mask is False.
    """

    def __init__(self, *, processor: Any, max_seq_length: int) -> None:
        if processor.tokenizer.pad_token_id is None:
            raise RuntimeError(
                "processor.tokenizer.pad_token_id is None — set it (e.g. to "
                "eos_token_id) before constructing LangSliceCollator. Padding a "
                "batch of variable-length examples requires a pad token."
            )
        self.processor = processor
        self.max_seq_length = max_seq_length

    def __call__(self, examples: list[RenderedExample]) -> dict[str, torch.Tensor]:
        # Apply chat template per-example (not as a batch) so the per-example
        # assistant_mask aligns 1:1 with that example's input_ids.
        per_example: list[dict[str, torch.Tensor]] = []
        for ex in examples:
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
            if "assistant_masks" in out:
                assistant_mask = out["assistant_masks"]  # 1 where assistant, 0 elsewhere
            else:
                # Fallback: re-tokenize each assistant turn separately and find their token
                # spans in the full sequence. Triggers when the chat template lacks
                # {% generation %} markers.
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

        # Pad to the longest example in the batch
        return _pad_batch(per_example, pad_token_id=self.processor.tokenizer.pad_token_id)

    def _manual_span_mask(self, example: RenderedExample, input_ids: torch.Tensor) -> torch.Tensor:
        """Build a per-token assistant mask by re-tokenizing each assistant turn."""
        mask = torch.zeros_like(input_ids)
        for msg in example.messages:
            if msg["role"] != "assistant":
                continue
            # Render just this assistant turn and find its span in input_ids
            sub = self.processor.apply_chat_template(
                [msg],
                tools=example.tools,
                chat_template_kwargs={"enable_thinking": False},
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            sub_ids = sub["input_ids"][0]
            # Linear search for the sub-sequence (cheap; assistant turns are short)
            n = sub_ids.shape[0]
            for start in range(0, input_ids.shape[0] - n + 1):
                if torch.equal(input_ids[start:start + n], sub_ids):
                    mask[start:start + n] = 1
                    break
        return mask.unsqueeze(0)

    def _sanity_check_no_image_tokens_in_labels(self, ids: torch.Tensor, labels: torch.Tensor) -> None:
        """Optional safety net — disabled if the processor doesn't expose image-token IDs."""
        candidate_names = ("<image_soft_token>", "<image>", "<|image|>")
        for name in candidate_names:
            tok_id = self.processor.tokenizer.convert_tokens_to_ids(name)
            if tok_id is not None and tok_id >= 0:
                bad = ((labels != -100) & (ids == tok_id)).any().item()
                if bad:
                    raise RuntimeError(
                        f"labels mask is keeping image-token positions (token id={tok_id}, "
                        f"name={name}); the chat template's assistant-mask logic is wrong "
                        f"for this trace shape — investigate before training"
                    )
                return


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
                pad_value = pad_token_id if k == "input_ids" else (0 if k == "attention_mask" else -100)
                padding = torch.full((pad_len,), pad_value, dtype=t.dtype)
                t = torch.cat([t, padding], dim=0)
            padded.append(t)
        out[k] = torch.stack(padded, dim=0)

    # Image-related tensors: stack along batch dim. All examples must produce
    # stackable shapes; ragged image tensors indicate a processor schema mismatch
    # we don't currently handle.
    image_keys: set[str] = set()
    for ex in per_example:
        image_keys.update(
            k for k in ex if k not in keys_with_seq_dim and isinstance(ex[k], torch.Tensor)
        )
    for k in image_keys:
        try:
            out[k] = torch.stack(
                [
                    ex[k] if ex[k].dim() == per_example[0][k].dim() else ex[k].squeeze(0)
                    for ex in per_example
                ],
                dim=0,
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"failed to stack image-tensor key {k!r} across batch: {e}. "
                "Examples in the same batch must produce stackable image-tensor "
                "shapes; check the processor configuration."
            ) from e
    return out

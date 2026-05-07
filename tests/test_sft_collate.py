"""Tests for models/langslice-gemma-4/training/sft/collate.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sft.collate import LangSliceCollator
from sft.dataset import load_examples
from sft.render import AtlasMetaCache, render_example

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sft_traces"


def _has_gemma4() -> bool:
    """Check whether the Gemma 4 processor is locally available."""
    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained("unsloth/gemma-4-E4B-it", trust_remote_code=False)
        return True
    except Exception:
        return False


_GEMMA4_AVAILABLE = _has_gemma4()
_GEMMA4_SKIP_REASON = (
    "Gemma 4 processor not available locally; run after Task 1 verification "
    "downloads it via Unsloth."
)


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained("unsloth/gemma-4-E4B-it", trust_remote_code=False)


@pytest.fixture
def rendered_single_slice(tmp_path):
    """Stage the single-slice fixture with dummy images for the renderer."""
    from PIL import Image
    src = FIXTURES / "single_slice_minimal.jsonl"
    dest = tmp_path / "single_slice_minimal.jsonl"
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    for name in ("query.png", "a3.png", "a5.png", "a7.png"):
        Image.new("RGB", (224, 224), color=(128, 128, 128)).save(tmp_path / name)
    return render_example(load_examples(dest)[0], atlas_meta_cache=AtlasMetaCache())


def test_collate_rejects_over_length_example(monkeypatch):
    """Cover the max_seq_length overflow branch without needing Gemma 4."""
    from sft.render import RenderedExample, RenderMetadata

    class StubTokenizer:
        pad_token_id = 0

    class StubProcessor:
        tokenizer = StubTokenizer()

        def apply_chat_template(self, *args, **kwargs):
            return {
                "input_ids": torch.zeros((1, 5000), dtype=torch.long),
                "attention_mask": torch.ones((1, 5000), dtype=torch.long),
                "assistant_masks": torch.zeros((1, 5000), dtype=torch.long),
            }

    rendered = RenderedExample(
        messages=[],
        tools=[],
        metadata=RenderMetadata(
            atlas_name="x", atlas_version="x", plane="coronal",
            subject_id="overlong_subject", system_prompt_kind="single_slice",
        ),
    )
    collator = LangSliceCollator(processor=StubProcessor(), max_seq_length=4096)
    with pytest.raises(ValueError, match="exceeds max_seq_length"):
        collator([rendered])


def test_pad_batch_pads_input_ids_with_pad_token():
    per_example = [
        {"input_ids": torch.tensor([1, 2, 3]),
         "attention_mask": torch.tensor([1, 1, 1]),
         "labels": torch.tensor([1, 2, 3])},
        {"input_ids": torch.tensor([4, 5]),
         "attention_mask": torch.tensor([1, 1]),
         "labels": torch.tensor([4, 5])},
    ]
    from sft.collate import _pad_batch
    out = _pad_batch(per_example, pad_token_id=99)
    assert out["input_ids"].shape == (2, 3)
    assert out["input_ids"][1, 2].item() == 99


def test_pad_batch_pads_attention_mask_with_zero():
    per_example = [
        {"input_ids": torch.tensor([1, 2, 3]),
         "attention_mask": torch.tensor([1, 1, 1]),
         "labels": torch.tensor([1, 2, 3])},
        {"input_ids": torch.tensor([4, 5]),
         "attention_mask": torch.tensor([1, 1]),
         "labels": torch.tensor([4, 5])},
    ]
    from sft.collate import _pad_batch
    out = _pad_batch(per_example, pad_token_id=99)
    assert out["attention_mask"][1, 2].item() == 0


def test_pad_batch_pads_labels_with_minus_100():
    per_example = [
        {"input_ids": torch.tensor([1, 2, 3]),
         "attention_mask": torch.tensor([1, 1, 1]),
         "labels": torch.tensor([1, 2, 3])},
        {"input_ids": torch.tensor([4, 5]),
         "attention_mask": torch.tensor([1, 1]),
         "labels": torch.tensor([4, 5])},
    ]
    from sft.collate import _pad_batch
    out = _pad_batch(per_example, pad_token_id=99)
    assert out["labels"][1, 2].item() == -100


@pytest.mark.skipif(not _GEMMA4_AVAILABLE, reason=_GEMMA4_SKIP_REASON)
def test_collate_labels_match_input_ids_on_assistant_tokens(processor, rendered_single_slice):
    collator = LangSliceCollator(processor=processor, max_seq_length=4096)
    batch = collator([rendered_single_slice])
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    assert input_ids.shape == labels.shape
    # Where labels != -100, they must equal input_ids
    keep_mask = labels != -100
    assert torch.equal(labels[keep_mask], input_ids[keep_mask])


@pytest.mark.skipif(not _GEMMA4_AVAILABLE, reason=_GEMMA4_SKIP_REASON)
def test_collate_labels_minus_100_outside_assistant(processor, rendered_single_slice):
    collator = LangSliceCollator(processor=processor, max_seq_length=4096)
    batch = collator([rendered_single_slice])
    labels = batch["labels"][0]
    # At least one token should be -100 (system + user content)
    assert (labels == -100).sum().item() > 0
    # At least one token should not be -100 (assistant content)
    assert (labels != -100).sum().item() > 0
    # The fraction of kept tokens should be small relative to total
    keep_fraction = (labels != -100).float().mean().item()
    assert keep_fraction < 0.5, f"unexpectedly high keep fraction: {keep_fraction}"

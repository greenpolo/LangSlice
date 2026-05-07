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


pytestmark = pytest.mark.skipif(
    not _has_gemma4(),
    reason="Gemma 4 processor not available locally; run after Task 1 verification "
            "downloads it via Unsloth.",
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


def test_collate_labels_match_input_ids_on_assistant_tokens(processor, rendered_single_slice):
    collator = LangSliceCollator(processor=processor, max_seq_length=4096)
    batch = collator([rendered_single_slice])
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    assert input_ids.shape == labels.shape
    # Where labels != -100, they must equal input_ids
    keep_mask = labels != -100
    assert torch.equal(labels[keep_mask], input_ids[keep_mask])


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

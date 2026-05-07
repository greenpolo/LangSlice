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


@pytest.mark.skipif(not _GEMMA4_AVAILABLE, reason=_GEMMA4_SKIP_REASON)
def test_collate_image_token_sanity_check(processor, rendered_single_slice):
    """No labels position should fall on an image-placeholder token ID."""
    collator = LangSliceCollator(processor=processor, max_seq_length=4096)
    batch = collator([rendered_single_slice])
    labels = batch["labels"][0]
    input_ids = batch["input_ids"][0]
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image_soft_token>")
    if image_token_id is None or image_token_id < 0:
        pytest.skip("Gemma 4 image-placeholder token not exposed under that name")
    keep_positions = (labels != -100).nonzero(as_tuple=True)[0]
    for pos in keep_positions:
        assert input_ids[pos].item() != image_token_id, (
            f"labels[{pos}] is {input_ids[pos].item()} which is the image token id"
        )


@pytest.mark.skipif(not _GEMMA4_AVAILABLE, reason=_GEMMA4_SKIP_REASON)
def test_collate_falls_back_to_manual_span_when_no_assistant_mask(
    monkeypatch, processor, rendered_single_slice
):
    """Stripping assistant_masks from the processor output forces the fallback,
    which must reconstruct (approximately) the same labels mask the primary
    path would have produced."""
    # Primary path
    collator = LangSliceCollator(processor=processor, max_seq_length=4096)
    primary_batch = collator([rendered_single_slice])
    primary_labels = primary_batch["labels"][0]

    # Force fallback by stripping assistant_masks from processor output
    real = processor.apply_chat_template

    def fake_apply(*args, **kwargs):
        out = real(*args, **kwargs)
        if isinstance(out, dict) and "assistant_masks" in out:
            del out["assistant_masks"]
        return out

    monkeypatch.setattr(processor, "apply_chat_template", fake_apply)

    fallback_collator = LangSliceCollator(processor=processor, max_seq_length=4096)
    fallback_batch = fallback_collator([rendered_single_slice])
    fallback_labels = fallback_batch["labels"][0]

    # The two label sequences should agree on the vast majority of positions.
    # Allow some slack for BOS / template-edge tokens (these may differ by 1-2
    # positions per assistant turn).
    n_total = primary_labels.shape[0]
    n_match = (primary_labels == fallback_labels).sum().item()
    agreement = n_match / n_total
    assert agreement > 0.9, (
        f"fallback labels disagree with primary labels at {n_total - n_match} of "
        f"{n_total} positions (agreement={agreement:.3f}); fallback is likely broken."
    )


def test_manual_span_mask_finds_assistant_turns_via_incremental_render():
    """Stub processor: full render is concat of system+user+assistant+user+assistant
    token sequences. Verify _manual_span_mask marks exactly the assistant spans."""
    from sft.render import RenderedExample, RenderMetadata

    # Token id legend (synthetic): 1-9 = system, 10-19 = user1, 20-29 = assistant1,
    # 30-39 = user2, 40-49 = assistant2. Stub returns SEGMENTS[len(messages)],
    # i.e. SEGMENTS[N] is the rendering of the first N messages.
    SEGMENTS = {
        0: torch.tensor([], dtype=torch.long),
        1: torch.tensor([1, 2, 3]),                                                  # +system
        2: torch.tensor([1, 2, 3, 10, 11, 12, 13]),                                  # +user1
        3: torch.tensor([1, 2, 3, 10, 11, 12, 13, 20, 21, 22]),                      # +assistant1
        4: torch.tensor([1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 30, 31]),              # +user2
        5: torch.tensor(  # +assistant2
            [1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 30, 31, 40, 41, 42, 43]
        ),
    }
    full = SEGMENTS[5]

    class StubTokenizer:
        unk_token_id = 0
        pad_token_id = 0

        def convert_tokens_to_ids(self, name):
            return -1

    class StubProcessor:
        tokenizer = StubTokenizer()

        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": SEGMENTS[len(messages)].unsqueeze(0)}

    example = RenderedExample(
        messages=[
            {"role": "system", "content": "x"},
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "x"},
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "x"},
        ],
        tools=[],
        metadata=RenderMetadata(
            atlas_name="x", atlas_version="x", plane="coronal",
            subject_id="stub", system_prompt_kind="single_slice",
        ),
    )
    collator = LangSliceCollator(processor=StubProcessor(), max_seq_length=4096)
    mask = collator._manual_span_mask(example, full)
    expected = torch.zeros_like(full)
    expected[7:10] = 1   # assistant1 span
    expected[12:16] = 1  # assistant2 span
    assert torch.equal(mask.squeeze(0), expected), (
        f"got {mask.squeeze(0).tolist()}, expected {expected.tolist()}"
    )


def test_manual_span_mask_raises_when_turn_unfindable():
    """If incremental render produces tokens that don't match input_ids, raise."""
    from sft.render import RenderedExample, RenderMetadata

    class StubTokenizer:
        unk_token_id = 0
        pad_token_id = 0

        def convert_tokens_to_ids(self, name):
            return -1

    class StubProcessor:
        tokenizer = StubTokenizer()

        def apply_chat_template(self, messages, **kwargs):
            n = len(messages)
            return {"input_ids": torch.tensor([[100 + i for i in range(n + 1)]])}

    example = RenderedExample(
        messages=[
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "x"},
        ],
        tools=[],
        metadata=RenderMetadata(
            atlas_name="x", atlas_version="x", plane="coronal",
            subject_id="stub", system_prompt_kind="single_slice",
        ),
    )
    collator = LangSliceCollator(processor=StubProcessor(), max_seq_length=4096)
    full_input_ids = torch.tensor([999, 999, 999])  # nothing matches
    with pytest.raises(RuntimeError, match="could not locate assistant turn"):
        collator._manual_span_mask(example, full_input_ids)


def test_sanity_check_passes_when_no_image_tokens_in_labels():
    """No labels positions match the image-token ids → no error."""
    IMAGE_ID = 999

    class StubTokenizer:
        unk_token_id = 0
        pad_token_id = 0

        def convert_tokens_to_ids(self, name):
            return IMAGE_ID if name == "<image_soft_token>" else 0

    class StubProcessor:
        tokenizer = StubTokenizer()

    collator = LangSliceCollator(processor=StubProcessor(), max_seq_length=4096)
    # ids has no IMAGE_ID; labels keeps a few positions (not -100). Should pass.
    ids = torch.tensor([1, 2, 3, 4, 5])
    labels = torch.tensor([-100, -100, 3, 4, -100])
    collator._sanity_check_no_image_tokens_in_labels(ids, labels)  # no raise


def test_sanity_check_raises_when_image_token_in_kept_labels():
    """A kept (non -100) label position with the image-token id → raise."""
    IMAGE_ID = 999

    class StubTokenizer:
        unk_token_id = 0
        pad_token_id = 0

        def convert_tokens_to_ids(self, name):
            return IMAGE_ID if name == "<image_soft_token>" else 0

    class StubProcessor:
        tokenizer = StubTokenizer()

    collator = LangSliceCollator(processor=StubProcessor(), max_seq_length=4096)
    ids = torch.tensor([1, 2, IMAGE_ID, 4, 5])
    labels = torch.tensor([-100, -100, IMAGE_ID, -100, -100])
    with pytest.raises(RuntimeError, match="keeping image-token positions"):
        collator._sanity_check_no_image_tokens_in_labels(ids, labels)


def test_sanity_check_warns_when_no_candidate_resolves():
    """If no candidate name resolves (or all UNK), warn but don't raise."""

    class StubTokenizer:
        unk_token_id = 3
        pad_token_id = 0

        def convert_tokens_to_ids(self, name):
            return self.unk_token_id  # all candidates → UNK

    class StubProcessor:
        tokenizer = StubTokenizer()

    collator = LangSliceCollator(processor=StubProcessor(), max_seq_length=4096)
    ids = torch.tensor([1, 2, 3, 4, 5])
    labels = torch.tensor([-100, -100, 3, 4, -100])
    with pytest.warns(UserWarning, match="image-token sanity check disabled"):
        collator._sanity_check_no_image_tokens_in_labels(ids, labels)

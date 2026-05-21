"""Tests for models/langslice-gemma-4/training/sft/collate.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from sft.collate import (
    LangSliceCollator,
    _BoundaryTokens,
    _resolve_gemma4_boundary_tokens,
    _token_level_assistant_mask,
)
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
        unk_token_id = 0

        def convert_tokens_to_ids(self, name):
            return -1

        def __call__(self, text, **kwargs):
            class _Tokenized:
                input_ids = [-1]
            return _Tokenized()

    class StubProcessor:
        tokenizer = StubTokenizer()

        def apply_chat_template(self, *args, **kwargs):
            return {
                "input_ids": torch.zeros((1, 5000), dtype=torch.long),
                "attention_mask": torch.ones((1, 5000), dtype=torch.long),
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


def test_pad_batch_pads_processor_batched_seq_tensors():
    # Gemma 4's processor returns ``mm_token_type_ids`` as ``[1, seq_len]`` —
    # the batch dim is preserved unlike input_ids/attention_mask/labels which
    # are explicitly ``[0]``-indexed in collate. Verify the per-token shape
    # detection handles both cases at PDBS>1.
    per_example = [
        {"input_ids": torch.tensor([1, 2, 3]),
         "attention_mask": torch.tensor([1, 1, 1]),
         "labels": torch.tensor([1, 2, 3]),
         "mm_token_type_ids": torch.tensor([[0, 1, 1]]),
         "pixel_values": torch.zeros(2, 3, 4, 4)},
        {"input_ids": torch.tensor([4, 5]),
         "attention_mask": torch.tensor([1, 1]),
         "labels": torch.tensor([4, 5]),
         "mm_token_type_ids": torch.tensor([[0, 1]]),
         "pixel_values": torch.zeros(1, 3, 4, 4)},
    ]
    from sft.collate import _pad_batch
    out = _pad_batch(per_example, pad_token_id=99)
    assert out["mm_token_type_ids"].shape == (2, 3)
    # Padding position (row 1, col 2) is 0 (non-image marker)
    assert out["mm_token_type_ids"][1, 2].item() == 0
    # Pre-padding positions are preserved
    assert out["mm_token_type_ids"][0].tolist() == [0, 1, 1]
    assert out["mm_token_type_ids"][1, :2].tolist() == [0, 1]
    # Pixel values still concat along image-count dim
    assert out["pixel_values"].shape == (3, 3, 4, 4)


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
    # ``<|image|>`` is the canonical Gemma 4 image-soft-token (id 258880).
    # The earlier name ``<image_soft_token>`` resolves to UNK (id 3) on this
    # tokenizer, which would make the check trivially pass — use the real one.
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image|>")
    assert image_token_id is not None and image_token_id > 0, (
        "<|image|> token must resolve to a real id on Gemma 4"
    )
    keep_positions = (labels != -100).nonzero(as_tuple=True)[0]
    for pos in keep_positions:
        assert input_ids[pos].item() != image_token_id, (
            f"labels[{pos}] is {input_ids[pos].item()} which is the image token id"
        )


@pytest.mark.skipif(not _GEMMA4_AVAILABLE, reason=_GEMMA4_SKIP_REASON)
def test_collate_falls_back_to_manual_span_when_no_boundary_tokens(
    processor, rendered_single_slice
):
    """Disabling the token-level walker (``_boundary_tokens = None``) forces
    the ``_manual_span_mask`` fallback, which must reconstruct (approximately)
    the same labels mask the primary path would have produced."""
    # Primary path: token-level walker
    collator = LangSliceCollator(processor=processor, max_seq_length=4096)
    primary_batch = collator([rendered_single_slice])
    primary_labels = primary_batch["labels"][0]

    # Force fallback by nulling the boundary-token cache
    fallback_collator = LangSliceCollator(processor=processor, max_seq_length=4096)
    fallback_collator._boundary_tokens = None
    fallback_batch = fallback_collator([rendered_single_slice])
    fallback_labels = fallback_batch["labels"][0]

    # The two label sequences should agree on the vast majority of positions.
    # The token-level walker includes the trailing ``<turn|>\n`` of each
    # assistant span in the mask (the model learns to emit eos), while
    # ``_manual_span_mask`` derives spans from the incremental-render diff
    # which may or may not capture that pair depending on Jinja whitespace.
    # Some image-token positions are also handled defensively by the walker
    # but cannot be by the span-diff path. Allow some slack here.
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


# --- Token-level assistant-mask walker ----------------------------------


def _synth_boundary_tokens() -> _BoundaryTokens:
    """Synthetic token IDs for the standalone walker tests.

    These don't correspond to the real Gemma 4 vocab — the walker only cares
    that all 9 IDs are distinct and that the role-name IDs follow ``sot`` and
    precede ``newline`` in turn-header triples.
    """
    return _BoundaryTokens(
        sot=100,
        model_role=200,
        user_role=201,
        system_role=202,
        newline=300,
        eot=400,
        tool_resp_open=500,
        tool_resp_close=501,
        image=900,
    )


def test_token_level_mask_simple_two_turn():
    """user → assistant text. Mask covers ONLY the assistant span (incl. eot)."""
    tok = _synth_boundary_tokens()
    # Layout:
    #   [sot, user, nl, 10, 11, eot, nl,     <- user turn (mask=0)
    #    sot, model, nl, 20, 21, 22, eot, nl] <- model turn (mask=1 inside)
    ids = torch.tensor([
        100, 201, 300, 10, 11, 400, 300,
        100, 200, 300, 20, 21, 22, 400, 300,
    ], dtype=torch.long)
    mask = _token_level_assistant_mask(ids, tok).squeeze(0)
    expected = torch.tensor([
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 1, 1, 1,  # 20/21/22 + <turn|> + \n masked ON
    ], dtype=torch.long)
    assert torch.equal(mask, expected), (
        f"got {mask.tolist()}, expected {expected.tolist()}"
    )


def test_token_level_mask_tool_response_inline_excluded():
    """Model turn with inlined tool response. Tool-response span (incl. images) masked OFF.

    Mirrors a Gemma 4 multi-turn agent trace where the assistant emits a
    tool_call, the template inlines ``<|tool_response>...<image>...<tool_response|>``,
    then the assistant emits a final tool_call.
    """
    tok = _synth_boundary_tokens()
    # Token legend:
    #   30, 31  = tool_call content #1 (mask ON)
    #   900     = image-soft-token (always OFF)
    #   40, 41  = tool response payload text (mask OFF — inside tool_resp)
    #   50, 51  = tool_call content #2 (mask ON)
    ids = torch.tensor([
        100, 201, 300, 10, 11, 400, 300,                     # user turn
        100, 200, 300,                                       # model turn start
        30, 31,                                              # tool_call 1
        500, 40, 900, 900, 41, 501,                          # <|tool_resp> ... <tool_resp|>
        50, 51,                                              # tool_call 2
        400, 300,                                            # <turn|> \n
    ], dtype=torch.long)
    mask = _token_level_assistant_mask(ids, tok).squeeze(0)
    # Expected mask: 30/31, 50/51, eot, nl masked ON; tool-resp body OFF;
    # image tokens always OFF.
    expected_idx_on = {
        # tool_call 1 positions:
        10, 11,                  # indices of 30, 31 in `ids`
        # tool_call 2 positions:
        18, 19,                  # indices of 50, 51
        # eot, nl:
        20, 21,
    }
    for i in range(ids.shape[0]):
        if i in expected_idx_on:
            assert mask[i].item() == 1, f"expected mask[{i}]=1 (tok={ids[i].item()})"
        else:
            assert mask[i].item() == 0, f"expected mask[{i}]=0 (tok={ids[i].item()})"
    # Sanity: zero image tokens kept
    for i in range(ids.shape[0]):
        if ids[i].item() == tok.image:
            assert mask[i].item() == 0


def test_token_level_mask_image_in_user_turn_stays_masked():
    """Image tokens in user turns must NOT be in the mask either."""
    tok = _synth_boundary_tokens()
    ids = torch.tensor([
        100, 201, 300, 900, 900, 11, 400, 300,    # user turn with images
        100, 200, 300, 20, 400, 300,              # model turn
    ], dtype=torch.long)
    mask = _token_level_assistant_mask(ids, tok).squeeze(0)
    # Only positions 11 (val=20), 12 (val=400=eot), 13 (val=300=nl) should be 1
    for i in range(8):
        assert mask[i].item() == 0
    assert mask[11].item() == 1
    assert mask[12].item() == 1
    assert mask[13].item() == 1


def test_token_level_mask_returns_empty_when_no_model_turn():
    """No ``<|turn>model\\n`` header → mask is all zeros."""
    tok = _synth_boundary_tokens()
    ids = torch.tensor([
        100, 201, 300, 10, 11, 400, 300,
        100, 202, 300, 1, 2, 3, 400, 300,
    ], dtype=torch.long)
    mask = _token_level_assistant_mask(ids, tok).squeeze(0)
    assert mask.sum().item() == 0


def test_token_level_mask_trailing_dangling_tool_response():
    """Last model turn ends with a dangling ``<|tool_response>`` (no close)
    when the template emits a tool-call generation prompt with no follow-up.
    The trailing region must be masked OFF; no IndexError on EOS.
    """
    tok = _synth_boundary_tokens()
    ids = torch.tensor([
        100, 200, 300,  # model turn start
        30, 31,         # tool_call (mask ON)
        500,            # dangling <|tool_response> (no close)
    ], dtype=torch.long)
    mask = _token_level_assistant_mask(ids, tok).squeeze(0)
    expected = torch.tensor([0, 0, 0, 1, 1, 0], dtype=torch.long)
    assert torch.equal(mask, expected)


# --- Gemma 4 integration tests (require local processor) ------------------


@pytest.fixture
def fresh_processor():
    """Function-scoped processor; the new collator never mutates the
    processor's chat_template, so this fixture exists purely so tests that
    expect a clean state aren't affected by other tests' side effects."""
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained("unsloth/gemma-4-E4B-it", trust_remote_code=False)


@pytest.mark.skipif(not _GEMMA4_AVAILABLE, reason=_GEMMA4_SKIP_REASON)
def test_resolve_gemma4_boundary_tokens_succeeds(fresh_processor):
    """The real Gemma 4 tokenizer resolves all 9 boundary tokens."""
    bt = _resolve_gemma4_boundary_tokens(fresh_processor.tokenizer)
    assert bt is not None
    # Spot-check known IDs (probed 2026-05-12).
    assert bt.sot == 105
    assert bt.eot == 106
    assert bt.user_role == 2364
    assert bt.model_role == 4368
    assert bt.system_role == 9731
    assert bt.newline == 107
    assert bt.tool_resp_open == 50
    assert bt.tool_resp_close == 51
    assert bt.image == 258880


@pytest.mark.skipif(not _GEMMA4_AVAILABLE, reason=_GEMMA4_SKIP_REASON)
def test_collator_multi_turn_agent_fixture_no_image_tokens_in_mask(processor):
    """Multi-turn agent trace with image-in-tool-response.

    Constructs the canonical bug-reproduction fixture (user query image,
    assistant tool_call, tool result containing an image, final assistant
    tool_call). Verifies:
      - mask is non-empty
      - no kept position holds the image-soft-token id (258880)
      - tool_call payloads ARE in the mask
      - tool-response body is NOT in the mask
    """
    from PIL import Image
    from sft.render import RenderedExample, RenderMetadata

    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image|>")
    # Skip if the canonical name doesn't resolve — would indicate a
    # tokenizer regression worth investigating before training.
    assert image_token_id is not None and image_token_id > 0

    tools = [
        {"type": "function", "function": {
            "name": "fetch_atlas", "description": "Fetch atlas slices",
            "parameters": {"type": "object", "properties": {
                "section_index": {"type": "integer"},
            }, "required": ["section_index"]},
        }},
        {"type": "function", "function": {
            "name": "submit_estimate", "description": "Submit final estimate",
            "parameters": {"type": "object", "properties": {
                "position_mm": {"type": "number"},
            }, "required": ["position_mm"]},
        }},
    ]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are an AP estimator."}]},
        {"role": "user", "content": [
            {"type": "image", "image": Image.new("RGB", (224, 224))},
            {"type": "text", "text": "Where is this slice?"},
        ]},
        {"role": "assistant", "content": [], "tool_calls": [{
            "id": "1", "type": "function",
            "function": {"name": "fetch_atlas", "arguments": "{\"section_index\":31}"},
        }]},
        {"role": "tool", "tool_call_id": "1", "name": "fetch_atlas",
         "content": [
             {"type": "image", "image": Image.new("RGB", (224, 224))},
             {"type": "text", "text": "<|image|>Atlas at section 31"},
         ]},
        {"role": "assistant", "content": [], "tool_calls": [{
            "id": "2", "type": "function",
            "function": {"name": "submit_estimate", "arguments": "{\"position_mm\":6.5}"},
        }]},
    ]
    rendered = RenderedExample(
        messages=messages, tools=tools,
        metadata=RenderMetadata(
            atlas_name="x", atlas_version="x", plane="coronal",
            subject_id="multi_turn_fixture", system_prompt_kind="single_slice",
        ),
        image_paths=["q.png", "a.png"], n_query_images=1,
    )
    collator = LangSliceCollator(processor=processor, max_seq_length=4096)
    batch = collator([rendered])
    labels = batch["labels"][0]
    input_ids = batch["input_ids"][0]
    # Non-empty mask
    n_kept = (labels != -100).sum().item()
    assert n_kept > 0, "mask is empty — token-level walker failed to find model spans"
    # No kept position holds the image-soft-token id (the bug we're fixing)
    keep_positions = (labels != -100).nonzero(as_tuple=True)[0]
    for pos in keep_positions:
        assert input_ids[pos].item() != image_token_id, (
            f"labels[{pos}] = {input_ids[pos].item()} = image-soft-token id"
        )
    # tool_call tokens ARE in the mask (sanity: 48 = <|tool_call>, 49 = <tool_call|>)
    TOOL_CALL_OPEN = 48
    TOOL_CALL_CLOSE = 49
    assert ((labels != -100) & (input_ids == TOOL_CALL_OPEN)).sum().item() == 2
    assert ((labels != -100) & (input_ids == TOOL_CALL_CLOSE)).sum().item() == 2
    # tool_response markers are NOT in the mask
    TOOL_RESP_OPEN = 50
    TOOL_RESP_CLOSE = 51
    assert ((labels != -100) & (input_ids == TOOL_RESP_OPEN)).sum().item() == 0
    assert ((labels != -100) & (input_ids == TOOL_RESP_CLOSE)).sum().item() == 0

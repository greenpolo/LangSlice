"""Foundation tests for the langslice_traces factory layer.

Pass 2 introduces CanonicalTrace + parser. Pass 3 adds the first renderer
(``render_sft_full``) with a golden-equality test against the SFT trainer's
``render_example`` — the two implementations must produce byte-identical
``RenderedExample`` instances for equivalent input data.

Task 4 adds three new renderers (``render_sft_answer_only``,
``render_rl_prefix``, ``render_isft_prefix``) that have no existing reference
implementation; behavior tests below cover shape, prefix-sharing, and label
fields.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from langslice_traces._empirical import P_NFETCH_STEP0
from langslice_traces.generator import (
    CANONICAL_ATLAS_ROOT,
    canonical_atlas_repo_path,
    generate_trace,
    load_atlas_grid,
)
from langslice_traces.parser import iter_canonical_traces, parse_canonical_trace
from langslice_traces.renderers import (
    render_isft_prefix,
    render_rl_prefix,
    render_sft_answer_only,
    render_sft_full,
)
from langslice_traces.renderers._common import AtlasMeta
from langslice_traces.schema import CanonicalTrace, FinalAnswer, ToolStep
from PIL import Image

# Make the SFT trainer importable for the golden-comparison branch.
# pyproject.toml already adds models/langslice-gemma-4/training to pythonpath
# for pytest, but be defensive for direct script invocation.
_REPO = Path(__file__).resolve().parents[1]
_SFT_TRAINING = _REPO / "models" / "langslice-gemma-4" / "training"
if str(_SFT_TRAINING) not in sys.path:
    sys.path.insert(0, str(_SFT_TRAINING))


def _make_synthetic_row(
    *,
    subject_id: str = "M01",
    final_position_mm: float = 3.6,
) -> dict[str, Any]:
    """Build a realistic SFT-shaped row with 2 tool_steps + submit."""
    return {
        "bucket": 1,
        "atlas_name": "allen_mouse_25um",
        "atlas_version": "CCFv3",
        "plane": "coronal",
        "subject_id": subject_id,
        "system_prompt_kind": "single_slice",
        "query_image_paths": [f"queries/single_{subject_id}.jpg"],
        "user_prompt_text": (
            "Determine this coronal slice's AP position in the allen_mouse_25um atlas."
        ),
        "trace": [
            {
                "tool_call": {
                    "name": "fetch_atlas",
                    "args": {"positions_mm": [2.0, 3.0, 4.0, 5.0]},
                },
                "tool_result": {
                    "image_paths": [
                        "atlas/allen_mouse_25um/coronal/2.00mm.jpg",
                        "atlas/allen_mouse_25um/coronal/3.00mm.jpg",
                        "atlas/allen_mouse_25um/coronal/4.00mm.jpg",
                        "atlas/allen_mouse_25um/coronal/5.00mm.jpg",
                    ],
                    "text": "Fetched 4 atlas sections: 2.00 mm, 3.00 mm, 4.00 mm, 5.00 mm",
                },
            },
            {
                "tool_call": {
                    "name": "fetch_atlas",
                    "args": {"positions_mm": [3.2, 3.4, 3.6, 3.8]},
                },
                "tool_result": {
                    "image_paths": [
                        "atlas/allen_mouse_25um/coronal/3.20mm.jpg",
                        "atlas/allen_mouse_25um/coronal/3.40mm.jpg",
                        "atlas/allen_mouse_25um/coronal/3.60mm.jpg",
                        "atlas/allen_mouse_25um/coronal/3.80mm.jpg",
                    ],
                    "text": "Fetched 4 atlas sections: 3.20 mm, 3.40 mm, 3.60 mm, 3.80 mm",
                },
            },
            {
                "submit": {
                    "name": "submit_estimate",
                    "args": {
                        "position_mm": final_position_mm,
                        "reasoning": (
                            "The 3.60 mm atlas section most closely matches the "
                            "characteristic cytoarchitecture of the query."
                        ),
                    },
                },
            },
        ],
        "quality": {
            "accuracy": "in_tolerance",
            "max_error_mm": 0.085,
        },
    }


def test_parse_canonical_trace_minimal_row():
    """Round-trip a synthetic SFT-shaped row through parse_canonical_trace.

    Assert all fields populated correctly: trace structure correctly split into
    tool_steps + final_answer, atlas/plane/subject_id preserved, quality dict
    passed through, bucket=1, gemini_reasoning=None when absent.
    """
    row = _make_synthetic_row()
    trace = parse_canonical_trace(row)

    # Type + identity.
    assert isinstance(trace, CanonicalTrace)

    # Top-level metadata preserved verbatim.
    assert trace.atlas_name == "allen_mouse_25um"
    assert trace.atlas_version == "CCFv3"
    assert trace.plane == "coronal"
    assert trace.subject_id == "M01"
    assert trace.system_prompt_kind == "single_slice"
    assert trace.bucket == 1
    assert trace.query_image_paths == ["queries/single_M01.jpg"]
    assert trace.user_prompt_text.startswith("Determine this coronal slice")

    # Trace decomposition: 2 tool_steps, 1 final_answer.
    assert len(trace.tool_steps) == 2
    first, second = trace.tool_steps
    assert isinstance(first, ToolStep)
    assert first.call_name == "fetch_atlas"
    assert first.call_args == {"positions_mm": [2.0, 3.0, 4.0, 5.0]}
    assert first.result_image_paths == [
        "atlas/allen_mouse_25um/coronal/2.00mm.jpg",
        "atlas/allen_mouse_25um/coronal/3.00mm.jpg",
        "atlas/allen_mouse_25um/coronal/4.00mm.jpg",
        "atlas/allen_mouse_25um/coronal/5.00mm.jpg",
    ]
    assert first.result_text.startswith("Fetched 4 atlas sections")

    assert second.call_name == "fetch_atlas"
    assert second.call_args == {"positions_mm": [3.2, 3.4, 3.6, 3.8]}
    assert len(second.result_image_paths) == 4

    # Final answer is typed and parsed.
    assert isinstance(trace.final_answer, FinalAnswer)
    assert trace.final_answer.name == "submit_estimate"
    assert trace.final_answer.position_mm == 3.6
    assert "3.60 mm atlas section" in trace.final_answer.reasoning

    # Quality dict passed through.
    assert trace.quality == {"accuracy": "in_tolerance", "max_error_mm": 0.085}

    # gemini_reasoning absent in row → None on trace.
    assert trace.gemini_reasoning is None

    # dataset_root defaults to None when not supplied.
    assert trace.dataset_root is None


def test_iter_canonical_traces_from_jsonl(tmp_path: Path):
    """Write 2 rows to a tmp JSONL, iter_canonical_traces yields 2 CanonicalTrace
    instances with dataset_root set to tmp_path.
    """
    jsonl_path = tmp_path / "examples.jsonl"
    row_a = _make_synthetic_row(subject_id="M01", final_position_mm=3.6)
    row_b = _make_synthetic_row(subject_id="M02", final_position_mm=4.2)
    with jsonl_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(row_a) + "\n")
        f.write(json.dumps(row_b) + "\n")

    traces = list(iter_canonical_traces(jsonl_path))

    assert len(traces) == 2
    assert all(isinstance(t, CanonicalTrace) for t in traces)
    assert traces[0].subject_id == "M01"
    assert traces[0].final_answer.position_mm == 3.6
    assert traces[1].subject_id == "M02"
    assert traces[1].final_answer.position_mm == 4.2

    # dataset_root set to the JSONL's parent so renderers can hydrate images.
    assert traces[0].dataset_root == tmp_path
    assert traces[1].dataset_root == tmp_path


# ---------- render_sft_full byte-identity ----------


class _StubAtlasMetaCache:
    """Duck-typed atlas-meta cache that returns a fixed AtlasMeta.

    Both the SFT trainer's ``render_example`` and the new ``render_sft_full``
    accept any object exposing ``.get(atlas_name, plane) -> AtlasMeta``; using a
    stub keeps the golden test hermetic (no real atlas files required) and
    deterministic across both call sites.
    """

    def __init__(self, meta: AtlasMeta) -> None:
        self._meta = meta

    def get(self, atlas_name: str, plane: str) -> AtlasMeta:  # noqa: ARG002
        return self._meta


def _write_tiny_png(path: Path, *, color: tuple[int, int, int]) -> None:
    """Write a deterministic 8x8 RGB PNG used as a placeholder query/atlas image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (8, 8), color)
    img.save(path, format="PNG")


def _stage_dataset_root(tmp_path: Path, row: dict[str, Any]) -> Path:
    """Drop a deterministic PNG at every image path referenced by *row*."""
    root = tmp_path / "dataset"
    for rel in row["query_image_paths"]:
        _write_tiny_png(root / rel, color=(123, 45, 67))
    for step in row["trace"][:-1]:
        for j, rel in enumerate(step["tool_result"]["image_paths"]):
            _write_tiny_png(root / rel, color=(10 + j * 30, 20 + j * 20, 200 - j * 25))
    return root


def _pil_fingerprint(img: Image.Image) -> tuple[Any, ...]:
    """Stable fingerprint of a PIL image: (mode, size, raw-bytes)."""
    return (img.mode, img.size, img.tobytes())


def _message_fingerprint(msg: dict[str, Any]) -> dict[str, Any]:
    """Replace embedded PIL images with their fingerprint so dicts compare cleanly."""
    out: dict[str, Any] = {}
    for k, v in msg.items():
        if k == "content" and isinstance(v, list):
            out[k] = [_content_fp(c) for c in v]
        else:
            out[k] = v
    return out


def _content_fp(block: dict[str, Any]) -> dict[str, Any]:
    if block.get("type") == "image" and isinstance(block.get("image"), Image.Image):
        return {"type": "image", "image": _pil_fingerprint(block["image"])}
    return block


def test_render_sft_full_byte_identical_to_render_example(tmp_path: Path):
    """``render_sft_full(canonical)`` must produce a RenderedExample byte-identical
    to ``sft.render.render_example(example)`` for the same input data.

    This is the load-bearing guarantee that lets Pass 3 swap one consumer's
    import to ``langslice_traces.renderers.render_sft_full`` with zero
    behavioral change.
    """
    # 1. Stage a dataset_root containing every image path the synthetic row
    #    references — both the SFT trainer's ``load_examples`` and the factory
    #    parser are happy as long as the files exist on disk.
    row = _make_synthetic_row(subject_id="M01", final_position_mm=3.6)
    dataset_root = _stage_dataset_root(tmp_path, row)
    jsonl_path = dataset_root / "examples.jsonl"
    jsonl_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    # 2. Parse the row through both code paths.
    canonical = parse_canonical_trace(row, dataset_root=dataset_root)

    from sft.dataset import load_examples as old_load_examples  # noqa: PLC0415

    examples = old_load_examples(jsonl_path)
    assert len(examples) == 1
    example = examples[0]

    # 3. Build a stub AtlasMetaCache shared by both renderers — both code paths
    #    call ``.get(atlas_name, plane)`` and we want the same AtlasMeta on
    #    both sides so build_system_prompt produces identical strings.
    stub_meta = AtlasMeta(pos_lo=0.0, pos_hi=10.0, species="mouse")
    cache = _StubAtlasMetaCache(stub_meta)

    # 4. Run both renderers.
    new = render_sft_full(canonical, atlas_meta_cache=cache)

    from sft.render import render_example as old_render_example  # noqa: PLC0415

    old = old_render_example(example, atlas_meta_cache=cache)

    # 5. Tools schema must match exactly (both go through deepcopy of the same
    #    constants, so dict structure + key ordering should be identical).
    assert new.tools == old.tools

    # 6. Metadata fields must match. The two RenderMetadata dataclasses are
    #    declared in different modules (factory schema.py vs sft/render.py)
    #    but carry identical fields — compare field-by-field.
    assert new.metadata.atlas_name == old.metadata.atlas_name
    assert new.metadata.atlas_version == old.metadata.atlas_version
    assert new.metadata.plane == old.metadata.plane
    assert new.metadata.subject_id == old.metadata.subject_id
    assert new.metadata.system_prompt_kind == old.metadata.system_prompt_kind

    # 7. image_paths order must match exactly — this is what downstream
    #    consumers (atlas-embedding splice) align against.
    assert new.image_paths == old.image_paths

    # 8. Messages: same count, same per-message structure. The embedded PIL
    #    images are hydrated independently by each renderer, so compare by
    #    fingerprint (mode + size + raw bytes), not object identity.
    assert len(new.messages) == len(old.messages)
    for i, (new_msg, old_msg) in enumerate(zip(new.messages, old.messages)):
        new_fp = _message_fingerprint(new_msg)
        old_fp = _message_fingerprint(old_msg)
        assert new_fp == old_fp, f"message[{i}] diverged: {new_fp!r} vs {old_fp!r}"

    # 9. The system prompt itself (text block of message[0]) must match the
    #    output of build_single_slice_prompt with the stub AtlasMeta.
    sys_msg = new.messages[0]
    assert sys_msg["role"] == "system"
    sys_text = sys_msg["content"][0]["text"]
    assert "allen_mouse_25um (mouse)" in sys_text
    assert "0.00-10.00 mm" in sys_text

    # 10. The assistant tool_calls' JSON-serialized arguments are byte-equal,
    #     which is the strongest single check: it covers tool-call ids,
    #     fetch_atlas positions_mm, and the terminal submit's
    #     {position_mm, reasoning} payload in dict-insertion order.
    new_assistant_calls = [m for m in new.messages if m["role"] == "assistant"]
    old_assistant_calls = [m for m in old.messages if m["role"] == "assistant"]
    assert len(new_assistant_calls) == len(old_assistant_calls)
    for n_msg, o_msg in zip(new_assistant_calls, old_assistant_calls):
        assert n_msg["tool_calls"] == o_msg["tool_calls"]

    # 11. Tool-response messages must carry the matching tool_call_id and the
    #     normalized text with ``<|image|>`` markers prepended.
    new_tool_msgs = [m for m in new.messages if m["role"] == "tool"]
    old_tool_msgs = [m for m in old.messages if m["role"] == "tool"]
    assert len(new_tool_msgs) == len(old_tool_msgs)
    for n_msg, o_msg in zip(new_tool_msgs, old_tool_msgs):
        assert n_msg["tool_call_id"] == o_msg["tool_call_id"]
        # text block content (last entry per normalize_tool_message_content)
        n_text = n_msg["content"][-1]["text"]
        o_text = o_msg["content"][-1]["text"]
        assert n_text == o_text


def test_render_sft_full_raises_without_dataset_root():
    """The renderer must refuse a CanonicalTrace whose dataset_root was never set."""
    row = _make_synthetic_row()
    trace = parse_canonical_trace(row)  # no dataset_root supplied
    cache = _StubAtlasMetaCache(AtlasMeta(pos_lo=0.0, pos_hi=10.0, species="mouse"))

    with pytest.raises(ValueError, match="dataset_root"):
        render_sft_full(trace, atlas_meta_cache=cache)


# ---------- Task 4: render_sft_answer_only ----------


def _make_submit_only_row(*, subject_id: str = "M03") -> dict[str, Any]:
    """SFT-shaped row whose trace contains only the submit step (no tool_steps).

    Used to verify the prefix renderers' empty-tool_steps degenerate case.
    """
    return {
        "bucket": 1,
        "atlas_name": "allen_mouse_25um",
        "atlas_version": "CCFv3",
        "plane": "coronal",
        "subject_id": subject_id,
        "system_prompt_kind": "single_slice",
        "query_image_paths": [f"queries/single_{subject_id}.jpg"],
        "user_prompt_text": (
            "Determine this coronal slice's AP position in the allen_mouse_25um atlas."
        ),
        "trace": [
            {
                "submit": {
                    "name": "submit_estimate",
                    "args": {"position_mm": 4.0, "reasoning": "no-tool sanity case"},
                },
            },
        ],
        "quality": {"accuracy": "in_tolerance", "max_error_mm": 0.05},
    }


def test_render_sft_answer_only_strips_tool_steps(tmp_path: Path):
    """No assistant fetch_atlas tool_calls; exactly one assistant message with
    submit_estimate; reasoning is empty string; image_paths only contains
    query_image_paths (no atlas images).
    """
    row = _make_synthetic_row(subject_id="M01", final_position_mm=3.6)
    dataset_root = _stage_dataset_root(tmp_path, row)
    canonical = parse_canonical_trace(row, dataset_root=dataset_root)
    cache = _StubAtlasMetaCache(AtlasMeta(pos_lo=0.0, pos_hi=10.0, species="mouse"))

    rendered = render_sft_answer_only(canonical, atlas_meta_cache=cache)

    # Exactly one assistant message, and it's submit_estimate.
    assistant_msgs = [m for m in rendered.messages if m["role"] == "assistant"]
    assert len(assistant_msgs) == 1
    tool_calls = assistant_msgs[0]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "submit_estimate"

    # No fetch_atlas anywhere.
    for m in rendered.messages:
        for tc in m.get("tool_calls", []) or []:
            assert tc["function"]["name"] != "fetch_atlas"

    # No tool messages at all (would imply a fetch_atlas response).
    assert all(m["role"] != "tool" for m in rendered.messages)

    # reasoning is the empty string. The arguments are JSON; parse to inspect.
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert args["reasoning"] == ""
    # And the position_mm defaults to the teacher's value.
    assert args["position_mm"] == 3.6

    # image_paths is just the query images — no atlas images.
    assert rendered.image_paths == list(canonical.query_image_paths)

    # Neither target_mm nor label_mm is set (this is an SFT renderer).
    assert rendered.target_mm is None
    assert rendered.label_mm is None


def test_render_sft_answer_only_prefers_gt_when_flagged(tmp_path: Path):
    """With prefer_ground_truth=True and ground_truth_mm=12.5, the submit arg
    is 12.5 not the teacher's value (3.6).
    """
    row = _make_synthetic_row(subject_id="M01", final_position_mm=3.6)
    dataset_root = _stage_dataset_root(tmp_path, row)
    canonical = parse_canonical_trace(row, dataset_root=dataset_root)
    cache = _StubAtlasMetaCache(AtlasMeta(pos_lo=0.0, pos_hi=20.0, species="mouse"))

    rendered = render_sft_answer_only(
        canonical,
        atlas_meta_cache=cache,
        prefer_ground_truth=True,
        ground_truth_mm=12.5,
    )

    assistant_msgs = [m for m in rendered.messages if m["role"] == "assistant"]
    args = json.loads(assistant_msgs[0]["tool_calls"][0]["function"]["arguments"])
    assert args["position_mm"] == 12.5
    # Teacher's 3.6 must not leak through.
    assert args["position_mm"] != canonical.final_answer.position_mm


def test_render_sft_answer_only_requires_gt_when_prefer_flag(tmp_path: Path):
    """With prefer_ground_truth=True and ground_truth_mm=None, raises ValueError."""
    row = _make_synthetic_row(subject_id="M01", final_position_mm=3.6)
    dataset_root = _stage_dataset_root(tmp_path, row)
    canonical = parse_canonical_trace(row, dataset_root=dataset_root)
    cache = _StubAtlasMetaCache(AtlasMeta(pos_lo=0.0, pos_hi=10.0, species="mouse"))

    with pytest.raises(ValueError, match="ground_truth_mm"):
        render_sft_answer_only(
            canonical,
            atlas_meta_cache=cache,
            prefer_ground_truth=True,
            ground_truth_mm=None,
        )


# ---------- Task 4: render_rl_prefix / render_isft_prefix ----------


def test_render_rl_prefix_excludes_submit(tmp_path: Path):
    """No assistant message with submit_estimate. Last message is a tool message
    (the final tool response). target_mm == ground_truth_mm. label_mm is None.
    """
    row = _make_synthetic_row(subject_id="M01", final_position_mm=3.6)
    dataset_root = _stage_dataset_root(tmp_path, row)
    canonical = parse_canonical_trace(row, dataset_root=dataset_root)
    cache = _StubAtlasMetaCache(AtlasMeta(pos_lo=0.0, pos_hi=10.0, species="mouse"))

    rendered = render_rl_prefix(canonical, atlas_meta_cache=cache, ground_truth_mm=4.2)

    # No submit_estimate anywhere in the prefix.
    for m in rendered.messages:
        for tc in m.get("tool_calls", []) or []:
            assert tc["function"]["name"] != "submit_estimate"

    # Last message is a tool response (since this trace has tool_steps).
    assert rendered.messages[-1]["role"] == "tool"

    # target_mm carries the reward GT; label_mm stays None.
    assert rendered.target_mm == 4.2
    assert rendered.label_mm is None


def test_render_rl_prefix_includes_tool_steps(tmp_path: Path):
    """For a canonical with 2 tool_steps, the prefix has 2 fetch_atlas assistant
    messages and 2 tool responses, in order.
    """
    row = _make_synthetic_row(subject_id="M01", final_position_mm=3.6)
    dataset_root = _stage_dataset_root(tmp_path, row)
    canonical = parse_canonical_trace(row, dataset_root=dataset_root)
    assert len(canonical.tool_steps) == 2
    cache = _StubAtlasMetaCache(AtlasMeta(pos_lo=0.0, pos_hi=10.0, species="mouse"))

    rendered = render_rl_prefix(canonical, atlas_meta_cache=cache, ground_truth_mm=4.2)

    assistant_msgs = [m for m in rendered.messages if m["role"] == "assistant"]
    tool_msgs = [m for m in rendered.messages if m["role"] == "tool"]
    assert len(assistant_msgs) == 2
    assert len(tool_msgs) == 2
    # All assistant tool_calls are fetch_atlas (no terminal submit).
    for m in assistant_msgs:
        assert m["tool_calls"][0]["function"]["name"] == "fetch_atlas"

    # Order: assistant_0 → tool_0 → assistant_1 → tool_1 (after system + user).
    # Verify by call_id pairing.
    assert assistant_msgs[0]["tool_calls"][0]["id"] == "call_0"
    assert tool_msgs[0]["tool_call_id"] == "call_0"
    assert assistant_msgs[1]["tool_calls"][0]["id"] == "call_1"
    assert tool_msgs[1]["tool_call_id"] == "call_1"

    # Full sequence: system, user, assistant_0, tool_0, assistant_1, tool_1.
    roles = [m["role"] for m in rendered.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool"]


def test_render_isft_prefix_same_prefix_diff_label(tmp_path: Path):
    """render_isft_prefix(t).messages == render_rl_prefix(t, ground_truth_mm=X).messages
    (modulo PIL identity; compare via _message_fingerprint). label_mm equals teacher's
    final_answer.position_mm. target_mm is None.
    """
    row = _make_synthetic_row(subject_id="M01", final_position_mm=3.6)
    dataset_root = _stage_dataset_root(tmp_path, row)
    canonical = parse_canonical_trace(row, dataset_root=dataset_root)
    cache = _StubAtlasMetaCache(AtlasMeta(pos_lo=0.0, pos_hi=10.0, species="mouse"))

    rl = render_rl_prefix(canonical, atlas_meta_cache=cache, ground_truth_mm=4.2)
    isft = render_isft_prefix(canonical, atlas_meta_cache=cache)

    # Messages compare byte-for-byte modulo PIL identity.
    assert len(rl.messages) == len(isft.messages)
    for i, (rl_msg, isft_msg) in enumerate(zip(rl.messages, isft.messages)):
        rl_fp = _message_fingerprint(rl_msg)
        isft_fp = _message_fingerprint(isft_msg)
        assert rl_fp == isft_fp, f"message[{i}] diverged: {rl_fp!r} vs {isft_fp!r}"

    # Tools and image_paths also identical.
    assert rl.tools == isft.tools
    assert rl.image_paths == isft.image_paths

    # The mode-specific label fields differ.
    assert isft.label_mm == canonical.final_answer.position_mm  # 3.6
    assert isft.target_mm is None
    assert rl.target_mm == 4.2
    assert rl.label_mm is None


def test_render_prefix_handles_empty_tool_steps(tmp_path: Path):
    """If canonical has zero tool_steps (just user prompt and terminal submit),
    the prefix is just system + user. Last message is the user message.
    """
    row = _make_submit_only_row(subject_id="M03")
    dataset_root = _stage_dataset_root(tmp_path, row)
    canonical = parse_canonical_trace(row, dataset_root=dataset_root)
    assert len(canonical.tool_steps) == 0
    cache = _StubAtlasMetaCache(AtlasMeta(pos_lo=0.0, pos_hi=10.0, species="mouse"))

    rl = render_rl_prefix(canonical, atlas_meta_cache=cache, ground_truth_mm=4.0)

    # Only system + user; nothing else.
    assert [m["role"] for m in rl.messages] == ["system", "user"]
    # Last message is the user message.
    assert rl.messages[-1]["role"] == "user"
    # image_paths is just the query images (no tool results to extend).
    assert rl.image_paths == list(canonical.query_image_paths)

    # Same check for the iSFT variant.
    isft = render_isft_prefix(canonical, atlas_meta_cache=cache)
    assert [m["role"] for m in isft.messages] == ["system", "user"]
    assert isft.label_mm == canonical.final_answer.position_mm
    assert isft.target_mm is None


# ---------- Task 5: TraceIterator + AtlasFetchJitter ----------


from langslice_traces.augmentations import AtlasFetchJitter, _snap_to_grid  # noqa: E402
from langslice_traces.iterator import TraceIterator  # noqa: E402


def _make_synthetic_manifest_row(
    *,
    record_id: str,
    subject_id: str,
    image_rel: str,
    position_mm: float,
    plane: str = "coronal",
    atlas: str = "allen_mouse_25um",
) -> dict[str, Any]:
    """Build one TraceManifestRow-compatible JSON record (kind='single').

    Mirrors the shape produced by the trace-collection driver: a single-slice
    job with an explicit ``subject_id`` carried in ``metadata`` (since the
    top-level manifest schema doesn't have one).
    """
    return {
        "id": record_id,
        "kind": "single",
        "image": image_rel,
        "atlas": atlas,
        "plane": plane,
        "position_mm": position_mm,
        "metadata": {"subject_id": subject_id},
    }


def _stage_corpus_jsonl(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    """Stage a corpus JSONL + all referenced image files under tmp_path/dataset."""
    dataset_root = tmp_path / "dataset"
    for row in rows:
        for rel in row["query_image_paths"]:
            _write_tiny_png(dataset_root / rel, color=(123, 45, 67))
        for step in row["trace"][:-1]:
            for j, rel in enumerate(step["tool_result"]["image_paths"]):
                _write_tiny_png(
                    dataset_root / rel,
                    color=(10 + j * 30, 20 + j * 20, 200 - j * 25),
                )
    jsonl_path = dataset_root / "examples.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return jsonl_path


def _stub_cache() -> _StubAtlasMetaCache:
    return _StubAtlasMetaCache(AtlasMeta(pos_lo=0.0, pos_hi=10.0, species="mouse"))


def test_iterator_seeding_deterministic(tmp_path: Path):
    """Same seed + same corpus + same augmentations -> identical messages."""
    rows = [
        _make_synthetic_row(subject_id="M01", final_position_mm=3.6),
        _make_synthetic_row(subject_id="M02", final_position_mm=4.2),
    ]
    jsonl_path = _stage_corpus_jsonl(tmp_path, rows)

    grid = [round(x * 0.05, 2) for x in range(0, 200)]  # 0.00, 0.05, ..., 9.95

    def resolve(_atlas: str, _plane: str) -> list[float]:
        return grid

    aug_config = {"atlas_fetch_jitter": {"sigma_mm": 0.2, "max_calls_jittered": 2}}

    it_a = TraceIterator(
        jsonl_path,
        mode="sft_full",
        seed=1234,
        augmentations=aug_config,
        atlas_meta_cache=_stub_cache(),
        grid_resolver=resolve,
    )
    it_b = TraceIterator(
        jsonl_path,
        mode="sft_full",
        seed=1234,
        augmentations=aug_config,
        atlas_meta_cache=_stub_cache(),
        grid_resolver=resolve,
    )

    out_a = list(it_a)
    out_b = list(it_b)
    assert len(out_a) == len(out_b) == 2
    for ex_a, ex_b in zip(out_a, out_b):
        assert len(ex_a.messages) == len(ex_b.messages)
        for msg_a, msg_b in zip(ex_a.messages, ex_b.messages):
            assert _message_fingerprint(msg_a) == _message_fingerprint(msg_b)


def test_iterator_mode_dispatch_sft_full(tmp_path: Path):
    """sft_full mode yields the same shape as render_sft_full directly."""
    row = _make_synthetic_row(subject_id="M01", final_position_mm=3.6)
    jsonl_path = _stage_corpus_jsonl(tmp_path, [row])

    cache = _stub_cache()
    it = TraceIterator(
        jsonl_path,
        mode="sft_full",
        atlas_meta_cache=cache,
    )
    examples = list(it)
    assert len(examples) == 1
    ex = examples[0]
    # Last assistant message is the terminal submit_estimate.
    assistant_msgs = [m for m in ex.messages if m["role"] == "assistant"]
    last_call = assistant_msgs[-1]["tool_calls"][0]
    assert last_call["function"]["name"] == "submit_estimate"
    # target_mm and label_mm both None (SFT renderer).
    assert ex.target_mm is None
    assert ex.label_mm is None


def test_iterator_mode_dispatch_sft_answer_only(tmp_path: Path):
    """sft_answer_only mode skips tool_steps; one assistant message only."""
    row = _make_synthetic_row(subject_id="M01", final_position_mm=3.6)
    jsonl_path = _stage_corpus_jsonl(tmp_path, [row])

    it = TraceIterator(
        jsonl_path,
        mode="sft_answer_only",
        atlas_meta_cache=_stub_cache(),
    )
    ex = next(iter(it))
    assistant_msgs = [m for m in ex.messages if m["role"] == "assistant"]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0]["tool_calls"][0]["function"]["name"] == "submit_estimate"
    # No tool responses appear.
    assert all(m["role"] != "tool" for m in ex.messages)


def test_iterator_mode_dispatch_rl_prefix_requires_manifest():
    """rl_prefix mode without manifest path raises ValueError at construction."""
    with pytest.raises(ValueError, match="manifest"):
        TraceIterator(
            Path("/nonexistent"),
            mode="rl_prefix",
            atlas_meta_cache=_stub_cache(),
            manifest=None,
        )


def test_iterator_mode_dispatch_rl_prefix_joins_manifest_gt(tmp_path: Path):
    """rl_prefix mode looks up GT from the manifest by (plane, subject_id, stem)."""
    row = _make_synthetic_row(subject_id="M01", final_position_mm=3.6)
    jsonl_path = _stage_corpus_jsonl(tmp_path, [row])

    # Manifest row: subject_id under metadata; image stem matches query.
    manifest_row = _make_synthetic_manifest_row(
        record_id="job_M01",
        subject_id="M01",
        image_rel="queries/single_M01.jpg",
        position_mm=4.25,
    )
    manifest_path = tmp_path / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(manifest_row) + "\n")

    it = TraceIterator(
        jsonl_path,
        mode="rl_prefix",
        atlas_meta_cache=_stub_cache(),
        manifest=manifest_path,
    )
    ex = next(iter(it))
    assert ex.target_mm == 4.25
    # No submit step in prefix.
    for m in ex.messages:
        for tc in m.get("tool_calls", []) or []:
            assert tc["function"]["name"] != "submit_estimate"


def test_iterator_mode_dispatch_isft_prefix(tmp_path: Path):
    """isft_prefix mode sets label_mm to teacher's final_answer.position_mm."""
    row = _make_synthetic_row(subject_id="M01", final_position_mm=3.6)
    jsonl_path = _stage_corpus_jsonl(tmp_path, [row])

    it = TraceIterator(
        jsonl_path,
        mode="isft_prefix",
        atlas_meta_cache=_stub_cache(),
    )
    ex = next(iter(it))
    assert ex.label_mm == 3.6
    assert ex.target_mm is None
    # No submit in prefix.
    for m in ex.messages:
        for tc in m.get("tool_calls", []) or []:
            assert tc["function"]["name"] != "submit_estimate"


def test_iterator_unknown_mode_raises():
    """An unknown mode string raises ValueError at construction."""
    with pytest.raises(ValueError, match="unknown mode"):
        TraceIterator(
            Path("/nonexistent"),
            mode="bogus_mode",  # type: ignore[arg-type]
            atlas_meta_cache=_stub_cache(),
        )


def test_iterator_atlas_meta_cache_required():
    """Constructor refuses atlas_meta_cache=None for any mode."""
    with pytest.raises(ValueError, match="atlas_meta_cache"):
        TraceIterator(
            Path("/nonexistent"),
            mode="sft_full",
            atlas_meta_cache=None,
        )


def test_iterator_prefer_ground_truth_requires_manifest():
    """sft_answer_only + prefer_ground_truth=True needs a manifest at construction.

    Previously this failed lazily (mid-iteration) when the renderer received
    ``ground_truth_mm=None``. Now it must fail at ``__init__`` so misconfigured
    training runs are caught at startup, not after partial data flows through.
    """
    with pytest.raises(ValueError, match="prefer_ground_truth"):
        TraceIterator(
            Path("/nonexistent"),
            mode="sft_answer_only",
            atlas_meta_cache=_stub_cache(),
            prefer_ground_truth=True,
            manifest=None,
        )


def test_iterator_unknown_augmentation_raises():
    """An unknown augmentation key (e.g. deferred clahe_mix_toggle) raises NotImplementedError.

    Asserts the iterator fails loud when callers try to use planned-but-deferred
    augmentation keys, instead of silently dropping them.
    """
    with pytest.raises(NotImplementedError, match="clahe_mix_toggle"):
        TraceIterator(
            Path("/nonexistent"),
            mode="sft_full",
            atlas_meta_cache=_stub_cache(),
            augmentations={"clahe_mix_toggle": {"probability": 0.3}},
        )


def test_iterator_iterable_corpus_reiterable(tmp_path: Path):
    """A one-shot generator corpus can still be iterated multiple times.

    The constructor materializes any non-Path iterable so that the second call
    to ``iter(my_iterator)`` does not silently yield nothing — the failure
    mode this guards against is a real one-shot generator (not a list, which
    is already re-iterable). Using a generator here actually exercises the
    materialization path.
    """
    row = _make_synthetic_row(subject_id="M01", final_position_mm=3.6)
    dataset_root = _stage_dataset_root(tmp_path, row)
    canonical = parse_canonical_trace(row, dataset_root=dataset_root)

    # Feed a one-shot generator. Without the materialization fix, the second
    # ``iter(it)`` below would yield nothing.
    gen = iter([canonical])
    it = TraceIterator(
        gen,
        mode="sft_full",
        atlas_meta_cache=_stub_cache(),
    )

    # The constructor eagerly consumed the generator into a list.
    assert list(gen) == []

    out_a = list(it)
    out_b = list(it)
    assert len(out_a) == 1
    assert len(out_b) == 1
    # Output must be byte-identical across the two passes (modulo PIL identity).
    for ex_a, ex_b in zip(out_a, out_b):
        assert len(ex_a.messages) == len(ex_b.messages)
        for msg_a, msg_b in zip(ex_a.messages, ex_b.messages):
            assert _message_fingerprint(msg_a) == _message_fingerprint(msg_b)
        assert ex_a.image_paths == ex_b.image_paths


def test_iterator_augmentation_missing_required_field():
    """Augmentation config missing a required sub-field fails at __init__.

    Previously this surfaced as a bare ``KeyError: 'sigma_mm'`` mid-iteration
    when ``_build_augmentations`` tried to read the missing kwarg. The
    validation now runs at construction so misconfigured training runs are
    caught at startup, not after data has begun flowing through.
    """
    with pytest.raises(ValueError, match="sigma_mm"):
        TraceIterator(
            Path("/nonexistent"),
            mode="sft_full",
            atlas_meta_cache=_stub_cache(),
            augmentations={"atlas_fetch_jitter": {}},
            grid_resolver=lambda _atlas, _plane: [],
        )

    # The companion required field is also enforced.
    with pytest.raises(ValueError, match="max_calls_jittered"):
        TraceIterator(
            Path("/nonexistent"),
            mode="sft_full",
            atlas_meta_cache=_stub_cache(),
            augmentations={"atlas_fetch_jitter": {"sigma_mm": 0.05}},
            grid_resolver=lambda _atlas, _plane: [],
        )


# ---------- AtlasFetchJitter behavior ----------


def _make_canonical_with_steps(
    *,
    positions_list: list[list[float]],
    dataset_root: Path,
) -> CanonicalTrace:
    """Build a CanonicalTrace with N tool_steps whose positions_mm are set explicitly.

    Image hydration is not exercised here — these traces are only fed to the
    augmentation, never to a renderer.
    """
    tool_steps = [
        ToolStep(
            call_name="fetch_atlas",
            call_args={"positions_mm": list(positions)},
            result_image_paths=[],
            result_text="",
        )
        for positions in positions_list
    ]
    return CanonicalTrace(
        atlas_name="allen_mouse_25um",
        atlas_version="CCFv3",
        plane="coronal",
        subject_id="M01",
        system_prompt_kind="single_slice",
        bucket=1,
        query_image_paths=["queries/single_M01.jpg"],
        user_prompt_text="Determine this slice's AP.",
        tool_steps=tool_steps,
        final_answer=FinalAnswer(
            name="submit_estimate",
            position_mm=3.6,
            reasoning="test",
        ),
        quality={"accuracy": "in_tolerance"},
        dataset_root=dataset_root,
    )


def test_snap_to_grid_basic():
    """_snap_to_grid returns the nearest grid entry; empty grid raises."""
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    assert _snap_to_grid(0.3, grid) == 0.25
    assert _snap_to_grid(0.4, grid) == 0.5
    assert _snap_to_grid(-100.0, grid) == 0.0
    assert _snap_to_grid(100.0, grid) == 1.0
    with pytest.raises(ValueError, match="non-empty"):
        _snap_to_grid(0.0, [])


def test_atlas_fetch_jitter_snaps_to_grid(tmp_path: Path):
    """Output positions are guaranteed to be members of the resolver's grid."""
    grid = [round(x * 0.1, 1) for x in range(0, 100)]  # 0.0, 0.1, ..., 9.9

    def resolve(atlas: str, plane: str) -> list[float]:
        assert atlas == "allen_mouse_25um"
        assert plane == "coronal"
        return grid

    aug = AtlasFetchJitter(
        sigma_mm=2.0,  # large sigma to force the jitter off-grid every time
        max_calls_jittered=10,
        grid_resolver=resolve,
    )
    canonical = _make_canonical_with_steps(
        positions_list=[[3.0, 5.0], [2.0, 4.0, 6.0]],
        dataset_root=tmp_path,
    )
    rng = random.Random(42)
    out = aug(canonical, rng)

    # Every emitted position must be a member of the grid.
    for step in out.tool_steps:
        for p in step.call_args["positions_mm"]:
            assert p in grid, f"position {p!r} not in grid"


def test_atlas_fetch_jitter_preserves_metadata(tmp_path: Path):
    """Only positions_mm in call_args may differ; everything else stays equal."""
    aug = AtlasFetchJitter(
        sigma_mm=0.1,
        max_calls_jittered=5,
        grid_resolver=None,  # no snap; raw jittered values
    )
    canonical = _make_canonical_with_steps(
        positions_list=[[3.0, 5.0], [2.0, 4.0]],
        dataset_root=tmp_path,
    )
    rng = random.Random(0)
    out = aug(canonical, rng)

    # Top-level identity preservation.
    assert out is not canonical  # new instance
    assert out.atlas_name == canonical.atlas_name
    assert out.atlas_version == canonical.atlas_version
    assert out.plane == canonical.plane
    assert out.subject_id == canonical.subject_id
    assert out.system_prompt_kind == canonical.system_prompt_kind
    assert out.bucket == canonical.bucket
    assert out.query_image_paths == canonical.query_image_paths
    assert out.user_prompt_text == canonical.user_prompt_text
    assert out.final_answer == canonical.final_answer
    assert out.quality == canonical.quality
    assert out.dataset_root == canonical.dataset_root

    # Same number of tool_steps; call_name, result_image_paths, result_text unchanged.
    assert len(out.tool_steps) == len(canonical.tool_steps)
    for orig, new in zip(canonical.tool_steps, out.tool_steps):
        assert new.call_name == orig.call_name
        assert new.result_image_paths == orig.result_image_paths
        assert new.result_text == orig.result_text
        # Only positions_mm may have changed; other call_args keys must persist.
        for k, v in orig.call_args.items():
            if k == "positions_mm":
                continue
            assert new.call_args[k] == v

    # Input was NOT mutated.
    assert canonical.tool_steps[0].call_args["positions_mm"] == [3.0, 5.0]
    assert canonical.tool_steps[1].call_args["positions_mm"] == [2.0, 4.0]


def test_atlas_fetch_jitter_respects_max_calls_jittered(tmp_path: Path):
    """With max_calls_jittered=1 on 3 tool_steps, exactly 1 step's positions differ.

    Compares each output step's positions to the original; exactly one must be
    different and two must be identical.
    """
    aug = AtlasFetchJitter(
        sigma_mm=1.0,  # large enough that jittered values won't coincidentally equal originals
        max_calls_jittered=1,
        grid_resolver=None,
    )
    canonical = _make_canonical_with_steps(
        positions_list=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dataset_root=tmp_path,
    )
    rng = random.Random(7)
    out = aug(canonical, rng)

    differences = [
        out.tool_steps[i].call_args["positions_mm"]
        != canonical.tool_steps[i].call_args["positions_mm"]
        for i in range(len(canonical.tool_steps))
    ]
    assert sum(differences) == 1, f"expected exactly 1 jittered step, got {sum(differences)}: {differences}"


def test_atlas_fetch_jitter_no_tool_steps_passthrough(tmp_path: Path):
    """Empty tool_steps -> canonical returned unchanged (no exception)."""
    aug = AtlasFetchJitter(sigma_mm=0.1, max_calls_jittered=3, grid_resolver=None)
    canonical = _make_canonical_with_steps(positions_list=[], dataset_root=tmp_path)
    rng = random.Random(0)
    out = aug(canonical, rng)
    # Same instance is acceptable (no transformation needed).
    assert out is canonical
    assert out.tool_steps == []


# ---------- Task 6: procedural trace generator ----------


def _synthetic_grid_01() -> list[float]:
    """A 0.1mm-spaced grid spanning [0.0, 10.0] inclusive."""
    return [round(i * 0.1, 1) for i in range(101)]


def _lane_a_kwargs(*, gt: float = 5.0, plane: str = "coronal", seed: int = 0) -> dict[str, Any]:
    return {
        "image_path": "queries/img.jpg",
        "ground_truth_mm": gt,
        "plane": plane,
        "atlas_name": "allen_mouse_25um",
        "atlas_version": "CCFv3",
        "subject_id": "M01",
        "grid": _synthetic_grid_01(),
        "strategy": "lane_a_prefix",
        "rng": random.Random(seed),
    }


def _lane_b_kwargs(*, gt: float = 5.0, plane: str = "coronal", seed: int = 0) -> dict[str, Any]:
    return {
        "image_path": "queries/img.jpg",
        "ground_truth_mm": gt,
        "plane": plane,
        "atlas_name": "allen_mouse_25um",
        "atlas_version": "CCFv3",
        "subject_id": "M01",
        "grid": _synthetic_grid_01(),
        "strategy": "lane_b_broad_slate",
        "rng": random.Random(seed),
    }


# ---------- Lane A structural ----------


def test_lane_a_prefix_two_steps_majority():
    """Empirically 99% of corpus traces have 2 tool_steps. Allow >=90% margin."""
    grid = _synthetic_grid_01()
    rng = random.Random(11)
    two_step = 0
    n = 200
    for _ in range(n):
        gt = rng.uniform(1.0, 9.0)
        trace = generate_trace(
            image_path="queries/img.jpg",
            ground_truth_mm=gt,
            plane="coronal",
            atlas_name="allen_mouse_25um",
            atlas_version="CCFv3",
            subject_id="M01",
            grid=grid,
            strategy="lane_a_prefix",
            rng=rng,
        )
        if len(trace.tool_steps) == 2:
            two_step += 1
    frac = two_step / n
    assert frac >= 0.90, f"only {frac:.2%} of traces had 2 tool_steps (expected >=90%)"


def test_lane_a_prefix_step1_center_not_always_gt():
    """Real corpus has 50% of step-1 brackets offset from GT. Allow >=40% margin."""
    grid = _synthetic_grid_01()
    rng = random.Random(13)
    offset_count = 0
    measured = 0
    n = 200
    for _ in range(n):
        gt = rng.uniform(1.0, 9.0)
        trace = generate_trace(
            image_path="queries/img.jpg",
            ground_truth_mm=gt,
            plane="coronal",
            atlas_name="allen_mouse_25um",
            atlas_version="CCFv3",
            subject_id="M01",
            grid=grid,
            strategy="lane_a_prefix",
            rng=rng,
        )
        if len(trace.tool_steps) < 2:
            continue
        measured += 1
        step1_positions = trace.tool_steps[1].call_args["positions_mm"]
        center = sum(step1_positions) / len(step1_positions)
        if abs(center - gt) > 0.01:
            offset_count += 1
    frac = offset_count / max(measured, 1)
    assert frac >= 0.40, f"only {frac:.2%} of step-1 centers were offset from GT"


def test_lane_a_prefix_final_answer_is_none():
    """Lane A always produces a prefix; final_answer must be None."""
    trace = generate_trace(**_lane_a_kwargs())
    assert trace.final_answer is None


def test_lane_a_prefix_grid_compliance():
    """Every position in every tool_step is an exact element of the grid."""
    grid = _synthetic_grid_01()
    rng = random.Random(17)
    for _ in range(50):
        trace = generate_trace(
            image_path="queries/img.jpg",
            ground_truth_mm=rng.uniform(1.0, 9.0),
            plane="coronal",
            atlas_name="allen_mouse_25um",
            atlas_version="CCFv3",
            subject_id="M01",
            grid=grid,
            strategy="lane_a_prefix",
            rng=rng,
        )
        for step in trace.tool_steps:
            for p in step.call_args["positions_mm"]:
                assert p in grid, f"position {p!r} not in grid"


# ---------- Lane B structural ----------


def test_lane_b_broad_slate_contains_gt():
    """min(positions) <= gt <= max(positions) always holds."""
    grid = _synthetic_grid_01()
    rng = random.Random(19)
    n = 200
    for _ in range(n):
        gt = rng.uniform(1.0, 9.0)
        trace = generate_trace(
            image_path="queries/img.jpg",
            ground_truth_mm=gt,
            plane="coronal",
            atlas_name="allen_mouse_25um",
            atlas_version="CCFv3",
            subject_id="M01",
            grid=grid,
            strategy="lane_b_broad_slate",
            rng=rng,
        )
        positions = trace.tool_steps[0].call_args["positions_mm"]
        assert min(positions) <= gt <= max(positions), (
            f"slate does not bracket GT: gt={gt}, positions={positions}"
        )


def test_lane_b_broad_slate_gt_not_centered():
    """At least 60% of slates have |gt_fraction - 0.5| > 0.1."""
    grid = _synthetic_grid_01()
    rng = random.Random(23)
    off_center = 0
    n = 200
    for _ in range(n):
        gt = rng.uniform(2.0, 8.0)  # leave room so the broad slate fits
        trace = generate_trace(
            image_path="queries/img.jpg",
            ground_truth_mm=gt,
            plane="coronal",
            atlas_name="allen_mouse_25um",
            atlas_version="CCFv3",
            subject_id="M01",
            grid=grid,
            strategy="lane_b_broad_slate",
            rng=rng,
        )
        frac = trace.quality["gt_fraction_in_slate"]
        if abs(frac - 0.5) > 0.1:
            off_center += 1
    proportion = off_center / n
    assert proportion >= 0.60, f"only {proportion:.2%} of slates were off-center"


def test_lane_b_broad_slate_one_step():
    """Every Lane B trace has exactly 1 tool_step."""
    grid = _synthetic_grid_01()
    rng = random.Random(29)
    for _ in range(50):
        trace = generate_trace(
            image_path="queries/img.jpg",
            ground_truth_mm=rng.uniform(2.0, 8.0),
            plane="coronal",
            atlas_name="allen_mouse_25um",
            atlas_version="CCFv3",
            subject_id="M01",
            grid=grid,
            strategy="lane_b_broad_slate",
            rng=rng,
        )
        assert len(trace.tool_steps) == 1


def test_lane_b_broad_slate_final_answer_is_none():
    """Lane B always produces a prefix; final_answer must be None."""
    trace = generate_trace(**_lane_b_kwargs())
    assert trace.final_answer is None


def test_lane_b_broad_slate_grid_compliance():
    """Every position in the slate is an exact element of the grid."""
    grid = _synthetic_grid_01()
    rng = random.Random(31)
    for _ in range(50):
        trace = generate_trace(
            image_path="queries/img.jpg",
            ground_truth_mm=rng.uniform(2.0, 8.0),
            plane="coronal",
            atlas_name="allen_mouse_25um",
            atlas_version="CCFv3",
            subject_id="M01",
            grid=grid,
            strategy="lane_b_broad_slate",
            rng=rng,
        )
        for p in trace.tool_steps[0].call_args["positions_mm"]:
            assert p in grid, f"position {p!r} not in grid"


# ---------- Lane A realism distributions ----------


def _pearson_r(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation between two same-length lists."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return num / (dx * dy)


def test_lane_a_prefix_step0_gt_correlation():
    """Step-0 center vs GT should correlate r > 0.85.

    Real corpus shows r=0.98; allow statistical slack for a 1000-sample test.
    """
    grid = _synthetic_grid_01()
    rng = random.Random(37)
    gts: list[float] = []
    centers: list[float] = []
    n = 1000
    while len(gts) < n:
        gt = rng.uniform(0.5, 9.5)
        trace = generate_trace(
            image_path="queries/img.jpg",
            ground_truth_mm=gt,
            plane="coronal",
            atlas_name="allen_mouse_25um",
            atlas_version="CCFv3",
            subject_id="M01",
            grid=grid,
            strategy="lane_a_prefix",
            rng=rng,
        )
        if not trace.tool_steps:
            continue
        step0 = trace.tool_steps[0].call_args["positions_mm"]
        center = sum(step0) / len(step0)
        gts.append(gt)
        centers.append(center)
    r = _pearson_r(centers, gts)
    assert r > 0.85, f"r(step0_center, gt) = {r:.4f} (expected > 0.85)"


def test_lane_a_prefix_integer_position_rate_step0():
    """Fraction of step-0 positions that are integer mm should match the
    empirical P_ROUNDNESS_STEP0 integer weight (0.495) plus a small lift
    from finer tiers that happen to land on integers. Allow [0.35, 0.65].
    """
    grid = _synthetic_grid_01()
    rng = random.Random(41)
    integer_count = 0
    total = 0
    while total < 1000:
        gt = rng.uniform(0.5, 9.5)
        trace = generate_trace(
            image_path="queries/img.jpg",
            ground_truth_mm=gt,
            plane="coronal",
            atlas_name="allen_mouse_25um",
            atlas_version="CCFv3",
            subject_id="M01",
            grid=grid,
            strategy="lane_a_prefix",
            rng=rng,
        )
        if not trace.tool_steps:
            continue
        step0 = trace.tool_steps[0].call_args["positions_mm"]
        for p in step0:
            total += 1
            if abs(p - round(p)) < 1e-6:
                integer_count += 1
            if total >= 1000:
                break
    rate = integer_count / total
    assert 0.35 <= rate <= 0.65, f"step-0 integer rate = {rate:.4f} (expected in [0.35, 0.65])"


def _kl_divergence(observed: dict[int, float], expected: dict[int, float]) -> float:
    """KL divergence (observed || expected). Skip terms where expected is 0."""
    eps = 1e-12
    total = 0.0
    for k, p_obs in observed.items():
        if p_obs <= 0.0:
            continue
        p_exp = expected.get(k, eps)
        total += p_obs * math.log((p_obs + eps) / (p_exp + eps))
    return total


def test_lane_a_prefix_nfetch_step0_within_3sigma():
    """Empirical n_fetch distribution for step-0 should match P_NFETCH_STEP0.

    Uses KL divergence < 0.1 as the agreement threshold over 1000 traces.
    """
    grid = _synthetic_grid_01()
    rng = random.Random(43)
    counts: Counter[int] = Counter()
    expected = {k: v for k, v in P_NFETCH_STEP0}
    n_traces = 0
    while n_traces < 1000:
        gt = rng.uniform(1.0, 9.0)
        trace = generate_trace(
            image_path="queries/img.jpg",
            ground_truth_mm=gt,
            plane="coronal",
            atlas_name="allen_mouse_25um",
            atlas_version="CCFv3",
            subject_id="M01",
            grid=grid,
            strategy="lane_a_prefix",
            rng=rng,
        )
        if not trace.tool_steps:
            continue
        n_traces += 1
        # Use raw (pre-snap) intent length: positions_mm reflects post-snap
        # dedupe, which can shrink counts. Skip traces where dedupe affected
        # the count substantively by accepting only same-as-sampled counts...
        # actually that requires hooking into the generator. Easier path:
        # measure the post-dedupe count, since dedupe is rare on dense grids
        # and shifts mass at most by one bin.
        counts[len(trace.tool_steps[0].call_args["positions_mm"])] += 1
    total = sum(counts.values())
    observed = {k: c / total for k, c in counts.items()}
    kl = _kl_divergence(observed, expected)
    assert kl < 0.1, f"KL(observed || P_NFETCH_STEP0) = {kl:.4f}; observed={observed}"


# ---------- Determinism ----------


def test_lane_a_seeded_deterministic():
    """Two calls with Random(7) produce identical CanonicalTrace."""
    a = generate_trace(**_lane_a_kwargs(seed=7))
    b = generate_trace(**_lane_a_kwargs(seed=7))
    assert a == b
    assert a.tool_steps == b.tool_steps


def test_lane_b_seeded_deterministic():
    """Two calls with Random(7) produce identical CanonicalTrace."""
    a = generate_trace(**_lane_b_kwargs(seed=7))
    b = generate_trace(**_lane_b_kwargs(seed=7))
    assert a == b
    assert a.tool_steps == b.tool_steps


# ---------- Sort + default-bucket regression tests ----------


def test_lane_a_positions_sorted():
    """Every step's positions_mm must be sorted ascending.

    Per-position roundness draws + collision-fallback can produce out-of-order
    positions; real corpus is essentially 100% sorted (0.03% unsorted), so the
    generator sorts before emit. Regression test for that fix.
    """
    grid = _synthetic_grid_01()
    rng = random.Random(7)
    n = 1000
    for _ in range(n):
        gt = rng.uniform(1.0, 9.0)
        trace = generate_trace(
            image_path="queries/img.jpg",
            ground_truth_mm=gt,
            plane="coronal",
            atlas_name="allen_mouse_25um",
            atlas_version="CCFv3",
            subject_id="M01",
            grid=grid,
            strategy="lane_a_prefix",
            rng=rng,
        )
        for step_idx, step in enumerate(trace.tool_steps):
            positions = step.call_args["positions_mm"]
            for i in range(len(positions) - 1):
                assert positions[i] <= positions[i + 1], (
                    f"Lane A step {step_idx} unsorted at index {i}: {positions}"
                )


def test_lane_b_positions_sorted():
    """Lane B slate positions must be sorted ascending."""
    grid = _synthetic_grid_01()
    rng = random.Random(7)
    n = 1000
    for _ in range(n):
        gt = rng.uniform(2.0, 8.0)
        trace = generate_trace(
            image_path="queries/img.jpg",
            ground_truth_mm=gt,
            plane="coronal",
            atlas_name="allen_mouse_25um",
            atlas_version="CCFv3",
            subject_id="M01",
            grid=grid,
            strategy="lane_b_broad_slate",
            rng=rng,
        )
        positions = trace.tool_steps[0].call_args["positions_mm"]
        for i in range(len(positions) - 1):
            assert positions[i] <= positions[i + 1], (
                f"Lane B slate unsorted at index {i}: {positions}"
            )


def test_generate_trace_default_bucket_is_1():
    """Default bucket must be 1 (corpus is 100% bucket=1)."""
    kwargs = _lane_a_kwargs()
    # The helper sets no bucket explicitly; we drop any "bucket" key if it's
    # ever added by a future kwargs helper, so the default is exercised.
    kwargs.pop("bucket", None)
    trace = generate_trace(**kwargs)
    assert trace.bucket == 1


# ---------- Format / contract ----------


def test_image_path_format():
    """tool_result image paths match ``atlas/<atlas>/<plane>/<p:.2f>mm.jpg``."""
    grid = _synthetic_grid_01()
    atlas = "allen_mouse_25um"
    plane = "coronal"
    for strategy in ("lane_a_prefix", "lane_b_broad_slate"):
        trace = generate_trace(
            image_path="queries/img.jpg",
            ground_truth_mm=5.0,
            plane=plane,
            atlas_name=atlas,
            atlas_version="CCFv3",
            subject_id="M01",
            grid=grid,
            strategy=strategy,  # type: ignore[arg-type]
            rng=random.Random(53),
        )
        for step in trace.tool_steps:
            positions = step.call_args["positions_mm"]
            image_paths = step.result_image_paths
            assert len(positions) == len(image_paths)
            for p, path in zip(positions, image_paths):
                expected = f"atlas/{atlas}/{plane}/{p:.2f}mm.jpg"
                assert path == expected, f"mismatch: {path!r} vs {expected!r}"


def test_tool_result_text_format():
    """n=1 step says ``1 atlas section`` (singular); n>=2 says ``N atlas sections``."""
    grid = _synthetic_grid_01()
    # Generate a sample of traces and check whichever singular/plural cases occur.
    rng = random.Random(59)
    saw_singular = False
    saw_plural = False
    for _ in range(50):
        trace = generate_trace(
            image_path="queries/img.jpg",
            ground_truth_mm=rng.uniform(1.0, 9.0),
            plane="coronal",
            atlas_name="allen_mouse_25um",
            atlas_version="CCFv3",
            subject_id="M01",
            grid=grid,
            strategy="lane_a_prefix",
            rng=rng,
        )
        for step in trace.tool_steps:
            n = len(step.call_args["positions_mm"])
            text = step.result_text
            if n == 1:
                saw_singular = True
                assert text.startswith("Fetched 1 atlas section:"), text
                assert "atlas sections" not in text
            else:
                saw_plural = True
                assert text.startswith(f"Fetched {n} atlas sections:"), text
    # The generator drops <2-position steps; we expect plural to dominate.
    assert saw_plural, "no plural-form result_text was produced"
    # Singular is rare-by-design (dropped pre-emit). Don't assert seen.


def test_user_prompt_template():
    """User prompt for each plane uses the matching axis name (AP/ML/DV)."""
    grid = _synthetic_grid_01()
    atlas = "allen_mouse_25um"
    expected = {"coronal": "AP", "sagittal": "ML", "horizontal": "DV"}
    for plane, axis in expected.items():
        for strategy in ("lane_a_prefix", "lane_b_broad_slate"):
            trace = generate_trace(
                image_path="queries/img.jpg",
                ground_truth_mm=5.0,
                plane=plane,  # type: ignore[arg-type]
                atlas_name=atlas,
                atlas_version="CCFv3",
                subject_id="M01",
                grid=grid,
                strategy=strategy,  # type: ignore[arg-type]
                rng=random.Random(61),
            )
            assert (
                trace.user_prompt_text
                == f"Determine this {plane} slice's {axis} position in the {atlas} atlas."
            )


# ---------- Renderer integration ----------


def _stage_generated_trace(tmp_path: Path, trace: CanonicalTrace) -> None:
    """Drop a placeholder PNG at every image path referenced by ``trace``."""
    dataset_root = tmp_path / "dataset"
    for rel in trace.query_image_paths:
        _write_tiny_png(dataset_root / rel, color=(100, 100, 100))
    for step in trace.tool_steps:
        for j, rel in enumerate(step.result_image_paths):
            _write_tiny_png(
                dataset_root / rel,
                color=(10 + j * 30, 20 + j * 20, 200 - j * 25),
            )
    trace.dataset_root = dataset_root


def test_lane_a_prefix_renders_via_isft_prefix(tmp_path: Path):
    """A Lane A trace passes cleanly through render_isft_prefix."""
    gt = 5.0
    trace = generate_trace(**_lane_a_kwargs(gt=gt))
    _stage_generated_trace(tmp_path, trace)
    cache = _StubAtlasMetaCache(AtlasMeta(pos_lo=0.0, pos_hi=10.0, species="mouse"))

    rendered = render_isft_prefix(trace, atlas_meta_cache=cache)
    assert len(rendered.messages) > 0
    assert len(rendered.image_paths) >= 1
    # No teacher final answer was emitted, so label_mm stays None.
    assert rendered.label_mm is None


def test_lane_a_prefix_renders_via_rl_prefix(tmp_path: Path):
    """A Lane A trace passes cleanly through render_rl_prefix with target_mm == gt."""
    gt = 5.0
    trace = generate_trace(**_lane_a_kwargs(gt=gt))
    _stage_generated_trace(tmp_path, trace)
    cache = _StubAtlasMetaCache(AtlasMeta(pos_lo=0.0, pos_hi=10.0, species="mouse"))

    rendered = render_rl_prefix(trace, atlas_meta_cache=cache, ground_truth_mm=gt)
    assert len(rendered.messages) > 0
    assert rendered.target_mm == gt
    # Each tool step in the trace must produce one tool-response message.
    tool_msgs = [m for m in rendered.messages if m["role"] == "tool"]
    assert len(tool_msgs) == len(trace.tool_steps)


def test_lane_b_renders_via_isft_prefix(tmp_path: Path):
    """A Lane B trace passes cleanly through render_isft_prefix."""
    gt = 5.0
    trace = generate_trace(**_lane_b_kwargs(gt=gt))
    _stage_generated_trace(tmp_path, trace)
    cache = _StubAtlasMetaCache(AtlasMeta(pos_lo=0.0, pos_hi=10.0, species="mouse"))

    rendered = render_isft_prefix(trace, atlas_meta_cache=cache)
    assert len(rendered.messages) > 0
    assert rendered.label_mm is None
    tool_msgs = [m for m in rendered.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1  # single broad slate


# ---------- Renderer guards (preserved) ----------


def test_render_sft_full_rejects_none_final_answer(tmp_path: Path):
    """render_sft_full must refuse a CanonicalTrace whose final_answer is None."""
    trace = generate_trace(**_lane_a_kwargs())
    _stage_generated_trace(tmp_path, trace)
    cache = _StubAtlasMetaCache(AtlasMeta(pos_lo=0.0, pos_hi=10.0, species="mouse"))
    with pytest.raises(ValueError, match="final_answer"):
        render_sft_full(trace, atlas_meta_cache=cache)


def test_render_sft_answer_only_rejects_none_final_answer_without_gt_flag(tmp_path: Path):
    """render_sft_answer_only(prefer_ground_truth=False) refuses final_answer=None."""
    trace = generate_trace(**_lane_a_kwargs())
    _stage_generated_trace(tmp_path, trace)
    cache = _StubAtlasMetaCache(AtlasMeta(pos_lo=0.0, pos_hi=10.0, species="mouse"))
    with pytest.raises(ValueError, match="final_answer"):
        render_sft_answer_only(trace, atlas_meta_cache=cache)


def test_load_atlas_grid_parses_basenames():
    """load_atlas_grid round-trips real on-disk atlas embedding cache.

    Gated on the presence of the cache file; skipped in environments where
    the cache hasn't been built.
    """
    pytest.importorskip("torch")
    cache_dir = _REPO / "out" / "atlas_embeddings"
    cache_path = cache_dir / "allen_mouse_25um_coronal.pt"
    if not cache_path.is_file():
        pytest.skip(f"atlas embedding cache not present: {cache_path}")

    grid = load_atlas_grid(cache_dir, "allen_mouse_25um", "coronal")
    # Sorted, dense, well-populated, contains a known atlas mm position.
    assert grid == sorted(grid)
    assert len(grid) > 300
    assert 5.00 in grid


# ---------------------------------------------------------------------------
# canonical_atlas_repo_path — the prefix-unifier helper used by both Lane A
# (synthetic terminal_states) and Lane B (randomized section_state) so the
# trainer's ``repo_root / p`` resolver sees a single convention.
# ---------------------------------------------------------------------------


def test_canonical_atlas_repo_path_idempotent() -> None:
    """An already-canonical path must be returned unchanged. Idempotence
    matters because both consumers wrap every emitted atlas path — a
    non-idempotent helper would double-prefix on cache reload."""
    already = (
        "models/langslice-gemma-4/data/atlas/"
        "allen_mouse_25um/coronal/5.00mm.jpg"
    )
    assert canonical_atlas_repo_path(already) == already


def test_canonical_atlas_repo_path_from_bare() -> None:
    """The generator emits bare ``atlas/...`` paths; the helper must
    prepend the canonical root so the trainer can resolve under
    ``repo_root``."""
    bare = "atlas/X/coronal/3.20mm.jpg"
    out = canonical_atlas_repo_path(bare)
    assert out == "models/langslice-gemma-4/data/atlas/X/coronal/3.20mm.jpg"
    assert out.startswith(CANONICAL_ATLAS_ROOT + "/")


def test_canonical_atlas_repo_path_from_legacy_data_prefix() -> None:
    """Older Lane B JSONL ships with ``data/atlas/...``; the helper must
    rewrite that to the canonical root so reloaded section-state caches
    keep resolving against the on-disk tiles."""
    legacy = "data/atlas/allen_mouse_25um/coronal/0.65mm.jpg"
    out = canonical_atlas_repo_path(legacy)
    assert out == (
        "models/langslice-gemma-4/data/atlas/"
        "allen_mouse_25um/coronal/0.65mm.jpg"
    )


def test_canonical_atlas_repo_path_handles_backslashes() -> None:
    """Windows-encoded paths (back-slashes) must normalize to forward
    slashes and still hit the right rewrite rule."""
    win = r"atlas\allen_mouse_25um\coronal\5.00mm.jpg"
    out = canonical_atlas_repo_path(win)
    assert out == (
        "models/langslice-gemma-4/data/atlas/"
        "allen_mouse_25um/coronal/5.00mm.jpg"
    )


def test_canonical_atlas_repo_path_unknown_prefix_passthrough() -> None:
    """Unrecognized input shapes are returned unchanged — the helper is a
    rewriter, not a validator. (Absolute paths and arbitrary strings can
    flow through without exploding.)"""
    abs_like = "/tmp/whatever/atlas.jpg"
    assert canonical_atlas_repo_path(abs_like) == abs_like
    other = "queries/img_42.png"
    assert canonical_atlas_repo_path(other) == other

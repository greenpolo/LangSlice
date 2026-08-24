"""Positioning tools and the position node.

No live model calls: the agent step is driven by a scripted fake BaseLlm
swapped in through ``LLMRegistry.new_llm``.

The stack is placed against ``SlabAtlas`` (tests/fakes.py) — 20 slices at
1 mm, structures in known index ranges — because ``submit_positions``
checks its end anchors against a real annotation volume.
"""

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from google.adk.models import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from PIL import Image

from langslice.atlas import landmarks
from langslice.linear.whole_brain.engine import build_context
from langslice.linear.whole_brain.nodes import ingest, position
from langslice.linear.whole_brain.position import (
    build_position_prompt,
    build_position_seed_message,
    build_position_tools,
    position_rows,
)
from langslice.linear.whole_brain.state import BrainConfig, StackState
from tests.fakes import SlabAtlas

_RANGE = {"pos_lo": 0.0, "pos_hi": 13.18}

#: Vocabulary the positioning step must not put in front of the model: the
#: coaching that every traced benchmark failure came back to.
BANNED_COACHING = (
    "strategy",
    "compressed",
    "compress",
    "stretched",
    "warning sign",
    "stop sign",
    "believe the anchors",
    "advisory",
    "rigid shift",
)

#: One instance for the module: the landmark caches key on the atlas name, and
#: rebuilding the volume per tool call buys nothing.
_ATLAS = SlabAtlas()

#: The two EXTRA-narrow structures of _ATLAS, 1.0 mm each (7.6% of the
#: atlas's 13.18 mm span) and one per END of the placed test stack — the
#: default end anchors, since they are the only ones that clear
#: MAX_ANCHOR_SPAN_FRACTION. "FA" (2.0-6.0 mm, 30%) and "NS" (2.0-5.0 mm,
#: 23%) are the counterparts used to exercise the too-broad rejection.
_ANCHOR_FIRST = "XA"
_ANCHOR_LAST = "XP"
_BROAD_STRUCTURE = "FA"


@pytest.fixture(autouse=True)
def _clear_landmark_caches():
    landmarks._PRESENCE_CACHE.clear()
    landmarks._STRUCTURE_CACHE.clear()
    yield
    landmarks._PRESENCE_CACHE.clear()
    landmarks._STRUCTURE_CACHE.clear()


class _Actions:
    escalate = False


class _ToolContext:
    def __init__(self):
        self.actions = _Actions()


def _anchors(state: StackState, structure: str | None = None):
    """Valid end anchors for *state*: one narrow structure per END.

    Pass *structure* to use the same one at both ends — how the too-broad and
    unresolvable cases are exercised.
    """
    ordered = state.in_order()
    first = structure or _ANCHOR_FIRST
    last = structure or _ANCHOR_LAST
    return [
        {"id": ordered[0].id, "structure": first, "note": "seen at this end"},
        {"id": ordered[-1].id, "structure": last, "note": "seen at this end"},
    ]


def _stack(folder: Path, n: int = 6, *, placed: bool = True, **kwargs):
    """An ingested stack, optionally with a ladder of positions already on it.

    The unplaced form (``placed=False``) is what the positioning step actually
    receives now: the seed node no longer places anything. When placed, the
    ladder runs 2.0-4.5 mm, so its first section sits inside ``XA``
    (2.0-3.0 mm) and its last inside ``XP`` (4.0-5.0 mm): the end-anchor gate
    passes by default and each test can break exactly one thing.
    """
    for index in range(n):
        Image.new("RGB", (40, 30), (10 * index, 60, 120)).save(
            folder / f"slice_{index:02d}.png"
        )
    kwargs.setdefault("landmark_tools", True)
    ctx = build_context(
        BrainConfig(image_folder=str(folder), **kwargs),
        emit=lambda _m: None,
        atlas_loader=lambda _name: _ATLAS,
    )
    state = StackState()
    asyncio.run(ingest(state, ctx))
    if placed:
        for index, record in enumerate(state.in_order()):
            record.position_mm = 2.0 + 0.5 * index
            record.position_source = "refined"
    return state, ctx


def _tool(box, name: str):
    return next(t for t in box.tools if t.__name__ == name)


def _box(state, ctx):
    return build_position_tools(state, ctx, **_RANGE)


# --- set_positions -------------------------------------------------------


def test_set_positions_writes_a_batch_and_returns_the_position_rows(tmp_path: Path):
    state, ctx = _stack(tmp_path)
    set_positions = _tool(_box(state, ctx), "set_positions")

    result = set_positions(
        [
            {"id": "slice_00.png", "position_mm": 1.1, "confidence": "high"},
            {"id": "slice_01.png", "position_mm": 1.4},
            {"id": "ghost.png", "position_mm": 2.0},
            {"id": "slice_02.png", "position_mm": "not a number"},
        ]
    )

    assert result["status"] == "ok"
    assert [row["id"] for row in result["written"]] == ["slice_00.png", "slice_01.png"]
    assert result["unknown_ids"] == ["ghost.png"]
    assert result["rejected"][0]["id"] == "slice_02.png"
    assert result["clamped"] == []

    first = state.by_id("slice_00.png")
    assert first is not None
    assert first.position_mm == pytest.approx(1.1)
    assert first.position_source == "refined"
    assert first.confidence == "high"
    # An untouched section keeps whatever position it already had.
    second = state.by_id("slice_02.png")
    assert second is not None and second.position_mm == pytest.approx(3.0)

    rows = result["rows"]
    assert [row["id"] for row in rows] == [s.id for s in state.in_order()]
    assert rows[0]["spacing_to_next_mm"] == pytest.approx(0.3)
    assert rows[-1]["spacing_to_next_mm"] is None
    # Data only: no legend, no implied-vs-nominal commentary.
    assert "legend" not in result
    assert "interval_table" not in result
    assert set(rows[0]) == {"index", "id", "position_mm", "spacing_to_next_mm"}


def test_stack_positions_reports_the_stack_as_data(tmp_path: Path):
    state, ctx = _stack(tmp_path, n=5)
    for index, record in enumerate(state.in_order()):
        record.position_mm = 2.0 + 0.3 * index

    result = _tool(_box(state, ctx), "stack_positions")()

    assert result["status"] == "ok"
    assert set(result) == {"status", "rows"}
    rows = result["rows"]
    assert [row["index"] for row in rows] == [0, 1, 2, 3, 4]
    assert [row["id"] for row in rows] == [s.id for s in state.in_order()]
    assert rows[0]["position_mm"] == pytest.approx(2.0)
    assert all(
        row["spacing_to_next_mm"] == pytest.approx(0.3) for row in rows[:-1]
    )
    assert rows[-1]["spacing_to_next_mm"] is None
    # No nominal interval, no implied interval, no verdict of any kind.
    assert not any(
        key in row
        for row in rows
        for key in ("nominal_interval_mm", "implied_interval_mm", "legend", "note")
    )


def test_stack_positions_spans_an_unplaced_gap(tmp_path: Path):
    """Spacing is measured to the next PLACED section, whatever sits between."""
    state, _ctx = _stack(tmp_path, n=4)
    for index, record in enumerate(state.in_order()):
        record.position_mm = 2.0 + 0.3 * index
    state.in_order()[2].position_mm = None

    rows = position_rows(state)

    assert rows[1]["spacing_to_next_mm"] == pytest.approx(0.6)  # 2.3 -> 2.9
    assert rows[2]["position_mm"] is None
    assert rows[2]["spacing_to_next_mm"] is None


def test_stack_positions_on_an_unplaced_stack(tmp_path: Path):
    state, ctx = _stack(tmp_path, placed=False)

    rows = _tool(_box(state, ctx), "stack_positions")()["rows"]

    assert len(rows) == len(state.slices)
    assert all(row["position_mm"] is None for row in rows)
    assert all(row["spacing_to_next_mm"] is None for row in rows)


def test_set_positions_clamps_out_of_range_values(tmp_path: Path):
    state, ctx = _stack(tmp_path)
    set_positions = _tool(_box(state, ctx), "set_positions")

    result = set_positions(
        [
            {"id": "slice_00.png", "position_mm": -3.0},
            {"id": "slice_01.png", "position_mm": 99.0},
        ]
    )

    assert result["status"] == "ok"
    assert [entry["id"] for entry in result["clamped"]] == [
        "slice_00.png",
        "slice_01.png",
    ]
    assert "clamped" in result["warning"]
    assert state.by_id("slice_00.png").position_mm == pytest.approx(0.0)  # type: ignore[union-attr]
    assert state.by_id("slice_01.png").position_mm == pytest.approx(13.18)  # type: ignore[union-attr]


def test_set_positions_rejects_an_empty_or_useless_batch(tmp_path: Path):
    state, ctx = _stack(tmp_path)
    set_positions = _tool(_box(state, ctx), "set_positions")

    assert set_positions([])["error"] == "BAD_ARGS"
    assert set_positions([{"id": "ghost.png", "position_mm": 1.0}])["error"] == (
        "NOTHING_WRITTEN"
    )


# --- interpolate_between -------------------------------------------------


def test_interpolate_between_suggests_without_writing(tmp_path: Path):
    state, ctx = _stack(tmp_path, placed=False)
    interpolate_between = _tool(_box(state, ctx), "interpolate_between")

    result = interpolate_between(
        [
            {"id": "slice_01.png", "position_mm": 2.0},
            {"id": "slice_04.png", "position_mm": 3.5},
            {"id": "ghost.png", "position_mm": 9.0},
            {"id": "slice_05.png", "position_mm": "not a number"},
        ]
    )

    assert result["status"] == "ok"
    assert result["unknown_ids"] == ["ghost.png"]
    assert result["rejected"][0]["id"] == "slice_05.png"
    rows = {row["id"]: row for row in result["suggestions"]}
    assert [row["id"] for row in result["suggestions"]] == [
        s.id for s in state.in_order()
    ]
    # Between the fixed points the spacing is spread evenly.
    assert rows["slice_02.png"]["position_mm"] == pytest.approx(2.5)
    assert rows["slice_03.png"]["position_mm"] == pytest.approx(3.0)
    assert rows["slice_01.png"]["source"] == "fixed"
    assert rows["slice_02.png"]["source"] == "interpolated"
    # Outside them it steps by the interval the fixed points IMPLY (0.5),
    # never by the nominal one (0.2).
    assert rows["slice_00.png"]["position_mm"] == pytest.approx(1.5)
    assert rows["slice_05.png"]["position_mm"] == pytest.approx(4.0)
    assert result["implied_interval_mm"] == pytest.approx(0.5)
    # Numbers only: no advice riding along with the suggestions.
    assert "note" not in result
    assert "nominal_interval_mm" not in result

    # Nothing written.
    assert all(s.position_mm is None for s in state.slices)


def test_interpolate_between_extrapolates_at_the_implied_interval(
    tmp_path: Path,
):
    """Beyond the outermost fixed points, the step is what they imply."""
    state, ctx = _stack(tmp_path, n=6, placed=False)
    interpolate_between = _tool(_box(state, ctx), "interpolate_between")

    result = interpolate_between(
        [
            {"id": "slice_02.png", "position_mm": 4.0},
            {"id": "slice_03.png", "position_mm": 5.0},
        ]
    )

    rows = {row["id"]: row["position_mm"] for row in result["suggestions"]}
    assert result["implied_interval_mm"] == pytest.approx(1.0)
    assert rows["slice_00.png"] == pytest.approx(2.0)  # not 4.0 - 2 * 0.2
    assert rows["slice_05.png"] == pytest.approx(7.0)  # not 5.0 + 2 * 0.2


def test_interpolate_between_refuses_a_single_fixed_point(tmp_path: Path):
    state, ctx = _stack(tmp_path, placed=False)
    interpolate_between = _tool(_box(state, ctx), "interpolate_between")

    result = interpolate_between([{"id": "slice_02.png", "position_mm": 4.0}])

    assert result["error"] == "ONE_FIXED_POINT"
    assert "at least two fixed points" in result["message"]


def test_interpolate_between_follows_a_reversed_stack(tmp_path: Path):
    state, ctx = _stack(tmp_path, placed=False)
    interpolate_between = _tool(_box(state, ctx), "interpolate_between")

    result = interpolate_between(
        [
            {"id": "slice_01.png", "position_mm": 5.0},
            {"id": "slice_04.png", "position_mm": 3.5},
        ]
    )

    positions = [row["position_mm"] for row in result["suggestions"]]
    assert positions == sorted(positions, reverse=True)


def test_interpolate_between_respects_corrected_order(tmp_path: Path):
    """Suggestions follow index_corrected, not the discovery order."""
    state, ctx = _stack(tmp_path, n=4, placed=False)
    for record in state.slices:  # reverse the stack
        record.index_corrected = 3 - record.index_original
    interpolate_between = _tool(_box(state, ctx), "interpolate_between")

    result = interpolate_between(
        [
            {"id": "slice_03.png", "position_mm": 1.0},
            {"id": "slice_00.png", "position_mm": 4.0},
        ]
    )

    rows = {row["id"]: row["position_mm"] for row in result["suggestions"]}
    assert [row["id"] for row in result["suggestions"]] == [
        "slice_03.png", "slice_02.png", "slice_01.png", "slice_00.png",
    ]
    assert rows["slice_02.png"] == pytest.approx(2.0)
    assert rows["slice_01.png"] == pytest.approx(3.0)


def test_interpolate_between_needs_a_fixed_point(tmp_path: Path):
    state, ctx = _stack(tmp_path, placed=False)
    interpolate_between = _tool(_box(state, ctx), "interpolate_between")

    assert interpolate_between([])["error"] == "BAD_ARGS"
    assert interpolate_between([{"id": "ghost.png", "position_mm": 1.0}])["error"] == (
        "NO_FIXED_POINTS"
    )


# --- the toolset itself ---------------------------------------------------


def test_the_toolset_is_the_lean_one(tmp_path: Path):
    """No per-slice worker, no advisory tool: data tools and the gate only."""
    state, ctx = _stack(tmp_path)

    names = {t.__name__ for t in _box(state, ctx).tools}

    assert names == {
        "view_slices",
        "fetch_atlas",
        "atlas_structures_at",
        "structure_range",
        "stack_positions",
        "interpolate_between",
        "set_positions",
        "submit_positions",
    }


def test_no_tool_payload_carries_advice(tmp_path: Path):
    """Every payload the agent reads is numbers, not opinions."""
    state, ctx = _stack(tmp_path)
    box = _box(state, ctx)

    payloads = [
        _tool(box, "stack_positions")(),
        _tool(box, "set_positions")([{"id": "slice_00.png", "position_mm": 2.0}]),
        _tool(box, "interpolate_between")(
            [
                {"id": "slice_01.png", "position_mm": 2.5},
                {"id": "slice_04.png", "position_mm": 4.0},
            ]
        ),
        _tool(box, "atlas_structures_at")([4.0]),
        _tool(box, "structure_range")(["XA"]),
    ]

    blob = " ".join(str(payload) for payload in payloads).lower()
    for phrase in BANNED_COACHING:
        assert phrase not in blob, phrase


# --- submit_positions ----------------------------------------------------


def test_submit_positions_rejects_an_incomplete_stack(tmp_path: Path):
    state, ctx = _stack(tmp_path)
    box = _box(state, ctx)
    anchors = _anchors(state)
    state.in_order()[2].position_mm = None
    tool_context = _ToolContext()

    result = _tool(box, "submit_positions")(
        interval_breaks=[],
        notes=[],
        summary="done",
        end_anchors=anchors,
        tool_context=tool_context,
    )

    assert result["error"] == "MISSING_POSITIONS"
    assert result["missing_ids"] == ["slice_02.png"]
    assert tool_context.actions.escalate is False
    assert not box.submission


def test_submit_positions_escalates_and_captures_findings(tmp_path: Path):
    state, ctx = _stack(tmp_path)
    box = _box(state, ctx)
    ordered = state.in_order()
    for record in ordered[3:]:  # a real double gap before index 3
        record.position_mm = float(record.position_mm) + 0.5  # type: ignore[arg-type]
    tool_context = _ToolContext()

    result = _tool(box, "submit_positions")(
        interval_breaks=[3],
        notes=["sections 3-4 jump two intervals"],
        summary="Verified both key sections; one gap.",
        end_anchors=_anchors(state),
        tool_context=tool_context,
    )

    assert result["status"] == "ok"
    assert tool_context.actions.escalate is True
    assert box.submission["interval_breaks"] == [3]
    assert box.submission["notes"] == ["sections 3-4 jump two intervals"]
    assert [entry["structure"] for entry in box.submission["end_anchors"]] == [
        _ANCHOR_FIRST,
        _ANCHOR_LAST,
    ]


# --- the interval-break gate ----------------------------------------------


def _submit_breaks(box, state, breaks, tool_context=None):
    return _tool(box, "submit_positions")(
        interval_breaks=breaks,
        notes=[],
        summary="done",
        end_anchors=_anchors(state),
        tool_context=tool_context or _ToolContext(),
    )


def test_submit_refuses_a_break_its_own_positions_do_not_show(tmp_path: Path):
    """An evenly placed stack has no breaks, whatever the submission claims."""
    state, ctx = _stack(tmp_path)  # a flat 0.5 mm ladder
    box = _box(state, ctx)
    tool_context = _ToolContext()

    result = _submit_breaks(box, state, [2, 4], tool_context)

    assert result["error"] == "INTERVAL_BREAKS_UNSUPPORTED"
    assert [failure["index"] for failure in result["failures"]] == [2, 4]
    failure = result["failures"][0]
    assert failure["written_interval_mm"] == pytest.approx(0.5)
    assert failure["median_interval_mm"] == pytest.approx(0.5)
    assert failure["between"] == ["slice_01.png", "slice_02.png"]
    # The interval actually written there is in the message the agent reads.
    assert "are 0.500 mm apart" in failure["reason"]
    # Refused, not warned: nothing was submitted.
    assert tool_context.actions.escalate is False
    assert not box.submission


def test_submit_accepts_a_break_the_positions_really_show(tmp_path: Path):
    state, ctx = _stack(tmp_path)
    box = _box(state, ctx)
    for record in state.in_order()[4:]:  # a double gap before index 4
        record.position_mm = float(record.position_mm) + 0.5  # type: ignore[arg-type]

    assert _submit_breaks(box, state, [4])["status"] == "ok"
    assert box.submission["interval_breaks"] == [4]


def test_submit_refuses_a_break_index_with_no_interval(tmp_path: Path):
    state, ctx = _stack(tmp_path)
    box = _box(state, ctx)

    result = _submit_breaks(box, state, [0, 99], None)

    assert [failure["index"] for failure in result["failures"]] == [0, 99]
    assert all(f["error"] == "NOT_A_GAP" for f in result["failures"])
    assert "no section before it" in result["failures"][0]["reason"]


def test_submit_break_gate_is_on_with_the_landmark_tools_off(tmp_path: Path):
    state, ctx = _stack(tmp_path, landmark_tools=False)
    box = _box(state, ctx)

    refused = _tool(box, "submit_positions")(
        interval_breaks=[2], notes=[], summary="done", tool_context=_ToolContext()
    )

    assert refused["error"] == "INTERVAL_BREAKS_UNSUPPORTED"
    assert not box.submission


# --- the end-anchor gate --------------------------------------------------


def _submit(box, state, anchors, tool_context=None):
    return _tool(box, "submit_positions")(
        interval_breaks=[],
        notes=[],
        summary="done",
        end_anchors=anchors,
        tool_context=tool_context or _ToolContext(),
    )


def test_submit_refuses_a_placement_the_atlas_rules_out(tmp_path: Path):
    """XA exists over 2.0-3.0 mm; a section at 12 mm cannot show it."""
    state, ctx = _stack(tmp_path)
    box = _box(state, ctx)
    for record in state.in_order():
        record.position_mm = 12.0 + 0.5 * record.index_corrected
    tool_context = _ToolContext()

    result = _submit(box, state, _anchors(state), tool_context)

    assert result["error"] == "END_ANCHOR_FAILED"
    failure = result["failures"][0]
    assert failure["error"] == "OUTSIDE_STRUCTURE_SPAN"
    assert failure["structure"] == _ANCHOR_FIRST
    assert failure["atlas_span_mm"] == [2.0, 3.0]
    assert failure["position_mm"] == pytest.approx(12.0)
    # The span the model has to reconcile with is IN the message it reads.
    assert "2.000 to 3.000 mm" in failure["reason"]
    assert "12.000 mm" in failure["reason"]
    # Refused, not escalated: the agent has to fix it.
    assert tool_context.actions.escalate is False
    assert not box.submission


def test_submit_refuses_a_structure_too_broad_to_localize(tmp_path: Path):
    """FA spans 30% of the slicing axis — wide enough to "prove" anywhere.

    XA/XP, sitting over the same section range, are the narrow counterparts:
    a good end landmark only needs a SHORT span, not agreement everywhere.
    """
    state, ctx = _stack(tmp_path)
    box = _box(state, ctx)
    tool_context = _ToolContext()

    broad = _submit(box, state, _anchors(state, _BROAD_STRUCTURE), tool_context)

    assert broad["error"] == "END_ANCHOR_FAILED"
    failure = broad["failures"][0]
    assert failure["error"] == "STRUCTURE_TOO_BROAD"
    assert failure["structure"] == _BROAD_STRUCTURE
    assert failure["span_mm"] == [2.0, 6.0]
    assert failure["span_fraction"] == pytest.approx(4.0 / 13.18, abs=0.01)
    assert "%" in failure["reason"]
    assert tool_context.actions.escalate is False
    assert not box.submission

    # 3 mm over a 13.18 mm axis (23%) is still too wide to anchor an end: the
    # cap is tight enough to exclude structures wider than the errors it is
    # there to catch.
    assert _submit(box, state, _anchors(state, "NS"))["error"] == "END_ANCHOR_FAILED"

    # The extra-narrow structures, over the same sections, are accepted.
    accepted = _submit(box, state, _anchors(state), tool_context)
    assert accepted["status"] == "ok"


def test_submit_refuses_an_unresolvable_structure_and_names_near_misses(
    tmp_path: Path,
):
    state, ctx = _stack(tmp_path)
    box = _box(state, ctx)
    anchors = _anchors(state)
    anchors[0]["structure"] = "forebrain area"  # ambiguous: FA and FB

    result = _submit(box, state, anchors)

    assert result["error"] == "END_ANCHOR_FAILED"
    failure = result["failures"][0]
    assert failure["error"] == "UNKNOWN_STRUCTURE"
    assert set(failure["near_misses"]) == {"FA", "FB"}
    assert not box.submission


def test_submit_refuses_when_an_end_anchor_is_missing(tmp_path: Path):
    state, ctx = _stack(tmp_path)
    box = _box(state, ctx)
    ordered = state.in_order()

    result = _submit(box, state, _anchors(state)[:1])

    assert result["error"] == "END_ANCHORS_REQUIRED"
    assert result["required_ids"] == [ordered[0].id, ordered[-1].id]
    assert ordered[-1].id in result["message"]
    assert not box.submission

    # Anchoring the same end twice is not two ends either.
    inner = _anchors(state)
    inner[1]["id"] = ordered[0].id
    assert _submit(box, state, inner)["error"] == "END_ANCHORS_REQUIRED"
    # ...nor is anchoring a section in the middle.
    middle = _anchors(state)
    middle[1]["id"] = ordered[2].id
    assert _submit(box, state, middle)["error"] == "END_ANCHORS_REQUIRED"


def test_submit_accepts_an_anchor_within_a_thickness_of_the_span(tmp_path: Path):
    """The tolerance is one section thickness, not zero."""
    state, ctx = _stack(tmp_path)
    box = _box(state, ctx)
    ordered = state.in_order()
    ordered[0].position_mm = 2.0 - state.thickness_mm / 2  # just outside XA
    ordered[-1].position_mm = 5.0

    assert _submit(box, state, _anchors(state))["status"] == "ok"


def test_submit_survives_an_atlas_that_cannot_answer(tmp_path: Path):
    """A broken atlas must not deadlock the run — but it is on the record."""
    state, ctx = _stack(tmp_path)
    ctx.atlas_loader = lambda _name: object()
    box = _box(state, ctx)
    tool_context = _ToolContext()

    result = _submit(box, state, _anchors(state), tool_context)

    assert result["status"] == "ok"
    assert tool_context.actions.escalate is True
    assert any(
        "end-anchor check skipped" in note for note in box.submission["notes"]
    )


# --- the landmark tools ---------------------------------------------------


def test_atlas_structures_at_reports_what_the_annotation_carries(tmp_path: Path):
    state, ctx = _stack(tmp_path)
    atlas_structures_at = _tool(_box(state, ctx), "atlas_structures_at")

    result = atlas_structures_at([6.0, 15.0])

    assert result["status"] == "ok"
    first, second = result["levels"]
    assert first["position_mm"] == pytest.approx(6.0)
    assert [entry["acronym"] for entry in first["structures"]] == ["FB", "FA"]
    assert [entry["acronym"] for entry in second["structures"]] == ["HB"]


def test_atlas_structures_at_clamps_and_caps(tmp_path: Path):
    state, ctx = _stack(tmp_path)
    atlas_structures_at = _tool(_box(state, ctx), "atlas_structures_at")

    assert atlas_structures_at([])["error"] == "BAD_ARGS"
    assert atlas_structures_at([-5.0])["levels"][0]["position_mm"] == 0.0
    assert len(atlas_structures_at([1.0] * 12)["levels"]) == 8


def test_structure_range_reports_spans_and_near_misses(tmp_path: Path):
    state, ctx = _stack(tmp_path)
    structure_range = _tool(_box(state, ctx), "structure_range")

    result = structure_range(["FA", "FOR", "ghost"])

    by_query = {entry["query"]: entry for entry in result["ranges"]}
    assert (by_query["FA"]["first_mm"], by_query["FA"]["last_mm"]) == (2.0, 6.0)
    # A parent's span rolls its descendants up.
    assert (by_query["FOR"]["first_mm"], by_query["FOR"]["last_mm"]) == (2.0, 9.0)
    assert by_query["ghost"]["error"] == "UNKNOWN_STRUCTURE"
    assert "near_misses" in by_query["ghost"]


def test_structure_range_names_the_structures_the_query_could_have_meant(
    tmp_path: Path,
):
    """A short acronym resolves silently; the alternatives make that visible."""
    state, ctx = _stack(tmp_path)
    structure_range = _tool(_box(state, ctx), "structure_range")

    entry = structure_range(["FOR"])["ranges"][0]

    # "FOR" resolves to the parent, but FA and FB match it too.
    assert entry["acronym"] == "FOR"
    assert entry["name"] == "Forebrain"
    assert set(entry["also_matches"]) == {"FA", "FB"}
    assert len(entry["also_matches"]) <= 3

    # An unambiguous acronym carries no alternatives to second-guess.
    assert "also_matches" not in structure_range(["HB"])["ranges"][0]


def test_structure_range_caps_the_batch(tmp_path: Path):
    state, ctx = _stack(tmp_path)
    structure_range = _tool(_box(state, ctx), "structure_range")

    assert structure_range([])["error"] == "BAD_ARGS"
    result = structure_range(["FA"] * 12)
    assert len(result["ranges"]) == 10
    assert len(result["skipped"]) == 2


# --- landmark_tools=False: the ablation switch ----------------------------


def test_landmark_tools_off_drops_the_tools_from_the_toolbox(tmp_path: Path):
    state, ctx = _stack(tmp_path, landmark_tools=False)
    names = {t.__name__ for t in _box(state, ctx).tools}

    assert "atlas_structures_at" not in names
    assert "structure_range" not in names
    assert {"view_slices", "set_positions", "submit_positions"} <= names


def test_landmark_tools_off_submit_positions_takes_no_end_anchors(tmp_path: Path):
    state, ctx = _stack(tmp_path, landmark_tools=False)
    box = _box(state, ctx)
    submit_positions = _tool(box, "submit_positions")
    tool_context = _ToolContext()

    result = submit_positions(
        interval_breaks=[],
        notes=["checked both ends by eye"],
        summary="done",
        tool_context=tool_context,
    )

    assert result["status"] == "ok"
    assert tool_context.actions.escalate is True
    assert "end_anchors" not in box.submission


# --- prompt + seed message ------------------------------------------------


def test_prompt_states_the_job_the_facts_the_tools_and_the_constraints(
    tmp_path: Path,
):
    state, _ctx = _stack(tmp_path, placed=False)
    state.interval_breaks = [4]

    prompt = build_position_prompt(
        state=state, species="mouse", pos_lo=0.0, pos_hi=13.18
    )

    assert "Your job:" in prompt
    assert "Run facts:" in prompt
    assert "Tools:" in prompt
    assert "Constraints:" in prompt
    # Run facts.
    assert "6 sections, coronal plane" in prompt
    assert "0.00-13.18 mm" in prompt
    assert "0.200 mm center-to-center" in prompt
    assert "0.050 mm" in prompt
    assert "No section carries a position yet" in prompt
    assert "[4]" in prompt  # the survey's interval-break flags
    assert "corrected order shown is fixed" in prompt  # keep_order
    # One factual line per tool.
    for tool in (
        "view_slices",
        "fetch_atlas",
        "atlas_structures_at",
        "structure_range",
        "stack_positions",
        "interpolate_between",
        "set_positions",
        "submit_positions",
    ):
        assert f"`{tool}`" in prompt


def test_prompt_carries_no_coaching(tmp_path: Path):
    """No strategies, no rules of thumb, no failure-mode warnings."""
    state, _ctx = _stack(tmp_path, placed=False)

    for landmark_tools in (True, False):
        prompt = build_position_prompt(
            state=state,
            species="mouse",
            pos_lo=0.0,
            pos_hi=13.18,
            landmark_tools=landmark_tools,
        ).lower()
        for phrase in (*BANNED_COACHING, "rules that apply", "ladder", "key section"):
            assert phrase not in prompt, phrase


def test_prompt_states_the_hard_constraints_only(tmp_path: Path):
    state, _ctx = _stack(tmp_path, placed=False)

    prompt = build_position_prompt(
        state=state, species="mouse", pos_lo=0.0, pos_hi=13.18
    )

    assert "Every section must have a position" in prompt
    assert "one end anchor for the FIRST and one for the LAST section" in prompt
    assert "8% of" in prompt  # MAX_ANCHOR_SPAN_FRACTION
    assert "1.5x the stack's median written spacing" in prompt


def test_prompt_omits_landmarks_when_the_gate_is_off(tmp_path: Path):
    state, _ctx = _stack(tmp_path, placed=False)

    prompt = build_position_prompt(
        state=state,
        species="mouse",
        pos_lo=0.0,
        pos_hi=13.18,
        landmark_tools=False,
    )

    assert "structure_range" not in prompt
    assert "atlas_structures_at" not in prompt
    assert "end anchor" not in prompt.lower()
    # The rest of the lean prompt is unchanged.
    assert "Every section must have a position" in prompt
    assert "1.5x the stack's median written spacing" in prompt


def test_prompt_stays_atlas_agnostic(tmp_path: Path):
    """One prompt for every BrainGlobe atlas: no region names, no landmarks."""
    state, _ctx = _stack(tmp_path, placed=False)
    prompt = build_position_prompt(
        state=state, species="zebra finch", pos_lo=0.0, pos_hi=13.18
    ).lower()

    for region in (
        "hippocamp", "cortex", "cerebellum", "olfactory", "bregma",
        "mid-brain", "midbrain", "thalam", "striat", "ventricle",
    ):
        assert region not in prompt


def test_prompt_notes_positions_that_are_already_on_the_stack(tmp_path: Path):
    state, _ctx = _stack(tmp_path)

    prompt = build_position_prompt(
        state=state, species="mouse", pos_lo=0.0, pos_hi=13.18
    )

    assert "6 of 6 sections already carry a position." in prompt


def test_seed_message_renders_unplaced_sections(tmp_path: Path):
    state, ctx = _stack(tmp_path, placed=False)
    state.notes.append("seed: no automatic seeding available")

    text = "\n".join(
        part.text or "" for part in build_position_seed_message(state, ctx).parts or []
    )

    assert text.count("unplaced") >= len(state.slices)
    assert "NO POSITION" not in text
    assert "seed: no automatic seeding available" in text


# --- the position node ---------------------------------------------------


class _ScriptedPositionLlm(BaseLlm):
    """Write one position, then submit."""

    async def generate_content_async(
        self, llm_request, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        responses = sum(
            1
            for content in (llm_request.contents or [])
            for part in (content.parts or [])
            if getattr(part, "function_response", None) is not None
        )
        if responses == 0:
            call = types.Part.from_function_call(
                name="set_positions",
                args={
                    "entries": [
                        {
                            "id": "slice_03.png",
                            "position_mm": 4.25,
                            "confidence": "medium",
                        }
                    ]
                },
            )
        else:
            call = types.Part.from_function_call(
                name="submit_positions",
                args={
                    "interval_breaks": [3],
                    "notes": ["one section missing before index 3"],
                    "summary": "Moved one section; found a gap.",
                    "end_anchors": [
                        {
                            "id": "slice_00.png",
                            "structure": "XA",
                            "note": "extra-narrow anterior structure fills it",
                        },
                        {
                            "id": "slice_05.png",
                            "structure": "XP",
                            "note": "extra-narrow posterior structure present",
                        },
                    ],
                },
            )
        yield LlmResponse(
            content=types.Content(role="model", parts=[call]),
            partial=False,
            turn_complete=True,
        )


class _PlacingLlm(BaseLlm):
    """Place an unplaced stack with the reduced toolset.

    look -> write two points -> interpolate -> check a structure span -> read
    the stack back -> write the whole ladder -> submit.
    """

    async def generate_content_async(
        self, llm_request, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        responses = sum(
            1
            for content in (llm_request.contents or [])
            for part in (content.parts or [])
            if getattr(part, "function_response", None) is not None
        )
        script = [
            ("view_slices", {"slice_ids": ["slice_01.png", "slice_04.png"]}),
            (
                "set_positions",
                {
                    "entries": [
                        {"id": "slice_01.png", "position_mm": 2.5},
                        {"id": "slice_04.png", "position_mm": 4.0},
                    ]
                },
            ),
            (
                "interpolate_between",
                {
                    "fixed": [
                        {"id": "slice_01.png", "position_mm": 2.5},
                        {"id": "slice_04.png", "position_mm": 4.0},
                    ]
                },
            ),
            ("structure_range", {"acronyms": ["FA"]}),
            (
                "set_positions",
                {
                    "entries": [
                        {"id": f"slice_{i:02d}.png", "position_mm": 2.0 + 0.5 * i}
                        for i in range(6)
                    ]
                },
            ),
            ("stack_positions", {}),
            (
                "submit_positions",
                {
                    "interval_breaks": [],
                    "notes": ["two points fixed, the rest interpolated"],
                    "summary": "Placed the stack from two verified sections.",
                    "end_anchors": [
                        {
                            "id": "slice_00.png",
                            "structure": "XA",
                            "note": "extra-narrow anterior structure, whole section",
                        },
                        {
                            "id": "slice_05.png",
                            "structure": "XP",
                            "note": "extra-narrow posterior structure, smaller",
                        },
                    ],
                },
            ),
        ]
        name, args = script[min(responses, len(script) - 1)]
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_function_call(name=name, args=args)],
            ),
            partial=False,
            turn_complete=True,
        )


class _SilentLlm(BaseLlm):
    """Never calls a tool — exercises the nudge and the turn budget."""

    async def generate_content_async(
        self, llm_request, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream, llm_request
        yield LlmResponse(
            content=types.Content(
                role="model", parts=[types.Part.from_text(text="Looks fine to me.")]
            ),
            partial=False,
            turn_complete=True,
        )


def _install_llm(monkeypatch, factory) -> None:
    from google.adk.models.registry import LLMRegistry

    monkeypatch.setattr(LLMRegistry, "new_llm", staticmethod(factory))


def test_position_node_applies_edits_and_routes_on(tmp_path: Path, monkeypatch):
    _install_llm(monkeypatch, lambda model: _ScriptedPositionLlm(model=model))
    state, ctx = _stack(tmp_path)

    assert asyncio.run(position(state, ctx)) == ""

    record = state.by_id("slice_03.png")
    assert record is not None
    assert record.position_mm == pytest.approx(4.25)
    assert record.position_source == "refined"
    assert record.confidence == "medium"
    assert state.interval_breaks == [3]
    assert any("Moved one section" in note for note in state.notes)
    assert any("one section missing" in note for note in state.notes)
    # Accepted breaks are recorded plainly, without editorial.
    assert any(
        "interval breaks accepted at corrected indices [3]" in note
        for note in state.notes
    )


def test_position_node_places_an_unplaced_stack_with_the_lean_toolset(
    tmp_path: Path, monkeypatch
):
    _install_llm(monkeypatch, lambda model: _PlacingLlm(model=model))
    state, ctx = _stack(tmp_path, placed=False)

    assert asyncio.run(position(state, ctx)) == ""

    positions = [s.position_mm for s in state.in_order()]
    assert positions == pytest.approx([2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    assert all(s.position_source == "refined" for s in state.slices)
    assert any("two points fixed" in note for note in state.notes)
    # The end anchors it submitted are on the record, not just in the gate.
    assert any("end anchor slice_00.png = XA" in note for note in state.notes)


def test_position_node_keeps_partial_work_when_the_budget_runs_out(
    tmp_path: Path, monkeypatch
):
    _install_llm(monkeypatch, lambda model: _SilentLlm(model=model))
    state, ctx = _stack(tmp_path)

    assert asyncio.run(position(state, ctx)) == ""

    assert any("position: incomplete" in note for note in state.notes)
    # Positions already on the stack are untouched, not discarded.
    assert all(s.position_mm is not None for s in state.slices)


def test_position_node_is_a_no_op_on_an_empty_stack(tmp_path: Path):
    ctx = build_context(
        BrainConfig(image_folder=str(tmp_path)),
        emit=lambda _m: None,
        atlas_loader=lambda _name: _ATLAS,
    )
    assert asyncio.run(position(StackState(), ctx)) == ""


def test_submission_refused_when_direction_reversed(tmp_path):
    from langslice.linear.whole_brain.position import _direction_error

    state, _ctx = _stack(tmp_path, placed=False)
    state.axis_directions = {"ap": "anterior_to_posterior"}
    ordered = state.in_order()
    n = len(ordered)
    for i, record in enumerate(ordered):
        record.position_mm = 5.0 - i * (3.0 / max(1, n - 1))  # descending
    err = _direction_error(state)
    assert err is not None and err["error"] == "DIRECTION_REVERSED"
    for i, record in enumerate(ordered):
        record.position_mm = 2.0 + i * (3.0 / max(1, n - 1))  # ascending
    assert _direction_error(state) is None
    state.axis_directions = {}
    for i, record in enumerate(ordered):
        record.position_mm = 5.0 - i * 0.1
    assert _direction_error(state) is None  # unknown direction: no check

"""Positioning tools and the position node.

No live model calls: the agent step is driven by a scripted fake BaseLlm
swapped in through ``LLMRegistry.new_llm``, and the single-slice escalation
worker is monkeypatched.
"""

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from google.adk.models import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from PIL import Image

from langslice.linear import APResult
from langslice.linear.whole_brain.engine import build_context
from langslice.linear.whole_brain.nodes import ingest, position
from langslice.linear.whole_brain.position import (
    build_position_prompt,
    build_position_tools,
    interval_table,
)
from langslice.linear.whole_brain.state import BrainConfig, StackState

_POSITION = "langslice.linear.whole_brain.position.run_slice_estimation"
_RANGE = {"pos_lo": 0.0, "pos_hi": 13.18}


class _FakeVolume:
    shape = (528, 320, 456)


class _FakeAtlas:
    atlas_name = "fake_mouse_25um"
    orientation = "asr"
    reference = _FakeVolume()
    resolution = (25.0, 25.0, 25.0)
    metadata = {"species": "mouse"}


class _Actions:
    escalate = False


class _ToolContext:
    def __init__(self):
        self.actions = _Actions()


def _seeded(folder: Path, n: int = 6, **kwargs) -> tuple[StackState, object]:
    """An ingested stack with seed-style positions already on it."""
    for index in range(n):
        Image.new("RGB", (40, 30), (10 * index, 60, 120)).save(
            folder / f"slice_{index:02d}.png"
        )
    ctx = build_context(
        BrainConfig(image_folder=str(folder), **kwargs),
        emit=lambda _m: None,
        atlas_loader=lambda _name: _FakeAtlas(),
    )
    state = StackState()
    asyncio.run(ingest(state, ctx))
    for index, record in enumerate(state.in_order()):
        record.position_mm = 1.0 + 0.5 * index
        record.position_source = "anchor" if index in (1, 4) else "interpolated"
    return state, ctx


def _tool(box, name: str):
    return next(t for t in box.tools if t.__name__ == name)


def _box(state, ctx):
    return build_position_tools(state, ctx, **_RANGE)


# --- set_positions -------------------------------------------------------


def test_set_positions_writes_a_batch_and_returns_the_interval_table(tmp_path: Path):
    state, ctx = _seeded(tmp_path)
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
    # An untouched section keeps its seeded source.
    second = state.by_id("slice_02.png")
    assert second is not None and second.position_source == "interpolated"

    table = result["interval_table"]
    assert [row["id"] for row in table] == [s.id for s in state.in_order()]
    assert table[0]["delta_to_next_mm"] == pytest.approx(0.3)
    assert table[-1]["delta_to_next_mm"] is None


def test_set_positions_clamps_out_of_range_values(tmp_path: Path):
    state, ctx = _seeded(tmp_path)
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
    state, ctx = _seeded(tmp_path)
    set_positions = _tool(_box(state, ctx), "set_positions")

    assert set_positions([])["error"] == "BAD_ARGS"
    assert set_positions([{"id": "ghost.png", "position_mm": 1.0}])["error"] == (
        "NOTHING_WRITTEN"
    )


# --- estimate_slice ------------------------------------------------------


def test_estimate_slice_reports_without_writing(tmp_path: Path, monkeypatch):
    state, ctx = _seeded(tmp_path)
    estimate_slice = _tool(_box(state, ctx), "estimate_slice")

    async def run(*, image_path: str, atlas_name: str, plane: str, model_name):
        del atlas_name, model_name
        assert plane == "coronal"
        assert Path(image_path).name == "slice_03.png"
        return APResult(position_mm=7.5, reasoning="matched the hippocampus")

    monkeypatch.setattr(_POSITION, run)
    result = asyncio.run(estimate_slice("slice_03.png"))

    assert result["status"] == "ok"
    assert result["position_mm"] == pytest.approx(7.5)
    assert "hippocampus" in result["reasoning"]
    record = state.by_id("slice_03.png")
    assert record is not None
    # Not written: the agent decides, then calls set_positions.
    assert record.position_mm == pytest.approx(2.5)
    assert record.position_source == "interpolated"
    assert result["current_position_mm"] == pytest.approx(2.5)


def test_estimate_slice_surfaces_failures_as_tool_errors(tmp_path: Path, monkeypatch):
    state, ctx = _seeded(tmp_path)
    estimate_slice = _tool(_box(state, ctx), "estimate_slice")

    async def boom(**_kwargs):
        raise RuntimeError("API quota exhausted")

    monkeypatch.setattr(_POSITION, boom)

    assert asyncio.run(estimate_slice("ghost.png"))["error"] == "UNKNOWN_SLICE_ID"
    failed = asyncio.run(estimate_slice("slice_03.png"))
    assert failed["error"] == "ESTIMATE_FAILED"
    assert "quota" in failed["message"]


# --- get_advisories ------------------------------------------------------


def test_get_advisories_returns_spacing_signals(tmp_path: Path):
    state, ctx = _seeded(tmp_path)
    advisories = _tool(_box(state, ctx), "get_advisories")()

    assert advisories["status"] == "ok"
    assert "verify" in advisories["advisory"]
    assert len(advisories["interval_table"]) == len(state.slices)

    residuals = advisories["interpolation_residuals"]
    assert residuals["status"] == "ok"
    assert len(residuals["rows"]) == len(state.slices)
    # The seeded ladder is exactly the anchor interpolation between anchors.
    between = {row["id"]: row for row in residuals["rows"]}["slice_02.png"]
    assert between["residual_mm"] == pytest.approx(0.0)

    fit = advisories["monotone_fit"]
    assert fit["status"] == "ok"
    assert {"id", "position_mm", "suggested_mm", "delta_mm"} == set(fit["rows"][0])


def test_get_advisories_degrades_without_anchors(tmp_path: Path):
    state, ctx = _seeded(tmp_path, n=2)
    for record in state.slices:
        record.position_source = "interpolated"
    advisories = _tool(_box(state, ctx), "get_advisories")()

    assert advisories["interpolation_residuals"]["status"] == "unavailable"
    assert advisories["monotone_fit"]["status"] == "unavailable"


def test_interval_table_tolerates_missing_positions(tmp_path: Path):
    state, ctx = _seeded(tmp_path, n=3)
    del ctx
    state.in_order()[1].position_mm = None

    rows = interval_table(state)

    assert rows[0]["delta_to_next_mm"] is None
    assert rows[1]["position_mm"] is None


# --- submit_positions ----------------------------------------------------


def test_submit_positions_rejects_an_incomplete_stack(tmp_path: Path):
    state, ctx = _seeded(tmp_path)
    box = _box(state, ctx)
    state.in_order()[2].position_mm = None
    tool_context = _ToolContext()

    result = _tool(box, "submit_positions")(
        interval_breaks=[], notes=[], summary="done", tool_context=tool_context
    )

    assert result["error"] == "MISSING_POSITIONS"
    assert result["missing_ids"] == ["slice_02.png"]
    assert tool_context.actions.escalate is False
    assert not box.submission


def test_submit_positions_escalates_and_captures_findings(tmp_path: Path):
    state, ctx = _seeded(tmp_path)
    box = _box(state, ctx)
    tool_context = _ToolContext()

    result = _tool(box, "submit_positions")(
        interval_breaks=[3],
        notes=["sections 3-4 jump two intervals"],
        summary="Verified both key sections; one gap.",
        tool_context=tool_context,
    )

    assert result["status"] == "ok"
    assert tool_context.actions.escalate is True
    assert box.submission["interval_breaks"] == [3]
    assert box.submission["notes"] == ["sections 3-4 jump two intervals"]


# --- prompt --------------------------------------------------------------


def test_prompt_carries_the_key_slice_and_interval_break_strategy(tmp_path: Path):
    state, _ctx = _seeded(tmp_path)
    state.interval_breaks = [4]

    prompt = build_position_prompt(
        state=state, species="mouse", pos_lo=0.0, pos_hi=13.18
    )

    assert "VERIFY THE KEY SECTIONS FIRST" in prompt
    assert "does NOT mean no sections are missing" in prompt
    assert "slice_01.png" in prompt  # the anchors are named
    assert "[4]" in prompt  # the survey's interval-break flags carry over
    assert "0.200 mm" in prompt


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
                },
            )
        yield LlmResponse(
            content=types.Content(role="model", parts=[call]),
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
    state, ctx = _seeded(tmp_path)

    assert asyncio.run(position(state, ctx)) == ""

    record = state.by_id("slice_03.png")
    assert record is not None
    assert record.position_mm == pytest.approx(4.25)
    assert record.position_source == "refined"
    assert record.confidence == "medium"
    assert state.interval_breaks == [3]
    assert any("Moved one section" in note for note in state.notes)
    assert any("one section missing" in note for note in state.notes)


def test_position_node_keeps_partial_work_when_the_budget_runs_out(
    tmp_path: Path, monkeypatch
):
    _install_llm(monkeypatch, lambda model: _SilentLlm(model=model))
    state, ctx = _seeded(tmp_path)

    assert asyncio.run(position(state, ctx)) == ""

    assert any("position: incomplete" in note for note in state.notes)
    # The seeded positions are untouched, not discarded.
    assert all(s.position_mm is not None for s in state.slices)


def test_position_node_is_a_no_op_on_an_empty_stack(tmp_path: Path):
    ctx = build_context(
        BrainConfig(image_folder=str(tmp_path)),
        emit=lambda _m: None,
        atlas_loader=lambda _name: _FakeAtlas(),
    )
    assert asyncio.run(position(StackState(), ctx)) == ""

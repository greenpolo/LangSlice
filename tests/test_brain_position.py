"""Positioning tools and the position node.

No live model calls: the agent step is driven by a scripted fake BaseLlm
swapped in through ``LLMRegistry.new_llm``, and the single-slice estimation
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
    MAX_ESTIMATE_SLICES,
    build_position_prompt,
    build_position_seed_message,
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


def _stack(folder: Path, n: int = 6, *, placed: bool = True, **kwargs):
    """An ingested stack, optionally with a ladder of positions already on it.

    The unplaced form (``placed=False``) is what the positioning step actually
    receives now: the seed node no longer places anything.
    """
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
    if placed:
        for index, record in enumerate(state.in_order()):
            record.position_mm = 1.0 + 0.5 * index
            record.position_source = "refined"
    return state, ctx


def _tool(box, name: str):
    return next(t for t in box.tools if t.__name__ == name)


def _box(state, ctx):
    return build_position_tools(state, ctx, **_RANGE)


# --- set_positions -------------------------------------------------------


def test_set_positions_writes_a_batch_and_returns_the_interval_table(tmp_path: Path):
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
    assert second is not None and second.position_mm == pytest.approx(2.0)

    table = result["interval_table"]
    assert [row["id"] for row in table] == [s.id for s in state.in_order()]
    assert table[0]["delta_to_next_mm"] == pytest.approx(0.3)
    assert table[-1]["delta_to_next_mm"] is None


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


# --- estimate_slices -----------------------------------------------------


def _fake_worker(monkeypatch, *, fail: tuple[str, ...] = (), spacing: float = 2.5):
    """Patch the single-slice worker; record what it was called with."""
    calls: list[dict] = []

    async def run(*, image_path: str, atlas_name: str, plane: str, model_name, **kwargs):
        del atlas_name, model_name
        slice_id = Path(image_path).name
        calls.append({"id": slice_id, "plane": plane, **kwargs})
        if slice_id in fail:
            raise RuntimeError("API quota exhausted")
        index = int(Path(image_path).stem.split("_")[-1])
        return APResult(position_mm=index * spacing, reasoning=f"level of {slice_id}")

    monkeypatch.setattr(_POSITION, run)
    return calls


def test_estimate_slices_reports_a_batch_without_writing(tmp_path: Path, monkeypatch):
    state, ctx = _stack(tmp_path)
    calls = _fake_worker(monkeypatch)
    estimate_slices = _tool(_box(state, ctx), "estimate_slices")

    result = asyncio.run(estimate_slices(["slice_03.png", "slice_01.png", "ghost.png"]))

    assert result["status"] == "ok"
    assert [call["id"] for call in calls] == ["slice_03.png", "slice_01.png"]
    assert all(call["plane"] == "coronal" for call in calls)
    assert result["unknown_ids"] == ["ghost.png"]
    estimates = {entry["id"]: entry for entry in result["estimates"]}
    assert estimates["slice_03.png"]["position_mm"] == pytest.approx(7.5)
    assert "slice_03.png" in estimates["slice_03.png"]["reasoning"]
    assert estimates["slice_03.png"]["current_position_mm"] == pytest.approx(2.5)
    assert "Nothing was written" in result["note"]

    # Not written: the agent judges the numbers, then calls set_positions.
    record = state.by_id("slice_03.png")
    assert record is not None
    assert record.position_mm == pytest.approx(2.5)


def test_estimate_slices_caps_the_batch(tmp_path: Path, monkeypatch):
    state, ctx = _stack(tmp_path, n=12)
    calls = _fake_worker(monkeypatch)
    estimate_slices = _tool(_box(state, ctx), "estimate_slices")

    requested = [f"slice_{i:02d}.png" for i in range(12)]
    result = asyncio.run(estimate_slices(requested))

    assert len(calls) == MAX_ESTIMATE_SLICES == 8
    assert len(result["estimates"]) == 8
    assert result["skipped_ids"] == requested[8:]


def test_estimate_slices_reports_per_section_failures(tmp_path: Path, monkeypatch):
    state, ctx = _stack(tmp_path)
    _fake_worker(monkeypatch, fail=("slice_02.png",))
    estimate_slices = _tool(_box(state, ctx), "estimate_slices")

    assert asyncio.run(estimate_slices([]))["error"] == "BAD_ARGS"
    assert asyncio.run(estimate_slices(["ghost.png"]))["error"] == "UNKNOWN_SLICE_IDS"

    result = asyncio.run(estimate_slices(["slice_02.png", "slice_03.png"]))
    entries = {entry["id"]: entry for entry in result["estimates"]}
    assert result["status"] == "ok"  # one section failed, the other did not
    assert entries["slice_02.png"]["error"] == "ESTIMATE_FAILED"
    assert "quota" in entries["slice_02.png"]["message"]
    assert entries["slice_03.png"]["status"] == "ok"


def test_estimate_slices_passes_the_configured_preprocessing(
    tmp_path: Path, monkeypatch
):
    """The worker preprocesses internally; it must not be done twice."""
    state, ctx = _stack(tmp_path, preprocess="none")
    calls = _fake_worker(monkeypatch)
    asyncio.run(_tool(_box(state, ctx), "estimate_slices")(["slice_01.png"]))
    assert calls[0]["apply_clahe"] is False

    state, ctx = _stack(tmp_path, preprocess="auto")
    calls = _fake_worker(monkeypatch)
    asyncio.run(_tool(_box(state, ctx), "estimate_slices")(["slice_01.png"]))
    assert calls[0]["apply_clahe"] is True


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
    # Outside them it steps by the nominal interval.
    assert rows["slice_00.png"]["position_mm"] == pytest.approx(1.8)
    assert rows["slice_05.png"]["position_mm"] == pytest.approx(3.7)

    # Nothing written.
    assert all(s.position_mm is None for s in state.slices)


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


# --- get_advisories ------------------------------------------------------


def test_get_advisories_returns_spacing_signals(tmp_path: Path):
    state, ctx = _stack(tmp_path)
    advisories = _tool(_box(state, ctx), "get_advisories")()

    assert advisories["status"] == "ok"
    assert "verify" in advisories["advisory"]
    assert len(advisories["interval_table"]) == len(state.slices)

    fit = advisories["monotone_fit"]
    assert fit["status"] == "ok"
    assert {"id", "position_mm", "suggested_mm", "delta_mm"} == set(fit["rows"][0])


def test_get_advisories_degrades_on_an_unplaced_stack(tmp_path: Path):
    state, ctx = _stack(tmp_path, placed=False)
    advisories = _tool(_box(state, ctx), "get_advisories")()

    assert advisories["monotone_fit"]["status"] == "unavailable"
    assert all(row["position_mm"] is None for row in advisories["interval_table"])


def test_interval_table_tolerates_missing_positions(tmp_path: Path):
    state, ctx = _stack(tmp_path, n=3)
    del ctx
    state.in_order()[1].position_mm = None

    rows = interval_table(state)

    assert rows[0]["delta_to_next_mm"] is None
    assert rows[1]["position_mm"] is None


# --- submit_positions ----------------------------------------------------


def test_submit_positions_rejects_an_incomplete_stack(tmp_path: Path):
    state, ctx = _stack(tmp_path)
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
    state, ctx = _stack(tmp_path)
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


# --- prompt + seed message ------------------------------------------------


def test_prompt_offers_a_strategy_menu_instead_of_prescribing_one(tmp_path: Path):
    state, _ctx = _stack(tmp_path, placed=False)
    state.interval_breaks = [4]

    prompt = build_position_prompt(
        state=state, species="mouse", pos_lo=0.0, pos_hi=13.18
    )

    assert "YOU CHOOSE THE STRATEGY" in prompt
    # The menu, not a prescription.
    assert "A. KEY SECTIONS, THEN INTERPOLATE" in prompt
    assert "B. FULL COVERAGE" in prompt
    assert "C. A MIX" in prompt
    # The failure modes that bite whichever strategy is chosen.
    assert "DISAGREEING ESTIMATES ARE A STOP SIGN" in prompt
    assert "CHECK BOTH ENDS BEFORE YOU SUBMIT" in prompt
    assert "PLACE KEY SECTIONS WHERE THE FEATURES ARE UNAMBIGUOUS" in prompt
    assert "does NOT mean no sections are missing" in prompt
    # Stack facts still carry over.
    assert "the stack is unplaced" in prompt
    assert "[4]" in prompt  # the survey's interval-break flags
    assert "0.200 mm" in prompt


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

    assert "6 of 6 sections already carry a position" in prompt


def test_seed_message_renders_unplaced_sections(tmp_path: Path):
    state, _ctx = _stack(tmp_path, placed=False)
    state.notes.append("seed: no automatic seeding available")

    text = "\n".join(
        part.text or "" for part in build_position_seed_message(state).parts or []
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
                },
            )
        yield LlmResponse(
            content=types.Content(role="model", parts=[call]),
            partial=False,
            turn_complete=True,
        )


class _StrategyLlm(BaseLlm):
    """Place an unplaced stack the way the prompt's strategy A describes.

    estimate two key sections -> write them -> interpolate the rest -> write
    the whole ladder -> submit.
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
            ("estimate_slices", {"slice_ids": ["slice_01.png", "slice_04.png"]}),
            (
                "set_positions",
                {
                    "entries": [
                        {"id": "slice_01.png", "position_mm": 0.5},
                        {"id": "slice_04.png", "position_mm": 2.0},
                    ]
                },
            ),
            (
                "interpolate_between",
                {
                    "fixed": [
                        {"id": "slice_01.png", "position_mm": 0.5},
                        {"id": "slice_04.png", "position_mm": 2.0},
                    ]
                },
            ),
            (
                "set_positions",
                {
                    "entries": [
                        {"id": f"slice_{i:02d}.png", "position_mm": 0.5 + 0.5 * (i - 1)}
                        for i in range(6)
                    ]
                },
            ),
            (
                "submit_positions",
                {
                    "interval_breaks": [],
                    "notes": ["strategy A: two key sections, rest interpolated"],
                    "summary": "Placed the stack from two verified key sections.",
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


def test_position_node_places_an_unplaced_stack_with_the_new_tools(
    tmp_path: Path, monkeypatch
):
    _install_llm(monkeypatch, lambda model: _StrategyLlm(model=model))
    state, ctx = _stack(tmp_path, placed=False)
    calls = _fake_worker(monkeypatch, spacing=0.5)

    assert asyncio.run(position(state, ctx)) == ""

    # The agent chose which sections to estimate; nothing upstream did.
    assert [call["id"] for call in calls] == ["slice_01.png", "slice_04.png"]
    positions = [s.position_mm for s in state.in_order()]
    assert positions == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    assert all(s.position_source == "refined" for s in state.slices)
    assert any("strategy A" in note for note in state.notes)


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
        atlas_loader=lambda _name: _FakeAtlas(),
    )
    assert asyncio.run(position(StackState(), ctx)) == ""

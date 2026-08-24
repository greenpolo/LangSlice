"""Survey tools, the survey/fix nodes, and the --stop-after seam.

No live model calls: the agent step is driven by a scripted fake BaseLlm
swapped in through ``LLMRegistry.new_llm`` (same seam as tests/fakes.py).
"""

import asyncio
import io
import json
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from google.adk.models import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from PIL import Image, ImageStat

from langslice.linear.whole_brain.engine import build_context, run_nodes
from langslice.linear.whole_brain.nodes import NODES, fix, ingest, survey
from langslice.linear.whole_brain.state import BrainConfig, StackState
from langslice.linear.whole_brain.survey import (
    build_stack_manifest,
    build_survey_prompt,
    build_survey_tools,
)


class _FakeVolume:
    shape = (528, 320, 456)


class _FakeAtlas:
    atlas_name = "fake_mouse_25um"
    orientation = "asr"
    reference = _FakeVolume()
    resolution = (25.0, 25.0, 25.0)
    metadata = {"species": "mouse"}


def _fake_atlas_loader(name: str) -> object:
    del name
    return _FakeAtlas()


def _half_dark_image() -> Image.Image:
    """Left half black, right half white — mirroring is visible in pixels."""
    img = Image.new("RGB", (64, 48), (0, 0, 0))
    img.paste(Image.new("RGB", (32, 48), (255, 255, 255)), (32, 0))
    return img


def _make_stack(folder: Path, n: int = 4) -> list[str]:
    names = [f"slice_{i}.png" for i in range(1, n + 1)]
    for name in names:
        _half_dark_image().save(folder / name)
    return names


def _ctx(folder: Path, **kwargs):
    return build_context(
        BrainConfig(image_folder=str(folder), **kwargs),
        emit=lambda _msg: None,
        atlas_loader=_fake_atlas_loader,
    )


def _ingested(folder: Path, **kwargs) -> tuple[StackState, object]:
    _make_stack(folder)
    ctx = _ctx(folder, **kwargs)
    state = StackState()
    asyncio.run(ingest(state, ctx))
    return state, ctx


def _tool(box, name: str):
    return next(t for t in box.tools if t.__name__ == name)


def _left_right_means(part: types.Part) -> tuple[float, float]:
    inline = part.inline_data
    assert inline is not None and inline.data is not None
    with Image.open(io.BytesIO(inline.data)) as img:
        gray = img.convert("L")
        width, height = gray.size
        return (
            ImageStat.Stat(gray.crop((0, 0, width // 2, height))).mean[0],
            ImageStat.Stat(gray.crop((width // 2, 0, width, height))).mean[0],
        )


# --- tools ---------------------------------------------------------------


def test_flip_slices_toggles_and_reports_unknown_ids(tmp_path: Path):
    state, ctx = _ingested(tmp_path)
    flip_slices = _tool(build_survey_tools(state, ctx), "flip_slices")

    result = flip_slices(["slice_2.png", "slice_3.png", "nope.png"])
    assert result["status"] == "ok"
    assert result["toggled"] == ["slice_2.png", "slice_3.png"]
    assert result["unknown_ids"] == ["nope.png"]
    assert [s.id for s in state.slices if s.flip] == ["slice_2.png", "slice_3.png"]

    # Toggling again undoes it.
    assert flip_slices(["slice_2.png"])["flipped_now"] == ["slice_3.png"]
    assert flip_slices([])["status"] == "error"


def test_reorder_slices_rewrites_corrected_order(tmp_path: Path):
    state, ctx = _ingested(tmp_path, keep_order=False)
    reorder = _tool(build_survey_tools(state, ctx), "reorder_slices")

    new_order = ["slice_2.png", "slice_1.png", "slice_4.png", "slice_3.png"]
    assert reorder(new_order)["status"] == "ok"
    assert [s.id for s in state.in_order()] == new_order
    assert [s.index_corrected for s in state.in_order()] == [0, 1, 2, 3]
    # index_original is untouched: the discovery order stays recoverable.
    assert {s.id: s.index_original for s in state.slices}["slice_1.png"] == 0


def test_reorder_slices_rejects_non_permutations(tmp_path: Path):
    state, ctx = _ingested(tmp_path, keep_order=False)
    reorder = _tool(build_survey_tools(state, ctx), "reorder_slices")

    result = reorder(["slice_2.png", "slice_1.png"])
    assert result["error"] == "NOT_A_PERMUTATION"
    assert result["missing_ids"] == ["slice_3.png", "slice_4.png"]
    assert [s.index_corrected for s in state.in_order()] == [0, 1, 2, 3]

    dupes = reorder(["slice_1.png"] * 4)
    assert dupes["error"] == "NOT_A_PERMUTATION"
    assert dupes["unknown_ids"] == []


def test_reorder_slices_refuses_when_order_is_host_locked(tmp_path: Path):
    state, ctx = _ingested(tmp_path, keep_order=True)
    box = build_survey_tools(state, ctx)

    result = _tool(box, "reorder_slices")(["slice_2.png", "slice_1.png"])
    assert result["error"] == "ORDER_LOCKED"
    assert "keep_order" in result["message"]
    assert [s.id for s in state.in_order()] == [f"slice_{i}.png" for i in range(1, 5)]

    # Flips and damage are still allowed under keep_order.
    assert _tool(box, "flip_slices")(["slice_1.png"])["status"] == "ok"
    assert box.corrections


def test_mark_damaged_records_notes(tmp_path: Path):
    state, ctx = _ingested(tmp_path)
    mark_damaged = _tool(build_survey_tools(state, ctx), "mark_damaged")

    result = mark_damaged(
        [
            {"id": "slice_2.png", "note": "large tear in cortex"},
            {"id": "ghost.png", "note": "not here"},
        ]
    )
    assert result["marked"] == ["slice_2.png"]
    assert result["unknown_ids"] == ["ghost.png"]
    record = state.by_id("slice_2.png")
    assert record is not None and record.damaged
    assert record.damage_note == "large tear in cortex"
    assert result["damaged_now"] == ["slice_2.png"]


def test_view_slices_caps_at_eight_and_mirrors_flipped(tmp_path: Path):
    _make_stack(tmp_path, n=10)
    ctx = _ctx(tmp_path)
    state = StackState()
    asyncio.run(ingest(state, ctx))
    box = build_survey_tools(state, ctx)
    view_slices = _tool(box, "view_slices")

    many = view_slices([f"slice_{i}.png" for i in range(1, 11)])
    assert len(many["images"]) == 8
    assert len(many["slice_ids"]) == 8

    upright = view_slices(["slice_1.png"])
    left, right = _left_right_means(upright["images"][0])
    assert left < right  # dark half on the left, as saved

    _tool(box, "flip_slices")(["slice_1.png"])
    flipped = view_slices(["slice_1.png"])
    left, right = _left_right_means(flipped["images"][0])
    assert left > right  # rendered through the corrected (flipped) view

    assert view_slices(["ghost.png"])["error"] == "UNKNOWN_SLICE_IDS"


def test_submit_survey_escalates_and_captures_findings(tmp_path: Path):
    state, ctx = _ingested(tmp_path)
    box = build_survey_tools(state, ctx)

    class _Actions:
        escalate = False

    class _ToolContext:
        actions = _Actions()

    tool_context = _ToolContext()
    result = _tool(box, "submit_survey")(
        axis_directions={"ap": "anterior_to_posterior"},
        interval_breaks=[2],
        notes=["section 3 looks thin"],
        clean=False,
        summary="Flipped one section.",
        tool_context=tool_context,
    )

    assert result == {"status": "ok", "clean": False}
    assert tool_context.actions.escalate is True
    assert box.submission["axis_directions"] == {"ap": "anterior_to_posterior"}
    assert box.submission["interval_breaks"] == [2]
    assert box.submission["clean"] is False


def test_prompt_and_manifest_carry_the_stack_facts(tmp_path: Path):
    state, ctx = _ingested(tmp_path, plane="coronal")
    box = build_survey_tools(state, ctx)
    _tool(box, "flip_slices")(["slice_2.png"])
    _tool(box, "mark_damaged")([{"id": "slice_3.png", "note": "fold"}])

    manifest = build_stack_manifest(state)
    assert "slice_2.png  [flipped]" in manifest
    assert "slice_3.png  [damaged: fold]" in manifest

    prompt = build_survey_prompt(
        state=state, species="mouse", pos_lo=0.0, pos_hi=13.18
    )
    assert "coronal" in prompt
    assert "notch" in prompt  # the expert's orientation-marker cue
    assert "ORDER ONLY" in prompt
    assert "host locked" in prompt  # keep_order defaults to True


# --- the survey node -----------------------------------------------------


class _ScriptedSurveyLlm(BaseLlm):
    """Flip one section, then submit; ``clean`` comes from the class attr."""

    clean: bool = False

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
                name="flip_slices", args={"slice_ids": ["slice_2.png"]}
            )
        else:
            call = types.Part.from_function_call(
                name="submit_survey",
                args={
                    "axis_directions": {"ap": "anterior_to_posterior"},
                    "interval_breaks": [2],
                    "notes": ["gap after section 1"],
                    "clean": self.clean,
                    "summary": "Mirrored one section; one interval break.",
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


def test_survey_applies_corrections_and_routes_to_fix(tmp_path: Path, monkeypatch):
    _install_llm(monkeypatch, lambda model: _ScriptedSurveyLlm(model=model))
    state, ctx = _ingested(tmp_path)

    assert asyncio.run(survey(state, ctx)) == "fix"

    record = state.by_id("slice_2.png")
    assert record is not None and record.flip
    assert state.axis_directions == {"ap": "anterior_to_posterior"}
    assert state.interval_breaks == [2]
    assert any("Mirrored one section" in note for note in state.notes)
    assert any("gap after section 1" in note for note in state.notes)
    assert "fix" not in state.completed_nodes


def test_survey_routes_past_fix_when_clean(tmp_path: Path, monkeypatch):
    from tests.fakes import install_fake_adk_model_clean_stack

    install_fake_adk_model_clean_stack(monkeypatch)
    state, ctx = _ingested(tmp_path)

    # "seed" jumps past fix; the engine books the skipped node as complete.
    assert asyncio.run(survey(state, ctx)) == "seed"

    assert not any(s.flip for s in state.slices)
    assert state.axis_directions == {"ap": "anterior_to_posterior"}
    assert state.interval_breaks == []


def test_survey_re_checks_when_the_agent_corrects_but_claims_clean(
    tmp_path: Path, monkeypatch
):
    _install_llm(monkeypatch, lambda model: _ScriptedSurveyLlm(model=model, clean=True))
    state, ctx = _ingested(tmp_path)

    assert asyncio.run(survey(state, ctx)) == "fix"


def test_survey_without_a_submission_proceeds_clean(tmp_path: Path, monkeypatch):
    _install_llm(monkeypatch, lambda model: _SilentLlm(model=model))
    state, ctx = _ingested(tmp_path)

    assert asyncio.run(survey(state, ctx)) == "seed"

    assert any("incomplete" in note for note in state.notes)


# --- the fix node --------------------------------------------------------


def test_fix_rebuilds_the_contact_sheet_and_routes_back_to_survey(tmp_path: Path):
    state, ctx = _ingested(tmp_path)
    sheet = Path(state.contact_sheet)
    before = sheet.read_bytes()
    record = state.by_id("slice_1.png")
    assert record is not None
    record.flip = True

    assert asyncio.run(fix(state, ctx)) == "survey"

    assert Path(state.contact_sheet) == sheet
    assert sheet.read_bytes() != before  # re-rendered through the flip


def test_fix_is_a_no_op_on_an_empty_stack(tmp_path: Path):
    assert asyncio.run(fix(StackState(), _ctx(tmp_path))) == ""


# --- the per-step seam ---------------------------------------------------


def test_stop_after_halts_and_checkpoints(tmp_path: Path):
    log: list[str] = []

    def _node(name: str):
        async def run(state: StackState, _ctx) -> str:
            log.append(name)
            return ""

        return run

    nodes = [(name, _node(name)) for name, _ in NODES]
    ctx = _ctx(tmp_path)
    state = asyncio.run(run_nodes(nodes, StackState(), ctx, stop_after="survey"))

    assert log == ["ingest", "survey"]
    assert state.completed_nodes == ["ingest", "survey"]
    saved = json.loads(Path(ctx.checkpoint_path).read_text())
    assert saved["completed_nodes"] == ["ingest", "survey"]


def test_stop_after_rejects_an_unknown_node(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown node"):
        asyncio.run(
            run_nodes(list(NODES), StackState(), _ctx(tmp_path), stop_after="nope")
        )

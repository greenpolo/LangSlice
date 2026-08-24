"""The review step: its tools, its manifest, and the node's verdict routing.

No live model calls: the pass is driven by a scripted fake BaseLlm swapped in
through ``LLMRegistry.new_llm``.
"""

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import ClassVar

from google.adk.models import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from PIL import Image

from langslice.linear.whole_brain.engine import EngineContext, build_context
from langslice.linear.whole_brain.nodes import ingest, review
from langslice.linear.whole_brain.position import build_position_seed_message
from langslice.linear.whole_brain.review import (
    build_review_prompt,
    build_review_seed_message,
    build_review_tools,
    review_manifest,
)
from langslice.linear.whole_brain.state import BrainConfig, StackState


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


def _reviewed(folder: Path, n: int = 4) -> tuple[StackState, EngineContext]:
    """A stack that has been through positioning and transforms."""
    for index in range(n):
        Image.new("RGB", (40, 30), (10 * index, 60, 120)).save(
            folder / f"slice_{index:02d}.png"
        )
    ctx = build_context(
        BrainConfig(image_folder=str(folder)),
        emit=lambda _m: None,
        atlas_loader=lambda _name: _FakeAtlas(),
    )
    state = StackState()
    asyncio.run(ingest(state, ctx))
    for index, record in enumerate(state.in_order()):
        record.position_mm = 1.0 + 0.2 * index
        record.position_source = "refined"
        record.confidence = "medium"
        if index == 2:
            record.damaged = True
            record.damage_note = "fold"
            record.interactive_transform = {"rotation_deg": 3.0}
        else:
            record.affine = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    return state, ctx


def _tool(box, name: str):
    return next(t for t in box.tools if t.__name__ == name)


# --- manifest + seed -----------------------------------------------------


def test_review_manifest_shows_positions_flags_and_transforms(tmp_path: Path):
    state, _ctx = _reviewed(tmp_path)
    state.in_order()[3].affine = None  # a section that got no transform at all

    lines = review_manifest(state).splitlines()

    assert "1.000 mm" in lines[0]
    assert "refined, medium confidence, affine" in lines[0]
    assert "damaged: fold" in lines[2]
    assert "interactive transform" in lines[2]
    assert "NO TRANSFORM" in lines[3]


def test_review_seed_carries_the_manifest_positions_and_notes(tmp_path: Path):
    state, ctx = _reviewed(tmp_path)
    state.notes.append("position: one section was moved 0.4 mm")

    text = "".join(
        part.text or "" for part in build_review_seed_message(state, ctx).parts or []
    )

    assert "slice_00.png" in text
    assert "Positions and neighbour spacing" in text
    assert "'spacing_to_next_mm': 0.2" in text
    assert "one section was moved 0.4 mm" in text


def test_review_seed_is_data_only(tmp_path: Path):
    """The seed hands over numbers; no advisories, no fitted suggestions."""
    state, ctx = _reviewed(tmp_path)

    text = "".join(
        part.text or "" for part in build_review_seed_message(state, ctx).parts or []
    ).lower()

    for phrase in ("advisory", "monotone", "suggested_mm", "arithmetic only"):
        assert phrase not in text, phrase


def test_review_prompt_states_the_verdict_contract(tmp_path: Path):
    state, _ctx = _reviewed(tmp_path)

    prompt = build_review_prompt(state=state, species="mouse")

    assert "Your job:" in prompt
    assert "Run facts:" in prompt
    assert "Tools:" in prompt
    assert "approved=False" in prompt
    assert "0.200 mm" in prompt


def test_review_prompt_carries_no_coaching(tmp_path: Path):
    state, _ctx = _reviewed(tmp_path)

    prompt = build_review_prompt(state=state, species="mouse").lower()

    for phrase in (
        "serial order",
        "stretched",
        "advisory",
        "1.3x",
        "costs the user",
        "do not expect",
    ):
        assert phrase not in prompt, phrase


# --- flag_slice ----------------------------------------------------------


def test_flag_slice_appends_caveats_and_updates_confidence(tmp_path: Path):
    state, ctx = _reviewed(tmp_path)
    flag_slice = _tool(build_review_tools(state, ctx), "flag_slice")

    result = flag_slice(
        [
            {"id": "slice_02.png", "caveat": "damaged; position from neighbours",
             "confidence": "low"},
            {"id": "slice_02.png", "caveat": "transform covers intact tissue only"},
            {"id": "ghost.png", "caveat": "nope"},
        ]
    )

    assert result["status"] == "ok"
    assert result["unknown_ids"] == ["ghost.png"]
    record = state.by_id("slice_02.png")
    assert record is not None
    assert record.caveats == [
        "damaged; position from neighbours",
        "transform covers intact tissue only",
    ]
    assert record.confidence == "low"


def test_flag_slice_rejects_an_empty_or_useless_batch(tmp_path: Path):
    state, ctx = _reviewed(tmp_path)
    flag_slice = _tool(build_review_tools(state, ctx), "flag_slice")

    assert flag_slice([])["error"] == "BAD_ARGS"
    assert flag_slice([{"id": "ghost.png", "caveat": "x"}])["error"] == "NOTHING_FLAGGED"


def test_submit_review_escalates_and_captures_the_verdict(tmp_path: Path):
    state, ctx = _reviewed(tmp_path)
    box = build_review_tools(state, ctx)
    tool_context = _ToolContext()

    result = _tool(box, "submit_review")(
        approved=False,
        notes=["slice_03 sits before slice_02"],
        summary="Order breaks at the end of the stack.",
        tool_context=tool_context,
    )

    assert result["status"] == "ok"
    assert tool_context.actions.escalate is True
    assert box.submission["approved"] is False
    assert box.submission["notes"] == ["slice_03 sits before slice_02"]


# --- the review node -----------------------------------------------------


class _ReviewLlm(BaseLlm):
    """Flag one section, then submit a fixed verdict."""

    # ClassVar, not a field: BaseLlm is a pydantic model.
    approved: ClassVar[bool] = True

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
                name="flag_slice",
                args={
                    "entries": [
                        {"id": "slice_02.png", "caveat": "damaged", "confidence": "low"}
                    ]
                },
            )
        else:
            call = types.Part.from_function_call(
                name="submit_review",
                args={
                    "approved": self.approved,
                    "notes": ["spacing after slice_01 is too wide"],
                    "summary": "One suspicious gap.",
                },
            )
        yield LlmResponse(
            content=types.Content(role="model", parts=[call]),
            partial=False,
            turn_complete=True,
        )


class _RejectingLlm(_ReviewLlm):
    approved: ClassVar[bool] = False


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


def test_review_node_approves_and_falls_through_to_emit(tmp_path: Path, monkeypatch):
    _install_llm(monkeypatch, lambda model: _ReviewLlm(model=model))
    state, ctx = _reviewed(tmp_path)

    assert asyncio.run(review(state, ctx)) == ""

    record = state.by_id("slice_02.png")
    assert record is not None
    assert record.caveats == ["damaged"] and record.confidence == "low"
    assert any("review: One suspicious gap." in note for note in state.notes)


def test_review_node_routes_back_to_position_and_leaves_its_notes_behind(
    tmp_path: Path, monkeypatch
):
    _install_llm(monkeypatch, lambda model: _RejectingLlm(model=model))
    state, ctx = _reviewed(tmp_path)

    assert asyncio.run(review(state, ctx)) == "position"

    assert any(
        "review: spacing after slice_01 is too wide" in note for note in state.notes
    )
    # And the positioning step's next pass actually reads them.
    seed = "".join(
        part.text or "" for part in build_position_seed_message(state, ctx).parts or []
    )
    assert "review: spacing after slice_01 is too wide" in seed


def test_review_node_approves_when_the_budget_runs_out(tmp_path: Path, monkeypatch):
    _install_llm(monkeypatch, lambda model: _SilentLlm(model=model))
    state, ctx = _reviewed(tmp_path)

    assert asyncio.run(review(state, ctx)) == ""

    assert any("review: incomplete" in note for note in state.notes)


def test_review_node_is_a_no_op_on_an_empty_stack(tmp_path: Path):
    ctx = build_context(
        BrainConfig(image_folder=str(tmp_path)),
        emit=lambda _m: None,
        atlas_loader=lambda _name: _FakeAtlas(),
    )
    assert asyncio.run(review(StackState(), ctx)) == ""

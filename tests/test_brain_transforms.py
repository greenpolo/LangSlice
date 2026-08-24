"""The transform step: the affine pass, the interactive loop, and the node.

No live model calls: the interactive loop is driven by a scripted fake BaseLlm
swapped in through ``LLMRegistry.new_llm``, and the affine registrar is
monkeypatched wherever the test is about the node rather than the fit.
"""

import asyncio
import io
from collections.abc import AsyncGenerator, Callable
from pathlib import Path

import numpy as np
import pytest
from google.adk.models import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from PIL import Image

from langslice.adk import TOOL_MEDIA_PARTS_KEY
from langslice.affine import SilhouetteFit
from langslice.linear.whole_brain.engine import build_context
from langslice.linear.whole_brain.nodes import ingest, transforms
from langslice.linear.whole_brain.state import BrainConfig, StackState
from langslice.linear.whole_brain.transforms import (
    build_transform_tools,
    run_affine_pass,
    run_transform_session,
)
from tests.fakes import EllipseAtlas, ellipse_section

_SILHOUETTE = "langslice.linear.whole_brain.transforms.silhouette_affine"


class _Actions:
    escalate = False


class _ToolContext:
    def __init__(self):
        self.actions = _Actions()


def _marker_section(_index: int) -> Image.Image:
    """A bright square on a black field: geometry is readable off the pixels."""
    canvas = np.zeros((160, 200, 3), dtype=np.uint8)
    canvas[70:90, 40:60] = 255
    return Image.fromarray(canvas, mode="RGB")


def _stack(
    folder: Path,
    n: int = 4,
    damaged: tuple[int, ...] = (),
    images: Callable[[int], Image.Image] = lambda index: ellipse_section(
        angle=5.0 * index
    ),
) -> tuple[StackState, object]:
    """An ingested, positioned stack of sections over the ellipse atlas."""
    for index in range(n):
        images(index).save(folder / f"slice_{index:02d}.png")
    ctx = build_context(
        BrainConfig(image_folder=str(folder), atlas="fake_ellipse_1mm"),
        emit=lambda _m: None,
        atlas_loader=lambda _name: EllipseAtlas(),
    )
    state = StackState()
    asyncio.run(ingest(state, ctx))
    for index, record in enumerate(state.in_order()):
        record.position_mm = 2.0 + index
        record.position_source = "refined"
        if index in damaged:
            record.damaged = True
            record.damage_note = "torn dorsal cortex"
    return state, ctx


def _tool(box, name: str):
    return next(t for t in box.tools if t.__name__ == name)


def _fake_fit(iou: float = 0.9) -> SilhouetteFit:
    return SilhouetteFit(
        matrix=np.array([[1.0, 0.0, 10.0], [0.0, 1.0, -5.0]]),
        iou=iou,
        size=(200, 100),
        slice_rgb=np.zeros((100, 200, 3), dtype=np.uint8),
        atlas_mask=np.zeros((100, 200), dtype=np.uint8),
        sign_pattern=(1, 1),
    )


# --- the affine pass -----------------------------------------------------


def test_affine_pass_writes_normalized_parameters_for_intact_sections(
    tmp_path: Path, monkeypatch
):
    state, ctx = _stack(tmp_path, damaged=(2,))
    seen: list[float] = []

    def fake(image, *, atlas, position_mm, plane, **kwargs):
        del image, atlas, plane, kwargs
        seen.append(position_mm)
        return _fake_fit()

    monkeypatch.setattr(_SILHOUETTE, fake)
    fitted, failed = run_affine_pass(state, ctx)

    assert (fitted, failed) == (3, 0)
    # The damaged section is left for the interactive route.
    assert seen == [2.0, 3.0, 5.0]
    assert state.in_order()[2].affine is None
    # [a, b, tx, c, d, ty], x as a fraction of width, y of height.
    assert state.in_order()[0].affine == pytest.approx(
        [1.0, 0.0, 10.0 / 200, 0.0, 1.0, -5.0 / 100]
    )
    assert all(not record.caveats for record in state.slices)


def test_affine_pass_caveats_a_weak_fit(tmp_path: Path, monkeypatch):
    state, ctx = _stack(tmp_path, n=1)
    monkeypatch.setattr(_SILHOUETTE, lambda *a, **k: _fake_fit(iou=0.2))

    fitted, failed = run_affine_pass(state, ctx)

    assert (fitted, failed) == (1, 0)
    assert state.slices[0].affine is not None
    assert "weak affine fit" in state.slices[0].caveats[0]


def test_affine_pass_survives_a_failing_section(tmp_path: Path, monkeypatch):
    state, ctx = _stack(tmp_path, n=2)

    def boom(image, *, atlas, position_mm, plane, **kwargs):
        del image, atlas, plane, kwargs
        if position_mm == 2.0:
            raise ValueError("Otsu likely failed")
        return _fake_fit()

    monkeypatch.setattr(_SILHOUETTE, boom)
    fitted, failed = run_affine_pass(state, ctx)

    assert (fitted, failed) == (1, 1)
    first, second = state.in_order()
    assert first.affine is None and first.caveats == ["affine failed"]
    assert second.affine is not None
    assert any("affine failed for slice_00.png" in note for note in state.notes)


def test_affine_pass_runs_for_real_against_a_synthetic_atlas(tmp_path: Path):
    """No monkeypatching: the real silhouette fit, end to end."""
    state, ctx = _stack(tmp_path, n=1)

    fitted, failed = run_affine_pass(state, ctx)

    assert (fitted, failed) == (1, 0)
    affine = state.slices[0].affine
    assert affine is not None and len(affine) == 6
    # An ellipse onto an ellipse: near-identity rotation, modest rescale.
    assert affine[0] == pytest.approx(1.0, abs=0.35)
    assert state.slices[0].caveats == []


# --- preview_transform ---------------------------------------------------


def _panel_cells(result) -> list[np.ndarray]:
    """The three panels of a preview, as grayscale arrays."""
    part = result[TOOL_MEDIA_PARTS_KEY][0]
    image = Image.open(io.BytesIO(part.inline_data.data)).convert("L")
    array = np.asarray(image)
    width = array.shape[1] // 3
    # The bottom strip is the label row; drop it.
    return [array[:-14, i * width : (i + 1) * width] for i in range(3)]


def _centroid_x(cell: np.ndarray) -> float:
    ys, xs = np.nonzero(cell > 40)
    del ys
    return float(xs.mean())


def test_preview_transform_returns_a_three_panel_image(tmp_path: Path):
    state, ctx = _stack(tmp_path, n=1)
    preview = _tool(build_transform_tools(state.slices[0], state, ctx), "preview_transform")

    result = preview(
        rotation_deg=0.0, scale_x=1.0, scale_y=1.0, translate_x=0.0, translate_y=0.0
    )

    assert result["status"] == "ok"
    assert result["id"] == "slice_00.png"
    assert result["params"]["scale_x"] == 1.0
    assert "magenta" in result["description"]
    parts = result[TOOL_MEDIA_PARTS_KEY]
    assert len(parts) == 1 and isinstance(parts[0], types.Part)
    cells = _panel_cells(result)
    assert len(cells) == 3 and all(cell.size > 0 for cell in cells)


def test_preview_transform_moves_the_section_as_asked(tmp_path: Path):
    state, ctx = _stack(tmp_path, n=1, images=_marker_section)
    preview = _tool(build_transform_tools(state.slices[0], state, ctx), "preview_transform")

    identity = _panel_cells(
        preview(
            rotation_deg=0.0, scale_x=1.0, scale_y=1.0, translate_x=0.0, translate_y=0.0
        )
    )[0]
    shifted = _panel_cells(
        preview(
            rotation_deg=0.0, scale_x=1.0, scale_y=1.0, translate_x=0.2, translate_y=0.0
        )
    )[0]

    width = identity.shape[1]
    assert _centroid_x(shifted) - _centroid_x(identity) == pytest.approx(
        0.2 * width, rel=0.15
    )


def test_preview_transform_tracks_the_last_parameters_it_rendered(tmp_path: Path):
    state, ctx = _stack(tmp_path, n=1)
    box = build_transform_tools(state.slices[0], state, ctx)

    _tool(box, "preview_transform")(
        rotation_deg=4.0, scale_x=1.0, scale_y=1.0, translate_x=0.0, translate_y=0.0
    )

    assert box.previews == 1
    assert box.last_preview == {
        "rotation_deg": 4.0,
        "scale_x": 1.0,
        "scale_y": 1.0,
        "translate_x": 0.0,
        "translate_y": 0.0,
    }


def test_submit_transform_escalates_and_captures_the_parameters(tmp_path: Path):
    state, ctx = _stack(tmp_path, n=1)
    box = build_transform_tools(state.slices[0], state, ctx)
    tool_context = _ToolContext()

    result = _tool(box, "submit_transform")(
        rotation_deg=-3.0,
        scale_x=1.05,
        scale_y=0.98,
        translate_x=0.01,
        translate_y=-0.02,
        confidence="medium",
        note="dorsal tear could not be matched",
        tool_context=tool_context,
    )

    assert result["status"] == "ok"
    assert tool_context.actions.escalate is True
    assert box.submission["params"]["rotation_deg"] == -3.0
    assert box.submission["confidence"] == "medium"


# --- the interactive session ---------------------------------------------


class _ScriptedTransformLlm(BaseLlm):
    """Preview once, then submit."""

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
        args = {
            "rotation_deg": 6.0,
            "scale_x": 1.02,
            "scale_y": 0.97,
            "translate_x": 0.03,
            "translate_y": -0.01,
        }
        if responses == 0:
            call = types.Part.from_function_call(name="preview_transform", args=args)
        else:
            call = types.Part.from_function_call(
                name="submit_transform",
                args={**args, "confidence": "low", "note": "half the cortex is missing"},
            )
        yield LlmResponse(
            content=types.Content(role="model", parts=[call]),
            partial=False,
            turn_complete=True,
        )


class _PreviewForeverLlm(BaseLlm):
    """Never submits — exercises the turn budget."""

    async def generate_content_async(
        self, llm_request, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream, llm_request
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_function_call(
                        name="preview_transform",
                        args={
                            "rotation_deg": 2.0,
                            "scale_x": 1.0,
                            "scale_y": 1.0,
                            "translate_x": 0.0,
                            "translate_y": 0.0,
                        },
                    )
                ],
            ),
            partial=False,
            turn_complete=True,
        )


def _install_llm(monkeypatch, factory) -> None:
    from google.adk.models.registry import LLMRegistry

    monkeypatch.setattr(LLMRegistry, "new_llm", staticmethod(factory))


def test_interactive_session_records_the_submitted_transform(tmp_path: Path, monkeypatch):
    _install_llm(monkeypatch, lambda model: _ScriptedTransformLlm(model=model))
    state, ctx = _stack(tmp_path, n=1, damaged=(0,))
    record = state.slices[0]

    outcome = asyncio.run(
        run_transform_session(
            record=record, state=state, ctx=ctx, pos_lo=0.0, pos_hi=19.0
        )
    )

    assert outcome.submitted is True
    assert outcome.previews == 1
    assert record.interactive_transform == {
        "rotation_deg": 6.0,
        "scale_x": 1.02,
        "scale_y": 0.97,
        "translate_x": 0.03,
        "translate_y": -0.01,
    }
    assert record.confidence == "low"
    assert record.caveats == ["interactive transform: half the cortex is missing"]
    assert record.affine is None


def test_interactive_session_keeps_the_last_preview_when_the_budget_runs_out(
    tmp_path: Path, monkeypatch
):
    _install_llm(monkeypatch, lambda model: _PreviewForeverLlm(model=model))
    state, ctx = _stack(tmp_path, n=1, damaged=(0,))
    record = state.slices[0]

    outcome = asyncio.run(
        run_transform_session(
            record=record,
            state=state,
            ctx=ctx,
            pos_lo=0.0,
            pos_hi=19.0,
            max_iterations=3,
        )
    )

    assert outcome.submitted is False
    assert outcome.previews >= 1
    assert record.interactive_transform == {
        "rotation_deg": 2.0,
        "scale_x": 1.0,
        "scale_y": 1.0,
        "translate_x": 0.0,
        "translate_y": 0.0,
    }
    assert record.caveats == ["interactive transform incomplete"]


# --- the transforms node -------------------------------------------------


def test_transforms_node_takes_both_routes_and_falls_through(tmp_path: Path, monkeypatch):
    _install_llm(monkeypatch, lambda model: _ScriptedTransformLlm(model=model))
    monkeypatch.setattr(_SILHOUETTE, lambda *a, **k: _fake_fit())
    state, ctx = _stack(tmp_path, n=4, damaged=(1,))

    assert asyncio.run(transforms(state, ctx)) == ""

    ordered = state.in_order()
    assert [record.affine is not None for record in ordered] == [
        True, False, True, True
    ]
    assert [record.interactive_transform is not None for record in ordered] == [
        False, True, False, False
    ]
    assert any(
        "3 affine fit(s), 0 affine failure(s), 1 interactive" in note
        for note in state.notes
    )


def test_transforms_node_is_a_no_op_on_an_empty_stack(tmp_path: Path):
    ctx = build_context(
        BrainConfig(image_folder=str(tmp_path)),
        emit=lambda _m: None,
        atlas_loader=lambda _name: EllipseAtlas(),
    )
    assert asyncio.run(transforms(StackState(), ctx)) == ""

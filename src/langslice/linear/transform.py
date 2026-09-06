"""In-plane transforms: the closed-form fit and the interactive sub-session.

Two routes to the same field. :func:`fit_silhouette` is plain code — the shared
moments fit (:func:`langslice.affine.silhouette_affine`) of a section's
silhouette onto its atlas section. :func:`run_align_session` is a bounded agent
loop for ONE section (preview → look → adjust → submit), for sections whose
silhouette is exactly what damage destroyed.

Both draw ONE picture, :func:`langslice.linear.render.physical_overlay`: the
section under its transform with the atlas family outlines on top at true
physical scale. Parameters are ABBA's — rotation about the canvas centre,
per-axis scales, translations in MILLIMETRES — which only mean anything once
the canvas is calibrated, so every payload carries the calibration and where it
came from.

Everything here is a PROPOSAL: six normalized numbers recorded on the state,
never applied to the user's images (see :mod:`langslice.affine` for the
convention).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from google.genai import types
from PIL import Image

from langslice.adk import TOOL_MEDIA_PARTS_KEY
from langslice.affine import (
    decompose_affine,
    normalized_affine,
    normalized_physical_affine,
    silhouette_affine,
)
from langslice.atlas.render import atlas_um_per_px
from langslice.linear.render import (
    OVERLAY_LONG_EDGE,
    PREVIEW_LONG_EDGE,
    canvas_geometry,
    canvas_um_per_px,
    estimate_um_per_px,
    image_to_part,
    physical_overlay,
    render_slice,
)
from langslice.linear.session import build_agent, run_agent_session
from langslice.linear.state import SliceState, StackState
from langslice.space import Plane

if TYPE_CHECKING:  # ponytail: import cycle — engine builds the toolbox
    from langslice.linear.engine import EngineContext

logger = logging.getLogger(__name__)

#: Tool calls one interactive alignment session may spend.
ALIGN_MAX_ITERATIONS = 12

#: Below this silhouette overlap the closed-form fit is not to be trusted.
WEAK_FIT_IOU = 0.65

_NUDGE_NO_TOOL = (
    "You did not call a tool. Render a candidate alignment with "
    "`preview_transform`, then finish with `submit_transform`."
)
_NUDGE_CONTINUE = (
    "Continue. Adjust the parameters and call `preview_transform` again, or "
    "call `submit_transform` if the current alignment is the best you can get."
)

# --- calibration ---------------------------------------------------------


def calibrate(
    state: StackState, ctx: EngineContext, record: SliceState, section: Image.Image
) -> tuple[float, str]:
    """``(canvas micrometres per pixel, source)``, never failing.

    The file's tags or the host answer first (``"file"`` / ``"host"``). With
    neither, the section's tissue width against the atlas anatomy's gives a
    scale, reported as ``"estimated"`` — a shape fit, not a calibration, and
    the payloads say so. If even that is degenerate the atlas's own voxel size
    stands in, still ``"estimated"``: an alignment loop with an honest guess
    beats one that crashes.
    """
    known, source = canvas_um_per_px(ctx, record, long_edge=PREVIEW_LONG_EDGE)
    if known is not None:
        return known, source
    estimated = estimate_um_per_px(
        section,
        ctx.atlas,
        record.position_mm or 0.0,
        cast(Plane, state.plane),
        state.pitch_deg,
        state.yaw_deg,
    )
    return (estimated or atlas_um_per_px(ctx.atlas)), "estimated"


# --- the closed-form fit -------------------------------------------------


def _fit_matrix_in_section_frame(
    fit_matrix: np.ndarray,
    fit_size: tuple[int, int],
    section: Image.Image,
    geometry: Any,
) -> np.ndarray:
    """The silhouette fit's 2x3, re-expressed on the physical overlay's canvas.

    The fit works in its own frame: the section resized to
    :data:`~langslice.affine.AFFINE_LONG_EDGE` and the atlas silhouette
    STRETCHED to that same frame. To draw it truthfully the whole chain has to
    be composed — section render -> fit frame -> atlas pixels -> canvas —
    otherwise the picture shows the fit against an atlas that is the wrong
    size, which is the very error physical calibration exists to remove.
    """
    fit_w, fit_h = fit_size
    rows, cols = geometry.annotation.shape[:2]
    to_section = fit_w / float(section.width)  # uniform: the fit keeps aspect
    kx = cols * geometry.atlas_scale / fit_w
    ky = rows * geometry.atlas_scale / fit_h
    ox = geometry.atlas_offset[0] - geometry.section_offset[0]
    oy = geometry.atlas_offset[1] - geometry.section_offset[1]
    chain = (
        np.array([[kx, 0.0, ox], [0.0, ky, oy], [0.0, 0.0, 1.0]])
        @ np.vstack([np.asarray(fit_matrix, dtype=np.float64), [0.0, 0.0, 1.0]])
        @ np.diag([to_section, to_section, 1.0])
    )
    return chain[:2]


def fit_silhouette(
    state: StackState, ctx: EngineContext, record: SliceState
) -> dict[str, Any]:
    """Fit the silhouette affine for one positioned, undamaged section.

    Returns the tool-shaped payload: on success ``params`` (six normalized
    numbers), ``iou``, the ``decomposition`` of those numbers, the
    ``calibration`` the panel was drawn with, and a ``panel`` image labelled
    with the section id. ``flat_atlas_fit`` is reported when the stack carries
    cutting angles: the moments fit measures against the flat atlas section,
    because it builds its own atlas silhouette from the voxel grid.
    """
    if record.position_mm is None:
        return {"status": "error", "error": "NO_POSITION", "id": record.id}
    section = render_slice(ctx, record, long_edge=PREVIEW_LONG_EDGE)
    try:
        fit = silhouette_affine(
            section,
            atlas=ctx.atlas,
            position_mm=record.position_mm,
            plane=cast(Plane, state.plane),
        )
    except Exception as exc:
        logger.warning("fit_affine: silhouette fit failed for %s: %s", record.id, exc)
        return {
            "status": "error",
            "error": "FIT_FAILED",
            "id": record.id,
            "message": str(exc),
        }

    um_per_px, source = calibrate(state, ctx, record, section)
    geometry = canvas_geometry(
        section.size,
        um_per_px,
        ctx.atlas,
        record.position_mm,
        cast(Plane, state.plane),
        state.pitch_deg,
        state.yaw_deg,
    )
    panel = physical_overlay(
        section,
        um_per_px,
        ctx.atlas,
        record.position_mm,
        cast(Plane, state.plane),
        state.pitch_deg,
        state.yaw_deg,
        _fit_matrix_in_section_frame(fit.matrix, fit.size, section, geometry),
        label=record.id,
    )
    params = normalized_affine(fit.matrix, fit.size)
    payload: dict[str, Any] = {
        "status": "ok",
        "id": record.id,
        "position_mm": round(record.position_mm, 3),
        "iou": round(float(fit.iou), 3),
        "params": params,
        "decomposition": decompose_affine(params, fit.size),
        "calibration": {
            "section_um_per_px": round(um_per_px, 4),
            "source": source,
        },
        "panel": panel,
    }
    if state.is_oblique:
        payload["flat_atlas_fit"] = True
    return payload


# --- the interactive sub-session -----------------------------------------


@dataclass
class _AlignBox:
    """Tool callables plus the results the sub-session reads back."""

    tools: list[Any] = field(default_factory=list)
    submission: dict[str, Any] = field(default_factory=dict)
    last_params: dict[str, float] | None = None
    last_panel: Image.Image | None = None
    um_per_px: float = 1.0
    calibration_source: str = ""
    previews: int = 0


def _build_align_tools(
    state: StackState, ctx: EngineContext, record: SliceState
) -> _AlignBox:
    box = _AlignBox()
    section = render_slice(ctx, record, long_edge=PREVIEW_LONG_EDGE)
    position_mm = record.position_mm if record.position_mm is not None else 0.0
    box.um_per_px, box.calibration_source = calibrate(state, ctx, record, section)

    def _params(
        rotation_deg: float,
        scale_x: float,
        scale_y: float,
        translate_x_mm: float,
        translate_y_mm: float,
    ) -> dict[str, float]:
        return {
            "rotation_deg": float(rotation_deg),
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
            "translate_x_mm": float(translate_x_mm),
            "translate_y_mm": float(translate_y_mm),
        }

    def preview_transform(
        rotation_deg: float,
        scale_x: float,
        scale_y: float,
        translate_x_mm: float,
        translate_y_mm: float,
        show_template: bool,
    ) -> dict[str, Any]:
        """Render this section under a candidate transform, with atlas outlines.

        Nothing is written. Call it as often as you need.

        Args:
            rotation_deg: Counter-clockwise rotation about the centre of the
                image, in degrees. Negative turns it clockwise.
            scale_x: Horizontal scale multiplier. 1.0 leaves the width alone.
            scale_y: Vertical scale multiplier, same convention.
            translate_x_mm: Horizontal shift in millimetres; positive moves
                right.
            translate_y_mm: Vertical shift in millimetres; positive moves down.
            show_template: True blends the atlas reference image under the
                outlines.

        Returns:
            The parameters you passed, their decomposition, the calibration
            used, and one image: the transformed section with the atlas
            region outlines drawn over it at true physical scale.
        """
        try:
            params = _params(
                rotation_deg, scale_x, scale_y, translate_x_mm, translate_y_mm
            )
        except (TypeError, ValueError):
            return {"status": "error", "error": "BAD_ARGS"}
        try:
            overlay = physical_overlay(
                section,
                box.um_per_px,
                ctx.atlas,
                position_mm,
                cast(Plane, state.plane),
                state.pitch_deg,
                state.yaw_deg,
                params,
                show_template=bool(show_template),
                label=record.id,
                long_edge=OVERLAY_LONG_EDGE,
            )
        except Exception as exc:
            logger.warning("align_slice: overlay failed for %s: %s", record.id, exc)
            return {
                "status": "error",
                "error": "ATLAS_RENDER_FAILED",
                "message": str(exc),
            }
        box.last_params = params
        box.last_panel = overlay
        box.previews += 1
        return {
            "status": "ok",
            "id": record.id,
            "params": params,
            "decomposition": decompose_affine(
                normalized_physical_affine(
                    size=section.size, um_per_px=box.um_per_px, **params
                ),
                section.size,
            ),
            "calibration": {
                "section_um_per_px": round(box.um_per_px, 4),
                "source": box.calibration_source,
            },
            "description": (
                f"{record.id} under the transform above, with the outlines of "
                f"the {state.plane} atlas section at {position_mm:.3f} mm drawn "
                "over it at true physical scale, in each region's own color."
            ),
            TOOL_MEDIA_PARTS_KEY: [image_to_part(overlay)],
        }

    def submit_transform(
        rotation_deg: float,
        scale_x: float,
        scale_y: float,
        translate_x_mm: float,
        translate_y_mm: float,
        confidence: str,
        note: str,
        tool_context: Any = None,
    ) -> dict[str, Any]:
        """Finish this section. Call this exactly once, last.

        Args:
            rotation_deg: Rotation of the accepted alignment, in degrees.
            scale_x: Horizontal scale of the accepted alignment.
            scale_y: Vertical scale of the accepted alignment.
            translate_x_mm: Horizontal shift in millimetres.
            translate_y_mm: Vertical shift in millimetres.
            confidence: "low", "medium" or "high".
            note: Short remark for the record. May be empty.
        """
        try:
            params = _params(
                rotation_deg, scale_x, scale_y, translate_x_mm, translate_y_mm
            )
        except (TypeError, ValueError):
            return {"status": "error", "error": "BAD_ARGS"}
        # Model output is a trust boundary: a malformed submission must not
        # take the run down.
        box.submission.update(
            {
                "params": params,
                "confidence": str(confidence or "").strip().lower(),
                "note": str(note or "").strip(),
            }
        )
        if tool_context is not None:
            tool_context.actions.escalate = True
        return {"status": "ok", "id": record.id, "params": params}

    box.tools = [preview_transform, submit_transform]
    return box


def _align_prompt(
    record: SliceState,
    state: StackState,
    notes: str,
    um_per_px: float,
    calibration_source: str,
) -> str:
    """System instruction for one section's interactive alignment."""
    position = record.position_mm if record.position_mm is not None else 0.0
    damage = record.damage_note.strip()
    lines = [
        f"You are aligning ONE {state.plane} histology section to the reference "
        f"atlas section it belongs to, by eye.",
        "",
        "Facts:",
        f"- Section {record.id} (corrected index {record.index_corrected}), "
        f"placed at {position:.3f} mm along the {state.atlas} atlas.",
        f"- Cutting angles: pitch {state.pitch_deg:.2f} deg, yaw "
        f"{state.yaw_deg:.2f} deg.",
        f"- Canvas: {um_per_px:.2f} um/px, from {calibration_source or 'nothing'}. "
        f"The atlas outlines are drawn at that scale.",
    ]
    if damage:
        lines.append(f"- Recorded damage: {damage}.")
    if notes.strip():
        lines.append(f"- Notes from the caller: {notes.strip()}")
    lines += [
        "",
        "Tools:",
        "- `preview_transform`: renders the section under a candidate "
        "rotation/scale/translation with the atlas region outlines over it, "
        "each in its own color, at true physical scale. Writes nothing.",
        "- `submit_transform`: ends this alignment with the parameters you "
        "settled on.",
        "",
        "Constraints:",
        f"- You have about {ALIGN_MAX_ITERATIONS} tool calls; if you do not "
        "submit, the last preview is taken as your answer.",
        "- Translations are in millimetres; scales are multipliers; rotation "
        "is counter-clockwise degrees about the centre of the image.",
    ]
    return "\n".join(lines)


async def run_align_session(
    state: StackState,
    ctx: EngineContext,
    record: SliceState,
    notes: str,
    *,
    max_iterations: int = ALIGN_MAX_ITERATIONS,
) -> dict[str, Any]:
    """Run the bounded interactive alignment for ONE section.

    Writes nothing: returns ``{"status": "ok", "params": {...physical
    knobs...}, "matrix_params": [...six normalized numbers...],
    "decomposition": {...}, "calibration": {...}, "submitted": bool,
    "note": str, "panel": Image}`` for the caller to record.
    """
    box = _build_align_tools(state, ctx, record)
    agent = build_agent(
        model=ctx.model,
        name="linear_align_slice",
        instruction=_align_prompt(
            record, state, notes, box.um_per_px, box.calibration_source
        ),
        tools=box.tools,
        max_output_tokens=2000,
        reasoning=ctx.spec.reasoning,
    )
    seed = types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text=(
                    f"Align {record.id} to the {state.plane} atlas section at "
                    f"{(record.position_mm or 0.0):.3f} mm, then call "
                    "`submit_transform`."
                )
            )
        ],
    )
    tool_calls, turns = await run_agent_session(
        agent=agent,
        seed_message=seed,
        done=lambda: bool(box.submission),
        nudge_no_tool=_NUDGE_NO_TOOL,
        nudge_continue=_NUDGE_CONTINUE,
        max_iterations=max_iterations,
        run_label=f"linear_align_{record.index_corrected:03d}",
    )

    submitted = bool(box.submission)
    params = dict(box.submission["params"]) if submitted else box.last_params
    if params is None:
        return {
            "status": "error",
            "error": "NO_TRANSFORM",
            "id": record.id,
            "turns": turns,
            "tool_calls": tool_calls,
        }
    section = render_slice(ctx, record, long_edge=PREVIEW_LONG_EDGE)
    matrix_params = normalized_physical_affine(
        size=section.size, um_per_px=box.um_per_px, **params
    )
    return {
        "status": "ok",
        "id": record.id,
        "params": params,
        "matrix_params": matrix_params,
        "decomposition": decompose_affine(matrix_params, section.size),
        "calibration": {
            "section_um_per_px": round(box.um_per_px, 4),
            "source": box.calibration_source,
        },
        "submitted": submitted,
        "confidence": str(box.submission.get("confidence", "")),
        "note": str(box.submission.get("note", "")),
        "previews": box.previews,
        "turns": turns,
        "tool_calls": tool_calls,
        "panel": box.last_panel,
    }

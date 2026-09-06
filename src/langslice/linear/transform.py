"""In-plane transforms: the closed-form fit and the interactive sub-session.

Two routes to the same field. :func:`fit_silhouette` is plain code — the shared
moments fit (:func:`langslice.affine.silhouette_affine`) of a section's
silhouette onto its atlas section. :func:`run_align_session` is a bounded agent
loop for ONE section (preview → look → adjust → submit), for sections whose
silhouette is exactly what damage destroyed.

Everything here is a PROPOSAL: six normalized numbers recorded on the state,
never applied to the user's images (see :mod:`langslice.affine` for the
convention).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from google.genai import types
from PIL import Image

from langslice.adk import TOOL_MEDIA_PARTS_KEY
from langslice.affine import affine_matrix, normalized_affine, silhouette_affine
from langslice.linear.atlas_fetch import atlas_section
from langslice.linear.render import (
    PREVIEW_LONG_EDGE,
    image_to_part,
    preview_panel,
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


# --- the closed-form fit -------------------------------------------------


def fit_silhouette(
    state: StackState, ctx: EngineContext, record: SliceState
) -> dict[str, Any]:
    """Fit the silhouette affine for one positioned, undamaged section.

    Returns the tool-shaped payload: on success ``params`` (six normalized
    numbers), ``iou``, and a ``panel`` image. ``flat_atlas_fit`` is reported
    when the stack carries cutting angles: the moments fit measures against the
    flat atlas section, because it builds its own atlas silhouette from the
    voxel grid.
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

    fitted = Image.fromarray(fit.slice_rgb)
    panel = preview_panel(
        fitted, atlas_section(ctx, state, record.position_mm), fit.matrix
    )
    payload: dict[str, Any] = {
        "status": "ok",
        "id": record.id,
        "position_mm": round(record.position_mm, 3),
        "iou": round(float(fit.iou), 3),
        "params": normalized_affine(fit.matrix, fit.size),
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
    previews: int = 0


def _build_align_tools(
    state: StackState, ctx: EngineContext, record: SliceState
) -> _AlignBox:
    box = _AlignBox()
    section = render_slice(ctx, record, long_edge=PREVIEW_LONG_EDGE)
    position_mm = record.position_mm if record.position_mm is not None else 0.0

    def _params(
        rotation_deg: float,
        scale_x: float,
        scale_y: float,
        translate_x: float,
        translate_y: float,
    ) -> dict[str, float]:
        return {
            "rotation_deg": float(rotation_deg),
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
            "translate_x": float(translate_x),
            "translate_y": float(translate_y),
        }

    def preview_transform(
        rotation_deg: float,
        scale_x: float,
        scale_y: float,
        translate_x: float,
        translate_y: float,
    ) -> dict[str, Any]:
        """Render this section under a candidate transform, over its atlas section.

        Nothing is written. Call it as often as you need.

        Args:
            rotation_deg: Counter-clockwise rotation about the centre of the
                section, in degrees. Negative turns it clockwise.
            scale_x: Horizontal scale multiplier. 1.0 leaves the width alone.
            scale_y: Vertical scale multiplier, same convention.
            translate_x: Horizontal shift as a FRACTION of image width;
                positive moves right.
            translate_y: Vertical shift as a fraction of image height;
                positive moves down.

        Returns:
            The parameters you passed plus a panel: the transformed section,
            the atlas section it has to match, and the two overlaid.
        """
        try:
            params = _params(rotation_deg, scale_x, scale_y, translate_x, translate_y)
        except (TypeError, ValueError):
            return {"status": "error", "error": "BAD_ARGS"}
        matrix = affine_matrix(size=section.size, **params)
        try:
            reference = atlas_section(ctx, state, position_mm)
        except Exception as exc:
            logger.warning("align_slice: atlas render failed for %s: %s", record.id, exc)
            return {
                "status": "error",
                "error": "ATLAS_RENDER_FAILED",
                "message": str(exc),
            }
        panel = preview_panel(section, reference, matrix)
        box.last_params = params
        box.last_panel = panel
        box.previews += 1
        return {
            "status": "ok",
            "id": record.id,
            "params": params,
            "description": (
                f"{record.id} under the transform above, against the "
                f"{state.plane} atlas section at {position_mm:.3f} mm. Panels, "
                "left to right: transformed section, atlas section, the two "
                "overlaid (magenta = section, green = atlas; white where they "
                "agree)."
            ),
            TOOL_MEDIA_PARTS_KEY: [image_to_part(panel)],
        }

    def submit_transform(
        rotation_deg: float,
        scale_x: float,
        scale_y: float,
        translate_x: float,
        translate_y: float,
        confidence: str,
        note: str,
        tool_context: Any = None,
    ) -> dict[str, Any]:
        """Finish this section. Call this exactly once, last.

        Args:
            rotation_deg: Rotation of the accepted alignment, in degrees.
            scale_x: Horizontal scale of the accepted alignment.
            scale_y: Vertical scale of the accepted alignment.
            translate_x: Horizontal shift, as a fraction of image width.
            translate_y: Vertical shift, as a fraction of image height.
            confidence: "low", "medium" or "high".
            note: Short remark for the record. May be empty.
        """
        try:
            params = _params(rotation_deg, scale_x, scale_y, translate_x, translate_y)
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


def _align_prompt(record: SliceState, state: StackState, notes: str) -> str:
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
    ]
    if damage:
        lines.append(f"- Recorded damage: {damage}.")
    if notes.strip():
        lines.append(f"- Notes from the caller: {notes.strip()}")
    lines += [
        "",
        "Tools:",
        "- `preview_transform`: renders the section under a candidate "
        "rotation/scale/translation, over its atlas section, plus an overlay "
        "(magenta = section, green = atlas, white where they agree). Writes "
        "nothing.",
        "- `submit_transform`: ends this alignment with the parameters you "
        "settled on.",
        "",
        "Constraints:",
        f"- You have about {ALIGN_MAX_ITERATIONS} tool calls; if you do not "
        "submit, the last preview is taken as your answer.",
        "- Translations are fractions of image width/height; scales are "
        "multipliers; rotation is counter-clockwise degrees.",
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

    Writes nothing: returns ``{"status": "ok", "params": {...five knobs...},
    "matrix_params": [...six numbers...], "submitted": bool, "note": str,
    "panel": Image}`` for the caller to record.
    """
    box = _build_align_tools(state, ctx, record)
    agent = build_agent(
        model=ctx.model,
        name="linear_align_slice",
        instruction=_align_prompt(record, state, notes),
        tools=box.tools,
        max_output_tokens=2000,
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
    matrix = affine_matrix(size=section.size, **params)
    return {
        "status": "ok",
        "id": record.id,
        "params": params,
        "matrix_params": normalized_affine(matrix, section.size),
        "submitted": submitted,
        "confidence": str(box.submission.get("confidence", "")),
        "note": str(box.submission.get("note", "")),
        "previews": box.previews,
        "turns": turns,
        "tool_calls": tool_calls,
        "panel": box.last_panel,
    }

"""The transform step: one in-plane alignment proposal per section.

Two routes, decided by the damage flag the survey set:

* INTACT sections take the plain-code route — the shared silhouette affine
  (:func:`langslice.affine.silhouette_affine`) against the atlas section the
  positioning step assigned. No model, no turns. A section whose fit fails
  keeps its position and picks up a caveat; the run never dies for it.
* DAMAGED sections take the interactive route — one agent session per section,
  computer-use style: propose rotation/scale/translation, look at the rendered
  overlay against the atlas section, adjust, repeat, submit. Their silhouette
  is exactly what damage destroyed, so the closed-form fit has nothing to bite
  on and a pair of eyes does better.

Everything written here is a PROPOSAL — parameters recorded on the state, never
applied to the user's images. See :mod:`langslice.affine` for the parameter
convention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from google.adk.agents import LlmAgent
from google.genai import types
from PIL import Image, ImageDraw

from langslice.adk import TOOL_MEDIA_PARTS_KEY
from langslice.adk.model_resolver import default_http_options, resolve_adk_model
from langslice.affine import affine_matrix, normalized_affine, silhouette_affine
from langslice.atlas.core import get_composite_slice
from langslice.linear.tools import _image_to_part
from langslice.linear.whole_brain._step_common import render_slice, run_agent_session
from langslice.linear.whole_brain.engine import EngineContext
from langslice.linear.whole_brain.state import (
    SliceState,
    StackState,
    apply_confidence,
)

logger = logging.getLogger(__name__)

#: Model turns (counted as tool calls) one damaged section may spend.
DEFAULT_TRANSFORM_MAX_ITERATIONS = 12

DEFAULT_TRANSFORM_MODEL = "gemini-3-flash-preview"

#: Working frame for the interactive loop. The preview panel is three of these
#: side by side, which is as much detail as the overlay judgement needs.
PREVIEW_LONG_EDGE = 512

#: Below this silhouette overlap the closed-form fit is not to be trusted.
WEAK_FIT_IOU = 0.65

AFFINE_FAILED_CAVEAT = "affine failed"
INCOMPLETE_CAVEAT = "interactive transform incomplete"

_NUDGE_NO_TOOL = (
    "You did not call a tool. Do not answer in prose: render a candidate "
    "alignment with `preview_transform`, then finish with `submit_transform`."
)
_NUDGE_CONTINUE = (
    "Keep going. Adjust the parameters and call `preview_transform` again, or "
    "call `submit_transform` if the current alignment is the best you can get."
)


# --- intact sections: the closed-form affine -----------------------------


def run_affine_pass(state: StackState, ctx: EngineContext) -> tuple[int, int]:
    """Fit the silhouette affine for every intact, positioned section.

    Returns ``(fitted, failed)``. A failure is logged, noted and caveated —
    never raised: one unreadable section must not cost the whole stack its
    transforms.
    """
    atlas = ctx.atlas_loader(state.atlas or ctx.config.atlas)
    fitted = 0
    failed = 0
    for record in state.in_order():
        if record.damaged or record.position_mm is None:
            continue
        try:
            fit = silhouette_affine(
                render_slice(ctx, record, long_edge=PREVIEW_LONG_EDGE),
                atlas=atlas,
                position_mm=record.position_mm,
                plane=state.plane,  # type: ignore[arg-type]
            )
        except Exception as exc:
            logger.warning("transforms: affine failed for %s: %s", record.id, exc)
            failed += 1
            _add_caveat(record, AFFINE_FAILED_CAVEAT)
            state.notes.append(f"transforms: affine failed for {record.id} ({exc})")
            continue
        record.affine = normalized_affine(fit.matrix, fit.size)
        fitted += 1
        if fit.iou < WEAK_FIT_IOU:
            _add_caveat(record, f"weak affine fit (silhouette overlap {fit.iou:.2f})")
        ctx.progress(
            f"[transforms] {record.id}: affine fitted "
            f"(silhouette overlap {fit.iou:.2f})"
        )
    return fitted, failed


def _add_caveat(record: SliceState, caveat: str) -> None:
    if caveat not in record.caveats:
        record.caveats.append(caveat)


# --- damaged sections: the interactive loop ------------------------------


@dataclass
class TransformOutcome:
    """What one interactive session produced for one section.

    ``params`` is the submitted transform, or the last previewed one when the
    turn budget ran out before a submission, or ``None`` when the session
    produced nothing at all.
    """

    params: dict[str, float] | None = None
    submitted: bool = False
    previews: int = 0
    tool_calls: int = 0
    turns: int = 0


@dataclass
class TransformToolBox:
    """Tool callables plus the mutable results the step reads back."""

    tools: list[Any] = field(default_factory=list)
    submission: dict[str, Any] = field(default_factory=dict)
    last_preview: dict[str, float] | None = None
    previews: int = 0


def preview_panel(
    section: Image.Image, atlas_section: Image.Image, matrix: np.ndarray
) -> Image.Image:
    """Transformed section | atlas section | overlay, side by side, labelled.

    The overlay puts the transformed section in magenta and the atlas section
    in green: where the two agree the pixels go white, and every mismatch shows
    up as a coloured fringe on the side that is out of place.
    """
    width, height = section.size
    section_gray = np.asarray(section.convert("L"), dtype=np.uint8)
    warped = cv2.warpAffine(
        section_gray, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=0
    )
    atlas_resized = atlas_section.convert("RGB").resize(
        (width, height), resample=Image.Resampling.BILINEAR
    )
    atlas_gray = np.asarray(atlas_resized.convert("L"), dtype=np.uint8)
    overlay = np.dstack([warped, atlas_gray, warped])

    cells = [
        (Image.fromarray(warped, mode="L").convert("RGB"), "transformed section"),
        (atlas_resized, "atlas section"),
        (Image.fromarray(overlay, mode="RGB"), "overlay: magenta=section, green=atlas"),
    ]
    label_px = 14
    panel = Image.new("RGB", (width * len(cells), height + label_px), (16, 16, 16))
    draw = ImageDraw.Draw(panel)
    for index, (image, label) in enumerate(cells):
        panel.paste(image, (index * width, 0))
        draw.text((index * width + 3, height + 2), label, fill=(235, 235, 235))
    return panel


def build_transform_tools(
    record: SliceState, state: StackState, ctx: EngineContext
) -> TransformToolBox:
    """Build the interactive tool set, closed over ONE section."""
    box = TransformToolBox()
    atlas = ctx.atlas_loader(state.atlas or ctx.config.atlas)
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

        Nothing is written and no file is touched — this is your eye. Call it
        as often as you need, changing one parameter at a time so you can see
        what each change did.

        Args:
            rotation_deg: Counter-clockwise rotation about the centre of the
                section, in degrees. Negative turns it clockwise.
            scale_x: Horizontal scale multiplier. 1.0 leaves the width alone,
                1.1 widens by 10%, 0.9 narrows by 10%.
            scale_y: Vertical scale multiplier, same convention.
            translate_x: Horizontal shift as a FRACTION of image width.
                Positive moves right; 0.05 is a twentieth of the width.
            translate_y: Vertical shift as a fraction of image height.
                Positive moves down.

        Returns:
            The parameters you passed plus three images: the transformed
            section, the atlas section it has to match, and the two overlaid.
        """
        try:
            params = _params(
                rotation_deg, scale_x, scale_y, translate_x, translate_y
            )
        except (TypeError, ValueError):
            return {"status": "error", "error": "BAD_ARGS"}
        matrix = affine_matrix(size=section.size, **params)
        try:
            atlas_section = get_composite_slice(atlas, position_mm, plane=state.plane)  # type: ignore[arg-type]
        except Exception as exc:
            logger.warning("transforms: atlas render failed for %s: %s", record.id, exc)
            return {"status": "error", "error": "ATLAS_RENDER_FAILED", "message": str(exc)}
        panel = preview_panel(section, atlas_section, matrix)

        box.last_preview = params
        box.previews += 1
        return {
            "status": "ok",
            "id": record.id,
            "params": params,
            "description": (
                f"{record.id} under the transform above, against the "
                f"{state.plane} atlas section at {position_mm:.2f} mm. Panels, "
                "left to right: transformed section, atlas section, the two "
                "overlaid (magenta = section, green = atlas; white where they "
                "agree)."
            ),
            TOOL_MEDIA_PARTS_KEY: [_image_to_part(panel)],
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

        Submit the parameters of the alignment you settled on — the same five
        numbers you last previewed, unless you are correcting them.

        Args:
            rotation_deg: Rotation of the accepted alignment, in degrees.
            scale_x: Horizontal scale of the accepted alignment.
            scale_y: Vertical scale of the accepted alignment.
            translate_x: Horizontal shift, as a fraction of image width.
            translate_y: Vertical shift, as a fraction of image height.
            confidence: "low", "medium" or "high" — how well the damaged
                section could be aligned at all.
            note: Short remark for the record, e.g. which part of the section
                is missing and could not be matched. May be empty.
        """
        try:
            params = _params(
                rotation_deg, scale_x, scale_y, translate_x, translate_y
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


def build_transform_prompt(record: SliceState, state: StackState) -> str:
    """System instruction for one damaged section's interactive loop."""
    damage = record.damage_note or "damage was reported but not described"
    position = record.position_mm if record.position_mm is not None else 0.0
    return (
        f"You are aligning ONE damaged {state.plane} histology section to the "
        f"reference atlas section it belongs to, by eye.\n\n"
        f"Section: {record.id} (corrected index {record.index_corrected}), "
        f"placed at {position:.3f} mm along the "
        f"{state.atlas} atlas.\n"
        f"Reported damage: {damage}.\n\n"
        f"The automatic shape-matching alignment is not used for this section: "
        f"it matches the OUTLINE of the tissue, and this section's outline is "
        f"exactly what the damage destroyed. You align it by looking instead.\n\n"
        f"HOW TO WORK:\n\n"
        f"1. Start with `preview_transform(rotation_deg=0, scale_x=1, "
        f"scale_y=1, translate_x=0, translate_y=0)` to see where the section "
        f"sits before any transform.\n\n"
        f"2. Read the overlay panel. The section is magenta, the atlas is "
        f"green, and where they agree the pixels turn white. Magenta sticking "
        f"out on one side means the section has to move or shrink that way; "
        f"green showing through means the section does not reach that far.\n\n"
        f"3. Change ONE parameter at a time and preview again. Rotation first "
        f"(a tilted section never lines up by translation), then translation "
        f"to centre it, then scale. Translations are fractions of the image: "
        f"0.02 is a small nudge, 0.1 a large one.\n\n"
        f"4. IGNORE THE MISSING TISSUE. Line up the parts of the section that "
        f"survived — the midline, the ventricles, the outer border where it is "
        f"intact. Do not stretch the section to cover a tear; that would "
        f"misplace everything that is still there.\n\n"
        f"5. Finish with `submit_transform`, giving the parameters you settled "
        f"on, an honest confidence, and a note on what could not be matched. "
        f"You have about {DEFAULT_TRANSFORM_MAX_ITERATIONS} tool calls — "
        f"submit before you run out, or the last preview is taken as your "
        f"answer."
    )


def build_transform_seed_message(record: SliceState, state: StackState) -> types.Content:
    """The opening turn: what this section is and what it has to match."""
    position = record.position_mm if record.position_mm is not None else 0.0
    return types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text=(
                    f"Align {record.id} to the {state.plane} atlas section at "
                    f"{position:.3f} mm. Start by previewing the "
                    f"identity transform (rotation 0, scales 1, translations "
                    f"0) to see how the untransformed section sits, then work "
                    f"from what the overlay shows you."
                )
            )
        ],
    )


def build_transform_agent(
    *,
    record: SliceState,
    state: StackState,
    tools: list[Any],
    model: str | object = DEFAULT_TRANSFORM_MODEL,
    media_resolution: str = "MEDIA_RESOLUTION_MEDIUM",
) -> LlmAgent:
    """Construct the interactive-transform LlmAgent for one section."""
    # Same shape as the other whole-brain agents: kwargs dict so the enum-typed
    # media_resolution string is accepted as-is.
    config_kwargs: dict[str, Any] = {
        "temperature": 1.0,
        # Short arguments, plenty of looking: this step needs far less output
        # room than the batch steps.
        "max_output_tokens": 2000,
        "media_resolution": media_resolution,
        "http_options": default_http_options(),
    }
    return LlmAgent(
        model=resolve_adk_model(model),  # type: ignore[arg-type]
        name="whole_brain_transform",
        instruction=build_transform_prompt(record, state),
        tools=tools,
        generate_content_config=types.GenerateContentConfig(**config_kwargs),
    )


async def run_transform_session(
    *,
    record: SliceState,
    state: StackState,
    ctx: EngineContext,
    pos_lo: float,
    pos_hi: float,
    max_iterations: int = DEFAULT_TRANSFORM_MAX_ITERATIONS,
) -> TransformOutcome:
    """Drive the interactive loop for ONE damaged section."""
    box = build_transform_tools(record, state, ctx)
    agent = build_transform_agent(
        record=record,
        state=state,
        tools=box.tools,
        model=ctx.model or DEFAULT_TRANSFORM_MODEL,
    )

    tool_calls, turns = await run_agent_session(
        agent=agent,
        state=state,
        pos_lo=pos_lo,
        pos_hi=pos_hi,
        seed_message=build_transform_seed_message(record, state),
        done=lambda: bool(box.submission),
        nudge_no_tool=_NUDGE_NO_TOOL,
        nudge_continue=_NUDGE_CONTINUE,
        max_iterations=max_iterations,
        run_label=f"whole_brain_transform_{record.index_corrected:03d}",
    )

    submitted = bool(box.submission)
    params = (
        dict(box.submission["params"]) if submitted else box.last_preview
    )
    if submitted:
        apply_confidence(record, box.submission.get("confidence"))
        note = str(box.submission.get("note", ""))
        if note:
            _add_caveat(record, f"interactive transform: {note}")
    elif params is not None:
        # Out of turns with a preview on the table: the last thing the agent
        # looked at is a better proposal than nothing, but it never blessed it.
        _add_caveat(record, INCOMPLETE_CAVEAT)

    if params is not None:
        record.interactive_transform = dict(params)

    return TransformOutcome(
        params=params,
        submitted=submitted,
        previews=box.previews,
        tool_calls=tool_calls,
        turns=turns,
    )


async def run_interactive_transforms(
    state: StackState, ctx: EngineContext, *, pos_lo: float, pos_hi: float
) -> tuple[int, int]:
    """Run one interactive session per damaged section; return ``(done, empty)``.

    Sequential on purpose: these are full agent sessions, and running a stack's
    worth of them concurrently is the fastest way to hit a provider rate limit.
    # ponytail: one section at a time; fan out with a semaphore if wall-clock
    # on damaged-heavy stacks ever matters.
    """
    damaged = [record for record in state.in_order() if record.damaged]
    done = 0
    empty = 0
    for record in damaged:
        try:
            outcome = await run_transform_session(
                record=record, state=state, ctx=ctx, pos_lo=pos_lo, pos_hi=pos_hi
            )
        except Exception as exc:
            logger.warning(
                "transforms: interactive session failed for %s: %s", record.id, exc
            )
            state.notes.append(
                f"transforms: interactive session failed for {record.id} ({exc})"
            )
            _add_caveat(record, INCOMPLETE_CAVEAT)
            empty += 1
            continue
        if outcome.params is None:
            empty += 1
            _add_caveat(record, INCOMPLETE_CAVEAT)
            state.notes.append(
                f"transforms: {record.id} produced no transform in "
                f"{outcome.turns} turns"
            )
            ctx.progress(f"[transforms] {record.id}: no transform proposed")
            continue
        done += 1
        ctx.progress(
            f"[transforms] {record.id}: interactive transform "
            f"{'submitted' if outcome.submitted else 'incomplete, last preview kept'} "
            f"after {outcome.previews} preview(s)"
        )
    return done, empty

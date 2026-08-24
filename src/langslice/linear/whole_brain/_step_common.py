"""Pieces shared by the whole-brain agent steps (survey, position, transforms,
review).

Every agent step looks at the same stack the same way: sections rendered
through the corrected view (``render_slice``, ``view_slices``,
``stack_image_parts``), a text manifest of the stack, and the same ADK session
loop (create session, run turns, nudge when the model answers in prose, stop
when the step's submit tool has escalated). Only the tools, the prompt and the
seed message differ, so those stay with the step.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.apps.app import App
from google.adk.runners import InMemoryRunner
from google.genai import types
from PIL import Image

from langslice.adk import TOOL_MEDIA_PARTS_KEY
from langslice.image_prep import (
    adaptive_preprocess,
    normalize_image,
    prepare_image_for_vlm,
)
from langslice.linear.runner import _APP_NAME, _USER_ID, _build_plugins
from langslice.linear.session import build_initial_state
from langslice.linear.tools import _image_to_part
from langslice.linear.whole_brain.engine import EngineContext
from langslice.linear.whole_brain.state import SliceState, StackState
from langslice.linear.whole_brain.trace import open_trace
from langslice.space import Plane

logger = logging.getLogger(__name__)

#: Sections one ``view_slices`` call may return.
MAX_VIEW_SLICES = 8
#: Long edge for ``view_slices`` images: enough to zoom past the seed-message
#: stack images without paying full-resolution tokens.
VIEW_LONG_EDGE = 1024
#: Long edge for the per-section images in a step's seed message. Small on
#: purpose: the whole stack (40-odd sections) rides in one user message and
#: stays in context for the whole session.
SEED_IMAGE_LONG_EDGE = 512


def split_known_ids(state: StackState, slice_ids: list[str]) -> tuple[list[str], list[str]]:
    """Partition *slice_ids* into ids the stack knows and ids it does not."""
    known = {s.id for s in state.slices}
    return (
        [sid for sid in slice_ids if sid in known],
        [sid for sid in slice_ids if sid not in known],
    )


def render_slice(
    ctx: EngineContext, record: SliceState, *, long_edge: int = VIEW_LONG_EDGE
) -> Image.Image:
    """One section as the engine sees it: normalized, downsampled, enhanced, flipped.

    Every step that shows or measures a section goes through here, so the
    pixels the agent judges are the pixels the transform step fits against.

    With ``config.preprocess == "auto"`` (the default) the section is run
    through :func:`~langslice.image_prep.adaptive_preprocess` — per-channel
    CLAHE plus a DAPI-weighted grayscale blend — so dim fluorescence reads like
    the atlas instead of like a black field. Display only: the user's file is
    never touched.

    Renders are cached on *ctx* (see :attr:`EngineContext.render_cache`), so
    the returned image is shared: read it, never mutate it in place.
    """
    key = (record.id, record.flip, long_edge, ctx.config.preprocess)
    cached = ctx.render_cache.get(key)
    if cached is not None:
        return cached

    with Image.open(ctx.image_path(record.id)) as handle:
        # Detach from the file handle: prepare_image_for_vlm can hand back the
        # very object it was given when no resize is needed.
        source = normalize_image(handle.copy())
    prepped = prepare_image_for_vlm(source, max_long_edge=long_edge).image
    if ctx.config.preprocess == "auto":
        prepped = adaptive_preprocess(prepped)
    if record.flip:
        prepped = prepped.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    ctx.render_cache[key] = prepped
    return prepped


def make_view_slices(
    state: StackState, ctx: EngineContext
) -> Callable[[list[str]], dict[str, Any]]:
    """Build the shared ``view_slices`` tool, closed over *state*."""

    def view_slices(slice_ids: list[str]) -> dict[str, Any]:
        """Look at up to 8 named sections at higher resolution.

        Use this whenever the stack images you were given are too small to
        judge damage, a notch, a hemisphere flip, or which atlas level a
        section matches. Sections are rendered through the corrected state: a
        section that has been flipped comes back mirrored, i.e. as it will be
        used.

        Args:
            slice_ids: Filenames from the stack manifest (max 8 per call).

        Returns:
            status/slice_ids/description plus the images, in the requested
            order. On failure, ``{"status": "error", "error": ...}``.
        """
        if not slice_ids:
            return {"status": "error", "error": "BAD_ARGS"}
        wanted, unknown = split_known_ids(state, list(slice_ids)[:MAX_VIEW_SLICES])
        if not wanted:
            return {"status": "error", "error": "UNKNOWN_SLICE_IDS", "unknown": unknown}

        parts: list[types.Part] = []
        for slice_id in wanted:
            record = state.by_id(slice_id)
            assert record is not None
            parts.append(_image_to_part(render_slice(ctx, record)))

        return {
            "status": "ok",
            "slice_ids": wanted,
            "unknown_ids": unknown,
            # The attached images are unlabelled, so this ordering note is the
            # only way the model can tie an image back to a filename.
            "description": (
                "Attached images are "
                + ", ".join(wanted)
                + ", in that order, rendered with any flips already applied."
            ),
            TOOL_MEDIA_PARTS_KEY: parts,
        }

    return view_slices


def slice_flags(record: SliceState) -> list[str]:
    """The section's current corrections, as short human-readable flags."""
    flags: list[str] = []
    if record.flip:
        flags.append("flipped")
    if record.damaged:
        flags.append(
            f"damaged: {record.damage_note}" if record.damage_note else "damaged"
        )
    return flags


def build_stack_manifest(state: StackState, *, with_positions: bool = False) -> str:
    """One line per section in corrected order, with its current flags.

    With *with_positions*, each line also carries the section's current
    position and where that position came from.
    """
    lines: list[str] = []
    for record in state.in_order():
        flags = slice_flags(record)
        suffix = f"  [{'; '.join(flags)}]" if flags else ""
        position = ""
        if with_positions and record.position_mm is None:
            position = "  unplaced"
        elif with_positions:
            origin = record.position_source or "unknown"
            if record.confidence:
                origin += f", {record.confidence} confidence"
            position = f"  {record.position_mm:.3f} mm ({origin})"
        lines.append(f"{record.index_corrected:>3}  {record.id}{position}{suffix}")
    return "\n".join(lines)


def stack_image_parts(
    state: StackState, ctx: EngineContext, *, long_edge: int = SEED_IMAGE_LONG_EDGE
) -> list[types.Part]:
    """The whole stack as labelled text+image pairs, in corrected order.

    Each section gets a one-line label — ``"<corrected index>: <filename>"``
    plus any flags — immediately followed by its own image, rendered through
    :func:`render_slice` so preprocessing and flips are already applied.

    One image per section rather than one thumbnail grid: a grid splits a fixed
    vision-encoder patch budget across every section at once and lets
    neighbouring sections share patch boundaries. A labelled sequence at a
    modest resolution reads better than a big sheet, and the label is what
    binds each set of pixels to a filename the model can quote back.
    """
    parts: list[types.Part] = [
        types.Part.from_text(
            text=(
                f"The {len(state.slices)} sections of the stack follow, in "
                "their current corrected order, one image each. Every image is "
                "preceded by its label '<index>: <filename>' and is rendered "
                "with any flips already applied."
            )
        )
    ]
    for record in state.in_order():
        flags = slice_flags(record)
        label = f"{record.index_corrected}: {record.id}"
        if flags:
            label += f"  [{'; '.join(flags)}]"
        parts.append(types.Part.from_text(text=label))
        parts.append(_image_to_part(render_slice(ctx, record, long_edge=long_edge)))
    return parts


async def run_agent_session(
    *,
    agent: LlmAgent,
    state: StackState,
    pos_lo: float,
    pos_hi: float,
    seed_message: types.Content,
    done: Callable[[], bool],
    nudge_no_tool: str,
    nudge_continue: str,
    max_iterations: int,
    run_label: str,
) -> tuple[int, int]:
    """Drive one whole-stack agent pass; return ``(tool_calls, turns)``.

    Ends when *done* reports the step's submit tool has fired, or when the
    turn/tool-call budget runs out. Tools mutate the state as they are called,
    so a pass that never submits still leaves its writes behind.

    With ``LANGSLICE_TRACE_DIR`` set, everything the agent is shown, says, calls
    and gets back is appended to a JSONL trace named after *run_label* (see
    :mod:`langslice.linear.whole_brain.trace`).
    """
    trace = open_trace(run_label, agent=agent)
    app = App(name=_APP_NAME, root_agent=agent, plugins=_build_plugins(run_label))
    runner = InMemoryRunner(app=app)
    assert runner.session_service is not None

    plane: Plane = state.plane  # type: ignore[assignment]
    await runner.session_service.create_session(
        app_name=_APP_NAME,
        user_id=_USER_ID,
        session_id=run_label,
        # fetch_atlas reads atlas/plane/pos_lo/pos_hi from session state.
        state=build_initial_state(
            atlas_name=state.atlas,
            plane=plane,
            pos_lo=pos_lo,
            pos_hi=pos_hi,
            n_slices=len(state.slices),
            interval_mm=state.interval_mm,
            thickness_um=int(round(state.thickness_mm * 1000)),
            max_iterations=max_iterations,
        ),
    )

    message = seed_message
    if trace is not None:
        trace.seed(seed_message)
    # The nudge is traced where it is sent, not where it is written: the last
    # one built before the budget runs out never reaches the model.
    nudge: str | None = None
    tool_calls = 0
    turns = 0
    while turns < max_iterations and not done():
        turns += 1
        if trace is not None and nudge is not None:
            trace.nudge(nudge, turn=turns)
        saw_tool_call = False
        async for event in runner.run_async(
            user_id=_USER_ID, session_id=run_label, new_message=message
        ):
            if trace is not None:
                trace.event(event, turn=turns)
            calls = event.get_function_calls() or []
            if calls:
                saw_tool_call = True
                tool_calls += len(calls)
                logger.info(
                    "%s turn %d: %s (tool calls=%d)",
                    run_label,
                    turns,
                    [getattr(call, "name", "?") for call in calls],
                    tool_calls,
                )
            if done() or tool_calls > max_iterations:
                break
        if done():
            break
        if tool_calls > max_iterations:
            logger.warning(
                "%s hit max_iterations=%d; ending pass", run_label, max_iterations
            )
            break
        nudge = nudge_continue if saw_tool_call else nudge_no_tool
        message = types.Content(role="user", parts=[types.Part.from_text(text=nudge)])

    if trace is not None:
        trace.summary(tool_calls=tool_calls, turns=turns, submitted=done())
    return tool_calls, turns

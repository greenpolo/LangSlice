"""The run: ingest, one agent session, results.

``run(spec)`` is the whole of ``langslice linear``. There is no node graph: one
state, one toolbox, one job statement, and a session that ends at ``submit`` or
the turn budget. Every write tool checkpoints, so a run that dies mid-way
resumes from the checkpoint with the state it had — the agent is re-seeded, not
replayed.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from google.genai import types

from langslice.adk.media import opening_parts
from langslice.atlas.core import load_atlas
from langslice.linear.checkpoint import (
    default_checkpoint_path,
    load_checkpoint,
    observe_checkpoints,
    save_checkpoint,
)
from langslice.linear.discovery import discover_slices
from langslice.linear.display import display_facts
from langslice.linear.live import LiveCallback
from langslice.linear.prompt import build_job_statement
from langslice.linear.render import status_text
from langslice.linear.session import (
    DEFAULT_MAX_ITERATIONS,
    build_agent,
    run_agent_session,
)
from langslice.linear.spec import JobSpec
from langslice.linear.state import SliceState, StackState
from langslice.linear.toolbox import ToolBox, build_tools
from langslice.linear.workspace import Workspace, log_progress

RESULTS_FILENAME = "linear_results.json"

_RUN_LABEL = "linear_stack"

_NUDGE_NO_TOOL = (
    "You did not call a tool. Continue with the tools rather than in prose; "
    "call `submit` when the job is done."
)
_NUDGE_CONTINUE = "Continue; call `submit` when the job is done."

#: Asked once, after submit, in the same context. The answer is recorded on the
#: state (``debrief``) for the people building this environment; nothing about
#: the run changes. Answer in text: tools are still live, and a call here would
#: be a write after submit.
DEBRIEF_PROMPT = (
    "The job is submitted and nothing you say now changes it. This is a "
    "debrief for the people building this tool environment; answer in text "
    "and do not call any tool.\n"
    "1. Which tools or pieces of information did you reach for, or wish "
    "existed, that were not available — including while aligning sections: "
    "views, overlays, controls, measurements?\n"
    "2. Which tools behaved differently from what you expected, or were "
    "awkward to use as specified?\n"
    "3. What did you have to work around?\n"
    "4. Which tool environments you know does this resemble, and what did "
    "those have that this lacks?\n"
    "5. Your wishlist: what would you want in this environment to do this "
    "job well?"
)


@dataclass(kw_only=True)
class EngineContext(Workspace):
    """The agent driver's context: the core :class:`Workspace` plus what only
    the driver needs — the job's files, the model, and the cache of encoded
    message images (:mod:`langslice.adk.media`)."""

    checkpoint_path: str
    results_path: str
    model: str
    #: Encoded, captioned reference images shared by the seed and comparison tools.
    reference_parts: dict[tuple[Any, ...], types.Part] = field(default_factory=dict, repr=False)


def build_context(
    spec: JobSpec,
    *,
    emit: Callable[[str], None] | None = None,
    atlas_loader: Callable[[str], Any] | None = None,
) -> EngineContext:
    """Assemble an :class:`EngineContext` from a job spec."""
    from langslice.providers.openai_oauth import DEFAULT_REVIEW_MODEL

    folder = os.path.abspath(spec.image_folder)
    return EngineContext(
        spec=spec,
        image_folder=folder,
        checkpoint_path=default_checkpoint_path(folder),
        results_path=spec.out or os.path.join(folder, RESULTS_FILENAME),
        model=spec.model or DEFAULT_REVIEW_MODEL,
        emit=emit or log_progress,
        atlas_loader=atlas_loader or load_atlas,
    )


# --- ingest --------------------------------------------------------------


def ingest(spec: JobSpec, ctx: EngineContext) -> StackState:
    """Discover the folder and build the stack state. Plain code, no model."""
    paths = discover_slices(ctx.image_folder)
    if not paths:
        raise ValueError(f"No slice images found in {ctx.image_folder}")

    pos_lo, pos_hi = ctx.position_range
    state = StackState(
        image_folder=ctx.image_folder,
        atlas=spec.atlas,
        plane=spec.plane,
        interval_mm=spec.interval_mm,
        thickness_mm=spec.thickness_mm,
        spec=spec.to_dict(),
        slices=[
            SliceState(
                id=os.path.basename(path), index_original=index, index_corrected=index
            )
            for index, path in enumerate(paths)
        ],
    )
    state.notes.append(
        f"ingest: {len(paths)} sections, atlas {spec.atlas} ({spec.plane}) "
        f"spans {pos_lo:.2f}-{pos_hi:.2f} mm"
    )
    ctx.progress(f"[ingest] {len(paths)} sections from {ctx.image_folder}")
    return state


def apply_host_inputs(state: StackState, spec: JobSpec) -> None:
    """Write the host's answers for the tasks that are switched off.

    Order arrives as a list of filenames, positions as a filename -> mm
    mapping, angles as ``{"pitch": deg, "yaw": deg}``, damage as a filename
    -> note mapping, and transforms as filename -> stored transform dictionaries.
    Anything the host
    supplies for a task that IS on is applied too — it is a starting point,
    not a constraint. Two exceptions are constraints: ``damaged`` flags the
    agent cannot clear, and ``locked`` sections (a list of filenames) whose
    flip, rotation and transform the agent cannot change; a locked section
    without a supplied transform carries the ``"host"`` identity
    (:func:`langslice.linear.toolbox.host_transform`), because its snapshot
    is already aligned.
    """
    inputs = spec.inputs or {}

    order = inputs.get("order") or []
    if order:
        known = [state.by_id(str(name)) for name in order]
        missing = [str(name) for name, hit in zip(order, known, strict=True) if hit is None]
        if missing:
            raise ValueError(f"inputs.order names sections that are not here: {missing}")
        tail = [s for s in state.in_order() if s.id not in {str(name) for name in order}]
        for index, record in enumerate([r for r in known if r is not None] + tail):
            record.index_corrected = index
        state.notes.append(f"inputs: order set by the host ({len(order)} sections)")

    positions = inputs.get("positions") or {}
    if positions:
        applied = 0
        for name, value in positions.items():
            record = state.by_id(str(name))
            if record is None:
                raise ValueError(f"inputs.positions names an unknown section: {name!r}")
            record.position_mm = float(value)
            applied += 1
        state.notes.append(f"inputs: {applied} position(s) set by the host")

    angles = inputs.get("angles") or {}
    if angles:
        state.cutting_angles_deg = {
            "pitch": float(angles.get("pitch", 0.0)),
            "yaw": float(angles.get("yaw", 0.0)),
        }
        state.notes.append(
            f"inputs: cutting angles set by the host "
            f"(pitch {state.pitch_deg:.2f}, yaw {state.yaw_deg:.2f})"
        )

    transforms = inputs.get("transforms") or {}
    if transforms:
        for name, value in transforms.items():
            record = state.by_id(str(name))
            if record is None:
                raise ValueError(f"inputs.transforms names an unknown section: {name!r}")
            if not isinstance(value, dict):
                raise ValueError(f"inputs.transforms[{name!r}] must be a transform dictionary")
            # Preserve complete historical mappings, including splines. The image
            # correction handoff explicitly refuses unsupported spline inputs.
            record.transform = copy.deepcopy(value)
        state.notes.append(f"inputs: {len(transforms)} transform(s) set by the host")

    damaged = inputs.get("damaged") or {}
    if damaged:
        # Damage is normally the agent's own classification; a host (or a
        # benchmark) may assert it up front so the automatic fits refuse the
        # section and it is aligned by hand.
        for name, note in damaged.items():
            record = state.by_id(str(name))
            if record is None:
                raise ValueError(f"inputs.damaged names an unknown section: {name!r}")
            record.damaged = True
            record.damage_note = str(note or "")
        state.notes.append(f"inputs: {len(damaged)} section(s) marked damaged by the host")

    locked = inputs.get("locked") or []
    if locked:
        from langslice.linear.toolbox import host_transform

        if not isinstance(locked, (list, tuple)):
            raise ValueError("inputs.locked must be a list of section filenames")
        for name in locked:
            record = state.by_id(str(name))
            if record is None:
                raise ValueError(f"inputs.locked names an unknown section: {name!r}")
            if record.transform is None:
                record.transform = host_transform()
        state.notes.append(
            f"inputs: {len(locked)} section(s) locked by the host (in-plane alignment done)"
        )


# --- the session ---------------------------------------------------------


def build_seed_message(state: StackState, ctx: EngineContext) -> types.Content:
    """The stack as ABBA-style strips (:mod:`langslice.linear.opening`), the table."""
    parts: list[types.Part] = opening_parts(state, ctx)
    parts.append(
        types.Part.from_text(
            text=(
                "Status table (corrected index, filename, position, spacing to "
                "the next placed section, flags):\n"
                f"{status_text(state)}\n\n"
                "Recent run notes:\n"
                + ("\n".join(f"- {note}" for note in state.notes[-12:]) or "- (none)")
            )
        )
    )
    return types.Content(role="user", parts=parts)


async def run_session(
    state: StackState,
    ctx: EngineContext,
    spec: JobSpec,
    box: ToolBox,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    on_event: LiveCallback | None = None,
) -> tuple[int, int]:
    """Drive the one stack session; return ``(tool_calls, turns)``."""
    pos_lo, pos_hi = ctx.position_range
    agent = build_agent(
        model=ctx.model,
        name="linear_stack",
        instruction=build_job_statement(
            spec,
            state,
            tool_names=box.names,
            species=ctx.species,
            pos_lo=pos_lo,
            pos_hi=pos_hi,
            axis_ends=ctx.axis_ends,
            **display_facts(ctx, state),
        ),
        tools=box.tools,
        reasoning=spec.reasoning,
    )
    sink: list[str] = []
    outcome = await run_agent_session(
        agent=agent,
        seed_message=build_seed_message(state, ctx),
        done=lambda: state.submitted,
        nudge_no_tool=_NUDGE_NO_TOOL,
        nudge_continue=_NUDGE_CONTINUE,
        max_iterations=max_iterations,
        run_label=_RUN_LABEL,
        debrief=DEBRIEF_PROMPT if spec.debrief else None,
        debrief_sink=sink,
        progress=ctx.progress,
        max_input_tokens=spec.max_input_tokens,
        max_quota_percent=spec.max_quota_percent,
        tool_media_delivered=box.mark_placement_views_delivered,
        on_event=on_event,
    )
    if sink and sink[0]:
        state.debrief = sink[0]
        save_checkpoint(state, ctx.checkpoint_path)
    return outcome


# --- results -------------------------------------------------------------


def emit_results(state: StackState, ctx: EngineContext) -> StackState:
    """Write the results JSON — the same shape as the checkpoint."""
    os.makedirs(os.path.dirname(os.path.abspath(ctx.results_path)), exist_ok=True)
    with open(ctx.results_path, "w", encoding="utf-8") as handle:
        json.dump(state.to_dict(), handle, indent=2)
    ctx.progress(f"[emit] results -> {ctx.results_path}")
    return state


async def run(
    spec: JobSpec,
    *,
    emit: Callable[[str], None] | None = None,
    atlas_loader: Callable[[str], Any] | None = None,
    on_write: Callable[[StackState], None] | None = None,
    on_event: LiveCallback | None = None,
) -> StackState:
    """Run one linear job and return the final stack state.

    *on_write* is a host adapter that wants to watch the run live (the ABBA
    mirror, :mod:`langslice.integrations.abba_linear`): it is called once
    with the state as ingested, then again after every checkpoint the
    session writes (:func:`langslice.linear.checkpoint.observe_checkpoints`),
    through to the final result.
    """
    ctx = build_context(spec, emit=emit, atlas_loader=atlas_loader)

    state = load_checkpoint(ctx.checkpoint_path) if spec.resume else None
    if state is not None:
        ctx.progress(f"[ingest] resuming from {ctx.checkpoint_path}")
        state.spec = spec.to_dict()
        state.submitted = False
    else:
        state = ingest(spec, ctx)
        apply_host_inputs(state, spec)
    save_checkpoint(state, ctx.checkpoint_path)
    if on_write is not None:
        on_write(state)

    box = build_tools(state, ctx, spec, on_event=on_event)
    watch = observe_checkpoints(on_write) if on_write is not None else contextlib.nullcontext()
    with watch:
        try:
            tool_calls, turns = await run_session(state, ctx, spec, box, on_event=on_event)
        finally:
            # Background image corrections finish and are recorded even when
            # the session ends without a submit.
            if box.settle_image_corrections(state):
                save_checkpoint(state, ctx.checkpoint_path)
        ctx.progress(
            f"[session] {tool_calls} tool call(s) over {turns} turn(s); "
            + ("submitted" if state.submitted else "no submission")
        )
        if not state.submitted:
            state.notes.append(f"session: no submission within {turns} turns")

        return emit_results(state, ctx)

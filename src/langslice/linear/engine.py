"""The run: ingest, one agent session, the post pass, results.

``run(spec)`` is the whole of ``langslice linear``. There is no node graph: one
state, one toolbox, one job statement, and a session that ends at ``submit`` or
the turn budget. Every write tool checkpoints, so a run that dies mid-way
resumes from the checkpoint with the state it had — the agent is re-seeded, not
replayed.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from google.genai import types
from PIL import Image

from langslice.atlas.core import get_position_range_mm, load_atlas
from langslice.linear.checkpoint import (
    default_checkpoint_path,
    load_checkpoint,
    save_checkpoint,
)
from langslice.linear.discovery import discover_slices
from langslice.linear.prompt import build_job_statement
from langslice.linear.render import stack_image_parts, status_text
from langslice.linear.session import (
    DEFAULT_MAX_ITERATIONS,
    build_agent,
    run_agent_session,
)
from langslice.linear.spec import JobSpec
from langslice.linear.state import SliceState, StackState, add_caveat
from langslice.linear.toolbox import ToolBox, build_tools
from langslice.linear.transform import fit_silhouette, run_align_session
from langslice.space import Plane

logger = logging.getLogger(__name__)

RESULTS_FILENAME = "linear_results.json"

_RUN_LABEL = "linear_stack"

_NUDGE_NO_TOOL = (
    "You did not call a tool. Continue with the tools rather than in prose; "
    "call `submit` when the job is done."
)
_NUDGE_CONTINUE = "Continue; call `submit` when the job is done."


def _log_progress(message: str) -> None:
    logger.info(message)


@dataclass
class EngineContext:
    """Everything a tool needs that is not stack state."""

    spec: JobSpec
    image_folder: str
    checkpoint_path: str
    results_path: str
    model: str
    emit: Callable[[str], None] = _log_progress
    #: Atlas accessor, injectable so tests (and offline hosts) can supply one.
    atlas_loader: Callable[[str], Any] = load_atlas
    #: Rendered sections, keyed ``(id, flip, rotation, long_edge, preprocess,
    #: frame)``. The user's files never change during a run, and a correction
    #: or a different size is a different key. Cached images are shared —
    #: callers read them, never mutate them.
    render_cache: dict[tuple[str, bool, int, int, str, bool], Image.Image] = field(
        default_factory=dict, repr=False
    )
    _atlas: Any = field(default=None, repr=False)
    _range: tuple[float, float] | None = field(default=None, repr=False)

    def progress(self, message: str) -> None:
        self.emit(message)
        logger.info(message)

    def image_path(self, slice_id: str) -> str:
        """Absolute path of a section image. Ids are folder-relative names."""
        return os.path.join(self.image_folder, slice_id)

    @property
    def atlas(self) -> Any:
        if self._atlas is None:
            self._atlas = self.atlas_loader(self.spec.atlas)
        return self._atlas

    @property
    def position_range(self) -> tuple[float, float]:
        if self._range is None:
            self._range = get_position_range_mm(
                self.atlas, plane=cast(Plane, self.spec.plane)
            )
        return self._range

    @property
    def species(self) -> str:
        metadata = getattr(self.atlas, "metadata", None)
        return str((metadata or {}).get("species", "mouse"))


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
        emit=emit or _log_progress,
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
    mapping, angles as ``{"pitch": deg, "yaw": deg}``. Anything the host
    supplies for a task that IS on is applied too — it is a starting point,
    not a constraint.
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


# --- the session ---------------------------------------------------------


def build_seed_message(state: StackState, ctx: EngineContext) -> types.Content:
    """Every section as its own labelled image, plus the status table."""
    parts: list[types.Part] = stack_image_parts(state, ctx)
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
        ),
        tools=box.tools,
    )
    return await run_agent_session(
        agent=agent,
        seed_message=build_seed_message(state, ctx),
        done=lambda: state.submitted,
        nudge_no_tool=_NUDGE_NO_TOOL,
        nudge_continue=_NUDGE_CONTINUE,
        max_iterations=max_iterations,
        run_label=_RUN_LABEL,
    )


# --- the post pass -------------------------------------------------------


async def run_post_pass(state: StackState, ctx: EngineContext, spec: JobSpec) -> None:
    """Fill in the transforms the agent did not, when subagents are off.

    Intact sections take the closed-form silhouette fit; damaged ones take the
    interactive session, one at a time.

    # ponytail: sequential; a semaphore would fan out if damaged-heavy stacks
    # ever cost real wall-clock.
    """
    if not spec.has("transform") or spec.transform.subagents:
        return
    for record in state.in_order():
        if record.transform is not None or record.position_mm is None:
            continue
        if record.damaged:
            try:
                outcome = await run_align_session(state, ctx, record, "")
            except Exception as exc:
                logger.warning("post: alignment failed for %s: %s", record.id, exc)
                add_caveat(record, "interactive alignment failed")
                continue
            if outcome["status"] != "ok":
                add_caveat(record, "interactive alignment produced no transform")
                continue
            record.transform = {
                "kind": "interactive",
                "params": outcome["matrix_params"],
                "note": outcome["note"],
            }
            ctx.progress(f"[post] {record.id}: interactive transform recorded")
            continue

        outcome = fit_silhouette(state, ctx, record)
        outcome.pop("panel", None)
        if outcome["status"] != "ok":
            add_caveat(record, "affine fit failed")
            continue
        record.transform = {
            "kind": "silhouette",
            "params": outcome["params"],
            "iou": outcome["iou"],
        }
        ctx.progress(
            f"[post] {record.id}: silhouette affine (overlap {outcome['iou']:.2f})"
        )
    save_checkpoint(state, ctx.checkpoint_path)


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
) -> StackState:
    """Run one linear job and return the final stack state."""
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

    box = build_tools(state, ctx, spec)
    tool_calls, turns = await run_session(state, ctx, spec, box)
    ctx.progress(
        f"[session] {tool_calls} tool call(s) over {turns} turn(s); "
        + ("submitted" if state.submitted else "no submission")
    )
    if not state.submitted:
        state.notes.append(f"session: no submission within {turns} turns")

    await run_post_pass(state, ctx, spec)
    return emit_results(state, ctx)

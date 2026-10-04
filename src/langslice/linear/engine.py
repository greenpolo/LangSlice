"""The run: open the job, one agent session, results.

``run(spec)`` is the whole of ``langslice linear``. There is no node graph: one
job (:class:`langslice.linear.job.Job`), one toolbox over it, one job
statement, and a session that ends at ``submit`` or the turn budget. Every
write tool checkpoints, so a run that dies mid-way resumes from the checkpoint
with the state it had (and its undo history) — the agent is re-seeded, not
replayed.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from google.genai import types

from langslice.adk.media import opening_parts
from langslice.atlas.core import load_atlas
from langslice.linear.checkpoint import default_checkpoint_path
from langslice.linear.job import Job

# Re-exported for the sibling SliceBench adapters, which import them from here;
# they live in the job layer (langslice.linear.job) since layered-core phase 2.
from langslice.linear.job import apply_host_inputs as apply_host_inputs  # noqa: E402
from langslice.linear.job import ingest as ingest  # noqa: E402
from langslice.linear.live import LiveCallback
from langslice.linear.prompt import build_job_statement, display_facts
from langslice.linear.render import status_text
from langslice.linear.session import (
    DEFAULT_MAX_ITERATIONS,
    build_agent,
    run_agent_session,
)
from langslice.linear.spec import JobSpec
from langslice.linear.state import StackState
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
    the driver needs — where the job's files go (the paths the
    :class:`~langslice.linear.job.Job` is opened at; the job owns them from
    then on), the model, and the cache of encoded message images
    (:mod:`langslice.adk.media`)."""

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
            max_resolution=box.max_view_edge,
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
        box.job.checkpoint()
    return outcome


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
    with the state as opened, then again after every checkpoint the job
    writes (and every reload of a state file changed on disk,
    :meth:`langslice.linear.job.Job.observe`), through to the final result.
    """
    ctx = build_context(spec, emit=emit, atlas_loader=atlas_loader)
    job = Job.open(spec, ctx, checkpoint_path=ctx.checkpoint_path,
                   results_path=ctx.results_path)
    state = job.state
    if on_write is not None:
        on_write(state)

    box = build_tools(state, ctx, spec, job=job, on_event=on_event)
    watch = job.observe(on_write) if on_write is not None else contextlib.nullcontext()
    with watch:
        try:
            tool_calls, turns = await run_session(state, ctx, spec, box, on_event=on_event)
        finally:
            # Background image corrections finish and are recorded even when
            # the session ends without a submit.
            job.settle_image_corrections()
        ctx.progress(
            f"[session] {tool_calls} tool call(s) over {turns} turn(s); "
            + ("submitted" if state.submitted else "no submission")
        )
        if not state.submitted:
            state.notes.append(f"session: no submission within {turns} turns")

        return job.emit_results(ctx.progress)

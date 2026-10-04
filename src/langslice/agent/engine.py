"""The run: open the job, one agent session, results.

``run(spec)`` is the whole of ``langslice linear``. There is no node graph: one
job (:class:`langslice.job.job.Job`), one toolbox over it, one job
statement, and a session that ends at ``submit`` or the turn budget. Every
write tool checkpoints, so a run that dies mid-way resumes from the checkpoint
with the state it had (and its undo history) — the agent is re-seeded, not
replayed.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.genai import types

from langslice.agent.live import LiveCallback
from langslice.agent.prompt import build_job_statement, display_facts
from langslice.agent.session import (
    DEFAULT_MAX_ITERATIONS,
    build_agent,
    run_agent_session,
)
from langslice.core.atlas.core import load_atlas
from langslice.core.spec import JobSpec
from langslice.core.state import StackState
from langslice.core.status import status_text
from langslice.core.workspace import log_progress
from langslice.doors.card import write_card
from langslice.doors.jobs import JobContext
from langslice.doors.tools.media import opening_parts, packaged_tools
from langslice.doors.tools.toolbox import ToolBox, build_tools
from langslice.job.job import Job

# Re-exported for the sibling SliceBench adapters, which import them from here;
# they live in the job layer (langslice.job.job) since layered-core phase 2.
from langslice.job.job import apply_host_inputs as apply_host_inputs  # noqa: E402
from langslice.job.job import ingest as ingest  # noqa: E402
from langslice.job.layout import JobLayout, locate_job_folder

logger = logging.getLogger(__name__)

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
class EngineContext(JobContext):
    """The agent driver's context: the core :class:`Workspace` and where the
    job's files go (:class:`~langslice.doors.jobs.JobContext`: the job folder,
    ``<images>/langslice``, and the results path the
    :class:`~langslice.job.job.Job` is opened at; the job owns them from
    then on), plus what only the driver needs: the model."""

    model: str


def build_context(
    spec: JobSpec,
    *,
    emit: Callable[[str], None] | None = None,
    atlas_loader: Callable[[str], Any] | None = None,
    job_folder: str | os.PathLike[str] | None = None,
) -> EngineContext:
    """Assemble an :class:`EngineContext` from a job spec.

    The job folder is *job_folder* when given (a saved job's), else
    :func:`~langslice.job.layout.locate_job_folder` (``spec.job_dir``, next
    to the images, or ``~/.langslice/jobs/<id>/`` for a read-only folder).
    """
    from langslice.providers.openai_oauth import DEFAULT_REVIEW_MODEL

    folder = os.path.abspath(spec.image_folder)
    if job_folder is None:
        job_folder, _fallback = locate_job_folder(folder, spec.job_dir, emit=emit or log_progress)
    job_folder = Path(job_folder)
    return EngineContext(
        spec=spec,
        image_folder=folder,
        job_folder=str(job_folder),
        results_path=spec.out or str(JobLayout(job_folder).results_file),
        model=spec.model or DEFAULT_REVIEW_MODEL,
        emit=emit or log_progress,
        atlas_loader=atlas_loader or load_atlas,
    )


# --- the session ---------------------------------------------------------


def build_seed_message(state: StackState, ctx: EngineContext) -> types.Content:
    """The stack as ABBA-style strips (:mod:`langslice.core.opening`), the table."""
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


def save_opening(box: ToolBox, seed: types.Content) -> None:
    """Save the opening's pictures in the job folder, as the bytes sent."""
    pictures = [part.inline_data.data for part in seed.parts or []
                if part.inline_data is not None and part.inline_data.data]
    try:
        box.job.views.save(tool="opening", pictures=[(data, None) for data in pictures])
    except Exception:  # saving must never break the run
        logger.warning("Could not queue the opening pictures", exc_info=True)


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
        # The tools return plain pictures; ADK takes them as message parts.
        tools=packaged_tools(box.tools),
        reasoning=spec.reasoning,
    )
    sink: list[str] = []
    seed = build_seed_message(state, ctx)
    save_opening(box, seed)
    outcome = await run_agent_session(
        agent=agent,
        seed_message=seed,
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
    mirror, :mod:`langslice.hosts.integrations.abba_linear`): it is called once
    with the state as opened, then again after every checkpoint the job
    writes (and every reload of a state file changed on disk,
    :meth:`langslice.job.job.Job.observe`), through to the final result.
    """
    ctx = build_context(spec, emit=emit, atlas_loader=atlas_loader)
    job = Job.open(spec, ctx, folder=ctx.job_folder, results_path=ctx.results_path)
    write_card(job.layout)
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

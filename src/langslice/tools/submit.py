"""Ending the run: the submit gates, the final write, the ``submit`` tool."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langslice.core.refusal import RefusalCode, Refused
from langslice.core.stack.stack import break_ids, normalize_to_atlas_order
from langslice.tools.exports import Exported
from langslice.tools.refusal import unknown_sections
from langslice.tools.run import as_list

if TYPE_CHECKING:
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job
    from langslice.tools.run import Run

logger = logging.getLogger(__name__)

#: The note the write adds when the corrected order is reversed to run the atlas way.
REVERSED_NOTE = "submit: corrected order reversed to run the atlas way"


@dataclass(frozen=True)
class Submitted:
    """What :func:`submit` wrote."""

    summary: str
    #: The notes added from the call (cleaned), the summary line aside.
    notes: list[str] = field(default_factory=list)
    #: The interval breaks written: the filenames of the sections after each gap.
    interval_breaks: list[str] = field(default_factory=list)
    #: Whether the corrected order was reversed to run the atlas way.
    reversed: bool = False
    #: The stack's exports written after the step
    #: (``tools.exports.export_stack``; no per-section maps), when the call had
    #: the workspace; None otherwise or when writing failed.
    exported: Exported | None = None


def clean_breaks(job: Job, values: Any) -> list[int]:
    """The interval breaks *values* name (a list of filenames from a model: a
    trust boundary), as the corrected indices the state keeps.
    ``UNKNOWN_SLICE_IDS`` for a name that is no section's; ``BAD_ARGS`` for
    anything but a list."""
    if values is None:
        return []
    if not isinstance(values, (list, tuple)):
        raise Refused(
            RefusalCode.BAD_ARGS,
            message="interval_breaks is a list of filenames.",
        )
    breaks: list[int] = []
    unknown: list[str] = []
    for raw in values:
        record = job.state.resolve(raw)
        if record is None:
            unknown.append(str(raw))
        else:
            breaks.append(record.index_corrected)
    if unknown:
        raise unknown_sections(job.state, unknown)
    return breaks


def submit(
    job: Job,
    *,
    summary: str = "",
    notes: Sequence[Any] = (),
    interval_breaks: Sequence[Any] = (),
    traces: bool = False,
    workspace: Workspace | None = None,
) -> Submitted:
    """Check the job's submit gates, then end the run: ONE undo step.

    First every piece of the job's background work is waited for
    (:meth:`langslice.job.background.BackgroundWork.wait_all`: a packaged
    trace lands and its fit applies, each its own undo step).
    The gates, in order: the job's (:meth:`~langslice.job.job.Job.submit_errors`:
    positions, interval breaks, transforms, and, when the host requires one
    (``NonlinearSpec.require_deformation``), a deformation on every section).
    A refusal is :class:`Refused` with the gate's payload, nothing written.
    With *traces* (the image model is part of the run), image-model traces
    still running are waited for and recorded first; tracing is the agent's
    choice, so no section needs one.

    The write: the interval breaks (*interval_breaks*: the filenames of the
    sections after each gap, :func:`clean_breaks`; kept as corrected indices,
    sorted, unique), the corrected order reversed when it runs against the
    atlas (a convention, not an inference: noted, never asked of the agent),
    the cleaned *notes* and a ``submit: <summary>`` note, ``submitted``. Then
    every queued picture is written to the job folder before this returns, and,
    with *workspace*, the stack's exports
    (:func:`langslice.tools.exports.export_stack`: QuickNII, VisuAlign,
    ``registration.json``, plugin layers; a failure there is logged, the submit
    stands). The per-section maps are not written: the ``export_maps`` verb
    writes them on demand.
    """
    breaks = clean_breaks(job, interval_breaks)
    job.background.wait_all()
    if traces:
        job.settle_border_traces()
    refusal = job.submit_errors(breaks)
    if refusal is not None:
        raise refusal

    state = job.state
    before = job.snapshot()
    state.interval_breaks = sorted(set(breaks))
    # Direction is a convention, not an inference: a posterior-first stack
    # is emitted in atlas order without the agent being told about it.
    reversed_order = bool(normalize_to_atlas_order(state))
    if reversed_order:
        state.notes.append(REVERSED_NOTE)
    # Model output is a trust boundary: a malformed submission must not
    # take the run down.
    clean_notes = (
        [str(item).strip() for item in notes if str(item).strip()]
        if isinstance(notes, (list, tuple))
        else []
    )
    state.notes.extend(clean_notes)
    summary_text = str(summary or "").strip()
    if summary_text:
        state.notes.append(f"submit: {summary_text}")
    state.submitted = True
    job.commit(before)
    job.views.flush()
    exported = None
    if workspace is not None and job.persist:
        from langslice.tools.exports import export_stack

        try:
            exported = export_stack(job, workspace)
        except Exception:  # the derived files must never undo a submit
            logger.warning(
                "Could not write the exports at submit", exc_info=True
            )
    return Submitted(
        summary=summary_text,
        notes=clean_notes,
        interval_breaks=break_ids(state),
        reversed=reversed_order,
        exported=exported,
    )


def submit_tool(
    run: Run,
    summary: str,
    notes: list[str],
    interval_breaks: list[str],
    *,
    tool_context: Any = None,
) -> dict[str, Any]:
    """Run the ``submit`` tool: end the run and answer with the whole table.

    Parameters
    ----------
    run
        The run the tool is called in.
    summary, notes, interval_breaks, tool_context
        The declared arguments as the model sent them
        (``tools/descriptions/bookkeeping.py``).

    Returns
    -------
    dict
        ``status`` and every section's row, the cutting angles and the
        interval breaks.
    """
    done = submit(
        run.job,
        summary=summary,
        notes=as_list(notes),
        interval_breaks=as_list(interval_breaks),
        traces=run.traces_on,
        workspace=run.workspace,
    )
    run.box.submission.update(
        {
            "summary": done.summary,
            "notes": done.notes,
            "interval_breaks": done.interval_breaks,
        }
    )
    if tool_context is not None:
        tool_context.actions.escalate = True
    return {"status": "ok", **run.rows()}

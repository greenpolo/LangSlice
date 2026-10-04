"""Ending the run: the job's submit gates, then the final write."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langslice.linear.state import normalize_to_atlas_order
from langslice.ops.refusal import Refused

if TYPE_CHECKING:
    from langslice.linear.job import Job
    from langslice.linear.workspace import Workspace

#: The note the write adds when the corrected order is reversed to run the atlas way.
REVERSED_NOTE = "submit: corrected order reversed to run the atlas way"


@dataclass(frozen=True)
class Submitted:
    """What :func:`submit` wrote."""

    summary: str
    #: The notes added from the call (cleaned), the summary line aside.
    notes: list[str] = field(default_factory=list)
    interval_breaks: list[int] = field(default_factory=list)
    #: Whether the corrected order was reversed to run the atlas way.
    reversed: bool = False


def clean_breaks(values: Any) -> list[int]:
    """The whole numbers in *values* (a list from a model: a trust boundary)."""
    breaks: list[int] = []
    for raw in values if isinstance(values, (list, tuple)) else []:
        try:
            breaks.append(int(raw))
        except (TypeError, ValueError):
            continue
    return breaks


def submit(
    job: Job,
    *,
    summary: str = "",
    notes: Sequence[Any] = (),
    interval_breaks: Sequence[Any] = (),
    traces: bool = False,
    workspace: Workspace | None = None,
    gate: Callable[[], Mapping[str, Any] | None] | None = None,
) -> Submitted:
    """Check the job's submit gates, then end the run: ONE undo step.

    The gates, in order: the job's (:meth:`~langslice.linear.job.Job.submit_errors`:
    positions, order, interval breaks, transforms, deformations), then, with
    *traces* (the image model is part of the run; *workspace*, the run's,
    reads each section's current geometry) and the nonlinear task on, every
    section's completed image correction at its current placement
    (:meth:`~langslice.linear.job.Job.missing_image_corrections`), which is
    reported before a missing deformation since a deformation may be fitted
    to its section's trace. *gate*, when given, runs last, before
    anything is written: a door's own check (the tool door's "view_stack
    first"); a payload it returns refuses the call. A refusal is
    :class:`Refused` with the gate's payload, nothing written.

    The write: the interval breaks (sorted, unique), the corrected order
    reversed when it runs against the atlas (a convention, not an inference:
    noted, never asked of the agent), the cleaned *notes* and a
    ``submit: <summary>`` note, ``submitted``. Then every queued picture is
    written to the job folder before this returns.
    """
    breaks = clean_breaks(interval_breaks)
    refusal = job.submit_errors(breaks)
    # With the image model, missing traces are reported before missing
    # deformations: a deformation may be fitted to its section's trace.
    if traces and workspace is None:
        raise ValueError("submit: checking the image corrections needs the workspace")
    if refusal is not None and not (traces and refusal.get("error") == "MISSING_DEFORMATIONS"):
        raise Refused.of(refusal)
    if traces and job.spec.has("nonlinear"):
        assert workspace is not None
        pending = job.missing_image_corrections(workspace)
        if pending:
            raise Refused(
                "MISSING_IMAGE_CORRECTIONS", status="refused", sections=pending,
                message="Each section requires a completed image correction at its current "
                "linear placement. This checks completion and geometry, "
                "not anatomical quality.",
            )
    if refusal is not None:
        raise Refused.of(refusal)
    if gate is not None:
        held = gate()
        if held is not None:
            raise Refused.of(held)

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
    return Submitted(summary=summary_text, notes=clean_notes,
                     interval_breaks=list(state.interval_breaks), reversed=reversed_order)

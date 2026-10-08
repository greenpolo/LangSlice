"""Ending the run: the job's submit gates, then the final write."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langslice.core import deformation
from langslice.core.state import break_ids, normalize_to_atlas_order
from langslice.ops.exports import Exported
from langslice.ops.refusal import Refused, unknown_sections

if TYPE_CHECKING:
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job

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
    #: The sections left without a deformation, ``{id: reason}`` (``left_linear``).
    left_linear: dict[str, str] = field(default_factory=dict)
    #: The maps and exports written after the step (``ops.exports``), when
    #: the call had the workspace; None otherwise or when writing failed.
    exported: Exported | None = None


def clean_breaks(job: Job, values: Any) -> list[int]:
    """The interval breaks *values* name (a list of filenames from a model: a
    trust boundary), as the corrected indices the state keeps.
    ``UNKNOWN_SLICE_IDS`` for a name that is no section's; ``BAD_ARGS`` for
    anything but a list."""
    if values is None:
        return []
    if not isinstance(values, (list, tuple)):
        raise Refused("BAD_ARGS", message="interval_breaks is a list of filenames.")
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


def clean_left_linear(job: Job, entries: Any) -> dict[str, str]:
    """``{id: reason}`` from ``[{id, reason}]`` (a list from a model: a trust
    boundary). Refused: ``BAD_ARGS`` (not a list of ``{id, reason}`` with a
    reason, a section named twice, or any entry in a run without Nonlinear),
    ``UNKNOWN_SLICE_IDS``, ``DEFORMATION_REQUIRED`` (the host requires a
    deformation on every section), ``HAS_DEFORMATION`` (a listed section
    carries one at its current placement: undo it first),
    ``INVALID_LINEAR_PLACEMENT`` (a listed section has no position or no
    transform, so it has no linear placement to stand)."""
    if entries is None or (isinstance(entries, (list, tuple)) and not entries):
        return {}
    if not isinstance(entries, (list, tuple)) or not all(
            isinstance(entry, dict) and set(entry) <= {"id", "reason"} for entry in entries):
        raise Refused("BAD_ARGS", message="left_linear is a list of {id, reason}.")
    if not job.spec.has("nonlinear"):
        raise Refused("BAD_ARGS", message="left_linear is for runs with the Nonlinear task "
                      "on; this run fits no deformation.")
    if job.spec.nonlinear.require_deformation:
        raise Refused("DEFORMATION_REQUIRED", message="The user requires a deformation on "
                      "every section, so no section can be left linear.")
    left: dict[str, str] = {}
    unknown: list[str] = []
    fitted: list[str] = []
    for entry in entries:
        record = job.state.resolve(entry.get("id"))
        reason = str(entry.get("reason") or "").strip()
        if record is None:
            unknown.append(str(entry.get("id")))
            continue
        if not reason:
            raise Refused("BAD_ARGS", id=record.id,
                          message="Each left_linear entry needs the reason its linear "
                          "placement stands.")
        if record.id in left:
            raise Refused("BAD_ARGS", id=record.id, message="A section is named twice.")
        left[record.id] = reason
        if job.deformations.current(job.state, record) is not None:
            fitted.append(record.id)
    if unknown:
        raise unknown_sections(job.state, unknown)
    unplaced = [name for name in left if (held := job.state.by_id(name)) is not None
                and (held.position_mm is None or held.transform is None)]
    if unplaced:
        from langslice.core.handoff import NO_TRANSFORM_LINEAR_OFF

        raise Refused("INVALID_LINEAR_PLACEMENT", ids=unplaced,
                      message="A section left linear needs a position and a transform."
                      + ("" if job.spec.has("transform") else " " + NO_TRANSFORM_LINEAR_OFF))
    if fitted:
        raise Refused("HAS_DEFORMATION", ids=fitted,
                      message="These sections carry a deformation at their current "
                      "placement; undo it to leave them linear, or leave them off the list.")
    return left


def submit(
    job: Job,
    *,
    summary: str = "",
    notes: Sequence[Any] = (),
    interval_breaks: Sequence[Any] = (),
    left_linear: Any = (),
    traces: bool = False,
    workspace: Workspace | None = None,
    gate: Callable[[], Mapping[str, Any] | None] | None = None,
) -> Submitted:
    """Check the job's submit gates, then end the run: ONE undo step.

    First every piece of the job's background work is waited for
    (:meth:`langslice.job.background.BackgroundWork.wait_all`: a packaged
    trace lands and its fit applies, each its own undo step). *left_linear*
    (``[{id, reason}]``, runs with Nonlinear on) names the sections left
    without a deformation, each with the reason its linear placement stands
    (:func:`clean_left_linear`; refused when the host requires a deformation
    on every section, ``NonlinearSpec.require_deformation``).
    The gates, in order: the job's (:meth:`~langslice.job.job.Job.submit_errors`:
    positions, interval breaks, transforms, deformations, *left_linear*
    counted as covered), then
    *gate*, when given, before anything is written: a door's own check (the
    tool door's "look at the whole stack in mode positioning first"); a
    payload it returns refuses the call.
    A refusal is :class:`Refused` with the gate's payload, nothing written.
    With *traces* (the image model is part of the run), image-model traces
    still running are waited for and recorded first; tracing is the agent's
    choice, so no section needs one.

    The write: each *left_linear* section's record that its linear placement
    stands (the record ``keep_linear`` writes, its reason), the interval
    breaks (*interval_breaks*: the filenames of the sections after each gap,
    :func:`clean_breaks`; kept as corrected indices, sorted, unique), the corrected order
    reversed when it runs against the atlas (a convention, not an inference:
    noted, never asked of the agent), the cleaned *notes* and a
    ``submit: <summary>`` note, ``submitted``. Then every queued picture is
    written to the job folder before this returns, and, with *workspace*,
    every placed section's maps and the stack's exports
    (:func:`langslice.ops.exports.export_maps`; a failure there is logged,
    the submit stands).
    """
    breaks = clean_breaks(job, interval_breaks)
    job.background.wait_all()
    if traces:
        job.settle_image_corrections()
    left = clean_left_linear(job, left_linear)
    refusal = job.submit_errors(breaks, left)
    if refusal is not None:
        raise Refused.of(refusal)
    if gate is not None:
        held = gate()
        if held is not None:
            raise Refused.of(held)

    state = job.state
    before = job.snapshot()
    for section, reason in left.items():
        record = state.by_id(section)
        if record is not None:
            record.deformation = {"keep_linear": reason,
                                  "linear_key": deformation.linear_key(state, record)}
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
        from langslice.ops.exports import export_maps

        try:
            exported = export_maps(job, workspace)
        except Exception:  # the derived files must never undo a submit
            logger.warning("Could not write the maps and exports at submit", exc_info=True)
    return Submitted(summary=summary_text, notes=clean_notes,
                     interval_breaks=break_ids(state), reversed=reversed_order,
                     exported=exported, left_linear=left)

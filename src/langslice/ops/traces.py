"""The image model's border trace of a section: started in the background, recorded.

:func:`trace_borders` takes the image model as an argument
(:class:`langslice.providers.registry.ImageModel`, resolved by the door), so
this module never chooses or imports a provider. The section's current
geometry is the core's (:func:`langslice.core.handoff.correction_fingerprint`)
and the edit is prepared by ``registration_tool.start_correction``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langslice.core import handoff
from langslice.core.damage import exclusions
from langslice.core.nonlinear import registration_tool
from langslice.ops.inputs import STALE_INPUT
from langslice.ops.refusal import Refused

if TYPE_CHECKING:
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job
    from langslice.providers.registry import ImageModel


@dataclass(frozen=True)
class TraceStarted:
    """What :func:`trace_borders` did for one section."""

    id: str
    #: The section's image-correction record as stored (paths relative to
    #: the job folder): ``status``, ``cached``, ``attempt``, the geometry
    #: fingerprint, ``error`` / ``message`` when it failed...
    record: dict[str, Any]
    #: True when an image call was started in the background now.
    started: bool = False
    #: True when this section's call at this geometry was already running
    #: (nothing started, nothing written).
    running: bool = False


#: Prepares one section's edit: ``(state, workspace, section id, calls_dir)``
#: -> ``(record, call or None)`` (``registration_tool.start_correction`` or
#: ``start_atlas_correction`` with the call's own arguments bound).
Prepare = Callable[[Any, Any, str, Any], tuple[dict[str, Any], Any]]

_STALE_MESSAGE = ("The section's placement changed while the image call was prepared; "
                  "nothing was started or written for it.")


@dataclass(frozen=True)
class _Prepared:
    """One section's edit prepared outside the lock, or why there is none."""

    id: str
    result: dict[str, Any] = field(default_factory=dict)
    call: Any = None
    #: The call at this geometry is already running (nothing prepared).
    running: bool = False
    #: ``(code, facts)`` when the section cannot be traced.
    problem: tuple[str, dict[str, Any]] | None = None


def _prepare(job: Job, workspace: Workspace, ref: object, prepare: Prepare) -> _Prepared:
    """Resolve *ref*, apply the job's Nonlinear refusal and prepare its edit
    (outside the job's write lock)."""
    record = job.state.resolve(ref)
    if record is None:
        return _Prepared(id=str(ref), problem=("UNKNOWN_SLICE_IDS", {"unknown": [str(ref)]}))
    refusal = job.nonlinear_refusal(record.id)
    if refusal is not None:
        return _Prepared(id=record.id, problem=(refusal[0], {"message": refusal[1]}))
    try:
        current = handoff.correction_fingerprint(job.state, workspace, record.id)
        if job.image_job_running(record.id, current):
            return _Prepared(id=record.id, running=True)
        result, call = prepare(job.state, workspace, record.id,
                               job.layout.image_correction_dir(record.id))
    except ValueError as exc:
        return _Prepared(id=record.id, problem=("INVALID_LINEAR_PLACEMENT",
                                                {"message": str(exc)}))
    except OSError as exc:
        return _Prepared(id=record.id, problem=("IMAGE_CORRECTION_IO_ERROR",
                                                {"message": str(exc)}))
    return _Prepared(id=record.id, result=result, call=call)


def _apply(job: Job, workspace: Workspace, prepared: _Prepared, workers: int,
           ) -> tuple[dict[str, Any], bool] | None:
    """Under the job's write lock: start the prepared call and set the
    section's record. ``(record as stored, changed)``, or None when the
    section's geometry changed since it was prepared (nothing started or set)."""
    now = job.state.by_id(prepared.id)
    fingerprint = prepared.result["geometry_fingerprint"]
    if now is None or handoff.correction_fingerprint(
            job.state, workspace, prepared.id) != fingerprint:
        return None
    if prepared.call is not None:
        job.start_image_job(prepared.id, fingerprint, prepared.call, workers=workers)
    portable = job.portable(prepared.result)
    changed = portable != now.image_correction
    if changed:
        now.image_correction = portable
    return portable, changed


def trace_borders(
    job: Job,
    workspace: Workspace,
    ref: object,
    *,
    image_model: ImageModel,
    prompt: str = "",
    restrict_to: tuple[str, ...] = (),
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
    workers: int = registration_tool.MAX_CONCURRENT_IMAGE_CALLS,
) -> TraceStarted:
    """Start one section's image correction in the background, or reuse it.

    The section needs a position and a linear transform. A call already
    running at the section's current geometry is not started again
    (``running``). Otherwise ``registration_tool.start_correction`` prepares
    the edit for *image_model* (the first result at a placement is reused,
    ``cached``) and the call, if any, runs on the job's image executor
    (*workers* at most at once; :meth:`~langslice.job.job.Job.start_image_job`).
    The section's ``image_correction`` record is written as ONE undo step
    when it changed; the call's result lands on it later (submit waits for
    it). *restrict_to* (*include*, its older name, when it is empty) and
    *exclude* choose the regions whose borders the model is shown
    (``registration_tool.shown_labels``); the section's marked regions are
    excluded on their own (:func:`langslice.core.damage.exclusions`). A
    traced ``fit_deformable`` reads them from the record. The edit is
    prepared outside the job's write lock; the start and the write happen
    under it (:meth:`~langslice.job.job.Job.writing`),
    refused ``STALE_INPUT`` when the section's geometry changed meanwhile;
    the result lands only at the geometry it was made for. Refused:
    ``UNKNOWN_SLICE_IDS``, ``KEEPS_HOST_WARP`` / ``NONLINEAR_SKIPPED`` (the
    host kept the section out of Nonlinear, ``Job.nonlinear_refusal``),
    ``INVALID_LINEAR_PLACEMENT`` (the placement cannot be prepared),
    ``IMAGE_CORRECTION_IO_ERROR``, ``STALE_INPUT``.
    """
    chosen = tuple(restrict_to) or tuple(include)

    def prepare(state: Any, ctx: Any, section_id: str, calls_dir: Any,
                ) -> tuple[dict[str, Any], Any]:
        kept, dropped = exclusions(state.by_id(section_id), chosen, exclude)
        return registration_tool.start_correction(
            state, ctx, section_id, prompt=prompt, image_model=image_model,
            calls_dir=calls_dir, include=kept, exclude=dropped)

    prepared = _prepare(job, workspace, ref, prepare)
    if prepared.problem is not None:
        code, facts = prepared.problem
        raise Refused(code, **({} if "unknown" in facts else {"id": prepared.id}), **facts)
    if prepared.running:
        record = job.state.by_id(prepared.id)
        return TraceStarted(id=prepared.id, running=True,
                            record=dict((record.image_correction if record else None) or {}))
    with job.writing():  # prepared outside the lock; written under it
        before = job.snapshot()
        applied = _apply(job, workspace, prepared, workers)
        if applied is None:
            raise Refused(STALE_INPUT, id=prepared.id,
                          message=_STALE_MESSAGE + " Run trace_borders again.")
        result, changed = applied
        if changed:
            job.commit(before)
    return TraceStarted(id=prepared.id, record=result, started=prepared.call is not None)


@dataclass(frozen=True)
class AtlasTraces:
    """What :func:`trace_from_atlas` did: one row per section asked for."""

    #: Per section, in call order: ``id``, ``status`` ("running", "ok" for a
    #: reused reply, or "error" with ``error`` / ``message``), ``cached``,
    #: ``attempt``, ``started``.
    rows: list[dict[str, Any]]
    #: The sections whose image-correction record this call wrote.
    written: list[str]


def trace_from_atlas(
    job: Job,
    workspace: Workspace,
    refs: list[object],
    *,
    image_model: ImageModel,
    passes: int = 1,
    workers: int = registration_tool.MAX_CONCURRENT_IMAGE_CALLS,
) -> AtlasTraces:
    """The placement-free trace (route "atlas") of each section in *refs*.

    A scripting verb, hidden from every listing (``registry.Verb.hidden``):
    the model is shown each clean section and the outlined grayscale atlas
    plane at its position and its own cutting angles (the section's, not
    the stack's: sections supplied per section keep each their plane), never
    its placement
    (``registration_tool.start_atlas_correction``; *passes* 2 adds the
    corrective second call). The result is the section's
    ``image_correction`` record exactly as :func:`trace_borders` writes it,
    so ``fit_deformable`` with a traced fit section (starting from the
    section's written linear placement) and the maps read it unchanged.
    Each section needs a position and a written transform.

    Long-verb semantics, as :func:`trace_borders`: every edit is prepared
    outside the job's write lock; under it each section's geometry is
    checked again (a changed one is its row ``STALE_INPUT``), its call
    started on the job's image executor and its record written; every
    changed record is ONE undo step. A call already running for a section
    at its geometry is not started again (``running``). Per-section problems
    are rows: ``UNKNOWN_SLICE_IDS``, ``KEEPS_HOST_WARP`` /
    ``NONLINEAR_SKIPPED``, ``INVALID_LINEAR_PLACEMENT``,
    ``IMAGE_CORRECTION_IO_ERROR``, ``STALE_INPUT``. Refused (nothing
    written): ``BAD_ARGS`` (no sections, or *passes* not 1 or 2).
    """
    if passes not in (1, 2):
        raise Refused("BAD_ARGS", message="passes must be 1 or 2.")
    if not refs:
        raise Refused("BAD_ARGS", message="Name the sections to trace (slices).")
    rows: list[dict[str, Any]] = []
    pending: list[tuple[int, _Prepared]] = []
    for ref in refs:
        prepared = _prepare(job, workspace, ref, lambda state, ctx, section_id, calls_dir:
                            registration_tool.start_atlas_correction(
                                state, ctx, section_id, passes=passes,
                                image_model=image_model, calls_dir=calls_dir))
        if prepared.problem is not None:
            code, facts = prepared.problem
            rows.append({"id": prepared.id, "status": "error", "error": code,
                         **{key: value for key, value in facts.items() if key == "message"}})
        elif prepared.running:
            rows.append({"id": prepared.id, "status": "running", "started": False,
                         "message": "This section's image correction is already running."})
        else:
            rows.append({"id": prepared.id})
            pending.append((len(rows) - 1, prepared))
    written: list[str] = []
    if not pending:
        return AtlasTraces(rows=rows, written=written)
    with job.writing():  # prepared outside the lock; started and written under it
        before = job.snapshot()
        for index, prepared in pending:
            applied = _apply(job, workspace, prepared, workers)
            if applied is None:
                rows[index] = {"id": prepared.id, "status": "error", "error": STALE_INPUT,
                               "message": _STALE_MESSAGE}
                continue
            result, changed = applied
            if changed:
                written.append(prepared.id)
            rows[index] = {"id": prepared.id, "started": prepared.call is not None, **{
                key: result[key] for key in ("status", "error", "message", "cached", "attempt")
                if key in result}}
        if written:
            job.commit(before)
    return AtlasTraces(rows=rows, written=written)

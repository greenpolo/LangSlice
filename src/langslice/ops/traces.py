"""The image model's border trace of a section: started in the background, recorded.

:func:`trace_borders` takes the image model as an argument
(:class:`langslice.providers.registry.ImageModel`, resolved by the door), so
this module never chooses or imports a provider. The section's current
geometry is the core's (:func:`langslice.core.handoff.correction_fingerprint`)
and the edit is prepared by ``registration_tool.start_correction``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langslice.core import handoff
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


def trace_borders(
    job: Job,
    workspace: Workspace,
    ref: object,
    *,
    image_model: ImageModel,
    prompt: str = "",
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
    it). The edit is prepared outside the job's write lock; the start and
    the write happen under it (:meth:`~langslice.job.job.Job.writing`),
    refused ``STALE_INPUT`` when the section's geometry changed meanwhile;
    the result lands only at the geometry it was made for. Refused:
    ``UNKNOWN_SECTION``, ``KEEPS_HOST_WARP`` / ``NONLINEAR_SKIPPED`` (the host
    kept the section out of Nonlinear, ``Job.nonlinear_refusal``),
    ``INVALID_LINEAR_PLACEMENT`` (the placement cannot be prepared),
    ``IMAGE_CORRECTION_IO_ERROR``, ``STALE_INPUT``.
    """
    record = job.state.resolve(ref)
    if record is None:
        raise Refused("UNKNOWN_SECTION", id=str(ref))
    refusal = job.nonlinear_refusal(record.id)
    if refusal is not None:
        raise Refused(refusal[0], id=record.id, message=refusal[1])
    try:
        current = handoff.correction_fingerprint(job.state, workspace, record.id)
        if job.image_job_running(record.id, current):
            return TraceStarted(id=record.id, record=dict(record.image_correction or {}),
                                running=True)
        result, call = registration_tool.start_correction(
            job.state, workspace, record.id, prompt=prompt, image_model=image_model,
            calls_dir=job.layout.image_correction_dir(record.id))
    except ValueError as exc:
        raise Refused("INVALID_LINEAR_PLACEMENT", id=record.id, message=str(exc)) from exc
    except OSError as exc:
        raise Refused("IMAGE_CORRECTION_IO_ERROR", id=record.id, message=str(exc)) from exc
    with job.writing():  # prepared outside the lock; written under it
        now = job.state.by_id(record.id)
        if now is None or handoff.correction_fingerprint(
                job.state, workspace, record.id) != result["geometry_fingerprint"]:
            raise Refused(STALE_INPUT, id=record.id,
                          message="The section's placement changed while the image call was "
                          "prepared; nothing was started or written. Run trace_borders again.")
        if call is not None:
            job.start_image_job(record.id, result["geometry_fingerprint"], call,
                                workers=workers)
        result = job.portable(result)
        if result != now.image_correction:
            before = job.snapshot()
            now.image_correction = result
            job.commit(before)
    return TraceStarted(id=record.id, record=result, started=call is not None)


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
    section's written linear placement), the submit gate and the maps read
    it unchanged. Each section needs a position and a written transform.

    Long-verb semantics, as :func:`trace_borders`: every edit is prepared
    outside the job's write lock; under it each section's geometry is
    checked again (a changed one is its row ``STALE_INPUT``), its call
    started on the job's image executor and its record written; every
    changed record is ONE undo step. A call already running for a section
    at its geometry is not started again (``running``). Per-section problems
    are rows: ``UNKNOWN_SECTION``, ``INVALID_LINEAR_PLACEMENT``,
    ``IMAGE_CORRECTION_IO_ERROR``, ``STALE_INPUT``. Refused (nothing
    written): ``BAD_ARGS`` (no sections, or *passes* not 1 or 2).
    """
    if passes not in (1, 2):
        raise Refused("BAD_ARGS", message="passes must be 1 or 2.")
    if not refs:
        raise Refused("BAD_ARGS", message="Name the sections to trace (slices).")
    rows: list[dict[str, Any]] = []
    prepared: list[tuple[int, str, dict[str, Any], Any]] = []
    for ref in refs:
        record = job.state.resolve(ref)
        if record is None:
            rows.append({"id": str(ref), "status": "error", "error": "UNKNOWN_SECTION"})
            continue
        try:
            current = handoff.correction_fingerprint(job.state, workspace, record.id)
            if job.image_job_running(record.id, current):
                rows.append({"id": record.id, "status": "running", "started": False,
                             "message": "This section's image correction is already running."})
                continue
            result, call = registration_tool.start_atlas_correction(
                job.state, workspace, record.id, passes=passes, image_model=image_model,
                calls_dir=job.layout.image_correction_dir(record.id))
        except ValueError as exc:
            rows.append({"id": record.id, "status": "error",
                         "error": "INVALID_LINEAR_PLACEMENT", "message": str(exc)})
            continue
        except OSError as exc:
            rows.append({"id": record.id, "status": "error",
                         "error": "IMAGE_CORRECTION_IO_ERROR", "message": str(exc)})
            continue
        rows.append({"id": record.id})
        prepared.append((len(rows) - 1, record.id, result, call))
    written: list[str] = []
    if not prepared:
        return AtlasTraces(rows=rows, written=written)
    with job.writing():  # prepared outside the lock; started and written under it
        before = job.snapshot()
        for index, section_id, result, call in prepared:
            now = job.state.by_id(section_id)
            if now is None or handoff.correction_fingerprint(
                    job.state, workspace, section_id) != result["geometry_fingerprint"]:
                rows[index] = {"id": section_id, "status": "error", "error": STALE_INPUT,
                               "message": "The section's placement changed while the image "
                               "call was prepared; nothing was started or written for it."}
                continue
            if call is not None:
                job.start_image_job(section_id, result["geometry_fingerprint"], call,
                                    workers=workers)
            portable = job.portable(result)
            if portable != now.image_correction:
                now.image_correction = portable
                written.append(section_id)
            rows[index] = {"id": section_id, "started": call is not None, **{
                key: portable[key] for key in ("status", "error", "message", "cached", "attempt")
                if key in portable}}
        if written:
            job.commit(before)
    return AtlasTraces(rows=rows, written=written)

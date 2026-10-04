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

from langslice import registration_tool
from langslice.core import handoff
from langslice.ops.refusal import Refused

if TYPE_CHECKING:
    from langslice.linear.job import Job
    from langslice.linear.workspace import Workspace
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
    (*workers* at most at once; :meth:`~langslice.linear.job.Job.start_image_job`).
    The section's ``image_correction`` record is written as ONE undo step
    when it changed; the call's result lands on it later (submit waits for
    it). Refused: ``UNKNOWN_SECTION``, ``INVALID_LINEAR_PLACEMENT`` (the
    placement cannot be prepared), ``IMAGE_CORRECTION_IO_ERROR``.
    """
    record = job.state.resolve(ref)
    if record is None:
        raise Refused("UNKNOWN_SECTION", id=str(ref))
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
    if call is not None:
        job.start_image_job(record.id, result["geometry_fingerprint"], call, workers=workers)
    result = job.portable(result)
    if result != record.image_correction:
        before = job.snapshot()
        record.image_correction = result
        job.commit(before)
    return TraceStarted(id=record.id, record=result, started=call is not None)

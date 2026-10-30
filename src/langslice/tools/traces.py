"""The image model's border trace of a section, and the fit that follows it.

:func:`trace_borders` is LangSlice's own nonlinear method packaged: the image
call runs in the background, and when it lands a fit of the traced borders is
applied as background work (:mod:`langslice.job.background`,
:func:`land_trace`). It takes the image model as an argument
(:class:`langslice.providers.connections.ImageModel`, resolved by the door), so
this module never chooses or imports a provider. The section's current geometry
is the core's (:func:`langslice.core.fits.frame.correction_fingerprint`) and
the edit is prepared by ``registration_tool.start_correction``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from langslice.core.fits import deformation
from langslice.core.fits import frame as handoff
from langslice.core.fits.trace import trace as registration_tool
from langslice.core.jsonable import JsonValue
from langslice.core.refusal import RefusalCode, Refused
from langslice.core.sections.damage import exclusions
from langslice.core.stack.codec import trace_from_json, trace_to_json
from langslice.core.stack.trace import BorderTrace
from langslice.job.background import DONE, FAILED, Landed
from langslice.tools.refusal import frozen_row, unknown_sections
from langslice.tools.stale import section_inputs

if TYPE_CHECKING:
    from langslice.core.pictures.options import DisplayOptions
    from langslice.core.stack.section import Section
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job
    from langslice.providers.connections import ImageModel


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
    #: True when this section's call at this geometry (or its packaged trace)
    #: was already running (nothing started, nothing written).
    running: bool = False
    #: The packaged trace's background work id (:func:`trace_borders`); None
    #: when nothing was started, and when the trace had already landed.
    work: str | None = None
    #: The packaged trace at this placement and region choice had already
    #: landed: the section's deformation holds its fit (nothing started).
    landed: bool = False


_STALE_MESSAGE = ("The section's placement changed while the image call was prepared; "
                  "nothing was started or written for it.")


def _trace_json(section: Section) -> dict[str, JsonValue]:
    """Return the section's border trace in its JSON form (empty: none)."""
    if section.border_trace is None:
        return {}
    return trace_to_json(section.border_trace)


def _without_cached(trace: BorderTrace | None) -> BorderTrace | None:
    return None if trace is None else replace(trace, cached=None)


def _start_trace(
    job: Job,
    workspace: Workspace,
    ref: object,
    *,
    image_model: ImageModel,
    prompt: str,
    chosen: tuple[str, ...],
    workers: int,
) -> TraceStarted:
    """Start one section's border trace in the background, or reuse it
    (the trace alone; :func:`trace_borders` adds the fit). The call is
    prepared outside the job's write lock and started under it."""
    record = job.state.resolve(ref)
    if record is None:
        raise unknown_sections(job.state, [str(ref)])
    section_id = record.id
    frozen = frozen_row(job, section_id, "deformation")
    if frozen is not None:
        raise Refused(frozen["error"], id=section_id, properties=frozen["properties"],
                      message=frozen["message"])
    try:
        current = handoff.correction_fingerprint(job.state, workspace, section_id)
        if job.image_job_running(section_id, current):
            return TraceStarted(id=section_id, running=True,
                                record=_trace_json(record))
        kept, dropped = exclusions(record, chosen)
        result, call = registration_tool.start_correction(
            job.state, workspace, section_id, prompt=prompt, image_model=image_model,
            calls_dir=job.layout.border_trace_dir(section_id), include=kept, exclude=dropped)
    except ValueError as exc:
        raise Refused(RefusalCode.INVALID_LINEAR_PLACEMENT,
                      id=section_id, message=str(exc)) from exc
    except OSError as exc:
        raise Refused(RefusalCode.IMAGE_CORRECTION_IO_ERROR,
                      id=section_id, message=str(exc)) from exc
    with job.writing():  # prepared outside the lock; written under it
        before = job.snapshot()
        now = job.state.by_id(section_id)
        fingerprint = result["geometry_fingerprint"]
        if now is None or handoff.correction_fingerprint(
                job.state, workspace, section_id) != fingerprint:
            raise Refused(RefusalCode.STALE_INPUT, id=section_id,
                          message=_STALE_MESSAGE + " Run trace_borders again.")
        if call is not None:
            job.start_image_job(section_id, fingerprint, call, workers=workers)
        portable = job.portable(result)
        trace = trace_from_json(portable)
        # A saved reply reused as it stands differs only by its ``cached`` flag:
        # nothing to write, so no undo step.
        if _without_cached(trace) != _without_cached(now.border_trace):
            now.border_trace = trace
            job.commit(before)
    return TraceStarted(id=section_id, record=portable, started=call is not None)


#: The background work's kind (:mod:`langslice.job.background`).
TRACE_WORK = "trace_borders"
#: The fit a landed trace gets: the traced lines as named regions against the
#: atlas borders, FireANTs at 80 um smoothness.
TRACE_FIT = deformation.Choice(fit_section=deformation.TRACED_BORDERS, fit_atlas="borders",
                               smoothness_um=80.0)
#: Fit rows that mean the section or its trace changed before the fit applied.
_STALE_CODES = (
    RefusalCode.STALE_INPUT,
    RefusalCode.TRACE_STALE,
    RefusalCode.TRACE_RUNNING,
    RefusalCode.NO_TRACE,
)


def trace_borders(
    job: Job,
    workspace: Workspace,
    ref: object,
    *,
    image_model: ImageModel,
    prompt: str = "",
    restrict_to: Any = (),
    options: DisplayOptions | None = None,
    workers: int = registration_tool.MAX_CONCURRENT_IMAGE_CALLS,
) -> TraceStarted:
    """LangSlice's own nonlinear method on one section, packaged: the image
    model traces the atlas borders onto the section, then FireANTs fits what it
    drew and the deformation is applied. Returns at once (``work``: the id
    of the background work, :mod:`langslice.job.background`).

    Now: the section's border trace is started in the background, or a
    saved reply at its geometry reused (``cached``), the section's
    ``image_correction`` record written as ONE undo step when it changed, and a
    piece of background work started. The model is shown the section's
    ``preprocessed`` image and the same picture with the placed atlas borders
    (``registration_tool.start_correction``): with *restrict_to*, only those
    regions' borders (descendants included, ``"CTX:left"`` for one side), and
    the section's marked regions left out
    (:func:`langslice.core.sections.damage.exclusions`). *prompt* replaces the
    base prompt (blank sends it unchanged); the prompt sent is saved with each
    attempt in the job folder (``sections/<name>/image_correction/``).

    When the reply lands (:func:`land_trace`): the trace is recorded on the
    section, then a fit of the traced borders runs (:data:`TRACE_FIT`, on
    top of the section's applied deformation where it holds one, else its
    linear placement; with *restrict_to*, by those regions only) and is
    applied as its own undo step. A section whose inputs (placement,
    section images, damage, deformation) changed since this call gets
    nothing, its notice saying ``STALE_INPUT``. With *options*, the fit and
    the trace are drawn and saved with the work.

    A section whose packaged trace is still running is not started again
    (``running``, that work's id). A call whose trace is the saved reply at
    this placement and region choice, and whose fit the section's applied
    deformation already holds, fits nothing again (``landed``, no work).
    Refused before any image call:
    ``ENGINE_MISSING``, ``BAD_ARGS`` / ``UNKNOWN_REGIONS`` / ``NO_SIDES`` (a
    bad *restrict_to*), ``UNKNOWN_SLICE_IDS``, ``FROZEN`` (the
    section's deformation is frozen), ``INVALID_LINEAR_PLACEMENT``, ``IMAGE_CORRECTION_IO_ERROR``,
    ``STALE_INPUT`` (the geometry changed while the call was prepared).
    """
    from langslice.tools.deformable import (
        refuse_without_engine,
        region_entries,
    )

    refuse_without_engine()
    regions = region_entries(workspace, job.state, restrict_to)
    record = job.state.resolve(ref)
    if record is None:
        raise unknown_sections(job.state, [str(ref)])
    held = job.background.running_for(record.id, TRACE_WORK)
    if held is not None:
        return TraceStarted(id=record.id, running=True, work=held.id,
                            record=_trace_json(record))
    started = _start_trace(job, workspace, record.id, image_model=image_model, prompt=prompt,
                           chosen=regions, workers=workers)
    now = job.state.by_id(record.id) or record
    if not started.started and _landed(job, now):
        return replace(started, landed=True)
    expected = section_inputs(job.state, now, deformation=True)
    section_id = record.id

    def land() -> Landed:
        return land_trace(job, workspace, section_id, expected, regions=regions,
                          options=options)

    work = job.background.start(TRACE_WORK, (section_id,), land)
    return replace(started, work=work.id)


def land_trace(
    job: Job,
    workspace: Workspace,
    section_id: str,
    expected: str,
    *,
    regions: tuple[str, ...] = (),
    options: DisplayOptions | None = None,
) -> Landed:
    """A packaged trace's landing (background work, after the image call):
    the fit of the traced borders, applied as its own undo step, or why not.

    *expected* is :func:`~langslice.tools.stale.section_inputs` (with the
    deformation) when the trace was asked for: a section whose inputs moved
    since gets nothing (``STALE_INPUT``), as does one whose trace was undone or
    made at another placement. A failed image call is a failed work.
    """
    from langslice.tools.deformable import fit_deformable

    state = job.state
    record = state.by_id(section_id)
    if record is None:
        return Landed(status=FAILED, text=f"{section_id} is no longer in the stack; nothing "
                      "was applied.",
                      result={"error": RefusalCode.UNKNOWN_SLICE_IDS})
    held = record.border_trace
    form = _trace_json(record)
    trace = {key: form[key] for key in ("status", "attempt", "cached", "prompt_edited",
                                        "include", "exclude", "error", "message")
             if key in form}
    if held is not None and held.status == "error":
        reason = held.message or held.error or "no image"
        return Landed(status=FAILED, text=f"the image model's call failed ({reason}); "
                      "nothing was applied.",
                      result={"error": RefusalCode.TRACE_FAILED,
                              "trace": trace})
    if (section_inputs(state, record, deformation=True) != expected
            or held is None or held.status != "ok"):
        return _stale(trace)
    done = fit_deformable(job, workspace, [record], TRACE_FIT, restrict_to=regions,
                          options=options)
    row: dict[str, Any] = (done.rows[0] if done.rows
                           else Refused(RefusalCode.FIT_FAILED).payload())
    if row.get("status") == "error":
        if row.get("error") in _STALE_CODES:
            return _stale(trace)
        return Landed(status=FAILED, text=f"the fit of the traced borders failed "
                      f"({row.get('error')}: {row.get('message', '')}); nothing was applied.",
                      result={"error": row.get("error"), "row": row, "trace": trace})
    numbers = row.get("displacement_mm") or {}
    built_on = ("its previous deformation" if row.get("steps", 0) > 1
                else "its linear placement")
    facts = (f"Displacement median {numbers.get('median')} mm, max {numbers.get('max')} mm; "
             f"fold fraction {row.get('fold_fraction')}"
             + (f"; flags {deformation.flag_words(row)}" if row.get("flags") else "") + ".")
    if not row.get("written"):
        text = ("the trace landed; its fit equals the deformation the section already holds, "
                "so nothing changed. " + facts)
    else:
        text = ("the trace landed and FireANTs fitted the traced borders (80 um smoothness"
                + (f", restricted to {', '.join(regions)}" if regions else "")
                + f") on top of {built_on}. The deformation is applied as its own undo "
                "step: undo removes it while it is the latest step. " + facts)
    return Landed(status=DONE, text=text, pictures=list(done.pictures),
                  result={"row": row, "trace": trace})


def _landed(job: Job, section: Section) -> bool:
    """Whether *section*'s saved trace has landed: its border trace is a
    completed reply and a step of its applied deformation, at its current
    linear placement, fitted that very trace
    (:func:`~langslice.tools.deformable.fit_deformable`
    records the trace a traced step read)."""
    held = section.border_trace
    applied = section.deformation
    if held is None or held.status != "ok":
        return False
    if applied is None or not applied.steps:
        return False
    if applied.linear_key != deformation.linear_key(job.state, section):
        return False
    trace = held.artifact_dir or ""
    return bool(trace) and any(step.get("trace") == trace for step in applied.steps)


def _stale(trace: dict[str, Any]) -> Landed:
    return Landed(status=FAILED,
                  text=f"{RefusalCode.STALE_INPUT}: the section's placement, section "
                  "images, damage, deformation or trace changed before the trace landed; "
                  "nothing was applied. Run trace_borders again.",
                  result={"error": RefusalCode.STALE_INPUT, "trace": trace})

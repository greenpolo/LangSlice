"""The image model's border trace of a section, and the fit that follows it.

:func:`trace_borders` is LangSlice's own nonlinear method packaged: the
image call runs in the background, and when it lands a fit of the traced
borders is applied as background work (:mod:`langslice.job.background`,
:func:`land_trace`). It takes the image model as an argument
(:class:`langslice.providers.registry.ImageModel`, resolved by the door), so
this module never chooses or imports a provider. The section's current
geometry is the core's (:func:`langslice.core.handoff.correction_fingerprint`)
and the edit is prepared by ``registration_tool.start_correction``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from langslice.core import deformation, handoff
from langslice.core.damage import exclusions
from langslice.core.nonlinear import registration_tool
from langslice.job.background import DONE, FAILED, Landed
from langslice.ops.inputs import STALE_INPUT, section_inputs
from langslice.ops.refusal import Refused

if TYPE_CHECKING:
    from langslice.core.display import DisplayOptions
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
    #: True when this section's call at this geometry (or its packaged trace)
    #: was already running (nothing started, nothing written).
    running: bool = False
    #: The packaged trace's background work id (:func:`trace_borders`); None
    #: when nothing was started, and when the trace had already landed.
    work: str | None = None
    #: The packaged trace at this placement and region choice had already
    #: landed: the section's deformation holds its fit (nothing started).
    landed: bool = False


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
    # A saved reply reused as it stands differs only by its ``cached`` flag:
    # nothing to write, so no undo step.
    changed = _without_cached(portable) != _without_cached(now.image_correction)
    if changed:
        now.image_correction = portable
    return portable, changed


def _without_cached(record: dict[str, Any] | None) -> dict[str, Any] | None:
    return None if record is None else {k: v for k, v in record.items() if k != "cached"}


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
    """Start one section's image correction in the background, or reuse it
    (the trace alone; :func:`trace_borders` adds the fit)."""

    def prepare(state: Any, ctx: Any, section_id: str, calls_dir: Any,
                ) -> tuple[dict[str, Any], Any]:
        kept, dropped = exclusions(state.by_id(section_id), chosen, ())
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


#: The background work's kind (:mod:`langslice.job.background`).
TRACE_WORK = "trace_borders"
#: The fit a landed trace gets: the traced lines as named regions against the
#: atlas borders, ANTs at medium stiffness.
TRACE_FIT = deformation.Choice(fit_section=deformation.TRACED_BORDERS, fit_atlas="borders",
                               engine="ants", stiffness="medium")
#: Fit rows that mean the section or its trace changed before the fit applied.
_STALE_CODES = (STALE_INPUT, "TRACE_STALE", "TRACE_RUNNING", "NO_TRACE")


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
    model traces the atlas borders onto the section, then ANTs fits what it
    drew and the deformation is applied. Returns at once (``work``: the id
    of the background work, :mod:`langslice.job.background`).

    Now: the section's image correction is started in the background, or a
    saved reply at its geometry reused (``cached``), the section's
    ``image_correction`` record written as ONE undo step when it changed,
    and a piece of background work started. The model is shown the
    section's preprocessed channel and the same picture with the placed
    atlas borders (``registration_tool.start_correction``): with
    *restrict_to*, only those regions' borders (descendants included,
    ``"CTX:left"`` for one side), and the section's marked regions left out
    (:func:`langslice.core.damage.exclusions`). *prompt* replaces the base
    prompt (blank sends it unchanged); the prompt sent is saved with each
    attempt in the job folder (``sections/<name>/image_correction/``).

    When the reply lands (:func:`land_trace`): the trace is recorded on the
    section, then a fit of the traced borders runs (:data:`TRACE_FIT`, on
    top of the section's applied deformation where it holds one, else its
    linear placement; with *restrict_to*, by those regions only) and is
    applied as its own undo step. A section whose inputs (placement,
    preprocessed channel, damage, deformation) changed since this call gets
    nothing, its notice saying ``STALE_INPUT``. With *options*, the fit and
    the trace are drawn and saved with the work.

    A section whose packaged trace is still running is not started again
    (``running``, that work's id). A call whose trace is the saved reply at
    this placement and region choice, and whose fit the section's applied
    deformation already holds, fits nothing again (``landed``, no work).
    Refused before any image call:
    ``ANTS_MISSING``, ``BAD_ARGS`` / ``UNKNOWN_REGIONS`` / ``NO_SIDES`` (a
    bad *restrict_to*), ``UNKNOWN_SLICE_IDS``, ``KEEPS_HOST_WARP`` /
    ``NONLINEAR_SKIPPED`` (the host kept the section out of Nonlinear),
    ``INVALID_LINEAR_PLACEMENT``, ``IMAGE_CORRECTION_IO_ERROR``,
    ``STALE_INPUT`` (the geometry changed while the call was prepared).
    """
    from langslice.ops.deformable import START_LATEST, refuse_without_ants, region_entries

    refuse_without_ants()
    regions = region_entries(workspace, job.state, restrict_to)
    record = job.state.resolve(ref)
    if record is None:
        raise Refused("UNKNOWN_SLICE_IDS", unknown=[str(ref)])
    held = job.background.running_for(record.id, TRACE_WORK)
    if held is not None:
        return TraceStarted(id=record.id, running=True, work=held.id,
                            record=dict(record.image_correction or {}))
    started = _start_trace(job, workspace, record.id, image_model=image_model, prompt=prompt,
                           chosen=regions, workers=workers)
    now = job.state.by_id(record.id) or record
    if not started.started and _landed(job, now):
        return replace(started, landed=True)
    expected = section_inputs(job.state, now, deformation=True)
    section_id = record.id

    def land() -> Landed:
        return land_trace(job, workspace, section_id, expected, regions=regions,
                          start=START_LATEST, options=options)

    work = job.background.start(TRACE_WORK, (section_id,), land)
    return replace(started, work=work.id)


def land_trace(
    job: Job,
    workspace: Workspace,
    section_id: str,
    expected: str,
    *,
    regions: tuple[str, ...] = (),
    start: str = "linear",
    options: DisplayOptions | None = None,
) -> Landed:
    """A packaged trace's landing (background work, after the image call):
    the fit of the traced borders, applied as its own undo step, or why not.

    *expected* is :func:`~langslice.ops.inputs.section_inputs` (with the
    deformation) when the trace was asked for: a section whose inputs moved
    since gets nothing (``STALE_INPUT``), as does one whose trace was undone
    or made at another placement. A failed image call is a failed work.
    """
    from langslice.ops.deformable import fit_deformable

    state = job.state
    record = state.by_id(section_id)
    if record is None:
        return Landed(status=FAILED, text=f"{section_id} is no longer in the stack; nothing "
                      "was applied.", result={"error": "UNKNOWN_SLICE_IDS"})
    held = record.image_correction or {}
    trace = {key: held[key] for key in ("status", "attempt", "cached", "prompt_edited",
                                        "include", "exclude", "error", "message")
             if key in held}
    if held.get("status") == "error":
        reason = str(held.get("message") or held.get("error") or "no image")
        return Landed(status=FAILED, text=f"the image model's call failed ({reason}); "
                      "nothing was applied.", result={"error": "TRACE_FAILED", "trace": trace})
    if section_inputs(state, record, deformation=True) != expected or held.get("status") != "ok":
        return _stale(trace)
    done = fit_deformable(job, workspace, [record], TRACE_FIT, restrict_to=regions,
                          start=start, options=options)
    row: dict[str, Any] = (done.rows[0] if done.rows
                           else {"status": "error", "error": "FIT_FAILED"})
    if row.get("status") == "error":
        if row.get("error") in _STALE_CODES:
            return _stale(trace)
        return Landed(status=FAILED, text=f"the fit of the traced borders failed "
                      f"({row.get('error')}: {row.get('message', '')}); nothing was applied.",
                      result={"error": row.get("error"), "row": row, "trace": trace})
    numbers = row.get("displacement_mm") or {}
    flags = [flag.get("code") for flag in row.get("flags") or []]
    built_on = ("its previous deformation" if row.get("steps", 0) > 1
                else "its linear placement")
    facts = (f"Displacement median {numbers.get('median')} mm, max {numbers.get('max')} mm; "
             f"fold fraction {row.get('fold_fraction')}"
             + (f"; flags {', '.join(str(code) for code in flags)}" if flags else "") + ".")
    if not row.get("written"):
        text = ("the trace landed; its fit equals the deformation the section already holds, "
                "so nothing changed. " + facts)
    else:
        text = ("the trace landed and ANTs fitted the traced borders (medium stiffness"
                + (f", restricted to {', '.join(regions)}" if regions else "")
                + f") on top of {built_on}. The deformation is applied as its own undo "
                "step: undo removes it while it is the latest step. " + facts)
    return Landed(status=DONE, text=text, pictures=list(done.pictures),
                  result={"row": row, "trace": trace})


def _landed(job: Job, record: Any) -> bool:
    """Whether *record*'s saved trace has landed: its image correction is a
    completed reply and a step of its applied deformation, at its current
    linear placement, fitted that very trace (:func:`~langslice.ops.deformable.fit_deformable`
    records the trace a traced step read)."""
    held = record.image_correction or {}
    applied = record.deformation or {}
    if held.get("status") != "ok" or not applied.get("steps"):
        return False
    if applied.get("linear_key") != deformation.linear_key(job.state, record):
        return False
    trace = str(held.get("artifact_dir") or "")
    return bool(trace) and any(step.get("trace") == trace for step in applied["steps"])


def _stale(trace: dict[str, Any]) -> Landed:
    return Landed(status=FAILED, text=f"{STALE_INPUT}: the section's placement, preprocessed "
                  "channel, damage, deformation or trace changed before the trace landed; "
                  "nothing was applied. Run trace_borders again.",
                  result={"error": STALE_INPUT, "trace": trace})


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
    so :func:`langslice.ops.deformable.fit_deformable` with a traced fit
    section (from the section's written linear placement), the submit gate
    and the maps read it unchanged; no verb fits it: a script does.
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

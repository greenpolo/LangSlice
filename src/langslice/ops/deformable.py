"""The deformation on top of a section's linear placement: fit it, or keep the placement.

:func:`fit_deformable` runs one or more deformable fits per section
(:mod:`langslice.core.deformation`: the fit grid, the image each fit
reads, the record cache and the engines) and, when the call has exactly one
setting, APPLIES each section's result as its deformation: one undo step
for the call. Several settings (candidates) are a preview: every fit runs
and is cached, nothing is written. :func:`keep_linear` records instead that
a section's linear placement stands without a deformation.

The arguments arrive checked (the door validates them and resolves each
candidate into a :class:`~langslice.core.deformation.Choice`); what can go
wrong per section (no valid placement, ``start="current"`` with nothing to
compose onto, a trace that is missing, stale or failed, a fit that failed,
a record that cannot be saved) is that section's row, not a refusal of the
call. No pictures: the result carries the image each fit read and its
records, which the door draws.
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from PIL import Image

from langslice.core import deformation
from langslice.core.state import SliceState
from langslice.ops.inputs import section_inputs, stale_row
from langslice.ops.refusal import Refused

if TYPE_CHECKING:
    import numpy as np

    from langslice.core.display import DisplayOptions
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fitted:
    """One fit that produced a record, for the door to draw: its row (the
    same dict as in :attr:`DeformableFit.rows`), the fit itself (the image
    it read, the record it started from) and its record."""

    row: dict[str, Any]
    fit: deformation.Job
    outcome: deformation.DeformableRecord


@dataclass(frozen=True)
class DeformableFit:
    """What one :func:`fit_deformable` call did.

    ``rows``: one per section and setting, in call order (``candidate``
    numbered when previewing): the settings, the engine numbers, the
    displacement summary, ``cached``, and when applied ``written`` and
    ``steps``; or ``status: error`` with the code. ``fitted``: the rows that
    hold a fit to draw, in the same order. ``traced``: per section whose
    settings read its trace, the image the fit read and the trace's lines on
    the fit grid. ``written``: the sections whose deformation changed.
    """

    applied: bool
    rows: list[dict[str, Any]] = field(default_factory=list)
    fitted: list[Fitted] = field(default_factory=list)
    traced: dict[str, tuple[Image.Image, np.ndarray]] = field(default_factory=dict)
    written: list[str] = field(default_factory=list)
    #: With display options (:func:`pictures`): the pictures, in order (each
    #: drawn row's ``image_indexes`` point into them), ``{"id",
    #: "image_indexes"}`` per traced section's trace picture, and ``{"id",
    #: "message"}`` per picture that failed (the write stands).
    pictures: list[Image.Image] = field(default_factory=list)
    traces: list[dict[str, Any]] = field(default_factory=list)
    render_failed: list[dict[str, str]] = field(default_factory=list)


def fit_deformable(
    job: Job,
    workspace: Workspace,
    records: list[SliceState],
    choices: list[deformation.Choice],
    *,
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
    start: str = "linear",
    options: DisplayOptions | None = None,
) -> DeformableFit:
    """Fit every choice on every section; with one choice, apply it.

    *include* / *exclude* are region entries (the engine's ``structures`` /
    ``exclude``, sides allowed); *start* is ``linear`` (from the linear
    placement) or ``current`` (composed onto the section's applied
    deformation). A traced choice waits for the section's trace still running
    (one :data:`~langslice.core.deformation.TRACE_WAIT_S` deadline for the
    call; a trace that lands is checkpointed without an undo step).
    Identical inputs reuse a cached or saved result; applying a section's own
    current key again writes nothing (``written: false``). With *options*,
    every fit and every traced section's trace is then drawn (:func:`pictures`).

    The fits run outside the job's write lock, from the state as it stood;
    applying takes the lock (:meth:`~langslice.job.job.Job.writing`) and
    refuses a section whose inputs changed meanwhile
    (:data:`langslice.ops.inputs.STALE_INPUT`, that section's row), applying
    the others.
    """
    state = job.state
    store = job.deformations
    applying = len(choices) == 1
    rows: list[dict[str, Any]] = []
    jobs: list[deformation.Job] = []
    traced: dict[str, tuple[Image.Image, Any]] = {}
    trace_deadline = time.monotonic() + deformation.TRACE_WAIT_S
    traced_choice = any(choice.fit_section in deformation.TRACED for choice in choices)
    expected: dict[str, str] = {}
    for record in records:
        running = False
        if traced_choice:
            # A traced image waits for the section's trace still running; a
            # result lands under the job's lock, which may reload the state.
            running = not job.wait_image_job(record.id, trace_deadline - time.monotonic())
            record = state.by_id(record.id) or record
        expected[record.id] = section_inputs(state, record, deformation=start == "current",
                                             trace=traced_choice)
        try:
            grid = deformation.fit_grid(state, workspace, record)
        except (ValueError, OSError) as exc:
            rows.append({"id": record.id, "status": "error",
                         "error": "INVALID_LINEAR_PLACEMENT", "message": str(exc)})
            continue
        previous = None
        previous_key: str | None = None
        if start == "current":
            previous = store.current(state, record)
            if previous is None:
                rows.append({"id": record.id, "status": "error", "error": "NO_DEFORMATION",
                             "message": "start='current' composes onto the section's "
                             "applied deformation; this section has none."})
                continue
            previous_key = str((record.deformation or {}).get("key"))
        for number, choice in enumerate(choices, start=1):
            failure = {"id": record.id, "status": "error", "settings": choice.echo(),
                       **({} if applying else {"candidate": number})}
            try:
                image, identity = deformation.stain_image(workspace, state, grid,
                                                          choice.fit_section)
                lines = None
                if choice.fit_section in deformation.TRACED:
                    lines, trace = deformation.traced_lines(
                        state, workspace, grid, running=running,
                        waited_s=deformation.TRACE_WAIT_S, root=job.folder)
                    identity = {**identity, "trace": trace}
                    traced.setdefault(record.id, (image, lines))
                settings = choice.settings(include, exclude)
            except deformation.FitRefusal as refusal:
                rows.append({**failure, **refusal.payload, "id": record.id})
                continue
            except (ValueError, OSError) as exc:
                rows.append({**failure, "error": "BAD_SETTINGS", "message": str(exc)})
                continue
            key = deformation.cache_key(state, grid, settings, identity, previous_key)
            cached = store.get(record.id, key)
            jobs.append(deformation.Job(
                grid=grid, choice=choice, settings=settings, key=key,
                image_identity=identity, previous=previous, image=image, lines=lines,
                result=cached, cached=cached is not None,
            ))
            rows.append({"id": record.id, "job": len(jobs) - 1,
                         **({} if applying else {"candidate": number})})
    deformation.run_jobs(workspace, jobs)

    with job.writing() if applying else contextlib.nullcontext():
        before = job.snapshot()
        done = _apply(job, rows, jobs, applying=applying, expected=expected,
                      include=include, exclude=exclude, start=start)
        if done.written:
            job.commit(before)
    done = replace(done, traced=traced)
    if options is None:
        return done
    return pictures(workspace, done, options, candidates=len(choices), include=include,
                    exclude=exclude, start=start)


def _apply(
    job: Job, rows: list[dict[str, Any]], jobs: list[deformation.Job], *, applying: bool,
    expected: dict[str, str], include: tuple[str, ...], exclude: tuple[str, ...], start: str,
) -> DeformableFit:
    """Every fit's row; with *applying* (under the job's lock), each section's
    result as its deformation when its inputs are unchanged."""
    state = job.state
    store = job.deformations
    written: list[str] = []
    fitted: list[Fitted] = []
    for row in rows:
        index = row.pop("job", None)
        if index is None:
            continue
        fit = jobs[index]
        outcome = fit.result
        row["settings"] = fit.choice.echo()
        if not isinstance(outcome, deformation.DeformableRecord):
            row.update(status="error", error="FIT_FAILED",
                       message=getattr(outcome, "error", "no result"))
            continue
        store.put(fit.key, outcome)
        numbers = deformation.summary(outcome, fit.previous)
        record = fit.grid.record
        if applying:
            # The state may have been reloaded: the section as it is now.
            now = state.by_id(record.id)
            if now is None or section_inputs(
                    state, now, deformation=start == "current",
                    trace=fit.choice.fit_section in deformation.TRACED) != expected[record.id]:
                row.update(stale_row(record.id))
                continue
            record = now
        row.update(status="ok", engine_settings=deformation.engine_settings(fit.settings),
                   **numbers, runtime_s=round(float(outcome.engine.get("runtime_s", 0.0)), 1),
                   cached=fit.cached)
        if applying:
            linear = deformation.linear_key(state, record)
            outcome.provenance = deformation.provenance(
                fit.grid, fit.choice, include, exclude, start, fit.image_identity, linear)
            held = record.deformation or {}
            if held.get("key") == fit.key:
                row["written"] = False
            else:
                try:
                    folder = store.save(record.id, fit.key, outcome)
                except OSError as exc:
                    row.update(status="error", error="RECORD_WRITE_FAILED", message=str(exc))
                    continue
                record.deformation = deformation.reference(
                    folder=job.layout.relative(folder), key=fit.key, linear=linear, record=outcome,
                    choice=fit.choice, include=include, exclude=exclude, start=start,
                    previous=held, numbers=numbers,
                )
                row["written"] = True
                written.append(record.id)
            row["steps"] = len((record.deformation or {}).get("steps") or [])
        fitted.append(Fitted(row=row, fit=fit, outcome=outcome))
    return DeformableFit(applied=applying, rows=rows, fitted=fitted, written=written)


def pictures(
    workspace: Workspace,
    done: DeformableFit,
    options: DisplayOptions,
    *,
    candidates: int,
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
    start: str = "linear",
) -> DeformableFit:
    """*done* with its pictures: per drawn row, the final atlas borders on the
    image the fit read (:func:`langslice.core.deformation.picture`), titled
    with the section, applied or candidate number of *candidates*, engine,
    stiffness and what was fitted; mode ``ab`` adds what the fit started from
    (the previous deformation, or the linear placement). Then each traced
    section's trace on the section (:func:`~langslice.core.deformation.trace_picture`).
    Included or ``view.regions`` borders strong, excluded ones pink. Each
    drawn row gets ``image_indexes``; a picture that fails is listed in
    ``render_failed``.
    """
    from langslice.core import layers
    from langslice.core.canvas import normalize_border_style

    parts: list[Image.Image] = []
    failed: list[dict[str, str]] = []
    highlight = [name for name, _ids in options.regions] or list(include)
    color, thickness = normalize_border_style(options.border_color, options.border_thickness)
    style = deformation.Style(
        zoom=() if options.full_view else tuple(options.zoom), highlight=tuple(highlight),
        marked=exclude, outlines=options.layer, color=color, thickness=thickness,
        atlas_opacity=options.atlas_opacity, long_edge=options.long_edge,
    )
    for fitted in done.fitted:
        row, fit, outcome = fitted.row, fitted.fit, fitted.outcome
        record = fit.grid.record
        heading = (f"{record.id}  " + ("applied" if done.applied else
                   f"candidate {row['candidate']}/{candidates}")
                   + f": {fit.choice.engine} {fit.choice.stiffness}")
        detail_line = (f"{fit.choice.fit_section} vs {fit.choice.fit_atlas}, start {start}"
                       + (f", include {','.join(include)}" if include else "")
                       + (f", exclude {','.join(exclude)}" if exclude else ""))
        shown_atlas = options.atlas_images
        first = len(parts)
        try:
            def noted(index: int, section: str = record.id) -> dict[str, Any]:
                return {"sections": (section,),
                        "mode": (("after" if index == 0 else "before")
                                 if options.mode == "ab" else options.mode)}

            images = [deformation.picture(workspace, fit.image, outcome, warped=True,
                                          style=style, atlas_images=shown_atlas,
                                          title=f"{heading}\n{detail_line}", note=noted(0))]
            if options.mode == "ab":
                if fit.previous is not None:
                    images.append(deformation.picture(
                        workspace, fit.image, fit.previous, warped=True, style=style,
                        atlas_images=shown_atlas,
                        title=f"{record.id}  before: the deformation it started from",
                        note=noted(1)))
                else:
                    images.append(deformation.picture(
                        workspace, fit.image, outcome, warped=False, style=style,
                        atlas_images=shown_atlas,
                        title=f"{record.id}  before: the linear placement", note=noted(1)))
            parts.extend(images)
        except Exception as exc:
            logger.warning("fit_deformable picture failed for %s", record.id, exc_info=True)
            del parts[first:]
            failed.append({"id": record.id, "message": str(exc)})
            continue
        row["image_indexes"] = list(range(first, len(parts)))
    # Each traced section's trace, once per call, so it can be reviewed.
    traces: list[dict[str, Any]] = []
    for section_id, (image, lines) in done.traced.items():
        try:
            picture = deformation.trace_picture(
                image, lines, style=style,
                title=f"{section_id}  trace_borders result: the image model's lines")
        except Exception as exc:
            logger.warning("trace picture failed for %s", section_id, exc_info=True)
            failed.append({"id": section_id, "message": str(exc)})
            continue
        traces.append({"id": section_id, "image_indexes": [len(parts)]})
        parts.append(layers.note(picture, sections=(section_id,), mode="trace"))
    return replace(done, pictures=parts, traces=traces, render_failed=failed)


@dataclass(frozen=True)
class KeptLinear:
    """The sections that now record their linear placement as standing."""

    touched: list[str]
    reason: str


def keep_linear(job: Job, records: list[SliceState], reason: str) -> KeptLinear:
    """Record that each section's linear placement stands: no warp, *reason*.

    Held on ``SliceState.deformation`` with the placement's ``linear_key``,
    so it satisfies `submit` like an applied fit, and a later change to the
    placement clears it the same way. One undo step. Refused
    (``NOTHING_WRITTEN``, each offending section under ``results``) when any
    section lacks a position or a transform.
    """
    with job.writing():
        records = [job.state.by_id(record.id) or record for record in records]
        refused = [
            {"id": record.id, "status": "error", "error": "INVALID_LINEAR_PLACEMENT",
             "message": "keep_linear needs a position and a transform."}
            for record in records if record.position_mm is None or record.transform is None
        ]
        if refused:
            raise Refused("NOTHING_WRITTEN", results=refused)
        before = job.snapshot()
        for record in records:
            record.deformation = {"keep_linear": reason,
                                  "linear_key": deformation.linear_key(job.state, record)}
        job.commit(before)
    return KeptLinear(touched=[record.id for record in records], reason=reason)

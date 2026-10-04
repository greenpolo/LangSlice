"""The deformation on top of a section's linear placement: fit it, or keep the placement.

:func:`fit_deformable` runs one or more deformable fits per section
(:mod:`langslice.linear.deformation`: the fit grid, the image each fit
reads, the record cache and the engines) and, when the call has exactly one
setting, APPLIES each section's result as its deformation: one undo step
for the call. Several settings (candidates) are a preview: every fit runs
and is cached, nothing is written. :func:`keep_linear` records instead that
a section's linear placement stands without a deformation.

The arguments arrive checked (the door validates them and resolves each
candidate into a :class:`~langslice.linear.deformation.Choice`); what can go
wrong per section (no valid placement, ``start="current"`` with nothing to
compose onto, a trace that is missing, stale or failed, a fit that failed,
a record that cannot be saved) is that section's row, not a refusal of the
call. No pictures: the result carries the image each fit read and its
records, which the door draws.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from PIL import Image

from langslice.linear import deformation
from langslice.linear.state import SliceState
from langslice.ops.refusal import Refused

if TYPE_CHECKING:
    import numpy as np

    from langslice.linear.job import Job
    from langslice.linear.workspace import Workspace


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


def fit_deformable(
    job: Job,
    workspace: Workspace,
    records: list[SliceState],
    choices: list[deformation.Choice],
    *,
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
    start: str = "linear",
) -> DeformableFit:
    """Fit every choice on every section; with one choice, apply it.

    *include* / *exclude* are region entries (the engine's ``structures`` /
    ``exclude``, sides allowed); *start* is ``linear`` (from the linear
    placement) or ``current`` (composed onto the section's applied
    deformation). A traced choice waits for the section's trace still running
    (one :data:`~langslice.linear.deformation.TRACE_WAIT_S` deadline for the
    call; a trace that lands is checkpointed without an undo step).
    Identical inputs reuse a cached or saved result; applying a section's own
    current key again writes nothing (``written: false``).
    """
    state = job.state
    store = job.deformations
    applying = len(choices) == 1
    rows: list[dict[str, Any]] = []
    jobs: list[deformation.Job] = []
    traced: dict[str, tuple[Image.Image, Any]] = {}
    trace_deadline = time.monotonic() + deformation.TRACE_WAIT_S
    for record in records:
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
        running = False
        if any(choice.fit_section in deformation.TRACED for choice in choices):
            # A traced image waits for the section's trace still running.
            landed = record.id in job.image_jobs
            running = not job.wait_image_job(record.id, trace_deadline - time.monotonic())
            if landed and not running:
                job.checkpoint()
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
                        waited_s=deformation.TRACE_WAIT_S)
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

    before = job.snapshot()
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
                    folder=folder, key=fit.key, linear=linear, record=outcome,
                    choice=fit.choice, include=include, exclude=exclude, start=start,
                    previous=held, numbers=numbers,
                )
                row["written"] = True
                written.append(record.id)
            row["steps"] = len((record.deformation or {}).get("steps") or [])
        fitted.append(Fitted(row=row, fit=fit, outcome=outcome))
    if written:
        job.commit(before)
    return DeformableFit(applied=applying, rows=rows, fitted=fitted, traced=traced,
                         written=written)


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

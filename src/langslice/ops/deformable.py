"""The deformation on top of a section's linear placement.

:func:`fit_deformable` runs one deformable fit per section
(:mod:`langslice.core.deformation`: the fit grid, the image the fit reads,
the record cache and the engine) and APPLIES each section's result as its
deformation: one undo step for the call. Its callers are :func:`fireants_syn`
(FireANTs SyN on the preprocessed channel, building on whatever deformation
each section holds) and the landing of a packaged trace
(:func:`langslice.ops.traces.land_trace`: the traced borders fitted).

What can go wrong per section (no valid placement, an applied deformation
that cannot be read, a trace that is missing, stale or failed, a
fit that failed, a record that cannot be saved) is that section's row, not
a refusal of the call. With display options, :func:`pictures` draws each
fit and each traced section's trace through the core.

A deformation made elsewhere (a tool plugin's engine) is written the same
way: :func:`warp_from_map` builds its record from the full map,
:func:`apply_warps` writes it as the section's deformation, one undo step.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from langslice.core import deformation
from langslice.core.damage import exclusions
from langslice.core.handoff import digest
from langslice.core.state import SliceState
from langslice.ops.inputs import section_inputs, stale_row
from langslice.ops.refusal import Refused, frozen_row, unknown_sections

if TYPE_CHECKING:
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

    ``rows``: one per section, in call order: the settings, the engine
    numbers, the displacement summary, ``cached``, ``written`` and
    ``steps``; or ``status: error`` with the code. ``fitted``: the rows that
    hold a fit to draw, in the same order. ``traced``: per section whose
    settings read its trace, the image the fit read and the trace's lines on
    the fit grid. ``written``: the sections whose deformation changed.
    """

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


def traced_regions(
    record: SliceState, include: tuple[str, ...], exclude: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The regions a traced fit uses: the call's own, completed by the trace's.

    The model was shown only the trace's regions (``trace_borders`` include /
    exclude), so the atlas side drops the same ones: the call's *exclude*
    plus the trace's, and the call's *include*, else the trace's.
    """
    held = record.image_correction or {}
    traced_in = tuple(str(name) for name in held.get("include") or ())
    traced_out = tuple(str(name) for name in held.get("exclude") or ())
    return include or traced_in, tuple(dict.fromkeys((*exclude, *traced_out)))


def fit_regions(fit: deformation.Job) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(include, exclude)`` one fit ran with, as region entries."""
    return (tuple(str(name) for name in fit.settings.structures),
            tuple(str(name) for name in fit.settings.exclude))


def fit_deformable(
    job: Job,
    workspace: Workspace,
    records: list[SliceState],
    choice: deformation.Choice,
    *,
    restrict_to: tuple[str, ...] = (),
    options: DisplayOptions | None = None,
) -> DeformableFit:
    """Fit *choice* on every section and apply each result.

    *restrict_to* are region entries (the engine's ``structures``, sides
    allowed; empty: every region). Each section's marked regions are
    excluded on their own (:func:`langslice.core.damage.exclusions`).
    Each fit builds on the section's applied deformation where it holds one
    (composed onto it), else on its linear placement. A traced choice waits
    for the section's trace still running (one
    :data:`~langslice.core.deformation.TRACE_WAIT_S` deadline for the call;
    a trace that lands is checkpointed without an undo step) and adds the
    trace's own regions (:func:`traced_regions`; the row's ``trace_regions``).
    Identical inputs reuse a cached or saved result. With *options*,
    every fit and every traced section's trace is then drawn (:func:`pictures`).
    A section whose deformation is frozen is refused (its row ``FROZEN``).

    The fits run outside the job's write lock, from the state as it stood;
    applying takes the lock (:meth:`~langslice.job.job.Job.writing`) and
    refuses a section whose inputs changed meanwhile
    (:data:`langslice.ops.inputs.STALE_INPUT`, that section's row), applying
    the others.
    """
    state = job.state
    store = job.deformations
    rows: list[dict[str, Any]] = []
    jobs: list[deformation.Job] = []
    traced: dict[str, tuple[Image.Image, Any]] = {}
    trace_deadline = time.monotonic() + deformation.TRACE_WAIT_S
    traced_choice = choice.fit_section == deformation.TRACED_BORDERS
    expected: dict[str, str] = {}
    starts: dict[str, str] = {}
    for record in records:
        frozen = frozen_row(job, record.id, "deformation")
        if frozen is not None:
            rows.append(frozen)
            continue
        running = False
        if traced_choice:
            # A traced image waits for the section's trace still running; a
            # result lands under the job's lock, which may reload the state.
            running = not job.wait_image_job(record.id, trace_deadline - time.monotonic())
            record = state.by_id(record.id) or record
        held = record.deformation or {}
        try:
            current = store.current(state, record)
            unreadable = False
        except ValueError:  # its record.json does not load (the error names it)
            logger.warning("Applied deformation of %s unreadable", record.id, exc_info=True)
            current, unreadable = None, True
        if unreadable or (current is None and held
                          and held.get("linear_key") == deformation.linear_key(state, record)):
            rows.append({"id": record.id, "status": "error", "error": "DEFORMATION_UNREADABLE",
                         "message": "The section's applied deformation record cannot be read "
                         f"({held.get('record')}); it is not replaced by a fit from the "
                         "linear placement."})
            continue
        own_start = "current" if current is not None else "linear"
        starts[record.id] = own_start
        expected[record.id] = section_inputs(state, record, deformation=own_start == "current",
                                             trace=traced_choice)
        try:
            grid = deformation.fit_grid(state, workspace, record)
        except (ValueError, OSError) as exc:
            rows.append({"id": record.id, "status": "error",
                         "error": "INVALID_LINEAR_PLACEMENT", "message": str(exc)})
            continue
        previous = current
        previous_key = None if current is None else str((record.deformation or {}).get("key"))
        own = exclusions(record, tuple(restrict_to))
        failure = {"id": record.id, "status": "error", "settings": choice.echo()}
        regions = traced_regions(record, *own) if traced_choice else own
        try:
            image, identity = deformation.stain_image(workspace, state, grid, choice.fit_section)
            lines = None
            if traced_choice:
                lines, trace = deformation.traced_lines(
                    state, workspace, grid, running=running,
                    waited_s=deformation.TRACE_WAIT_S, root=job.folder)
                identity = {**identity, "trace": trace}
                traced.setdefault(record.id, (image, lines))
            settings = choice.settings(*regions, brain_scale=workspace.brain_scale)
        except deformation.FitRefusal as refusal:
            rows.append({**failure, **refusal.payload, "id": record.id})
            continue
        except (ValueError, OSError) as exc:
            rows.append({**failure, "error": "BAD_SETTINGS", "message": str(exc)})
            continue
        key = deformation.cache_key(state, grid, settings, identity, previous_key,
                                    workspace.brain_scale)
        cached = store.get(record.id, key)
        jobs.append(deformation.Job(
            grid=grid, choice=choice, settings=settings, key=key,
            image_identity=identity, previous=previous, image=image, lines=lines,
            result=cached, cached=cached is not None,
        ))
        inherited = regions != own
        rows.append({"id": record.id, "job": len(jobs) - 1,
                     **({"trace_regions": {"include": list(regions[0]),
                                           "exclude": list(regions[1])}}
                        if inherited else {})})
    deformation.run_jobs(workspace, jobs)

    with job.writing():
        before = job.snapshot()
        done = _apply(job, rows, jobs, expected=expected, starts=starts,
                      brain_scale=workspace.brain_scale)
        if done.written:
            job.commit(before)
    done = replace(done, traced=traced)
    if options is None:
        return done
    return pictures(workspace, done, options)


def _apply(
    job: Job, rows: list[dict[str, Any]], jobs: list[deformation.Job], *,
    expected: dict[str, str], starts: dict[str, str], brain_scale: float = 1.0,
) -> DeformableFit:
    """Every fit's row (its engine numbers at the atlas's *brain_scale*) and,
    under the job's lock, each section's result as its deformation when its
    inputs are unchanged."""
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
        start = starts[record.id]
        # The state may have been reloaded: the section as it is now.
        now = state.by_id(record.id)
        if now is None or section_inputs(
                state, now, deformation=start == "current",
                trace=fit.choice.fit_section == deformation.TRACED_BORDERS) != expected[record.id]:
            row.update(stale_row(record.id))
            continue
        record = now
        row.update(status="ok",
                   engine_settings=deformation.engine_settings(fit.settings, brain_scale),
                   **numbers, runtime_s=round(float(outcome.engine.get("runtime_s", 0.0)), 1),
                   cached=fit.cached)
        linear = deformation.linear_key(state, record)
        include, exclude = fit_regions(fit)
        outcome.provenance = deformation.provenance(
            fit.grid, fit.choice, include, exclude, start, fit.image_identity, linear)
        held = record.deformation or {}
        try:
            folder = store.save(record.id, fit.key, outcome)
        except OSError as exc:
            row.update(status="error", error="RECORD_WRITE_FAILED", message=str(exc))
            continue
        record.deformation = deformation.reference(
            folder=job.layout.relative(folder), key=fit.key, linear=linear, record=outcome,
            step={"start": start, **fit.choice.echo(), "include": list(include),
                  "exclude": list(exclude)},
            previous=held, numbers=numbers,
        )
        traced = (fit.image_identity or {}).get("trace") if isinstance(
            fit.image_identity, dict) else None
        if traced:
            # The trace this step fitted (its artifact directory):
            # a packaged trace called again on it reuses this step.
            record.deformation["steps"][-1]["trace"] = traced
        row["written"] = True
        written.append(record.id)
        row["steps"] = len((record.deformation or {}).get("steps") or [])
        fitted.append(Fitted(row=row, fit=fit, outcome=outcome))
    return DeformableFit(rows=rows, fitted=fitted, written=written)


@dataclass(frozen=True)
class MappedWarp:
    """A deformation made elsewhere, as a record not written yet
    (:func:`warp_from_map`): the section, the record and its key, the linear
    placement it sits on (``linear_key``), the step the section's reference
    will list, and the record's numbers (:func:`langslice.core.deformation.summary`)."""

    id: str
    key: str
    linear_key: str
    record: deformation.DeformableRecord
    step: dict[str, Any]
    numbers: dict[str, Any]


def warp_from_map(
    job: Job, workspace: Workspace, section: Any, native_xy: Any, *, method: str,
    step: dict[str, Any] | None = None, provenance: dict[str, Any] | None = None,
) -> MappedWarp:
    """The section's deformation from a full map made elsewhere, not written.

    *native_xy* is a float ``(H, W, 2)`` array: for each pixel centre of the
    section's fit grid (:func:`langslice.core.deformation.fit_grid`'s image,
    ``(W, H)`` its size), the native atlas-plane pixel ``(x, y)`` it
    corresponds to, linear placement and deformation together. It becomes a
    record as every applied deformation is stored: the residual over the
    section's current linear placement
    (:func:`langslice.core.deformation.record_from_map`), built on nothing the
    section held before (``start`` ``linear``). *method* names it (the
    record's engine and the step's ``method``); *step* adds fields to the
    step, *provenance* to the record's.

    Refused (nothing built): ``UNKNOWN_SLICE_IDS``; ``INVALID_LINEAR_PLACEMENT``
    (no position or no usable transform, as a fit refuses); ``BAD_FIELD`` (not
    a float array of the grid's shape with two channels, or not finite).
    """
    record = job.state.resolve(section)
    if record is None:
        raise unknown_sections(job.state, [str(section)])
    try:
        grid = deformation.fit_grid(job.state, workspace, record)
    except (ValueError, OSError) as exc:
        raise Refused("INVALID_LINEAR_PLACEMENT", message=f"{record.id}: {exc}") from exc
    width, height = grid.image.size
    array = np.asarray(native_xy)
    if not np.issubdtype(array.dtype, np.floating):
        raise Refused("BAD_FIELD", message=f"{record.id}: the map is a float array, not "
                      f"{array.dtype}.")
    if array.shape != (height, width, 2):
        raise Refused("BAD_FIELD", message=f"{record.id}: the map is (H, W, 2) = "
                      f"({height}, {width}, 2) on the fit grid; got {array.shape}.",
                      expected_shape=[height, width, 2])
    if not np.isfinite(array).all():
        raise Refused("BAD_FIELD", message=f"{record.id}: the map has values that are not "
                      "finite.")
    linear = deformation.linear_key(job.state, record)
    array = np.ascontiguousarray(array, dtype=np.float64)
    key = digest({"method": method, "linear": linear,
                  "map": hashlib.sha256(array.tobytes()).hexdigest()})
    outcome = deformation.record_from_map(
        workspace, grid, array, method=method,
        provenance={"tool": method, "linear_key": linear, **(provenance or {})})
    return MappedWarp(id=record.id, key=key, linear_key=linear, record=outcome,
                      step={"method": method, "start": "linear", **(step or {})},
                      numbers=deformation.summary(outcome, None))


@dataclass(frozen=True)
class WarpsApplied:
    """What :func:`apply_warps` did: one row per warp (``status`` ``ok`` with
    the record's numbers, or ``error`` with the code) and the sections written."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    written: list[str] = field(default_factory=list)


def apply_warps(job: Job, warps: Sequence[MappedWarp]) -> WarpsApplied:
    """Write each warp as its section's deformation: ONE undo step.

    Under the job's write lock, per warp: a section whose deformation is
    frozen is its row ``FROZEN``; one whose linear placement is no longer the
    one the warp was built on is ``STALE_INPUT``; otherwise the record is saved
    in the section's folder (``job.deformations``) and the section's
    ``deformation`` names it, as an applied fit's does, replacing what it held
    (``undo`` brings that back). A later change to the placement clears it as
    it clears any deformation.
    """
    store = job.deformations
    rows: list[dict[str, Any]] = []
    written: list[str] = []
    with job.writing():
        before = job.snapshot()
        for warp in warps:
            record = job.state.by_id(warp.id)
            if record is None:
                rows.append({"id": warp.id, "status": "error", "error": "UNKNOWN_SLICE_IDS"})
                continue
            frozen = frozen_row(job, record.id, "deformation")
            if frozen is not None:
                rows.append(frozen)
                continue
            if deformation.linear_key(job.state, record) != warp.linear_key:
                rows.append(stale_row(record.id))
                continue
            store.put(warp.key, warp.record)
            try:
                folder = store.save(record.id, warp.key, warp.record)
            except OSError as exc:
                rows.append({"id": record.id, "status": "error",
                             "error": "RECORD_WRITE_FAILED", "message": str(exc)})
                continue
            record.deformation = deformation.reference(
                folder=job.layout.relative(folder), key=warp.key, linear=warp.linear_key,
                record=warp.record, step=dict(warp.step), previous=None, numbers=warp.numbers)
            written.append(record.id)
            rows.append({"id": record.id, "status": "ok", **warp.numbers})
        if written:
            job.commit(before)
    return WarpsApplied(rows=rows, written=written)


def pictures(
    workspace: Workspace,
    done: DeformableFit,
    options: DisplayOptions,
) -> DeformableFit:
    """*done* with its pictures: per drawn row, the final atlas borders on the
    image the fit read (:func:`langslice.core.deformation.picture`), titled
    with the section, engine, stiffness and what was fitted. Then each
    traced section's trace on the section
    (:func:`~langslice.core.deformation.trace_picture`).
    Included or ``view.regions`` borders strong, excluded ones pink. Each
    drawn row gets ``image_indexes``; a picture that fails is listed in
    ``render_failed``.
    """
    import numpy as np

    from langslice.core import layers
    from langslice.core.canvas import (
        border_words,
        resolve_border_color,
        validate_border_thickness,
    )
    from langslice.core.deformable.render import drawn_border_coverage

    parts: list[Image.Image] = []
    failed: list[dict[str, str]] = []
    thickness = validate_border_thickness(options.border_thickness)
    # `color` is the image model's own fixed yellow (`trace_picture` below
    # draws it plain: its prompt says yellow and its reply is read back by
    # colour). Each fitted-atlas picture gets its cased core (`core`) per row
    # below, resolved against that row's own tissue -- never one for the whole
    # batch, since different sections in one call can need different ones.
    style = deformation.Style(
        zoom=() if options.full_view else tuple(options.zoom),
        highlight=tuple(name for name, _ids in options.regions), marked=(),
        outlines=options.layer, color=(255, 255, 0), thickness=thickness,
        atlas_opacity=options.atlas_opacity, depth=options.depth, long_edge=options.long_edge,
    )
    for fitted in done.fitted:
        row, fit, outcome = fitted.row, fitted.fit, fitted.outcome
        record = fit.grid.record
        include, exclude = fit_regions(fit)
        fit_style = replace(style, highlight=style.highlight or include, marked=exclude)
        coverage = drawn_border_coverage(
            outcome, workspace.atlas, highlight=fit_style.highlight, marked=fit_style.marked,
            warped=True, outlines=fit_style.outlines, depth=fit_style.depth,
            width_px=fit_style.thickness,
        )
        core, _ = resolve_border_color(
            options.border_color, options.border_thickness,
            rgb=np.asarray(fit.image.convert("RGB")), coverage=coverage,
        )
        fit_style = replace(fit_style, core=core)
        heading = f"{record.id}  applied: {outcome.engine['name']} {fit.choice.stiffness}"
        start = "current" if fit.previous is not None else "linear"
        detail_line = (f"{fit.choice.fit_section} vs {fit.choice.fit_atlas}, start {start}"
                       + (f", include {','.join(include)}" if include else "")
                       + (f", exclude {','.join(exclude)}" if exclude else ""))
        shown_atlas = options.atlas_images
        first = len(parts)
        try:
            parts.append(deformation.picture(
                workspace, fit.image, outcome, warped=True, style=fit_style,
                atlas_images=shown_atlas, title=f"{heading}\n{detail_line}",
                note={"sections": (record.id,), "mode": options.mode,
                      "caption": fit_caption(record, fit, row, include, exclude,
                                             border_words(core))}))
        except Exception as exc:
            logger.warning("fit_deformable picture failed for %s", record.id, exc_info=True)
            del parts[first:]
            failed.append({"id": record.id, "message": str(exc)})
            continue
        row["image_indexes"] = list(range(first, len(parts)))
    # Each traced section's trace, once per call, so it can be reviewed.
    traces: list[dict[str, Any]] = []
    where = {fitted.fit.grid.record.id: fitted.fit.grid.record for fitted in done.fitted}
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
        held = where.get(section_id)
        at = (f", at {float(held.position_mm):.2f} mm"
              if held is not None and held.position_mm is not None else "")
        parts.append(layers.note(
            picture, sections=(section_id,), mode="trace",
            caption=f"{section_id} trace_borders: the image model's traced lines drawn on "
            f"the section{at}"))
    return replace(done, pictures=parts, traces=traces, render_failed=failed)


#: How a fit picture's caption names what of the section a fit read.
_FIT_SECTION_WORDS = {
    deformation.FIT_LOOK: "the preprocessed channel",
    deformation.TRACED_BORDERS: "the image model's traced borders",
}


def fit_caption(record: SliceState, fit: deformation.Job, row: dict[str, Any],
                include: Sequence[str], exclude: Sequence[str],
                border_word: str = "") -> str:
    """The index caption of a deformable fit's picture: the section, the
    fit (engine, stiffness, what it read against which atlas image), the
    position, the regions drawn thick or left out, the colour the fitted
    borders are drawn in, and the fit's numbers."""
    choice = fit.choice
    read = _FIT_SECTION_WORDS.get(choice.fit_section, choice.fit_section)
    at = (f", at {float(record.position_mm):.2f} mm"
          if record.position_mm is not None else "")
    numbers = row.get("displacement_mm") or {}
    facts = (f"displacement median {numbers.get('median')} mm, max {numbers.get('max')} mm; "
             f"fold fraction {row.get('fold_fraction')}") if numbers else ""
    engine = getattr(fit.result, "engine", {}).get("name")
    return (f"{record.id} deformation fit ({engine} {choice.stiffness}, {read} "
            f"against the atlas {choice.fit_atlas}){at}; the fitted atlas borders "
            + (f"({border_word}) " if border_word else "")
            + "on the image the fit read"
            + (f"; regions {', '.join(include)} drawn thick, the other borders faint"
               if include else "")
            + (f"; left out (second colour): {', '.join(exclude)}" if exclude else "")
            + (f"; {facts}" if facts else ""))


#: Stiffness levels (``core.deformable.settings.Stiffness``).
STIFFNESSES = ("soft", "medium", "firm")
#: Sections one :func:`fireants_syn` call takes.
MAX_FIREANTS_SYN_SECTIONS = 4


def engine_ready() -> bool:
    """Whether the deformable engine (FireANTs on torch) imports here
    (``ENGINE_MISSING`` when not)."""
    if not deformation.engine_available():
        return False
    from langslice.core.deformable.engines import import_fireants

    try:
        import_fireants()
    except Exception:  # an install that does not load is as missing as none
        logger.warning("FireANTs is installed but does not import", exc_info=True)
        return False
    return True


def refuse_without_engine() -> None:
    """Refuse ``ENGINE_MISSING`` (nothing done) when the deformable engine does
    not import."""
    if not engine_ready():
        raise Refused("ENGINE_MISSING", message=deformation.ENGINE_MISSING + ".")


def region_entries(workspace: Workspace, state: Any, entries: Any) -> tuple[str, ...]:
    """*entries* (region acronyms, names or ids, ``"CTX:left"`` for one side)
    checked against the atlas and normalized. Refused: ``BAD_ARGS`` (not a
    list of names, or a side that does not exist), ``UNKNOWN_REGIONS``,
    ``NO_SIDES`` (a side on a sagittal stack)."""
    from langslice.core.atlas.sides import has_sides
    from langslice.core.damage import normalized_entries
    from langslice.core.deformable.plane_images import resolve_entries

    if isinstance(entries, str) or not isinstance(entries, (list, tuple)):
        raise Refused("BAD_ARGS", message="restrict_to is a list of atlas regions.")
    try:
        names = normalized_entries(str(entry) for entry in entries)
    except ValueError as exc:
        raise Refused("BAD_ARGS", message=str(exc)) from exc
    if names:
        try:
            resolve_entries(workspace.atlas, names)
        except ValueError as exc:
            raise Refused("UNKNOWN_REGIONS", message=str(exc)) from exc
        if state.plane == "sagittal" and has_sides(names):
            raise Refused("NO_SIDES", message="A sagittal section lies within one "
                          "hemisphere, so a region cannot be limited to one side.")
    return tuple(names)


def fireants_syn(
    job: Job,
    workspace: Workspace,
    sections: list[Any],
    *,
    restrict_to: Any = (),
    atlas_image: str = "template",
    stiffness: str = "medium",
) -> DeformableFit:
    """A deformable fit with FireANTs SyN on each section, applied: ONE undo step.

    One choice, no candidates: the section's preprocessed channel
    (:mod:`langslice.core.appearance`) against *atlas_image* (one of the atlas's
    images, :func:`langslice.core.atlas.images.atlas_images`), at *stiffness*
    (``soft`` / ``medium`` / ``firm``), on top of each section's applied
    deformation where it holds one, else its linear placement
    (undo is how to start over). *restrict_to*: warp by
    these regions only (descendants included, ``"CTX:left"`` for one side;
    empty: every region); each section's marked regions are left out on
    their own (:func:`langslice.core.damage.exclusions`). The fit runs as
    :func:`fit_deformable` runs: computed outside the job's write lock,
    applied under it, a section whose inputs changed meanwhile its row
    ``STALE_INPUT``; per-section problems are rows. The tool door draws the
    picture of what was written (:func:`langslice.ops.look.show_result`).

    Refused (nothing done): ``ENGINE_MISSING``; ``BAD_ARGS`` (no sections,
    an image the atlas does not
    have, an unknown stiffness, a bad region entry); ``UNKNOWN_SLICE_IDS``;
    ``UNKNOWN_REGIONS``; ``NO_SIDES``.
    """
    from langslice.core.atlas.images import atlas_images

    refs = list(sections or [])
    if not refs:
        raise Refused("BAD_ARGS", message="Name the sections to fit.")
    kind = str(atlas_image or "template").strip()
    offered = atlas_images(workspace.atlas)
    if kind not in offered:
        raise Refused("BAD_ARGS", message=f"atlas_image is one of this atlas's images: "
                      f"{list(offered)}.", atlas_image=list(offered))
    level = str(stiffness or "medium").strip().lower()
    if level not in STIFFNESSES:
        raise Refused("BAD_ARGS", message=f"stiffness is one of {list(STIFFNESSES)}.")
    refuse_without_engine()
    regions = region_entries(workspace, job.state, restrict_to)
    records: list[SliceState] = []
    unknown: list[str] = []
    for ref in refs:
        record = job.state.resolve(ref)
        if record is None:
            unknown.append(str(ref))
        elif all(held.id != record.id for held in records):
            records.append(record)
    if unknown:
        raise unknown_sections(job.state, unknown)
    choice = deformation.Choice(fit_section=deformation.FIT_LOOK, fit_atlas=kind,
                                stiffness=level)
    return fit_deformable(job, workspace, records, choice, restrict_to=regions)

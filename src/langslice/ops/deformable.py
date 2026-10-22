"""The deformation on top of a section's linear placement.

:func:`fit_deformable` runs one deformable fit per section
(:mod:`langslice.core.deformation`: the fit grid, the image the fit reads,
the record cache and the engine) and APPLIES each section's result as its
deformation: one undo step for the call. Its callers are :func:`fireants_syn`
(FireANTs SyN on a section image, building on whatever deformation
each section holds) and the landing of a packaged trace
(:func:`langslice.ops.traces.land_trace`: the traced borders fitted).

What can go wrong per section (no valid placement, an applied deformation
that cannot be read, a trace that is missing, stale or failed, a
fit that failed, a record that cannot be saved) is that section's row, not
a refusal of the call. With display options, :func:`pictures` draws each
fit and each traced section's trace through the core.

A deformation made elsewhere (a tool plugin's engine, a script's) is written
the same way: :func:`warp_from_map` builds its record from the full map,
:func:`apply_warps` writes it as the section's deformation, one undo step.
:func:`fit_record` runs the deformable engine once with settings of the
caller's own and writes nothing (a script's ``job.context.fit_fireants``).
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from langslice.core import deformation
from langslice.core import section_images as images_
from langslice.core.damage import exclusions
from langslice.core.engine_inputs import (
    NATIVE_BRIGHTNESS,
    EstimatedBlock,
    InputsBlock,
    damage_block,
    mask_block,
)
from langslice.core.engine_output import engine_output, failure_message
from langslice.core.handoff import digest
from langslice.core.state import SliceState
from langslice.ops import masks as ops_masks
from langslice.ops.images import check_image
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
    #: The call's own settings, as the reply states them (:func:`fireants_syn`).
    settings: dict[str, Any] = field(default_factory=dict)


# --- what a fit reads and estimated, named (the reply's and the record's blocks) --------


def damage_margin_mm(workspace: Workspace) -> float:
    """The margin every native fit leaves around the marked damage regions, in
    millimetres at the atlas's brain size."""
    from langslice.core.deformable.masks import EXCLUSION_MARGIN_MM

    return float(EXCLUSION_MARGIN_MM * workspace.brain_scale)


def inputs_block(
    state: Any, workspace: Workspace, record: SliceState, *, section_image: str,
    atlas_image: str, mask: str, damage: Sequence[str], restrict_to: Sequence[str] = (),
    brightness: str = NATIVE_BRIGHTNESS, resolved: str | None = None,
) -> InputsBlock:
    """What one native fit of *record* read, named: the section image (name,
    kind, file, digest), the atlas image as asked and as resolved, the section mask
    (``"none"`` or its name, kind and staleness), the atlas-side damage and
    the regions the fit was restricted to."""
    from langslice.core.atlas.images import nissl_note, resolve_image

    entry = (record.masks or {}).get(mask) if mask != ops_masks.NO_MASK else None
    held = mask_block()
    if mask != ops_masks.NO_MASK:
        held = mask_block(mask, (entry or {}).get("kind"), False,
                          ops_masks.made_earlier(state, record, mask))
    return InputsBlock(
        section_image=images_.read_image(workspace, record, section_image).payload(),
        atlas_image=atlas_image,
        atlas_image_resolved=resolved or resolve_image(workspace.atlas, atlas_image),
        mask=held, damage=damage_block(list(damage), damage_margin_mm(workspace)),
        restrict_to=list(restrict_to), brightness=brightness,
        atlas_image_note=nissl_note(workspace.atlas, atlas_image))


def estimated_block(record: deformation.DeformableRecord) -> EstimatedBlock | None:
    """What the engine estimated for itself on *record*: its own sorting of the
    section (saved with the deformation) and the fraction of the placed footprint
    it left out; None when it estimated nothing."""
    weights = getattr(record, "weights", None)
    if weights is None:
        return None
    footprint = np.asarray(record.footprint, dtype=bool)
    sorted_out = np.asarray(weights, dtype=np.float32) < 0.5
    inside = sorted_out[footprint] if footprint.shape == sorted_out.shape and footprint.any() \
        else sorted_out.ravel()
    return EstimatedBlock(sorting="saved with the deformation; make_mask source fit",
                          left_out_fraction=round(float(inside.mean()), 4) if inside.size else 0.0)


def _resolved(workspace: Workspace, atlas_image: str) -> str:
    from langslice.core.atlas.images import resolve_image

    return resolve_image(workspace.atlas, atlas_image) if atlas_image not in (
        deformation.FIT_BORDERS,) else atlas_image


def default_settings(workspace: Workspace) -> Any:
    """``fireants_syn``'s own settings (a :class:`~langslice.core.deformable.settings.FitSettings`:
    Nissl, 80 um smoothness, mutual information and edges, standard detail),
    the base a script's ``fit_fireants`` changes."""
    return deformation.Choice(
        fit_section=images_.PREPROCESSED, fit_atlas="nissl").settings(
        (), (), brain_scale=workspace.brain_scale)


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


#: A fit's row names its wait for the machine's GPU queue from this many seconds.
GPU_WAIT_SHOWN_S = 1.0


def gpu_rows(engine: Mapping[str, Any]) -> dict[str, Any]:
    """What a fit's row says of the machine's GPU queue, from its record's
    ``engine``: ``gpu_wait_s`` when it waited for its turn
    (:data:`GPU_WAIT_SHOWN_S` or more), ``retried`` in words when the GPU ran
    out of memory and the fit ran again."""
    out: dict[str, Any] = {}
    waited = float(engine.get("gpu_wait_s") or 0.0)
    if waited >= GPU_WAIT_SHOWN_S:
        out["gpu_wait_s"] = round(waited)
    retries = int(engine.get("oom_retries") or 0)
    if retries:
        times = "once" if retries == 1 else f"{retries} times"
        out["retried"] = (f"The GPU ran out of memory {times} (another process held it); "
                          "the fit waited and ran again from the start, so this result is "
                          "whole.")
    return out


def fit_deformable(
    job: Job,
    workspace: Workspace,
    records: list[SliceState],
    choice: deformation.Choice,
    *,
    restrict_to: tuple[str, ...] = (),
    options: DisplayOptions | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> DeformableFit:
    """Fit *choice* on every section and apply each result.

    *restrict_to* are region entries (the engine's ``structures``, sides
    allowed; empty: every region). Each section's marked regions are
    excluded on their own (:func:`langslice.core.damage.exclusions`).
    *overrides* are :class:`~langslice.core.deformable.settings.FitSettings`
    fields set over the choice's own (``stain_metric``, ``stain_edges``,
    ``detail``). ``choice.fit_section`` is the section image the fit reads (a raw
    channel or a made image, by name; ``trace_borders`` fits the traced lines) and
    ``choice.mask`` the section mask by name (:mod:`langslice.ops.masks`): a
    stale mask is that section's row (``STALE_MASK``). Every ok row carries
    the ``inputs`` the fit read and what it ``estimated``
    (:mod:`langslice.core.engine_inputs`).
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
    blocks: dict[int, InputsBlock] = {}
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
        failure = {"id": record.id, "status": "error", "settings": choice.tool_settings()}
        regions = traced_regions(record, *own) if traced_choice else own
        try:
            weights = ops_masks.section_weights(job, record, choice.mask)
        except Refused as refusal:
            rows.append({**failure, **refusal.payload()})
            continue
        try:
            image, identity = deformation.section_image(
                workspace, state, grid,
                images_.PREPROCESSED if traced_choice else choice.fit_section)
            lines = None
            if traced_choice:
                lines, trace = deformation.traced_lines(
                    state, workspace, grid, running=running,
                    waited_s=deformation.TRACE_WAIT_S, root=job.folder)
                identity = {**identity, "trace": trace}
                traced.setdefault(record.id, (image, lines))
            settings = choice.settings(*regions, brain_scale=workspace.brain_scale)
            if overrides:
                settings = replace(settings, **dict(overrides))
        except deformation.FitRefusal as refusal:
            rows.append({**failure, **refusal.payload, "id": record.id})
            continue
        except (ValueError, OSError) as exc:
            rows.append({**failure, "error": "BAD_SETTINGS", "message": str(exc)})
            continue
        held_mask = (record.masks or {}).get(choice.mask)
        key = deformation.cache_key(
            state, grid, settings, identity, previous_key, workspace.brain_scale,
            mask_identity=None if held_mask is None else {
                "file": held_mask.get("file"), "kind": held_mask.get("kind")})
        cached = store.get(record.id, key)
        jobs.append(deformation.Job(
            grid=grid, choice=choice, settings=settings, key=key,
            image_identity=identity, previous=previous, image=image, lines=lines,
            result=cached, cached=cached is not None, section_weights=weights,
        ))
        if not traced_choice:
            blocks[len(jobs) - 1] = inputs_block(
                state, workspace, record, section_image=choice.fit_section,
                atlas_image=choice.fit_atlas, mask=choice.mask,
                damage=[str(name) for name in own[1]],
                restrict_to=[str(name) for name in restrict_to],
                resolved=_resolved(workspace, choice.fit_atlas))
        inherited = regions != own
        rows.append({"id": record.id, "job": len(jobs) - 1,
                     **({"trace_regions": {"include": list(regions[0]),
                                           "exclude": list(regions[1])}}
                        if inherited else {})})
    with engine_output(job.engine_log, f"deformable fits of {len(jobs)} section(s)"):
        deformation.run_jobs(workspace, jobs)

    with job.writing():
        before = job.snapshot()
        done = _apply(job, rows, jobs, expected=expected, starts=starts, blocks=blocks,
                      brain_scale=workspace.brain_scale)
        if done.written:
            job.commit(before)
    done = replace(done, traced=traced)
    if options is None:
        return done
    return pictures(workspace, done, options)


def _apply(
    job: Job, rows: list[dict[str, Any]], jobs: list[deformation.Job], *,
    expected: dict[str, str], starts: dict[str, str], blocks: Mapping[int, InputsBlock],
    brain_scale: float = 1.0,
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
        block = blocks.get(index)
        row["settings"] = fit.choice.tool_settings()
        if not isinstance(outcome, deformation.DeformableRecord):
            row.update(status="error", error="FIT_FAILED", message=failure_message(
                str(getattr(outcome, "error", "no result")), job.engine_log,
                f"the fit of {fit.grid.record.id} failed"))
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
        if not fit.cached:
            row.update(gpu_rows(outcome.engine))
        linear = deformation.linear_key(state, record)
        include, exclude = fit_regions(fit)
        outcome.provenance = deformation.provenance(
            fit.grid, fit.choice, include, exclude, start, fit.image_identity, linear)
        if block is not None:
            estimated = estimated_block(outcome)
            row["inputs"] = block.payload()
            row["estimated"] = None if estimated is None else estimated.payload()
            # The record's provenance holds the same two blocks; what the
            # fit's choice and image were moves to ``fit``.
            outcome.provenance["fit"] = outcome.provenance.pop("inputs", None)
            outcome.provenance["inputs"] = row["inputs"]
            outcome.provenance["estimated"] = row["estimated"]
        held = record.deformation or {}
        try:
            folder = store.save(record.id, fit.key, outcome)
        except OSError as exc:
            row.update(status="error", error="RECORD_WRITE_FAILED", message=str(exc))
            continue
        traced_fit = fit.choice.fit_section == deformation.TRACED_BORDERS
        record.deformation = deformation.reference(
            folder=job.layout.relative(folder), key=fit.key, linear=linear, record=outcome,
            step={"method": "trace_borders" if traced_fit else "fireants_syn",
                  "start": start, **fit.choice.echo(),
                  # the tool's settings the choice does not hold (a step saved
                  # before they were recorded has none)
                  **({} if traced_fit else {"metric": fit.settings.stain_metric,
                                            "edges": bool(fit.settings.stain_edges),
                                            "detail": fit.settings.detail}),
                  "include": list(include), "exclude": list(exclude)},
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
    #: The key of the deformation it was built on (``start`` ``current``), which
    #: the section must still hold when it is applied; None: built on the
    #: linear placement.
    parent_key: str | None = None


def warp_from_map(
    job: Job, workspace: Workspace, section: Any, native_xy: Any, *, method: str,
    step: dict[str, Any] | None = None, provenance: dict[str, Any] | None = None,
    weights: Any = None, start: str = "linear", settings: Mapping[str, Any] | None = None,
    engine_settings: Mapping[str, Any] | None = None, engine_version: str = "",
) -> MappedWarp:
    """The section's deformation from a full map made elsewhere, not written.

    *native_xy* is a float ``(H, W, 2)`` array: for each pixel centre of the
    section's fit grid (:func:`langslice.core.deformation.fit_grid`'s image,
    ``(W, H)`` its size), the native atlas-plane pixel ``(x, y)`` it
    corresponds to, linear placement and deformation together. It becomes a
    record as every applied deformation is stored: the residual over the
    section's current linear placement
    (:func:`langslice.core.deformation.record_from_map`). *start*
    ``linear``: built on nothing the section held before; ``current``: the
    map was made on top of the deformation the section holds, so that record
    is its parent and the section's steps are kept before its own, as a
    native fit chains (the map is still the whole registration). *method*
    names it (the record's engine and the step's ``method``); *settings*
    (the call's settings, by the tool's names) are the step's ``settings``;
    *engine_settings* (what the engine ran at, in its own units) and
    *engine_version* are the record's ``engine.native_parameters`` and
    ``engine.version``, as a FireANTs record holds them; *step* adds fields to
    the step, *provenance* to the record's. *weights* is the weighting the
    engine estimated for itself, a float ``(H, W)`` array on the fit grid in
    [0, 1] (stored as float16 with the record; ``make_mask`` source ``fit``
    copies it); None when it estimated none.

    Refused (nothing built): ``UNKNOWN_SLICE_IDS``; ``INVALID_LINEAR_PLACEMENT``
    (no position or no usable transform, as a fit refuses); ``BAD_FIELD`` (not
    a float array of the grid's shape with two channels, or not finite;
    *weights* not a finite ``(H, W)`` array); ``BAD_ARGS`` (a *start* that is
    neither, ``current`` on a section that holds no deformation);
    ``DEFORMATION_UNREADABLE``.
    """
    record = job.state.resolve(section)
    if record is None:
        raise unknown_sections(job.state, [str(section)])
    if start not in ("linear", "current"):
        raise Refused("BAD_ARGS", message="start is 'linear' or 'current'.")
    try:
        grid = deformation.fit_grid(job.state, workspace, record)
    except (ValueError, OSError) as exc:
        raise Refused("INVALID_LINEAR_PLACEMENT", message=f"{record.id}: {exc}") from exc
    previous: deformation.DeformableRecord | None = None
    if start == "current":
        try:
            previous = job.deformations.current(job.state, record)
        except ValueError as exc:
            raise Refused("DEFORMATION_UNREADABLE", message=f"{record.id}: {exc}") from exc
        if previous is None:
            raise Refused("BAD_ARGS", message=f"{record.id} holds no deformation to build on: "
                          "start is 'linear'.")
    parent_key = (str((record.deformation or {}).get("key")) if previous is not None
                  else None)
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
    stored_weights = None
    if weights is not None:
        held = np.asarray(weights)
        if held.shape != (height, width) or not np.isfinite(held.astype(np.float64)).all():
            raise Refused("BAD_FIELD", message=f"{record.id}: weights are a finite (H, W) = "
                          f"({height}, {width}) array on the fit grid; got {held.shape}.",
                          expected_shape=[height, width])
        stored_weights = np.clip(held.astype(np.float32), 0.0, 1.0).astype(np.float16)
    linear = deformation.linear_key(job.state, record)
    array = np.ascontiguousarray(array, dtype=np.float64)
    key = digest({"method": method, "linear": linear,
                  "map": hashlib.sha256(array.tobytes()).hexdigest(),
                  **({"parent": parent_key} if parent_key else {}),
                  **({"weights": hashlib.sha256(stored_weights.tobytes()).hexdigest()}
                     if stored_weights is not None else {})})
    outcome = deformation.record_from_map(
        workspace, grid, array, method=method,
        provenance={"tool": method, "linear_key": linear, "start": start,
                    **({"settings": dict(settings)} if settings else {}),
                    **(provenance or {})},
        native_parameters=dict(engine_settings or {}), engine_version=str(engine_version))
    outcome.weights = stored_weights
    if previous is not None:
        outcome.parent = previous
        outcome.step = previous.step + 1
    return MappedWarp(id=record.id, key=key, linear_key=linear, record=outcome,
                      step={"method": method, "start": start, **(step or {}),
                            **({"settings": dict(settings)} if settings else {})},
                      numbers=deformation.summary(outcome, previous), parent_key=parent_key)


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
    one the warp was built on, or (a warp built on the section's deformation,
    ``parent_key``) that no longer holds that deformation, is ``STALE_INPUT``;
    otherwise the record is saved in the section's folder (``job.deformations``)
    and the section's ``deformation`` names it, as an applied fit's does: a
    warp built on the linear placement replaces what the section held, one
    built on its deformation keeps that deformation's steps before its own
    (``undo`` brings the earlier one back). A later change to the placement
    clears it as it clears any deformation.
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
            held = record.deformation or {}
            if deformation.linear_key(job.state, record) != warp.linear_key or (
                    warp.parent_key is not None and held.get("key") != warp.parent_key):
                rows.append(stale_row(record.id))
                continue
            store.put(warp.key, warp.record)
            try:
                folder = store.save(record.id, warp.key, warp.record)
            except (OSError, ValueError) as exc:
                rows.append({"id": record.id, "status": "error",
                             "error": "RECORD_WRITE_FAILED", "message": str(exc)})
                continue
            record.deformation = deformation.reference(
                folder=job.layout.relative(folder), key=warp.key, linear=warp.linear_key,
                record=warp.record, step=dict(warp.step),
                previous=held if warp.parent_key is not None else None, numbers=warp.numbers)
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
    with the section, engine, smoothness and what was fitted. Then each
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
        heading = (f"{record.id}  applied: {outcome.engine['name']} "
                   f"smoothness {fit.choice.smoothness_um:g} um")
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
    images_.PREPROCESSED: "the preprocessed image",
    deformation.TRACED_BORDERS: "the image model's traced borders",
}


def fit_caption(record: SliceState, fit: deformation.Job, row: dict[str, Any],
                include: Sequence[str], exclude: Sequence[str],
                border_word: str = "") -> str:
    """The index caption of a deformable fit's picture: the section, the
    fit (engine, smoothness, what it read against which atlas image), the
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
    return (f"{record.id} deformation fit ({engine} smoothness {choice.smoothness_um:g} um, "
            f"{read} against the atlas {choice.fit_atlas}){at}; the fitted atlas borders "
            + (f"({border_word}) " if border_word else "")
            + "on the image the fit read"
            + (f"; regions {', '.join(include)} drawn thick, the other borders faint"
               if include else "")
            + (f"; left out (second colour): {', '.join(exclude)}" if exclude else "")
            + (f"; {facts}" if facts else ""))


#: The metrics and detail levels ``fireants_syn`` takes.
METRICS = ("mutual_information", "local_correlation")
DETAILS = ("coarse", "standard", "fine")
#: Sections one :func:`fireants_syn` call takes: the fits run one after
#: another on the GPU, but every fitted section is held in memory until the
#: call writes them as one undo step (measured on 1536-pixel fit grids: about
#: 3.2 GB for the process, then 170 MB a section; 20 sections peaked at 6.4 GB).
MAX_FIREANTS_SYN_SECTIONS = 20
#: Why a call of more sections is refused (``TOO_MANY_SECTIONS``'s message).
SECTION_CAP_REASON = ("every fitted section is held in memory (about 170 MB) until the call "
                      "writes them all as one undo step")


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


def fit_record(
    job: Job, workspace: Workspace, section: Any, settings: Any, *,
    start: str = "linear", mask: str = "none", marked: bool = True,
    section_image: str = images_.PREPROCESSED,
) -> deformation.DeformableRecord:
    """One FireANTs SyN fit of the section with *settings* (a
    :class:`~langslice.core.deformable.settings.FitSettings`: a stain fit, any
    smoothness, detail, atlas image, metric, edge channel, regions), written
    nowhere: the record (``field_mm`` the residual over the section's linear
    placement, composed onto its applied deformation with *start*
    ``current``). Its ``provenance`` holds the fit's ``inputs`` and
    ``estimated`` blocks (:mod:`langslice.core.engine_inputs`).

    *mask* is ``"none"`` or the name of one of the section's masks
    (:mod:`langslice.ops.masks`): its weights, thresholded at 0.5, narrow the
    section mask (:func:`langslice.core.deformable.fit.prepare_fit`);
    *marked* adds the section's marked damage regions to the settings'
    ``exclude``, as every native fit leaves them out.

    Refused: ``UNKNOWN_SLICE_IDS``, ``ENGINE_MISSING``, ``BAD_ARGS`` (settings
    that are not a FireANTs stain fit, a start that is neither, ``current``
    without a deformation), ``INVALID_LINEAR_PLACEMENT``, ``UNKNOWN_MASK``,
    ``STALE_MASK``, ``FIT_FAILED`` (the engine or the masks failed, the
    message says how).
    """
    from langslice.core.deformable import FitSettings
    from langslice.core.deformable.fit import prepare_fit, run_prepared

    record = job.state.resolve(section)
    if record is None:
        raise unknown_sections(job.state, [str(section)])
    if not isinstance(settings, FitSettings) or settings.engine != "fireants" or (
            settings.section_image != "stain" or settings.labels != "none"):
        raise Refused("BAD_ARGS", message="settings are a FireANTs stain fit's FitSettings "
                      "(engine fireants, section_image stain, labels none).")
    if start not in ("linear", "current"):
        raise Refused("BAD_ARGS", message="start is 'linear' or 'current'.")
    refuse_without_engine()
    label = str(mask or ops_masks.NO_MASK).strip()
    weights = ops_masks.section_weights(job, record, label)
    try:
        grid = deformation.fit_grid(job.state, workspace, record)
    except (ValueError, OSError) as exc:
        raise Refused("INVALID_LINEAR_PLACEMENT", message=f"{record.id}: {exc}") from exc
    previous = None
    if start == "current":
        previous = job.deformations.current(job.state, record)
        if previous is None:
            raise Refused("BAD_ARGS", message=f"{record.id} has no deformation to start from.")
    image_name = check_image(job, workspace, [record], section_image)
    _kept, dropped = exclusions(record, tuple(str(name) for name in settings.structures))
    if marked:
        settings = replace(settings, exclude=tuple(dict.fromkeys(
            (*settings.exclude, *dropped))))
    try:
        image, _look = deformation.section_image(workspace, job.state, grid, image_name)
        prepared = prepare_fit(image, workspace.atlas, grid.placement, settings,
                               previous=previous, section_weights=weights)
        with deformation.engine_lock(), engine_output(job.engine_log,
                                                      f"FireANTs fit of {record.id}"):
            outcome = run_prepared(prepared)
    except Exception as exc:  # noqa: BLE001 - the engine's failure is the caller's refusal
        raise Refused("FIT_FAILED", message=f"{record.id}: " + failure_message(
            f"{type(exc).__name__}: {exc}", job.engine_log,
            f"the FireANTs fit of {record.id} failed")) from exc
    block = inputs_block(
        job.state, workspace, record, section_image=image_name,
        atlas_image=str(settings.atlas_image), mask=label,
        damage=[str(name) for name in dropped] if marked else [],
        restrict_to=[str(name) for name in settings.structures],
        resolved=_resolved(workspace, str(settings.atlas_image)))
    estimated = estimated_block(outcome)
    outcome.provenance = {**outcome.provenance, "inputs": block.payload(),
                          "estimated": None if estimated is None else estimated.payload()}
    return outcome


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
    section_image: str = "preprocessed",
    atlas_image: str = "nissl",
    mask: str = "none",
    smoothness_um: float = 80.0,
    metric: str = "mutual_information",
    edges: bool = True,
    detail: str = "standard",
) -> DeformableFit:
    """A deformable fit with FireANTs SyN on each section, applied: ONE undo step.

    One choice, no candidates: the section image *section_image* (a raw
    channel, ``preprocessed`` or another made image,
    :func:`langslice.ops.images.image_names`) against *atlas_image*
    (one of the atlas's images, :func:`langslice.core.atlas.images.atlas_images`;
    ``nissl`` is its Nissl image where it has one, else its template), with
    the warp's smoothness *smoothness_um* in micrometres, the image *metric*
    (``mutual_information`` / ``local_correlation``), the *edges* channel pair
    and *detail* (``coarse`` / ``standard`` / ``fine``), on top of each
    section's applied deformation where it holds one, else its linear
    placement (undo is how to start over). *mask* is ``"none"`` or the name of a
    mask each section holds (:func:`langslice.ops.masks.make_mask`).
    *restrict_to*: warp by
    these regions only (descendants included, ``"CTX:left"`` for one side;
    empty: every region); each section's marked regions are left out on
    their own (:func:`langslice.core.damage.exclusions`). The fit runs as
    :func:`fit_deformable` runs: computed outside the job's write lock,
    applied under it, a section whose inputs changed meanwhile its row
    ``STALE_INPUT``; per-section problems are rows. The result's ``settings``
    is the call as it was made, each ok row's ``inputs`` and ``estimated``
    what the fit read and worked out itself. The tool door draws the
    picture of what was written (:func:`langslice.ops.look.show_result`).

    Refused (nothing done): ``ENGINE_MISSING``; ``BAD_ARGS`` (no sections,
    an image the atlas does not have, a metric, detail or smoothness that is
    not one of the allowed, a bad region entry); ``UNKNOWN_SLICE_IDS``;
    ``UNKNOWN_IMAGE``; ``UNKNOWN_MASK`` (a section without that mask);
    ``UNKNOWN_REGIONS``; ``NO_SIDES``.
    """
    from langslice.core.atlas.images import resolve_image
    from langslice.core.deformable.settings import SMOOTHNESS_BOUNDS_UM

    refs = list(sections or [])
    if not refs:
        raise Refused("BAD_ARGS", message="Name the sections to fit.")
    kind = str(atlas_image or "nissl").strip()
    try:
        resolve_image(workspace.atlas, kind)
    except ValueError as exc:
        raise Refused("BAD_ARGS", message=str(exc)) from exc
    metric_name = str(metric or "mutual_information").strip()
    if metric_name not in METRICS:
        raise Refused("BAD_ARGS", message=f"metric is one of {list(METRICS)}.")
    level = str(detail or "standard").strip().lower()
    if level not in DETAILS:
        raise Refused("BAD_ARGS", message=f"detail is one of {list(DETAILS)}.")
    low, high = SMOOTHNESS_BOUNDS_UM
    if isinstance(smoothness_um, bool) or not isinstance(smoothness_um, (int, float)) or not (
            low <= smoothness_um <= high):
        raise Refused("BAD_ARGS", message=f"smoothness_um is a number of micrometres from "
                      f"{low:g} to {high:g}.")
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
    image = refuse_unknown_inputs(job, workspace, records, section_image, mask)
    choice = deformation.Choice(fit_section=image, fit_atlas=kind,
                                smoothness_um=float(smoothness_um), mask=_mask_label(mask))
    done = fit_deformable(
        job, workspace, records, choice, restrict_to=regions,
        overrides={"stain_metric": metric_name, "stain_edges": bool(edges),
                   # ``standard`` is the tool's own level (``deformation.DETAIL_LEVEL``)
                   **({} if level == "standard" else {"detail": level})})
    return replace(done, settings={
        "section_image": image, "atlas_image": kind,
        "atlas_image_resolved": _resolved(workspace, kind), "mask": choice.mask,
        "restrict_to": list(regions), "smoothness_um": float(smoothness_um),
        "metric": metric_name, "edges": bool(edges), "detail": level})


def _mask_label(mask: Any) -> str:
    return str(mask or ops_masks.NO_MASK).strip()


def refuse_unknown_inputs(
    job: Job, workspace: Workspace, records: Sequence[SliceState], section_image: Any,
    mask: Any,
) -> str:
    """The section image a fit reads, after checking every one of *records* has
    it (``UNKNOWN_IMAGE``) and has the mask *mask* unless it is ``"none"``
    (``UNKNOWN_MASK``); a stale mask is the section's own row, later."""
    image = check_image(job, workspace, records, section_image)
    label = _mask_label(mask)
    if label != ops_masks.NO_MASK:
        lacking = [record.id for record in records if label not in (record.masks or {})]
        if lacking:
            raise Refused("UNKNOWN_MASK", sections=lacking,
                          message=f"No mask {label!r} on {', '.join(lacking)}; make it with "
                          "make_mask.")
    return image

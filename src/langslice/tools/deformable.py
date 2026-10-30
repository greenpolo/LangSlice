"""The deformation on top of a section's linear placement.

:func:`fit_deformable` runs one deformable fit per section
(:mod:`langslice.core.fits.deformation`: the fit grid, the image the fit reads,
the record cache and the engine) and APPLIES each section's result as its
deformation: one undo step for the call. Its callers are :func:`fireants_syn`
(FireANTs SyN on a section image, building on whatever deformation each section
holds) and the landing of a packaged trace
(:func:`langslice.tools.traces.land_trace`: the traced borders fitted).

What can go wrong per section (no valid placement, an applied deformation
that cannot be read, a trace that is missing, stale or failed, a
fit that failed, a record that cannot be saved) is that section's row, not
a refusal of the call. :func:`langslice.tools.fit_pictures.draw_fits` draws
each fit and each traced section's trace through the core.

The ``fireants_syn`` tool's body is :func:`fireants_syn_tool`; a script's
own FireANTs fit, written nowhere, is :func:`langslice.tools.warps.fit_record`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from PIL import Image

from langslice.core.atlas.images import resolve_image
from langslice.core.fits import deformation
from langslice.core.fits.engine_log import engine_output, failure_message
from langslice.core.fits.inputs import (
    NATIVE_BRIGHTNESS,
    EstimatedBlock,
    InputsBlock,
    damage_block,
    mask_block,
)
from langslice.core.refusal import RefusalCode, Refused, ReplyStatus
from langslice.core.sections import images as images_
from langslice.core.sections.damage import exclusions
from langslice.core.stack.section import Section
from langslice.core.stack.stack import Stack
from langslice.tools import masks as ops_masks
from langslice.tools.images import check_image
from langslice.tools.look import show_result
from langslice.tools.refusal import frozen_row, unknown_sections
from langslice.tools.run import Run, as_list, drawn_image, pictured
from langslice.tools.stale import section_inputs, stale_row

if TYPE_CHECKING:
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
    #: The record's folder, relative to the job folder.
    folder: str


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
    #: Once drawn (``fit_pictures.draw_fits``): the pictures, in order
    #: (each drawn row's ``image_indexes`` point into them), ``{"id",
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
    from langslice.core.fits.deformable.masks import EXCLUSION_MARGIN_MM

    return float(EXCLUSION_MARGIN_MM * workspace.brain_scale)


def inputs_block(
    state: Stack, workspace: Workspace, record: Section, *, section_image: str,
    atlas_image: str, mask: str, damage: Sequence[str], restrict_to: Sequence[str] = (),
    brightness: str = NATIVE_BRIGHTNESS, resolved: str | None = None,
) -> InputsBlock:
    """What one native fit of *record* read, named: the section image (name,
    kind, file, digest), the atlas image as asked and as resolved, the section mask
    (``"none"`` or its name, kind and staleness), the atlas-side damage and
    the regions the fit was restricted to."""
    from langslice.core.atlas.images import nissl_note, resolve_image

    entry = record.masks.get(mask) if mask != ops_masks.NO_MASK else None
    kind = None if entry is None else entry.kind
    held = mask_block()
    if mask != ops_masks.NO_MASK:
        held = mask_block(mask, kind, False, ops_masks.made_earlier(state, record, mask))
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


def resolved_atlas_image(workspace: Workspace, atlas_image: str) -> str:
    """Return the atlas image a fit reads, resolved for the atlas.

    Parameters
    ----------
    workspace
        The job's workspace, whose atlas resolves the name.
    atlas_image
        The name a call gave (``nissl`` resolves per atlas); the traced
        fit's ``borders`` stays as it is.

    Returns
    -------
    str
        The atlas's own image name.
    """
    return resolve_image(workspace.atlas, atlas_image) if atlas_image not in (
        deformation.FIT_BORDERS,) else atlas_image


def traced_regions(
    record: Section, include: tuple[str, ...], exclude: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The regions a traced fit uses: the call's own, completed by the trace's.

    The model was shown only the trace's regions (``trace_borders`` include /
    exclude), so the atlas side drops the same ones: the call's *exclude*
    plus the trace's, and the call's *include*, else the trace's.
    """
    held = record.border_trace
    traced_in: tuple[str, ...] = ()
    traced_out: tuple[str, ...] = ()
    if held is not None:
        traced_in = held.include or ()
        traced_out = held.exclude or ()
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
    records: list[Section],
    choice: deformation.Choice,
    *,
    restrict_to: tuple[str, ...] = (),
    overrides: Mapping[str, Any] | None = None,
) -> DeformableFit:
    """Fit *choice* on every section and apply each result.

    *restrict_to* are region entries (the engine's ``structures``, sides
    allowed; empty: every region). Each section's marked regions are excluded
    on their own (:func:`langslice.core.sections.damage.exclusions`).
    *overrides* are
    :class:`~langslice.core.fits.deformable.settings.FitSettings` fields set
    over the choice's own (``stain_metric``, ``stain_edges``, ``detail``).
    ``choice.fit_section`` is the section image the fit reads (a raw channel or
    a made image, by name; ``trace_borders`` fits the traced lines) and
    ``choice.mask`` the section mask by name (:mod:`langslice.tools.masks`): a
    stale mask is that section's row (``STALE_MASK``). Every ok row carries the
    ``inputs`` the fit read and what it ``estimated``
    (:mod:`langslice.core.fits.inputs`). Each fit builds on the section's
    applied deformation where it holds one (composed onto it), else on its
    linear placement. A traced choice waits for the section's trace still
    running (one :data:`~langslice.core.fits.deformation.TRACE_WAIT_S` deadline
    for the call; a trace that lands is checkpointed without an undo step) and
    adds the trace's own regions (:func:`traced_regions`; the row's
    ``trace_regions``). Identical inputs reuse a cached or saved result. A
    section whose deformation is frozen is refused (its row ``FROZEN``).

    The fits run outside the job's write lock, from the state as it stood;
    applying takes the lock (:meth:`~langslice.job.job.Job.writing`) and
    refuses a section whose inputs changed meanwhile
    (``STALE_INPUT``, that section's row), applying
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
        held = record.deformation
        try:
            current = store.current(state, record)
            unreadable = False
        except ValueError:  # its record.json does not load (the error names it)
            logger.warning("Applied deformation of %s unreadable", record.id, exc_info=True)
            current, unreadable = None, True
        if unreadable or (current is None and held is not None
                          and held.linear_key == deformation.linear_key(state, record)):
            folder = None if held is None else held.record
            rows.append(Refused(
                RefusalCode.DEFORMATION_UNREADABLE,
                "The section's applied deformation record cannot be read "
                f"({folder}); it is not replaced by a fit from the "
                "linear placement.").payload(section_id=record.id))
            continue
        own_start = "current" if current is not None else "linear"
        starts[record.id] = own_start
        expected[record.id] = section_inputs(
            state, record, deformation=own_start == deformation.FitStart.CURRENT,
            trace=traced_choice)
        try:
            grid = deformation.fit_grid(state, workspace, record)
        except (ValueError, OSError) as exc:
            rows.append(Refused(RefusalCode.INVALID_LINEAR_PLACEMENT,
                                message=str(exc)).payload(section_id=record.id))
            continue
        previous = current
        previous_key = None
        if current is not None and record.deformation is not None:
            previous_key = record.deformation.key
        own = exclusions(record, tuple(restrict_to))
        failure = {"id": record.id, "status": ReplyStatus.ERROR,
                   "settings": choice.tool_settings()}
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
        except Refused as refusal:  # the trace the fit reads
            rows.append({**failure, **refusal.payload()})
            continue
        except (ValueError, OSError) as exc:
            refusal = Refused(RefusalCode.BAD_SETTINGS, message=str(exc))
            rows.append({**failure, **refusal.payload()})
            continue
        held_mask = record.masks.get(choice.mask)
        key = deformation.cache_key(
            state, grid, settings, identity, previous_key, workspace.brain_scale,
            mask_identity=None if held_mask is None else {
                "file": held_mask.file, "kind": held_mask.kind})
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
                resolved=resolved_atlas_image(workspace, choice.fit_atlas))
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
    return replace(done, traced=traced)


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
        position = cast("int", index)  # the job index fit_deformable set on the row
        fit = jobs[position]
        outcome = fit.result
        block = blocks.get(position)
        row["settings"] = fit.choice.tool_settings()
        if not isinstance(outcome, deformation.DeformableRecord):
            message = failure_message(
                str(getattr(outcome, "error", "no result")), job.engine_log,
                f"the fit of {fit.grid.record.id} failed")
            row.update(Refused(RefusalCode.FIT_FAILED,
                               message=message).payload())
            continue
        store.put(fit.key, outcome)
        numbers = deformation.summary(outcome, fit.previous)
        record = fit.grid.record
        start = starts[record.id]
        # The state may have been reloaded: the section as it is now.
        now = state.by_id(record.id)
        if now is None or section_inputs(
                state, now, deformation=start == deformation.FitStart.CURRENT,
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
        held = record.deformation
        try:
            folder = store.save(record.id, fit.key, outcome)
        except OSError as exc:
            refusal = Refused(RefusalCode.RECORD_WRITE_FAILED,
                              message=str(exc))
            row.update(refusal.payload())
            continue
        traced_fit = fit.choice.fit_section == deformation.TRACED_BORDERS
        step: dict[str, Any] = {
            "method": "trace_borders" if traced_fit else "fireants_syn",
            "start": start, **fit.choice.echo(),
            # the tool's settings the choice does not hold (a step saved
            # before they were recorded has none)
            **({} if traced_fit else {"metric": fit.settings.stain_metric,
                                      "edges": bool(fit.settings.stain_edges),
                                      "detail": fit.settings.detail}),
            "include": list(include), "exclude": list(exclude)}
        traced = fit.image_identity.get("trace")
        if traced:
            # The trace this step fitted (its artifact directory):
            # a packaged trace called again on it reuses this step.
            step["trace"] = traced
        record.deformation = deformation.reference(
            folder=job.layout.relative(folder), key=fit.key, linear=linear, record=outcome,
            step=step, previous=held, numbers=numbers,
        )
        row["written"] = True
        written.append(record.id)
        row["steps"] = len(record.deformation.steps or ())
        fitted.append(Fitted(row=row, fit=fit, outcome=outcome,
                             folder=job.layout.relative(folder)))
    return DeformableFit(rows=rows, fitted=fitted, written=written)


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
    from langslice.core.fits.deformable.engines import import_fireants

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
        raise Refused(RefusalCode.ENGINE_MISSING,
                      message=deformation.ENGINE_MISSING + ".")


def region_entries(workspace: Workspace, state: Any, entries: Any) -> tuple[str, ...]:
    """*entries* (region acronyms, names or ids, ``"CTX:left"`` for one side)
    checked against the atlas and normalized. Refused: ``BAD_ARGS`` (not a
    list of names, or a side that does not exist), ``UNKNOWN_REGIONS``,
    ``NO_SIDES`` (a side on a sagittal stack)."""
    from langslice.core.atlas.sides import has_sides
    from langslice.core.fits.deformable.plane_images import resolve_entries
    from langslice.core.sections.damage import normalized_entries

    if isinstance(entries, str) or not isinstance(entries, (list, tuple)):
        raise Refused(RefusalCode.BAD_ARGS,
                      message="restrict_to is a list of atlas regions.")
    try:
        names = normalized_entries(str(entry) for entry in entries)
    except ValueError as exc:
        raise Refused(RefusalCode.BAD_ARGS, message=str(exc)) from exc
    if names:
        try:
            resolve_entries(workspace.atlas, names)
        except ValueError as exc:
            raise Refused(RefusalCode.UNKNOWN_REGIONS,
                          message=str(exc)) from exc
        if state.plane == "sagittal" and has_sides(names):
            raise Refused(RefusalCode.NO_SIDES,
                          message="A sagittal section lies within one "
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
    :func:`langslice.tools.images.image_names`) against *atlas_image* (one of
    the atlas's images, :func:`langslice.core.atlas.images.atlas_images`;
    ``nissl`` is its Nissl image where it has one, else its template), with the
    warp's smoothness *smoothness_um* in micrometres, the image *metric*
    (``mutual_information`` / ``local_correlation``), the *edges* channel pair
    and *detail* (``coarse`` / ``standard`` / ``fine``), on top of each
    section's applied deformation where it holds one, else its linear placement
    (undo is how to start over). *mask* is ``"none"`` or the name of a mask
    each section holds (:func:`langslice.tools.masks.make_mask`).
    *restrict_to*: warp by these regions only (descendants included,
    ``"CTX:left"`` for one side; empty: every region); each section's marked
    regions are left out on their own
    (:func:`langslice.core.sections.damage.exclusions`). The fit runs as
    :func:`fit_deformable` runs: computed outside the job's write lock, applied
    under it, a section whose inputs changed meanwhile its row ``STALE_INPUT``;
    per-section problems are rows. The result's ``settings`` is the call as it
    was made, each ok row's ``inputs`` and ``estimated`` what the fit read and
    worked out itself. The tool door draws the picture of what was written
    (:func:`langslice.tools.look.show_result`).

    Refused (nothing done): ``ENGINE_MISSING``; ``BAD_ARGS`` (no sections,
    an image the atlas does not have, a metric, detail or smoothness that is
    not one of the allowed, a bad region entry); ``UNKNOWN_SLICE_IDS``;
    ``UNKNOWN_IMAGE``; ``UNKNOWN_MASK`` (a section without that mask);
    ``UNKNOWN_REGIONS``; ``NO_SIDES``.
    """
    from langslice.core.fits.deformable.settings import SMOOTHNESS_BOUNDS_UM

    refs = list(sections or [])
    if not refs:
        raise Refused(RefusalCode.BAD_ARGS,
                      message="Name the sections to fit.")
    kind = check_atlas_image(workspace, atlas_image)
    metric_name = str(metric or "mutual_information").strip()
    if metric_name not in METRICS:
        raise Refused(RefusalCode.BAD_ARGS,
                      message=f"metric is one of {list(METRICS)}.")
    level = str(detail or "standard").strip().lower()
    if level not in DETAILS:
        raise Refused(RefusalCode.BAD_ARGS,
                      message=f"detail is one of {list(DETAILS)}.")
    low, high = SMOOTHNESS_BOUNDS_UM
    if isinstance(smoothness_um, bool) or not isinstance(smoothness_um, (int, float)) or not (
            low <= smoothness_um <= high):
        raise Refused(RefusalCode.BAD_ARGS,
                      message=f"smoothness_um is a number of micrometres from "
                      f"{low:g} to {high:g}.")
    refuse_without_engine()
    regions = region_entries(workspace, job.state, restrict_to)
    records: list[Section] = []
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
        "atlas_image_resolved": resolved_atlas_image(workspace, kind), "mask": choice.mask,
        "restrict_to": list(regions), "smoothness_um": float(smoothness_um),
        "metric": metric_name, "edges": bool(edges), "detail": level})


def _mask_label(mask: object) -> str:
    return str(mask or ops_masks.NO_MASK).strip()


def refuse_unknown_inputs(
    job: Job, workspace: Workspace, records: Sequence[Section], section_image: object,
    mask: object,
) -> str:
    """The section image a fit reads, after checking every one of *records* has
    it (``UNKNOWN_IMAGE``) and has the mask *mask* unless it is ``"none"``
    (``UNKNOWN_MASK``); a stale mask is the section's own row, later."""
    image = check_image(job, workspace, records, section_image)
    label = _mask_label(mask)
    if label != ops_masks.NO_MASK:
        lacking = [record.id for record in records if label not in record.masks]
        if lacking:
            raise Refused(RefusalCode.UNKNOWN_MASK, sections=lacking,
                          message=f"No mask {label!r} on {', '.join(lacking)}; make it with "
                          "make_mask.")
    return image


def check_atlas_image(workspace: Workspace, atlas_image: object) -> str:
    """Return the atlas image a fit reads, by the name the call gave.

    Parameters
    ----------
    workspace
        The job's workspace, whose atlas must have the image.
    atlas_image
        The call's ``atlas_image``; empty is ``nissl``.

    Returns
    -------
    str
        The name, stripped (``nissl`` resolves per atlas when the fit runs).

    Raises
    ------
    Refused
        ``BAD_ARGS`` when the atlas has no image of that name.
    """
    kind = str(atlas_image or "nissl").strip()
    try:
        resolve_image(workspace.atlas, kind)
    except ValueError as exc:
        raise Refused(RefusalCode.BAD_ARGS, message=str(exc)) from exc
    return kind


def nothing_fitted(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Return the top-level error of a fit call that fitted nothing.

    Parameters
    ----------
    rows
        The call's per-section rows, each refused.

    Returns
    -------
    dict
        When every section was refused for the same reason, that reason's
        ``error`` and ``message`` (the rows keep theirs); otherwise
        ``NOTHING_FITTED``, each row naming its own.
    """
    codes = {str(row.get("error")) for row in rows if row.get("error")}
    if not rows or len(codes) != 1 or not all(row.get("error") for row in rows):
        return {"error": RefusalCode.NOTHING_FITTED}
    code = codes.pop()
    message = str(rows[0].get("message") or "")
    if message and len(rows) > 1:
        message = f"Every section was refused for this reason. {message}"
    if not message:
        return {"error": code}
    return {"error": code, "message": message}


# --- the tool ----------------------------------------------------------------


def fireants_syn_tool(
    run: Run,
    sections: Sequence[str],
    restrict_to: Sequence[str],
    *,
    section_image: str = "preprocessed",
    atlas_image: str = "nissl",
    mask: str = "none",
    smoothness_um: float = 80.0,
    metric: str = "mutual_information",
    edges: bool = True,
    detail: str = "standard",
    view: bool = True,
) -> dict[str, Any]:
    """Run the ``fireants_syn`` tool: fit, answer, picture.

    The call is capped at :data:`MAX_FIREANTS_SYN_SECTIONS` sections
    (``TOO_MANY_SECTIONS``) and fitted by :func:`fireants_syn`. The picture
    is each fitted section under its new registration, drawn on the image
    the fit read, zoomed to the ``restrict_to`` regions' box when that
    magnifies it, those regions' borders thick.

    Parameters
    ----------
    run
        The run the tool belongs to.
    sections, restrict_to, section_image, atlas_image, mask, smoothness_um
        The call's arguments, as :func:`fireants_syn` takes them.
    metric, edges, detail
        The engine's options, as :func:`fireants_syn` takes them.
    view
        Whether to answer with the picture.

    Returns
    -------
    dict
        The reply: ``status``, the call's ``settings``, one row per section
        (``results``), the status rows the write changed.
    """
    from langslice.tools.catalog import (  # local: cycle with tools.catalog
        check_section_count,
    )

    job, workspace = run.job, run.ctx
    check_section_count(job, "fireants_syn", len(as_list(sections)))
    done = fireants_syn(
        job, workspace, as_list(sections), restrict_to=as_list(restrict_to),
        section_image=section_image, atlas_image=atlas_image, mask=mask,
        smoothness_um=smoothness_um, metric=metric, edges=edges, detail=detail)
    fitted = [str(row["id"]) for row in done.rows
              if row.get("status") == ReplyStatus.OK]
    result: dict[str, Any] = {"status": "ok" if fitted else "error"}
    if not fitted:
        result.update(nothing_fitted(done.rows))
    result["settings"] = done.settings
    result["results"] = done.rows
    if done.written:
        result["changed"] = run.changed(done.written)["changed"]
    if not fitted or not run.shows(view):
        return result

    regions = region_entries(workspace, job.state, as_list(restrict_to))
    zooms = _region_zooms(run, fitted, regions)
    shown = show_result(
        job, workspace, "fireants_syn", "overlay", fitted, zooms=zooms,
        highlight=regions, images=drawn_image(section_image))
    return pictured(result, shown)


def _region_zooms(
    run: Run, fitted: Sequence[str], regions: Sequence[str]
) -> dict[str, list[float]]:
    """Return each fitted section's zoom to the regions' box.

    None without regions, and none where it would not magnify enough
    (:func:`langslice.tools.affine_fit.zoom_window`).
    """
    from langslice.tools import affine_fit  # local: cycle with tools.affine_fit

    zooms: dict[str, list[float]] = {}
    if not regions:
        return zooms
    for name in fitted:
        section = run.job.state.by_id(name)
        box = None
        if section is not None:
            box = affine_fit.region_box(run.ctx, run.job.state, section, regions)
        window = affine_fit.zoom_window(box)
        if window is not None:
            zooms[name] = window
    return zooms

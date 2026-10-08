"""The in-plane transform: the stored record, from knobs or from a fit, and its write.

Every stored transform has one shape (``SliceState.transform``): ``kind``
(``interactive``, ``elastix``, or the host's), ``params``
(the six normalized numbers on the section's working frame, the exact map,
:mod:`langslice.core.affine`), ``physical`` (the same map as the knobs about a
pivot in canvas fractions), the ``calibration`` it was drawn with, and
``mirrored``. :func:`transform_record` and :func:`fit_transform` build
it; :func:`set_transforms` writes any number of them as one undo step.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from langslice.core.affine import normalized_physical_affine
from langslice.core.damage import exclusions
from langslice.core.display import canonical_atlas_name
from langslice.core.transform import physical_decomposition
from langslice.ops.inputs import section_inputs, stale_row
from langslice.ops.refusal import Refused

if TYPE_CHECKING:
    from langslice.core.placement import Staged
    from langslice.core.state import SliceState
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job

logger = logging.getLogger(__name__)

def transform_record(
    *,
    size: tuple[int, int],
    um_per_px: float,
    calibration: Mapping[str, Any],
    pivot: tuple[float, float] | None,
    pivot_frac: Sequence[float],
    knobs: Mapping[str, float],
    note: str = "",
) -> dict[str, Any]:
    """The stored record of a transform given as knobs (``interactive_transform``).

    *size* and *um_per_px* are the section's working frame and its
    calibration; *pivot* is the rotation/scale centre on that frame (None:
    its centre) and *pivot_frac* the same point as canvas fractions, which
    the record keeps beside the knobs. *knobs* are ``rotation_deg``,
    ``scale_x``, ``scale_y``, ``translate_x_mm``, ``translate_y_mm`` and
    ``shear`` (:func:`langslice.core.affine.decompose_affine`'s; may be left
    out: none); the record's ``physical`` lists all six, so a
    tweak that copies a fit's knobs, shear included, keeps the fit's map.
    """
    knobs = {**knobs, "shear": knobs.get("shear", 0.0)}
    six = normalized_physical_affine(size=size, um_per_px=um_per_px, pivot=pivot, **knobs)
    return {
        "kind": "interactive",
        "params": six,
        "physical": {**knobs, "pivot": list(pivot_frac)},
        "calibration": dict(calibration),
        "mirrored": physical_decomposition(six, size)["mirrored"],
        "note": str(note or "").strip(),
    }


def same_transform(held: Mapping[str, Any] | None, new: Mapping[str, Any]) -> bool:
    """Whether *new* is the transform already held, its note aside.

    The same numbers again are a look, not a write: no undo step for it.
    """
    return held is not None and {**held, "note": ""} == {**new, "note": ""}


def fit_transform(
    fit: Mapping[str, Any],
    *,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    fit_atlas: str = "template",
) -> dict[str, Any]:
    """The stored record of one Elastix affine fit (kind ``elastix``).

    *fit* is :func:`langslice.core.transform.fit_elastix`'s ok payload.
    Regions and a non-default atlas image are recorded only when they were
    used.
    """
    fit_atlas = canonical_atlas_name(fit_atlas)
    return {
        "kind": "elastix",
        "params": list(fit["params"]),
        "physical": fit["physical"],
        "iou": fit["iou"],
        "calibration": fit["calibration"],
        "mirrored": fit["mirrored"],
        **({"regions": {"include": list(include), "exclude": list(exclude)}}
           if include or exclude else {}),
        **({"fit_atlas": fit_atlas} if fit_atlas != "template" else {}),
    }


#: A fit that turns a section further than this from its previous transform
#: reports the turn in its row (``turn_deg``): the overlap it reports cannot
#: tell such a turn from a correct one.
LARGE_TURN_DEG = 45.0


def _turn_deg(previous: Mapping[str, Any] | None, outcome: Mapping[str, Any]) -> float:
    """How far a fit turns the section from its previous transform, in degrees (0-180)."""
    before = float(((previous or {}).get("physical") or {}).get("rotation_deg") or 0.0)
    after = float((outcome.get("physical") or {}).get("rotation_deg") or 0.0)
    return abs((after - before + 180.0) % 360.0 - 180.0)


def set_transforms(job: Job, transforms: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Write each ``{section id: transform record}`` as ONE undo step.

    The record replaces the section's transform whole. The host's locked
    sections are refused (:class:`Refused` ``LOCKED``, nothing written).
    Returns the ids written; nothing is committed for an empty mapping.
    """
    locked = [name for name in transforms if name in job.locked]
    if locked:
        raise Refused("LOCKED", ids=locked)
    with job.writing():
        records = []
        for name in transforms:
            record = job.state.by_id(name)
            if record is None:
                raise Refused("UNKNOWN_SLICE_IDS", unknown=[name])
            records.append(record)
        if not records:
            return []
        before = job.snapshot()
        for record in records:
            record.transform = dict(transforms[record.id])
        job.commit(before)
        return [record.id for record in records]


# --- elastix_affine ------------------------------------------------------------------

def fit_targets(job: Job) -> list[SliceState]:
    """The sections ``elastix_affine`` fits when none are named: every
    positioned section the host did not lock, in corrected order."""
    return [
        record for record in job.state.in_order()
        if record.position_mm is not None and record.id not in job.locked
    ]


@dataclass(frozen=True)
class AffineFit:
    """What :func:`elastix_affine` did.

    ``rows``: one per section asked, in order: the fit's payload (``id``,
    ``status: ok``, ``iou``, ``physical``, ``params`` (the six stored
    numbers), ``calibration``, ``mirrored``, ``regions`` when restricted)
    or ``{"id", "status": "error", "error", ...}`` (``LOCKED``, the fitter's
    code, ``STALE_INPUT``). ``fitted``: the sections written.
    """

    rows: list[dict[str, Any]] = field(default_factory=list)
    fitted: list[str] = field(default_factory=list)


def _fit_elastix(
    job: Job, workspace: Workspace, records: Sequence[SliceState], *, fit_atlas: str,
    restrict_to: tuple[str, ...],
) -> AffineFit:
    """Fit an Elastix affine per section and write each fit as its transform.

    Each section's current placement is refined against the atlas image
    *fit_atlas* (:func:`langslice.core.transform.fit_elastix`), by the
    *restrict_to* regions only (empty: every region); each section's marked
    regions are left out on their own (:func:`langslice.core.damage.exclusions`).
    A locked section is refused (``LOCKED``). Every fit that succeeded is
    written as ONE undo step (:func:`set_transforms`, :func:`fit_transform`
    records).

    The fits run outside the job's write lock, from the state as it stood;
    the write takes the lock (:meth:`~langslice.job.job.Job.writing`) and
    refuses a section whose inputs changed meanwhile
    (:data:`langslice.ops.inputs.STALE_INPUT`, that section's row), writing
    the others.
    """
    from langslice.core.transform import fit_elastix

    state = job.state
    rows: list[dict[str, Any]] = []
    fits: list[tuple[SliceState, dict[str, Any], tuple[str, ...], tuple[str, ...]]] = []
    expected: dict[str, str] = {}
    for record in records:
        expected[record.id] = section_inputs(state, record)
        if record.id in job.locked:
            rows.append({"id": record.id, "status": "error", "error": "LOCKED"})
            continue
        kept, dropped = exclusions(record, restrict_to, ())
        try:
            outcome = fit_elastix(state, workspace, record, include=kept, exclude=dropped,
                                  atlas_image=fit_atlas)
        except Exception as exc:  # nothing is written for this section
            logger.warning("elastix_affine failed for %s: %s", record.id, exc)
            rows.append({"id": record.id, "status": "error",
                         "error": getattr(exc, "code", "FIT_FAILED"), "message": str(exc)})
            continue
        rows.append(outcome)
        if outcome["status"] != "ok":
            continue
        turn = _turn_deg(record.transform, outcome)
        if turn > LARGE_TURN_DEG:
            outcome["turn_deg"] = round(turn, 1)
        fits.append((record, outcome, kept, dropped))
    with job.writing():
        records_out: dict[str, dict[str, Any]] = {}
        for record, outcome, kept, dropped in fits:
            now = state.by_id(record.id)
            if now is None or section_inputs(state, now) != expected[record.id]:
                outcome.clear()
                outcome.update(stale_row(record.id))
                continue
            records_out[record.id] = fit_transform(outcome, include=kept, exclude=dropped,
                                                   fit_atlas=fit_atlas)
        written = set_transforms(job, records_out)
    return AffineFit(rows=rows, fitted=written)


# --- interactive_transform -------------------------------------------------------------

@dataclass(frozen=True)
class Adjustment:
    """One ``interactive_transform`` entry: what it set, or why not."""

    #: ``{"status": "error", "error", ...}`` when the entry was refused
    #: (``BAD_ARGS``, ``LOCKED``, ``UNKNOWN_SLICE_IDS``, ``NO_POSITION``,
    #: ``ATLAS_RENDER_FAILED``, ``BAD_PIVOT``, ...); else None.
    error: dict[str, Any] | None = None
    #: The section, its working frame, canvas and pivot
    #: (:class:`langslice.core.placement.Staged`), with the knobs used.
    staged: Staged | None = None
    #: The section's transform before the call.
    previous: dict[str, Any] | None = None
    #: The stored record these knobs make (:func:`transform_record`).
    transform: dict[str, Any] | None = None
    #: Whether it differs from *previous* (the same numbers again write nothing).
    written: bool = False
    #: The section's id; its flip and quarter turn after the call
    #: (``{"flip", "rotate_quarter", "changed"}``); and, when the call
    #: changed them under a stored transform, the knob values the rebuilt
    #: transform kept.
    id: str = ""
    orientation: dict[str, Any] | None = None
    kept: dict[str, float] | None = None

    @property
    def message(self) -> str:
        """The reply's line when the orientation changed under a stored transform."""
        if not self.kept:
            return ""
        return ("The orientation changed. The knob values were kept as they were; the "
                "transform was rebuilt on the new orientation, so the picture moved.")


@dataclass(frozen=True)
class Adjusted:
    """What :func:`interactive_transform` did: one :class:`Adjustment` per
    entry, in order, and the sections written."""

    entries: list[Adjustment] = field(default_factory=list)
    written: list[str] = field(default_factory=list)


#: The knobs ``interactive_transform`` takes, with the value of no change.
KNOB_DEFAULTS: dict[str, float] = {
    "rotation_deg": 0.0, "scale_x": 1.0, "scale_y": 1.0,
    "translate_x_mm": 0.0, "translate_y_mm": 0.0, "shear": 0.0,
}


def held_knobs(record: SliceState) -> dict[str, float]:
    """The six knob values the section's stored transform has (identity without one)."""
    physical = (record.transform or {}).get("physical") or {}
    held: dict[str, float] = {}
    for name, default in KNOB_DEFAULTS.items():
        value = physical.get(name)
        held[name] = default if value is None else float(value)
    return held


def interactive_transform(
    job: Job,
    workspace: Workspace,
    sections: Sequence[Any],
) -> Adjusted:
    """Set each section's orientation and in-plane knobs; ONE undo step.

    A section is ``{"id", "flip"?, "rotate_quarter"? (0, 90, 180 or 270),
    "rotation_deg"?, "scale_x"?, "scale_y"?, "shear"?, "translate_x_mm"?,
    "translate_y_mm"?, "pivot"?, "note"?}``. Values are absolute and a value
    left out keeps what the section has now (identity without a transform;
    the pivot a stored transform used stays). The quarter turn is applied
    first, then the flip (the renderer's order).
    A changed orientation keeps the section's knob values: its transform is
    rebuilt on the new orientation with the same numbers, not dropped, and
    the entry says so (:attr:`Adjustment.kept`, :attr:`Adjustment.message`).
    A changed orientation of a section with no transform, and no knob given,
    sets the orientation alone.

    Per entry, refused (nothing written for it): ``LOCKED``,
    ``UNKNOWN_SLICE_IDS``, ``BAD_ARGS`` (not a mapping, a knob that is not a
    finite number, a flip that is not true/false), ``BAD_ROTATION``,
    ``FLIP_DISABLED`` (the spec turns flipping off and the call changes the
    flip), ``NO_POSITION`` (a transform to build on a section without a
    position) and the stager's codes (:func:`langslice.core.placement.stage`).
    All the changed sections are written together, under the job's lock; the
    tool door draws the picture of what was written
    (:func:`langslice.ops.look.show_result`).
    """
    from dataclasses import replace

    from langslice.core.placement import StageFailure, stage
    from langslice.core.state import ROTATIONS

    state = job.state
    done: list[Adjustment] = []
    pending: dict[str, tuple[bool, int, dict[str, Any] | None]] = {}
    for entry in sections:
        if not isinstance(entry, Mapping):
            done.append(Adjustment(error={"status": "error", "error": "BAD_ARGS"}))
            continue
        record = state.resolve(entry.get("id", ""))
        if record is None:
            done.append(Adjustment(error={"status": "error", "error": "UNKNOWN_SLICE_IDS",
                                          "unknown": [str(entry.get("id", ""))]}))
            continue
        name = record.id
        if name in job.locked:
            done.append(Adjustment(error={"status": "error", "error": "LOCKED", "id": name},
                                   id=name))
            continue

        def refuse(error: str, name: str = name, **facts: Any) -> None:
            done.append(Adjustment(error={"status": "error", "error": error, "id": name,
                                          **facts}, id=name))

        flip = record.flip
        if entry.get("flip") is not None:
            if not isinstance(entry["flip"], bool):
                refuse("BAD_ARGS", message="flip is true or false")
                continue
            if entry["flip"] != flip and not job.spec.transform.flip:
                refuse("FLIP_DISABLED")
                continue
            flip = entry["flip"]
        rotation = record.rotation_deg
        if entry.get("rotate_quarter") is not None:
            try:
                value = float(entry["rotate_quarter"])
                rotation = int(value) % 360 if value == int(value) else -1
            except (TypeError, ValueError):
                rotation = -1
            if rotation not in ROTATIONS:
                refuse("BAD_ROTATION", allowed=list(ROTATIONS))
                continue
        knobs = held_knobs(record)
        given = {key: entry[key] for key in KNOB_DEFAULTS if entry.get(key) is not None}
        try:
            knobs.update({key: float(value) for key, value in given.items()})
        except (TypeError, ValueError):
            refuse("BAD_ARGS", message="every transform number must be a number")
            continue
        if not all(math.isfinite(value) for value in knobs.values()):
            refuse("BAD_ARGS", message="every transform number must be finite")
            continue
        changed = (flip, rotation) != (record.flip, record.rotation_deg)
        orientation = {"flip": flip, "rotate_quarter": rotation, "changed": changed}
        held = record.transform
        if held is None and not given:
            # No transform and none asked for: the orientation alone.
            if changed:
                pending[name] = (flip, rotation, None)
            done.append(Adjustment(id=name, orientation=orientation, written=changed))
            continue
        if record.position_mm is None:
            refuse("NO_POSITION")
            continue
        pivot = ((held or {}).get("physical") or {}).get("pivot")
        twin = replace(record, flip=flip, rotation_deg=rotation)
        try:
            staged = stage(workspace, state, twin, knobs,
                           entry.get("pivot", pivot if pivot else "canvas"))
        except StageFailure as failure:
            refuse(failure.code, message=str(failure))
            continue
        transform = transform_record(
            size=staged.section.size, um_per_px=staged.um_per_px,
            calibration=staged.calibration, pivot=staged.pivot_in_section,
            pivot_frac=staged.pivot_frac, knobs=staged.params,
            note=str(entry.get("note", "")),
        )
        wrote = changed or not same_transform(held, transform)
        if wrote:
            pending[name] = (flip, rotation, transform)
        done.append(Adjustment(
            staged=staged, previous=held, transform=transform, written=wrote,
            id=name, orientation=orientation,
            kept=(held_knobs(record) if changed and held is not None else None)))
    written: list[str] = []
    if pending:
        with job.writing():
            before = job.snapshot()
            for name, (flip, rotation, transform) in pending.items():
                target = job.state.by_id(name)
                if target is None:
                    continue
                target.flip, target.rotation_deg, target.transform = flip, rotation, transform
                written.append(name)
            if written:
                job.commit(before)
    return Adjusted(entries=done, written=written)


# --- elastix_affine --------------------------------------------------------------------

#: The atlas images an Elastix affine fit reads.
ELASTIX_ATLAS_IMAGES: tuple[str, ...] = ("template", "nissl")
#: Margin added round the regions' bounding box, as a fraction of the canvas.
BOX_MARGIN = 0.05


def region_box(
    workspace: Workspace, state: Any, record: SliceState, regions: Sequence[str],
) -> list[float] | None:
    """The atlas regions' bounding box on the section's canvas.

    ``[x0, y0, x1, y1]`` as fractions of the canvas, with a small margin: the
    zoom :func:`langslice.core.canvas.zoom_box` takes. None when the regions
    are not in the plane there. A side (``CTX:left``) is read as the region
    on both sides.
    """
    from langslice.core.atlas.sides import split_side
    from langslice.core.deformable.atlas_images import regions_mask
    from langslice.core.placement import stage

    if not regions:
        return None
    try:
        names = [split_side(name)[0] for name in regions]
        geometry = stage(workspace, state, record, dict(KNOB_DEFAULTS), "canvas").geometry
        mask = regions_mask(workspace.atlas, geometry.annotation, names)
    except Exception as exc:
        logger.warning("region box failed for %s: %s", record.id, exc)
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    width, height = geometry.size
    scale = geometry.atlas_scale
    left, top = geometry.atlas_offset
    x0, x1 = left + cols[0] * scale, left + (cols[-1] + 1) * scale
    y0, y1 = top + rows[0] * scale, top + (rows[-1] + 1) * scale
    return [round(max(0.0, x0 / width - BOX_MARGIN), 4),
            round(max(0.0, y0 / height - BOX_MARGIN), 4),
            round(min(1.0, x1 / width + BOX_MARGIN), 4),
            round(min(1.0, y1 / height + BOX_MARGIN), 4)]


def elastix_affine(
    job: Job,
    workspace: Workspace,
    sections: Sequence[str] = (),
    restrict_to: Sequence[str] = (),
    atlas_image: str = "template",
) -> AffineFit:
    """Fit an in-plane affine with Elastix, per section; ONE undo step for the fits.

    Each section's current placement is refined against *atlas_image* (``template`` or ``nissl``;
    ``ara`` is still read as ``template``). *sections* are ids (filenames or
    corrected indices); empty, every section :func:`fit_targets` names.
    *restrict_to* fits by those atlas regions only (sides allowed; empty:
    every region). Each section's marked damage regions are left out
    automatically (:func:`langslice.core.damage.exclusions`).

    Every ok row of a call with *restrict_to* carries ``restrict_box``: the
    regions' bounding box on that section's canvas as ``[x0, y0, x1, y1]``
    fractions (``core.canvas.zoom_box``'s zoom), so a later picture can zoom
    on them; absent when the regions are not in that plane. Refused (nothing
    fitted): ``UNKNOWN_SLICE_IDS``, ``BAD_ARGS`` (``atlas_image`` not one of
    :data:`ELASTIX_ATLAS_IMAGES`, a region that is not text).
    """
    image = canonical_atlas_name(str(atlas_image))
    if image not in ELASTIX_ATLAS_IMAGES:
        raise Refused("BAD_ARGS", message=f"atlas_image is one of {list(ELASTIX_ATLAS_IMAGES)}")
    if any(not isinstance(name, str) for name in restrict_to):
        raise Refused("BAD_ARGS", message="restrict_to lists atlas regions")
    if sections:
        records = []
        for ref in sections:
            record = job.state.resolve(ref)
            if record is None:
                raise Refused("UNKNOWN_SLICE_IDS", unknown=[str(ref)])
            records.append(record)
    else:
        records = fit_targets(job)
    result = _fit_elastix(job, workspace, records, fit_atlas=image,
                          restrict_to=tuple(restrict_to))
    if restrict_to:
        for row in result.rows:
            record = job.state.by_id(str(row.get("id")))
            if row.get("status") == "ok" and record is not None:
                box = region_box(workspace, job.state, record, restrict_to)
                if box is not None:
                    row["restrict_box"] = box
    return result

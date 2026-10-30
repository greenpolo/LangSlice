"""The in-plane transform: its stored record, its write, interactive_transform.

Every stored transform has one shape (``Section.transform``): ``kind``
(``interactive``, ``elastix``, or the host's), ``params`` (the six normalized
numbers on the section's working frame, the exact map,
:mod:`langslice.core.geometry.affine`), ``physical`` (the same map as the knobs
about a pivot in canvas fractions), the ``calibration`` it was drawn with, and
``mirrored``. :func:`transform_record` builds it from knobs (an Elastix fit's
is ``tools.affine_fit.fit_transform``); :func:`set_transforms` writes any
number of them as one undo step. :func:`interactive_transform` sets
orientations and knobs; the tool's function (``interactive_transform_tool``)
adds its rows and the picture of each section written.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from langslice.core.fits.affine import physical_decomposition
from langslice.core.geometry.affine import normalized_physical_affine
from langslice.core.jsonable import JsonObject
from langslice.core.pictures.placement import stage
from langslice.core.refusal import RefusalCode, Refused
from langslice.core.stack import codec
from langslice.core.stack.affine import PhysicalAffine, StoredAffine
from langslice.core.stack.json_reads import flag_at
from langslice.core.stack.section import ROTATIONS
from langslice.tools.look import show_result
from langslice.tools.refusal import (
    check_frozen,
    check_section_count,
    section_refusal,
    unknown_sections,
)
from langslice.tools.run import as_list, pictured

if TYPE_CHECKING:
    from langslice.core.pictures.placement import Staged
    from langslice.core.stack.section import Section
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job
    from langslice.tools.arguments import SectionTransform
    from langslice.tools.run import Run


def transform_record(
    *,
    size: tuple[int, int],
    um_per_px: float,
    calibration: Mapping[str, Any],
    pivot: tuple[float, float] | None,
    pivot_frac: Sequence[float],
    knobs: Mapping[str, float],
) -> StoredAffine:
    """The stored record of a transform given as knobs (``interactive_transform``).

    *size* and *um_per_px* are the section's working frame and its calibration;
    *pivot* is the rotation/scale centre on that frame (None: its centre) and
    *pivot_frac* the same point as canvas fractions, which the record keeps
    beside the knobs. *knobs* are ``rotation_deg``, ``scale_x``, ``scale_y``,
    ``translate_x_mm``, ``translate_y_mm`` and ``shear``
    (:func:`langslice.core.geometry.affine.decompose_affine`'s; may be left
    out: none); the record's ``physical`` lists all six, so a tweak that copies
    a fit's knobs, shear included, keeps the fit's map.
    """
    knobs = {**knobs, "shear": knobs.get("shear", 0.0)}
    six = normalized_physical_affine(
        size=size, um_per_px=um_per_px, pivot=pivot, **knobs
    )
    pivot_at = (pivot_frac[0], pivot_frac[1])
    physical = PhysicalAffine(**knobs, pivot=pivot_at, extra={})
    # reportAny: the staged calibration and the decomposition are plain dicts
    scale = codec.calibration_from_json(cast("JsonObject", calibration))
    decomposed = cast("JsonObject", physical_decomposition(six, size))
    mirrored = flag_at(decomposed, "mirrored")
    return StoredAffine(
        kind="interactive",
        params=tuple(six),
        physical=physical,
        calibration=scale,
        mirrored=mirrored,
        note="",
    )


def same_transform(held: StoredAffine | None, new: StoredAffine) -> bool:
    """Whether *new* is the transform already held, its note aside.

    The same numbers again are a look, not a write: no undo step for it.
    """
    return held is not None and replace(held, note="") == replace(new, note="")


def physical_knobs(transform: StoredAffine | None) -> PhysicalAffine:
    """Return a stored transform's knobs; none set without a transform.

    Parameters
    ----------
    transform
        The stored transform, or None.

    Returns
    -------
    PhysicalAffine
        Its knobs about its pivot.
    """
    if transform is None or transform.physical is None:
        return PhysicalAffine()
    return transform.physical


def det_sign(transform: StoredAffine | None) -> int:
    """The sign (1, -1, or 0 for a transform with none to read) of a stored
    transform's 2x2 determinant: positive keeps the section's handedness in
    the frame its orientation sets, negative turns it over."""
    numbers = () if transform is None else transform.params
    params = np.asarray(numbers, dtype=float)
    if params.shape != (6,) or not np.isfinite(params).all():
        return 0
    det = params[0] * params[4] - params[1] * params[3]
    return 1 if det > 0 else -1 if det < 0 else 0


def reverses_orientation(record: Section, transform: StoredAffine) -> bool:
    """Whether writing *transform* on *record* would turn the section over,
    which its orientation, when frozen, forbids: its determinant's sign
    differs from the sign of the transform it replaces (a supplied transform
    may carry a mirror of its own; a section with none is unmirrored)."""
    held = det_sign(record.transform) or 1
    return det_sign(transform) != held


def mirror_message(name: str) -> str:
    """Return why a transform that would mirror section *name* is refused.

    Parameters
    ----------
    name
        The section's filename.

    Returns
    -------
    str
        The refusal's message: its orientation is frozen.
    """
    return (
        f"This transform would mirror {name}, which changes its orientation; the "
        "orientation is frozen there, so the transform has to keep the section's "
        "handedness."
    )


def set_transforms(
    job: Job, transforms: Mapping[str, StoredAffine]
) -> list[str]:
    """Write each ``{section id: transform record}`` as ONE undo step.

    The record replaces the section's transform whole. A section whose affine
    is frozen is refused (:class:`Refused` ``FROZEN``, nothing written), and so
    is a transform that would turn over a section whose orientation is frozen
    (:func:`reverses_orientation`). Returns the ids written; nothing is
    committed for an empty mapping.
    """
    check_frozen(job, list(transforms), "affine")
    for name, transform in transforms.items():
        record = job.state.by_id(name)
        if (
            record is not None
            and "orientation" in job.frozen_for(name)
            and reverses_orientation(record, transform)
        ):
            raise Refused(
                RefusalCode.FROZEN,
                ids=[name],
                properties=["orientation"],
                message=mirror_message(name),
            )
    with job.writing():
        records: list[Section] = []
        for name in transforms:
            record = job.state.by_id(name)
            if record is None:
                raise unknown_sections(job.state, [name])
            records.append(record)
        if not records:
            return []
        before = job.snapshot()
        for record in records:
            record.transform = transforms[record.id]
        job.commit(before)
        return [record.id for record in records]


# --- interactive_transform ---------------------------------------------------


@dataclass(frozen=True)
class Adjustment:
    """One ``interactive_transform`` entry: what it set, or why not."""

    #: Why the entry was refused (``BAD_ARGS``, ``FROZEN``,
    #: ``UNKNOWN_SLICE_IDS``, ``NO_POSITION``, ``ATLAS_RENDER_FAILED``,
    #: ``BAD_PIVOT``, ...); None when it was not.
    refusal: Refused | None = None
    #: The section, its working frame, canvas and pivot
    #: (:class:`langslice.core.pictures.placement.Staged`), with the knobs
    #: used.
    staged: Staged | None = None
    #: The section's transform before the call.
    previous: StoredAffine | None = None
    #: The stored record these knobs make (:func:`transform_record`).
    transform: StoredAffine | None = None
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
        return (
            "The orientation changed. The knob values were kept as they were; the "
            "transform was rebuilt on the new orientation, so the picture moved."
        )


@dataclass(frozen=True)
class Adjusted:
    """What :func:`interactive_transform` did: one :class:`Adjustment` per
    entry, in order, and the sections written."""

    entries: list[Adjustment] = field(default_factory=list)
    written: list[str] = field(default_factory=list)


#: ``interactive_transform``'s refusals of one entry (the section's name, an
#: option's name, filled in).
_ENTRY = "Each entry is an object: the section's id and what to change."
_ROTATIONS = "rotate_quarter is 0, 90, 180 or 270."
_OPTION_OFF = "{} is switched off for this job; leave it out."
#: The refusal of a transform on a section without a position.
NO_POSITION = "{} has no position yet; write its position first."
#: The knobs ``interactive_transform`` takes, with the value of no change.
KNOB_DEFAULTS: dict[str, float] = {
    "rotation_deg": 0.0,
    "scale_x": 1.0,
    "scale_y": 1.0,
    "translate_x_mm": 0.0,
    "translate_y_mm": 0.0,
    "shear": 0.0,
}


def held_knobs(section: Section) -> dict[str, float]:
    """The six knob values the section's stored transform has (identity without one)."""
    physical = physical_knobs(section.transform)
    if section.transform is None or section.transform.physical is None:
        return dict(KNOB_DEFAULTS)
    knobs: dict[str, float] = {}
    for name, default in KNOB_DEFAULTS.items():  # the knobs are its fields
        value = cast("float | None", getattr(physical, name))
        knobs[name] = default if value is None else float(value)
    return knobs


def interactive_transform(
    job: Job,
    workspace: Workspace,
    sections: Sequence[Any],
) -> Adjusted:
    """Set each section's orientation and in-plane knobs; ONE undo step.

    A section is ``{"id", "flip"?, "rotate_quarter"? (0, 90, 180 or 270),
    "rotation_deg"?, "scale_x"?, "scale_y"?, "shear"?, "translate_x_mm"?,
    "translate_y_mm"?}``. Values are absolute and a value
    left out keeps what the section has now (identity without a transform;
    the pivot a stored transform used stays). The quarter turn is applied
    first, then the flip (the renderer's order).
    A changed orientation keeps the section's knob values: its transform is
    rebuilt on the new orientation with the same numbers, not dropped, and
    the entry says so (:attr:`Adjustment.kept`, :attr:`Adjustment.message`).
    A changed orientation of a section with no transform, and no knob given,
    sets the orientation alone.

    A section whose affine is frozen keeps its stored transform whatever the
    orientation does (its numbers are never rewritten), and the same numbers
    again write nothing, so it can still be looked at.

    Per entry, refused (nothing written for it): ``FROZEN`` (the entry would
    change a frozen orientation, change the knobs of a frozen affine, or, under
    a frozen orientation, turn the section over: a transform whose determinant
    changes sign), ``UNKNOWN_SLICE_IDS``, ``BAD_ARGS`` (not a mapping, a knob
    that is not a finite number, a flip that is not true/false),
    ``BAD_ROTATION``, ``OPTION_OFF`` (the job disables
    ``interactive_transform:flip`` or ``:rotate`` and the call changes it),
    ``NO_POSITION`` (a transform to build on a section without a position) and
    the stager's codes (:func:`langslice.core.pictures.placement.stage`). All
    the changed sections are written together, under the job's lock; the tool
    door draws the picture of what was written
    (:func:`langslice.tools.look.show_result`).
    """
    state = job.state
    done: list[Adjustment] = []
    pending: dict[str, tuple[bool, int, StoredAffine | None]] = {}
    for entry in sections:
        if not isinstance(entry, Mapping):
            not_entry = Refused(RefusalCode.BAD_ARGS, _ENTRY)
            done.append(Adjustment(refusal=not_entry))
            continue
        record = state.resolve(entry.get("id", ""))
        if record is None:
            done.append(
                Adjustment(
                    refusal=unknown_sections(state, [str(entry.get("id", ""))])
                )
            )
            continue
        name = record.id
        frozen_now = job.frozen_for(name)

        flip = record.flip
        if entry.get("flip") is not None:
            if not isinstance(entry["flip"], bool):
                done.append(
                    _refused(
                        name, RefusalCode.BAD_ARGS, "flip is true or false."
                    )
                )
                continue
            if entry["flip"] != flip and job.spec.off(
                "interactive_transform:flip"
            ):
                done.append(
                    _refused(
                        name,
                        RefusalCode.OPTION_OFF,
                        _OPTION_OFF.format("flip"),
                    )
                )
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
                bad_rotation = Refused(
                    RefusalCode.BAD_ROTATION,
                    _ROTATIONS,
                    id=name,
                    allowed=list(ROTATIONS),
                )
                done.append(Adjustment(refusal=bad_rotation, id=name))
                continue
            if rotation != record.rotation_deg and job.spec.off(
                "interactive_transform:rotate"
            ):
                done.append(
                    _refused(
                        name,
                        RefusalCode.OPTION_OFF,
                        _OPTION_OFF.format("rotate_quarter"),
                    )
                )
                continue
        knobs = held_knobs(record)
        given = {
            key: entry[key]
            for key in KNOB_DEFAULTS
            if entry.get(key) is not None
        }
        try:
            knobs.update({key: float(value) for key, value in given.items()})
        except (TypeError, ValueError):
            done.append(
                _refused(
                    name,
                    RefusalCode.BAD_ARGS,
                    "every transform number is a number.",
                )
            )
            continue
        if not all(math.isfinite(value) for value in knobs.values()):
            done.append(
                _refused(
                    name,
                    RefusalCode.BAD_ARGS,
                    "every transform number is finite.",
                )
            )
            continue
        changed = (flip, rotation) != (record.flip, record.rotation_deg)
        held = record.transform
        hit: list[str] = []
        if frozen_now:
            if changed:
                hit.append("orientation")
            if any(
                knobs[key] != value
                for key, value in held_knobs(record).items()
            ):
                hit.append("affine")
        frozen = section_refusal(job, name, *hit) if hit else None
        if frozen is not None:
            done.append(Adjustment(refusal=frozen, id=name))
            continue
        orientation = {
            "flip": flip,
            "rotate_quarter": rotation,
            "changed": changed,
        }
        if held is None and not given:
            # No transform and none asked for: the orientation alone.
            if changed:
                pending[name] = (flip, rotation, None)
            done.append(
                Adjustment(id=name, orientation=orientation, written=changed)
            )
            continue
        if record.position_mm is None:
            done.append(
                _refused(
                    name, RefusalCode.NO_POSITION, NO_POSITION.format(name)
                )
            )
            continue
        stored_pivot = physical_knobs(held).pivot
        pivot = None if stored_pivot is None else list(stored_pivot)
        twin = replace(record, flip=flip, rotation_deg=rotation)
        try:
            staged = stage(
                workspace, state, twin, knobs, pivot if pivot else "canvas"
            )
        except Refused as failure:
            done.append(_refused(name, failure.code, failure.message))
            continue
        transform = transform_record(
            size=staged.section.size,
            um_per_px=staged.um_per_px,
            calibration=staged.calibration,
            pivot=staged.pivot_in_section,
            pivot_frac=staged.pivot_frac,
            knobs=staged.params,
        )
        if "affine" in frozen_now:
            wrote, written_transform = (
                changed,
                held,
            )  # nothing of the affine moves
        elif "orientation" in frozen_now and reverses_orientation(
            record, transform
        ):
            mirrored = _mirror_refusal(job, name)
            done.append(Adjustment(refusal=mirrored, id=name))
            continue
        else:
            wrote = changed or not same_transform(held, transform)
            written_transform = transform
        if wrote:
            pending[name] = (flip, rotation, written_transform)
        done.append(
            Adjustment(
                staged=staged,
                previous=held,
                transform=transform,
                written=wrote,
                id=name,
                orientation=orientation,
                kept=(
                    held_knobs(record)
                    if changed and held is not None
                    else None
                ),
            )
        )
    written: list[str] = []
    if pending:
        with job.writing():
            before = job.snapshot()
            for name, (flip, rotation, transform) in pending.items():
                target = job.state.by_id(name)
                if target is None:
                    continue
                target.flip, target.rotation_deg, target.transform = (
                    flip,
                    rotation,
                    transform,
                )
                written.append(name)
            if written:
                job.commit(before)
    return Adjusted(entries=done, written=written)


def _refused(section_id: str, code: str, message: str) -> Adjustment:
    """Return an entry refused for its own reason; its row names the section
    after the message.
    """
    return Adjustment(
        refusal=Refused(code, message, id=section_id), id=section_id
    )


def _mirror_refusal(job: Job, section_id: str) -> Refused:
    """Return the section's busy or ``FROZEN`` refusal of a transform that
    would mirror it under a frozen orientation, with the mirror's message.
    """
    message = mirror_message(section_id)
    frozen = section_refusal(job, section_id, "orientation")
    if frozen is None:  # the caller saw its orientation frozen
        return Refused(
            RefusalCode.FROZEN, properties=["orientation"], message=message
        )
    return Refused(frozen.code, **{**frozen.facts, "message": message})


# --- the tool ----------------------------------------------------------------


def interactive_transform_tool(
    run: Run,
    sections: Sequence[SectionTransform],
    *,
    view: bool = True,
) -> dict[str, Any]:
    """Run the ``interactive_transform`` tool: write, answer, picture.

    Parameters
    ----------
    run
        The run the tool is called in.
    sections, view
        The declared arguments as the model sent them
        (``tools/descriptions/changing.py``).

    Returns
    -------
    dict
        The reply: ``status``, one row per entry, the rows the write changed.
    """
    job = run.job
    entries = as_list(sections)
    if not entries:
        raise Refused(
            RefusalCode.BAD_ARGS, "sections must name one or more sections"
        )
    check_section_count(job, "interactive_transform", len(entries))
    _check_named_once(job, entries)
    done = interactive_transform(job, run.workspace, entries)

    rows = [_entry_row(entry) for entry in done.entries]
    in_picture = [entry.id for entry in done.entries if entry.refusal is None]
    status = "ok" if in_picture else "error"
    result: dict[str, Any] = {"status": status}
    if not in_picture:
        result["error"] = RefusalCode.NOTHING_WRITTEN
    result["results"] = rows
    if done.written:
        result["changed"] = run.changed(done.written)["changed"]
    if not in_picture or not run.shows(view):
        return result

    # A deformation this write made stale goes before the picture is
    # drawn, so the picture shows the section as it now stands.
    cleared = job.clear_stale_deformations()
    if cleared:
        result["deformation_cleared"] = cleared
    looked = show_result(
        job, run.workspace, "interactive_transform", "overlay", in_picture
    )
    return pictured(result, looked)


def _check_named_once(job: Job, entries: Sequence[object]) -> None:
    """Refuse a call that names a section twice (``DUPLICATE_SLICE_IDS``); an
    entry that is not an object is the operation's to refuse.
    """
    objects = [
        cast("Mapping[str, object]", entry)
        for entry in entries
        if isinstance(entry, dict)
    ]
    named = [job.state.resolve(entry.get("id", "")) for entry in objects]
    ids = [section.id for section in named if section is not None]
    duplicates = sorted({name for name in ids if ids.count(name) > 1})
    if duplicates:
        raise Refused(
            RefusalCode.DUPLICATE_SLICE_IDS,
            duplicate_ids=duplicates,
            message="A call sets each section once; look at the first "
            "result before a dependent correction.",
        )


def _entry_row(entry: Adjustment) -> dict[str, Any]:
    """Return one entry's row: its refusal (a frozen section's row names it
    first), or its orientation, whether it was written, its knobs and the
    line on a changed orientation.
    """
    if entry.refusal is not None:
        # a refusal that names the section among its facts keeps it there
        first = (
            entry.id if entry.id and "id" not in entry.refusal.facts else None
        )
        return entry.refusal.payload(section_id=first)
    row: dict[str, Any] = {
        "id": entry.id,
        "status": "ok",
        "orientation": entry.orientation,
        "written": entry.written,
    }
    if entry.transform is not None:
        physical = entry.transform.physical
        row["transform"] = None
        if physical is not None:
            row["transform"] = codec.physical_to_json(physical)
    if entry.message:
        row["message"] = entry.message
    return row

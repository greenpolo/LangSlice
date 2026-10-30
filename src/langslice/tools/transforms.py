"""The in-plane transform: the stored record, from knobs or from a fit, and its write.

Every stored transform has one shape (``Section.transform``): ``kind``
(``interactive``, ``elastix``, or the host's), ``params`` (the six normalized
numbers on the section's working frame, the exact map,
:mod:`langslice.core.geometry.affine`), ``physical`` (the same map as the knobs
about a pivot in canvas fractions), the ``calibration`` it was drawn with, and
``mirrored``. :func:`transform_record` and :func:`fit_transform` build it;
:func:`set_transforms` writes any number of them as one undo step.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from langslice.core.fits.affine import physical_decomposition
from langslice.core.fits.engine_log import engine_output, failure_message
from langslice.core.fits.flags import FitFlag
from langslice.core.geometry.affine import normalized_physical_affine
from langslice.core.jsonable import JsonObject
from langslice.core.refusal import RefusalCode, Refused, ReplyStatus
from langslice.core.sections.damage import exclusions
from langslice.core.stack import codec
from langslice.core.stack.affine import (
    FitRegions,
    PhysicalAffine,
    StoredAffine,
)
from langslice.core.stack.json_reads import flag_at, numbers_at, object_at
from langslice.job.settings.needs import marked, render
from langslice.tools.refusal import check_frozen, frozen_row, unknown_sections
from langslice.tools.stale import section_inputs, stale_row

if TYPE_CHECKING:
    from langslice.core.pictures.placement import Staged
    from langslice.core.stack.section import Section
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
    six = normalized_physical_affine(size=size, um_per_px=um_per_px, pivot=pivot, **knobs)
    pivot_at = (pivot_frac[0], pivot_frac[1])
    physical = PhysicalAffine(**knobs, pivot=pivot_at, extra={})
    # reportAny: the staged calibration and the decomposition are plain dicts
    scale = codec.calibration_from_json(cast("JsonObject", calibration))
    decomposed = cast("JsonObject", physical_decomposition(six, size))
    mirrored = flag_at(decomposed, "mirrored")
    return StoredAffine(kind="interactive", params=tuple(six),
                        physical=physical, calibration=scale,
                        mirrored=mirrored, note="")


def same_transform(held: StoredAffine | None, new: StoredAffine) -> bool:
    """Whether *new* is the transform already held, its note aside.

    The same numbers again are a look, not a write: no undo step for it.
    """
    return held is not None and replace(held, note="") == replace(new, note="")


def fit_transform(
    fit: Mapping[str, Any],
    *,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    fit_atlas: str = "template",
) -> StoredAffine:
    """The stored record of one Elastix affine fit (kind ``elastix``).

    *fit* is :func:`langslice.core.fits.affine.fit_elastix`'s ok payload.
    Regions and a non-default atlas image are recorded only when they were
    used.
    """
    regions = None
    if include or exclude:
        regions = FitRegions(include=tuple(include), exclude=tuple(exclude))
    atlas = None if fit_atlas == "template" else fit_atlas
    payload = cast("JsonObject", fit)  # reportAny: the fit's plain dict
    knobs = codec.physical_from_json(object_at(payload, "physical"))
    scale = codec.calibration_from_json(object_at(payload, "calibration"))
    return StoredAffine(kind="elastix", params=numbers_at(payload, "params"),
                        physical=knobs, calibration=scale,
                        mirrored=flag_at(payload, "mirrored"), regions=regions,
                        fit_atlas=atlas)


#: A fit that turns a section further than this from its previous transform
#: reports the turn in its row (``turn_deg``).
LARGE_TURN_DEG = 45.0


def _knobs(transform: StoredAffine | None) -> PhysicalAffine:
    """Return a stored transform's knobs; none set without a transform."""
    if transform is None or transform.physical is None:
        return PhysicalAffine()
    return transform.physical


def _turn_deg(previous: StoredAffine | None, outcome: Mapping[str, Any]) -> float:
    """How far a fit turns the section from its previous transform, in degrees (0-180)."""
    before = float(_knobs(previous).rotation_deg or 0.0)
    after = float((outcome.get("physical") or {}).get("rotation_deg") or 0.0)
    return abs((after - before + 180.0) % 360.0 - 180.0)


#: An affine fit whose scale along either axis changes by more than this
#: ratio either way from the previous transform is flagged ``SCALE_CHANGED``
#: (a section shrinks or swells in processing by much less than that).
SCALE_CHANGE_LIMITS = (0.8, 1.25)
#: An affine fit that changes the shear by more than this is flagged ``SHEAR``.
SHEAR_CHANGE_LIMIT = 0.15


def plausibility_flags(previous: StoredAffine | None,
                       outcome: Mapping[str, Any]) -> list[dict[str, Any]]:
    """An affine fit's implausible changes from the section's *previous*
    transform (none: the identity), as the deformable fits' flags are given:
    ``SCALE_CHANGED`` (``scale``, ``previous``, ``limits`` per axis ratio) and
    ``SHEAR`` (``shear``, ``previous``, ``limit`` on the change)."""
    before = _knobs(previous)
    after = outcome.get("physical") or {}
    scale = [float(after.get(key) or 1.0) for key in ("scale_x", "scale_y")]
    held = [float(value or 1.0) for value in (before.scale_x, before.scale_y)]
    low, high = SCALE_CHANGE_LIMITS
    flags: list[dict[str, Any]] = []
    if any(not low <= new / old <= high for new, old in zip(scale, held, strict=True) if old):
        flags.append({"code": FitFlag.SCALE_CHANGED, "scale": [round(v, 3) for v in scale],
                      "previous": [round(v, 3) for v in held], "limits": [low, high]})
    shear, was = float(after.get("shear") or 0.0), float(before.shear or 0.0)
    if abs(shear - was) > SHEAR_CHANGE_LIMIT:
        flags.append({"code": FitFlag.SHEAR, "shear": round(shear, 3), "previous": round(was, 3),
                      "limit": SHEAR_CHANGE_LIMIT})
    return flags


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


def _mirrors(name: str) -> str:
    return (f"This transform would mirror {name}, which changes its orientation; the "
            "orientation is frozen there, so the transform has to keep the section's "
            "handedness.")


def set_transforms(job: Job, transforms: Mapping[str, StoredAffine]) -> list[str]:
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
        if (record is not None and "orientation" in job.frozen_for(name)
                and reverses_orientation(record, transform)):
            raise Refused(RefusalCode.FROZEN,
                          ids=[name], properties=["orientation"],
                          message=_mirrors(name))
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


# --- elastix_affine ------------------------------------------------------------------

@dataclass(frozen=True)
class AffineFit:
    """What :func:`elastix_affine` did.

    ``rows``: one per section asked, in order: the fit's payload (``id``,
    ``status: ok``, ``physical``, ``params`` (the six stored
    numbers), ``calibration``, ``mirrored``, ``regions`` when restricted)
    or ``{"id", "status": "error", "error", ...}`` (``FROZEN``, the fitter's
    code, ``STALE_INPUT``). ``fitted``: the sections written.
    """

    rows: list[dict[str, Any]] = field(default_factory=list)
    fitted: list[str] = field(default_factory=list)


def refit_refusal(section: Section) -> Refused:
    """Refuse a linear refit of a section that holds a deformation.

    Parameters
    ----------
    section
        The section asked for, which holds a deformation.

    Returns
    -------
    Refused
        ``HAS_DEFORMATION``, saying to undo the deformation first.
    """
    return Refused(
        RefusalCode.HAS_DEFORMATION,
        f"{section.id} holds a deformation, which a new transform would "
        "leave misplaced; undo the deformation first, or fit a warp on it.")


def _fit_elastix(
    job: Job, workspace: Workspace, records: Sequence[Section], *, fit_atlas: str,
    restrict_to: tuple[str, ...], section_image: str = "preprocessed", asked_atlas: str = "",
    mask: str = "none", edges: bool = True,
) -> AffineFit:
    """Fit an Elastix affine per section and write each fit as its transform.

    Each section's current placement is refined against the atlas image
    *fit_atlas* (:func:`langslice.core.fits.affine.fit_elastix`), by the
    *restrict_to* regions only (empty: every region); each section's marked
    regions are left out on their own
    (:func:`langslice.core.sections.damage.exclusions`). The fit reads
    *section_image* (a section image, by name) with the section mask *mask* by
    name (``"none"`` or one the section holds; a stale one is that section's
    row, ``STALE_MASK``) and, with *edges*, the edge pair; every ok row carries
    the ``inputs`` the fit read
    (:class:`langslice.core.fits.inputs.InputsBlock`) and ``estimated`` (null:
    Elastix estimates no weighting). A section that holds a deformation is
    refused (``HAS_DEFORMATION``), and so is one whose affine is frozen, and so
    is a fit that turns over a section whose orientation is frozen
    (:func:`reverses_orientation`; its row says ``FROZEN``). Every fit that
    succeeded is written as ONE undo step (:func:`set_transforms`,
    :func:`fit_transform` records).

    The fits run outside the job's write lock, from the state as it stood; the
    write takes the lock (:meth:`~langslice.job.job.Job.writing`) and refuses a
    section whose inputs changed meanwhile
    (``STALE_INPUT``, that section's row), writing
    the others.
    """
    from langslice.core.fits.affine import fit_elastix
    from langslice.tools import masks as ops_masks
    from langslice.tools.deformable import inputs_block

    state = job.state
    rows: list[dict[str, Any]] = []
    fits: list[tuple[Section, dict[str, Any], tuple[str, ...], tuple[str, ...]]] = []
    expected: dict[str, str] = {}
    for record in records:
        expected[record.id] = section_inputs(state, record)
        frozen = frozen_row(job, record.id, "affine")
        if frozen is not None:
            rows.append(frozen)
            continue
        if record.deformation:
            rows.append(refit_refusal(record).payload(section_id=record.id))
            continue
        kept, dropped = exclusions(record, restrict_to)
        try:
            weights = ops_masks.section_weights(job, record, mask)
        except Refused as refusal:
            rows.append(refusal.payload(section_id=record.id))
            continue
        try:
            with engine_output(job.engine_log, f"Elastix affine of {record.id}"):
                outcome = fit_elastix(state, workspace, record, include=kept, exclude=dropped,
                                      atlas_image=fit_atlas, section_image=section_image,
                                      edges=edges, section_weights=weights)
        except Exception as exc:  # nothing is written for this section
            logger.warning("elastix_affine failed for %s: %s", record.id, exc)
            code = getattr(exc, "code", RefusalCode.FIT_FAILED)
            message = str(exc)
            if code == RefusalCode.FIT_FAILED:
                message = failure_message(
                    f"{type(exc).__name__}: {exc}", job.engine_log,
                    f"the Elastix affine of {record.id} failed")
            failure = Refused(code, message=message)
            rows.append(failure.payload(section_id=record.id))
            continue
        if outcome.get("error") == RefusalCode.FIT_FAILED:
            outcome["message"] = failure_message(
                str(outcome.get("message", "")), job.engine_log,
                f"the Elastix affine of {record.id} failed")
        else:
            text = str(cast(object, outcome.get("message", "")))  # plain dict
            if marked(text):  # the message names another tool
                outcome["message"] = render(text, job.spec.switched_off)
        rows.append(outcome)
        if outcome["status"] != ReplyStatus.OK:
            continue
        outcome["inputs"] = inputs_block(
            state, workspace, record, section_image=section_image,
            atlas_image=asked_atlas or fit_atlas, mask=mask, damage=dropped,
            restrict_to=restrict_to, resolved=fit_atlas).payload()
        outcome["estimated"] = None
        turn = _turn_deg(record.transform, outcome)
        if turn > LARGE_TURN_DEG:
            outcome["turn_deg"] = round(turn, 1)
        flags = plausibility_flags(record.transform, outcome)
        if flags:
            outcome["flags"] = flags
        fits.append((record, outcome, kept, dropped))
    with job.writing():
        records_out: dict[str, StoredAffine] = {}
        for record, outcome, kept, dropped in fits:
            now = state.by_id(record.id)
            if now is None or section_inputs(state, now) != expected[record.id]:
                outcome.clear()
                outcome.update(stale_row(record.id))
                continue
            made = fit_transform(outcome, include=kept, exclude=dropped, fit_atlas=fit_atlas)
            if "orientation" in job.frozen_for(record.id) and reverses_orientation(now, made):
                outcome.clear()
                outcome.update(frozen_row(job, record.id, "orientation") or {},
                               message=_mirrors(record.id))
                continue
            records_out[record.id] = made
        written = set_transforms(job, records_out)
    return AffineFit(rows=rows, fitted=written)


def elastix_fit(
    job: Job, workspace: Workspace, section: Any, *, restrict_to: Sequence[str] = (),
    exclude: Sequence[str] = (), atlas_image: str = "nissl", mask: str = "none",
    marked: bool = True, section_image: str = "preprocessed", edges: bool = True,
) -> Any:
    """One Elastix affine of the section from its current placement, written
    nowhere (a script's ``job.context.fit_elastix``):
    :class:`langslice.core.fits.affine.ElastixFit` (``params``, the six
    normalized numbers; ``report``; ``engine``; ``inputs``, what it read).
    *restrict_to* / *exclude*
    are region entries (checked, ``"CTX:left"`` for one side); *marked* adds
    the section's marked damage regions to *exclude*, as ``elastix_affine``
    leaves them out; *mask* is ``"none"`` or the name of one of the section's
    masks (:mod:`langslice.tools.masks`), its weights under 0.5 left out.

    Refused: ``UNKNOWN_SLICE_IDS``, ``NO_POSITION``, ``BAD_ARGS`` (an image
    the atlas does not have), ``UNKNOWN_IMAGE``, ``UNKNOWN_MASK``,
    ``STALE_MASK``, ``HAS_DEFORMATION``, ``UNKNOWN_REGIONS``, ``NO_SIDES``,
    ``REGION_TOO_SMALL``, ``FIT_FAILED``.
    """
    from dataclasses import replace

    from langslice.core.atlas.images import resolve_image
    from langslice.core.fits.affine import elastix_affine
    from langslice.core.geometry.affine import IDENTITY_PARAMS
    from langslice.tools import masks as ops_masks
    from langslice.tools.deformable import inputs_block, region_entries
    from langslice.tools.images import check_image

    record = job.state.resolve(section)
    if record is None:
        raise unknown_sections(job.state, [str(section)])
    if record.position_mm is None:
        raise Refused(RefusalCode.NO_POSITION,
                      message=_NO_POSITION.format(record.id))
    if record.deformation:
        raise refit_refusal(record)
    try:
        resolved = resolve_image(workspace.atlas, str(atlas_image))
    except ValueError as exc:
        raise Refused(RefusalCode.BAD_ARGS, message=str(exc)) from exc
    section_image = check_image(job, workspace, [record], section_image)
    include = region_entries(workspace, job.state, list(restrict_to))
    dropped = region_entries(workspace, job.state, list(exclude))
    weights = ops_masks.section_weights(job, record, mask)
    marks: tuple[str, ...] = ()
    if marked:
        include, marks = exclusions(record, include)
        dropped = tuple(dict.fromkeys((*dropped, *marks)))
    stored = None if record.transform is None else record.transform.params
    start = list(stored) if stored is not None and len(stored) == 6 else list(IDENTITY_PARAMS)
    try:
        with engine_output(job.engine_log, f"Elastix affine of {record.id}"):
            fit = elastix_affine(job.state, workspace, record, start, include=include,
                                 exclude=dropped, atlas_image=resolved,
                                 section_image=section_image,
                                 edges=edges, section_weights=weights)
    except Refused as refusal:  # its own code, as the tool's row carries it
        told = render(refusal.message, job.spec.switched_off)
        facts = {**refusal.facts, "message": told} if told else refusal.facts
        raise Refused(refusal.code, **facts) from refusal
    except Exception as exc:  # noqa: BLE001 - the engine's failure is the caller's refusal
        raise Refused(RefusalCode.FIT_FAILED,
                      message=f"{record.id}: " + failure_message(
            f"{type(exc).__name__}: {exc}", job.engine_log,
            f"the Elastix affine of {record.id} failed")) from exc
    block = inputs_block(job.state, workspace, record, section_image=section_image,
                         atlas_image=str(atlas_image),
                         mask=str(mask or ops_masks.NO_MASK).strip(), damage=dropped,
                         restrict_to=include, resolved=resolved)
    return replace(fit, inputs=block.payload())


# --- interactive_transform -------------------------------------------------------------

@dataclass(frozen=True)
class Adjustment:
    """One ``interactive_transform`` entry: what it set, or why not."""

    #: ``{"status": "error", "error", ...}`` when the entry was refused
    #: (``BAD_ARGS``, ``FROZEN``, ``UNKNOWN_SLICE_IDS``, ``NO_POSITION``,
    #: ``ATLAS_RENDER_FAILED``, ``BAD_PIVOT``, ...); else None.
    error: dict[str, Any] | None = None
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
        return ("The orientation changed. The knob values were kept as they were; the "
                "transform was rebuilt on the new orientation, so the picture moved.")


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
_NO_POSITION = "{} has no position yet; write its position first."
#: The knobs ``interactive_transform`` takes, with the value of no change.
KNOB_DEFAULTS: dict[str, float] = {
    "rotation_deg": 0.0, "scale_x": 1.0, "scale_y": 1.0,
    "translate_x_mm": 0.0, "translate_y_mm": 0.0, "shear": 0.0,
}


def held_knobs(section: Section) -> dict[str, float]:
    """The six knob values the section's stored transform has (identity without one)."""
    physical = _knobs(section.transform)
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
    from dataclasses import replace

    from langslice.core.pictures.placement import stage
    from langslice.core.stack.section import ROTATIONS

    state = job.state
    done: list[Adjustment] = []
    pending: dict[str, tuple[bool, int, StoredAffine | None]] = {}
    for entry in sections:
        if not isinstance(entry, Mapping):
            entry_refusal = Refused(RefusalCode.BAD_ARGS, _ENTRY)
            done.append(Adjustment(error=entry_refusal.payload()))
            continue
        record = state.resolve(entry.get("id", ""))
        if record is None:
            done.append(Adjustment(error=unknown_sections(
                state, [str(entry.get("id", ""))]).payload()))
            continue
        name = record.id
        frozen_now = job.frozen_for(name)

        def refuse(error: str, message: str, name: str = name,
                   **facts: object) -> None:
            refusal = Refused(error, message, id=name, **facts)
            done.append(Adjustment(error=refusal.payload(), id=name))

        flip = record.flip
        if entry.get("flip") is not None:
            if not isinstance(entry["flip"], bool):
                refuse(RefusalCode.BAD_ARGS, "flip is true or false.")
                continue
            if entry["flip"] != flip and job.spec.off("interactive_transform:flip"):
                refuse(RefusalCode.OPTION_OFF, _OPTION_OFF.format("flip"))
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
                refuse(RefusalCode.BAD_ROTATION, _ROTATIONS,
                       allowed=list(ROTATIONS))
                continue
            if rotation != record.rotation_deg and job.spec.off("interactive_transform:rotate"):
                refuse(RefusalCode.OPTION_OFF,
                       _OPTION_OFF.format("rotate_quarter"))
                continue
        knobs = held_knobs(record)
        given = {key: entry[key] for key in KNOB_DEFAULTS if entry.get(key) is not None}
        try:
            knobs.update({key: float(value) for key, value in given.items()})
        except (TypeError, ValueError):
            refuse(RefusalCode.BAD_ARGS, "every transform number is a number.")
            continue
        if not all(math.isfinite(value) for value in knobs.values()):
            refuse(RefusalCode.BAD_ARGS, "every transform number is finite.")
            continue
        changed = (flip, rotation) != (record.flip, record.rotation_deg)
        held = record.transform
        hit: list[str] = []
        if frozen_now:
            if changed:
                hit.append("orientation")
            if any(knobs[key] != value for key, value in held_knobs(record).items()):
                hit.append("affine")
        row = frozen_row(job, name, *hit) if hit else None
        if row is not None:
            done.append(Adjustment(error=row, id=name))
            continue
        orientation = {"flip": flip, "rotate_quarter": rotation, "changed": changed}
        if held is None and not given:
            # No transform and none asked for: the orientation alone.
            if changed:
                pending[name] = (flip, rotation, None)
            done.append(Adjustment(id=name, orientation=orientation, written=changed))
            continue
        if record.position_mm is None:
            refuse(RefusalCode.NO_POSITION, _NO_POSITION.format(name))
            continue
        stored_pivot = _knobs(held).pivot
        pivot = None if stored_pivot is None else list(stored_pivot)
        twin = replace(record, flip=flip, rotation_deg=rotation)
        try:
            staged = stage(workspace, state, twin, knobs,
                           pivot if pivot else "canvas")
        except Refused as failure:
            refuse(failure.code, message=failure.message)
            continue
        transform = transform_record(
            size=staged.section.size, um_per_px=staged.um_per_px,
            calibration=staged.calibration, pivot=staged.pivot_in_section,
            pivot_frac=staged.pivot_frac, knobs=staged.params,
        )
        if "affine" in frozen_now:
            wrote, written_transform = changed, held  # nothing of the affine moves
        elif "orientation" in frozen_now and reverses_orientation(record, transform):
            done.append(Adjustment(error={
                **(frozen_row(job, name, "orientation") or {}), "message": _mirrors(name)},
                id=name))
            continue
        else:
            wrote = changed or not same_transform(held, transform)
            written_transform = transform
        if wrote:
            pending[name] = (flip, rotation, written_transform)
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

#: Margin added round the regions' bounding box, as a fraction of the canvas.
BOX_MARGIN = 0.05
#: The least magnification a restricted fit's picture zooms to its regions'
#: box for (:func:`zoom_window`); a box covering more of the canvas is drawn
#: whole, its regions highlighted only.
MIN_ZOOM_GAIN = 1.5


def zoom_window(box: Sequence[float] | None) -> list[float] | None:
    """*box* (:func:`region_box`) as the zoom of a restricted fit's picture,
    or None when it would magnify less than :data:`MIN_ZOOM_GAIN` times
    (its larger side over 1 / ``MIN_ZOOM_GAIN`` of the canvas's)."""
    if not box:
        return None
    extent = max(float(box[2]) - float(box[0]), float(box[3]) - float(box[1]))
    return [float(v) for v in box] if extent * MIN_ZOOM_GAIN <= 1.0 else None


def region_box(
    workspace: Workspace, state: Any, record: Section, regions: Sequence[str],
) -> list[float] | None:
    """The atlas regions' bounding box on the section's canvas.

    ``[x0, y0, x1, y1]`` as fractions of the canvas, with a small margin: the
    zoom :func:`langslice.core.pictures.canvas.zoom_box` takes. None when the
    regions are not in the plane there. A side (``CTX:left``) is read as the
    region on both sides.
    """
    from langslice.core.atlas.sides import split_side
    from langslice.core.fits.deformable.plane_images import regions_mask
    from langslice.core.pictures.placement import stage

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
    sections: Sequence[str],
    restrict_to: Sequence[str] = (),
    section_image: str = "preprocessed",
    atlas_image: str = "nissl",
    mask: str = "none",
    edges: bool = True,
) -> AffineFit:
    """Fit an in-plane affine with Elastix, per section; ONE undo step for the fits.

    Each section's current placement is refined by matching its section image
    *section_image* against *atlas_image* (one of the atlas's images,
    :func:`langslice.core.atlas.images.atlas_images`; ``nissl`` is its Nissl
    image where it has one, else its template), with the section mask *mask* by
    name (``"none"`` or one each section holds) and, with *edges*, the edge
    pair. *sections* are one or more filenames (``Stack.resolve``).
    *restrict_to* fits by those atlas regions only (sides allowed; empty: every
    region). Each section's marked damage regions are left out automatically
    (:func:`langslice.core.sections.damage.exclusions`). A section that holds a
    deformation is that section's row ``HAS_DEFORMATION``.

    Every ok row of a call with *restrict_to* carries ``restrict_box``: the
    regions' bounding box on that section's canvas as ``[x0, y0, x1, y1]``
    fractions (``core.pictures.canvas.zoom_box``'s zoom), so a later picture
    can zoom on them; absent when the regions are not in that plane. Refused
    (nothing fitted): ``UNKNOWN_SLICE_IDS``, ``BAD_ARGS`` (``atlas_image`` not
    one of the atlas's images, a region that is not text, no section named),
    ``UNKNOWN_IMAGE``, ``UNKNOWN_MASK``.
    """
    from langslice.core.atlas.images import resolve_image
    from langslice.tools.deformable import refuse_unknown_inputs

    if not sections:
        raise Refused(RefusalCode.BAD_ARGS,
                      message="sections must name one or more sections")

    asked = str(atlas_image or "nissl").strip()
    try:
        image = resolve_image(workspace.atlas, asked)
    except ValueError as exc:
        raise Refused(RefusalCode.BAD_ARGS, message=str(exc)) from exc
    if any(not isinstance(name, str) for name in restrict_to):
        raise Refused(RefusalCode.BAD_ARGS,
                      message="restrict_to lists atlas regions")
    records = []
    for ref in sections:
        record = job.state.resolve(ref)
        if record is None:
            raise unknown_sections(job.state, [str(ref)])
        records.append(record)
    named = refuse_unknown_inputs(job, workspace, records, section_image, mask)
    label = str(mask or "none").strip()
    result = _fit_elastix(job, workspace, records, fit_atlas=image,
                          restrict_to=tuple(restrict_to), section_image=named,
                          asked_atlas=asked, mask=label, edges=bool(edges))
    if restrict_to:
        for row in result.rows:
            record = job.state.by_id(str(row.get("id")))
            if row.get("status") == ReplyStatus.OK and record is not None:
                box = region_box(workspace, job.state, record, restrict_to)
                if box is not None:
                    row["restrict_box"] = box
    return result

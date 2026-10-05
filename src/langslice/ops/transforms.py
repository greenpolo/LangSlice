"""The in-plane transform: the stored record, from knobs or from a fit, and its write.

Every stored transform has one shape (``SliceState.transform``): ``kind``
(``interactive``, ``elastix``, ``silhouette``, or the host's), ``params``
(the six normalized numbers on the section's working frame, the exact map,
:mod:`langslice.core.affine`), ``physical`` (the same map as the knobs about a
pivot in canvas fractions), the ``calibration`` it was drawn with, and
``mirrored``. :func:`interactive_transform` and :func:`fit_transform` build
it; :func:`set_transforms` writes any number of them as one undo step.
"""

from __future__ import annotations

import functools
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langslice.core.affine import normalized_physical_affine
from langslice.core.transform import physical_decomposition
from langslice.ops.inputs import section_inputs, stale_row
from langslice.ops.refusal import Refused

if TYPE_CHECKING:
    from PIL import Image

    from langslice.core.display import DisplayOptions
    from langslice.core.placement import Staged
    from langslice.core.state import SliceState
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job

logger = logging.getLogger(__name__)

def interactive_transform(
    *,
    size: tuple[int, int],
    um_per_px: float,
    calibration: Mapping[str, Any],
    pivot: tuple[float, float] | None,
    pivot_frac: Sequence[float],
    knobs: Mapping[str, float],
    note: str = "",
) -> dict[str, Any]:
    """The stored record of a transform given as knobs (``adjust_transforms``).

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
    method: str,
    fit: Mapping[str, Any],
    *,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    fit_atlas: str = "ara",
) -> dict[str, Any]:
    """The stored record of one ``fit_affine`` result.

    *fit* is the fitter's ok payload (:func:`langslice.core.transform.fit_silhouette`,
    :func:`~langslice.core.transform.fit_elastix`); *method* becomes the
    ``kind``. Regions and a non-default atlas image are recorded only when
    they were used.
    """
    return {
        "kind": method,
        "params": list(fit["params"]),
        "physical": fit["physical"],
        "iou": fit["iou"],
        "calibration": fit["calibration"],
        "mirrored": fit["mirrored"],
        **({"regions": {"include": list(include), "exclude": list(exclude)}}
           if include or exclude else {}),
        **({"fit_atlas": fit_atlas} if method == "elastix" and fit_atlas != "ara" else {}),
    }


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


# --- fit_affine ------------------------------------------------------------------------

def fit_targets(job: Job) -> list[SliceState]:
    """The sections ``fit_affine`` fits when none are named: every positioned,
    undamaged section the host did not lock, in corrected order."""
    return [
        record for record in job.state.in_order()
        if record.position_mm is not None and not record.damaged
        and record.id not in job.locked
    ]


@dataclass(frozen=True)
class AffineFit:
    """What :func:`fit_affine` did.

    ``rows``: one per section asked, in order: the fit's payload (``id``,
    ``status: ok``, ``iou``, ``physical``, ``params`` (the six stored
    numbers), ``calibration``, ``mirrored``, ``regions`` when restricted;
    with pictures, ``image_indexes`` into ``pictures``) or ``{"id", "status":
    "error", "error", ...}`` (``LOCKED``, ``DAMAGED``, the fitter's code,
    ``RENDER_FAILED``). ``fitted``: the sections written.
    """

    rows: list[dict[str, Any]] = field(default_factory=list)
    fitted: list[str] = field(default_factory=list)
    pictures: list[Image.Image] = field(default_factory=list)


def fit_affine(
    job: Job,
    workspace: Workspace,
    records: Sequence[SliceState],
    *,
    method: str = "elastix",
    fit_atlas: str = "ara",
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
    options: DisplayOptions | None = None,
) -> AffineFit:
    """Fit an in-plane affine per section and write each fit as its transform.

    *method* ``elastix`` refines the section's current placement against the
    atlas image *fit_atlas* (``ara`` or ``nissl``,
    :func:`langslice.core.transform.fit_elastix`); ``silhouette`` fits the
    tissue outline from scratch (:func:`~langslice.core.transform.fit_silhouette`).
    *include* / *exclude* restrict the fit to atlas regions (sides allowed).
    The job's rules, per section: a locked section is refused (``LOCKED``),
    a damaged one too unless regions restrict the fit (``DAMAGED``).

    With *options*, each fit is drawn under its new transform
    (:func:`langslice.core.placement.fit_picture`) BEFORE anything is
    written: a section whose fit or picture fails is that section's error
    row and is not written. Every fit that succeeded is written as ONE undo
    step (:func:`set_transforms`, :func:`fit_transform` records).

    The fits run outside the job's write lock, from the state as it stood;
    the write takes the lock (:meth:`~langslice.job.job.Job.writing`) and
    refuses a section whose inputs changed meanwhile
    (:data:`langslice.ops.inputs.STALE_INPUT`, that section's row; its
    picture stays), writing the others.
    """
    from langslice.core.placement import fit_picture
    from langslice.core.transform import FIT_FRAME_KEY, fit_elastix, fit_silhouette

    state = job.state
    fitter = (functools.partial(fit_elastix, atlas_image=fit_atlas) if method == "elastix"
              else fit_silhouette)
    restricted = bool(include or exclude)
    rows: list[dict[str, Any]] = []
    pictures: list[Image.Image] = []
    fits: list[tuple[SliceState, dict[str, Any]]] = []
    expected: dict[str, str] = {}
    for record in records:
        expected[record.id] = section_inputs(state, record)
        if record.id in job.locked:
            rows.append({"id": record.id, "status": "error", "error": "LOCKED"})
            continue
        if record.damaged and not restricted:
            rows.append({"id": record.id, "status": "error", "error": "DAMAGED"})
            continue
        try:
            outcome = fitter(state, workspace, record, include=include, exclude=exclude)
            frame = outcome.pop(FIT_FRAME_KEY, None)
            panels = (fit_picture(workspace, state, record, frame, options)
                      if frame is not None and options is not None else [])
        except Exception as exc:  # nothing is written for this section
            logger.warning("fit_affine failed for %s: %s", record.id, exc)
            rows.append({"id": record.id, "status": "error",
                         "error": getattr(exc, "code", "RENDER_FAILED"),
                         "message": str(exc)})
            continue
        rows.append(outcome)
        if outcome["status"] != "ok":
            continue
        fits.append((record, outcome))
        if options is not None:
            # Every fit returns its picture, however many sections the call fits.
            outcome["image_indexes"] = list(range(len(pictures), len(pictures) + len(panels)))
            pictures.extend(panels)
    with job.writing():
        kept: dict[str, dict[str, Any]] = {}
        for record, outcome in fits:
            now = state.by_id(record.id)
            if now is None or section_inputs(state, now) != expected[record.id]:
                indexes = outcome.get("image_indexes")
                outcome.clear()
                outcome.update(stale_row(record.id))
                if indexes is not None:
                    outcome["image_indexes"] = indexes
                continue
            kept[record.id] = fit_transform(method, outcome, include=include,
                                            exclude=exclude, fit_atlas=fit_atlas)
        written = set_transforms(job, kept)
    return AffineFit(rows=rows, fitted=written, pictures=pictures)


# --- adjust_transforms ---------------------------------------------------------------


@dataclass(frozen=True)
class Adjustment:
    """One ``adjust_transforms`` entry: the transform its knobs make, or why not."""

    #: ``{"status": "error", "error", ...}`` when the entry was refused
    #: (``BAD_ARGS``, ``LOCKED``, ``UNKNOWN_SLICE_IDS``, ``NO_POSITION``,
    #: ``ATLAS_RENDER_FAILED``, ``BAD_PIVOT``, ``RENDER_FAILED``); else None.
    error: dict[str, Any] | None = None
    #: True for an entry refused before its section was looked at (not an
    #: object, or a locked section).
    early: bool = False
    #: The section, its working frame, canvas and pivot
    #: (:class:`langslice.core.placement.Staged`), with the knobs used.
    staged: Staged | None = None
    #: The section's transform before the call.
    previous: dict[str, Any] | None = None
    #: The stored record these knobs make (:func:`interactive_transform`).
    transform: dict[str, Any] | None = None
    #: Whether it differs from *previous* (the same numbers again only redraw).
    written: bool = False
    pictures: list[Image.Image] = field(default_factory=list)


@dataclass(frozen=True)
class Adjusted:
    """What :func:`adjust_transforms` did: one :class:`Adjustment` per entry,
    in order, and the sections written."""

    entries: list[Adjustment] = field(default_factory=list)
    written: list[str] = field(default_factory=list)


def _knobs(entry: Mapping[str, Any], record: SliceState) -> dict[str, float] | dict[str, Any]:
    """The entry's knobs as finite floats, or the refusal payload."""
    shear = entry.get("shear")
    if shear is None:
        # Left out: the shear the section's transform already has stays.
        held = (record.transform or {}).get("physical") or {}
        shear = held.get("shear") or 0.0
    try:
        params = {
            "rotation_deg": float(entry.get("rotation_deg")),  # type: ignore[arg-type]
            "scale_x": float(entry.get("scale_x")),  # type: ignore[arg-type]
            "scale_y": float(entry.get("scale_y")),  # type: ignore[arg-type]
            "translate_x_mm": float(entry.get("translate_x_mm")),  # type: ignore[arg-type]
            "translate_y_mm": float(entry.get("translate_y_mm")),  # type: ignore[arg-type]
            "shear": float(shear),
        }
    except (TypeError, ValueError):
        return {"status": "error", "error": "BAD_ARGS"}
    if not all(math.isfinite(value) for value in params.values()):
        return {"status": "error", "error": "BAD_ARGS",
                "message": "every transform number must be finite"}
    return params


def adjust_transforms(
    job: Job,
    workspace: Workspace,
    entries: Sequence[Any],
    *,
    options: DisplayOptions | None = None,
) -> Adjusted:
    """Set each entry's section to the transform its knobs make; ONE undo step.

    An entry is ``{"id", "rotation_deg", "scale_x", "scale_y",
    "translate_x_mm", "translate_y_mm", "shear"?, "pivot"?, "note"?}``
    (*pivot* "canvas" (default), "tissue" or ``[fx, fy]``
    canvas fractions). A left-out ``shear`` keeps the section's current
    shear (0 without one); an explicit 0 drops it. The record replaces the
    whole transform; the flip and rotation flags are not
    touched. Per entry, refused: a locked section (``LOCKED``), an unknown
    one, one without a position, numbers that are not finite, a canvas or
    pivot that cannot be built (:func:`langslice.core.placement.stage`).

    With *options*, each entry is drawn first
    (:func:`langslice.core.placement.transform_views`; mode ``ab`` adds what
    the section carried before) and an entry whose picture fails is not
    written (``RENDER_FAILED``). The records that differ from the section's
    current transform are written together (:func:`set_transforms`).
    """
    from langslice.core.placement import StageFailure, stage, transform_views

    state = job.state
    done: list[Adjustment] = []
    writes: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            done.append(Adjustment(error={"status": "error", "error": "BAD_ARGS"}, early=True))
            continue
        target = state.resolve(entry.get("id", ""))
        if target is not None and target.id in job.locked:
            done.append(Adjustment(error={"status": "error", "error": "LOCKED",
                                          "id": target.id}, early=True))
            continue
        slice_id = str(entry.get("id", ""))
        record = state.resolve(slice_id)
        if record is None:
            done.append(Adjustment(error={"status": "error", "error": "UNKNOWN_SLICE_IDS",
                                          "unknown": [slice_id]}))
            continue
        if record.position_mm is None:
            done.append(Adjustment(error={"status": "error", "error": "NO_POSITION",
                                          "id": record.id}))
            continue
        params = _knobs(entry, record)
        if "error" in params:
            done.append(Adjustment(error=dict(params)))
            continue
        try:
            staged = stage(workspace, state, record, params, entry.get("pivot", "canvas"))
        except StageFailure as failure:
            done.append(Adjustment(error={"status": "error", "error": failure.code,
                                          "message": str(failure)}))
            continue
        previous = record.transform
        transform = interactive_transform(
            size=staged.section.size, um_per_px=staged.um_per_px,
            calibration=staged.calibration, pivot=staged.pivot_in_section,
            pivot_frac=staged.pivot_frac, knobs=staged.params,
            note=str(entry.get("note", "")),
        )
        wrote = not same_transform(previous, transform)
        pictures: list[Image.Image] = []
        if options is not None:
            try:
                pictures = transform_views(workspace, state, staged, options, previous)
            except Exception as exc:
                logger.warning("adjust_transform failed for %s: %s", record.id, exc)
                done.append(Adjustment(error={"status": "error", "error": "RENDER_FAILED",
                                              "message": str(exc)}))
                continue
        if wrote:
            writes[record.id] = transform
        done.append(Adjustment(staged=staged, previous=previous, transform=transform,
                               written=wrote, pictures=pictures))
    return Adjusted(entries=done, written=set_transforms(job, writes))

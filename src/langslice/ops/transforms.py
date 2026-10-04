"""The in-plane transform: the stored record, from knobs or from a fit, and its write.

Every stored transform has one shape (``SliceState.transform``): ``kind``
(``interactive``, ``elastix``, ``silhouette``, or the host's), ``params``
(the six normalized numbers on the section's working frame, the exact map,
:mod:`langslice.affine`), ``physical`` (the same map as the knobs about a
pivot in canvas fractions), the ``calibration`` it was drawn with, and
``mirrored``. :func:`interactive_transform` and :func:`fit_transform` build
it; :func:`set_transforms` writes any number of them as one undo step.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from langslice.affine import normalized_physical_affine
from langslice.linear.transform import physical_decomposition
from langslice.ops.refusal import Refused

if TYPE_CHECKING:
    from langslice.linear.job import Job

#: The knobs of a transform, in the order every payload lists them. ``shear``
#: is :func:`langslice.affine.decompose_affine`'s (0: none).
KNOBS: tuple[str, ...] = (
    "rotation_deg", "scale_x", "scale_y", "translate_x_mm", "translate_y_mm", "shear",
)


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
    the record keeps beside the knobs. *knobs* are :data:`KNOBS` (``shear``
    may be left out: none); the record's ``physical`` lists all six, so a
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

    *fit* is the fitter's ok payload (:func:`langslice.linear.transform.fit_silhouette`,
    :func:`~langslice.linear.transform.fit_elastix`); *method* becomes the
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

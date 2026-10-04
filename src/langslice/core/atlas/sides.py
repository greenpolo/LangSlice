"""One side of a region: ``"CTX:left"``, as the section is shown.

A region entry anywhere a tool takes regions (acronym or id, descendants
included) may carry a side: ``"CTX"`` is both hemispheres, ``"CTX:left"``
and ``"CTX:right"`` one of them. Damage is often one-sided (cortex torn off
one hemisphere), and excluding a region by name alone also drops the side
that is present.

Left and right are the SECTION'S, in its displayed frame: the oriented
section render (rotation and flip applied) that ``view_slices``,
``fit_deformable`` and ``trace_borders`` pictures show, never the animal's
anatomical left. They are derived, never guessed from the tissue:

1. Each native atlas-plane pixel's medio-lateral volume index comes from
   :func:`langslice.core.oblique.plane_index_coordinates` (the same pixel grid as
   ``annotation_slice``, cutting angles included) on the ML axis that
   :func:`langslice.core.space.atlas_space_context` derives from the atlas
   orientation through ``brainglobe_space``.
2. The two halves are split where BrainGlobe splits a symmetric atlas
   (``BrainGlobeAtlas.hemispheres``: index ``round(n_ml / 2)`` starts the
   second half); an atlas whose metadata says it is not symmetric is split
   by its own ``hemispheres`` volume. Which half BrainGlobe calls "left" is
   NOT used — its own docstring and code disagree about it, and on a
   symmetric atlas no label agreement could settle it.
3. The linear placement (native plane -> section pixels) carries the ML
   index gradient into the section frame; the half whose ML direction points
   toward smaller section x is the section's left. A mirrored placement thus
   swaps the halves, as it swaps the tissue.

A sagittal plane lies within one hemisphere (``NO_SIDES``), and a placement
that turns the midline closer to horizontal than vertical has no left or
right (``SIDES_AMBIGUOUS``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

SIDES: tuple[str, ...] = ("left", "right")
#: Separator between a region and its side.
SEPARATOR = ":"


class SideError(ValueError):
    """A side the placement cannot define, with its error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def split_side(entry: str | int) -> tuple[str, str | None]:
    """``(region, side or None)`` of one entry; ``ValueError`` for an unknown side."""
    text = str(entry).strip()
    if SEPARATOR not in text:
        return text, None
    region, _, side = text.rpartition(SEPARATOR)
    side = side.strip().lower()
    if side not in SIDES:
        raise ValueError(f"Unknown side {side!r} in {text!r}: use {' or '.join(SIDES)}")
    return region.strip(), side


def has_sides(entries: Any) -> bool:
    """Whether any entry names one side."""
    return any(split_side(entry)[1] is not None for entry in entries or ())


def overlapping(first: Any, second: Any) -> list[str]:
    """Entries of *first* that cover part of an entry of *second* (case-insensitive).

    ``CTX`` overlaps ``CTX:left``; ``CTX:left`` does not overlap ``CTX:right``.
    """
    others = [(region.lower(), side) for region, side in map(split_side, second or ())]
    found = []
    for entry in first or ():
        region, side = split_side(entry)
        if any(region.lower() == other and (side is None or theirs is None or side == theirs)
               for other, theirs in others):
            found.append(str(entry).strip())
    return found


def ml_halves(
    atlas: Any, position_mm: float, plane: str, pitch_deg: float = 0.0, yaw_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """``(high, gradient)`` on the native plane grid.

    *high* marks the pixels in the half with the larger ML volume index;
    *gradient* is (d ML / d native x, d ML / d native y), in volume voxels per
    native pixel (the plane is flat, so one vector holds everywhere).
    """
    from langslice.core.oblique import plane_index_coordinates
    from langslice.core.space import atlas_space_context

    if plane == "sagittal":
        raise SideError("NO_SIDES", "A sagittal section lies within one hemisphere, so a "
                        "region cannot be limited to one side.")
    context = atlas_space_context(atlas)
    axis = context.ml_axis_index
    coords = plane_index_coordinates(atlas, position_mm, plane, pitch_deg, yaw_deg)  # type: ignore[arg-type]
    ml = np.asarray(coords[axis], dtype=np.float64)
    if ml.shape[0] < 2 or ml.shape[1] < 2:
        raise SideError("NO_SIDES", "The atlas plane is too small to have sides.")
    gradient = np.array([ml[0, 1] - ml[0, 0], ml[1, 0] - ml[0, 0]], dtype=np.float64)
    count = context.shape[axis]
    metadata = getattr(atlas, "metadata", None) or {}
    if metadata.get("symmetric", True) is False and getattr(atlas, "hemispheres", None) is not None:
        index = np.rint(coords).astype(np.intp)
        for k, size in enumerate(context.shape):
            np.clip(index[k], 0, size - 1, out=index[k])
        values = np.asarray(atlas.hemispheres)[index[0], index[1], index[2]]
        present = [int(v) for v in np.unique(values) if v]
        if len(present) < 2:
            # The plane misses one hemisphere: compare with the volume's centre.
            high = ml >= (count / 2.0 - 0.5)
        else:
            means = {v: float(ml[values == v].mean()) for v in present}
            top = max(means, key=lambda v: means[v])
            high = values == top
    else:
        high = ml >= (round(count / 2) - 0.5)
    return high, gradient


def native_left(
    atlas: Any,
    position_mm: float,
    plane: str,
    pitch_deg: float,
    yaw_deg: float,
    native_to_display: np.ndarray,
) -> np.ndarray:
    """Native-plane pixels on the displayed LEFT, for a placement *native_to_display*.

    *native_to_display* is the placement's linear part (2x2, or the 3x3/2x3
    matrix; only its first two columns of the first two rows are read)
    mapping native plane pixels to the displayed section frame.
    """
    high, gradient = ml_halves(atlas, position_mm, plane, pitch_deg, yaw_deg)
    linear = np.asarray(native_to_display, dtype=np.float64)[:2, :2]
    if not np.isfinite(linear).all() or abs(np.linalg.det(linear)) < 1e-12:
        raise SideError("SIDES_AMBIGUOUS", "The placement is degenerate, so it has no sides.")
    # d ML / d display = inverse(L)^T @ d ML / d native.
    shown = np.linalg.inv(linear).T @ gradient
    if not np.isfinite(shown).all() or np.hypot(*shown) < 1e-9:
        raise SideError("NO_SIDES", "The atlas plane does not cross the midline.")
    if abs(shown[0]) <= abs(shown[1]):
        raise SideError("SIDES_AMBIGUOUS", "This placement turns the midline closer to "
                        "horizontal than vertical, so left and right are undefined.")
    # ML increasing toward the right: the low half is on the left.
    return ~high if shown[0] > 0 else high


def restrict(mask: np.ndarray, side: str | None, left: np.ndarray | None) -> np.ndarray:
    """*mask* limited to one side (unchanged for no side)."""
    if side is None:
        return mask
    if left is None:
        raise ValueError("A one-sided region needs the placement's sides")
    return mask & (left if side == "left" else ~left)

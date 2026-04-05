"""Nano-banana search window construction."""

from __future__ import annotations

import math
from dataclasses import dataclass

_IMAGE_SPACING_MM = 0.025
_MIN_IMAGES = 5
_MAX_IMAGES = 13


@dataclass(frozen=True)
class RefinementWindow:
    """Search window for a single slice's nano-banana refinement."""

    lo: float
    hi: float
    center: float
    n_images: int
    skip: bool


def compute_refinement_window(
    *,
    position_mm: float,
    left_locked_mm: float | None,
    right_locked_mm: float | None,
    thickness_mm: float,
    interval_mm: float,
) -> RefinementWindow:
    """Compute the atlas search window for one slice's nano-banana pass.

    Bounds are derived from locked neighbors.  If no locked neighbor exists on
    a side, fall back to ``position_mm +/- interval_mm``.  Returns ``skip=True``
    when the window is narrower than *thickness_mm*.
    """
    if left_locked_mm is not None:
        lo = left_locked_mm + thickness_mm
    else:
        lo = position_mm - interval_mm

    if right_locked_mm is not None:
        hi = right_locked_mm - thickness_mm
    else:
        hi = position_mm + interval_mm

    width = hi - lo

    # lo > hi means the thickness margins have crossed — no valid position exists.
    if width < thickness_mm:
        return RefinementWindow(lo=lo, hi=hi, center=position_mm, n_images=0, skip=True)

    n_images = max(_MIN_IMAGES, min(_MAX_IMAGES, math.floor(width / _IMAGE_SPACING_MM)))

    return RefinementWindow(lo=lo, hi=hi, center=position_mm, n_images=n_images, skip=False)

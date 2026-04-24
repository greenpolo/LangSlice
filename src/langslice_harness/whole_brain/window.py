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


def compute_search_bounds(
    *,
    center_mm: float,
    atlas_range: tuple[float, float],
    window_half_mm: float = 3.0,
    edge_anchor_mm: float | None = None,
    edge_side: str | None = None,
    edge_padding_mm: float = 1.0,
) -> tuple[float, float]:
    """Return AP search bounds for a slice estimation pass.

    Interior slices use a symmetric ``center_mm +/- window_half_mm`` window.
    Extrapolated edge slices can instead use a tighter asymmetric window tied
    to the nearest outer anchor to avoid flooding the model with irrelevant
    far-posterior or far-anterior atlas sections.
    """
    lo, hi = atlas_range

    if edge_side is None:
        return max(lo, center_mm - window_half_mm), min(hi, center_mm + window_half_mm)

    if edge_anchor_mm is None:
        raise ValueError("edge_anchor_mm is required when edge_side is set")

    if edge_side == "leading":
        bounded_lo = max(lo, center_mm - edge_padding_mm)
        bounded_hi = min(hi, edge_anchor_mm + edge_padding_mm)
    elif edge_side == "trailing":
        bounded_lo = max(lo, edge_anchor_mm - edge_padding_mm)
        bounded_hi = min(hi, center_mm + edge_padding_mm)
    else:
        raise ValueError(f"Unsupported edge_side: {edge_side}")

    if bounded_lo >= bounded_hi:
        return max(lo, center_mm - window_half_mm), min(hi, center_mm + window_half_mm)

    return bounded_lo, bounded_hi


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

"""Plane-relative tolerance helpers and trace accuracy categorisation."""

from __future__ import annotations

import math
from typing import Any

from langslice_harness.atlas.core import get_position_range_mm, load_atlas

from .constants import RESCUE_PCT, TOLERANCE_FLOOR_MM, TOLERANCE_PCT


def plane_tolerance_mm(atlas_name: str, plane: str) -> float:
    """Atlas-aware accept tolerance for `categorize_trace`.

    For sagittal we use canonical-hemisphere extent (full ML extent / 2),
    matching the production constraint that the agent estimates within one
    hemisphere only.
    """
    atlas = load_atlas(atlas_name)
    _, hi = get_position_range_mm(atlas, plane=plane)
    extent = hi
    if plane == "sagittal":
        extent = extent / 2.0
    return max(extent * TOLERANCE_PCT, TOLERANCE_FLOOR_MM)


def plane_rescue_threshold_mm(atlas_name: str, plane: str) -> float:
    """Atlas-aware rescue (near-miss) threshold."""
    atlas = load_atlas(atlas_name)
    _, hi = get_position_range_mm(atlas, plane=plane)
    extent = hi
    if plane == "sagittal":
        extent = extent / 2.0
    return max(extent * RESCUE_PCT, TOLERANCE_FLOOR_MM * 4)


def canonicalize_positions(
    positions: list[float] | None, atlas_name: str, plane: str
) -> list[float] | None:
    """Map sagittal positions to canonical (single-hemisphere) form.

    The two hemispheres are mirror images so any GT (or model submission) in
    the upper-half ML range is equivalent to its mirror in the lower-half.
    Comparing GT and submission in canonical space removes the mirror flip
    that otherwise drives ~50% of sagittal "errors". No-op for coronal/horizontal.
    """
    if positions is None or plane != "sagittal" or not positions:
        return positions
    atlas = load_atlas(atlas_name)
    _, hi = get_position_range_mm(atlas, plane=plane)
    return [min(float(p), hi - float(p)) for p in positions]


def categorize_trace(
    *,
    truth_positions: list[float],
    submitted_positions: list[float] | None,
    fetched_positions: list[float],
    tolerance_mm: float,
    turn_count: int,
    median_turn_count: int,
    had_broad_restart: bool,
    submit_rejections: int,
    fallback: bool,
) -> dict[str, Any]:
    """Categorize a trace for SFT filtering and hard-negative accounting."""

    if fallback:
        return {"accepted": False, "label": "fallback"}
    if not submitted_positions:
        return {"accepted": False, "label": "no_submit"}
    if len(submitted_positions) != len(truth_positions):
        return {"accepted": False, "label": "length_mismatch"}

    errors = [abs(a - b) for a, b in zip(submitted_positions, truth_positions, strict=True)]
    max_error = max(errors) if errors else math.inf
    accepted = max_error <= tolerance_mm
    if not accepted:
        return {
            "accepted": False,
            "label": "out_of_tolerance",
            "max_error_mm": round(max_error, 4),
        }

    excursion = 0.0
    if fetched_positions and submitted_positions:
        centers = submitted_positions
        excursion = max(
            min(abs(fetched - center) for center in centers)
            for fetched in fetched_positions
        )
    hard = (
        turn_count > median_turn_count
        or had_broad_restart
        or submit_rejections > 0
    )
    return {
        "accepted": True,
        "label": "hard_negative_recovered" if hard else "clean_success",
        "max_error_mm": round(max_error, 4),
        "trajectory_excursion_mm": round(excursion, 4),
    }

"""Nano-banana search window construction."""

from __future__ import annotations


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

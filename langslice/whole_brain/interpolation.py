"""Interval-based interpolation and extrapolation of slice positions."""

from __future__ import annotations

from langslice.whole_brain.types import SlicePosition


def interpolate_positions(
    slices: list[SlicePosition],
    *,
    interval_mm: float,
    atlas_range: tuple[float, float],
    z_axis: str,
) -> list[SlicePosition]:
    """Assign positions to unlocked slices based on locked anchors.

    Between adjacent anchors: distribute evenly (interval as baseline, residual
    spread across gaps).  Beyond outermost anchors: step outward using
    *interval_mm*, clamped to *atlas_range*.

    Returns a new list — locked slices are copied unchanged.
    """
    lo, hi = atlas_range
    n = len(slices)
    if n == 0:
        return []

    # Copy so we don't mutate the input
    out = [
        SlicePosition(s.filename, s.index, s.position_mm, s.source, s.locked)
        for s in slices
    ]

    # Direction multiplier: AP means increasing index -> increasing mm
    direction = 1.0 if z_axis == "AP" else -1.0
    step = interval_mm * direction

    # Collect locked anchor indices
    anchor_idxs = [i for i, s in enumerate(out) if s.locked]
    if not anchor_idxs:
        return out

    # --- Interpolate between each pair of adjacent anchors ---
    for a_idx, b_idx in zip(anchor_idxs, anchor_idxs[1:], strict=False):
        a_pos = out[a_idx].position_mm
        b_pos = out[b_idx].position_mm
        n_gaps = b_idx - a_idx
        if n_gaps <= 1:
            continue
        gap_step = (b_pos - a_pos) / n_gaps
        for k in range(1, n_gaps):
            i = a_idx + k
            out[i] = SlicePosition(
                out[i].filename,
                out[i].index,
                a_pos + k * gap_step,
                "interpolated",
                locked=False,
            )

    # --- Extrapolate before the first anchor ---
    first_anchor = anchor_idxs[0]
    for k in range(1, first_anchor + 1):
        i = first_anchor - k
        pos = out[first_anchor].position_mm - k * step
        pos = max(lo, min(hi, pos))
        out[i] = SlicePosition(
            out[i].filename, out[i].index, pos, "extrapolated", locked=False
        )

    # --- Extrapolate after the last anchor ---
    last_anchor = anchor_idxs[-1]
    for k in range(1, n - last_anchor):
        i = last_anchor + k
        pos = out[last_anchor].position_mm + k * step
        pos = max(lo, min(hi, pos))
        out[i] = SlicePosition(
            out[i].filename, out[i].index, pos, "extrapolated", locked=False
        )

    return out

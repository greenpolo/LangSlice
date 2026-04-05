"""Center-out anchor index selection.

Anchors are placed starting from the center of the slice list and expanding
outward.  Anterior brain slices (especially in mouse) lack visually distinct
tissue and are unreliable for AP estimation, so we avoid placing anchors at
the extremes unless the user requests enough to cover the full range.
"""

from __future__ import annotations


def select_anchor_indices(n_slices: int, n_anchors: int) -> list[int]:
    """Return sorted 0-based indices for anchor slices.

    Places anchors center-out: the midpoint is chosen first, then positions
    expand symmetrically.  When *n_anchors* >= *n_slices*, every slice is an
    anchor.
    """
    if n_slices <= 0:
        return []
    n_anchors = min(n_anchors, n_slices)
    if n_anchors == n_slices:
        return list(range(n_slices))

    # Evenly space n_anchors points across the range, offset inward from edges.
    # gap = total_range / (n_anchors + 1) keeps anchors away from index 0 and n-1.
    gap = n_slices / (n_anchors + 1)
    indices: list[int] = []
    for k in range(1, n_anchors + 1):
        idx = int(round(k * gap)) - 1  # -1 for 0-based
        idx = max(0, min(idx, n_slices - 1))
        if idx not in indices:
            indices.append(idx)

    # If rounding caused duplicates or we're short, fill from center outward
    if len(indices) < n_anchors:
        mid = n_slices // 2
        for offset in range(n_slices):
            for candidate in [mid + offset, mid - offset]:
                if 0 <= candidate < n_slices and candidate not in indices:
                    indices.append(candidate)
                    if len(indices) == n_anchors:
                        break
            if len(indices) == n_anchors:
                break

    indices.sort()
    return indices

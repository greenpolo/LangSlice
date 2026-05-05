"""Hemisphere-split bbox computation.

Two backends:
- `bbox_from_atlas_slice`: consumes a 2D annotation array (atlas-aligned).
- `bbox_from_real_section`: projects probes through a per-section pixel->voxel
  registration helper (see `_local/eval/lib/registration.py`).

Coverage gate: the qualifying pixel/probe area must be >=1% and <=40% of total
image area, else the bbox fails. Whole-brain coronal/horizontal returns
`{left, right}`; either side empty causes the example to fail (returns None).
Sagittal / hemisphere returns a single bbox.
"""
from __future__ import annotations

import numpy as np

_COVERAGE_MIN = 0.01
_COVERAGE_MAX = 0.40


def _bbox_of_mask(mask: np.ndarray) -> list[int] | None:
    """Return [x1, y1, x2, y2] or None if mask is empty."""
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def bbox_from_atlas_slice(
    annotation_slice: np.ndarray,
    region_ids: set[int],
    is_hemisphere: bool,
    hemisphere_slice: np.ndarray | None = None,
) -> dict | list | None:
    """Compute hemisphere-split bbox from a 2D atlas annotation.

    Returns:
        - {"left": [...], "right": [...]} for whole-brain coronal/horizontal.
        - [...] for sagittal or hemisphere sections.
        - None if coverage gate fails or whole-brain has empty side.
    """
    if annotation_slice.ndim != 2:
        raise ValueError(
            f"annotation_slice must be 2D, got shape {annotation_slice.shape}"
        )
    h, w = annotation_slice.shape
    total = h * w

    mask = np.isin(annotation_slice, list(region_ids))
    coverage = float(mask.sum()) / float(total)
    if coverage < _COVERAGE_MIN or coverage > _COVERAGE_MAX:
        return None

    if is_hemisphere:
        return _bbox_of_mask(mask)

    if hemisphere_slice is None:
        raise ValueError("hemisphere_slice is required for whole-brain atlas bboxes")
    if hemisphere_slice.shape != annotation_slice.shape:
        raise ValueError(
            "hemisphere_slice must match annotation_slice shape, got "
            f"{hemisphere_slice.shape} vs {annotation_slice.shape}"
        )

    left_mask = mask & (hemisphere_slice == 1)
    right_mask = mask & (hemisphere_slice == 2)

    left_bbox = _bbox_of_mask(left_mask)
    right_bbox = _bbox_of_mask(right_mask)
    if left_bbox is None or right_bbox is None:
        return None
    return {"left": left_bbox, "right": right_bbox}

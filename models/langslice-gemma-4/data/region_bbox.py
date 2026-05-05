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

from collections.abc import Callable

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


def bbox_from_real_section(
    section_image_shape: tuple[int, int],
    pixel_to_voxel: Callable[[float, float], np.ndarray],
    annotation_volume: np.ndarray,
    region_ids: set[int],
    midline_x_at_row: Callable[[float], float],
    is_hemisphere: bool,
    grid_step: int = 8,
) -> dict | list | None:
    """Compute hemisphere-split bbox by projecting probes into atlas voxel space.

    Args:
        section_image_shape: (H, W) of the section image (post-resize).
        pixel_to_voxel: returns QuickNII voxel coords as (x_ml, y_ap, z_dv)
            for a given section pixel (i, j). This matches
            `_local/eval/lib/registration.py::pixel_to_voxel`.
        annotation_volume: 3D BrainGlobe annotation, shape (AP, DV, ML).
        region_ids: BrainGlobe region IDs to match (use `LandmarkLoader.resolve`).
        midline_x_at_row: returns the pixel x-coord of the projected atlas
            midline at a given row i. For atlas-symmetric sections, a simple
            constant `lambda _: W/2`.
        is_hemisphere: if True, return a single bbox; otherwise hemisphere-split.
        grid_step: probe stride in pixels.

    Returns:
        Same shape as `bbox_from_atlas_slice`. None on coverage-gate failure
        or empty hemisphere on whole-brain.
    """
    H, W = section_image_shape
    AP, DV, ML = annotation_volume.shape

    is_left = np.zeros((H, W), dtype=bool)
    is_match = np.zeros((H, W), dtype=bool)

    for i in range(0, H, grid_step):
        midline_at_i = float(midline_x_at_row(float(i)))
        for j in range(0, W, grid_step):
            voxel = pixel_to_voxel(float(j), float(i))
            apv, dvv, mlv = quicknii_to_brainglobe_indices(voxel)
            if not (0 <= apv < AP and 0 <= dvv < DV and 0 <= mlv < ML):
                continue
            ann = int(annotation_volume[apv, dvv, mlv])
            if ann in region_ids:
                is_match[i, j] = True
                if j < midline_at_i:
                    is_left[i, j] = True

    n_match = int(is_match.sum())
    n_probes = (H // grid_step) * (W // grid_step)
    if n_probes == 0:
        return None
    coverage = n_match / float(n_probes)
    if coverage < _COVERAGE_MIN or coverage > _COVERAGE_MAX:
        return None

    def _padded_bbox(mask: np.ndarray) -> list[int] | None:
        bbox = _bbox_of_mask(mask)
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - grid_step)
        y1 = max(0, y1 - grid_step)
        x2 = min(W - 1, x2 + grid_step)
        y2 = min(H - 1, y2 + grid_step)
        return [x1, y1, x2, y2]

    if is_hemisphere:
        return _padded_bbox(is_match)

    left_mask = is_match & is_left
    right_mask = is_match & ~is_left

    left_bbox = _padded_bbox(left_mask)
    right_bbox = _padded_bbox(right_mask)
    if left_bbox is None or right_bbox is None:
        return None
    return {"left": left_bbox, "right": right_bbox}


def quicknii_to_brainglobe_indices(voxel: np.ndarray) -> tuple[int, int, int]:
    """Convert QuickNII (x_ml, y_ap, z_dv) to BrainGlobe (AP, DV, ML)."""
    x_ml, y_ap, z_dv = float(voxel[0]), float(voxel[1]), float(voxel[2])
    return (int(round(y_ap)), int(round(z_dv)), int(round(x_ml)))


def scale_bbox(bbox: dict | list, scale_x: float, scale_y: float) -> dict | list:
    """Scale bbox coordinates into a resized image frame."""
    if isinstance(bbox, dict):
        return {
            side: scale_bbox(coords, scale_x=scale_x, scale_y=scale_y)
            for side, coords in bbox.items()
            if coords is not None
        }
    x1, y1, x2, y2 = bbox
    return [
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
        int(round(x2 * scale_x)),
        int(round(y2 * scale_y)),
    ]

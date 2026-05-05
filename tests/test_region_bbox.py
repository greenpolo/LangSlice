"""Tests for region_bbox: hemisphere-split bbox computation."""
from __future__ import annotations

import numpy as np
import region_bbox


def test_atlas_slice_whole_brain_returns_left_and_right():
    # 100-px-wide annotation: region ID {1} appears in both hemispheres.
    annotation = np.zeros((100, 100), dtype=np.int32)
    annotation[20:50, 10:40] = 1   # left side, region 1
    annotation[20:50, 60:90] = 1   # right side, region 1
    hemispheres = np.zeros((100, 100), dtype=np.int32)
    hemispheres[:, :50] = 1
    hemispheres[:, 50:] = 2
    bbox = region_bbox.bbox_from_atlas_slice(
        annotation_slice=annotation,
        hemisphere_slice=hemispheres,
        region_ids={1},
        is_hemisphere=False,
    )
    assert bbox == {
        "left": [10, 20, 39, 49],
        "right": [60, 20, 89, 49],
    }


def test_atlas_slice_whole_brain_drops_when_one_side_empty():
    annotation = np.zeros((100, 100), dtype=np.int32)
    annotation[20:50, 10:40] = 1   # left only
    hemispheres = np.zeros((100, 100), dtype=np.int32)
    hemispheres[:, :50] = 1
    hemispheres[:, 50:] = 2
    bbox = region_bbox.bbox_from_atlas_slice(
        annotation_slice=annotation,
        hemisphere_slice=hemispheres,
        region_ids={1},
        is_hemisphere=False,
    )
    assert bbox is None  # whole-brain example dropped if either side empty


def test_atlas_slice_hemisphere_returns_single_bbox():
    annotation = np.zeros((100, 100), dtype=np.int32)
    annotation[20:50, 10:40] = 1
    bbox = region_bbox.bbox_from_atlas_slice(
        annotation_slice=annotation,
        region_ids={1},
        is_hemisphere=True,
    )
    assert bbox == [10, 20, 39, 49]


def test_atlas_slice_coverage_gate_too_small():
    annotation = np.zeros((100, 100), dtype=np.int32)
    annotation[0, 0] = 1  # 1 pixel out of 10000 = 0.01%, below 1% gate
    bbox = region_bbox.bbox_from_atlas_slice(
        annotation_slice=annotation,
        region_ids={1},
        is_hemisphere=True,
    )
    assert bbox is None


def test_atlas_slice_coverage_gate_too_large():
    annotation = np.full((100, 100), 1, dtype=np.int32)  # 100% of image
    bbox = region_bbox.bbox_from_atlas_slice(
        annotation_slice=annotation,
        region_ids={1},
        is_hemisphere=True,
    )
    assert bbox is None


def test_real_section_identity_transform_matches_atlas_path():
    """Identity pixel→voxel: realhist path should match atlas-slice path."""
    H, W = 96, 96
    # Tiny BrainGlobe volume is the same as a single annotation slice. The fake
    # QuickNII transform returns (x_ml, y_ap, z_dv), so x=col, AP=0, z=row.
    annotation_volume = np.zeros((1, H, W), dtype=np.int32)
    annotation_volume[0, 20:50, 10:40] = 1   # left
    annotation_volume[0, 20:50, 60:90] = 1   # right

    def pixel_to_voxel(i: float, j: float) -> np.ndarray:
        return np.array([i, 0.0, j], dtype=np.float64)  # QuickNII: (x_ml, y_ap, z_dv)

    def midline_x(i: float) -> float:  # vertical midline at x=W/2 in pixel coords
        return W / 2.0

    bbox = region_bbox.bbox_from_real_section(
        section_image_shape=(H, W),
        pixel_to_voxel=pixel_to_voxel,
        annotation_volume=annotation_volume,
        region_ids={1},
        midline_x_at_row=midline_x,
        is_hemisphere=False,
        grid_step=4,
    )
    assert bbox is not None
    assert "left" in bbox and "right" in bbox
    # With grid_step=4 and ~30x30 region, expect coords near the atlas truth
    # (10,20)-(39,49) and (60,20)-(89,49) within one grid step of padding.
    assert bbox["left"][0] <= 12
    assert bbox["left"][1] <= 22
    assert bbox["left"][2] >= 36
    assert bbox["left"][3] >= 47
    assert bbox["right"][0] <= 62
    assert bbox["right"][2] >= 86


def test_real_section_hemisphere_returns_single_bbox():
    H, W = 96, 96
    annotation_volume = np.zeros((1, H, W), dtype=np.int32)
    annotation_volume[0, 20:50, 10:40] = 1

    def pixel_to_voxel(i: float, j: float) -> np.ndarray:
        return np.array([i, 0.0, j], dtype=np.float64)

    bbox = region_bbox.bbox_from_real_section(
        section_image_shape=(H, W),
        pixel_to_voxel=pixel_to_voxel,
        annotation_volume=annotation_volume,
        region_ids={1},
        midline_x_at_row=lambda i: W / 2.0,
        is_hemisphere=True,
        grid_step=4,
    )
    assert isinstance(bbox, list)
    assert len(bbox) == 4


def test_real_section_coverage_gate_drops_too_small():
    H, W = 96, 96
    annotation_volume = np.zeros((1, H, W), dtype=np.int32)
    annotation_volume[0, 0, 0] = 1  # 1 voxel only

    def pixel_to_voxel(i: float, j: float) -> np.ndarray:
        return np.array([i, 0.0, j], dtype=np.float64)

    bbox = region_bbox.bbox_from_real_section(
        section_image_shape=(H, W),
        pixel_to_voxel=pixel_to_voxel,
        annotation_volume=annotation_volume,
        region_ids={1},
        midline_x_at_row=lambda i: W / 2.0,
        is_hemisphere=True,
        grid_step=4,
    )
    assert bbox is None


def test_quicknii_to_brainglobe_indices_converts_axis_order():
    # QuickNII pixel_to_voxel returns (x_ml, y_ap, z_dv); BrainGlobe annotation
    # arrays are indexed as (AP, DV, ML).
    assert region_bbox.quicknii_to_brainglobe_indices(
        np.array([7.2, 3.0, 5.9], dtype=np.float64)
    ) == (3, 6, 7)


def test_scale_bbox_handles_split_and_single_boxes():
    split = {"left": [10, 20, 30, 40], "right": [60, 20, 80, 40]}
    assert region_bbox.scale_bbox(split, scale_x=0.5, scale_y=0.25) == {
        "left": [5, 5, 15, 10],
        "right": [30, 5, 40, 10],
    }
    assert region_bbox.scale_bbox([10, 20, 30, 40], scale_x=0.5, scale_y=0.25) == [
        5, 5, 15, 10,
    ]

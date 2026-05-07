"""Tests for region_bbox: hemisphere-split bbox computation."""
from __future__ import annotations

import numpy as np
import region_bbox


def test_atlas_slice_whole_brain_returns_left_and_right():
    # 200-px-wide annotation: each 30x30 region is well below the dense-core
    # trigger so the basic-bbox path is exercised.
    annotation = np.zeros((200, 200), dtype=np.int32)
    annotation[20:50, 10:40] = 1     # left hemisphere region
    annotation[20:50, 160:190] = 1   # right hemisphere region
    hemispheres = np.zeros((200, 200), dtype=np.int32)
    hemispheres[:, :100] = 1
    hemispheres[:, 100:] = 2
    bbox = region_bbox.bbox_from_atlas_slice(
        annotation_slice=annotation,
        hemisphere_slice=hemispheres,
        region_ids={1},
        is_hemisphere=False,
    )
    assert bbox == {
        "left": [10, 20, 39, 49],
        "right": [160, 20, 189, 49],
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
    annotation = np.zeros((200, 200), dtype=np.int32)
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
    H, W = 200, 200
    # Tiny BrainGlobe volume is the same as a single annotation slice. The fake
    # QuickNII transform returns (x_ml, y_ap, z_dv), so x=col, AP=0, z=row.
    annotation_volume = np.zeros((1, H, W), dtype=np.int32)
    annotation_volume[0, 20:50, 10:40] = 1      # left hemisphere
    annotation_volume[0, 20:50, 160:190] = 1    # right hemisphere

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
    # With grid_step=4 and ~30x30 regions in 200x200, expect coords near
    # (10,20)-(39,49) and (160,20)-(189,49) within one grid step of padding.
    assert bbox["left"][0] <= 12
    assert bbox["left"][1] <= 22
    assert bbox["left"][2] >= 36
    assert bbox["left"][3] >= 47
    assert bbox["right"][0] <= 162
    assert bbox["right"][2] >= 186


def test_real_section_hemisphere_returns_single_bbox():
    H, W = 200, 200
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


def test_dense_core_clamps_thin_sprawling_region():
    """A thin C-shape spans almost the whole image; dense-core clamps to its body."""
    H, W = 100, 100
    mask = np.zeros((H, W), dtype=bool)
    # Thick body in the middle (40-60 vertical, 30-70 horizontal): 20*40 = 800px
    mask[40:60, 30:70] = True
    # Thin tail along the bottom edge spanning the whole width: 5*100 = 500px
    mask[80:85, 0:100] = True
    # Without dense-core, bbox would be x:0-99, y:40-84 = 100*45 = 4500 px
    # With dense-core, the body is the largest connected component once the
    # thin tail erodes away → much tighter bbox.
    full = region_bbox._bbox_of_mask(mask)
    assert full == [0, 40, 99, 84]
    full_area = (full[2] - full[0] + 1) * (full[3] - full[1] + 1)
    assert full_area / (H * W) > region_bbox._BBOX_DENSE_CORE_TRIGGER

    dense = region_bbox._bbox_with_dense_core(mask, H * W)
    assert dense is not None
    dense_area = (dense[2] - dense[0] + 1) * (dense[3] - dense[1] + 1)
    assert dense_area < full_area
    # The dense-core bbox should hug the body block, not span the full width.
    assert dense[0] >= 25 and dense[2] <= 75


def test_dense_core_passthrough_when_bbox_already_compact():
    """Compact regions are returned untouched - no erosion needed."""
    H, W = 100, 100
    mask = np.zeros((H, W), dtype=bool)
    mask[40:60, 40:60] = True   # 20x20 = 400 px = 4% of image, well under trigger
    bbox = region_bbox._bbox_with_dense_core(mask, H * W)
    assert bbox == [40, 40, 59, 59]


def test_dense_core_returns_none_for_empty_mask():
    H, W = 50, 50
    mask = np.zeros((H, W), dtype=bool)
    assert region_bbox._bbox_with_dense_core(mask, H * W) is None


def test_scale_bbox_handles_split_and_single_boxes():
    split = {"left": [10, 20, 30, 40], "right": [60, 20, 80, 40]}
    assert region_bbox.scale_bbox(split, scale_x=0.5, scale_y=0.25) == {
        "left": [5, 5, 15, 10],
        "right": [30, 5, 40, 10],
    }
    assert region_bbox.scale_bbox([10, 20, 30, 40], scale_x=0.5, scale_y=0.25) == [
        5, 5, 15, 10,
    ]

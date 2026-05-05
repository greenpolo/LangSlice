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

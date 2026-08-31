"""Tests for the pure-numpy core of the ABBA adapter (no JVM required)."""

import numpy as np
import pytest

from langslice.integrations.abba import (
    classify_to_ids,
    landmarks_from_field,
    normalize_to_rgb,
    render_atlas_at_coords,
)


class FakeAtlas:
    """Tiny synthetic BrainGlobe-shaped atlas: 10x10x10 voxels at 100um."""

    resolution = (100, 100, 100)  # um

    def __init__(self):
        self.annotation = np.zeros((10, 10, 10), dtype=np.int32)
        self.annotation[:, 2:8, 2:8] = 7
        self.annotation[:, 4:6, 4:6] = 9
        self.reference = np.zeros((10, 10, 10), dtype=np.uint16)
        self.reference[:, 2:8, 2:8] = 1000
        self.structures = {}


def test_render_atlas_at_coords_samples_and_masks_oob():
    atlas = FakeAtlas()
    # 2x2 grid: one pixel inside region 7, one inside 9, one in background,
    # one out of volume.
    coords = np.array(
        [
            [[0.5, 0.25, 0.25], [0.5, 0.45, 0.45]],
            [[0.5, 0.05, 0.05], [0.5, -1.0, 0.25]],
        ]
    )
    colored, reference, ids = render_atlas_at_coords(atlas, coords)
    assert ids[0, 0] == 7
    assert ids[0, 1] == 9
    assert ids[1, 0] == 0
    assert ids[1, 1] == 0  # out of volume
    assert reference[0, 0] == 255  # normalized peak
    assert colored.shape == (2, 2, 3)
    assert tuple(colored[1, 1]) == (0, 0, 0)


def test_classify_to_ids_nearest_color_and_background():
    lut = {7: (200, 40, 40), 9: (40, 200, 40)}
    ids_present = np.array([[0, 7], [9, 7]])
    rgb = np.array(
        [
            [[190, 50, 50], [45, 190, 45]],
            [[5, 5, 5], [210, 30, 30]],
        ],
        dtype=np.uint8,
    )
    out = classify_to_ids(rgb, ids_present, lut)
    assert out[0, 0] == 7
    assert out[0, 1] == 9
    assert out[1, 0] == 0  # near-black -> background
    assert out[1, 1] == 7


def test_landmarks_from_field_direction_and_mask():
    h = w = 50
    field = np.zeros((h, w, 2))
    field[..., 0] = 3.0  # dx
    field[..., 1] = -2.0  # dy
    mask = np.zeros((h, w), dtype=bool)
    mask[10:40, 10:40] = True
    src, tgt = landmarks_from_field(field, mask, grid=8)
    assert len(src) >= 4
    np.testing.assert_allclose(tgt - src, np.tile([3.0, -2.0], (len(src), 1)))
    # all landmarks inside the mask
    assert all(mask[int(y), int(x)] for x, y in src)


def test_landmarks_from_field_raises_when_empty():
    field = np.zeros((20, 20, 2))
    with pytest.raises(RuntimeError):
        landmarks_from_field(field, np.zeros((20, 20), dtype=bool), grid=5)


def test_normalize_to_rgb_range():
    img = np.random.default_rng(0).normal(100.0, 20.0, size=(30, 30))
    out = np.asarray(normalize_to_rgb(img))
    assert out.shape == (30, 30, 3)
    assert out.dtype == np.uint8
    assert out.max() == 255 and out.min() == 0

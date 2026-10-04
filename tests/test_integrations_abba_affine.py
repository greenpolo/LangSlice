"""Host transforms preserve landmarks under pivots, shear and quarter-turns."""

from typing import Any

import numpy as np
import pytest

from langslice.core.abba_affine import normalized_to_abba_affine
from langslice.core.affine import normalized_affine, physical_affine_matrix


@pytest.mark.parametrize("rotation_deg", [0, 90, 180, 270])
def test_offcentre_pivot_maps_world_landmarks_like_the_snapshot(rotation_deg):
    original_size = (800, 480)
    size = (original_size[1], original_size[0]) if rotation_deg % 180 else original_size
    pixel_size_um = 25.0
    pixel_matrix = physical_affine_matrix(
        rotation_deg=31, scale_x=1.12, scale_y=0.83,
        translate_x_mm=0.4, translate_y_mm=-0.23,
        size=size, um_per_px=pixel_size_um,
        pivot=(size[0] * 0.21, size[1] * 0.74),
    )
    params = normalized_affine(pixel_matrix, size)
    actual = normalized_to_abba_affine(
        params, size=original_size, pixel_size_um=pixel_size_um,
        rotation_deg=rotation_deg,
    )
    points_px = np.array([[0, 0], [125, 219], [size[0] / 2, size[1] / 2]])
    points_mm = (points_px - np.array(size) / 2) * pixel_size_um / 1000
    expected_px = points_px @ pixel_matrix[:, :2].T + pixel_matrix[:, 2]
    expected_mm = (expected_px - np.array(size) / 2) * pixel_size_um / 1000
    mapped = points_mm @ actual[:2, :2].T + actual[:2, 3]
    np.testing.assert_allclose(mapped, expected_mm, atol=1e-12)
    np.testing.assert_array_equal(actual[2], [0, 0, 1, 0])
    np.testing.assert_array_equal(actual[:2, 2], [0, 0])


def test_shear_survives_and_result_is_independent_of_snapshot_resolution():
    matrix = np.array([[1.1, 0.28, -31.0], [-0.12, 0.93, 24.0]])
    params = normalized_affine(matrix, (800, 480))
    full = normalized_to_abba_affine(params, size=(800, 480), pixel_size_um=25)
    half = normalized_to_abba_affine(params, size=(400, 240), pixel_size_um=50)
    np.testing.assert_allclose(full, half, atol=1e-12)
    np.testing.assert_allclose(full[:2, :2], matrix[:, :2], atol=1e-12)


def test_identity_and_translation_have_no_centering_offset():
    result = normalized_to_abba_affine(
        [1, 0, 0.025, 0, 1, -0.05], size=(800, 480), pixel_size_um=25,
    )
    np.testing.assert_allclose(result, [[1, 0, 0, 0.5], [0, 1, 0, -0.6], [0, 0, 1, 0]])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"params": [1, 0, 0]},
        {"params": [1, 0, float("nan"), 0, 1, 0]},
        {"size": (0, 100)},
        {"pixel_size_um": 0},
        {"pixel_size_um": float("inf")},
        {"rotation_deg": 45},
    ],
)
def test_invalid_geometry_is_rejected(kwargs):
    arguments: dict[str, Any] = dict(
        params=[1, 0, 0, 0, 1, 0], size=(800, 480), pixel_size_um=25,
    )
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        normalized_to_abba_affine(**arguments)

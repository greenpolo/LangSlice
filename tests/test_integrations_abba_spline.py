"""Geometry boundary for native BigWarp landmarks, independent of Java."""
import numpy as np
import pytest

from langslice.integrations.abba_spline import spline_world_landmarks


def test_spline_world_landmarks_preserve_axis_direction_and_orientation():
    source = np.array([[0, 0], [1, 0], [0, 1], [.5, .5]])
    target = source + [.1, -.2]
    src, tgt = spline_world_landmarks(
        {"source": source, "target": target, "extent_mm": [15, 20]},
        size=(800, 600), pixel_size_um=25., rotation_deg=90,
    )
    np.testing.assert_allclose(src, [[-7.5, -10], [7.5, -10], [-7.5, 10], [0, 0]])
    np.testing.assert_allclose(tgt-src, np.tile([1.5, -4], (4, 1)))


@pytest.mark.parametrize("points", [
    [[0, 0], [0, 0], [1, 1]], [[0, 0], [.5, .5], [1, 1]],
    [[0, 0], [1, 0], [float("nan"), 1]],
])
def test_spline_rejects_invalid_geometry(points):
    with pytest.raises(ValueError):
        spline_world_landmarks(
            {"source": points, "target": points, "extent_mm": [20, 15]},
            size=(800, 600), pixel_size_um=25.,
        )

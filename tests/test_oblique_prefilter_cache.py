"""Cubic oblique sampling filters the atlas volume once, with unchanged values."""

import numpy as np
from scipy.ndimage import map_coordinates

from langslice.core import oblique
from tests.deformable_synthetic import SyntheticAtlas


def test_cubic_samples_match_uncached_and_filter_once():
    atlas = SyntheticAtlas()
    oblique._filtered_cache.clear()
    first = oblique.sample_oblique_plane(atlas, 0.1, "coronal", 3.0, -2.0, volume="template", order=3)
    second = oblique.sample_oblique_plane(atlas, 0.15, "coronal", 3.0, -2.0, volume="template", order=3)
    assert len(oblique._filtered_cache) == 1  # one filter for both planes
    vol, res = oblique._scaled_volume(atlas, "template", 1)
    axes = oblique.plane_axes(atlas, "coronal")
    coords = oblique._plane_coordinates(vol.shape, res, axes, 0.1 / res[axes[0]], 3.0, -2.0)
    reference = map_coordinates(vol, coords, order=3, mode="constant", cval=0.0)
    expected = oblique.orient_slice_for_display(reference, "coronal")
    assert np.allclose(first, expected, atol=1e-3 * max(1.0, float(np.abs(vol).max())))
    assert second.shape == first.shape

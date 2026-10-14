"""Oblique sampling: the cubic prefilter runs once per volume, on threads, a
large plane is sampled in bands and an annotation plane is sampled once, all
with unchanged values."""

from collections import OrderedDict
from typing import Any, cast

import numpy as np
import pytest
from scipy.ndimage import map_coordinates, spline_filter

from langslice.core import oblique
from tests.deformable_synthetic import SyntheticAtlas


@pytest.fixture(autouse=True)
def _fresh_caches(monkeypatch):
    monkeypatch.setattr(oblique, "_filtered_cache", {})
    monkeypatch.setattr(oblique, "_annotation_plane_cache", OrderedDict())


def test_cubic_samples_match_uncached_and_filter_once():
    atlas = SyntheticAtlas()
    first = oblique.sample_oblique_plane(
        atlas, 0.1, "coronal", 3.0, -2.0, volume="template", order=3)
    second = oblique.sample_oblique_plane(
        atlas, 0.15, "coronal", 3.0, -2.0, volume="template", order=3)
    assert len(oblique._filtered_cache) == 1  # one filter for both planes
    vol, res = oblique._scaled_volume(atlas, "template", 1)
    axes = oblique.plane_axes(atlas, "coronal")
    coords = oblique._plane_coordinates(vol.shape, res, axes, 0.1 / res[axes[0]], 3.0, -2.0)
    reference = map_coordinates(vol, coords, order=3, mode="constant", cval=0.0)
    expected = oblique.orient_slice_for_display(reference, "coronal")
    assert np.allclose(first, expected, atol=1e-3 * max(1.0, float(np.abs(vol).max())))
    assert second.shape == first.shape


def test_threaded_filter_matches_scipy_exactly():
    rng = np.random.default_rng(0)
    vol = rng.random((37, 23, 29), dtype=np.float32)
    expected = spline_filter(vol, order=3, mode="constant", output=cast(Any, np.float32))
    assert np.array_equal(oblique._spline_filter_threaded(vol, 3), expected)


def test_banded_sampling_matches_one_call(monkeypatch):
    rng = np.random.default_rng(1)
    vol = rng.random((30, 20, 25), dtype=np.float32)
    coords = rng.random((3, 41, 37)) * np.array([29, 19, 24])[:, None, None]
    monkeypatch.setattr(oblique, "SAMPLE_SPLIT_PIXELS", 1)
    for order, prefilter in ((0, True), (1, True), (3, False)):
        whole = map_coordinates(vol, coords, order=order, mode="constant", cval=0.0,
                                prefilter=prefilter)
        banded = oblique._sample(vol, coords, order=order, prefilter=prefilter)
        assert np.array_equal(banded, whole)


def test_annotation_planes_sampled_once_and_handed_out_as_copies():
    atlas = SyntheticAtlas()
    first = oblique.sample_oblique_annotation(atlas, 0.1, "coronal", 3.0, -2.0)
    first[:] = 0  # a caller's own array: the cache is untouched
    second = oblique.sample_oblique_annotation(atlas, 0.1, "coronal", 3.0, -2.0)
    assert second.any() and second.dtype == np.asarray(atlas.annotation).dtype

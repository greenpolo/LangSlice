"""Viewer-only layout policies (paging, camera geometry, atlas channel choice) need no JVM."""

from unittest.mock import Mock

import pytest

from langslice.hosts.integrations.abba_compare import (
    ComparisonWindow,
    atlas_channel_indices,
    camera_parameters,
    target_pages,
)


def test_disparate_targets_keep_execution_order_and_page_without_duplicates():
    assert target_pages([1, 20, 40, 20, 7, 11]) == [[1, 20, 40, 7], [11]]
    assert target_pages([]) == []


def test_camera_centres_native_sampling_plane_at_calibrated_scale():
    values = camera_parameters(800, 600, 10, 8, 6.2)
    assert values is not None
    scale, tx, ty, tz = values
    assert scale == pytest.approx(600 / 8 / 1.08)
    assert (tx, ty) == (400, 300)
    assert scale * 6.2 + tz == pytest.approx(0)


@pytest.mark.parametrize(
    "geometry", [(0, 800, 10, 10, 2), (800, 800, 0, 10, 2), (800, 800, 10, 10, float("nan"))]
)
def test_invalid_camera_geometry_does_not_move_view(geometry):
    assert camera_parameters(*geometry) is None


def test_atlas_channel_prefers_real_boundaries_and_never_coordinate_channels():
    assert atlas_channel_indices(["X", "Y", "Z", "reference", "borders"]) == [4]
    assert atlas_channel_indices(["X", "Y", "Z", "coordinate ML", "reference"]) == [4]
    assert atlas_channel_indices(["X", "Y", "Z"]) == []


def bare_window():
    window = ComparisonWindow.__new__(ComparisonWindow)
    window.closed = False
    window.pages = [[1, 20, 40, 5], [9]]
    window.page = 1
    window.timer = Mock()
    window.frame = Mock()
    window.panels = [Mock()]
    window.overview = Mock()
    return window


def test_page_cycle_includes_final_partial_page():
    window = bare_window()
    window._rebuild = Mock()
    window._page(1)
    assert window.page == 0
    window._page(-1)
    assert window.page == 1



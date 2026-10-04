"""Native positioning geometry, channel choices, and read-only visual state."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from langslice.hosts.integrations.abba_overview import (
    native_atlas_sources,
    native_centers,
    native_handle_style,
    native_position_x,
)


def test_position_matches_native_live_half_mm_interval():
    assert native_position_x(4.985, 0.01, 13.11, 50) == pytest.approx(137.1306)
    assert native_position_x(5.785, 0.01, 13.11, 50) == pytest.approx(158.1066)


def test_position_snaps_like_java_towards_zero_not_nearest_voxel():
    assert native_position_x(0.019, 0.01, 10, 50) == pytest.approx(5.2)
    assert native_position_x(-0.019, 0.01, 10, 50) == pytest.approx(4.8)


@pytest.mark.parametrize("args", [(1, 0, 10, 50), (1, 0.01, 10, 0), (float("nan"), 0.01, 10, 50)])
def test_invalid_calibration_does_not_invent_native_positions(args):
    with pytest.raises(ValueError):
        native_position_x(*args)


def test_positioning_view_preserves_actual_native_stairs_and_x_shifts():
    view = Mock()
    view.getDisplayMode.return_value = 0
    slices = [Mock(), Mock()]
    points = [Mock(), Mock()]
    points[0].getDoublePosition.side_effect = [137.1306, 12.4065]
    points[1].getDoublePosition.side_effect = [158.1066, 17.002]
    view.getDisplayedCenter.side_effect = points
    assert native_centers(view, slices, 0.01, 13.11, 9.19, 50) == [
        (137.1306, 12.4065),
        (158.1066, 17.002),
    ]
    view.setDisplayMode.assert_not_called()
    for sl in slices:
        sl.select.assert_not_called()
        sl.setSlicingAxisPosition.assert_not_called()


def test_review_view_uses_native_positioning_formula_without_switching_main_mode():
    view, sl = Mock(), Mock()
    view.getDisplayMode.return_value = 1
    sl.getSlicingAxisPosition.return_value = 4.985
    centers = native_centers(view, [sl], 0.01, 13.11, 9.19, 50)
    assert centers[0] == pytest.approx((137.1306, 9.19))
    view.getDisplayedCenter.assert_not_called()
    view.setDisplayMode.assert_not_called()


@pytest.mark.parametrize("mode", [0, 1])
def test_atlas_uses_host_visible_channels_and_borrows_existing_mosaic(mode):
    view = Mock()
    view.getDisplayMode.return_value = mode
    atlas = SimpleNamespace(
        extendedSlicedSources=["e0", "e1", "e2", "label"],
        nonExtendedSlicedSources=["n0", "n1", "n2", "label"],
    )
    active = {"e0", "e2"} if mode == 0 else {"n0", "n2"}
    state = view.getBdvh.return_value.getViewerPanel.return_value.state.return_value
    state.isSourceActive.side_effect = lambda source: source in active
    assert native_atlas_sources(view, atlas, 3) == ["e0", "e2"]
    view.setDisplayMode.assert_not_called()


def test_selection_colors_and_radii_match_abba_slice_gui_state():
    assert native_handle_style(True, False) == (12, (0, 255, 0, 180))
    assert native_handle_style(True, True) == (16, (0, 255, 0, 255))
    assert native_handle_style(False, False) == (12, (255, 255, 0, 64))
    assert native_handle_style(False, True) == (16, (255, 255, 0, 128))


def test_native_key_square_uses_key_status_for_size_and_magenta_for_selected_key():
    from langslice.hosts.integrations.abba_overview import native_key_style

    assert native_key_style(True, False) == (6, (0, 255, 0, 180))
    assert native_key_style(False, False) == (6, (255, 255, 0, 64))
    assert native_key_style(True, True) == (16, (255, 0, 255, 200))
    assert native_key_style(False, True) == (16, (255, 255, 0, 128))

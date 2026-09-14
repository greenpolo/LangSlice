"""Viewer-only layout and lifecycle policies require no running JVM."""

from typing import cast
from unittest.mock import Mock

import pytest

from langslice.integrations.abba_compare import (
    ComparisonWindow,
    _Panel,
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


def test_refresh_uses_selected_page_targets_only():
    window = bare_window()
    window.refresh()
    cast(Mock, window.panels[0]).update.assert_called_once_with(9)


def test_hiding_stops_refresh_without_disposing_native_panels():
    window = bare_window()
    window.hide()
    window.timer.stop.assert_called_once()
    window.frame.setVisible.assert_called_once_with(False)
    cast(Mock, window.panels[0]).dispose.assert_not_called()


def test_closing_disposes_only_owned_handles_once():
    window = bare_window()
    panel = cast(Mock, window.panels[0])
    window.dispose()
    window.dispose()
    panel.dispose.assert_called_once()
    window.frame.dispose.assert_called_once()
    assert window.pages == []


def test_hidden_window_timer_stops_without_refresh():
    window = bare_window()
    window.frame.isVisible.return_value = False
    window._tick()
    window.timer.stop.assert_called_once()
    cast(Mock, window.panels[0]).update.assert_not_called()


def test_show_reopens_last_targets_without_changing_page():
    window = bare_window()
    window.show_targets = Mock()
    window.show()
    window.show_targets.assert_called_once_with([1, 20, 40, 5, 9])
    assert window.page == 1
    window.frame.toFront.assert_called_once()


def test_page_cycle_includes_final_partial_page():
    window = bare_window()
    window._rebuild = Mock()
    window._page(1)
    assert window.page == 0
    window._page(-1)
    assert window.page == 1


def test_copy_owns_outer_transform_and_converter_before_setting_brightness():
    panel = _Panel.__new__(_Panel)
    source, display = Mock(), Mock()
    wrapper, helper, settings = Mock(), Mock(), Mock()
    classes = {
        "bdv.tools.transformation.TransformedSource": wrapper,
        "sc.fiji.bdvpg.sourceandconverter.SourceAndConverterHelper": helper,
        "spimdata.util.Displaysettings": settings,
    }
    panel.owner = Mock(jimport=classes.__getitem__)
    copied = panel._copy_source(source, display)
    wrapper.assert_called_once_with(source.getSpimSource())
    helper.createSourceAndConverter.assert_called_once_with(wrapper.return_value)
    settings.applyDisplaysettings.assert_called_once_with(copied, display)
    assert copied is helper.createSourceAndConverter.return_value
    source.getConverter.assert_not_called()


def test_disposed_comparison_cannot_be_reopened():
    window = bare_window()
    window.dispose()
    window.frame.reset_mock()
    window.show()
    window.show_targets([1, 2])
    window.frame.setVisible.assert_not_called()
    window.frame.toFront.assert_not_called()


def test_unchanged_poll_does_not_restart_progressive_renderer():
    panel = _Panel.__new__(_Panel)
    panel.last_content = None
    viewer = Mock()
    content = ((1.0, 0.0), 0.0, 0.0)
    panel._repaint_changed_content(viewer, content, sources_changed=True)
    for _ in range(8):
        panel._repaint_changed_content(viewer, content, sources_changed=False)
    viewer.requestRepaint.assert_called_once()
    # An in-place affine edit must still trigger rendering with identical sources.
    panel._repaint_changed_content(viewer, ((1.0, 0.2), 0.0, 0.0), sources_changed=False)
    assert viewer.requestRepaint.call_count == 2
    # So must a nonlinear registration replacing a source without changing its affine.
    panel._repaint_changed_content(viewer, ((1.0, 0.2), 0.0, 0.0), sources_changed=True)
    assert viewer.requestRepaint.call_count == 3


def test_late_bdv_camera_override_is_corrected_without_repainting_settled_view():
    from langslice.integrations.abba_compare import ensure_camera

    viewer, desired, current = Mock(), Mock(), Mock()
    desired.get.side_effect = lambda row, column: 10.0 if row == column else 0.0
    current.get.side_effect = lambda row, column: 1.0 if row == column else 0.0
    state = viewer.state.return_value
    state.getViewerTransform.return_value = current
    state.setViewerTransform.side_effect = lambda value: setattr(
        state.getViewerTransform,
        "return_value",
        value,
    )
    ensure_camera(viewer, desired)
    for _ in range(8):
        ensure_camera(viewer, desired)
    state.setViewerTransform.assert_called_once_with(desired)
    # Native late initialization moves the camera after our first fitting pass.
    state.getViewerTransform.return_value = current
    ensure_camera(viewer, desired)
    assert state.setViewerTransform.call_count == 2
    viewer.requestRepaint.assert_not_called()

"""Native positioning geometry matching ABBA's Java formula, without a JVM."""

from unittest.mock import Mock

import pytest

from langslice.hosts.integrations.abba_overview import (
    native_centers,
    native_position_x,
)


def test_position_matches_the_native_formula_and_snaps_towards_zero():
    assert native_position_x(4.985, 0.01, 13.11, 50) == pytest.approx(137.1306)
    assert native_position_x(5.785, 0.01, 13.11, 50) == pytest.approx(158.1066)
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



"""Position arithmetic: gap interpolation."""

import pytest

from langslice.linear.whole_brain.signals import interpolate_positions


def test_interpolate_between_two_knowns():
    out = interpolate_positions([1.0, None, None, 1.6], interval_mm=0.2)
    assert out[1] == pytest.approx(1.2, abs=1e-6)
    assert out[2] == pytest.approx(1.4, abs=1e-6)


def test_interpolate_extrapolates_outward_by_interval():
    out = interpolate_positions([None, None, 2.0, None, None, 2.6], interval_mm=0.2)
    assert out[1] == pytest.approx(1.8, abs=1e-6)
    assert out[0] == pytest.approx(1.6, abs=1e-6)
    assert out[5] == pytest.approx(2.6, abs=1e-6)


def test_interpolate_keeps_known_values():
    out = interpolate_positions([1.0, None, 1.5], interval_mm=0.2)
    assert out[0] == 1.0
    assert out[2] == 1.5


def test_interpolate_needs_one_known():
    with pytest.raises(ValueError, match="at least one known"):
        interpolate_positions([None, None], interval_mm=0.2)


def test_interpolate_empty_is_empty():
    assert interpolate_positions([], interval_mm=0.2) == []

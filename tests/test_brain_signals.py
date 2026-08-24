"""Advisory spacing signals: gap interpolation and the monotone fit."""

import pytest

from langslice.linear.whole_brain.signals import interpolate_positions, monotone_fit


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


def test_monotone_fit_leaves_monotone_input_alone():
    fitted = monotone_fit(
        [2.0, 3.0, 4.0, 5.0, 6.0], interval_mm=1.0, thickness_mm=0.05
    )
    assert fitted == sorted(fitted)
    for original, value in zip([2.0, 3.0, 4.0, 5.0, 6.0], fitted, strict=True):
        assert abs(original - value) < 0.5


def test_monotone_fit_tames_an_outlier():
    fitted = monotone_fit(
        [2.0, 3.0, 8.0, 5.0, 6.0], interval_mm=1.0, thickness_mm=0.05
    )
    assert fitted == sorted(fitted)
    assert fitted[2] < 7.0


def test_monotone_fit_enforces_minimum_spacing():
    fitted = monotone_fit(
        [1.0, 1.01, 1.02, 1.03, 2.0], interval_mm=0.2, thickness_mm=0.05
    )
    for a, b in zip(fitted, fitted[1:], strict=False):
        assert b - a >= 0.05 - 1e-6


def test_monotone_fit_corrects_non_monotone_input():
    fitted = monotone_fit(
        [5.0, 3.0, 4.0, 2.0, 6.0], interval_mm=1.0, thickness_mm=0.05
    )
    assert fitted == sorted(fitted)


def test_monotone_fit_weights_pull_toward_trusted_points():
    positions = [2.0, 3.0, 8.0, 5.0, 6.0]
    trusted = monotone_fit(
        positions,
        interval_mm=1.0,
        thickness_mm=0.05,
        weights=[1.0, 1.0, 20.0, 1.0, 1.0],
    )
    plain = monotone_fit(positions, interval_mm=1.0, thickness_mm=0.05)
    assert trusted[2] > plain[2]


def test_monotone_fit_short_inputs_pass_through():
    assert monotone_fit([], interval_mm=1.0, thickness_mm=0.05) == []
    assert monotone_fit([3.0], interval_mm=1.0, thickness_mm=0.05) == [3.0]


def test_monotone_fit_rejects_mismatched_weights():
    with pytest.raises(ValueError, match="weights length"):
        monotone_fit(
            [1.0, 2.0], interval_mm=1.0, thickness_mm=0.05, weights=[1.0]
        )

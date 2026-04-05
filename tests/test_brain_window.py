import pytest

from langslice.brain.window import compute_refinement_window


def test_window_locked_both_sides():
    """Window bounded by two locked neighbors."""
    win = compute_refinement_window(
        position_mm=2.5,
        left_locked_mm=2.3,
        right_locked_mm=2.7,
        thickness_mm=0.050,
        interval_mm=0.200,
    )
    assert win.lo == pytest.approx(2.35, abs=0.001)  # 2.3 + 0.05
    assert win.hi == pytest.approx(2.65, abs=0.001)  # 2.7 - 0.05
    assert win.center == pytest.approx(2.5)
    assert win.n_images >= 5


def test_window_locked_left_only():
    """Only left neighbor locked; right uses interpolated + interval."""
    win = compute_refinement_window(
        position_mm=2.5,
        left_locked_mm=2.3,
        right_locked_mm=None,
        thickness_mm=0.050,
        interval_mm=0.200,
    )
    assert win.lo == pytest.approx(2.35, abs=0.001)
    assert win.hi == pytest.approx(2.7, abs=0.001)  # center + interval
    assert win.n_images >= 5


def test_window_locked_right_only():
    """Only right neighbor locked."""
    win = compute_refinement_window(
        position_mm=2.5,
        left_locked_mm=None,
        right_locked_mm=2.7,
        thickness_mm=0.050,
        interval_mm=0.200,
    )
    assert win.lo == pytest.approx(2.3, abs=0.001)  # center - interval
    assert win.hi == pytest.approx(2.65, abs=0.001)


def test_window_skip_when_too_narrow():
    """Window smaller than thickness -> skip (n_images=0)."""
    win = compute_refinement_window(
        position_mm=2.5,
        left_locked_mm=2.48,
        right_locked_mm=2.52,
        thickness_mm=0.050,
        interval_mm=0.200,
    )
    assert win.skip is True
    assert win.n_images == 0


def test_window_image_count_scales():
    """Wide window gets more images than narrow window."""
    wide = compute_refinement_window(
        position_mm=5.0,
        left_locked_mm=4.7,
        right_locked_mm=5.3,
        thickness_mm=0.050,
        interval_mm=0.200,
    )
    narrow = compute_refinement_window(
        position_mm=5.0,
        left_locked_mm=4.9,
        right_locked_mm=5.1,
        thickness_mm=0.050,
        interval_mm=0.200,
    )
    assert wide.n_images > narrow.n_images


def test_window_max_images_capped():
    """Even a very wide window caps at 13 images."""
    win = compute_refinement_window(
        position_mm=5.0,
        left_locked_mm=3.0,
        right_locked_mm=7.0,
        thickness_mm=0.050,
        interval_mm=0.200,
    )
    assert win.n_images <= 13

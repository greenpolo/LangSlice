import pytest

from langslice.brain.window import compute_refinement_window, compute_search_bounds


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


def test_compute_search_bounds_uses_default_window_for_interior_slice():
    bounds = compute_search_bounds(
        center_mm=5.0,
        atlas_range=(0.0, 13.0),
        window_half_mm=3.0,
    )

    assert bounds == pytest.approx((2.0, 8.0), abs=0.001)


def test_compute_search_bounds_tightens_leading_edge_window():
    bounds = compute_search_bounds(
        center_mm=2.691,
        atlas_range=(0.0, 13.0),
        window_half_mm=3.0,
        edge_anchor_mm=3.691,
        edge_side="leading",
    )

    assert bounds == pytest.approx((1.691, 4.691), abs=0.001)


def test_compute_search_bounds_tightens_trailing_edge_window():
    bounds = compute_search_bounds(
        center_mm=11.255,
        atlas_range=(0.0, 13.0),
        window_half_mm=3.0,
        edge_anchor_mm=10.255,
        edge_side="trailing",
    )

    assert bounds == pytest.approx((9.255, 12.255), abs=0.001)

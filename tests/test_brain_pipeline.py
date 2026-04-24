"""Tests for isotonic fitting in the brain pipeline."""

from langslice_harness.whole_brain.pipeline import _fit_isotonic
from langslice_harness.whole_brain.types import SlicePosition


def _make_slices(positions: list[float]) -> list[SlicePosition]:
    return [
        SlicePosition(f"slice_{i:03d}.tif", i, p, "test", locked=False)
        for i, p in enumerate(positions)
    ]


def test_fit_isotonic_already_monotonic():
    """Monotonic input should stay close to original values."""
    slices = _make_slices([2.0, 3.0, 4.0, 5.0, 6.0])
    result = _fit_isotonic(slices, interval_mm=1.0, thickness_mm=0.05)
    positions = [s.position_mm for s in result]
    # Should be monotonically increasing
    for i in range(len(positions) - 1):
        assert positions[i] < positions[i + 1]
    # Should stay close to input
    for orig, fitted in zip([2.0, 3.0, 4.0, 5.0, 6.0], positions, strict=True):
        assert abs(orig - fitted) < 0.5


def test_fit_isotonic_corrects_outlier():
    """A single outlier should be pulled toward the trend."""
    slices = _make_slices([2.0, 3.0, 8.0, 5.0, 6.0])  # 8.0 is an outlier
    result = _fit_isotonic(slices, interval_mm=1.0, thickness_mm=0.05)
    positions = [s.position_mm for s in result]
    # Must be monotonic
    for i in range(len(positions) - 1):
        assert positions[i] < positions[i + 1]
    # The outlier at index 2 should be pulled down from 8.0
    assert positions[2] < 7.0


def test_fit_isotonic_enforces_minimum_spacing():
    """Adjacent slices must be at least thickness_mm apart."""
    slices = _make_slices([1.0, 1.01, 1.02, 1.03, 2.0])
    result = _fit_isotonic(slices, interval_mm=0.2, thickness_mm=0.05)
    positions = [s.position_mm for s in result]
    for i in range(len(positions) - 1):
        spacing = positions[i + 1] - positions[i]
        assert spacing >= 0.05 - 1e-9, f"spacing {spacing} < thickness 0.05"


def test_fit_isotonic_non_monotonic_input():
    """Non-monotonic input should be corrected to monotonic."""
    slices = _make_slices([5.0, 3.0, 4.0, 2.0, 6.0])
    result = _fit_isotonic(slices, interval_mm=1.0, thickness_mm=0.05)
    positions = [s.position_mm for s in result]
    for i in range(len(positions) - 1):
        assert positions[i] < positions[i + 1]

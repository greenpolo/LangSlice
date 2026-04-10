import pytest

from langslice.whole_brain.interpolation import interpolate_positions
from langslice.whole_brain.types import SlicePosition


def _locked(filename: str, index: int, position_mm: float) -> SlicePosition:
    return SlicePosition(filename, index, position_mm, "anchor", locked=True)


def _unlocked(filename: str, index: int) -> SlicePosition:
    return SlicePosition(filename, index, 0.0, "", locked=False)


def test_interpolate_between_two_anchors():
    """Even spacing between two anchors."""
    slices = [
        _locked("s0.tif", 0, 1.0),
        _unlocked("s1.tif", 1),
        _unlocked("s2.tif", 2),
        _locked("s3.tif", 3, 1.6),
    ]
    result = interpolate_positions(slices, interval_mm=0.200, atlas_range=(0.0, 13.0), z_axis="AP")
    assert result[1].position_mm == pytest.approx(1.2, abs=0.001)
    assert result[2].position_mm == pytest.approx(1.4, abs=0.001)
    assert result[1].source == "interpolated"
    assert result[1].locked is False


def test_extrapolate_before_first_anchor():
    """Slices before the first anchor use the average interval."""
    slices = [
        _unlocked("s0.tif", 0),
        _unlocked("s1.tif", 1),
        _locked("s2.tif", 2, 2.0),
        _locked("s5.tif", 5, 2.6),
    ]
    result = interpolate_positions(slices, interval_mm=0.200, atlas_range=(0.0, 13.0), z_axis="AP")
    assert result[1].position_mm == pytest.approx(1.8, abs=0.001)
    assert result[0].position_mm == pytest.approx(1.6, abs=0.001)
    assert result[0].source == "extrapolated"


def test_extrapolate_clamped_to_atlas_bounds():
    """Extrapolation does not go below 0.0mm."""
    slices = [
        _unlocked("s0.tif", 0),
        _locked("s1.tif", 1, 0.1),
        _locked("s3.tif", 3, 0.5),
    ]
    result = interpolate_positions(slices, interval_mm=0.200, atlas_range=(0.0, 13.0), z_axis="AP")
    assert result[0].position_mm >= 0.0


def test_pa_axis_extrapolates_correctly():
    """PA z-axis: first slice has highest AP, last has lowest."""
    slices = [
        _locked("s0.tif", 0, 8.0),
        _unlocked("s1.tif", 1),
        _locked("s2.tif", 2, 7.6),
    ]
    result = interpolate_positions(slices, interval_mm=0.200, atlas_range=(0.0, 13.0), z_axis="PA")
    assert result[1].position_mm == pytest.approx(7.8, abs=0.001)


def test_anchors_unchanged():
    """Locked anchor positions are never modified."""
    slices = [
        _locked("s0.tif", 0, 1.0),
        _unlocked("s1.tif", 1),
        _locked("s2.tif", 2, 1.5),
    ]
    result = interpolate_positions(slices, interval_mm=0.200, atlas_range=(0.0, 13.0), z_axis="AP")
    assert result[0].position_mm == 1.0
    assert result[2].position_mm == 1.5
    assert result[0].locked is True

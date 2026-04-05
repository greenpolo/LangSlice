from langslice.brain.constraints import enforce_constraints
from langslice.brain.types import SlicePosition


def _sp(index: int, pos: float) -> SlicePosition:
    return SlicePosition(f"s{index}.tif", index, pos, "refined", locked=True)


def test_strict_already_monotonic():
    """No changes when positions are already in order."""
    slices = [_sp(0, 1.0), _sp(1, 1.2), _sp(2, 1.4)]
    result = enforce_constraints(slices, ordering="strict", thickness_mm=0.050, z_axis="AP")
    assert [s.position_mm for s in result] == [1.0, 1.2, 1.4]


def test_strict_clamps_violation():
    """Strict mode clamps a non-monotonic slice to midpoint of neighbors."""
    slices = [_sp(0, 1.0), _sp(1, 1.5), _sp(2, 1.3)]  # index 2 violates
    result = enforce_constraints(slices, ordering="strict", thickness_mm=0.050, z_axis="AP")
    positions = [s.position_mm for s in result]
    # Slice 2 should be clamped to > slice 1
    assert positions[2] > positions[1]


def test_strict_enforces_minimum_spacing():
    """Two slices closer than thickness get nudged apart."""
    slices = [_sp(0, 1.0), _sp(1, 1.02), _sp(2, 1.3)]  # 0.02 < 0.05 thickness
    result = enforce_constraints(slices, ordering="strict", thickness_mm=0.050, z_axis="AP")
    positions = [s.position_mm for s in result]
    assert positions[1] - positions[0] >= 0.050 - 0.001


def test_loose_swaps_reversed_pair():
    """Loose mode swaps two adjacent slices that are reversed."""
    slices = [_sp(0, 1.0), _sp(1, 1.4), _sp(2, 1.2), _sp(3, 1.6)]
    result = enforce_constraints(slices, ordering="loose", thickness_mm=0.050, z_axis="AP")
    positions = [s.position_mm for s in result]
    # After swap, should be monotonic
    assert positions == sorted(positions)


def test_none_no_reordering():
    """None mode does not enforce monotonicity."""
    slices = [_sp(0, 1.0), _sp(1, 1.5), _sp(2, 1.3)]
    result = enforce_constraints(slices, ordering="none", thickness_mm=0.050, z_axis="AP")
    positions = [s.position_mm for s in result]
    assert positions == [1.0, 1.5, 1.3]


def test_none_still_enforces_minimum_spacing():
    """Even in none mode, minimum spacing is enforced."""
    slices = [_sp(0, 1.0), _sp(1, 1.02)]
    result = enforce_constraints(slices, ordering="none", thickness_mm=0.050, z_axis="AP")
    positions = [s.position_mm for s in result]
    assert abs(positions[1] - positions[0]) >= 0.050 - 0.001


def test_pa_axis_strict():
    """PA axis: monotonically decreasing is valid."""
    slices = [_sp(0, 8.0), _sp(1, 7.8), _sp(2, 7.6)]
    result = enforce_constraints(slices, ordering="strict", thickness_mm=0.050, z_axis="PA")
    positions = [s.position_mm for s in result]
    assert positions == [8.0, 7.8, 7.6]

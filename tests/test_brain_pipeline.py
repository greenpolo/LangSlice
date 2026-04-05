"""Tests for wave computation and pipeline orchestration."""

from langslice.brain.pipeline import compute_waves


def test_compute_waves_simple():
    """4 anchors among 12 slices, indices 0-11."""
    anchor_indices = {2, 5, 8, 11}
    n_slices = 12
    waves = compute_waves(n_slices, anchor_indices)
    # Wave 1: distance 1 from any anchor -> {1,3, 4,6, 7,9, 10}
    assert 1 in waves[0]
    assert 3 in waves[0]
    # All non-anchor indices should appear exactly once across all waves
    all_assigned = set()
    for wave in waves:
        for idx in wave:
            assert idx not in all_assigned, f"index {idx} in multiple waves"
            assert idx not in anchor_indices, f"anchor {idx} in wave"
            all_assigned.add(idx)
    expected = set(range(n_slices)) - anchor_indices
    assert all_assigned == expected


def test_compute_waves_all_anchors():
    """When every slice is an anchor, no waves needed."""
    waves = compute_waves(4, {0, 1, 2, 3})
    assert waves == []


def test_compute_waves_single_anchor():
    """Single anchor at midpoint, waves radiate outward."""
    waves = compute_waves(7, {3})
    # Wave 1: {2, 4}
    assert set(waves[0]) == {2, 4}
    # Wave 2: {1, 5}
    assert set(waves[1]) == {1, 5}
    # Wave 3: {0, 6}
    assert set(waves[2]) == {0, 6}


def test_compute_waves_adjacent_anchors():
    """Two adjacent anchors: only outer slices need waves."""
    waves = compute_waves(5, {2, 3})
    all_in_waves = set()
    for w in waves:
        all_in_waves.update(w)
    assert all_in_waves == {0, 1, 4}

from langslice.brain.anchor_selection import select_anchor_indices


def test_single_anchor_picks_midpoint():
    # 20 slices (indices 0..19), 1 anchor -> midpoint index 9
    result = select_anchor_indices(n_slices=20, n_anchors=1)
    assert result == [9]


def test_two_anchors_trisect():
    # 40 slices (0..39), 2 anchors -> indices ~13, 26
    result = select_anchor_indices(n_slices=40, n_anchors=2)
    assert len(result) == 2
    assert result == sorted(result)
    # Both should be in the middle third, not at extremes
    assert result[0] > 5
    assert result[1] < 35


def test_four_anchors_center_weighted():
    # 20 slices, 4 anchors -> should be spread with center priority
    result = select_anchor_indices(n_slices=20, n_anchors=4)
    assert len(result) == 4
    assert result == sorted(result)
    # No duplicates
    assert len(set(result)) == 4
    # All within bounds
    assert all(0 <= i < 20 for i in result)


def test_anchors_equal_slices():
    # 4 slices, 4 anchors -> every slice is an anchor
    result = select_anchor_indices(n_slices=4, n_anchors=4)
    assert result == [0, 1, 2, 3]


def test_one_slice_one_anchor():
    result = select_anchor_indices(n_slices=1, n_anchors=1)
    assert result == [0]


def test_anchors_never_exceed_slices():
    # More anchors than slices -> clamp
    result = select_anchor_indices(n_slices=3, n_anchors=10)
    assert len(result) == 3
    assert result == [0, 1, 2]


def test_center_out_avoids_extremes():
    # With few anchors on a large set, none should be at index 0 or n-1
    result = select_anchor_indices(n_slices=60, n_anchors=3)
    assert 0 not in result
    assert 59 not in result

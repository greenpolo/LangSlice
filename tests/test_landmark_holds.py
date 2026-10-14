"""Automatic holds: the lattice between a section's pairs that keeps the tissue
between them where it was, added by ``landmark_triangulation`` only."""

from __future__ import annotations

import numpy as np

from langslice.core import deformation
from langslice.core import landmarks as geometry
from langslice.core.triangulation import PiecewiseAffine
from langslice.ops import landmarks
from langslice.ops.landmarks import HOLD_CLEARANCE, HOLD_SPACING, automatic_holds
from tests.test_landmarks import saved_pairs, setup

SIZE = (700, 500)


def test_a_lattice_covers_the_frame_with_tens_of_holds():
    holds = automatic_holds(np.empty((0, 2)), SIZE)
    assert 20 <= len(holds) <= 60
    assert holds[:, 0].min() >= 0 and holds[:, 0].max() < SIZE[0]
    assert holds[:, 1].min() >= 0 and holds[:, 1].max() < SIZE[1]
    # regular: equal steps along each axis
    xs = np.unique(holds[:, 0])
    assert np.allclose(np.diff(xs), np.diff(xs)[0])


def test_holds_are_skipped_near_every_pair_and_kept_elsewhere():
    spacing = max(SIZE) * HOLD_SPACING
    everything = automatic_holds(np.empty((0, 2)), SIZE)
    pairs = np.array([everything[3] + [2.0, -3.0], [10.0, 10.0]])
    holds = automatic_holds(pairs, SIZE)
    gap = np.hypot(*(holds[:, None] - pairs[None]).transpose(2, 0, 1))
    assert gap.min() > HOLD_CLEARANCE * spacing
    # only the holds near a pair went
    dropped = len(everything) - len(holds)
    assert 1 <= dropped <= 6
    for hold in everything:
        near = np.hypot(*(pairs - hold).T).min() <= HOLD_CLEARANCE * spacing
        assert near == (not any(np.allclose(hold, kept) for kept in holds))


def test_a_hold_maps_to_itself_and_is_not_a_pair(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        edits = saved_pairs(job, ws, record)
        edits[2]["atlas"]["x"] += 12
        landmarks.set_landmarks(job, ws, record.id, edits)
        reply = landmarks.landmark_triangulation(job, ws, record.id)
        assert reply["automatic_holds"] > 5
        assert "automatic holds were added" in reply["holds_note"]
        assert reply["built_from"] == ["L1", "L2", "L3"] and reply["landmark_count"] == 3
        listed = landmarks.set_landmarks(job, ws, record.id)["landmarks"]
        assert [row["id"] for row in listed] == ["L1", "L2", "L3"]  # holds are no pairs
        current = job.deformations.current(job.state, record)
        assert current is not None
        params = current.engine["native_parameters"]
        n_pairs, n_holds = params["pair_count"], params["hold_count"]
        assert (n_pairs, n_holds) == (3, reply["automatic_holds"])
        mesh = PiecewiseAffine(
            np.array(params["source_xy"]), np.array(params["target_xy"]),
            np.array(params["simplices"]))
        source = np.array(params["source_xy"])[4 + n_pairs:]
        target = np.array(params["target_xy"])[4 + n_pairs:]
        assert np.array_equal(source, target)  # identity correspondences
        assert np.allclose(mesh.map(source), source)
        # the holds are in the record's provenance and the section's reference
        assert record.deformation is not None
        assert len(record.deformation["landmark_holds"]) == n_holds
        assert len(current.provenance["automatic_holds_xy"]) == n_holds
        assert len(record.landmarks["pairs"]) == 3
    finally:
        job.close()


def test_a_far_region_is_unchanged_by_one_moved_pair(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        edits = saved_pairs(job, ws, record)
        edits[2]["atlas"]["x"] += 14  # L3 moves; L1 and L2 stay where they are
        landmarks.set_landmarks(job, ws, record.id, edits)
        landmarks.landmark_triangulation(job, ws, record.id)
        current = job.deformations.current(job.state, record)
        assert current is not None
        field = np.asarray(current.field_mm)
        grid = deformation.fit_grid(job.state, ws, record)
        size = np.asarray(grid.image.size)
        moved = np.hypot(field[..., 0], field[..., 1])
        l3 = geometry.transform(
            np.linalg.inv(geometry.orientation(record)),
            record.landmarks["pairs"][2]["section_fraction_xy"]) * size - 0.5
        yy, xx = np.indices(moved.shape)
        distance = np.hypot(xx - l3[0], yy - l3[1])
        spacing = size.max() * HOLD_SPACING
        far = distance > 2.2 * spacing
        assert far.sum() > 100
        assert moved[far].max() < 1e-6 * max(moved.max(), 1.0) + 1e-9
        assert moved[distance < 0.3 * spacing].max() > 0
    finally:
        job.close()


def test_holds_are_kept_only_on_the_placed_footprint():
    size = (700, 500)
    footprint = np.zeros((size[1], size[0]), bool)
    footprint[100:400, 150:550] = True
    everywhere = automatic_holds(np.empty((0, 2)), size)
    holds = automatic_holds(np.empty((0, 2)), size, footprint=footprint)
    assert 0 < len(holds) < len(everywhere)
    rows, cols = np.rint(holds[:, 1]).astype(int), np.rint(holds[:, 0]).astype(int)
    assert footprint[rows, cols].all()

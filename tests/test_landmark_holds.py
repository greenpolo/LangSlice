"""Holds: the automatic lattice between a section's pairs that keeps the tissue
between them where it was, added by ``landmark_triangulation`` unless its
``holds`` is ``"none"``, and the holds an agent places with ``set_landmarks``."""

from __future__ import annotations

import numpy as np
import pytest

from langslice.core import deformation
from langslice.core import landmarks as geometry
from langslice.core.triangulation import PiecewiseAffine
from langslice.ops import landmarks, look
from langslice.ops.landmarks import HOLD_CLEARANCE, HOLD_SPACING, automatic_holds
from langslice.ops.refusal import Refused
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


# --- the holds choice, and the holds an agent places ------------------------------------


def section_points(job, ws, record, fractions):
    """Section points {picture, x, y} at *fractions* of the section's working frame."""
    shown = look.look(job, ws, "section", sections=[record.id])
    job.views.flush()
    seq = shown.entries[0]["id"]
    panel = job.views.lookup(seq).recipe["endpoint_panels"][0]
    points = []
    for fraction in fractions:
        x, y = geometry.section_picture_point(panel, fraction)
        points.append({"picture": seq, "x": float(x), "y": float(y)})
    return points


def _moved(job, record):
    """The applied deformation's displacement length per fit-grid pixel (mm)."""
    current = job.deformations.current(job.state, record)
    assert current is not None
    field = np.asarray(current.field_mm)
    return np.hypot(field[..., 0], field[..., 1]), current


def _pixel(job, ws, record, fraction):
    size = np.asarray(deformation.fit_grid(job.state, ws, record).image.size)
    return np.rint(np.asarray(fraction) * size - 0.5).astype(int)


def test_holds_none_adds_no_automatic_holds_and_lets_a_pair_move_far_tissue(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        edits = saved_pairs(job, ws, record)
        edits[2]["atlas"]["x"] += 14
        landmarks.set_landmarks(job, ws, record.id, edits)
        reply = landmarks.landmark_triangulation(job, ws, record.id, holds="none")
        assert reply["holds"] == "none" and reply["automatic_holds"] == 0
        assert "placed_holds" not in reply and 'holds "none"' in reply["holds_note"]
        moved, current = _moved(job, record)
        params = current.engine["native_parameters"]
        assert (params["pair_count"], params["placed_hold_count"], params["hold_count"]) == (
            3, 0, 0)
        assert len(params["source_xy"]) == 4 + 3
        assert record.deformation["landmark_holds"] == []
        assert record.deformation["landmark_hold_mode"] == "none"
        assert record.deformation["steps"][-1]["holds"] == "none"
        # with no holds the pair drags tissue far from it, which holds keep still
        col, row = _pixel(job, ws, record, [0.9, 0.9])
        assert moved[row, col] > 0
        # the other choice is another warp, written again; the same choice is not
        automatic = landmarks.landmark_triangulation(job, ws, record.id)
        assert automatic["written"] and automatic["automatic_holds"] > 0
        assert automatic["holds"] == "automatic"
        again = landmarks.landmark_triangulation(job, ws, record.id)
        assert not again["written"] and again["automatic_holds"] == automatic["automatic_holds"]
    finally:
        job.close()


def test_placed_holds_pin_their_spots_with_holds_none(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        edits = saved_pairs(job, ws, record)
        edits[2]["atlas"]["x"] += 14
        fractions = [[0.15, 0.85], [0.85, 0.85], [0.5, 0.95]]
        reply = landmarks.set_landmarks(job, ws, record.id, edits,
                                        holds=section_points(job, ws, record, fractions))
        assert reply["changed"] == ["L1", "L2", "L3"]
        assert reply["added_holds"] == ["H1", "H2", "H3"]
        assert [r["state"] for r in reply["holds"]] == ["pending"] * 3
        assert reply["counts"]["total"] == 3  # holds are not counted as pairs
        assert reply["warp"]["not_applied"] == ["L1", "L2", "L3", "H1", "H2", "H3"]
        done = landmarks.landmark_triangulation(job, ws, record.id, holds="none")
        assert done["placed_holds"] == ["H1", "H2", "H3"] and done["automatic_holds"] == 0
        assert "H1, H2, H3" in done["holds_note"]
        assert done["built_from"] == ["L1", "L2", "L3"]
        moved, current = _moved(job, record)
        params = current.engine["native_parameters"]
        assert (params["pair_count"], params["placed_hold_count"]) == (3, 3)
        source = np.array(params["source_xy"])[4 + 3:]
        target = np.array(params["target_xy"])[4 + 3:]
        assert np.array_equal(source, target)  # each hold corresponds to itself
        mesh = PiecewiseAffine(np.array(params["source_xy"]), np.array(params["target_xy"]),
                               np.array(params["simplices"]))
        assert np.allclose(mesh.map(source), source)
        # the field barely moves the pixels at the holds, while the pair moves its own
        for fraction in fractions:
            col, row = _pixel(job, ws, record, fraction)
            assert moved[row, col] < 0.01 * moved.max()
        assert set(record.deformation["landmark_placed_holds"]) == {"H1", "H2", "H3"}
        assert [h["id"] for h in current.provenance["placed_holds"]] == ["H1", "H2", "H3"]
        listing = landmarks.set_landmarks(job, ws, record.id)
        assert [r["state"] for r in listing["holds"]] == ["applied"] * 3
        assert listing["warp"]["holds_in_warp"] == ["H1", "H2", "H3"]
        assert "not_applied" not in listing["warp"]
        # a deleted hold still shapes the warp until the set is applied again
        gone = landmarks.set_landmarks(job, ws, record.id, delete=["H2"])
        assert gone["deleted"] == ["H2"] and gone["warp"]["deleted_still_in_warp"] == ["H2"]
        assert [r["id"] for r in gone["holds"]] == ["H1", "H3"]
        assert job.undo()
        assert [h["id"] for h in job.state.by_id(record.id).landmarks["holds"]] == [
            "H1", "H2", "H3"]
        cleared = landmarks.set_landmarks(job, ws, record.id, delete=["all"])
        assert cleared["deleted"] == ["L1", "L2", "L3", "H1", "H2", "H3"]
        assert "holds" not in cleared
        again = landmarks.set_landmarks(job, ws, record.id,
                                        holds=section_points(job, ws, record, fractions[:1]))
        assert again["added_holds"] == ["H4"]  # ids are never reused
    finally:
        job.close()


def test_placed_holds_join_the_automatic_holds(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        edits = saved_pairs(job, ws, record)
        edits[2]["atlas"]["x"] += 14
        landmarks.set_landmarks(job, ws, record.id, edits,
                                holds=section_points(job, ws, record, [[0.5, 0.55]]))
        done = landmarks.landmark_triangulation(job, ws, record.id)
        assert done["holds"] == "automatic" and done["automatic_holds"] > 0
        assert done["placed_holds"] == ["H1"] and "Your holds H1" in done["holds_note"]
        _, current = _moved(job, record)
        params = current.engine["native_parameters"]
        assert params["placed_hold_count"] == 1
        assert len(params["source_xy"]) == 4 + 3 + 1 + params["hold_count"]
        source = np.array(params["source_xy"])[4 + 3:]
        assert np.array_equal(source, np.array(params["target_xy"])[4 + 3:])
        assert len(record.deformation["landmark_holds"]) == done["automatic_holds"]
    finally:
        job.close()


def test_bad_hold_points_are_refused_and_nothing_is_written(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        edits = saved_pairs(job, ws, record)
        good = section_points(job, ws, record, [[0.5, 0.5]])[0]
        atlas_only = look.look(job, ws, "atlas", sections=[record.id])
        job.views.flush()
        before = job.snapshot()
        cases = [
            ({**good, "y": 100000}, "BAD_LANDMARK_POINT"),
            ({**good, "picture": 999}, "UNKNOWN_PICTURE"),
            ({"picture": good["picture"], "x": 1.0}, "BAD_ARGS"),
            ({**good, "fraction": 0.5}, "BAD_ARGS"),
            ({**good, "picture": atlas_only.entries[0]["id"]}, "BAD_LANDMARK_PICTURE"),
        ]
        for point, code in cases:
            with pytest.raises(Refused) as caught:
                landmarks.set_landmarks(job, ws, record.id, edits, holds=[point])
            assert caught.value.code == code
            assert "holds[0]" in caught.value.details["message"]
            assert job.snapshot() == before
        with pytest.raises(Refused) as caught:
            landmarks.set_landmarks(job, ws, record.id, holds="H1")  # type: ignore[arg-type]
        assert caught.value.code == "BAD_ARGS"
        # a hold on a pair's own section point cannot be triangulated
        landmarks.set_landmarks(job, ws, record.id, edits, holds=[edits[0]["section"]])
        for mode in ("automatic", "none"):
            with pytest.raises(Refused) as caught:
                landmarks.landmark_triangulation(job, ws, record.id, holds=mode)
            assert caught.value.code == "LANDMARK_TRIANGULATION_FAILED"
            assert caught.value.details["pairs"] == ["L1"]
            assert caught.value.details["holds"] == ["H1"]
            assert "Pairs and holds with the same section point" in caught.value.details[
                "message"]
        with pytest.raises(Refused) as caught:
            landmarks.landmark_triangulation(job, ws, record.id, holds="some")
        assert caught.value.code == "BAD_ARGS"
    finally:
        job.close()


def test_the_tools_take_holds(tmp_path):
    from langslice.doors.declarations import arguments_schema
    from langslice.doors.tools.toolbox import build_tools

    job, ws, record = setup(tmp_path)
    try:
        box = build_tools(job.state, ws, job.spec, job=job)
        tools = {tool.__name__: tool for tool in box.tools}
        schema = arguments_schema("landmark_triangulation")
        assert schema["properties"]["holds"]["enum"] == ["automatic", "none"]
        hold = section_points(job, ws, record, [[0.5, 0.9]])[0]
        bad = tools["set_landmarks"](section=record.id, holds=[{**hold, "role": "section"}])
        assert bad["error"] == "UNKNOWN_ARGUMENTS"
        done = tools["set_landmarks"](section=record.id, pairs=saved_pairs(job, ws, record),
                                      holds=[hold], view=True)
        assert done["added_holds"] == ["H1"] and len(done["pictures"]) == 3  # pairs only
        applied = tools["landmark_triangulation"](section=record.id, holds="none", view=True)
        assert applied["status"] == "ok" and applied["placed_holds"] == ["H1"]
        assert applied["pictures"]
        refused = tools["landmark_triangulation"](section=record.id, holds="some", view=False)
        assert refused["status"] == "error"
    finally:
        job.close()

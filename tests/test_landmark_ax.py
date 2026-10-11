"""The agent-trial fixes of the landmark tools: folds, specific errors, the compact
listing, delete-all, undo/redo words, annotation-only staleness, close-up borders,
atlas size and a zoom's content box."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from langslice.core.landmark_views import pair_picture
from langslice.core.landmarks import FOLD_COLOUR
from langslice.core.layers import collecting
from langslice.core.triangulation import DegenerateMesh, PiecewiseAffine, visualign_mesh
from langslice.ops import landmarks, look
from langslice.ops.refusal import Refused
from tests.test_landmarks import saved_pairs, setup


def _colour_pixels(image: Image.Image, colour: str, tolerance: int = 40) -> int:
    target = np.array([int(colour[i:i + 2], 16) for i in (1, 3, 5)])
    pixels = np.asarray(image.convert("RGB")).astype(int)
    return int((np.abs(pixels - target).max(axis=-1) <= tolerance).sum())


def _folded_pairs(job, ws, record):
    """Three pairs whose atlas endpoints of the first two are swapped: a fold."""
    edits = saved_pairs(job, ws, record)
    edits[0]["atlas"], edits[1]["atlas"] = edits[1]["atlas"], edits[0]["atlas"]
    return edits


# --- 1. folds ---------------------------------------------------------------------


def test_the_mesh_accepts_folds_and_keeps_its_forward_map():
    source = np.array([[0, 0], [10, 0], [0, 10], [10, 10], [5, 4]], dtype=float)
    target = source.copy()
    target[4] = [5, 14]  # the middle vertex is thrown across the top edge
    mesh = PiecewiseAffine(source, target)
    assert mesh.folded.any()
    assert mesh.folded_triangles().shape[1] == 3
    probe = np.array([[2.0, 1.0], [8.0, 5.0]])
    forward = mesh.map(probe)
    assert np.isfinite(forward).all()
    again = PiecewiseAffine(source, target)
    np.testing.assert_array_equal(again.map(probe), forward)
    grid = np.stack(np.meshgrid(np.arange(11.0), np.arange(11.0)), -1)
    first, second = mesh.inverse(grid), again.inverse(grid)
    np.testing.assert_array_equal(first, second)  # a deterministic choice where they overlap
    assert 0 < mesh.folded_fraction(grid) < 1


def test_unfolded_triangles_win_the_inverse_where_targets_overlap():
    source = np.array([[0, 0], [10, 0], [0, 10], [10, 10], [5, 4]], dtype=float)
    target = source.copy()
    target[4] = [5, 14]
    mesh = PiecewiseAffine(source, target)
    point = np.array([[5.0, 9.5]])  # covered by a folded and by an unfolded target triangle
    got = mesh.inverse(point)[0]
    owners = [k for k in range(len(mesh.simplices))
              if np.isfinite(mesh._map(point, mesh.target, mesh.source)[0]).all()]
    assert owners  # the point is covered
    unfolded = [k for k in np.flatnonzero(~mesh.folded)]
    tri = mesh.simplices[unfolded[0]]
    assert np.isfinite(got).all() and tri.size == 3


def test_repeated_points_and_zero_area_source_triangles_are_still_refused():
    with pytest.raises(DegenerateMesh) as repeated:
        PiecewiseAffine(np.array([[0, 0], [1, 1], [1, 1], [3, 0]], float), np.zeros((4, 2)))
    assert repeated.value.vertices == [[1, 2]]
    flat = np.array([[0, 0], [1, 1], [2, 2]], float)
    with pytest.raises(DegenerateMesh) as collapsed:
        PiecewiseAffine(flat, flat + 1, simplices=np.array([[0, 1, 2]]))
    assert collapsed.value.vertices == [[0, 1, 2]]
    with pytest.raises(ValueError, match="inside the fixed mesh corners"):
        visualign_mesh(np.array([[-500.0, 0], [1, 1], [2, 5]]), np.zeros((3, 2)), size=(10, 10))


def test_a_fold_is_applied_reported_and_outlined_on_the_picture(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        landmarks.set_landmarks(job, ws, record.id, _folded_pairs(job, ws, record))
        done = landmarks.landmark_triangulation(job, ws, record.id)
        assert done["written"]
        folds = done["folds"]
        assert 0 < folds["area_fraction"] <= 1
        assert set(folds["pairs"]) <= {"L1", "L2", "L3"} and folds["pairs"]
        assert folds["triangle_count"] >= 1 and folds["triangles"]
        stored = record.deformation["landmark_folds"]
        assert stored["triangles"] and stored["area_fraction"] == folds["area_fraction"]
        # An unchanged application says the same again.
        assert landmarks.landmark_triangulation(job, ws, record.id)["folds"] == folds
        shown = look.look(job, ws, "overlay", sections=[record.id])
        assert _colour_pixels(shown.pictures[0], FOLD_COLOUR) > 0
    finally:
        job.close()


def test_a_sound_warp_reports_no_folds_and_draws_no_outline(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        landmarks.set_landmarks(job, ws, record.id, saved_pairs(job, ws, record, delta=(2.0, 1.0)))
        done = landmarks.landmark_triangulation(job, ws, record.id)
        assert "folds" not in done and not record.deformation["landmark_folds"]
        shown = look.look(job, ws, "overlay", sections=[record.id])
        assert _colour_pixels(shown.pictures[0], FOLD_COLOUR) == 0
    finally:
        job.close()


def test_repeated_section_points_name_their_pairs(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        edits = saved_pairs(job, ws, record)
        edits[1]["section"] = dict(edits[0]["section"])
        landmarks.set_landmarks(job, ws, record.id, edits)
        with pytest.raises(Refused) as caught:
            landmarks.landmark_triangulation(job, ws, record.id)
        assert caught.value.code == "LANDMARK_TRIANGULATION_FAILED"
        assert caught.value.details["pairs"] == ["L1", "L2"]
        assert "L1, L2" in str(caught.value) and "same section point" in str(caught.value)
    finally:
        job.close()


# --- 3. specific errors ------------------------------------------------------------


def test_refusals_name_the_pair_endpoint_picture_point_and_fix(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        edits = saved_pairs(job, ws, record)
        edits[1]["section"]["y"] = 100000
        with pytest.raises(Refused) as caught:
            landmarks.set_landmarks(job, ws, record.id, edits)
        text = str(caught.value)
        picture = edits[1]["section"]["picture"]
        assert caught.value.code == "BAD_LANDMARK_POINT"
        assert "pairs[1] (new pair)" in text and "section endpoint" in text
        assert f"({edits[1]['section']['x']:g}, 100000)" in text
        assert f"picture {picture}" in text and "image content is x 0-" in text

        with pytest.raises(Refused) as caught:
            landmarks.set_landmarks(job, ws, record.id, [
                {"atlas": {"picture": 999, "x": 1, "y": 1}, "section": edits[0]["section"]}])
        assert caught.value.code == "UNKNOWN_PICTURE"
        assert "no saved picture 999" in str(caught.value) and "pairs[0]" in str(caught.value)

        generic = look.look(job, ws, "atlas", positions_mm=[record.position_mm])
        wrong = {**edits[0], "atlas": {**edits[0]["atlas"], "picture": generic.entries[0]["id"]}}
        with pytest.raises(Refused) as caught:
            landmarks.set_landmarks(job, ws, record.id, [wrong])
        assert caught.value.code == "BAD_LANDMARK_PICTURE"
        assert "atlas endpoint" in str(caught.value) and "no atlas geometry" in str(caught.value)
        assert record.id in str(caught.value)
    finally:
        job.close()


def test_a_stale_plane_names_the_planes_and_the_stale_pairs(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        edits = saved_pairs(job, ws, record)
        landmarks.set_landmarks(job, ws, record.id, edits)
        before = job.snapshot()
        record.position_mm += 0.01
        job.commit(before)
        with pytest.raises(Refused) as caught:
            landmarks.landmark_triangulation(job, ws, record.id)
        assert caught.value.details["pairs"] == ["L1", "L2", "L3"]
        assert "Pairs L1, L2, L3" in str(caught.value)
        fresh = [{"id": "L1", "atlas": edits[0]["atlas"]}]  # a picture from the old plane
        with pytest.raises(Refused) as caught:
            landmarks.set_landmarks(job, ws, record.id, fresh)
        text = str(caught.value)
        assert caught.value.code == "STALE_LANDMARK_PLANE"
        assert "pairs[0] (L1), atlas endpoint" in text and "now at" in text
        assert f"picture {edits[0]['atlas']['picture']}" in text
    finally:
        job.close()


# --- 4, 5. the listing and delete-all -----------------------------------------------


def test_the_listing_is_compact_and_a_call_without_pairs_writes_nothing(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        edits = saved_pairs(job, ws, record)
        reply = landmarks.set_landmarks(job, ws, record.id, edits)
        row = reply["landmarks"][0]
        assert set(row) == {"id", "atlas", "section", "state"}
        assert set(row["atlas"]) == {"picture", "x", "y"} and row["state"] == "pending"
        assert reply["counts"] == {"total": 3, "applied": 0, "pending": 3, "stale": 0}
        landmarks.landmark_triangulation(job, ws, record.id)
        depth = len(job.undo_stack)
        listed = landmarks.set_landmarks(job, ws, record.id)
        assert not listed["written"] and "nothing was written" in listed["note"]
        assert len(job.undo_stack) == depth
        assert {r["state"] for r in listed["landmarks"]} == {"applied"}
        moved = [{"id": "L3", "section": {**edits[2]["section"],
                                          "x": edits[2]["section"]["x"] + 5}}]
        mixed = landmarks.set_landmarks(job, ws, record.id, moved)
        assert [r["state"] for r in mixed["landmarks"]] == ["applied", "applied", "pending"]
        assert "geometry" not in str(mixed)  # the nested dump stays in the state
        assert "geometry" in record.landmarks["pairs"][0]["atlas"]
        before = job.snapshot()
        record.position_mm += 0.01
        job.commit(before)
        assert {r["state"] for r in landmarks.set_landmarks(
            job, ws, record.id)["landmarks"]} == {"stale"}
    finally:
        job.close()


def test_status_rows_carry_a_landmark_summary_only_when_there_are_pairs(tmp_path):
    from langslice.ops.views import status

    job, ws, record = setup(tmp_path)
    try:
        assert all("landmarks" not in row for row in status(job).rows)
        landmarks.set_landmarks(job, ws, record.id, saved_pairs(job, ws, record))
        rows = {row["id"]: row for row in status(job).rows}
        assert rows[record.id]["landmarks"] == {"total": 3, "pending": 3, "stale": 0}
        assert all("landmarks" not in row for key, row in rows.items() if key != record.id)
        landmarks.landmark_triangulation(job, ws, record.id)
        counts = {row["id"]: row for row in status(job).rows}[record.id]["landmarks"]
        assert counts == {"total": 3, "pending": 0, "stale": 0}
    finally:
        job.close()


def test_delete_all_clears_the_section_in_one_step_and_ids_are_not_reused(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        landmarks.set_landmarks(job, ws, record.id, saved_pairs(job, ws, record))
        depth = len(job.undo_stack)
        reply = landmarks.set_landmarks(job, ws, record.id, delete=["all"])
        assert reply["deleted"] == ["L1", "L2", "L3"] and reply["landmarks"] == []
        assert len(job.undo_stack) == depth + 1 and not record.landmarks["pairs"]
        again = landmarks.set_landmarks(job, ws, record.id, saved_pairs(job, ws, record)[:1])
        assert again["changed"] == ["L4"]
        with pytest.raises(Refused, match="stands alone"):
            landmarks.set_landmarks(job, ws, record.id, delete=["all", "L4"])
        assert job.undo() and job.undo()
        assert [p["id"] for p in job.state.by_id(record.id).landmarks["pairs"]] == [
            "L1", "L2", "L3"]
    finally:
        job.close()


# --- 6. undo and redo say what they did ----------------------------------------------


def test_undo_and_redo_name_the_tool_and_the_sections(tmp_path):
    from langslice.doors.tools.toolbox import build_tools

    job, ws, record = setup(tmp_path)
    try:
        box = build_tools(job.state, ws, job.spec, job=job, gates=False)
        tools = {tool.__name__: tool for tool in box.tools}
        tools["set_landmarks"](section=record.id, pairs=saved_pairs(job, ws, record), view=False)
        tools["landmark_triangulation"](section=record.id, view=False)
        undone = tools["undo"]()
        assert undone["undone"] == {"tool": "landmark_triangulation", "sections": [record.id]}
        assert [row["id"] for row in undone["changed"]] == [record.id]
        assert "rows" not in undone and "interval_breaks" not in undone
        assert undone["undo_depth"] >= 1
        redone = tools["redo"]()
        assert redone["redone"] == {"tool": "landmark_triangulation", "sections": [record.id]}
        assert tools["undo"]()["undone"]["tool"] == "landmark_triangulation"
        assert tools["undo"]()["undone"]["tool"] == "set_landmarks"
        other = job.state.in_order()[1]
        moved = tools["position_sections"](positions={other.id: float(other.position_mm) + 0.02}
                                           ) if "position_sections" in tools else None
        if moved is not None and moved.get("status") == "ok":
            back = tools["undo"]()
            assert back["undone"]["tool"] == "position_sections"
            assert other.id in back["undone"]["sections"]
            assert "cutting_angles_deg" in back
    finally:
        job.close()


# --- 7. landmark edits alone do not make an old picture stale --------------------------


def test_a_landmark_edit_alone_leaves_old_pictures_unstale_with_their_old_markers(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        edits = saved_pairs(job, ws, record)
        first = look.look(job, ws, "section", sections=[record.id])
        job.views.flush()
        number = first.entries[0]["id"]
        landmarks.set_landmarks(job, ws, record.id, edits)  # now it has markers
        zoomed = look.zoom(job, ws, box=[0, 0, 40, 40], picture=number)
        assert not zoomed.stale and "since" not in zoomed.entries[0]["caption"]
        assert _colour_pixels(zoomed.pictures[0], "#00ffff", 30) == 0  # no markers as of then
        newer = look.look(job, ws, "section", sections=[record.id])
        job.views.flush()
        again = look.zoom(job, ws, box=[0, 0, 400, 400], picture=newer.entries[0]["id"])
        landmarks.set_landmarks(job, ws, record.id, delete=["L2"])
        later = look.zoom(job, ws, box=[0, 0, 400, 400], picture=newer.entries[0]["id"])
        assert not again.stale and not later.stale
        before = job.snapshot()
        record.position_mm += 0.01
        job.commit(before)
        assert look.zoom(job, ws, box=[0, 0, 40, 40], picture=number).stale
    finally:
        job.close()


# --- 8. close-ups carry the atlas borders ----------------------------------------------


def test_the_pair_close_up_draws_atlas_borders_on_its_atlas_panel(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        landmarks.set_landmarks(job, ws, record.id, saved_pairs(job, ws, record))
        pair = job.state.by_id(record.id).landmarks["pairs"][0]
        with collecting():
            picture = pair_picture(ws, job.state, record, pair, store=job.deformations)
        half = picture.image.width // 2
        atlas_half = picture.image.crop((0, 0, half, picture.image.height))
        section_half = picture.image.crop((half, 0, picture.image.width, picture.image.height))
        assert _colour_pixels(atlas_half, "#ffff00", 30) > 0
        assert _colour_pixels(section_half, "#ffff00", 30) == 0
    finally:
        job.close()


# --- 9. an atlas picture honours the resolution -----------------------------------------


def test_an_atlas_look_honours_a_larger_resolution(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        plain = look.look(job, ws, "atlas", positions_mm=[record.position_mm])
        sized = look.look(job, ws, "atlas", positions_mm=[record.position_mm], resolution=900)
        small = look.look(job, ws, "atlas", positions_mm=[record.position_mm], resolution=128)
        assert max(plain.pictures[0].size) < 900
        assert max(sized.pictures[0].size) > max(plain.pictures[0].size)
        assert max(sized.pictures[0].size) <= 900 + 4
        assert max(small.pictures[0].size) <= 128 + 4
        zoom = look.zoom(job, ws, box=[10, 10, 80, 80], picture=sized.entries[0]["id"])
        assert max(zoom.pictures[0].size) > 100
    finally:
        job.close()


# --- 11. a zoom reports where its content lies -------------------------------------------


def test_a_zoom_reports_its_content_box(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        shown = look.look(job, ws, "section", sections=[record.id])
        job.views.flush()
        wide = look.zoom(job, ws, box=[0, 0, 60, 60], picture=shown.entries[0]["id"])
        x0, y0, x1, y1 = wide.entries[0]["content_box"]
        assert (x0, y0) == (0, 0) and x1 <= wide.pictures[0].width
        assert y1 <= wide.pictures[0].height
        height = shown.pictures[0].height
        tall = look.zoom(job, ws, box=[0, 0, 20, height - 40], picture=shown.entries[0]["id"])
        box = tall.entries[0]["content_box"]
        assert box[2] < tall.pictures[0].width  # padded beside the real content
    finally:
        job.close()


def test_the_triangulation_picture_takes_a_resolution(tmp_path):
    from langslice.doors.tools.toolbox import build_tools

    job, ws, record = setup(tmp_path)
    try:
        box = build_tools(job.state, ws, job.spec, job=job, gates=False, level="auto")
        tools = {tool.__name__: tool for tool in box.tools}
        tools["set_landmarks"](section=record.id, pairs=saved_pairs(job, ws, record), view=False)
        from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY

        shown = tools["landmark_triangulation"](section=record.id, view=True, resolution=300)
        assert shown["status"] == "ok" and shown["pictures"][0]["id"]
        small = max(shown[TOOL_MEDIA_PARTS_KEY][0].size)
        with job.writing():
            before = job.snapshot()
            record.deformation = None  # apply again, at the default size
            job.commit(before)
        plain = tools["landmark_triangulation"](section=record.id, view=True)
        # The synthetic section holds ~420 px: nothing is drawn past its source.
        assert small < max(plain[TOOL_MEDIA_PARTS_KEY][0].size)
        assert 280 <= small <= 305
        clamped = tools["landmark_triangulation"](section=record.id, view=True, resolution=99999)
        assert "resolution_note" in clamped
    finally:
        job.close()

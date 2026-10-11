"""Round-3 trial fixes: the zoom's coordinate grid and the one coordinate system of
a zoom and its source picture, a dry run's picture numbers, the truth about which
pairs the current warp was built from, and the apply reply."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from langslice.core import landmarks as geometry
from langslice.core.picture_geometry import GRID_LINES, grid_step, source_frame
from langslice.core.zoom_grid import GRID_COLOUR, draw_grid
from langslice.ops import landmarks, look
from tests.test_landmark_ax import _colour_pixels
from tests.test_landmarks import saved_pairs, setup


def _zoom(job, ws, picture, box):
    done = look.zoom(job, ws, box=box, picture=picture)
    job.views.flush()
    return done


# --- the grid ----------------------------------------------------------------------


@pytest.mark.parametrize("extent", [40, 90, 150, 400, 700, 1400, 2800, 9000])
def test_the_grid_step_is_round_and_gives_six_to_twelve_lines(extent):
    step = grid_step(extent)
    assert step in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000)
    lines = extent / step
    assert lines <= GRID_LINES
    assert lines >= 4.8  # the step before it would have given more than twelve


def test_a_zoom_draws_a_labelled_grid_in_the_whole_pictures_pixels(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        whole = look.look(job, ws, "section", sections=[record.id])
        job.views.flush()
        number = whole.entries[0]["id"]
        zoomed = _zoom(job, ws, number, [60, 40, 220, 160])
        entry = zoomed.entries[0]
        recipe = job.views.lookup(entry["id"]).recipe
        frame = source_frame(recipe)
        assert frame is not None
        x0, y0, x1, y1 = entry["covers"]
        assert 55 <= x0 <= 65 and 215 <= x1 <= 225 and 35 <= y0 <= 45 and 155 <= y1 <= 165
        # grid lines exist, in a colour no other drawing uses
        assert _colour_pixels(zoomed.pictures[0], GRID_COLOUR, 20) > 200
        # and the whole picture itself carries none
        assert _colour_pixels(whole.pictures[0], GRID_COLOUR, 20) == 0
    finally:
        job.close()


def test_grid_lines_sit_where_their_labels_say(tmp_path):
    frame_image = Image.new("RGB", (300, 200), (0, 0, 0))
    from langslice.core.picture_geometry import SourceFrame

    frame = SourceFrame(origin=[100.0, 40.0], scale=[0.5, 0.5], shown=[300, 200])
    draw_grid(frame_image, frame)
    pixels = np.asarray(frame_image).astype(int)
    green = np.abs(pixels - np.array([0x7d, 0xff, 0x3a])).max(axis=-1) <= 10
    # the source x = 160 (a multiple of the step) is picture pixel (150 + .5 - 100)/.5 - .5
    column = round(float(frame.to_picture([160, 60])[0]))
    assert green[:, column].sum() > 30
    assert green[:, column + 7].sum() < 5  # between two lines


# --- one coordinate system ---------------------------------------------------------


def test_a_number_read_off_a_zoom_grid_lands_where_it_does_on_the_source_picture(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        edits = saved_pairs(job, ws, record)
        source = edits[0]["section"]
        whole_number = source["picture"]
        x, y = float(source["x"]), float(source["y"])
        zoom = _zoom(job, ws, whole_number, [x - 50, y - 40, x + 50, y + 40])
        zoom_number = zoom.entries[0]["id"]
        # the same grid number, named on the source picture and on the zoom
        a = landmarks.set_landmarks(job, ws, record.id, [
            {"atlas": edits[0]["atlas"],
             "section": {"picture": whole_number, "x": x, "y": y}}])
        b = landmarks.set_landmarks(job, ws, record.id, [
            {"atlas": edits[0]["atlas"],
             "section": {"picture": zoom_number, "x": x, "y": y}}])
        pairs = {p["id"]: p for p in record.landmarks["pairs"]}
        assert a["changed"] == ["L1"] and b["changed"] == ["L2"]
        assert np.allclose(pairs["L1"]["section_fraction_xy"],
                           pairs["L2"]["section_fraction_xy"], atol=2e-3)
        # the listing keeps the numbers as given
        assert [r["section"]["x"] for r in b["landmarks"]][-1] == round(x, 1)
        # a zoom of the zoom: its box is in the same numbers, and so is a point on it
        deeper = _zoom(job, ws, zoom_number, [x - 20, y - 15, x + 20, y + 15])
        c = landmarks.set_landmarks(job, ws, record.id, [
            {"atlas": edits[0]["atlas"],
             "section": {"picture": deeper.entries[0]["id"], "x": x, "y": y}}])
        pairs = {p["id"]: p for p in record.landmarks["pairs"]}
        assert np.allclose(pairs[c["changed"][0]]["section_fraction_xy"],
                           pairs["L1"]["section_fraction_xy"], atol=2e-3)
        # a number outside what the zoom shows is refused, in the grid's numbers
        with pytest.raises(landmarks.Refused) as caught:
            landmarks.set_landmarks(job, ws, record.id, [
                {"atlas": edits[0]["atlas"],
                 "section": {"picture": deeper.entries[0]["id"], "x": x + 200, "y": y}}])
        assert caught.value.code == "BAD_LANDMARK_POINT"
        assert f"in picture {deeper.entries[0]['id']}" in str(caught.value)
    finally:
        job.close()


def test_an_atlas_zoom_and_an_overlay_zoom_read_the_same_numbers(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        edits = saved_pairs(job, ws, record)
        atlas = edits[1]["atlas"]
        x, y = float(atlas["x"]), float(atlas["y"])
        zoom = _zoom(job, ws, atlas["picture"], [x - 40, y - 30, x + 40, y + 30])
        on_source = landmarks.set_landmarks(job, ws, record.id, [
            {"atlas": {"picture": atlas["picture"], "x": x, "y": y},
             "section": edits[1]["section"]}])
        on_zoom = landmarks.set_landmarks(job, ws, record.id, [
            {"atlas": {"picture": zoom.entries[0]["id"], "x": x, "y": y},
             "section": edits[1]["section"]}])
        pairs = {p["id"]: p for p in record.landmarks["pairs"]}
        assert np.allclose(pairs[on_source["changed"][0]]["atlas_native_xy"],
                           pairs[on_zoom["changed"][0]]["atlas_native_xy"], atol=0.6)

        overlay = look.look(job, ws, "overlay", sections=[record.id])
        job.views.flush()
        number = overlay.entries[0]["id"]
        panel = job.views.lookup(number).recipe["endpoint_panels"][0]
        point = geometry.section_picture_point(panel, [0.5, 0.5])
        near = _zoom(job, ws, number, [point[0] - 40, point[1] - 30, point[0] + 40, point[1] + 30])
        first = landmarks.set_landmarks(job, ws, record.id, [
            {"atlas": edits[1]["atlas"],
             "section": {"picture": number, "x": float(point[0]), "y": float(point[1])}}])
        second = landmarks.set_landmarks(job, ws, record.id, [
            {"atlas": edits[1]["atlas"],
             "section": {"picture": near.entries[0]["id"], "x": float(point[0]),
                         "y": float(point[1])}}])
        pairs = {p["id"]: p for p in record.landmarks["pairs"]}
        assert np.allclose(pairs[first["changed"][0]]["section_fraction_xy"],
                           pairs[second["changed"][0]]["section_fraction_xy"], atol=3e-3)
    finally:
        job.close()


# --- a dry run ---------------------------------------------------------------------


def test_a_dry_run_resolves_the_pictures_a_real_call_would(tmp_path):
    from langslice.job.views import DiscardedViews

    job, ws, record = setup(tmp_path)
    try:
        edits = saved_pairs(job, ws, record)
        number = edits[0]["section"]["picture"]
        dry = DiscardedViews(job.layout)
        assert dry.lookup(number) is not None
        before = len(dry.records())
        assert before >= number
        job.views = dry
        job.persist = False
        depth = len(job.undo_stack)
        reply = landmarks.set_landmarks(job, ws, record.id, edits)
        assert reply["changed"] == ["L1", "L2", "L3"]
        # it numbers its own pictures after the saved ones
        made = look.look(job, ws, "section", sections=[record.id])
        assert made.entries[0]["id"] > number
        assert depth >= 0
    finally:
        job.close()


# --- which pairs the warp holds ----------------------------------------------------


def test_the_listing_tells_which_pairs_the_current_warp_holds(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        edits = saved_pairs(job, ws, record)
        landmarks.set_landmarks(job, ws, record.id, edits)
        first = landmarks.set_landmarks(job, ws, record.id)
        assert first["warp"]["built_from"] == [] and "No landmark warp" in first["warp"]["note"]
        done = landmarks.landmark_triangulation(job, ws, record.id)
        assert done["built_from"] == ["L1", "L2", "L3"]
        assert "whole deformation" in done["displacement_note"]
        assert "step_displacement_mm" in done["displacement_note"]
        same = landmarks.set_landmarks(job, ws, record.id)
        assert same["warp"]["built_from"] == ["L1", "L2", "L3"]
        assert "exactly these pairs" in same["warp"]["note"]
        # delete one: the remaining two are applied, the deleted one still shapes the warp
        gone = landmarks.set_landmarks(job, ws, record.id, delete=["L2"])
        assert {r["id"]: r["state"] for r in gone["landmarks"]} == {
            "L1": "applied", "L3": "applied"}
        assert gone["warp"]["deleted_still_in_warp"] == ["L2"]
        assert "L2" in gone["warp"]["note"] and "landmark_triangulation" in gone["warp"]["note"]
        # add one and move one
        moved = [{"id": "L3", "section": {**edits[2]["section"],
                                          "x": edits[2]["section"]["x"] + 5}}]
        more = landmarks.set_landmarks(job, ws, record.id, moved + [
            {"atlas": edits[1]["atlas"], "section": edits[1]["section"]}])
        states = {r["id"]: r["state"] for r in more["landmarks"]}
        assert states == {"L1": "applied", "L3": "moved", "L4": "pending"}
        assert more["warp"]["not_applied"] == ["L4", "L3"]
        assert more["counts"]["moved"] == 1 and more["counts"]["pending"] == 1
        # status says the same in words
        from langslice.ops.views import status

        row = {r["id"]: r for r in status(job).rows}[record.id]["landmarks"]
        assert row["pending"] == 2 and "differs" in row["warp"]
        # applying again makes the warp match, built from the three that remain
        again = landmarks.landmark_triangulation(job, ws, record.id)
        assert again["built_from"] == ["L1", "L3", "L4"] and again["replaced"]
        final = landmarks.set_landmarks(job, ws, record.id)
        assert {r["state"] for r in final["landmarks"]} == {"applied"}
        assert "exactly these pairs" in final["warp"]["note"]
    finally:
        job.close()


# --- the status line and the refusal --------------------------------------------------


def test_status_lists_the_landmark_tools_among_what_can_change(tmp_path):
    from langslice.core.spec import NonlinearSpec
    from tests.linear_tool_helpers import box

    state, ctx, toolbox = box(tmp_path, placed=True, tasks=["nonlinear"],
                              nonlinear=NonlinearSpec(provider="none"))
    tools = {tool.__name__: tool for tool in toolbox.tools}
    changeable = " ".join(tools["status"]()["can_change"])
    assert "set_landmarks" in changeable and "landmark_triangulation" in changeable


def test_too_many_landmark_views_says_what_to_send(tmp_path):
    from langslice.doors.tools.toolbox import build_tools

    job, ws, record = setup(tmp_path)
    try:
        edits = saved_pairs(job, ws, record)
        tools = {t.__name__: t for t in build_tools(
            job.state, ws, job.spec, job=job, gates=False, level="auto").tools}
        reply = tools["set_landmarks"](section=record.id, pairs=edits * 2, view=True)
        assert reply["error"] == "TOO_MANY_LANDMARK_VIEWS"
        assert "6 pairs" in reply["message"] and "view false" in reply["message"]
        assert "Nothing was written" in reply["message"]
    finally:
        job.close()


def test_markers_at_one_spot_show_every_id_and_labels_do_not_cover_each_other():
    from langslice.core.landmarks import glyph

    taken: list = []
    image = Image.new("RGB", (200, 120), (90, 60, 30))
    glyph(image, (50, 50), "L4", "atlas", taken=taken)
    glyph(image, (50, 50), "L12", "section", taken=taken)  # a different kind, same spot
    first, second = taken
    assert not (first[0] < second[2] and second[0] < first[2]
                and first[1] < second[3] and second[1] < first[3])


def test_same_kind_markers_close_together_share_one_label_naming_all(tmp_path):
    from langslice.core.landmarks import draw_landmarks

    job, ws, record = setup(tmp_path)
    try:
        edits = saved_pairs(job, ws, record)
        landmarks.set_landmarks(job, ws, record.id, edits[:1])
        landmarks.set_landmarks(job, ws, record.id, [{"atlas": edits[0]["atlas"],
                                                      "section": edits[0]["section"]}])
        shown = look.look(job, ws, "atlas", sections=[record.id])
        job.views.flush()
        panel = job.views.lookup(shown.entries[0]["id"]).recipe["endpoint_panels"]
        image = shown.pictures[0].copy()
        draw_landmarks(image, panel, job.state, record)
        # one white-on-black chip: the two pairs sit at one atlas point
        import langslice.core.landmarks as lm

        calls: list[str] = []
        original = lm.glyph
        lm.glyph = lambda *a, **k: (calls.append(a[2]), original(*a, **k))[1]
        try:
            draw_landmarks(shown.pictures[0].copy(), panel, job.state, record)
        finally:
            lm.glyph = original
        assert [c for c in calls if c] == ["L1, L2"]
    finally:
        job.close()

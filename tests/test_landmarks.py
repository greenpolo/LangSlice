"""Landmark verbs on a synthetic job: saved-pixel correspondences, batch edits, refusals,
folds, delete-all/undo/redo, staleness, explicit warp application, the one coordinate
system of a picture and its zooms, and the region at a picture point."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from langslice.core import deformation
from langslice.core import landmarks as geometry
from langslice.core.landmark_views import pair_picture
from langslice.core.layers import collecting
from langslice.core.picture_geometry import SourceFrame, source_frame
from langslice.core.spec import NonlinearSpec
from langslice.core.state import StackState
from langslice.core.zoom_grid import draw_grid
from langslice.job.job import Job
from langslice.job.views import DiscardedViews
from langslice.ops import atlas, damage, landmarks, look
from langslice.ops.refusal import Refused
from tests.test_ops_look import _job


def setup(tmp_path: Path):
    job, ws = _job(tmp_path)
    # The landmark tools belong to the nonlinear task: without it the deformation is frozen.
    job.spec = replace(job.spec, tasks=["nonlinear"], nonlinear=NonlinearSpec(provider="none"))
    record = job.state.by_id("s0.png")
    assert record is not None
    record.transform = {"kind": "interactive", "params": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]}
    job.checkpoint()
    return job, ws, record


def saved_pairs(job, ws, record, *, delta=(0.0, 0.0)):
    a = look.look(job, ws, "atlas", sections=[record.id])
    b = look.look(job, ws, "section", sections=[record.id])
    job.views.flush()
    grid = deformation.fit_grid(job.state, ws, record)
    ap = job.views.lookup(a.entries[0]["id"]).recipe["endpoint_panels"][0]
    bp = job.views.lookup(b.entries[0]["id"]).recipe["endpoint_panels"][0]
    pairs = []
    for fraction in [[0.3, 0.3], [0.7, 0.3], [0.5, 0.7]]:
        point = np.asarray(fraction) * grid.image.size - 0.5 + np.asarray(delta)
        native = geometry.transform(np.linalg.inv(grid.placement.atlas_to_section), point)
        ax = geometry.transform(np.linalg.inv(ap["pixel_to_native"]), native)
        bx = geometry.section_picture_point(bp, fraction)
        pairs.append(
            {
                "atlas": {"picture": a.entries[0]["id"], "x": ax[0], "y": ax[1]},
                "section": {"picture": b.entries[0]["id"], "x": bx[0], "y": bx[1]},
            }
        )
    return pairs


def test_batch_edit_persist_undo_delete_and_stale_plane(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        edits = saved_pairs(job, ws, record)
        depth = len(job.undo_stack)
        reply = landmarks.set_landmarks(job, ws, record.id, edits)
        assert reply["changed"] == ["L1", "L2", "L3"]
        assert len(job.undo_stack) == depth + 1 and record.deformation is None
        derived = job.layout.section_dir(record.id) / "landmarks.json"
        assert json.loads(derived.read_text())["pairs"] == record.landmarks["pairs"]
        from langslice.job.checkpoint import read_checkpoint

        restored = StackState.from_dict(read_checkpoint(job.checkpoint_path))
        assert restored.by_id(record.id).landmarks == record.landmarks
        landmarks.set_landmarks(job, ws, record.id, delete=["L2"])
        assert job.undo()
        record = job.state.by_id(record.id)
        assert [p["id"] for p in record.landmarks["pairs"]] == ["L1", "L2", "L3"]
        before = job.snapshot()
        record.position_mm += 0.01
        job.commit(before)
        assert all(p["stale"] for p in record.landmarks["pairs"])
        assert all(p["stale"] for p in json.loads(derived.read_text())["pairs"])
        with pytest.raises(Refused, match="re-place their atlas endpoints"):
            landmarks.landmark_triangulation(job, ws, record.id)
        assert job.undo()
        record = job.state.by_id(record.id)
        assert not any(p["stale"] for p in record.landmarks["pairs"])
        before = job.snapshot()
        job.state.appearance = {"channels": {"gray": {"gamma": 2}}}
        job.commit(before)
        assert not any(p["stale"] for p in record.landmarks["pairs"])
        assert job.undo()  # appearance
        assert job.undo()  # first landmark batch
        empty = json.loads(derived.read_text())
        assert empty["pairs"] == [] and empty["next_id"] == 1
    finally:
        job.close()


def test_bad_batch_is_atomic_and_caption_points_are_refused(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        edits = saved_pairs(job, ws, record)
        edits[1]["section"]["y"] = 100000
        before = job.snapshot()
        with pytest.raises(Refused) as caught:
            landmarks.set_landmarks(job, ws, record.id, edits)
        assert caught.value.code == "BAD_LANDMARK_POINT"
        assert job.snapshot() == before
        generic = look.look(job, ws, "atlas", positions_mm=[record.position_mm])
        edits[0]["atlas"]["picture"] = generic.entries[0]["id"]
        with pytest.raises(Refused) as caught:
            landmarks.set_landmarks(job, ws, record.id, edits[:1])
        assert caught.value.code == "BAD_LANDMARK_PICTURE"
        edits = saved_pairs(job, ws, record)
        edits[0]["section"]["x"] = -1
        with pytest.raises(Refused):
            landmarks.set_landmarks(job, ws, record.id, edits[:1])
        unknown = [{"atlas": {"picture": 999, "x": 1, "y": 1}, "section": edits[0]["section"]}]
        with pytest.raises(Refused) as caught:
            landmarks.set_landmarks(job, ws, record.id, unknown)
        assert caught.value.code == "UNKNOWN_PICTURE"
        depth = len(job.undo_stack)
        assert not landmarks.set_landmarks(job, ws, record.id)["written"]  # no pairs: no write
        assert len(job.undo_stack) == depth
    finally:
        job.close()


def test_side_by_side_zoom_preserves_offsets_gutters_and_old_landmarks(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        landmarks.set_landmarks(job, ws, record.id, saved_pairs(job, ws, record))
        pair = record.landmarks["pairs"][0]
        with collecting() as notes:
            picture = pair_picture(ws, job.state, record, pair, store=job.deformations)
        entry = look.save_pictures(job, ws, "set_landmarks", [picture], notes, {})[0]
        job.views.flush()
        panels = picture.recipe["endpoint_panels"]
        b = geometry.section_picture_point(panels[1], pair["section_fraction_xy"])
        assert b[0] > panels[0]["bounds"][2]
        landmarks.set_landmarks(
            job,
            ws,
            record.id,
            [{"id": "L1", "section": {"picture": entry["id"], "x": b[0], "y": b[1]}}],
        )
        with pytest.raises(Refused) as caught:
            landmarks.set_landmarks(
                job,
                ws,
                record.id,
                [
                    {
                        "id": "L1",
                        "section": {
                            "picture": entry["id"],
                            "x": panels[1]["bounds"][0] - 5,
                            "y": b[1],
                        },
                    }
                ],
            )
        assert caught.value.code == "BAD_LANDMARK_POINT"
        box = [b[0] - 30, b[1] - 25, b[0] + 30, b[1] + 25]
        zoom = look.zoom(job, ws, box=box, picture=entry["id"])
        job.views.flush()
        zoom_record = job.views.lookup(zoom.entries[0]["id"])
        assert zoom_record.recipe["args"]["box"] == [round(v, 2) for v in box]
        zp = zoom_record.recipe["endpoint_panels"][0]
        # On a zoom, endpoints (and further boxes) are the grid's numbers: the
        # pixels of the picture the zoom was cut from.
        point = source_frame(zoom_record.recipe).to_source(
            geometry.section_picture_point(zp, pair["section_fraction_xy"]))
        landmarks.set_landmarks(
            job,
            ws,
            record.id,
            [
                {
                    "id": "L1",
                    "section": {"picture": zoom.entries[0]["id"], "x": point[0], "y": point[1]},
                }
            ],
        )
        assert record.landmarks["pairs"][0]["section_fraction_xy"] == pytest.approx(
            pair["section_fraction_xy"]
        )
        again = look.zoom(
            job,
            ws,
            picture=zoom.entries[0]["id"],
            box=[point[0] - 10, point[1] - 10, point[0] + 10, point[1] + 10],
        )
        job.views.flush()
        again_panel = job.views.lookup(again.entries[0]["id"]).recipe["endpoint_panels"][0]
        again_point = source_frame(job.views.lookup(again.entries[0]["id"]).recipe).to_source(
            geometry.section_picture_point(again_panel, pair["section_fraction_xy"]))
        landmarks.set_landmarks(
            job,
            ws,
            record.id,
            [
                {
                    "id": "L1",
                    "section": {
                        "picture": again.entries[0]["id"],
                        "x": again_point[0],
                        "y": again_point[1],
                    },
                }
            ],
        )
        assert record.landmarks["pairs"][0]["section_fraction_xy"] == pytest.approx(
            pair["section_fraction_xy"]
        )
    finally:
        job.close()


def test_explicit_application_composes_without_running_an_engine(tmp_path, monkeypatch):
    job, ws, record = setup(tmp_path)
    try:
        monkeypatch.setattr(
            "langslice.core.deformable.engines.run_engine",
            lambda *_args: pytest.fail("No registration engine for landmarks"),
        )
        landmarks.set_landmarks(job, ws, record.id, saved_pairs(job, ws, record, delta=(2.0, 1.0)))
        first = landmarks.landmark_triangulation(job, ws, record.id)
        assert first["written"] and not first["composed"]
        applied = job.deformations.current(job.state, record)
        assert applied.engine["name"] == "landmark_triangulation"
        assert applied.inverse_field_mm is not None and not first["fold_fraction"]
        assert not landmarks.landmark_triangulation(job, ws, record.id)["written"]
        assert record.deformation is not None
        updates = saved_pairs(job, ws, record, delta=(3.0, 1.0))
        for k, update in enumerate(updates, 1):
            update["id"] = f"L{k}"
        landmarks.set_landmarks(job, ws, record.id, updates)
        second = landmarks.landmark_triangulation(job, ws, record.id)
        assert second["written"] and second["replaced"] and not second["composed"]
        composed = job.deformations.current(job.state, record)
        assert composed.parent is None
        assert job.undo()
        previous = job.deformations.current(job.state, job.state.by_id(record.id))
        np.testing.assert_array_equal(previous.field_mm, applied.field_mm)
        assert job.redo()
        np.testing.assert_array_equal(
            job.deformations.current(job.state, job.state.by_id(record.id)).field_mm,
            composed.field_mm,
        )
        reloaded = Job.load(job.spec, ws, folder=job.layout.folder)
        assert reloaded.state.by_id(record.id).landmarks
        reloaded.close()
    finally:
        job.close()


def test_replacing_landmarks_preserves_nonzero_baseline_and_intervening_method(tmp_path):
    import copy

    job, ws, record = setup(tmp_path)
    try:
        # A valid, nonzero field standing in for an already applied other method.
        landmarks.set_landmarks(job, ws, record.id, saved_pairs(job, ws, record, delta=(1., .5)))
        landmarks.landmark_triangulation(job, ws, record.id)
        baseline = copy.deepcopy(job.deformations.current(job.state, record))
        baseline.engine["name"] = "fireants"
        baseline_key = "other-method-baseline"
        job.deformations.put(baseline_key, baseline)
        folder = job.deformations.save(record.id, baseline_key, baseline)
        baseline_ref = {"key": baseline_key, "record": job.layout.relative(folder),
                        "linear_key": deformation.linear_key(job.state, record),
                        "steps": [{"method": "fireants_syn"}]}
        before = job.snapshot()
        record.deformation = copy.deepcopy(baseline_ref)
        job.commit(before)
        record.landmarks = {}
        edits = saved_pairs(job, ws, record, delta=(2., 1.))
        extra = copy.deepcopy(edits[0])
        grid = deformation.fit_grid(job.state, ws, record)
        fraction = [.5, .5]
        a_panel = job.views.lookup(extra["atlas"]["picture"]).recipe["endpoint_panels"][0]
        b_panel = job.views.lookup(extra["section"]["picture"]).recipe["endpoint_panels"][0]
        native = geometry.transform(np.linalg.inv(grid.placement.atlas_to_section),
                                    np.asarray(fraction)*grid.image.size-.5 + [3., -1.])
        a = geometry.transform(np.linalg.inv(a_panel["pixel_to_native"]), native)
        b = geometry.section_picture_point(b_panel, fraction)
        extra["atlas"].update(x=a[0], y=a[1])
        extra["section"].update(x=b[0], y=b[1])
        landmarks.set_landmarks(job, ws, record.id, [*edits, extra])
        first = landmarks.landmark_triangulation(job, ws, record.id)
        assert first["composed"] and not first["replaced"]
        landmarks.set_landmarks(job, ws, record.id, delete=["L4"])
        replaced = landmarks.landmark_triangulation(job, ws, record.id)
        assert replaced["composed"] and replaced["replaced"]
        reduced = job.deformations.current(job.state, record)
        np.testing.assert_array_equal(reduced.parent.field_mm, baseline.field_mm)
        assert record.deformation["landmark_baseline_key"] == baseline_key
        assert [step["method"] for step in record.deformation["steps"]] == [
            "fireants_syn", "landmark_triangulation"]
        # The remaining complete set applied fresh to the same original baseline
        # must produce exactly the same field, without influence from L4.
        before = job.snapshot()
        record.deformation = copy.deepcopy(baseline_ref)
        job.commit(before)
        landmarks.landmark_triangulation(job, ws, record.id)
        np.testing.assert_array_equal(
            job.deformations.current(job.state, record).field_mm, reduced.field_mm)
        # A different method applied after landmarks establishes a new baseline.
        intervening = copy.deepcopy(reduced)
        intervening.engine["name"] = "image_model"
        job.deformations.put("intervening-method", intervening)
        folder = job.deformations.save(record.id, "intervening-method", intervening)
        before = job.snapshot()
        record.deformation = {"key": "intervening-method", "record": job.layout.relative(folder),
                              "linear_key": deformation.linear_key(job.state, record),
                              "steps": [{"method": "trace_borders"}]}
        job.commit(before)
        latest = landmarks.landmark_triangulation(job, ws, record.id)
        assert latest["composed"] and not latest["replaced"]
        result = job.deformations.current(job.state, record)
        np.testing.assert_array_equal(result.parent.field_mm, intervening.field_mm)
        assert record.deformation["landmark_baseline_key"] == "intervening-method"
    finally:
        job.close()


def test_doors_have_typed_landmarks_and_apply_through_normal_tools(tmp_path):
    from langslice.doors.declarations import arguments_schema
    from langslice.doors.tools.toolbox import build_tools
    from langslice.ops.exports import export_maps

    job, ws, record = setup(tmp_path)
    try:
        spec = replace(job.spec, tasks=["nonlinear"], nonlinear=NonlinearSpec(provider="none"))
        job.spec = spec
        box = build_tools(job.state, ws, spec, job=job)
        assert "set_landmarks" in box.names and "landmark_triangulation" in box.names
        tools = {tool.__name__: tool for tool in box.tools}
        schema = arguments_schema("set_landmarks")
        assert schema["$defs"]["LandmarkPoint"]["required"] == ["picture", "x", "y"]
        bad = tools["set_landmarks"](
            section=record.id,
            pairs=[{"section": {"picture": 1, "x": 2.0, "y": 3.0, "fraction": 0.1}}],
        )
        assert bad["error"] == "UNKNOWN_ARGUMENTS"
        assert bad["problems"][0]["argument"] == "pairs[0].section"
        done = tools["set_landmarks"](
            section=record.id, pairs=saved_pairs(job, ws, record), view=False
        )
        assert done["changed"] == ["L1", "L2", "L3"]
        applied = tools["landmark_triangulation"](section=record.id, view=False)
        assert applied["status"] == "ok" and applied["written"]
        exported = export_maps(job, ws, ids=[record.id])
        assert record.id in exported.sections
        assert (job.layout.section_dir(record.id) / "coords.tif").is_file()
        assert (job.layout.exports_dir / "visualign.json").is_file()
    finally:
        job.close()


def test_mcp_shrinking_saves_delivered_geometry_without_changing_old_picture(tmp_path, monkeypatch):
    import base64
    from io import BytesIO

    from mcp.types import ImageContent
    from PIL import Image

    from langslice.doors.mcp import server
    from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
    from langslice.doors.tools.reply import fit_reply

    job, ws, record = setup(tmp_path)
    try:
        edits = saved_pairs(job, ws, record)
        point = edits[0]["section"]
        old = job.views.lookup(point["picture"])
        job.views.flush()
        original = Image.open(job.folder / old.path / "view.jpg").convert("RGB")
        expected = landmarks._endpoint(job, record, point, "section")["canonical_xy"]
        # A deliberately tiny transport budget guarantees multiple resizes.
        monkeypatch.setattr(server, "fit_reply", lambda items: fit_reply(items, budget=12000))
        blocks = server.result_blocks(
            {"pictures": [{"id": old.seq}], "regions": [{"id": old.seq}],
             "results": [{"pictures": [{"id": old.seq}]}],
             TOOL_MEDIA_PARTS_KEY: [original]}, job=job
        )
        new_id = json.loads(blocks[0].text)["pictures"][0]["id"]
        assert new_id != old.seq
        body = json.loads(blocks[0].text)
        assert body["regions"][0]["id"] == old.seq
        assert body["results"][0]["pictures"][0]["id"] == new_id
        block = next(block for block in blocks if isinstance(block, ImageContent))
        jpeg = base64.b64decode(block.data)
        delivered = Image.open(BytesIO(jpeg))
        assert delivered.width < original.width
        # Calculate expected returned pixels independently of saved metadata.
        x = (point["x"] + .5) * delivered.width / original.width - .5
        y = (point["y"] + .5) * delivered.height / original.height - .5
        actual = landmarks._endpoint(job, record, {"picture": new_id, "x": x, "y": y}, "section")
        np.testing.assert_allclose(actual["canonical_xy"], expected, atol=1e-12)
        new = job.views.lookup(new_id)
        job.views.flush()
        assert (job.folder / new.path / "view.jpg").read_bytes() == jpeg
        assert job.views.lookup(old.seq).recipe == old.recipe
        with pytest.raises(Refused, match="caption/gutter"):
            landmarks._endpoint(job, record,
                                {"picture": new_id, "x": x, "y": delivered.height-1}, "section")
        zoomed = look.zoom(job, ws, picture=new_id, box=[x-10, y-10, x+10, y+10])
        assert zoomed.redrawn
        assert server.result_blocks({TOOL_MEDIA_PARTS_KEY: [original]}, job=job)
    finally:
        job.close()


def test_a_frozen_deformation_refuses_pairs_and_triangulation_alike(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        edits = saved_pairs(job, ws, record)
        job.spec = replace(job.spec, frozen={record.id: ["deformation"]})
        depth = len(job.undo_stack)
        for call in (lambda: landmarks.set_landmarks(job, ws, record.id, edits),
                     lambda: landmarks.landmark_triangulation(job, ws, record.id)):
            with pytest.raises(Refused) as refused:
                call()
            assert refused.value.code == "FROZEN"
        assert len(job.undo_stack) == depth and not record.landmarks
    finally:
        job.close()


def _folded_pairs(job, ws, record):
    """Three pairs whose atlas endpoints of the first two are swapped: a fold."""
    edits = saved_pairs(job, ws, record)
    edits[0]["atlas"], edits[1]["atlas"] = edits[1]["atlas"], edits[0]["atlas"]
    return edits


@pytest.mark.parametrize("folded", [True, False])
def test_a_fold_is_applied_and_reported_and_a_sound_warp_reports_none(tmp_path, folded):
    job, ws, record = setup(tmp_path)
    try:
        edits = (_folded_pairs(job, ws, record) if folded
                 else saved_pairs(job, ws, record, delta=(2.0, 1.0)))
        landmarks.set_landmarks(job, ws, record.id, edits)
        done = landmarks.landmark_triangulation(job, ws, record.id)
        if not folded:
            assert "folds" not in done and not record.deformation["landmark_folds"]
            return
        assert done["written"]
        folds = done["folds"]
        assert 0 < folds["area_fraction"] <= 1
        assert folds["pairs"] and set(folds["pairs"]) <= {"L1", "L2", "L3"}
        assert folds["triangle_count"] >= 1 and folds["triangles"]
        stored = record.deformation["landmark_folds"]
        assert stored["triangles"] and stored["area_fraction"] == folds["area_fraction"]
        assert landmarks.landmark_triangulation(job, ws, record.id)["folds"] == folds
    finally:
        job.close()


def test_repeated_section_points_refuse_the_triangulation(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        edits = saved_pairs(job, ws, record)
        edits[1]["section"] = dict(edits[0]["section"])
        landmarks.set_landmarks(job, ws, record.id, edits)
        with pytest.raises(Refused) as caught:
            landmarks.landmark_triangulation(job, ws, record.id)
        assert caught.value.code == "LANDMARK_TRIANGULATION_FAILED"
        assert caught.value.details["pairs"] == ["L1", "L2"]
    finally:
        job.close()


def test_a_stale_plane_refuses_a_picture_from_the_old_plane(tmp_path):
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
        with pytest.raises(Refused) as caught:
            landmarks.set_landmarks(job, ws, record.id, [{"id": "L1", "atlas": edits[0]["atlas"]}])
        assert caught.value.code == "STALE_LANDMARK_PLANE"
        # a picture drawn for no section carries no atlas geometry for a pair
        generic = look.look(job, ws, "atlas", positions_mm=[record.position_mm])
        wrong = {**edits[0], "atlas": {**edits[0]["atlas"], "picture": generic.entries[0]["id"]}}
        with pytest.raises(Refused) as caught:
            landmarks.set_landmarks(job, ws, record.id, [wrong])
        assert caught.value.code == "BAD_LANDMARK_PICTURE"
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


def test_undo_and_redo_through_the_tools_report_what_they_reverted(tmp_path):
    from langslice.doors.tools.toolbox import build_tools

    job, ws, record = setup(tmp_path)
    try:
        box = build_tools(job.state, ws, job.spec, job=job)
        tools = {tool.__name__: tool for tool in box.tools}
        tools["set_landmarks"](section=record.id, pairs=saved_pairs(job, ws, record), view=False)
        tools["landmark_triangulation"](section=record.id, view=False)
        undone = tools["undo"]()
        assert undone["undone"] == {"sections": [record.id], "fields": ["deformation"]}
        redone = tools["redo"]()
        assert redone["redone"] == {"sections": [record.id], "fields": ["deformation"]}
        assert tools["undo"]()["undone"]["fields"] == ["deformation"]
        assert tools["undo"]()["undone"]["fields"] == ["landmarks"]
    finally:
        job.close()


def test_a_landmark_edit_alone_leaves_old_pictures_unstale(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        edits = saved_pairs(job, ws, record)
        first = look.look(job, ws, "section", sections=[record.id])
        job.views.flush()
        number = first.entries[0]["id"]
        landmarks.set_landmarks(job, ws, record.id, edits)
        assert not look.zoom(job, ws, box=[0, 0, 40, 40], picture=number).stale
        newer = look.look(job, ws, "section", sections=[record.id])
        job.views.flush()
        landmarks.set_landmarks(job, ws, record.id, delete=["L2"])
        assert not look.zoom(job, ws, box=[0, 0, 400, 400], picture=newer.entries[0]["id"]).stale
        before = job.snapshot()
        record.position_mm += 0.01
        job.commit(before)
        assert look.zoom(job, ws, box=[0, 0, 40, 40], picture=number).stale
    finally:
        job.close()


def test_the_listing_follows_the_warp_states(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        edits = saved_pairs(job, ws, record)
        landmarks.set_landmarks(job, ws, record.id, edits)
        assert landmarks.set_landmarks(job, ws, record.id)["warp"]["built_from"] == []
        done = landmarks.landmark_triangulation(job, ws, record.id)
        assert done["built_from"] == ["L1", "L2", "L3"]
        gone = landmarks.set_landmarks(job, ws, record.id, delete=["L2"])
        assert gone["warp"]["deleted_still_in_warp"] == ["L2"]
        moved = [{"id": "L3", "section": {**edits[2]["section"],
                                          "x": edits[2]["section"]["x"] + 5}}]
        more = landmarks.set_landmarks(job, ws, record.id, moved + [
            {"atlas": edits[1]["atlas"], "section": edits[1]["section"]}])
        assert {r["id"]: r["state"] for r in more["landmarks"]} == {
            "L1": "applied", "L3": "moved", "L4": "pending"}
        assert more["counts"]["moved"] == 1 and more["counts"]["pending"] == 1
        again = landmarks.landmark_triangulation(job, ws, record.id)
        assert again["built_from"] == ["L1", "L3", "L4"] and again["replaced"]
        final = landmarks.set_landmarks(job, ws, record.id)
        assert {r["state"] for r in final["landmarks"]} == {"applied"}
    finally:
        job.close()


def test_the_reply_names_the_region_at_each_atlas_endpoint_and_flags_one_outside(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        reply = landmarks.set_landmarks(job, ws, record.id, saved_pairs(job, ws, record))
        assert set(reply["atlas_endpoint_regions"]) == {"L1", "L2", "L3"}
        assert "outside_atlas" not in reply
        moved = saved_pairs(job, ws, record)[0]["section"]
        reply = landmarks.set_landmarks(job, ws, record.id, [{"id": "L1", "section": moved}])
        assert "atlas_endpoint_regions" not in reply  # a section-only edit places no atlas end
        seq = record.landmarks["pairs"][0]["atlas"]["picture"]
        panel = next(p for p in job.views.lookup(seq).recipe["endpoint_panels"]
                     if p["role"] == "atlas")
        x0, y0 = panel["bounds"][:2]
        corner = {"picture": seq, "x": x0 + 1, "y": y0 + 1}
        reply = landmarks.set_landmarks(job, ws, record.id, [{"id": "L2", "atlas": corner}])
        assert reply["written"] is True and reply["outside_atlas"] == ["L2"]
        assert reply["atlas_endpoint_regions"] == {"L2": "outside the atlas"}
    finally:
        job.close()


# --- one coordinate system: a picture, its zooms and the atlas drawn on it -------------


def _zoom(job, ws, picture, box):
    done = look.zoom(job, ws, box=box, picture=picture)
    job.views.flush()
    return done


def test_grid_lines_sit_where_their_labels_say():
    image = Image.new("RGB", (300, 200), (0, 0, 0))
    frame = SourceFrame(origin=[100.0, 40.0], scale=[0.5, 0.5], shown=[300, 200])
    draw_grid(image, frame)
    pixels = np.asarray(image).astype(int)
    green = np.abs(pixels - np.array([0x7d, 0xff, 0x3a])).max(axis=-1) <= 10
    column = round(float(frame.to_picture([160, 60])[0]))
    assert green[:, column].sum() > 30
    assert green[:, column + 7].sum() < 5  # between two lines


def test_a_number_read_off_a_zoom_grid_lands_where_it_does_on_the_source_picture(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        edits = saved_pairs(job, ws, record)
        source = edits[0]["section"]
        whole_number = source["picture"]
        x, y = float(source["x"]), float(source["y"])
        zoom = _zoom(job, ws, whole_number, [x - 50, y - 40, x + 50, y + 40])
        zoom_number = zoom.entries[0]["id"]
        a = landmarks.set_landmarks(job, ws, record.id, [
            {"atlas": edits[0]["atlas"], "section": {"picture": whole_number, "x": x, "y": y}}])
        b = landmarks.set_landmarks(job, ws, record.id, [
            {"atlas": edits[0]["atlas"], "section": {"picture": zoom_number, "x": x, "y": y}}])
        pairs = {p["id"]: p for p in record.landmarks["pairs"]}
        assert a["changed"] == ["L1"] and b["changed"] == ["L2"]
        assert np.allclose(pairs["L1"]["section_fraction_xy"],
                           pairs["L2"]["section_fraction_xy"], atol=2e-3)
        # a zoom of the zoom keeps the same numbers
        deeper = _zoom(job, ws, zoom_number, [x - 20, y - 15, x + 20, y + 15])
        c = landmarks.set_landmarks(job, ws, record.id, [
            {"atlas": edits[0]["atlas"],
             "section": {"picture": deeper.entries[0]["id"], "x": x, "y": y}}])
        pairs = {p["id"]: p for p in record.landmarks["pairs"]}
        assert np.allclose(pairs[c["changed"][0]]["section_fraction_xy"],
                           pairs["L1"]["section_fraction_xy"], atol=2e-3)
        with pytest.raises(Refused) as caught:
            landmarks.set_landmarks(job, ws, record.id, [
                {"atlas": edits[0]["atlas"],
                 "section": {"picture": deeper.entries[0]["id"], "x": x + 200, "y": y}}])
        assert caught.value.code == "BAD_LANDMARK_POINT"
    finally:
        job.close()


def test_an_atlas_zoom_and_an_overlay_zoom_read_the_same_numbers(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        edits = saved_pairs(job, ws, record)
        atlas_end = edits[1]["atlas"]
        x, y = float(atlas_end["x"]), float(atlas_end["y"])
        zoom = _zoom(job, ws, atlas_end["picture"], [x - 40, y - 30, x + 40, y + 30])
        on_source = landmarks.set_landmarks(job, ws, record.id, [
            {"atlas": {"picture": atlas_end["picture"], "x": x, "y": y},
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
            {"atlas": atlas_end,
             "section": {"picture": number, "x": float(point[0]), "y": float(point[1])}}])
        second = landmarks.set_landmarks(job, ws, record.id, [
            {"atlas": atlas_end,
             "section": {"picture": near.entries[0]["id"], "x": float(point[0]),
                         "y": float(point[1])}}])
        pairs = {p["id"]: p for p in record.landmarks["pairs"]}
        assert np.allclose(pairs[first["changed"][0]]["section_fraction_xy"],
                           pairs[second["changed"][0]]["section_fraction_xy"], atol=3e-3)
    finally:
        job.close()


def test_a_dry_run_resolves_the_pictures_a_real_call_would(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        edits = saved_pairs(job, ws, record)
        number = edits[0]["section"]["picture"]
        dry = DiscardedViews(job.layout)
        assert dry.lookup(number) is not None
        job.views = dry
        job.persist = False
        reply = landmarks.set_landmarks(job, ws, record.id, edits)
        assert reply["changed"] == ["L1", "L2", "L3"]
        made = look.look(job, ws, "section", sections=[record.id])
        assert made.entries[0]["id"] > number  # it numbers its own pictures after the saved ones
    finally:
        job.close()


# --- the region at a picture point -------------------------------------------------------


def _overlay(job, ws, record):
    shown = look.look(job, ws, "overlay", sections=[record.id])
    job.views.flush()
    return shown.entries[0]["id"]


def _inside_point(job, seq):
    """Pixel centre (x, y) with a region drawn, and the id drawn there."""
    labels = tifffile.imread(job.layout.resolve(job.views.lookup(seq).path) / "labels.tif")
    rows, cols = np.nonzero(labels)
    k = len(rows) // 2
    return float(cols[k]), float(rows[k]), int(labels[rows[k], cols[k]])


def test_the_region_at_a_point_is_what_the_overlay_drew_and_the_same_on_a_zoom(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        seq = _overlay(job, ws, record)
        x, y, region_id = _inside_point(job, seq)
        rows = atlas.region_at_points(job, ws, [{"picture": seq, "x": x, "y": y}])
        entry = ws.atlas.structures[region_id]
        assert rows[0]["region"] == {"acronym": entry["acronym"], "name": entry["name"],
                                     "id": region_id}
        outside = atlas.region_at_points(job, ws, [{"picture": seq, "x": 0.0, "y": 0.0}])
        assert outside[0]["region"] is None
        reply = atlas.grep_atlas(job, ws, points=[{"picture": seq, "x": x, "y": y}])
        assert reply["points"][0]["region"]["id"] == region_id and "rows" not in reply
        zoom_seq = _zoom(job, ws, seq, [x - 40, y - 30, x + 40, y + 30]).entries[0]["id"]
        on_zoom = atlas.region_at_points(job, ws, [{"picture": zoom_seq, "x": x, "y": y}])
        assert on_zoom[0]["region"] == rows[0]["region"]
    finally:
        job.close()


def test_a_section_only_picture_is_refused_for_atlas_lookup(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        shown = look.look(job, ws, "section", sections=[record.id])
        job.views.flush()
        seq = shown.entries[0]["id"]
        with pytest.raises(Refused) as caught:
            atlas.region_at_points(job, ws, [{"picture": seq, "x": 50, "y": 50}])
        assert caught.value.code == "NO_ATLAS_IN_PICTURE"
        with pytest.raises(Refused) as caught:
            atlas.region_at_points(job, ws, [{"picture": 999, "x": 1, "y": 1}])
        assert caught.value.code == "UNKNOWN_PICTURE"
        with pytest.raises(Refused):
            atlas.region_at_points(
                job, ws, [{"picture": seq, "x": 1, "y": 1}] * (atlas.MAX_POINTS + 1))
        with pytest.raises(Refused):
            atlas.grep_atlas(job, ws)
    finally:
        job.close()


def test_marking_damage_does_not_make_a_zoom_stale(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        seq = _overlay(job, ws, record)
        regions = [str(v["acronym"]) for v in ws.atlas.structures.values()][1:2]
        damage.mark_damage(job, ws, record.id, regions)
        assert look.zoom(job, ws, box=[10, 10, 100, 100], picture=seq).stale is False
        job.state.by_id(record.id).position_mm += 0.05  # a placement change does
        assert look.zoom(job, ws, box=[10, 10, 100, 100], picture=seq).stale is True
    finally:
        job.close()

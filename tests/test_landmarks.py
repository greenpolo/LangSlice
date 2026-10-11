"""Saved-pixel correspondences, landmark lifecycle and explicit warp application."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from langslice.core import deformation
from langslice.core import landmarks as geometry
from langslice.core.landmark_views import pair_picture
from langslice.core.layers import collecting
from langslice.core.spec import NonlinearSpec
from langslice.core.state import StackState
from langslice.job.job import Job
from langslice.ops import landmarks, look
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
        checkpoint = json.loads(Path(job.checkpoint_path).read_text())
        restored = StackState.from_dict(checkpoint)
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
        with pytest.raises(Refused, match="Update the stale"):
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
        a = geometry.transform(np.linalg.inv(panels[0]["pixel_to_native"]), pair["atlas_native_xy"])
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
        point = geometry.section_picture_point(zp, pair["section_fraction_xy"])
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
        again_point = geometry.section_picture_point(again_panel, pair["section_fraction_xy"])
        updated = landmarks.set_landmarks(
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
        assert updated["landmarks"][0]["section_fraction_xy"] == pytest.approx(
            pair["section_fraction_xy"]
        )
        # A batch can inspect each endpoint, and all pictures get their own IDs.
        batch = look.zoom(
            job,
            ws,
            requests=[
                {"picture": entry["id"], "box": box},
                {"picture": entry["id"], "box": [a[0] - 20, a[1] - 20, a[0] + 20, a[1] + 20]},
            ],
        )
        assert len(batch.entries) == 2 and len(batch.results) == 2
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
        baseline.engine["name"] = "ants"
        baseline_key = "other-method-baseline"
        job.deformations.put(baseline_key, baseline)
        folder = job.deformations.save(record.id, baseline_key, baseline)
        baseline_ref = {"key": baseline_key, "record": job.layout.relative(folder),
                        "linear_key": deformation.linear_key(job.state, record),
                        "steps": [{"method": "ants_syn"}]}
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
            "ants_syn", "landmark_triangulation"]
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


def test_batch_zoom_reports_each_failure_without_losing_successful_pictures(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        source = look.look(job, ws, "section", sections=[record.id])
        job.views.flush()
        done = look.zoom(
            job,
            ws,
            requests=[
                {"picture": source.entries[0]["id"], "box": [20, 20, 70, 70]},
                {"picture": 100000, "box": [0, 0, 20, 20]},
            ],
        )
        assert len(done.entries) == 1
        assert done.results[0]["status"] == "ok"
        assert done.results[1]["error"] == "UNKNOWN_PICTURE"
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
        box = build_tools(job.state, ws, spec, job=job, gates=False)
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

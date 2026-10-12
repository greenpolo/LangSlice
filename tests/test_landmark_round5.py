"""Round-5 landmark fixes: the region at each atlas endpoint and the warning for
one outside the atlas, holds over the tissue only, one label for a pair whose
endpoints meet, the warped overlay's caption, the job-not-found hint."""

from __future__ import annotations

import numpy as np
import pytest

from langslice.core import landmarks as drawing
from langslice.doors.jobs import NoJob, find
from langslice.ops import landmarks, look
from langslice.ops.landmarks import automatic_holds
from tests.test_landmarks import saved_pairs, setup


def test_the_reply_names_the_region_at_each_new_atlas_endpoint(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        reply = landmarks.set_landmarks(job, ws, record.id, saved_pairs(job, ws, record))
        assert set(reply["atlas_endpoint_regions"]) == {"L1", "L2", "L3"}
        assert all(name != "outside the atlas"
                   for name in reply["atlas_endpoint_regions"].values())
        assert "warning" not in reply and "outside_atlas" not in reply
        # a section-only edit places no atlas endpoint, so names none
        moved = saved_pairs(job, ws, record)[0]["section"]
        reply = landmarks.set_landmarks(job, ws, record.id, [{"id": "L1", "section": moved}])
        assert "atlas_endpoint_regions" not in reply
    finally:
        job.close()


def test_an_atlas_endpoint_outside_the_atlas_is_saved_with_a_warning(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        landmarks.set_landmarks(job, ws, record.id, saved_pairs(job, ws, record))
        seq = record.landmarks["pairs"][0]["atlas"]["picture"]
        panel = next(p for p in job.views.lookup(seq).recipe["endpoint_panels"]
                     if p["role"] == "atlas")
        x0, y0 = panel["bounds"][:2]
        corner = {"picture": seq, "x": x0 + 1, "y": y0 + 1}
        reply = landmarks.set_landmarks(job, ws, record.id, [{"id": "L2", "atlas": corner}])
        assert reply["written"] is True
        assert reply["atlas_endpoint_regions"] == {"L2": "outside the atlas"}
        assert reply["outside_atlas"] == ["L2"]
        assert "L2" in reply["warning"] and "outside the atlas" in reply["warning"]
    finally:
        job.close()


def test_holds_are_kept_only_on_the_tissue():
    size = (700, 500)
    tissue = np.zeros((size[1], size[0]), bool)
    tissue[100:400, 150:550] = True
    everywhere = automatic_holds(np.empty((0, 2)), size)
    holds = automatic_holds(np.empty((0, 2)), size, tissue=tissue)
    assert 0 < len(holds) < len(everywhere)
    rows, cols = np.rint(holds[:, 1]).astype(int), np.rint(holds[:, 0]).astype(int)
    assert tissue[rows, cols].all()


def test_an_applied_pair_whose_endpoints_meet_is_labelled_once(tmp_path, monkeypatch):
    job, ws, record = setup(tmp_path)
    try:
        landmarks.set_landmarks(job, ws, record.id, saved_pairs(job, ws, record, delta=(9, 0)))
        landmarks.landmark_triangulation(job, ws, record.id)
        labels: list[str] = []
        real = drawing.glyph
        monkeypatch.setattr(drawing, "glyph", lambda image, xy, label, role, **kw: (
            labels.append(label), real(image, xy, label, role, **kw))[1])
        shown = look.look(job, ws, "overlay", sections=[record.id])
        drawn = [label for label in labels if label]
        assert len(labels) == 6 and sorted(drawn) == ["L1", "L2", "L3"], labels
        assert "the section is warped onto the atlas" in shown.entries[0]["caption"]
    finally:
        job.close()


def test_no_job_names_the_job_dir_way(tmp_path):
    with pytest.raises(NoJob, match="--job-dir"):
        find(tmp_path)

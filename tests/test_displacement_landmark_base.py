"""DISPLACEMENT_OUTSIZED counts only what the dense fits add beyond the agent's landmark warp."""

from __future__ import annotations

import numpy as np

from langslice.core import deformation
from langslice.core.deformable import finish_fit, prepare_warp
from langslice.core.deformable.engines import EngineResult
from langslice.core.deformable.fit import WorkingGrid
from langslice.ops import landmarks
from tests.test_landmarks import saved_pairs, setup


def _codes(record) -> list[str]:
    return [f["code"] for f in record.diagnostics["flags"]]


def _added_step(job, ws, record, current, shift_mm):
    """A dense fit composed onto *current*: a uniform step of *shift_mm* along x."""
    grid = deformation.fit_grid(job.state, ws, record)
    size = grid.image.size
    prepared = prepare_warp(ws.atlas, grid.placement, size, deformation.DETAIL_LEVEL,
                            previous=current)
    prepared.grid = WorkingGrid(size, grid.placement.section_mm_per_px, size)
    field = np.zeros((size[1], size[0], 2), np.float32)
    field[..., 0] = shift_mm
    return finish_fit(prepared, EngineResult(
        field_mm=field, inverse_field_mm=-field, inverse_source="test step",
        native_parameters={}, runtime_s=0.0, engine_version=""))


def test_a_large_landmark_move_is_not_outsized_and_fits_count_only_their_addition(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        landmarks.set_landmarks(job, ws, record.id, saved_pairs(job, ws, record))
        landmarks.landmark_triangulation(job, ws, record.id)
        extent_mm = _added_step(job, ws, record, None, 0.0).diagnostics[
            "displacement"]["tissue_extent_mm"]
        # Replace the pairs by a 38 px move.
        record.landmarks = {}
        landmarks.set_landmarks(job, ws, record.id, saved_pairs(
            job, ws, record, delta=(38., 0.)))
        reply = landmarks.landmark_triangulation(job, ws, record.id)
        moved = job.deformations.current(job.state, record)
        whole = np.linalg.norm(moved.field_mm, axis=-1)[moved.footprint]
        assert whole.max() > 0.1 * extent_mm  # past the old limit
        assert "DISPLACEMENT_OUTSIZED" not in _codes(moved)
        assert all(f["code"] != "DISPLACEMENT_OUTSIZED" for f in reply["flags"])
        # A fit composed on it: measured from the landmark warp.
        small = _added_step(job, ws, record, moved, 0.02 * extent_mm)
        assert small.diagnostics["displacement"]["beyond_landmark_warp"]
        assert "DISPLACEMENT_OUTSIZED" not in _codes(small)
        large = _added_step(job, ws, record, moved, 0.2 * extent_mm)
        assert "DISPLACEMENT_OUTSIZED" in _codes(large)
        # Landmarks applied over that fit make the combined field the base.
        folder = job.deformations.save(record.id, "large", large)
        job.deformations.put("large", large)
        record.deformation = {
            "key": "large", "record": job.layout.relative(folder),
            "linear_key": deformation.linear_key(job.state, record), "steps": []}
        record.landmarks = {}
        landmarks.set_landmarks(job, ws, record.id, saved_pairs(job, ws, record, delta=(3., 0.)))
        landmarks.landmark_triangulation(job, ws, record.id)
        over = job.deformations.current(job.state, record)
        assert "DISPLACEMENT_OUTSIZED" not in _codes(over)
        assert "DISPLACEMENT_OUTSIZED" not in _codes(
            _added_step(job, ws, record, over, 0.02 * extent_mm))
    finally:
        job.close()


def test_a_chain_without_landmarks_is_measured_from_the_placement(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        first = _added_step(job, ws, record, None, 0.0)
        extent_mm = first.diagnostics["displacement"]["tissue_extent_mm"]
        assert not first.diagnostics["displacement"]["beyond_landmark_warp"]
        assert "DISPLACEMENT_OUTSIZED" not in _codes(first)
        big = _added_step(job, ws, record, None, 0.2 * extent_mm)
        assert "DISPLACEMENT_OUTSIZED" in _codes(big)
        # Two steps that add up past the limit are flagged as a whole.
        half = _added_step(job, ws, record, None, 0.06 * extent_mm)
        assert "DISPLACEMENT_OUTSIZED" not in _codes(half)
        both = _added_step(job, ws, record, half, 0.06 * extent_mm)
        assert "DISPLACEMENT_OUTSIZED" in _codes(both)
    finally:
        job.close()

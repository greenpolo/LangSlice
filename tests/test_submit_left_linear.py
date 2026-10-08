"""``submit``'s ``left_linear`` and the host's ``require_deformation``; the
job's starting positions and the gates that read them."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from langslice.agent.engine import build_context
from langslice.core.spec import JobSpec, NonlinearSpec
from langslice.job.job import DEFAULT_POSITION, Job, default_positions
from langslice.ops import submit as ops_submit
from langslice.ops.refusal import Refused
from tests.fakes import SlabAtlas

_ATLAS = SlabAtlas()
IDENTITY = {"kind": "interactive", "params": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]}


def _open(tmp_path: Path, n: int = 3, **spec_kwargs: Any) -> tuple[Job, Any]:
    for index in range(n):
        Image.fromarray(np.full((30, 40, 3), 40 + 10 * index, dtype=np.uint8)).save(
            tmp_path / f"s{index}.png")
    spec = JobSpec(image_folder=str(tmp_path), model="fake-model", preprocess="none",
                   **spec_kwargs)
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: _ATLAS)
    return Job.open(spec, ctx, folder=ctx.job_folder, results_path=ctx.results_path), ctx


def _placed(tmp_path: Path, **nonlinear: Any) -> Job:
    """A Nonlinear-only job whose sections carry a position and a transform."""
    job, _ = _open(tmp_path, tasks=["nonlinear"], nonlinear=NonlinearSpec(
        provider="none", **nonlinear))
    for index, record in enumerate(job.state.in_order()):
        record.position_mm = 2.0 + 0.2 * index
        record.transform = dict(IDENTITY)
    return job


def _refused(job: Job, **kwargs: Any) -> Refused:
    with pytest.raises(Refused) as refused:
        ops_submit.submit(job, **kwargs)
    return refused.value


def test_left_linear_covers_the_sections_it_names(tmp_path: Path):
    job = _placed(tmp_path)
    missing = _refused(job, summary="done")
    assert missing.code == "MISSING_DEFORMATIONS"
    assert "left_linear" in missing.payload()["message"]
    partial = _refused(job, left_linear=[{"id": "s0.png", "reason": "torn"}])
    assert [row["id"] for row in partial.payload()["sections"]] == ["s1.png", "s2.png"]
    done = ops_submit.submit(job, summary="done", left_linear=[
        {"id": "s0.png", "reason": "torn"}, {"id": "s1", "reason": "too faint"},
        {"id": "s2.png", "reason": "fits already"}])
    assert done.left_linear == {"s0.png": "torn", "s1.png": "too faint",
                                "s2.png": "fits already"}
    assert job.state.submitted
    held = job.state.by_id("s1.png").deformation  # type: ignore[union-attr]
    assert held is not None and held["keep_linear"] == "too faint"
    assert job.undo() and job.state.by_id("s1.png").deformation is None  # type: ignore[union-attr]


@pytest.mark.parametrize("entries,code", [
    ([{"id": "s0.png"}], "BAD_ARGS"),
    ([{"id": "s0.png", "reason": " "}], "BAD_ARGS"),
    ([{"id": "s0.png", "reason": "a", "extra": 1}], "BAD_ARGS"),
    ("s0.png", "BAD_ARGS"),
    ([{"id": "s0.png", "reason": "a"}, {"id": "s0", "reason": "b"}], "BAD_ARGS"),
    ([{"id": "nope.png", "reason": "a"}], "UNKNOWN_SLICE_IDS"),
    ([{"id": "0", "reason": "a"}], "UNKNOWN_SLICE_IDS"),
])
def test_bad_left_linear_is_refused(tmp_path: Path, entries, code):
    job = _placed(tmp_path)
    assert _refused(job, left_linear=entries).code == code
    assert not job.state.submitted


def test_the_host_can_require_a_deformation_on_every_section(tmp_path: Path):
    job = _placed(tmp_path, require_deformation=True)
    refused = _refused(job, left_linear=[{"id": "s0.png", "reason": "torn"}])
    assert refused.code == "DEFORMATION_REQUIRED"
    assert _refused(job).code == "MISSING_DEFORMATIONS"
    spec = JobSpec.from_dict(job.spec.to_dict())
    assert spec.nonlinear.require_deformation is True
    assert "require_deformation" not in JobSpec(image_folder=".").to_dict()["nonlinear"]


def test_left_linear_is_for_nonlinear_runs(tmp_path: Path):
    job, _ = _open(tmp_path, tasks=["transform"])
    for record in job.state.slices:
        record.position_mm, record.transform = 2.0, dict(IDENTITY)
    assert _refused(job, left_linear=[{"id": "s0.png", "reason": "a"}]).code == "BAD_ARGS"


def test_ingest_gives_starting_positions_that_submit_refuses(tmp_path: Path):
    job, ctx = _open(tmp_path, tasks=["position"])
    low, high = ctx.position_range
    positions = [record.position_mm for record in job.state.in_order()]
    assert all(position is not None and low <= position <= high for position in positions)
    steps = np.diff(np.asarray(positions, dtype=float))
    assert np.allclose(steps, job.state.interval_mm)  # evenly spaced, in stack order
    assert all(record.position_source == DEFAULT_POSITION for record in job.state.slices)
    refused = _refused(job)
    assert refused.code == "MISSING_POSITIONS"
    assert refused.payload()["at_default"] == ["s0.png", "s1.png", "s2.png"]
    with job.writing():  # a written position is the writer's own
        before = job.snapshot()
        job.state.by_id("s1.png").position_mm = 5.0  # type: ignore[union-attr]
        job.commit(before)
    assert job.state.by_id("s1.png").position_source == ""  # type: ignore[union-attr]
    assert _refused(job).payload()["at_default"] == ["s0.png", "s2.png"]
    assert job.undo()
    assert job.state.by_id("s1.png").position_source == DEFAULT_POSITION  # type: ignore[union-attr]


def test_starting_positions_fill_in_around_supplied_ones(tmp_path: Path):
    job, ctx = _open(tmp_path, n=5, tasks=["position"],
                     inputs={"positions": {"s1.png": 2.0, "s3.png": 3.0}})
    by = {record.id: record for record in job.state.slices}
    assert by["s1.png"].position_mm == 2.0 and by["s1.png"].position_source == ""
    assert by["s2.png"].position_mm == pytest.approx(2.5)
    assert by["s0.png"].position_mm == pytest.approx(2.0 - job.state.interval_mm)
    assert by["s4.png"].position_mm == pytest.approx(3.0 + job.state.interval_mm)
    assert {name for name, record in by.items() if record.position_source} == {
        "s0.png", "s2.png", "s4.png"}
    assert default_positions(job.state, ctx) == []  # nothing left without one


def test_without_positioning_no_starting_positions(tmp_path: Path):
    job, _ = _open(tmp_path, tasks=["transform"])
    assert all(record.position_mm is None for record in job.state.slices)


def test_saved_pictures_record_the_history_depth(tmp_path: Path):
    job, _ = _open(tmp_path, tasks=["position"])
    for position in (2.0, 3.0):  # two writes: the depth is not the store's default
        with job.writing():
            before = job.snapshot()
            job.state.by_id("s0.png").position_mm = position  # type: ignore[union-attr]
            job.commit(before)
    assert len(job.undo_stack) == 2
    picture = Image.new("RGB", (20, 10))
    job.views.save(tool="note", pictures=[(picture, None)])
    job.views.flush()
    latest = job.views.latest()
    assert latest is not None and latest.step == 2
    job.undo()
    job.views.save(tool="note", pictures=[(picture, None)])
    job.views.flush()
    latest = job.views.latest()
    assert latest is not None and latest.step == 1

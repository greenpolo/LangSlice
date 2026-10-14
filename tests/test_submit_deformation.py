"""The host's ``require_deformation`` at ``submit``; the
unplaced sections and the starting positions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from langslice.agent.engine import build_context
from langslice.core.spec import JobSpec, NonlinearSpec
from langslice.job.job import Job
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


def test_no_deformation_is_required_by_default(tmp_path: Path):
    job = _placed(tmp_path)
    done = ops_submit.submit(job, summary="done")
    assert done.summary == "done" and job.state.submitted
    assert all(record.deformation is None for record in job.state.slices)


def test_the_host_can_require_a_deformation_on_every_section(tmp_path: Path):
    job = _placed(tmp_path, require_deformation=True)
    refused = _refused(job)
    assert refused.code == "MISSING_DEFORMATIONS"
    assert [row["id"] for row in refused.payload()["sections"]] == ["s0.png", "s1.png", "s2.png"]
    assert "left_linear" not in refused.payload()["message"]
    assert not job.state.submitted
    spec = JobSpec.from_dict(job.spec.to_dict())
    assert spec.nonlinear.require_deformation is True
    assert "require_deformation" not in JobSpec(image_folder=".").to_dict()["nonlinear"]


def test_ingest_leaves_sections_unplaced_and_submit_refuses(tmp_path: Path):
    job, _ = _open(tmp_path, tasks=["position"])
    assert all(record.position_mm is None for record in job.state.slices)
    refused = _refused(job)
    assert refused.code == "MISSING_POSITIONS"
    assert refused.payload()["missing_ids"] == ["s0.png", "s1.png", "s2.png"]


def test_supplied_positions_do_not_place_other_sections(tmp_path: Path):
    job, _ = _open(tmp_path, n=5, tasks=["position"],
                   inputs={"positions": {"s1.png": 2.0, "s3.png": 3.0}})
    assert [record.position_mm for record in sorted(
        job.state.slices, key=lambda record: record.index_original)] == [None, 2.0, None, 3.0, None]


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

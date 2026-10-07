"""`ops.positions.position_sections`: positions and cutting angles in one undo step,
the stack numbered by position, supplied positions refused."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from langslice.agent.engine import build_context
from langslice.core.spec import JobSpec, TransformSpec
from langslice.job.job import Job
from langslice.ops import positions
from langslice.ops.refusal import Refused
from tests.fakes import SlabAtlas

_ATLAS = SlabAtlas()


def _open(tmp_path: Path, n: int = 4, **spec_kwargs: Any) -> tuple[Job, Any]:
    for index in range(n):
        Image.fromarray(np.full((30, 40, 3), 40 + 10 * index, dtype=np.uint8)).save(
            tmp_path / f"s{index}.png")
    spec = JobSpec(image_folder=str(tmp_path), model="fake-model", preprocess="none",
                   **spec_kwargs)
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: _ATLAS)
    job = Job.open(spec, ctx, folder=ctx.job_folder, results_path=ctx.results_path)
    return job, ctx


def _ids(job: Job) -> list[str]:
    return [record.id for record in job.state.in_order()]


def _position(job: Job, name: str) -> float | None:
    record = job.state.by_id(name)
    assert record is not None
    return record.position_mm


def test_positions_and_angles_are_one_undo_step(tmp_path: Path):
    job, ctx = _open(tmp_path)
    low, high = ctx.position_range
    middle = (low + high) / 2
    before_angles = dict(job.state.cutting_angles_deg)
    done = positions.position_sections(
        job, ctx,
        [{"id": "s0.png", "position_mm": middle}, {"id": "s1.png", "position_mm": high + 3}],
        {"pitch_deg": 2.0, "yaw_deg": -1.0})
    assert done.written == [("s0.png", middle), ("s1.png", high)]
    assert done.clamped == [("s1.png", high + 3, high)]
    assert done.cutting_angles == {"pitch": 2.0, "yaw": -1.0}
    assert job.state.cutting_angles_deg == {"pitch": 2.0, "yaw": -1.0}
    assert len(job.undo_stack) == 1
    assert job.undo()
    assert _position(job, "s0.png") is None
    assert job.state.cutting_angles_deg == before_angles


def test_angles_alone_and_nothing_to_write(tmp_path: Path):
    job, ctx = _open(tmp_path)
    done = positions.position_sections(job, ctx, [], {"pitch_deg": 1.0, "yaw_deg": 0.0})
    assert done.written == [] and done.cutting_angles == {"pitch": 1.0, "yaw": 0.0}
    assert len(job.undo_stack) == 1
    ghost = positions.position_sections(job, ctx, [{"id": "nope.png", "position_mm": 1.0}])
    assert ghost.unknown == ["nope.png"] and ghost.written == []
    assert len(job.undo_stack) == 1  # unknown ids write nothing
    with pytest.raises(Refused) as bad:
        positions.position_sections(job, ctx, [])
    assert bad.value.code == "BAD_ARGS"
    for entry in ({"id": "s0.png"}, {"id": "s0.png", "position_mm": float("nan")}, "s0.png"):
        with pytest.raises(Refused) as bad:
            positions.position_sections(job, ctx, [entry])
        assert bad.value.code == "BAD_ARGS"
    assert len(job.undo_stack) == 1


def test_order_follows_position(tmp_path: Path):
    job, ctx = _open(tmp_path)
    low, high = ctx.position_range
    step = (high - low) / 10
    done = positions.position_sections(job, ctx, [
        {"id": "s0.png", "position_mm": low + 3 * step},
        {"id": "s1.png", "position_mm": low + 1 * step},
        {"id": "s3.png", "position_mm": low + 2 * step},
    ])
    # s2 has no position and keeps its place (third); the placed sections fill the rest.
    assert _ids(job) == ["s1.png", "s3.png", "s2.png", "s0.png"]
    assert done.order == _ids(job)
    assert "s2.png" not in done.reordered and "s0.png" in done.reordered
    assert [record.index_corrected for record in job.state.in_order()] == [0, 1, 2, 3]
    # A later write renumbers again, in its own undo step.
    positions.position_sections(job, ctx, [{"id": "s1.png", "position_mm": low + 5 * step}])
    assert _ids(job) == ["s3.png", "s0.png", "s2.png", "s1.png"]
    assert len(job.undo_stack) == 2
    assert job.undo()
    assert _ids(job) == ["s1.png", "s3.png", "s2.png", "s0.png"]


def test_supplied_positions_are_refused_and_so_are_supplied_angles(tmp_path: Path):
    supplied = {"s0.png": 1.0, "s1.png": 2.0, "s2.png": 3.0, "s3.png": 4.0}
    job, ctx = _open(tmp_path, tasks=["transform"], inputs={"positions": supplied})
    with pytest.raises(Refused) as refused:
        positions.position_sections(job, ctx, [{"id": "s0.png", "position_mm": 5.0}])
    assert refused.value.code == "POSITIONS_SUPPLIED"
    with pytest.raises(Refused) as angles:
        positions.position_sections(job, ctx, [], {"pitch_deg": 1.0, "yaw_deg": 1.0})
    assert angles.value.code == "ANGLES_SUPPLIED"
    assert job.undo_stack == [] and _position(job, "s0.png") == 1.0


def test_supplied_positions_still_allow_angles_when_the_transform_task_sets_them(
        tmp_path: Path):
    supplied = {"s0.png": 1.0, "s1.png": 2.0, "s2.png": 3.0, "s3.png": 4.0}
    job, ctx = _open(tmp_path, tasks=["transform"], inputs={"positions": supplied},
                     transform=TransformSpec(angles=True))
    with pytest.raises(Refused) as refused:
        positions.position_sections(job, ctx, [{"id": "s0.png", "position_mm": 5.0}],
                                    {"pitch_deg": 1.0, "yaw_deg": 1.0})
    assert refused.value.code == "POSITIONS_SUPPLIED" and job.undo_stack == []
    done = positions.position_sections(job, ctx, [], {"pitch_deg": 1.0, "yaw_deg": 2.0})
    assert done.cutting_angles == {"pitch": 1.0, "yaw": 2.0} and done.written == []
    assert _position(job, "s0.png") == 1.0 and len(job.undo_stack) == 1

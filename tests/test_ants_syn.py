"""``ops.deformable.ants_syn``: one choice, applied, always building on the
section's current deformation; and its refusals."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from langslice.agent.engine import build_context
from langslice.core import deformation
from langslice.core.spec import JobSpec, NonlinearSpec
from langslice.job.job import Job
from langslice.ops import deformable as ops_deformable
from langslice.ops.refusal import Refused
from tests.deformable_synthetic import SMOOTH_FIELD, SyntheticAtlas, render_section

ID = "s0.png"


@pytest.fixture(autouse=True)
def _in_process(monkeypatch):
    monkeypatch.setattr(deformation, "USE_PROCESS_POOL", False)
    monkeypatch.setattr(deformation, "DETAIL_LEVEL", "coarse")


@pytest.fixture(scope="module")
def atlas() -> SyntheticAtlas:
    return SyntheticAtlas()


def _job(folder: Path, atlas: SyntheticAtlas) -> tuple[Job, Any]:
    image, _ = render_section(atlas, SMOOTH_FIELD())
    image.save(folder / ID)
    spec = JobSpec(image_folder=str(folder), model="fake-model", preprocess="none",
                   tasks=["nonlinear"], inputs={"pixel_size_um": 25.0},
                   nonlinear=NonlinearSpec(provider="none"))
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: atlas)
    job = Job.open(spec, ctx, folder=ctx.job_folder, results_path=ctx.results_path)
    with job.writing():
        before = job.snapshot()
        record = job.state.slices[0]
        record.position_mm = 0.1
        record.transform = {"kind": "interactive", "params": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            "calibration": {"section_um_per_px": 25.0, "source": "host"}}
        job.commit(before)
    return job, ctx


def test_each_call_applies_one_fit_on_top_of_the_last(tmp_path: Path, atlas):
    job, ctx = _job(tmp_path, atlas)
    depth = len(job.undo_stack)
    first = ops_deformable.ants_syn(job, ctx, [ID])
    (row,) = first.rows
    assert row["status"] == "ok" and row["written"]
    assert row["settings"] == {"engine": "ants", "stiffness": "medium", "fit_section": "fit",
                               "fit_atlas": "template"}
    assert len(job.undo_stack) == depth + 1
    held = job.state.slices[0].deformation
    assert held is not None and [step["start"] for step in held["steps"]] == ["linear"]

    second = ops_deformable.ants_syn(job, ctx, ["s0.png"], stiffness="soft", restrict_to=["TH"])
    assert second.rows[0]["status"] == "ok", second.rows
    held = job.state.slices[0].deformation
    assert held is not None
    assert [step["start"] for step in held["steps"]] == ["linear", "current"]
    assert held["steps"][1]["include"] == ["TH"] and held["steps"][1]["stiffness"] == "soft"
    assert len(job.undo_stack) == depth + 2
    assert job.undo()  # undo is how to start over
    assert len(job.state.slices[0].deformation["steps"]) == 1  # type: ignore[index]


def test_the_section_s_marked_regions_are_left_out(tmp_path: Path, atlas):
    job, ctx = _job(tmp_path, atlas)
    job.state.slices[0].damaged_regions = ["HY"]
    done = ops_deformable.ants_syn(job, ctx, [ID])
    assert done.rows[0]["status"] == "ok"
    step = job.state.slices[0].deformation["steps"][0]  # type: ignore[index]
    assert step["exclude"] == ["HY"]


@pytest.mark.parametrize("kwargs,code", [
    ({"sections": []}, "BAD_ARGS"),
    ({"sections": [ID] * 5}, "BAD_ARGS"),
    ({"sections": [ID], "stiffness": "stiff"}, "BAD_ARGS"),
    ({"sections": [ID], "atlas_image": "borders"}, "BAD_ARGS"),
    ({"sections": [ID], "atlas_image": "nissl"}, "FIT_ATLAS_UNAVAILABLE"),
    ({"sections": ["nope.png"]}, "UNKNOWN_SLICE_IDS"),
    ({"sections": [ID], "restrict_to": ["NOPE"]}, "UNKNOWN_REGIONS"),
    ({"sections": [ID], "restrict_to": "TH"}, "BAD_ARGS"),
])
def test_refusals_write_nothing(tmp_path: Path, atlas, kwargs, code):
    job, ctx = _job(tmp_path, atlas)
    depth = len(job.undo_stack)
    sections = kwargs.pop("sections")
    with pytest.raises(Refused) as refused:
        ops_deformable.ants_syn(job, ctx, sections, **kwargs)
    assert refused.value.code == code
    assert len(job.undo_stack) == depth and job.state.slices[0].deformation is None


def test_without_ants_it_refuses(tmp_path: Path, atlas, monkeypatch):
    job, ctx = _job(tmp_path, atlas)
    monkeypatch.setattr(deformation, "ants_available", lambda: False)
    with pytest.raises(Refused) as refused:
        ops_deformable.ants_syn(job, ctx, [ID])
    assert refused.value.code == "ANTS_MISSING"
    assert "antspyx" in refused.value.payload()["message"]


def test_the_old_atlas_name_is_accepted(tmp_path: Path, atlas):
    job, ctx = _job(tmp_path, atlas)
    done = ops_deformable.ants_syn(job, ctx, [ID], atlas_image="ara")
    assert done.rows[0]["settings"]["fit_atlas"] == "template"

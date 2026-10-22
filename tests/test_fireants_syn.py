"""``ops.deformable.fireants_syn``: one choice, applied, always building on the
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

pytest.importorskip("fireants", reason="fireants_syn needs FireANTs")

ID = "s0.png"


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
    first = ops_deformable.fireants_syn(job, ctx, [ID])
    (row,) = first.rows
    assert row["status"] == "ok" and row["written"]
    assert row["settings"] == {"smoothness_um": 80.0, "mask": "none",
                               "section_image": "preprocessed", "atlas_image": "nissl"}
    # The reply names what the fit read; the synthetic atlas has no Nissl, so
    # "nissl" is its template. FireANTs sorts nothing by itself any more.
    assert first.settings == {
        "section_image": "preprocessed", "atlas_image": "nissl",
        "atlas_image_resolved": "template", "mask": "none", "restrict_to": [],
        "smoothness_um": 80.0, "metric": "mutual_information", "edges": True,
        "detail": "standard"}
    assert row["inputs"]["section_image"]["name"] == "preprocessed"
    assert row["inputs"]["section_image"]["kind"] == "made"
    assert row["inputs"]["atlas_image_resolved"] == "template"
    assert row["inputs"]["mask"]["name"] == "none" and row["inputs"]["damage"]["side"] == "atlas"
    assert row["estimated"] is None
    assert len(job.undo_stack) == depth + 1
    held = job.state.slices[0].deformation
    assert held is not None and [step["start"] for step in held["steps"]] == ["linear"]

    second = ops_deformable.fireants_syn(job, ctx, ["s0.png"], smoothness_um=50.0,
                                         restrict_to=["TH"])
    assert second.rows[0]["status"] == "ok", second.rows
    held = job.state.slices[0].deformation
    assert held is not None
    assert [step["start"] for step in held["steps"]] == ["linear", "current"]
    assert held["steps"][1]["include"] == ["TH"]
    assert held["steps"][1]["smoothness_um"] == 50.0
    assert second.rows[0]["inputs"]["restrict_to"] == ["TH"]
    assert len(job.undo_stack) == depth + 2
    assert job.undo()  # undo is how to start over
    assert len(job.state.slices[0].deformation["steps"]) == 1  # type: ignore[index]


def test_the_section_s_marked_regions_are_left_out(tmp_path: Path, atlas):
    job, ctx = _job(tmp_path, atlas)
    job.state.slices[0].damaged_regions = ["HY"]
    done = ops_deformable.fireants_syn(job, ctx, [ID])
    assert done.rows[0]["status"] == "ok"
    step = job.state.slices[0].deformation["steps"][0]  # type: ignore[index]
    assert step["exclude"] == ["HY"]


@pytest.mark.parametrize("kwargs,code", [
    ({"sections": []}, "BAD_ARGS"),
    ({"sections": [ID], "smoothness_um": 1.0}, "BAD_ARGS"),
    ({"sections": [ID], "metric": "ssd"}, "BAD_ARGS"),
    ({"sections": [ID], "detail": "ultra"}, "BAD_ARGS"),
    ({"sections": [ID], "atlas_image": "borders"}, "BAD_ARGS"),
    ({"sections": [ID], "atlas_image": "no_such_image"}, "BAD_ARGS"),
    ({"sections": [ID], "section_image": "nope"}, "UNKNOWN_IMAGE"),
    ({"sections": [ID], "mask": "missing"}, "UNKNOWN_MASK"),
    ({"sections": ["nope.png"]}, "UNKNOWN_SLICE_IDS"),
    ({"sections": [ID], "restrict_to": ["NOPE"]}, "UNKNOWN_REGIONS"),
    ({"sections": [ID], "restrict_to": "TH"}, "BAD_ARGS"),
])
def test_refusals_write_nothing(tmp_path: Path, atlas, kwargs, code):
    job, ctx = _job(tmp_path, atlas)
    depth = len(job.undo_stack)
    sections = kwargs.pop("sections")
    with pytest.raises(Refused) as refused:
        ops_deformable.fireants_syn(job, ctx, sections, **kwargs)
    assert refused.value.code == code
    assert len(job.undo_stack) == depth and job.state.slices[0].deformation is None


def test_without_engine_it_refuses(tmp_path: Path, atlas, monkeypatch):
    job, ctx = _job(tmp_path, atlas)
    monkeypatch.setattr(deformation, "engine_available", lambda: False)
    with pytest.raises(Refused) as refused:
        ops_deformable.fireants_syn(job, ctx, [ID])
    assert refused.value.code == "ENGINE_MISSING"
    message = refused.value.payload()["message"]
    assert "torch" in message and "fireants" in message

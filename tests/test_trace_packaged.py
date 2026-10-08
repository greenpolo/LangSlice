"""The packaged ``trace_borders`` (``ops.traces``): the image call in the
background, then an ANTs fit of what it drew, applied as its own undo step.

The image model is the golden tests' stand-in (``tests/golden/record.py``
``stub_image_model``): the run's model with its network call replaced by one
that answers with the rough border overlay it was sent, so line extraction,
the artifacts and the traced fit all run on real data.
"""

from __future__ import annotations

import dataclasses
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from langslice.agent.engine import build_context
from langslice.core import deformation
from langslice.core.nonlinear.types import GeneratedSegmentation
from langslice.core.spec import JobSpec, NonlinearSpec
from langslice.job.job import Job
from langslice.ops import deformable as ops_deformable
from langslice.ops import traces
from langslice.ops.inputs import STALE_INPUT
from langslice.ops.refusal import Refused
from langslice.providers.registry import resolve_image_model
from tests.deformable_synthetic import SMOOTH_FIELD, SyntheticAtlas, render_section

ID = "s0.png"


@pytest.fixture(autouse=True)
def _in_process(monkeypatch):
    """Fits in this process at the coarse detail level, as the deformable tests run."""
    monkeypatch.setattr(deformation, "USE_PROCESS_POOL", False)
    monkeypatch.setattr(deformation, "DETAIL_LEVEL", "coarse")


@pytest.fixture(scope="module")
def atlas() -> SyntheticAtlas:
    return SyntheticAtlas()


class StubModel:
    """The run's image model, its call answered with the drawn borders (the
    attachment with more colour); *gate* holds the reply back until set."""

    def __init__(self, spec: JobSpec) -> None:
        self.calls: list[Any] = []
        self.gate = threading.Event()
        self.gate.set()
        resolved = resolve_image_model(spec.nonlinear.provider, spec.nonlinear.image_model)
        self.model = dataclasses.replace(resolved, call=self.generate)

    def generate(self, request: Any) -> GeneratedSegmentation:
        self.calls.append(request)
        assert self.gate.wait(10)
        images = [request.slice_image, *request.reference_images]

        def colour(image: Image.Image) -> float:
            pixels = np.asarray(image.convert("RGB")).astype(np.int16)
            return float(np.abs(pixels[..., 0] - pixels[..., 2]).mean())

        drawn = max(images, key=colour)
        return GeneratedSegmentation(image=drawn.convert("RGB"), provider=request.provider,
                                     model=str(request.model), route="test-stub")


def _job(folder: Path, atlas: SyntheticAtlas) -> tuple[Job, Any, StubModel]:
    image, _ = render_section(atlas, SMOOTH_FIELD())
    image.save(folder / ID)
    spec = JobSpec(image_folder=str(folder), model="fake-model", preprocess="none",
                   tasks=["nonlinear"], inputs={"pixel_size_um": 25.0},
                   nonlinear=NonlinearSpec(provider="openai-oauth"))
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: atlas)
    job = Job.open(spec, ctx, folder=ctx.job_folder, results_path=ctx.results_path)
    with job.writing():
        before = job.snapshot()
        record = job.state.slices[0]
        record.position_mm = 0.1
        record.transform = {"kind": "interactive", "params": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            "calibration": {"section_um_per_px": 25.0, "source": "host"}}
        job.commit(before)
    return job, ctx, StubModel(spec)


def test_the_trace_returns_at_once_and_its_fit_lands_as_one_undo_step(tmp_path, atlas):
    job, ctx, stub = _job(tmp_path, atlas)
    stub.gate.clear()
    depth = len(job.undo_stack)
    started = traces.trace_borders(job, ctx, ID, image_model=stub.model,
                                   prompt="Move each yellow line onto its edge.")
    assert started.work == "w1" and started.started
    assert started.record["status"] == "running"
    assert len(job.undo_stack) == depth + 1  # the running trace's record
    assert [work.id for work in job.background.running()] == ["w1"]
    # Asked again while it runs: the same work, nothing started.
    again = traces.trace_borders(job, ctx, ID, image_model=stub.model)
    assert again.running and again.work == "w1"
    stub.gate.set()
    job.background.wait_all()

    (work,) = job.background.notices()
    assert work.status == "done", work.notice
    assert work.notice.startswith("w1 trace_borders of s0.png finished: the trace landed "
                                  "and ANTs fitted the traced borders (medium stiffness) on "
                                  "top of its linear placement.")
    assert "undo removes it while it is the latest step" in work.notice
    record = job.state.by_id(ID)
    assert record is not None and record.image_correction["status"] == "ok"
    held = record.deformation
    assert held is not None and len(held["steps"]) == 1
    assert held["steps"][0]["fit_section"] == deformation.TRACED_BORDERS
    assert held["steps"][0]["engine"] == "ants" and held["steps"][0]["stiffness"] == "medium"
    assert len(job.undo_stack) == depth + 2  # the landing is its own step
    assert job.undo()
    record = job.state.by_id(ID)
    assert record is not None and record.deformation is None
    assert record.image_correction["status"] == "ok"  # the trace stays

    # The prompt sent is saved with the attempt in the job folder.
    folder = Path(job.folder) / str(record.image_correction["artifact_dir"])
    assert (folder / "prompt.txt").read_text() == "Move each yellow line onto its edge."
    assert len(stub.calls) == 1


def _landed_job(folder: Path, atlas: SyntheticAtlas) -> tuple[Job, Any, StubModel]:
    """A job whose section's packaged trace has landed (its fit applied)."""
    job, ctx, stub = _job(folder, atlas)
    traces.trace_borders(job, ctx, ID, image_model=stub.model)
    job.background.wait_all()
    (work,) = job.background.notices()
    assert work.status == "done", work.notice
    record = job.state.by_id(ID)
    assert record is not None and record.deformation is not None
    # The traced step records the trace it fitted.
    assert [step.get("trace") for step in record.deformation["steps"]] == [
        record.image_correction["artifact_dir"]]
    return job, ctx, stub


def test_a_trace_asked_again_after_its_fit_landed_fits_nothing(tmp_path, atlas):
    """The saved trace at this placement and region choice, whose fit is a step
    of the applied deformation: ``landed``, no image call, no new work."""
    job, ctx, stub = _landed_job(tmp_path, atlas)
    held = job.state.slices[0].deformation
    works = len(job.background.all())
    again = traces.trace_borders(job, ctx, ID, image_model=stub.model,
                                 prompt="A different prompt.")
    assert again.landed and again.work is None and not again.started and not again.running
    assert len(stub.calls) == 1  # the saved reply, no second image call
    assert len(job.background.all()) == works
    job.background.wait_all()
    assert job.background.notices() == []
    assert job.state.slices[0].deformation == held


def test_a_landed_trace_asked_again_takes_no_undo_step(tmp_path, atlas):
    """Nothing is fitted again, so nothing is written: undo still removes the fit.

    Fails on src as it stands: the re-call rewrites the section's
    ``image_correction`` (``cached`` False -> True) as an undo step before
    ``_landed`` is checked (``ops/traces.py`` ``trace_borders`` ->
    ``_start_trace`` commits), so the depth grows by one and the next undo
    reverts that flag instead of the fit.
    """
    job, ctx, stub = _landed_job(tmp_path, atlas)
    depth = len(job.undo_stack)
    again = traces.trace_borders(job, ctx, ID, image_model=stub.model)
    assert again.landed
    assert len(job.undo_stack) == depth
    assert job.undo()
    assert job.state.slices[0].deformation is None


def test_a_trace_builds_on_the_current_deformation_and_draws_its_pictures(tmp_path, atlas):
    from langslice.core.display import default_options

    job, ctx, stub = _job(tmp_path, atlas)
    fitted = ops_deformable.ants_syn(job, ctx, [ID])
    assert fitted.rows[0]["status"] == "ok", fitted.rows
    options = default_options("overlay")
    traces.trace_borders(job, ctx, ID, image_model=stub.model, options=options)
    job.background.wait_all()
    job.views.flush()
    (work,) = job.background.notices()
    assert work.status == "done", work.notice
    assert "on top of its previous deformation" in work.notice
    held = job.state.slices[0].deformation
    assert held is not None and len(held["steps"]) == 2
    # The fit and the trace, saved among the job's pictures.
    assert len(work.pictures) == 2 and len(work.images) == 2
    saved = [job.views.lookup(seq) for seq in work.pictures]
    assert all(entry is not None and entry.tool == "trace_borders" for entry in saved)


def test_restrict_to_traces_and_warps_that_region_only(tmp_path, atlas):
    job, ctx, stub = _job(tmp_path, atlas)
    started = traces.trace_borders(job, ctx, ID, image_model=stub.model, restrict_to=["TH"])
    job.background.wait_all()
    (work,) = job.background.notices()
    assert work.status == "done", work.notice
    assert "restricted to TH" in work.notice
    artifacts = Path(job.folder) / str(started.record["artifact_dir"])
    request = json.loads((artifacts / "request.json").read_text())
    assert request["include"] == ["TH"]
    step = job.state.slices[0].deformation["steps"][0]  # type: ignore[index]
    assert step["include"] == ["TH"]


def test_a_section_changed_while_the_model_draws_gets_nothing(tmp_path, atlas):
    job, ctx, stub = _job(tmp_path, atlas)
    stub.gate.clear()
    traces.trace_borders(job, ctx, ID, image_model=stub.model)
    with job.writing():  # the agent moves the section meanwhile
        before = job.snapshot()
        job.state.slices[0].position_mm = 0.15
        job.commit(before)
    depth = len(job.undo_stack)
    stub.gate.set()
    job.background.wait_all()
    (work,) = job.background.notices()
    assert work.status == "failed"
    assert f"{STALE_INPUT}: " in work.notice
    assert job.state.slices[0].deformation is None
    assert len(job.undo_stack) == depth


def test_without_ants_nothing_is_asked_of_the_image_model(tmp_path, atlas, monkeypatch):
    job, ctx, stub = _job(tmp_path, atlas)
    monkeypatch.setattr(ops_deformable, "ants_ready", lambda: False)
    depth = len(job.undo_stack)
    with pytest.raises(Refused) as refused:
        traces.trace_borders(job, ctx, ID, image_model=stub.model)
    assert refused.value.code == "ANTS_MISSING"
    assert "antspyx" in refused.value.payload()["message"]
    assert stub.calls == [] and len(job.undo_stack) == depth
    assert job.state.slices[0].image_correction is None
    with pytest.raises(Refused) as unknown:
        monkeypatch.setattr(ops_deformable, "ants_ready", lambda: True)
        traces.trace_borders(job, ctx, ID, image_model=stub.model, restrict_to=["NOPE"])
    assert unknown.value.code == "UNKNOWN_REGIONS" and stub.calls == []

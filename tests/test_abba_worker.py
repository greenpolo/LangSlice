"""External worker keeps host calibration and initial state intact without Java."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from langslice.doors.api.abba_worker import run_linear
from langslice.hosts.api.nonlinear_worker import run_nonlinear


@pytest.fixture
def params(tmp_path):
    Image.new("L", (80, 40)).save(tmp_path / "section_0001.tif")
    return {"image_folder": str(tmp_path), "pixel_size_um": 25,
            "positions_mm": {"section_0001.tif": 4.0},
            "spec": {"tasks": ["position", "transform"]}}


@pytest.mark.parametrize("initial_position", [4.0, -0.985])
def test_linear_initial_state_is_not_applied_and_affine_uses_snapshot_geometry(
    params, monkeypatch, initial_position,
):
    from langslice.agent import engine

    params["positions_mm"]["section_0001.tif"] = initial_position

    async def run(spec, *, on_write, on_event, emit):
        assert spec.inputs["positions"] == params["positions_mm"]
        assert spec.inputs["pixel_size_um"] == 25
        assert not spec.resume
        row = {"id": "section_0001.tif", "position_mm": initial_position, "rotation_deg": 0,
               "transform": None}
        value = {"slices": [row]}
        state = SimpleNamespace(to_dict=lambda: copy.deepcopy(value))
        on_write(state)
        row.update(position_mm=4.2, transform={"params": [1, 0, .1, 0, 1, .2]})
        on_write(state)
        on_event({"kind": "image", "images": [{"data": b"raw", "mime_type": "image/png"}],
                  "thought_signature": b"private"})
        return state

    monkeypatch.setattr(engine, "run", run)
    events = []
    result = run_linear(params, events.append)
    assert events[0]["initial"] and events[0]["host_updates"] == []
    update = events[1]["host_updates"][0]
    assert update["position_mm"] == 4.2
    np.testing.assert_allclose(update["affine_mm"],
                               [[1, 0, 0, .2], [0, 1, 0, .2], [0, 0, 1, 0]])
    assert "private" not in json.dumps(events)
    assert "raw" not in json.dumps(events)
    assert result["state"]["slices"][0]["position_mm"] == 4.2


@pytest.mark.parametrize("change, message", [
    ({"pixel_size_um": 0}, "positive"),
    ({"positions_mm": {"missing.tif": 4}}, "exactly"),
    ({"positions_mm": {"section_0001.tif": float("nan")}}, "finite"),
    ({"spec": {"plane": "sagittal"}}, "coronal"),
    ({"host_angles_deg": {"pitch": 4.0}}, "flat atlas"),
    ({"spec": {"transform": {"angles": True}}}, "Cutting-angle"),
    ({"registered_slices": ["missing.tif"]}, "Registered"),
    ({"locked": ["missing.tif"]}, "Locked"),
    ({"damaged": {"missing.tif": ""}}, "Damaged"),
    ({"damaged": {"section_0001.tif": 3}}, "Damaged"),
    ({"spec": {"image_resolution": "ultra"}}, "image_resolution"),
    ({"spec": {"transform": {"max_parallel": 5}}}, "max_parallel"),
    ({"spec": {"tasks": ["nonlinear"], "nonlinear": {"provider": "mystery"}}},
     "nonlinear.provider"),
])
def test_bad_host_inputs_refused_before_engine(params, change, message, monkeypatch):
    from langslice.agent import engine

    async def never(*args, **kwargs):
        raise AssertionError("the engine must not start")

    monkeypatch.setattr(engine, "run", never)
    params.update(change)
    with pytest.raises(ValueError, match=message):
        run_linear(params, lambda event: None)


def test_nonlinear_retains_pair_direction_and_grid(tmp_path, monkeypatch):
    import tifffile

    from langslice.hosts.integrations import abba

    coords = np.zeros((10, 12, 3), dtype=np.float32)
    np.save(tmp_path / "coords.npy", coords)
    tifffile.imwrite(tmp_path / "image.tif", np.ones((10, 12), dtype=np.uint8))
    source = np.array([[0., 0.], [1., 0.], [0., 1.]])
    target = source + 3

    def compute(actual_coords, histology, config):
        np.testing.assert_array_equal(actual_coords, coords)
        assert histology.shape == (10, 12)
        return source, target

    monkeypatch.setattr(abba, "compute_registration_landmarks", compute)
    result = run_nonlinear({"coords_path": str(tmp_path / "coords.npy"),
                            "histology_path": str(tmp_path / "image.tif")}, lambda event: None)
    assert result["source_points"] == source.tolist()
    assert result["target_points"] == target.tolist()
    assert result["coordinate_frame"] == "fixed_grid_pixels"


def test_trace_dir_saves_this_runs_trace_and_names_it(params, monkeypatch, tmp_path):
    import os

    from langslice.agent import engine
    from langslice.agent.trace import TRACE_DIR_ENV

    traces = tmp_path / "traces" / "nested"
    monkeypatch.delenv(TRACE_DIR_ENV, raising=False)

    async def run(spec, *, on_write, on_event, emit):
        # The engine's session opens its trace from the environment.
        folder = Path(os.environ[TRACE_DIR_ENV])
        (folder / "linear_stack_abcd1234.jsonl").write_text("{}\n")
        value = {"slices": [{"id": "section_0001.tif", "position_mm": 4.0,
                             "rotation_deg": 0, "transform": None}]}
        state = SimpleNamespace(to_dict=lambda: copy.deepcopy(value))
        on_write(state)
        return state

    monkeypatch.setattr(engine, "run", run)
    (tmp_path / "traces" / "nested").mkdir(parents=True)
    (traces / "older_run.jsonl").write_text("{}\n")
    result = run_linear({**params, "trace_dir": str(traces)}, lambda event: None)
    assert result["trace_files"] == [str(traces.resolve() / "linear_stack_abcd1234.jsonl")]
    assert TRACE_DIR_ENV not in os.environ


def test_no_trace_dir_means_no_trace(params, monkeypatch):
    import os

    from langslice.agent import engine
    from langslice.agent.trace import TRACE_DIR_ENV

    monkeypatch.delenv(TRACE_DIR_ENV, raising=False)

    async def run(spec, *, on_write, on_event, emit):
        assert TRACE_DIR_ENV not in os.environ
        value = {"slices": [{"id": "section_0001.tif", "position_mm": 4.0,
                             "rotation_deg": 0, "transform": None}]}
        state = SimpleNamespace(to_dict=lambda: copy.deepcopy(value))
        on_write(state)
        return state

    monkeypatch.setattr(engine, "run", run)
    assert "trace_files" not in run_linear(params, lambda event: None)


@pytest.mark.parametrize(("tasks", "sent"), [(["transform"], True), (["reorder"], False)])
def test_orientation_reaches_the_host_with_the_transform_task(tasks, sent):
    from langslice.doors.api.abba_worker import _host_updates

    before = {"slices": [{"id": "a.tif", "position_mm": 4.0, "index_corrected": 0,
                          "flip": False, "rotation_deg": 0, "transform": None}]}
    after = copy.deepcopy(before)
    after["slices"][0].update(flip=True, rotation_deg=90)
    updates = _host_updates(after, before, tasks, {"a.tif": (100, 80)}, 10.0)
    assert bool(updates) is sent
    if sent:
        assert updates[0]["flip"] is True and updates[0]["rotation_deg"] == 90


def test_nonlinear_without_an_image_model_is_accepted(params):
    from langslice.doors.api.abba_worker import prepare_linear

    params["spec"] = {"tasks": ["transform", "nonlinear"], "nonlinear": {"provider": "none"}}
    spec = prepare_linear(params).spec
    assert spec.has("nonlinear") and spec.nonlinear.uses_image_model is False


@pytest.mark.parametrize("where", ["job_dir", "read_only"])
def test_output_dir_is_the_job_folder_the_run_used(params, monkeypatch, tmp_path, where):
    """The result's output_dir is the run's actual job folder: the spec's
    job_dir, or the home fallback of a read-only snapshot folder (review
    finding 12)."""
    import os

    from langslice.agent import engine
    from langslice.job import index

    if where == "read_only" and hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root writes into read-only folders")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    async def run(spec, *, on_write, on_event, emit):
        value = {"slices": [{"id": "section_0001.tif", "position_mm": 4.0,
                             "rotation_deg": 0, "transform": None}]}
        state = SimpleNamespace(to_dict=lambda: copy.deepcopy(value))
        on_write(state)
        return state

    monkeypatch.setattr(engine, "run", run)
    images = Path(params["image_folder"])
    if where == "job_dir":
        expected = tmp_path / "elsewhere" / "job"
        params = {**params, "spec": {**params["spec"], "job_dir": str(expected)}}
        result = run_linear(params, lambda event: None)
    else:
        expected = tmp_path / "home" / ".langslice" / "jobs" / index.folder_id(images)
        images.chmod(0o555)
        try:
            result = run_linear(params, lambda event: None)
        finally:
            images.chmod(0o755)
    assert Path(result["output_dir"]) == expected


# --- one atlas angle for the whole stack --------------------------------------------


def _angled_state(*angles):
    from langslice.core.state import SliceState, StackState

    slices = [SliceState(id=f"s{i}.tif", index_original=i, index_corrected=i)
              for i in range(len(angles))]
    for record, (pitch, yaw) in zip(slices, angles, strict=True):
        record.cutting_angles_deg = {"pitch": pitch, "yaw": yaw}
    return StackState(slices=slices)


def test_a_checkpoint_with_an_angle_per_section_is_refused_before_abba_sees_it(params):
    from langslice.doors.api.abba_worker import (
        ABBA_MIXED_ANGLES,
        checkpoint_callback,
        prepare_linear,
    )

    events = []
    checkpoint = checkpoint_callback(prepare_linear(params), events.append)
    checkpoint(_angled_state((0.0, 0.0), (0.0, 0.0)))  # one angle: as before
    assert len(events) == 1 and events[0]["initial"]
    with pytest.raises(ValueError, match="cannot show") as refused:
        checkpoint(_angled_state((0.0, 0.0), (2.0, 0.0)))
    assert str(refused.value) == ABBA_MIXED_ANGLES
    assert len(events) == 1  # nothing emitted to the host


def test_a_run_ending_with_an_angle_per_section_reports_the_refusal(params, monkeypatch):
    from langslice.agent import engine

    async def run(spec, *, on_write, on_event, emit):
        return _angled_state((1.0, 0.0), (0.0, 0.0))

    monkeypatch.setattr(engine, "run", run)
    with pytest.raises(ValueError, match="cannot show"):
        run_linear(params, [].append)


def test_a_spec_with_differing_section_angles_is_refused_for_abba(tmp_path):
    from langslice.core.spec import JobSpec
    from langslice.doors.api.abba_worker import refuse_mixed_job

    def spec(angles, resume=False):
        return JobSpec(image_folder=str(tmp_path), resume=resume, inputs={"angles": angles})

    refuse_mixed_job(spec({"pitch": 2.0, "yaw": 0.0}))  # stack-wide: one angle
    refuse_mixed_job(spec({"a.tif": {"pitch": 1.0}, "b.tif": {"pitch": 1.0}}))
    with pytest.raises(ValueError, match="cannot show"):
        refuse_mixed_job(spec({"a.tif": {"pitch": 1.0}, "b.tif": {"pitch": 2.0}}))


def test_resuming_a_saved_job_with_an_angle_per_section_is_refused_for_abba(tmp_path):
    from langslice.core.spec import JobSpec
    from langslice.doors.api.abba_worker import refuse_mixed_job
    from langslice.job.checkpoint import write_checkpoint
    from langslice.job.layout import JobLayout

    layout = JobLayout.for_images(tmp_path)
    layout.folder.mkdir()
    write_checkpoint(_angled_state((0.0, 0.0), (1.5, 0.0)), str(layout.state_file))
    with pytest.raises(ValueError, match="cannot show"):
        refuse_mixed_job(JobSpec(image_folder=str(tmp_path), resume=True))
    refuse_mixed_job(JobSpec(image_folder=str(tmp_path), resume=False))  # a fresh job
    write_checkpoint(_angled_state((1.5, 0.0), (1.5, 0.0)), str(layout.state_file))
    refuse_mixed_job(JobSpec(image_folder=str(tmp_path), resume=True))

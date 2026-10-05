"""External worker keeps host calibration and initial state intact without Java."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from langslice.doors.api.abba_worker import run_linear


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

    async def run(spec, *, on_write, on_event, emit, **_):
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
    ({"angles_deg": {"pitch_deg": float("nan")}}, "finite"),
    ({"angles_deg": {"pitch": 4.0}}, "pitch_deg"),
    ({"z_offset_mm": float("inf")}, "z_offset_mm"),
    ({"existing_warp": ["missing.tif"]}, "existing_warp"),
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


def test_trace_dir_saves_this_runs_trace_and_names_it(params, monkeypatch, tmp_path):
    import os

    from langslice.agent import engine
    from langslice.agent.trace import TRACE_DIR_ENV

    traces = tmp_path / "traces" / "nested"
    monkeypatch.delenv(TRACE_DIR_ENV, raising=False)

    async def run(spec, *, on_write, on_event, emit, **_):
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

    async def run(spec, *, on_write, on_event, emit, **_):
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
    from langslice.job import index, layout

    if where == "read_only" and hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root writes into read-only folders")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(index, "default_root", lambda: tmp_path / "home" / ".langslice" / "jobs")

    async def run(spec, *, on_write, on_event, emit, **_):
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
        if os.name == "nt":
            original = layout.writable
            monkeypatch.setattr(layout, "writable", lambda target, source:
                                False if source == images else original(target, source))
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

    async def run(spec, *, on_write, on_event, emit, **_):
        return _angled_state((1.0, 0.0), (0.0, 0.0))

    monkeypatch.setattr(engine, "run", run)
    with pytest.raises(ValueError, match="cannot show"):
        run_linear(params, [].append)


# --- ABBA 0.24 contract: angles, z offset, existing warps, warp rows -------------------


def test_abba_angles_become_the_stack_wide_input_and_angle_tasks_are_allowed(params):
    from langslice.doors.api.abba_worker import prepare_linear

    params.update(angles_deg={"pitch_deg": 2.5, "yaw_deg": -1.0}, z_offset_mm=5.7)
    params["spec"] = {"tasks": ["position", "transform"], "transform": {"angles": True}}
    prepared = prepare_linear(params)
    assert prepared.spec.inputs["angles"] == {"pitch": 2.5, "yaw": -1.0}
    assert prepared.spec.transform.angles
    assert prepared.abba["z_offset_mm"] == 5.7
    assert prepared.abba["angles_deg"] == {"pitch_deg": 2.5, "yaw_deg": -1.0}


@pytest.mark.parametrize("locked, kept", [(["section_0001.tif"], True), ([], False)])
def test_existing_warp_is_kept_unless_the_user_allows_overwriting(params, locked, kept):
    """The dialog's overwrite option reaches the worker as `locked`: a locked
    section with the user's warp keeps it (inputs.keep_warp); an unlocked one
    (overwrite on) may be deformed."""
    from langslice.doors.api.abba_worker import prepare_linear

    params.update(locked=locked, existing_warp=["section_0001.tif"])
    params["spec"] = {"tasks": ["transform", "nonlinear"], "nonlinear": {"provider": "none"}}
    prepared = prepare_linear(params)
    assert ("keep_warp" in prepared.spec.inputs) is kept
    if kept:
        assert prepared.spec.inputs["keep_warp"] == ["section_0001.tif"]
    assert prepared.abba["existing_warp"] == ["section_0001.tif"]


def test_host_angles_are_sent_when_the_stack_wide_angles_change(tmp_path):
    from langslice.doors.api.abba_worker import checkpoint_callback, prepare_linear

    for name in ("s0.tif", "s1.tif"):
        Image.new("L", (80, 40)).save(tmp_path / name)
    events = []
    prepared = prepare_linear({"image_folder": str(tmp_path), "pixel_size_um": 25,
                               "positions_mm": {"s0.tif": 4.0, "s1.tif": 4.1},
                               "spec": {"tasks": ["position", "transform"]}})
    checkpoint = checkpoint_callback(prepared, events.append)
    checkpoint(_angled_state((0.0, 0.0), (0.0, 0.0)))
    checkpoint(_angled_state((0.0, 0.0), (0.0, 0.0)))
    checkpoint(_angled_state((2.0, -1.5), (2.0, -1.5)))
    checkpoint(_angled_state((2.0, -1.5), (2.0, -1.5)))
    assert "host_angles" not in events[0] and "host_angles" not in events[1]
    assert events[2]["host_angles"] == {"pitch_deg": 2.0, "yaw_deg": -1.5}
    assert "host_angles" not in events[3]
    assert prepared.final_angles == {"pitch_deg": 2.0, "yaw_deg": -1.5}


def test_the_abba_session_facts_are_kept_with_the_job(params, monkeypatch, tmp_path):
    from langslice.agent import engine
    from langslice.job.layout import JobLayout, read_job_file

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    params.update(z_offset_mm=5.7, angles_deg={"pitch_deg": 1.0, "yaw_deg": 0.0})

    async def run(spec, *, on_write, on_event, emit, on_open, **_):
        layout = JobLayout(Path(params["image_folder"]) / "langslice",
                           Path(params["image_folder"]))
        layout.ensure()
        on_open(SimpleNamespace(layout=layout, persist=True), None)
        value = {"slices": [{"id": "section_0001.tif", "position_mm": 4.0,
                             "rotation_deg": 0, "transform": None}]}
        state = SimpleNamespace(to_dict=lambda: copy.deepcopy(value))
        on_write(state)
        return state

    monkeypatch.setattr(engine, "run", run)
    result = run_linear(params, lambda event: None)
    assert result["abba"]["z_offset_mm"] == 5.7
    held = read_job_file(JobLayout(Path(params["image_folder"]) / "langslice"))
    assert held["host"]["abba"]["z_offset_mm"] == 5.7
    assert held["host"]["abba"]["angles_deg"] == {"pitch_deg": 1.0, "yaw_deg": 0.0}


@pytest.fixture(scope="module")
def warped(tmp_path_factory):
    """A real job (the golden recorder's synthetic stack) placed through the
    library, with the worker's checkpoint tracker attached to it."""
    import dataclasses

    import langslice
    from langslice.core.spec import NonlinearSpec
    from langslice.doors.api.abba_worker import PreparedLinear, checkpoint_callback
    from langslice.doors.jobs import create
    from tests.golden.record import (
        ID0,
        ID1,
        PIXEL_SIZE_UM,
        apply_patches,
        atlas_loader,
        full_spec,
        write_sections,
    )

    root = tmp_path_factory.mktemp("warp_rows")
    patch = pytest.MonkeyPatch()
    patch.setenv("HOME", str(root / "home"))
    apply_patches()
    folder = root / "stack"
    write_sections(folder)
    spec = dataclasses.replace(full_spec(folder), nonlinear=NonlinearSpec(provider="none"),
                               agent_preprocessing=False)
    create(spec, atlas_loader=atlas_loader()).close()
    job = langslice.open_job(str(folder), atlas_loader=atlas_loader())
    geometry = {}
    for name in (ID0, ID1):
        with Image.open(folder / name) as image:
            geometry[name] = image.size
    others = [p.name for p in folder.iterdir() if p.suffix == ".png" and p.name not in geometry]
    for name in others:
        with Image.open(folder / name) as image:
            geometry[name] = image.size
    prepared = PreparedLinear(folder, job.job.spec, geometry, PIXEL_SIZE_UM, frozenset(),
                              [], None, {})
    events: list = []
    tracker = checkpoint_callback(prepared, events.append)
    tracker.attach(job.job, job.workspace)
    job.set_positions(entries=[{"id": ID0, "position_mm": 0.1},
                               {"id": ID1, "position_mm": 0.15}])
    job.orient_slices(entries=[{"id": ID1, "flip": True, "rotate_deg": 90}])
    job.adjust_transforms(entries=[
        {"id": ID0, "rotation_deg": 3.0, "scale_x": 1.05, "scale_y": 0.97,
         "translate_x_mm": 0.1, "translate_y_mm": -0.05, "shear": 0.04},
        {"id": ID1, "rotation_deg": -4.0, "scale_x": 1.0, "scale_y": 1.0,
         "translate_x_mm": 0.0, "translate_y_mm": 0.02}])
    tracker(job.state)
    yield job, tracker, events, prepared
    job.close()
    patch.undo()


def _row(event, name):
    return next((row for row in event["host_updates"] if row["id"] == name), None)


def test_an_applied_deformation_lands_as_warp_pairs_on_the_affine(warped):
    from langslice.core.abba_warp import world_frame
    from langslice.core.maps import native_points, section_frame, stored_params
    from langslice.core.thin_plate import ThinPlateKernel
    from tests.golden.record import ID0, ID1

    job, tracker, events, prepared = warped
    assert job.fit_deformable(slices=[ID0], engine="elastix")["status"] == "ok"
    tracker(job.state)
    row = _row(events[-1], ID0)
    warp = row["warp"]
    assert set(warp) == {"source_mm", "target_mm", "record", "max_error_mm", "p99_error_mm",
                         "points"}
    assert 0 < warp["p99_error_mm"] <= warp["max_error_mm"]
    logged = [e["message"] for e in events if e.get("kind") == "log" and ID0 in e["message"]]
    assert any("deformation sent to ABBA" in m and "landmarks" in m for m in logged)
    source = np.asarray(warp["source_mm"]).T
    target = np.asarray(warp["target_mm"]).T
    assert len(warp["source_mm"]) == 2 and source.shape == (warp["points"], 2)
    assert warp["max_error_mm"] <= 0.005
    assert _row(events[-1], ID1) is None  # nothing changed there
    # ABBA's pull-back reproduces the record's own map at points off the grid.
    record = job.state.by_id(ID0)
    frame = section_frame(job.state, job.workspace, record)
    deformation = job.job.deformations.current(job.state, record)
    to_world, native_to_world = world_frame(frame, stored_params(record),
                                            size=prepared.geometry[ID0],
                                            pixel_size_um=prepared.calibration)
    rng = np.random.default_rng(3)
    width, height = prepared.geometry[ID0]
    points = rng.uniform([0, 0], [width - 1, height - 1], size=(500, 2))
    nx, ny = native_points(frame, deformation, points[None, :, 0], points[None, :, 1])
    homogeneous = np.column_stack([points, np.ones(len(points))])
    before = (to_world @ frame.file_to_render @ homogeneous.T).T[:, :2]
    after = (native_to_world @ np.stack([nx[0], ny[0], np.ones(len(points))])).T[:, :2]
    pullback = ThinPlateKernel(target, source, max_points=33 ** 2)
    assert np.linalg.norm(pullback.forward(after) - before, axis=1).max() <= 0.005


def test_keep_linear_and_a_moved_placement_remove_the_warp_step(warped):
    from tests.golden.record import ID0

    job, tracker, events, _prepared = warped
    if job.state.by_id(ID0).deformation is None:
        assert job.fit_deformable(slices=[ID0], engine="elastix")["status"] == "ok"
        tracker(job.state)
    job.fit_deformable(slices=[ID0], keep_linear="kept for the test")
    tracker(job.state)
    assert _row(events[-1], ID0)["warp"] is None
    assert job.fit_deformable(slices=[ID0], engine="elastix")["status"] == "ok"
    tracker(job.state)
    assert _row(events[-1], ID0)["warp"] is not None
    job.set_positions(entries=[{"id": ID0, "position_mm": 0.11}])
    tracker(job.state)
    moved = _row(events[-1], ID0)
    assert moved["position_mm"] == pytest.approx(0.11) and moved["warp"] is None
    # Since the start, the section has no LangSlice warp either.
    assert _row({"host_updates": events[-1]["updates_since_start"]}, ID0)["warp"] is None


def test_a_section_keeping_the_users_warp_is_refused_by_fit_deformable(tmp_path, monkeypatch):
    import dataclasses

    import langslice
    from langslice.core.spec import NonlinearSpec
    from langslice.doors.jobs import create
    from tests.golden.record import (
        ID0,
        PIXEL_SIZE_UM,
        apply_patches,
        atlas_loader,
        full_spec,
        write_sections,
    )

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    apply_patches()
    folder = tmp_path / "stack"
    write_sections(folder)
    spec = dataclasses.replace(
        full_spec(folder), nonlinear=NonlinearSpec(provider="none"), agent_preprocessing=False,
        inputs={"pixel_size_um": PIXEL_SIZE_UM, "keep_warp": [ID0]})
    create(spec, atlas_loader=atlas_loader()).close()
    job = langslice.open_job(str(folder), atlas_loader=atlas_loader())
    try:
        job.set_positions(entries=[{"id": ID0, "position_mm": 0.1}])
        result = job.fit_deformable(slices=[ID0], engine="elastix")
        assert "KEEPS_HOST_WARP" in json.dumps(result)
        assert job.state.by_id(ID0).deformation is None
    finally:
        job.close()


def test_the_seed_event_carries_the_saved_opening_views():
    from langslice.agent.engine import with_seed_views

    got: list = []
    forward = with_seed_views(got.append, ["/job/views/000001_opening/view.jpg"])
    assert forward is not None
    forward({"kind": "seed", "text": "", "images": []})
    forward({"kind": "text", "text": "hello"})
    assert got[0]["views"] == ["/job/views/000001_opening/view.jpg"]
    assert "views" not in got[1]
    assert with_seed_views(None, ["x"]) is None


def test_sections_left_out_of_nonlinear_are_refused_by_its_tools(tmp_path, monkeypatch):
    """`nonlinear_skip` (the dialog's "do not align these first" with
    Positioning on) reaches the job as inputs.nonlinear_skip; the fit and
    the image model's trace refuse those sections."""
    import dataclasses

    import langslice
    from langslice.core.spec import NonlinearSpec
    from langslice.doors.api.abba_worker import prepare_linear
    from langslice.doors.jobs import create
    from tests.golden.record import (
        ID0,
        PIXEL_SIZE_UM,
        apply_patches,
        atlas_loader,
        full_spec,
        write_sections,
    )

    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    Image.new("L", (80, 40)).save(snapshots / "a.tif")
    prepared = prepare_linear({"image_folder": str(snapshots), "pixel_size_um": 25,
                               "positions_mm": {"a.tif": 4.0}, "nonlinear_skip": ["a.tif"],
                               "spec": {"tasks": ["position", "nonlinear"],
                                        "nonlinear": {"provider": "none"}}})
    assert prepared.spec.inputs["nonlinear_skip"] == ["a.tif"]
    with pytest.raises(ValueError, match="nonlinear_skip"):
        prepare_linear({"image_folder": str(snapshots), "pixel_size_um": 25,
                        "positions_mm": {"a.tif": 4.0}, "nonlinear_skip": ["b.tif"]})

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    apply_patches()
    folder = tmp_path / "stack"
    write_sections(folder)
    spec = dataclasses.replace(
        full_spec(folder), nonlinear=NonlinearSpec(provider="none"), agent_preprocessing=False,
        inputs={"pixel_size_um": PIXEL_SIZE_UM, "nonlinear_skip": [ID0]})
    create(spec, atlas_loader=atlas_loader()).close()
    job = langslice.open_job(str(folder), atlas_loader=atlas_loader())
    try:
        job.set_positions(entries=[{"id": ID0, "position_mm": 0.1}])
        assert "NONLINEAR_SKIPPED" in json.dumps(job.fit_deformable(slices=[ID0],
                                                                    engine="elastix"))
        assert job.job.nonlinear_refusal(ID0)[0] == "NONLINEAR_SKIPPED"
    finally:
        job.close()


def test_the_run_ends_with_a_checkpoint_of_the_final_state(params, monkeypatch):
    """The connector applies the end-of-run checkpoint's rows (not
    final_updates), so the worker always sends one after the engine."""
    from langslice.agent import engine

    async def run(spec, *, on_write, on_event, emit, **_):
        row = {"id": "section_0001.tif", "position_mm": 4.0, "rotation_deg": 0,
               "transform": None}
        value = {"slices": [row]}
        on_write(SimpleNamespace(to_dict=lambda: copy.deepcopy(value)))
        row["position_mm"] = 4.3
        return SimpleNamespace(to_dict=lambda: copy.deepcopy(value))

    monkeypatch.setattr(engine, "run", run)
    events: list = []
    result = run_linear(params, events.append)
    last = [event for event in events if event["kind"] == "checkpoint"][-1]
    assert last["host_updates"] == [{"id": "section_0001.tif", "position_mm": 4.3}]
    assert result["final_updates"] == last["updates_since_start"]

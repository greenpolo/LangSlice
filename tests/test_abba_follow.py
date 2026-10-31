"""``abba.follow``: a prepared job's checkpoints streamed to ABBA as rows.

The follower is driven here one look at a time (:class:`Ticks` stands in for
the stop event the service gives it), so what lands between two looks is
known exactly. The checkpoint tracker's own rules (frozen properties, one
cutting angle, warp rows) are tested on it directly.

The warp-row tests (real deformation fits on the synthetic stack) are `slow`."""

from __future__ import annotations

import copy
import io
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import langslice
from langslice.core.stack.codec import deformation_to_json
from langslice.doors.abba.checkpoints import (
    ABBA_MIXED_ANGLES,
    WARP_CACHE_DIR,
    HostCheckpoints,
    HostFacts,
    _host_updates,
    baseline_rows,
    conversion_digest,
)
from langslice.doors.abba.contract import AbbaFollowRequest, AbbaPrepareRequest
from langslice.doors.abba.follow import follow
from langslice.doors.abba.prepare import prepare_job
from langslice.doors.abba.warp_workers import (
    THREAD_VARIABLES,
    started_with_threads,
)
from langslice.hosts.service import run_stdio
from tests.fakes import SlabAtlas

NAMES = ("s0.png", "s1.png", "s2.png")
_ATLAS = SlabAtlas()


class Ticks:
    """The follower's stop event, run by the test: each :meth:`look` lets the
    follower make exactly one more look at the job and waits until it has."""

    def __init__(self) -> None:
        self.stopped = False
        self.looks = 0
        self.turn = threading.Semaphore(0)
        self.changed = threading.Condition()

    def wait(self, timeout: float | None = None) -> bool:
        with self.changed:
            self.looks += 1
            self.changed.notify_all()
        self.turn.acquire()
        return self.stopped

    def is_set(self) -> bool:
        return self.stopped

    def ready(self, looks: int = 1) -> None:
        with self.changed:
            assert self.changed.wait_for(lambda: self.looks >= looks, timeout=30)

    def look(self) -> None:
        looks = self.looks
        self.turn.release()
        self.ready(looks + 1)

    def stop(self) -> None:
        self.stopped = True
        self.turn.release()


def _prepared(root: Path, **change: Any) -> Path:
    folder = root / "snapshots"
    folder.mkdir()
    for index, name in enumerate(NAMES):
        Image.fromarray(np.full((30, 40, 3), 60 + 30 * index, dtype=np.uint8)).save(
            folder / name)
    params: dict[str, Any] = {
        "image_folder": str(folder), "pixel_size_um": 25,
        "positions_mm": {"s0.png": 1.0, "s1.png": 2.0, "s2.png": 3.0},
        "angles_deg": {"pitch_deg": 0.0, "yaw_deg": 0.0},
        "registered": ["s2.png"],
        "abba": {"project_key": "0123456789ab", "atlas": "Allen Mouse V3p1",
                 "sections": {name: name.upper() for name in NAMES}},
        **change,
    }
    return prepare_job(AbbaPrepareRequest.model_validate(params))


def _init(job_folder: Path, *flags: str) -> None:
    from langslice.doors.agent_cli.calls import init

    envelope = init(str(job_folder.parent), list(flags), atlas_loader=lambda _n: _ATLAS)
    assert envelope.ok, envelope


class Following:
    """A follower on a thread, its events kept."""

    def __init__(self, job_folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from langslice.doors import jobs

        real = jobs.open_folder
        monkeypatch.setattr("langslice.doors.abba.follow.open_folder",
                            lambda folder, **kw: real(folder, atlas_loader=lambda _n: _ATLAS,
                                                      **kw))
        self.events: list[dict[str, Any]] = []
        self.ticks = Ticks()
        self.errors: list[BaseException] = []
        request = AbbaFollowRequest(job_folder=str(job_folder))
        self.thread = threading.Thread(target=self._run, args=(request,), daemon=True)
        self.thread.start()
        self.ticks.ready()

    def _run(self, request: AbbaFollowRequest) -> None:
        try:
            follow(request, self.events.append, self.ticks)  # type: ignore[arg-type]
        except BaseException as exc:  # noqa: BLE001 - the test reports it
            self.errors.append(exc)

    @property
    def checkpoints(self) -> list[dict[str, Any]]:
        return [event for event in self.events if event["kind"] == "checkpoint"]

    def close(self) -> None:
        self.ticks.stop()
        self.thread.join(30)
        assert not self.thread.is_alive() and not self.errors, self.errors


def _rows(checkpoint: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in checkpoint["host_updates"]}


def test_a_prepared_job_is_followed_from_its_first_opening(tmp_path, monkeypatch):
    job_folder = _prepared(tmp_path)
    following = Following(job_folder, monkeypatch)
    try:
        # prepared, not opened: the follower says so once and keeps looking
        following.ticks.look()
        assert following.events == [{"kind": "waiting"}]

        _init(job_folder)
        following.ticks.look()
        [first] = following.checkpoints
        # the whole state against ABBA with this job's steps removed
        assert first["initial"] is False and "host_angles" not in first
        rows = _rows(first)
        assert rows["s0.png"] == {"id": "s0.png", "position_mm": 1.0}
        assert rows["s1.png"] == {"id": "s1.png", "position_mm": 2.0}
        # a registered slice: its in-plane registration is ABBA's, no row for it
        assert rows["s2.png"] == {"id": "s2.png", "position_mm": 3.0}

        # nothing changed: no checkpoint, and once that everything up to
        # this state was sent
        following.ticks.look()
        assert len(following.checkpoints) == 1
        [idle] = [event for event in following.events if event["kind"] == "idle"]
        state = job_folder / "state.json"
        assert idle == {"kind": "idle", "state_mtime_ns": state.stat().st_mtime_ns}
        following.ticks.look()
        assert following.events[-1] == idle

        # several writes between two looks are one checkpoint
        job = langslice.open_job(str(job_folder), atlas_loader=lambda _n: _ATLAS)
        try:
            job.position_sections(sections=[{"id": "s0.png", "position_mm": 1.5}],
                                  view=False)
            job.interactive_transform(sections=[{"id": "s1.png", "rotation_deg": 3.0}],
                                      view=False)
            following.ticks.look()
            assert len(following.checkpoints) == 2
            rows = _rows(following.checkpoints[-1])
            assert rows["s0.png"] == {"id": "s0.png", "position_mm": 1.5}
            assert set(rows["s1.png"]) == {"id", "affine_mm"}
            assert "s2.png" not in rows

            # an undo is a forward step putting the section back
            job.undo()
            following.ticks.look()
            rows = _rows(following.checkpoints[-1])
            assert rows == {"s1.png": {"id": "s1.png", "affine_mm": None}}
        finally:
            job.close()
    finally:
        following.close()


def test_the_angles_are_sent_only_where_they_differ_from_abbas(tmp_path, monkeypatch):
    job_folder = _prepared(tmp_path, angles_deg={"pitch_deg": 2.0, "yaw_deg": -1.0})
    _init(job_folder)
    following = Following(job_folder, monkeypatch)
    try:
        [first] = following.checkpoints
        assert "host_angles" not in first
    finally:
        following.close()

    # the job's angles moved while ABBA was closed: the attach sends them
    job = langslice.open_job(str(job_folder), atlas_loader=lambda _n: _ATLAS)
    try:
        assert job.position_sections(
            sections=[{"id": "s0.png", "position_mm": 1.0}],
            cutting_angles={"pitch_deg": 1.0, "yaw_deg": 0.5}, view=False)["status"] == "ok"
    finally:
        job.close()
    following = Following(job_folder, monkeypatch)
    try:
        assert following.checkpoints[0]["host_angles"] == {"pitch_deg": 1.0, "yaw_deg": 0.5}
    finally:
        following.close()


def test_a_folder_without_an_abba_job_is_refused(tmp_path):
    (tmp_path / "langslice").mkdir()
    with pytest.raises(ValueError, match="no job prepared from ABBA"):
        follow(AbbaFollowRequest(job_folder=str(tmp_path / "langslice")), print,
               threading.Event())


def test_the_follower_ends_when_the_connector_closes_its_input(tmp_path):
    job_folder = _prepared(tmp_path)
    line = json.dumps({"id": "f", "method": "abba.follow",
                       "params": {"job_folder": str(job_folder)}})
    output = io.StringIO()
    run_stdio(input_stream=io.StringIO(line + "\n"), output_stream=output)
    messages = [json.loads(text) for text in output.getvalue().splitlines()]
    assert messages == [
        {"id": "f", "type": "event", "event": {"kind": "data", "payload": {"kind": "waiting"}}},
        {"id": "f", "type": "result", "result": {}},
    ]


# --- the tracker's rules ---------------------------------------------------------


def _facts(folder: Path, spec: Any = None) -> HostFacts:
    from langslice.job.settings.spec import JobSpec

    return HostFacts(folder, spec or JobSpec(image_folder=str(folder)),
                     {"s0.tif": (80, 40), "s1.tif": (80, 40)}, 25.0)


@pytest.mark.parametrize(("frozen", "sent"), [
    (frozenset(), True), (frozenset({"orientation"}), False), (frozenset({"affine"}), True)])
def test_orientation_reaches_the_host_unless_it_is_frozen(frozen, sent):
    before = {"slices": [{"id": "a.tif", "position_mm": 4.0, "index_corrected": 0,
                          "flip": False, "rotation_deg": 0, "transform": None}]}
    after = copy.deepcopy(before)
    after["slices"][0].update(flip=True, rotation_deg=90)
    updates = _host_updates(after, before, {"a.tif": (100, 80)}, 10.0, lambda _name: frozen)
    assert bool(updates) is sent
    if sent:
        assert updates[0]["flip"] is True and updates[0]["rotation_deg"] == 90


def _angled_state(*angles):
    from langslice.core.stack.section import Section
    from langslice.core.stack.stack import Stack

    slices = [Section(id=f"s{i}.tif", index_original=i, index_corrected=i)
              for i in range(len(angles))]
    for record, (pitch, yaw) in zip(slices, angles, strict=True):
        record.angles = (pitch, yaw)
    return Stack(slices=slices)


def test_an_angle_per_section_is_refused_before_abba_sees_it(tmp_path):
    events: list[dict[str, Any]] = []
    tracker = HostCheckpoints(_facts(tmp_path), events.append)
    tracker(_angled_state((0.0, 0.0), (0.0, 0.0)))  # one angle: as before
    assert len(events) == 1 and events[0]["initial"]
    with pytest.raises(ValueError, match="cannot show") as refused:
        tracker(_angled_state((0.0, 0.0), (2.0, 0.0)))
    assert str(refused.value) == ABBA_MIXED_ANGLES
    assert len(events) == 1  # nothing emitted to the host


def test_host_angles_are_sent_when_the_stack_wide_angles_change(tmp_path):
    events: list[dict[str, Any]] = []
    tracker = HostCheckpoints(_facts(tmp_path), events.append)
    tracker(_angled_state((0.0, 0.0), (0.0, 0.0)))
    tracker(_angled_state((0.0, 0.0), (0.0, 0.0)))
    tracker(_angled_state((2.0, -1.5), (2.0, -1.5)))
    tracker(_angled_state((2.0, -1.5), (2.0, -1.5)))
    assert "host_angles" not in events[0] and "host_angles" not in events[1]
    assert events[2]["host_angles"] == {"pitch_deg": 2.0, "yaw_deg": -1.5}
    assert "host_angles" not in events[3]


# --- warp rows (real deformation fits) ---------------------------------------------


@pytest.fixture(scope="module")
def warped(tmp_path_factory):
    """A real job (the shared synthetic stack) placed through the
    library, with the checkpoint tracker attached to it."""
    import dataclasses

    from langslice.doors.jobs import create
    from langslice.job.settings.spec import NonlinearSpec
    from tests.synthetic_stack import (
        ID0,
        ID1,
        PIXEL_SIZE_UM,
        atlas_loader,
        full_spec,
        write_sections,
    )

    root = tmp_path_factory.mktemp("warp_rows")
    patch = pytest.MonkeyPatch()
    patch.setenv("HOME", str(root / "home"))
    folder = root / "stack"
    write_sections(folder)
    spec = dataclasses.replace(full_spec(folder), nonlinear=NonlinearSpec(provider="none"))
    create(spec, atlas_loader=atlas_loader()).close()
    job = langslice.open_job(str(folder), atlas_loader=atlas_loader())
    geometry = {}
    for path in folder.iterdir():
        if path.suffix == ".png":
            with Image.open(path) as image:
                geometry[path.name] = image.size
    facts = HostFacts(folder, job.job.spec, geometry, PIXEL_SIZE_UM)
    events: list = []
    tracker = HostCheckpoints(facts, events.append)
    tracker.attach(job.job, job.workspace)
    assert job.position_sections(sections=[{"id": ID0, "position_mm": 0.1},
                                           {"id": ID1, "position_mm": 0.15}],
                                 view=False)["status"] == "ok"
    assert job.interactive_transform(sections=[
        {"id": ID0, "rotation_deg": 3.0, "scale_x": 1.05, "scale_y": 0.97,
         "translate_x_mm": 0.1, "translate_y_mm": -0.05, "shear": 0.04},
        {"id": ID1, "flip": True, "rotate_quarter": 90, "rotation_deg": -4.0,
         "scale_x": 1.0, "scale_y": 1.0, "translate_x_mm": 0.0, "translate_y_mm": 0.02}],
        view=False)["status"] == "ok"
    tracker(job.state)
    yield job, tracker, events, facts
    job.close()
    patch.undo()


def _row(event, name):
    return next((row for row in event["host_updates"] if row["id"] == name), None)


@pytest.mark.slow
def test_an_applied_deformation_lands_as_warp_pairs_on_the_affine(warped):
    from langslice.core.fits.deformable.geometry import sample_native
    from langslice.core.geometry.thin_plate import ThinPlateKernel
    from langslice.core.maps.abba import world_frame
    from langslice.core.maps.section_maps import (
        native_points,
        section_frame,
        stored_params,
    )
    from tests.synthetic_stack import ID0, ID1

    pytest.importorskip("fireants", reason="fireants_syn needs FireANTs")
    job, tracker, events, facts = warped
    # A SyN field varies on a finer scale than 1089 landmarks follow, so the
    # pairs approximate it with an error the row reports; the conversion's
    # own exactness on a smooth field is test_abba_warp's.
    assert job.fireants_syn(sections=[ID0], view=False)["status"] == "ok"
    tracker(job.state)
    row = _row(events[-1], ID0)
    warp = row["warp"]
    assert set(warp) == {"source_mm", "target_mm", "record", "max_error_mm", "p99_error_mm",
                         "p95_error_mm", "median_error_mm", "region", "points"}
    assert 0 < warp["median_error_mm"] <= warp["p95_error_mm"] <= warp["p99_error_mm"]
    assert warp["p99_error_mm"] <= warp["max_error_mm"]
    logged = [e["message"] for e in events if e.get("kind") == "log" and ID0 in e["message"]]
    assert any("deformation sent to ABBA" in m and "landmarks" in m for m in logged)
    source = np.asarray(warp["source_mm"]).T
    target = np.asarray(warp["target_mm"]).T
    assert len(warp["source_mm"]) == 2 and source.shape == (warp["points"], 2)
    assert _row(events[-1], ID1) is None  # nothing changed there
    # ABBA's pull-back reproduces the record's own map at points off the
    # grid inside the atlas brain, within the error the row reports.
    record = job.state.by_id(ID0)
    frame = section_frame(job.state, job.workspace, record)
    deformation = job.job.deformations.current(job.state, record)
    to_world, native_to_world = world_frame(frame, stored_params(record),
                                            size=facts.geometry[ID0],
                                            pixel_size_um=facts.calibration)
    rng = np.random.default_rng(3)
    width, height = facts.geometry[ID0]
    points = rng.uniform([0, 0], [width - 1, height - 1], size=(500, 2))
    nx, ny = native_points(frame, deformation, points[None, :, 0], points[None, :, 1])
    in_brain = sample_native(frame.annotation, np.stack([nx[0], ny[0]], axis=-1)) != 0
    homogeneous = np.column_stack([points, np.ones(len(points))])
    before = (to_world @ frame.file_to_render @ homogeneous.T).T[:, :2]
    after = (native_to_world @ np.stack([nx[0], ny[0], np.ones(len(points))])).T[:, :2]
    pullback = ThinPlateKernel(target, source, max_points=33 ** 2)
    errors = np.linalg.norm(pullback.forward(after) - before, axis=1)[in_brain]
    assert in_brain.sum() > 100
    assert np.percentile(errors, 99) <= warp["max_error_mm"]
    assert errors.max() <= 1.5 * warp["max_error_mm"]


@pytest.mark.slow
def test_a_warp_that_cannot_be_converted_is_sent_as_an_error_row(warped, monkeypatch):
    """CONN-4: the row names the failure (the connector keeps it as a failed
    update and leaves ABBA's warp as it is); it is not remembered, so the
    next checkpoint tries again."""
    from tests.synthetic_stack import ID0

    pytest.importorskip("fireants", reason="fireants_syn needs FireANTs")
    job, tracker, events, _facts = warped
    if job.state.by_id(ID0).deformation is None:
        assert job.fireants_syn(sections=[ID0], view=False)["status"] == "ok"
        tracker(job.state)
    record = job.state.by_id(ID0)
    key = str(deformation_to_json(record.deformation)["key"])
    real = tracker._warp_row

    def folds(*_args):
        raise ValueError("the thin-plate spline folds at every grid")

    tracker._warp_cache.clear()
    shutil.rmtree(job.job.layout.logs_dir / WARP_CACHE_DIR, ignore_errors=True)
    monkeypatch.setattr(tracker, "_warp_row", folds)
    assert tracker._warp(job.state, record, key) == {
        "error": "its deformation could not be converted "
                 "(the thin-plate spline folds at every grid)"}
    monkeypatch.setattr(tracker, "_warp_row", real)
    assert "source_mm" in tracker._warp(job.state, record, key)


@pytest.mark.slow
def test_a_moved_placement_removes_the_warp_step(warped):
    """A moved placement sends ABBA no warp for the section."""
    from tests.synthetic_stack import ID0

    pytest.importorskip("fireants", reason="fireants_syn needs FireANTs")
    job, tracker, events, _facts = warped
    assert job.fireants_syn(sections=[ID0], view=False)["status"] == "ok"
    tracker(job.state)
    assert _row(events[-1], ID0)["warp"] is not None
    assert job.position_sections(sections=[{"id": ID0, "position_mm": 0.11}],
                                 view=False)["status"] == "ok"
    tracker(job.state)
    moved = _row(events[-1], ID0)
    assert moved["position_mm"] == pytest.approx(0.11) and moved["warp"] is None


@pytest.mark.slow
def test_several_warps_are_converted_side_by_side_and_never_again(
        warped, monkeypatch):
    """Two sections fitted in one call are converted at once, in the
    conversion processes; a follower started again (ABBA reopened) sends
    the same rows from the job folder without converting anything."""
    from langslice.core.stack.rows import ANGLES_KEY
    from langslice.job.state_file import stack_to_json
    from tests.synthetic_stack import ID0, ID1

    pytest.importorskip("fireants", reason="fireants_syn needs FireANTs")
    job, tracker, events, facts = warped
    assert job.fireants_syn(sections=[ID0, ID1], view=False)["status"] == "ok"
    tracker(job.state)
    first = {name: _row(events[-1], name)["warp"] for name in (ID0, ID1)}
    assert all("source_mm" in warp for warp in first.values())
    logged = [e["message"] for e in events if e.get("kind") == "log"]
    assert any(m.startswith("2 deformations converted for ABBA") for m in logged)
    assert sum("converted in" in m for m in logged[-3:]) == 2
    assert len(list((job.job.layout.logs_dir / WARP_CACHE_DIR).glob("*.json"))) >= 2

    seen: list = []
    again = HostCheckpoints(facts, seen.append)
    again.attach(job.job, job.workspace)
    current = stack_to_json(job.state)
    angles = current[ANGLES_KEY]
    again.previous = baseline_rows(current, {"pitch_deg": angles["pitch"],
                                             "yaw_deg": angles["yaw"]})

    def never(*_args):
        raise AssertionError("a kept warp was converted again")

    monkeypatch.setattr(again, "_warp_row", never)
    monkeypatch.setattr(again, "_side_by_side", never)
    again(job.state)
    for name in (ID0, ID1):
        assert _row(seen[-1], name)["warp"] == first[name]
    assert sum("converted before" in e.get("message", "") for e in seen) == 2
    tracker.close()


def test_a_kept_warp_is_named_by_everything_its_rows_come_from():
    key = ("s0.png", "record", "linear")
    base = conversion_digest(key, (1, 0, 0, 0, 1, 0), (40, 30), 25.0)
    assert base == conversion_digest(key, (1, 0, 0, 0, 1, 0), (40, 30), 25.0)
    for other in (conversion_digest(("s0.png", "other", "linear"),
                                    (1, 0, 0, 0, 1, 0), (40, 30), 25.0),
                  conversion_digest(key, (1, 0, 0, 0, 1, 0.1), (40, 30), 25.0),
                  conversion_digest(key, (1, 0, 0, 0, 1, 0), (41, 30), 25.0),
                  conversion_digest(key, (1, 0, 0, 0, 1, 0), (40, 30), 20.0)):
        assert other != base


def test_conversion_processes_start_with_their_share_of_threads(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    with started_with_threads(3):
        assert all(os.environ[name] == "3" for name in THREAD_VARIABLES)
    assert os.environ["OMP_NUM_THREADS"] == "7"
    assert "MKL_NUM_THREADS" not in os.environ

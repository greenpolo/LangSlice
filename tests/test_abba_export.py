"""The ABBA project export (``core/maps/abba_world.py``, ``job/abba_project.py``,
``tools/abba_export.py``) and the window's Export menu (``hosts/gui/export.py``).

ABBA is not run here: its chain from a slice's pixels to the atlas is
replicated from what ABBA 0.25 writes into a QuPath project (SliceBench
``slicebench/ingest/abba.py`` reads it the same way), and the exported file
is read back through it. ``connectors/fiji``'s opt-in ``ExportCheck`` opens a
real export in ABBA itself.
"""

from __future__ import annotations

import http.client
import json
import math
import shutil
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from langslice.core.geometry.thin_plate import ThinPlateKernel
from langslice.core.maps import abba_world
from langslice.core.maps.abba_world import (
    AXIS_OFFSET_MM,
    CENTRE_MM,
    atlas_um_to_world,
    reslice_matrix,
    rotations,
)
from langslice.hosts.gui.export import Exports, Written
from langslice.hosts.gui.server import GuiServer
from langslice.hosts.gui.watch import JobList
from tests.synthetic_stack import ID0, ID1, ID2
from tests.test_gui_render import LOADER, synthetic_job

SYNTHETIC = "synthetic_ellipses_50um"


# --- ABBA's world frame -------------------------------------------------------


@pytest.mark.parametrize(("pitch", "yaw"), [(0.0, 0.0), (-12.0, 3.5), (8.0, -6.0)])
def test_the_world_frame_is_abba_s(pitch: float, yaw: float) -> None:
    rx, ry = rotations(pitch, yaw)
    assert (rx, ry) == (-math.radians(pitch), -math.radians(yaw))
    to_world = atlas_um_to_world(rx, ry)
    point = np.array([2500.0, 4000.0, 5690.0])  # asr: AP, DV, ML
    world = to_world[:, :3] @ point + to_world[:, 3]
    # ABBA's centre maps to the world origin in x, y; z is AP plus the offset
    assert world[:2] == pytest.approx([0.0, 0.0], abs=1e-12)
    assert world[2] == pytest.approx(2.5 + AXIS_OFFSET_MM)
    atlas_mm = np.array([point[2], point[1], point[0]]) / 1000.0
    expected = reslice_matrix(rx, ry) @ (atlas_mm - [*CENTRE_MM, 0.0]) + [0, 0, AXIS_OFFSET_MM]
    assert world == pytest.approx(expected)


# --- the export, read back through ABBA's chain ---------------------------------


@pytest.fixture(scope="module")
def made(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return synthetic_job(tmp_path_factory.mktemp("abba_export"))


@pytest.fixture
def job_folder(made: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    shutil.copytree(made.parent, tmp_path / "stack")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    # the job's settings name the Allen atlas; the loader gives the synthetic one
    names = frozenset({SYNTHETIC, "allen_mouse_25um"})
    monkeypatch.setattr(abba_world, "ABBA_ATLASES", names)
    return tmp_path / "stack" / "langslice"


def _open(folder: Path) -> Any:
    import langslice

    return langslice.open_job(str(folder), atlas_loader=LOADER)


def _chain(state: dict[str, Any], slice_entry: dict[str, Any], size: tuple[int, int],
           voxel: float, xy: np.ndarray, *, warp: bool) -> np.ndarray:
    """File pixels -> BrainGlobe atlas um through ABBA's chain, as ABBA's
    QuPath export composes it: the source centred at (W - 1) / 2, the
    preTransform, the location, the affine step (section -> world), and the
    warp step inverted (its pull-back is stored)."""
    actions = slice_entry["actions"]
    location = [a for a in actions if a["type"] == "MoveSliceAction"][-1]["location"]
    pre = np.array(slice_entry["preTransform"]["affinetransform3d"]).reshape(3, 4)
    assert np.allclose(pre, np.eye(3, 4))
    width, height = size
    mm = np.column_stack([(xy[:, 0] - (width - 1) / 2) * voxel,
                          (xy[:, 1] - (height - 1) / 2) * voxel])
    registrations = [a["registration"] for a in actions if a["type"] == "RegisterSliceAction"]
    affine = np.array(json.loads(registrations[0]["transform"])["affinetransform3d"]).reshape(3, 4)
    assert registrations[0]["type"] == "AffineRegistration"
    world = mm @ affine[:2, :2].T + affine[:2, 3]
    if warp and len(registrations) > 1:
        spline = json.loads(registrations[1]["transform"])["realTransform"]["wrappedTransform"][
            "wrappedTransform"]
        atlas_side = np.array(spline["srcPts"]).T
        section_side = np.array(spline["tgtPts"]).T
        # ABBA draws a section point where the pull-back (atlas -> section) lands on it
        pull = ThinPlateKernel(atlas_side, section_side, max_points=len(atlas_side))
        guess = world.copy()
        for _ in range(30):  # Newton on the pull-back
            value = pull.forward(guess)
            step = np.linalg.solve(pull.jacobian(guess), (value - world)[..., None])
            guess = guess - step[..., 0]
        world = guess
    points = np.column_stack([world, np.full(len(world), location)])
    to_world = atlas_um_to_world(state["rotationX"], state["rotationY"])
    return (points - to_world[:, 3]) @ np.linalg.inv(to_world[:, :3]).T


def _read(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], ET.Element]:
    with zipfile.ZipFile(path) as archive:
        assert sorted(archive.namelist()) == ["_bdvdataset_0.xml", "sources.json", "state.json"]
        return (json.loads(archive.read("state.json")), json.loads(archive.read("sources.json")),
                ET.fromstring(archive.read("_bdvdataset_0.xml")))


def test_the_export_reproduces_langslice_s_maps(job_folder: Path) -> None:
    from langslice.core.maps.section_maps import native_points, section_frame
    from langslice.tools.exports import applied_record

    with _open(job_folder) as job:
        done = job.export_abba()
        opened = job._opened  # noqa: SLF001
        state_now = opened.job.state
        frames = {sid: section_frame(state_now, opened.ctx, state_now.by_id(sid))
                  for sid in (ID0, ID1)}
        warps = {sid: applied_record(opened.job, state_now, state_now.by_id(sid))
                 for sid in (ID0, ID1)}
    path = Path(done.path)
    assert path == job_folder / "exports" / "stack.abba" and path.is_file()
    assert done.sections == [ID0, ID1] and done.warped == [ID0]
    assert [row["id"] for row in done.skipped] == [ID2]
    assert done.warp_error_um[ID0]["max"] < 10.0

    state, sources, dataset = _read(path)
    assert state["version"] == "0.25.0"
    assert len(state["slices_state_list"]) == 2
    # one source per channel of each file, opened by Bio-Formats
    assert sum(len(entry["actions"][0]["original_sources"]["source_indexes"])
               for entry in state["slices_state_list"]) == len(sources)
    openers = json.loads(dataset.findtext(".//openers") or "[]")
    assert [Path(o["location"]).name for o in openers] == [ID0, ID1]
    assert all(o["type"] == "BIOFORMATS" for o in openers)
    assert len(dataset.findall(".//ViewSetup")) == len(sources)

    by_file = {Path(o["location"]).name: entry for o, entry in
               zip(openers, state["slices_state_list"], strict=True)}
    for sid in (ID0, ID1):
        frame = frames[sid]
        width, height = frame.file_size
        voxel = frame.file_um_per_px / 1000.0
        grid = np.array([[x, y] for y in np.linspace(5, height - 6, 7)
                         for x in np.linspace(5, width - 6, 7)], dtype=np.float64)
        linear = _chain(state, by_file[sid], (width, height), voxel, grid, warp=False)
        rows = np.column_stack([grid[:, 1], grid[:, 0], np.ones(len(grid))])
        expected = rows @ np.asarray(frame.pixel_to_atlas_um()).T
        assert np.abs(linear - expected).max() < 1e-6  # um
        if warps[sid] is None:
            continue
        nx, ny = native_points(frame, warps[sid], grid[:, 0], grid[:, 1])
        native = np.column_stack([np.ravel(ny), np.ravel(nx), np.ones(len(grid))])
        warped_expected = native @ np.asarray(frame.native_to_um).T
        warped = _chain(state, by_file[sid], (width, height), voxel, grid, warp=True)
        assert np.linalg.norm(warped - warped_expected, axis=1).max() < 12.0  # um
        registration = by_file[sid]["actions"][-1]["registration"]
        assert registration["type"] == "BigWarpSource2DRegistration"
        assert registration["parameters"]["langslice_points"] == str(
            len(json.loads(registration["transform"])["realTransform"]["wrappedTransform"][
                "wrappedTransform"]["srcPts"][0]))
    record = json.loads((path.parent / "abba_export.json").read_text(encoding="utf-8"))
    assert record["project"] == "stack.abba" and record["warped"] == [ID0]


def test_linear_only_and_what_abba_cannot_open(job_folder: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    from langslice.core.refusal import RefusalCode, Refused

    with _open(job_folder) as job:
        few = job.export_abba(landmarks=30)
        state, _sources, _dataset = _read(Path(few.path))
        pairs = [json.loads(entry["actions"][-1]["registration"]["transform"])
                 for entry in state["slices_state_list"]
                 if entry["actions"][-1]["registration"]["type"]
                 == "BigWarpSource2DRegistration"]
        assert few.landmarks == 30 and pairs
        for transform in pairs:
            spline = transform["realTransform"]["wrappedTransform"]["wrappedTransform"]
            assert len(spline["srcPts"][0]) <= 30
        with pytest.raises(Refused):
            job.export_abba(landmarks=5000)
        done = job.export_abba(warps=False)
        assert done.warped == [] and done.sections == [ID0, ID1]
        state, _sources, _dataset = _read(Path(done.path))
        assert all(len([a for a in entry["actions"] if a["type"] == "RegisterSliceAction"]) == 1
                   for entry in state["slices_state_list"])
        monkeypatch.setattr(abba_world, "ABBA_ATLASES", frozenset())
        with pytest.raises(Refused) as refused:
            job.export_abba()
        assert refused.value.code == RefusalCode.BAD_ARGS


def test_mixed_cutting_angles_are_refused(job_folder: Path) -> None:
    from langslice.core.refusal import RefusalCode, Refused

    with _open(job_folder) as job:
        job.context  # noqa: B018 - the job is open
        held = job._opened.job  # noqa: SLF001
        with held.writing():
            before = held.snapshot()
            held.state.by_id(ID1).angles = (4.0, 0.0)  # type: ignore[union-attr]
            held.commit(before)
        with pytest.raises(Refused) as refused:
            job.export_abba()
        assert refused.value.code == RefusalCode.BAD_ARGS
        assert "one pair of cutting angles" in str(refused.value)


# --- the window's Export menu ---------------------------------------------------


def _request(gui: GuiServer, resource: str, body: Any = None, *, method: str = "POST",
             origin: str | None = "same") -> tuple[int, Any]:
    connection = http.client.HTTPConnection("127.0.0.1", gui.port, timeout=5)
    data = json.dumps(body).encode() if body is not None else b""
    connection.putrequest(method, f"/{gui.token}/{resource}", skip_host=True,
                          skip_accept_encoding=True)
    connection.putheader("Host", f"127.0.0.1:{gui.port}")
    if origin is not None:
        own = f"http://127.0.0.1:{gui.port}"
        connection.putheader("Origin", own if origin == "same" else origin)
    connection.putheader("Content-Type", "application/json")
    connection.putheader("Content-Length", str(len(data)))
    connection.endheaders(data if method == "POST" else None)
    response = connection.getresponse()
    out = response.read()
    connection.close()
    return response.status, json.loads(out) if out.startswith(b"{") else out


def test_the_export_menu_starts_polls_and_refuses(tmp_path: Path) -> None:
    folder = tmp_path / "images" / "langslice"
    folder.mkdir(parents=True)
    calls: list[tuple[Path, str]] = []

    def fake_run(job: Path, kind: str, landmarks: int) -> Written:
        calls.append((job, kind, landmarks))
        if kind == "visualign":
            raise ValueError("no section has a placement to write")
        return Written(str(job / "exports" / f"x.{kind}"), "2 sections", 1)

    gui = GuiServer(JobList([folder], recent=None), exports=Exports(run=fake_run))
    gui.start()
    try:
        assert _request(gui, "jobs/0/export", method="GET") == (200, {
            "state": "idle", "format": "", "path": "", "message": "", "skipped": 0,
            "started": 0.0, "seconds": 0.0})
        status, answer = _request(gui, "jobs/0/export", {"format": "abba"})
        assert status == 200 and answer["state"] in ("running", "done")
        for _ in range(100):
            status, answer = _request(gui, "jobs/0/export", method="GET")
            if answer["state"] != "running":
                break
            time.sleep(0.02)
        assert answer["state"] == "done" and answer["path"].endswith("x.abba")
        assert answer["skipped"] == 1 and calls == [(folder, "abba", 1089)]
        _request(gui, "jobs/0/export", {"format": "abba_linear", "landmarks": 100})
        for _ in range(100):
            status, answer = _request(gui, "jobs/0/export", method="GET")
            if answer["state"] != "running":
                break
            time.sleep(0.02)
        assert calls[-1] == (folder, "abba_linear", 100)
        assert _request(gui, "jobs/0/export", {"format": "abba", "landmarks": 50})[0] == 400

        _request(gui, "jobs/0/export", {"format": "visualign"})
        for _ in range(100):
            status, answer = _request(gui, "jobs/0/export", method="GET")
            if answer["state"] != "running":
                break
            time.sleep(0.02)
        assert answer["state"] == "failed" and "no section" in answer["message"]

        assert _request(gui, "jobs/0/export", {"format": "pdf"})[0] == 400
        assert _request(gui, "jobs/9/export", {"format": "abba"})[0] == 404
        assert _request(gui, "jobs/0/export", {"format": "abba"}, origin=None)[0] == 403
        assert _request(gui, "jobs/0/export", {"format": "abba"},
                        origin="http://evil.example")[0] == 403
        # nothing exported yet: the folder cannot be shown
        assert _request(gui, "jobs/0/export/reveal", {})[0] == 400
        assert _request(gui, "jobs/0/snapshot", {})[0] == 405
    finally:
        gui.close()

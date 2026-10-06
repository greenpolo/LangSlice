"""The job folder: layout, per-step undo files, refused older formats, and
every picture the model is shown saved with its layers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import tifffile
from PIL import Image

from langslice.agent.engine import build_context
from langslice.core.jpeg import encode_jpeg
from langslice.core.layers import coordinate_map
from langslice.core.spec import JobSpec
from langslice.doors.tools.toolbox import build_tools
from langslice.job.checkpoint import STATE_FORMAT_VERSION
from langslice.job.history import UNDO_DEPTH
from langslice.job.job import Job
from langslice.job.layout import JOB_FORMAT_VERSION, JobLayout, section_dirname
from langslice.job.views import flush_all
from tests.fakes import SlabAtlas
from tests.linear_tool_helpers import tool_named as _tool

_ATLAS = SlabAtlas()
KEY = "0123456789abcdef01234567"
CALL = "fedcba9876543210fedcba98"


def _folder(root: Path, n: int = 3) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for index in range(n):
        Image.fromarray(np.full((30, 40, 3), 40 + 10 * index, dtype=np.uint8)).save(
            root / f"s{index}.png")
    return root


def _open(folder: Path, **spec_kwargs: Any) -> tuple[Job, Any]:
    spec = JobSpec(image_folder=str(folder), model="fake-model", preprocess="none",
                   **spec_kwargs)
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: _ATLAS)
    return Job.open(spec, ctx, folder=ctx.job_folder, results_path=ctx.results_path), ctx


# --- the layout --------------------------------------------------------------------------


def test_section_folders_are_stems_unless_two_images_share_one(tmp_path: Path):
    assert section_dirname("s1.png", ["s0.png", "s1.png"]) == "s1"
    assert section_dirname("s1.png", ["s1.png", "s1.tif"]) == "s1.png"
    assert section_dirname("S1.png", ["S1.png", "s1.tif"]) == "S1.png"
    assert section_dirname("odd:name?.tif") == "odd_name_"
    layout = JobLayout.for_images(tmp_path)
    assert layout.folder == tmp_path / "langslice"
    assert layout.relative(layout.folder / "sections" / "s0" / "x") == "sections/s0/x"
    outside = tmp_path.parent / "elsewhere" / "file"
    assert layout.relative(outside) == str(outside)
    assert layout.resolve("sections/s0") == layout.folder / "sections" / "s0"


def test_open_lays_out_the_job_folder_next_to_the_images(tmp_path: Path):
    folder = _folder(tmp_path / "stack")
    job, _ctx = _open(folder)
    root = folder / "langslice"
    assert job.folder == root
    assert sorted(path.name for path in folder.iterdir()) == [
        "langslice", "s0.png", "s1.png", "s2.png"]
    for name in ("history", "sections", "views", "exports", "logs"):
        assert (root / name).is_dir()
    record = json.loads((root / "job.json").read_text())
    assert record["format_version"] == JOB_FORMAT_VERSION
    assert record["spec"]["image_folder"] == str(folder)
    assert json.loads((root / "state.json").read_text())["format_version"] == STATE_FORMAT_VERSION
    assert "open" in (root / "logs" / "events.jsonl").read_text()
    # The public rendering of the state, written with every checkpoint.
    registration = json.loads((root / "registration.json").read_text())
    assert registration["format_version"] == 1
    assert [entry["id"] for entry in registration["sections"]] == ["s0.png", "s1.png", "s2.png"]
    assert all(entry["pixel_to_atlas_um"] is None and entry["problem"] == "no position"
               for entry in registration["sections"])


def test_the_open_event_says_whether_a_checkpoint_was_resumed(tmp_path: Path):
    folder = _folder(tmp_path / "stack")
    _open(folder)
    _open(folder)
    events = [json.loads(line) for line in
              (folder / "langslice" / "logs" / "events.jsonl").read_text().splitlines()]
    assert [event["resumed"] for event in events if event["kind"] == "open"] == [False, True]


def test_a_job_folder_from_a_newer_langslice_is_refused(tmp_path: Path):
    folder = _folder(tmp_path / "stack")
    _open(folder)
    path = folder / "langslice" / "job.json"
    record = json.loads(path.read_text())
    record["format_version"] = JOB_FORMAT_VERSION + 1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="newer LangSlice"):
        _open(folder)
    assert json.loads(path.read_text())["format_version"] == JOB_FORMAT_VERSION + 1


def test_undo_is_one_file_per_step_and_bounded(tmp_path: Path):
    folder = _folder(tmp_path / "stack")
    job, _ctx = _open(folder)
    history = folder / "langslice" / "history"

    def steps() -> list[str]:
        return sorted(path.name for path in history.glob("step-*.json"))

    before = job.snapshot()
    job.state.notes.append("first")
    job.commit(before)
    first = steps()
    assert len(first) == 1
    stamp = (history / first[0]).stat()
    before = job.snapshot()
    job.state.notes.append("second")
    job.commit(before)
    assert steps()[:1] == first and len(steps()) == 2
    again_stamp = (history / first[0]).stat()  # the old step is not rewritten
    assert (again_stamp.st_ino, again_stamp.st_mtime_ns) == (stamp.st_ino, stamp.st_mtime_ns)
    for index in range(UNDO_DEPTH + 5):
        before = job.snapshot()
        job.state.notes.append(f"note {index}")
        job.commit(before)
    assert len(steps()) == UNDO_DEPTH and first[0] not in steps()
    index = json.loads((history / "index.json").read_text())
    assert index["undo"] == steps() and index["redo"] == []
    assert job.undo() and len(steps()) == UNDO_DEPTH  # one leaves undo, one joins redo
    again, _ = _open(folder)
    assert len(again.undo_stack) == UNDO_DEPTH - 1 and len(again.redo_stack) == 1
    assert again.redo() and again.state.notes[-1] == f"note {UNDO_DEPTH + 4}"



@pytest.mark.parametrize("damage", ["step", "newer_index"])
def test_a_history_that_cannot_be_read_is_never_deleted(tmp_path: Path, damage: str,
                                                       caplog: Any):
    """An unreadable step or an index from a newer LangSlice: the job opens
    without undo for the session, with a warning, and the history on disk
    is left exactly as it was, through the open and later writes."""
    import logging

    folder = _folder(tmp_path / "stack")
    job, _ctx = _open(folder)
    for index in range(3):
        before = job.snapshot()
        job.state.notes.append(f"note {index}")
        job.commit(before)
    history = folder / "langslice" / "history"
    if damage == "step":
        (history / "step-000002.json").write_text("{ not json")
    else:
        index = json.loads((history / "index.json").read_text())
        index["format_version"] = 99
        (history / "index.json").write_text(json.dumps(index))
    held = {path.name: path.read_bytes() for path in history.iterdir()}
    with caplog.at_level(logging.WARNING):
        again, _ = _open(folder)
    assert again.undo_stack == [] and again.redo_stack == []
    assert any("history" in record.getMessage() for record in caplog.records)
    before = again.snapshot()
    again.state.notes.append("after")
    again.commit(before)
    assert again.undo() and again.state.notes[-1] == "note 2"  # this session's own step
    assert {path.name: path.read_bytes() for path in history.iterdir()} == held
    # A fresh job on the folder leaves it alone too.
    fresh, _ = _open(folder, resume=False)
    assert {path.name: path.read_bytes() for path in history.iterdir()} == held
    assert fresh.undo_stack == []

# --- older formats ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", [None, 1, 2])
def test_a_checkpoint_of_an_older_format_is_refused(tmp_path: Path, version: int | None):
    """A pre-release checkpoint (unversioned, or an older format) is not read:
    the job asks to be started fresh and leaves the file alone."""
    folder = _folder(tmp_path / "stack")
    _open(folder)
    path = folder / "langslice" / "state.json"
    record = json.loads(path.read_text())
    record.pop("format_version")
    if version is not None:
        record["format_version"] = version
    path.write_text(json.dumps(record))
    held = path.read_bytes()
    with pytest.raises(ValueError, match="older pre-release LangSlice.*Start a new job"):
        _open(folder)
    assert path.read_bytes() == held
    fresh, _ = _open(folder, resume=False)
    assert json.loads(path.read_text())["format_version"] == STATE_FORMAT_VERSION
    assert fresh.state.notes[0].startswith("ingest:")


# --- the saved pictures ---------------------------------------------------------------------


def _golden_toolbox(folder: Path) -> tuple[Any, Any]:
    from langslice.job.job import ingest
    from tests.golden.record import atlas_loader, full_spec, write_sections

    write_sections(folder)
    spec = full_spec(folder)
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=atlas_loader())
    state = ingest(spec, ctx)
    return build_tools(state, ctx, spec), ctx


def _place(box: Any) -> None:
    from tests.golden.record import ID0, ID1, ID2

    _tool(box, "set_positions")([{"id": ID0, "position_mm": 0.1},
                                 {"id": ID1, "position_mm": 0.15},
                                 {"id": ID2, "position_mm": 0.2}])


def test_every_picture_is_saved_as_sent_with_layers_for_placements(tmp_path: Path):
    from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY

    box, _ctx = _golden_toolbox(tmp_path / "stack")
    shown = _tool(box, "view_slices")(["s0.png", "s1.png"])[TOOL_MEDIA_PARTS_KEY]
    _place(box)
    placed = _tool(box, "view_placement")([{"id": "s1.png", "positions_mm": [0.15]}],
                                          view={"mode": "overlay"})[TOOL_MEDIA_PARTS_KEY]
    sheet = _tool(box, "view_stack")()[TOOL_MEDIA_PARTS_KEY]
    flush_all()
    root = tmp_path / "stack" / "langslice"
    index = [json.loads(line) for line in (root / "views.jsonl").read_text().splitlines()]
    assert [entry["seq"] for entry in index] == list(range(1, len(index) + 1))
    first = root / index[0]["path"]
    assert index[0]["path"] == "sections/s0/views/000001_view_slices_section"
    assert (first / "view.jpg").read_bytes() == encode_jpeg(shown[0])
    assert not (first / "labels.tif").exists()
    overlay = next(entry for entry in index if entry["tool"] == "view_placement")
    assert overlay["layers"] and overlay["sections"] == ["s1.png"]
    view = root / overlay["path"]
    assert (view / "view.jpg").read_bytes() == encode_jpeg(placed[0])
    labels = tifffile.imread(view / "labels.tif")
    assert labels.dtype == np.uint32 and labels.shape == placed[0].size[::-1]
    record = json.loads((view / "view.json").read_text())
    assert record["tool"] == "view_placement" and record["arguments"]["view"]["mode"] == "overlay"
    assert record["frame"]["plane"]["position_mm"] == pytest.approx(0.15)
    stacked = [entry for entry in index if entry["tool"] == "view_stack"]
    assert [entry["path"].split("/")[0] for entry in stacked] == ["views", "views"]
    assert len(stacked) == len([item for item in sheet if not isinstance(item, str)])


def test_the_layers_agree_with_the_borders_drawn_on_a_golden_picture(tmp_path: Path):
    from tests.golden.record import GOLDEN_DIR, ID2

    box, _ctx = _golden_toolbox(tmp_path / "stack")
    _place(box)
    # Golden call 024: the outlines picture, atlas template under the lines.
    _tool(box, "view_placement")([{"id": ID2, "positions_mm": [0.2]}],
                                 view={"mode": "outlines", "atlas_channels": ["ara", "borders"]})
    flush_all()
    view = next((tmp_path / "stack" / "langslice").glob("sections/s2/views/*view_placement*"))
    picture = np.asarray(Image.open(view / "view.jpg").convert("RGB")).astype(int)
    golden = np.asarray(Image.open(GOLDEN_DIR / "024_tools_view_placement.0.png")
                        .convert("RGB")).astype(int)
    assert picture.shape == golden.shape and np.array_equal(picture, golden)

    borders = np.asarray(Image.open(view / "borders.png")).astype(int)
    labels = tifffile.imread(view / "labels.tif")
    yellow = ((picture[..., 0] + picture[..., 1]) / 2 - picture[..., 2]) > 80
    strong = borders >= 128
    assert strong.sum() > 500
    assert (yellow & strong).sum() / strong.sum() > 0.85  # the layer is where lines are drawn
    assert (yellow & (borders > 0)).sum() / yellow.sum() > 0.98  # and every drawn line is in it
    change = np.zeros(labels.shape, dtype=bool)
    change[:-1] |= labels[:-1] != labels[1:]
    change[1:] |= labels[:-1] != labels[1:]
    change[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    change[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    near = cv2.dilate(change.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert (strong & near).sum() / strong.sum() > 0.98  # the lines are label boundaries


@pytest.mark.parametrize("angles", [(0.0, 0.0), (1.0, 0.5)])
def test_the_coordinate_map_lands_on_the_labels(tmp_path: Path, angles: tuple[float, float]):
    from tests.deformable_synthetic import SyntheticAtlas
    from tests.golden.record import ID1

    box, _ctx = _golden_toolbox(tmp_path / "stack")
    _place(box)
    if angles != (0.0, 0.0):
        _tool(box, "set_cutting_angles")(*angles)
    _tool(box, "view_placement")([{"id": ID1, "positions_mm": [0.15]}],
                                 view={"mode": "overlay", "zoom": [40, 40, 330, 380]})
    flush_all()
    view = next((tmp_path / "stack" / "langslice").glob("sections/s1/views/*view_placement*"))
    coords = coordinate_map(view / "view.json")
    labels = tifffile.imread(view / "labels.tif")
    record = json.loads((view / "view.json").read_text())
    assert coords.shape == labels.shape + (3,) and coords.dtype == np.float32
    bottom = record["frame"]["content_box"][3]  # the caption band is below the content
    assert np.isnan(coords[bottom:]).all() and np.isfinite(coords[:bottom]).all()
    annotation = np.asarray(SyntheticAtlas().annotation)
    voxel = np.rint(coords[:bottom] / np.asarray(record["frame"]["atlas"]["resolution_um"]))
    inside = ((voxel >= 0) & (voxel < np.asarray(annotation.shape))).all(axis=-1)
    found = np.zeros(voxel.shape[:2], dtype=np.int64)
    hit = voxel[inside].astype(int)
    found[inside] = annotation[hit[:, 0], hit[:, 1], hit[:, 2]]
    assert (found == labels[:bottom]).mean() > 0.99


def test_saving_does_not_hold_up_a_tool(tmp_path: Path):
    box, _ctx = _golden_toolbox(tmp_path / "stack")
    _place(box)
    for _ in range(5):
        _tool(box, "view_placement")([{"id": "s0.png", "positions_mm": [0.1]}],
                                     view={"mode": "overlay"})
    views = box.job.views
    assert views.queue_seconds / 6 < 0.02  # numbering and queueing only
    flush_all()
    assert views.write_seconds > 0


# --- where the job folder goes: an explicit folder, a read-only image folder ----------------


def test_an_explicit_job_folder_per_arm_on_one_image_folder(tmp_path: Path):
    folder = _folder(tmp_path / "dataset")
    first, _ = _open(folder, job_dir=str(tmp_path / "arm-a"))
    second, _ = _open(folder, job_dir=str(tmp_path / "arm-b"))
    assert first.folder == tmp_path / "arm-a" and second.folder == tmp_path / "arm-b"
    assert (tmp_path / "arm-a" / "state.json").exists()
    assert not (folder / "langslice").exists()  # the dataset folder is never written
    before = first.snapshot()
    first.state.notes.append("arm a")
    first.commit(before)
    again, _ = _open(folder, job_dir=str(tmp_path / "arm-a"))  # the same images continue
    assert again.state.notes[-1] == "arm a"
    assert "job_dir" not in JobSpec(image_folder=str(folder)).to_dict()


def test_a_job_folder_holding_another_image_folders_job_is_refused(tmp_path: Path):
    _open(_folder(tmp_path / "one"), job_dir=str(tmp_path / "shared"))
    with pytest.raises(ValueError, match="already holds the job of"):
        _open(_folder(tmp_path / "two"), job_dir=str(tmp_path / "shared"))


def test_the_cli_takes_a_job_folder(tmp_path: Path):
    from langslice.doors.cli import build_parser
    from langslice.doors.cli.linear import build_linear_spec

    args = build_parser().parse_args(
        ["linear", "run", str(tmp_path), "--job-dir", str(tmp_path / "arm")])
    assert build_linear_spec(args, str(tmp_path)).job_dir == str(tmp_path / "arm")


@pytest.fixture
def read_only(tmp_path: Path, monkeypatch: Any) -> Any:
    import os

    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root writes into read-only folders")
    from langslice.job import index, layout

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(index, "default_root", lambda: tmp_path / "home" / ".langslice" / "jobs")
    folder = _folder(tmp_path / "shared-drive")
    folder.chmod(0o555)
    if os.name == "nt":
        # Windows chmod does not remove directory write permission; simulate a
        # folder with an ACL that denies writes instead.
        original = layout.writable
        monkeypatch.setattr(layout, "writable", lambda target, images:
                            False if images == folder else original(target, images))
    yield folder
    folder.chmod(0o755)


def test_a_read_only_image_folder_falls_back_to_the_home_job_folder(
    tmp_path: Path, read_only: Path,
):
    from langslice.job import index

    messages: list[str] = []
    spec = JobSpec(image_folder=str(read_only), model="fake-model", preprocess="none")
    ctx = build_context(spec, emit=messages.append, atlas_loader=lambda _n: _ATLAS)
    job = Job.open(spec, ctx, folder=ctx.job_folder, results_path=ctx.results_path)
    job_id = index.folder_id(read_only)
    target = tmp_path / "home" / ".langslice" / "jobs" / job_id
    assert job.folder == target and (target / "state.json").exists()
    assert len([m for m in messages if "cannot be created or written" in m]) == 1
    entry = json.loads((tmp_path / "home" / ".langslice" / "jobs" / f"{job_id}.json").read_text())
    assert entry["job_folder"] == str(target)
    assert entry["fallback"]["image_folder"] == str(read_only)
    before = job.snapshot()
    job.state.notes.append("kept")
    job.commit(before)
    again, _ = _open(read_only)  # found again on reopen
    assert again.folder == target and again.state.notes[-1] == "kept"


def test_a_read_only_folder_job_opens_over_mcp_by_its_job_folder(
    tmp_path: Path, read_only: Path,
):
    """The job of a read-only image folder lives under the home job folder;
    an MCP host opens it by that job folder, as saved."""
    from langslice.doors.mcp.server import open_folder

    job, _ = _open(read_only, tasks=["position"])
    job_folder = job.folder
    from langslice.doors.jobs import close_job

    close_job(job)

    def flags(_folder: str) -> JobSpec:
        raise AssertionError("A folder's own job must not use the server's job flags")

    session = open_folder(str(job_folder), flags, lambda _n: _ATLAS)
    try:
        assert session.job.folder == job_folder
        assert session.ctx.image_folder == str(read_only)
        assert session.spec.tasks == ["position"]
    finally:
        session.close()


# --- a job folder moves with its images ---------------------------------------------------


def _create(folder: Path, **spec_kwargs: Any) -> Any:
    from langslice.doors.jobs import create

    spec = JobSpec(image_folder=str(folder), model="fake-model", preprocess="none",
                   **spec_kwargs)
    opened = create(spec, atlas_loader=lambda _n: _ATLAS, emit=lambda _m: None)
    opened.close()
    return opened


def test_a_job_folder_moves_with_its_images(tmp_path: Path, monkeypatch: Any):
    """The default job folder stores its images as its parent, so renaming
    or moving the image folder (job folder inside) keeps the job: the CLI
    and the library open it, the images are read where they are now, and a
    host reopening the moved folder resumes it."""
    import langslice
    from langslice.doors.jobs import open_folder

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    folder = _folder(tmp_path / "stack")
    _create(folder)
    record = json.loads((folder / "langslice" / "job.json").read_text())
    assert record["image_folder"] == ".."
    moved = tmp_path / "elsewhere" / "renamed"
    moved.parent.mkdir()
    folder.rename(moved)
    opened = open_folder(moved, atlas_loader=lambda _n: _ATLAS)
    assert Path(opened.spec.image_folder) == moved
    assert Path(opened.ctx.image_path("s0.png")).is_file()
    opened.close()
    job = langslice.open_job(str(moved / "langslice"), atlas_loader=lambda _n: _ATLAS)
    job.set_positions(entries=[{"id": "s0.png", "position_mm": 0.1}])
    job.close()
    registration = json.loads((moved / "langslice" / "registration.json").read_text())
    assert registration["image_folder"] == ".."
    again = _create(moved)  # a host opening the moved folder resumes the job
    assert again.job.state.by_id("s0.png").position_mm == 0.1


def test_an_explicit_job_folder_whose_images_moved_says_how_to_reattach(
    tmp_path: Path, monkeypatch: Any,
):
    from langslice.doors.jobs import NoJob, open_folder

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    folder = _folder(tmp_path / "dataset")
    arm = tmp_path / "arm"
    _create(folder, job_dir=str(arm))
    assert json.loads((arm / "job.json").read_text())["image_folder"] == str(folder)
    moved = tmp_path / "dataset-moved"
    folder.rename(moved)
    with pytest.raises(NoJob) as raised:
        open_folder(arm, atlas_loader=lambda _n: _ATLAS)
    message = str(raised.value)
    assert str(folder) in message and "--job-dir" in message and "init" in message
    # Reattached from the images' new place, the same job continues.
    again = _create(moved, job_dir=str(arm))
    assert json.loads((arm / "job.json").read_text())["image_folder"] == str(moved)
    assert again.job.folder == arm
    # Another image folder's job is still refused while its images exist.
    with pytest.raises(ValueError, match="already holds the job of"):
        _create(_folder(tmp_path / "other"), job_dir=str(arm))

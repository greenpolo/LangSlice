"""The viewer's pictures (``langslice.hosts.gui.render``): each section under
its current registration, the strip's thumbnails and facts, drawn from the job
folder without opening the job or writing to it, cached by render key, one
drawing per job at a time, and refused with the reason."""

from __future__ import annotations

import copy
import io
import json
import os
import shutil
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from langslice.core.pictures.look import LookRequest, look
from langslice.core.stack.stack import Stack
from langslice.hosts.gui import render
from langslice.hosts.gui.render import (
    JobRenderer,
    RenderBusy,
    RenderError,
    RenderPool,
    UnknownSection,
    atlas_positions,
    edge_px,
    render_key,
    strip_step,
)
from langslice.hosts.gui.watch import JobWatcher
from tests.cli_child import install
from tests.synthetic_stack import (
    ID0,
    ID1,
    ID2,
    PIXEL_SIZE_UM,
    atlas_loader,
    write_sections,
)

IDENTITY = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
LOADER = atlas_loader()


def synthetic_job(root: Path) -> Path:
    """A job of the synthetic stack in *root*: s0.png placed and warped (a
    ripple on its linear placement), s1.png placed, s2.png with no position.
    Returns the job folder (``root/stack/langslice``)."""
    import langslice
    from tests.fixtures.example_plugin import ripple_map

    images = root / "stack"
    write_sections(images)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HOME", str(root / "home"))
        install(LOADER, patch.setattr)
        with langslice.create_job(
                images, positions={ID0: 0.1, ID1: 0.15}, transforms={ID0: IDENTITY, ID1: IDENTITY},
                pixel_size_um=PIXEL_SIZE_UM, preprocess="none",
                tasks=["position", "transform", "nonlinear"]) as job:
            frame = job.context.fit_frame(ID0)
            done = job.context.set_deformations([{"id": ID0, "map": ripple_map(frame, 3.0)}])
            assert done.written == [ID0]
            return Path(job.folder)


@pytest.fixture(scope="module")
def made(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return synthetic_job(tmp_path_factory.mktemp("gui_render"))


@pytest.fixture
def job(made: Path, tmp_path: Path) -> Path:
    """A copy of the module's job, so a test may change it."""
    shutil.copytree(made.parent, tmp_path / "stack")
    return tmp_path / "stack" / "langslice"


@pytest.fixture
def pool() -> Iterator[ThreadPoolExecutor]:
    executor = ThreadPoolExecutor(max_workers=2)
    yield executor
    executor.shutdown(wait=True)


def _renderer(job: Path, pool: ThreadPoolExecutor) -> JobRenderer:
    watcher = JobWatcher(job)
    return JobRenderer(job, watcher.current_stack, pool, atlas_loader=LOADER)


def _image(data: bytes) -> Image.Image:
    picture = Image.open(io.BytesIO(data))
    assert picture.format == "JPEG"
    return picture.convert("RGB")


def _files(folder: Path) -> dict[str, tuple[int, int]]:
    out = {}
    for root, _dirs, names in os.walk(folder):
        for name in names:
            info = (Path(root) / name).stat()
            out[str(Path(root, name).relative_to(folder))] = (info.st_size, info.st_mtime_ns)
    return out


def _stack(job: Path) -> Stack:
    state = JobWatcher(job).current_stack()
    assert state is not None
    return state


def _look_content(job: Path, request: LookRequest) -> Image.Image:
    """``look``'s picture as the job draws it, its caption band cut off."""
    from langslice.core.fits.deformation import RecordStore
    from langslice.doors.jobs import context, read_spec

    workspace = context(read_spec(job), job, atlas_loader=LOADER)
    layout = workspace.layout
    store = RecordStore(root=layout.sections_dir, folder_of=layout.deformable_dir)
    picture = look(workspace, _stack(job), request, store=store)[0]
    width, height = picture.recipe["shown"]
    assert picture.image.height > height  # the caption band, below the content
    return picture.image.crop((0, 0, width, height))


def _difference(first: Image.Image, second: Image.Image) -> float:
    assert first.size == second.size
    return float(np.abs(np.asarray(first, dtype=np.float32)
                        - np.asarray(second, dtype=np.float32)).mean())


def test_every_picture_is_drawn_without_writing_to_the_folder(
        job: Path, pool: ThreadPoolExecutor) -> None:
    before = _files(job.parent)
    renderer = _renderer(job, pool)
    strip = renderer.strip()
    for section in (ID0, ID1, ID2):
        for picture in (renderer.overlay(section, 256), renderer.thumb(section, 128)):
            assert max(_image(picture).size) <= 256
    for thumb in strip.atlas:
        assert _image(renderer.atlas_thumb(thumb.k, 128)).size[0] > 0
    assert _files(job.parent) == before


def test_an_overlay_is_looks_overlay_with_the_deformation_and_without_its_caption(
        job: Path, pool: ThreadPoolExecutor) -> None:
    renderer = _renderer(job, pool)
    warped = _image(renderer.overlay(ID0, 512))
    drawn = _look_content(job, LookRequest(mode="overlay", sections=(ID0,), long_edge=512))
    linear = _look_content(job, LookRequest(mode="overlay", sections=(ID0,), long_edge=512,
                                            warp="none"))
    assert _difference(warped, drawn) < 3.0  # the same picture, through JPEG
    assert _difference(drawn, linear) > 0.05  # the ripple is drawn
    assert _difference(warped, drawn) < _difference(warped, linear)
    # A section with no position is drawn alone, oriented.
    alone = _image(renderer.overlay(ID2, 512))
    section = _look_content(job, LookRequest(mode="section", sections=(ID2,), long_edge=512))
    assert _difference(alone, section) < 3.0


def test_the_render_key_changes_with_what_the_section_shows(job: Path) -> None:
    state = _stack(job)
    keys = {section: render_key(state, section) for section in (ID0, ID1, ID2)}
    assert len(set(keys.values())) == 3 and all(len(key) == 12 for key in keys.values())
    assert render_key(copy.deepcopy(state), ID1) == keys[ID1]
    moved = copy.deepcopy(state)
    section = moved.by_id(ID1)
    assert section is not None
    section.position_mm = 0.2
    assert render_key(moved, ID1) != keys[ID1]
    assert render_key(moved, ID0) == keys[ID0]


def test_the_strip(job: Path, pool: ThreadPoolExecutor) -> None:
    strip = _renderer(job, pool).strip()
    assert (strip.axis.low, strip.axis.high, strip.axis.min_mm) == ("anterior", "posterior", 0.0)
    assert strip.axis.max_mm == pytest.approx(0.25)
    assert strip.step_mm == 0.1
    assert [(thumb.k, thumb.position_mm) for thumb in strip.atlas] == [(0, 0.1), (1, 0.2)]
    assert strip.angles == ""
    rows = {row["id"]: row for row in _snapshot(job)["sections"]}
    assert [(s.id, s.order, s.position_mm) for s in strip.sections] == [
        (ID0, 1, 0.1), (ID1, 2, 0.15), (ID2, 3, None)]
    assert [s.render_key for s in strip.sections] == [rows[s.id]["render_key"]
                                                       for s in strip.sections]


def _snapshot(job: Path) -> dict:
    watcher = JobWatcher(job)
    watcher.poll(force=True)
    return watcher.snapshot(generation=watcher.generation)


def test_strip_spacing_and_edges() -> None:
    assert [edge_px(n) for n in (0, 1, 64, 65, 300, 2048, 9999)] == [
        64, 64, 64, 128, 320, 2048, 2048]
    # The adult mouse brain's axes (Allen 25 um): ABBA's 0.5 mm.
    assert strip_step(0.0, 13.175, 1.0) == 0.5  # coronal, 26 thumbnails
    assert strip_step(0.0, 7.975, 1.0) == 0.5  # horizontal, 15
    assert atlas_positions(0.0, 7.975, 0.5) == tuple(0.5 * k for k in range(1, 16))
    # A rat brain spaces them wider; a tiny volume as finely as allowed.
    assert strip_step(0.0, 39.9, 1.7) == 1.0
    assert strip_step(0.0, 0.25, 1.0) == 0.1


def test_a_picture_is_drawn_once_for_concurrent_requests_and_then_kept(
        job: Path, pool: ThreadPoolExecutor, monkeypatch: pytest.MonkeyPatch) -> None:
    drawn: list[int] = []
    slow = threading.Event()

    def overlay(_opened: object, _state: object, _section: object, edge: int) -> Image.Image:
        drawn.append(edge)
        slow.wait(5)
        return Image.new("RGB", (edge, edge // 2), (90, 90, 90))

    monkeypatch.setattr(render, "_overlay_picture", overlay)
    renderer = _renderer(job, pool)
    answers: list[bytes] = []
    threads = [threading.Thread(target=lambda: answers.append(renderer.overlay(ID0, 128)))
               for _ in range(4)]
    for thread in threads:
        thread.start()
    time.sleep(0.3)
    slow.set()
    for thread in threads:
        thread.join(10)
    assert drawn == [128] and len(answers) == 4 and len(set(answers)) == 1
    assert renderer.overlay(ID0, 128) == answers[0] and drawn == [128]  # kept
    renderer.overlay(ID0, 192)
    assert drawn == [128, 192]  # another edge is another picture


def test_a_slow_picture_is_busy_after_the_wait_and_kept_when_drawn(
        job: Path, pool: ThreadPoolExecutor, monkeypatch: pytest.MonkeyPatch) -> None:
    done = threading.Event()

    def thumb(*_args: object, **_kwargs: object) -> Image.Image:
        done.wait(5)
        return Image.new("RGB", (64, 48), (20, 20, 20))

    monkeypatch.setattr(render, "WAIT_S", 0.2)
    monkeypatch.setattr(render, "render_slice", thumb)
    renderer = _renderer(job, pool)
    with pytest.raises(RenderBusy):
        renderer.thumb(ID1, 64)
    done.set()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            assert _image(renderer.thumb(ID1, 64)).size == (64, 48)
            break
        except RenderBusy:
            continue
    else:
        pytest.fail("the picture was never kept")


def test_what_cannot_be_drawn_says_why(job: Path, pool: ThreadPoolExecutor) -> None:
    renderer = _renderer(job, pool)
    with pytest.raises(UnknownSection):
        renderer.overlay("../job.json", 128)
    with pytest.raises(UnknownSection):
        renderer.atlas_thumb(len(renderer.strip().atlas), 128)
    # A section's file gone: the job is read, the picture refused.
    shutil.move(job.parent / ID0, job.parent.parent / ID0)
    moved = job.parent.parent / "elsewhere"
    shutil.copytree(job, moved / "langslice")
    record = json.loads((moved / "langslice" / "job.json").read_text())
    record["image_folder"] = str(job.parent.parent / "gone")
    (moved / "langslice" / "job.json").write_text(json.dumps(record))
    # The images moved away: every render says what NoJob says.
    away = _renderer(moved / "langslice", pool)
    for call in (away.strip, lambda: away.overlay(ID1, 128), lambda: away.thumb(ID1, 128)):
        with pytest.raises(RenderError, match="are not in .* any more"):
            call()
    # A section whose file is gone is refused with the reason, the others drawn.
    with pytest.raises(RenderError, match="could not be drawn"):
        renderer.thumb(ID0, 96)
    assert _image(renderer.thumb(ID1, 96)).size[0] > 0
    # No state yet: nothing to draw.
    (job / "state.json").unlink()
    with pytest.raises(RenderError, match="no state"):
        _renderer(job, pool).overlay(ID1, 128)


def test_the_workspace_keeps_a_page_of_sections(
        job: Path, pool: ThreadPoolExecutor, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render, "KEPT_SOURCES", 1)
    monkeypatch.setattr(render, "KEPT_RENDERS", 2)
    renderer = _renderer(job, pool)
    for section in (ID0, ID1, ID2):
        renderer.overlay(section, 256)
        renderer.thumb(section, 64)
    opened = renderer._opened
    assert opened is not None
    workspace = opened.workspace
    assert len(workspace.source_cache) <= 1 and len(workspace.render_cache) <= 2
    assert len(workspace.render_scale) <= 2


def test_jobs_not_viewed_lately_let_go_of_their_workspace(
        made: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render, "KEPT_JOBS", 1)
    first, second = tmp_path / "a" / "stack", tmp_path / "b" / "stack"
    for images in (first, second):
        shutil.copytree(made.parent, images)
    jobs = RenderPool(atlas_loader=LOADER)
    try:
        viewed = jobs.renderer(first / "langslice", JobWatcher(first / "langslice").current_stack)
        picture = viewed.thumb(ID1, 64)
        assert viewed._opened is not None
        jobs.renderer(second / "langslice", JobWatcher(second / "langslice").current_stack)
        assert viewed._opened is None  # released
        again = jobs.renderer(first / "langslice", JobWatcher(first / "langslice").current_stack)
        assert again is viewed and again.thumb(ID1, 64) == picture  # read again
    finally:
        jobs.close()

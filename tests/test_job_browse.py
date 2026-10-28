"""Read-only browsing of the job folder (``job.browse``, ``ops.files``): paths kept
inside the folder, capped answers, picture files answered with their index record."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from langslice.agent.engine import build_context
from langslice.core.spec import JobSpec
from langslice.job import browse
from langslice.job.job import Job
from langslice.ops import files
from langslice.ops.refusal import Refused
from tests.fakes import SlabAtlas

_ATLAS = SlabAtlas()


def _open(tmp_path: Path, disabled: list[str] | None = None) -> Job:
    images = tmp_path / "images"
    images.mkdir(parents=True)
    Image.fromarray(np.full((30, 40, 3), 90, dtype=np.uint8)).save(images / "s0.png")
    spec = JobSpec(image_folder=str(images), model="fake-model", preprocess="none",
                   disabled=list(disabled or []))
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: _ATLAS)
    return Job.open(spec, ctx, folder=ctx.job_folder, results_path=ctx.results_path)


def _refused(call: Any, *args: Any, code: str = "BAD_PATH", **kwargs: Any) -> Refused:
    with pytest.raises(Refused) as caught:
        call(*args, **kwargs)
    assert caught.value.code == code
    return caught.value


def test_paths_outside_the_job_folder_are_refused(tmp_path: Path):
    job = _open(tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_text("outside")
    for bad in ("..", "../secret.txt", "../../etc/passwd", str(secret), "/etc/passwd",
                "history/../../secret.txt"):
        _refused(files.read_file, job, bad)
        _refused(files.list_files, job, bad)
        _refused(files.search_files, job, "outside", bad)
    missing = _refused(files.read_file, job, "history/no_such_file.txt", code="NOT_FOUND")
    # A path that does not exist names the nearest folder above it and what it holds.
    assert missing.details["nearest"] == "history"
    assert "the nearest folder above it that does, 'history', holds:" in str(
        missing.details["message"])
    deep = _refused(files.list_files, job, "sections/s0/deformable/x", code="NOT_FOUND")
    assert deep.details["nearest"] == "sections/s0"
    assert deep.details["holds"] == sorted(deep.details["holds"], key=lambda name: (
        not name.endswith("/"), name))


@pytest.mark.skipif(os.name == "nt", reason="symlinks")
def test_symlinks_out_of_the_folder_are_refused_and_not_walked(tmp_path: Path):
    job = _open(tmp_path)
    folder = job.layout.folder
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "leak.txt").write_text("LEAKED needle")
    (folder / "link_file").symlink_to(outside / "leak.txt")
    (folder / "link_dir").symlink_to(outside, target_is_directory=True)
    (folder / "inside.txt").write_text("needle inside")
    (folder / "alias").symlink_to(folder / "inside.txt")

    _refused(files.read_file, job, "link_file")
    _refused(files.read_file, job, "link_dir/leak.txt")
    _refused(files.list_files, job, "link_dir")
    listed = files.list_files(job).text
    assert "link_file" not in listed and "link_dir" not in listed
    assert "inside.txt" in listed
    found = files.search_files(job, "needle")
    assert "LEAKED" not in found.text
    assert any(line.startswith("inside.txt:1:") for line in found.matches)
    assert "needle inside" in files.read_file(job, "alias").text


def test_listing_search_and_reading_are_capped_and_paged(tmp_path: Path):
    job = _open(tmp_path)
    folder = job.layout.folder
    listed = files.list_files(job)
    assert listed.text.index("history/") < listed.text.index("state.json")  # folders first
    only = files.list_files(job, glob="*.json")
    assert only.entries and all(e.path.endswith(".json") for e in only.entries)
    many = folder / "many"
    many.mkdir()
    for index in range(browse.LIST_ENTRIES + 7):
        (many / f"f{index:04d}.txt").write_text("x")
    capped = files.list_files(job, "many")
    assert len(capped.entries) == browse.LIST_ENTRIES and capped.hidden == 7

    (folder / "a.txt").write_text("alpha\nBeta two\nalpha again\n")
    (folder / "b.log").write_text("alpha in a log\n")
    (folder / "bin.dat").write_bytes(b"alpha\x00\x00")
    found = files.search_files(job, "^alpha", glob="*.txt")
    assert found.matches == ("a.txt:1: alpha", "a.txt:3: alpha again")
    assert files.search_files(job, "beta").matches == ("a.txt:2: Beta two",)
    assert files.search_files(job, "alpha(", glob="*.txt").matches == ()  # invalid regex: plain
    assert not files.search_files(job, "no-such-text-anywhere").matches
    assert not any(m.startswith("bin.dat") for m in files.search_files(job, "alpha").matches)
    with pytest.raises(Refused) as caught:
        files.search_files(job, "")
    assert caught.value.code == "BAD_ARGS"
    (folder / "big.txt").write_text("hit\n" * (browse.SEARCH_MATCHES + 15))
    big = files.search_files(job, "hit", path="big.txt")
    assert len(big.matches) == browse.SEARCH_MATCHES and big.hidden == 15

    (folder / "long.txt").write_text("".join(f"line {n}\n" for n in range(1, 1001)))
    first = files.read_file(job, "long.txt", limit=3)
    assert "     1\tline 1" in first.text
    nxt = files.read_file(job, "long.txt", offset=3, limit=2)
    assert (nxt.first_line, nxt.last_line, nxt.total_lines) == (4, 5, 1000)
    assert files.read_file(job, "long.txt").last_line == browse.READ_LINES
    (folder / "heavy.txt").write_text(("z" * 300 + "\n") * 1000)
    heavy = files.read_file(job, "heavy.txt", limit=1000)
    assert len(heavy.text) < browse.TEXT_BYTES + 2000
    (folder / "blob.bin").write_bytes(bytes(range(256)) * 4)
    blob = files.read_file(job, "blob.bin")
    assert blob.kind == "binary" and "\x01" not in blob.text
    _refused(files.read_file, job, "history")


def test_picture_files_answer_with_their_index_record(tmp_path: Path):
    job = _open(tmp_path)
    job.views.save(tool="zoom", pictures=[(Image.new("RGB", (20, 10), (200, 10, 10)), None)],
                   captions=["A red test picture"])
    job.views.flush()
    record = job.views.lookup(1)
    assert record is not None and record.path
    reply = files.read_file(job, f"{record.path}/view.jpg")
    assert reply.kind == "picture" and reply.picture == record.seq
    assert "A red test picture" in reply.text
    assert "JFIF" not in reply.text  # no pixels, no bytes
    assert files.read_file(job, f"{record.path}/view.json").kind == "text"

    stray = job.layout.folder / "stray.png"
    Image.new("RGB", (4, 4)).save(stray)
    assert files.read_file(job, "stray.png").picture is None


def _look_recipe(**args: Any) -> dict[str, Any]:
    return {"renderer": "look", "args": {
        "mode": "overlay", "sections": ["s0.png"], "positions_mm": [], "images": [],
        "atlas_layers": None, "atlas_opacity": None, "warp": "applied", "long_edge": None,
        "zoom": [], "part": None, "highlight": [], "depth": 6, **args}}


def _saved(job: Job, tool: str, recipe: dict[str, Any] | None) -> str:
    job.views.save(tool=tool, pictures=[(Image.new("RGB", (20, 10)), None)],
                   recipes=[recipe], captions=["a picture"])
    job.views.flush()
    record = job.views.latest(exclude_tool=None)
    assert record is not None and record.path
    return f"{record.path}/view.jpg"


def test_a_picture_is_seen_again_with_zoom_or_drawn_again_with_look(tmp_path: Path):
    """With zoom, the picture's number; without it, the look call that draws a
    look picture's view again on the stack as it is now; else nothing."""
    with_zoom = _open(tmp_path / "zoom")
    shown = files.read_file(with_zoom, _saved(with_zoom, "look", _look_recipe())).text
    assert "call zoom with picture 1" in shown and "call look" not in shown

    job = _open(tmp_path / "nozoom", disabled=["zoom"])
    looked = files.read_file(job, _saved(job, "look", _look_recipe(
        registrations=[["s0.png", 2, False]], masks=["tissue"]))).text
    assert "call zoom" not in looked
    assert ("To draw its view again, call look with mode='overlay', sections=['s0.png'], "
            "masks=['tissue'], registrations={'s0.png': 2} (it draws the stack as it is now)."
            ) in looked
    zoomed = files.read_file(job, _saved(job, "interactive_transform", _look_recipe(
        zoom=[0.1, 0.1, 0.5, 0.5]))).text
    assert "call look with mode='overlay', sections=['s0.png'] (it draws the stack as it " \
        "is now, and the whole picture, unzoomed)." in zoomed
    atlas = files.read_file(job, _saved(job, "look", _look_recipe(
        mode="atlas", sections=[], positions_mm=[4.5]))).text
    assert "call look with mode='atlas', positions_mm=[4.5]" in atlas
    # A picture look did not draw (a crop, a sheet, no recipe) gets no line.
    for recipe in ({"renderer": "crop", "source": 1, "box": [0, 0, 5, 5]}, None):
        other = files.read_file(job, _saved(job, "zoom", recipe)).text
        assert "To draw" not in other and "To see" not in other
    # A look mode the job switches off is not offered either.
    no_atlas = _open(tmp_path / "noatlas", disabled=["zoom", "look:atlas"])
    text = files.read_file(no_atlas, _saved(no_atlas, "look", _look_recipe(
        mode="atlas", sections=[], positions_mm=[4.5]))).text
    assert "call look" not in text and "To draw" not in text


@pytest.mark.skipif(os.name == "nt", reason="symlinks")
def test_the_agent_transcripts_are_neither_listed_searched_nor_read(tmp_path: Path):
    job = _open(tmp_path)
    folder = job.layout.folder
    trace = job.layout.trace_dir
    (trace / "deep").mkdir(parents=True)
    (trace / "session.jsonl").write_text("needle in the transcript\n")
    (trace / "deep" / "more.jsonl").write_text("needle deeper\n")
    (folder / "notes.txt").write_text("needle outside\n")
    (folder / "into_trace").symlink_to(trace / "session.jsonl")

    listed = files.list_files(job)
    assert "trace" not in listed.text and "into_trace" not in listed.text
    assert "notes.txt" in listed.text
    every = files.list_files(job, glob="*")
    assert not any(e.path.startswith("trace") or e.path == "into_trace"
                   for e in every.entries)
    found = files.search_files(job, "needle")
    assert found.matches == ("notes.txt:1: needle outside",)
    for path in ("trace", "trace/", "./trace/session.jsonl", "trace/deep/more.jsonl",
                 "trace/no_such_file", "history/../trace/session.jsonl",
                 str(trace / "session.jsonl"), "into_trace"):
        for call, args in ((files.read_file, (path,)), (files.list_files, (path,)),
                           (files.search_files, ("needle", path))):
            refused = _refused(call, job, *args)
            assert "the agent transcripts, which these tools do not open" in str(
                refused.details["message"])
    # A missing path's refusal does not name the folder among the job folder's names.
    missing = _refused(files.read_file, job, "no_such_file", code="NOT_FOUND")
    assert "trace/" not in missing.details["holds"]

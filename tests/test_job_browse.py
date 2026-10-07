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
from langslice.ops import files, notes
from langslice.ops.refusal import Refused
from tests.fakes import SlabAtlas

_ATLAS = SlabAtlas()


def _open(tmp_path: Path) -> Job:
    images = tmp_path / "images"
    images.mkdir()
    Image.fromarray(np.full((30, 40, 3), 90, dtype=np.uint8)).save(images / "s0.png")
    spec = JobSpec(image_folder=str(images), model="fake-model", preprocess="none")
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: _ATLAS)
    return Job.open(spec, ctx, folder=ctx.job_folder, results_path=ctx.results_path)


def _refused(call: Any, *args: Any, **kwargs: Any) -> Refused:
    with pytest.raises(Refused) as caught:
        call(*args, **kwargs)
    assert caught.value.code == "BAD_PATH"
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
    _refused(files.read_file, job, "no_such_file.txt")


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


def test_list_files_names_kinds_and_sizes_and_caps(tmp_path: Path):
    job = _open(tmp_path)
    folder = job.layout.folder
    listed = files.list_files(job)
    assert "state.json" in listed.text and "history/" in listed.text
    assert listed.text.index("history/") < listed.text.index("state.json")  # folders first
    only = files.list_files(job, pattern="*.json")
    assert all(e.path.endswith(".json") for e in only.entries) and only.entries

    many = folder / "many"
    many.mkdir()
    for index in range(browse.LIST_ENTRIES + 7):
        (many / f"f{index:04d}.txt").write_text("x")
    capped = files.list_files(job, "many")
    assert len(capped.entries) == browse.LIST_ENTRIES and capped.hidden == 7
    assert capped.text.splitlines()[-1].startswith("... 7 more")
    (folder / "blob.bin").write_bytes(b"\x00\x01\x02")
    assert "blob.bin  binary, 3 B" in files.list_files(job).text


def test_search_files_regex_plain_text_glob_and_cap(tmp_path: Path):
    job = _open(tmp_path)
    folder = job.layout.folder
    (folder / "a.txt").write_text("alpha\nBeta two\nalpha again\n")
    (folder / "b.log").write_text("alpha in a log\n")
    (folder / "bin.dat").write_bytes(b"alpha\x00\x00")

    found = files.search_files(job, "^alpha", glob="*.txt")
    assert found.matches == ("a.txt:1: alpha", "a.txt:3: alpha again")
    assert files.search_files(job, "beta").matches == ("a.txt:2: Beta two",)
    assert files.search_files(job, "alpha(", glob="*.txt").matches == ()  # invalid regex: plain
    assert files.search_files(job, "no-such-text-anywhere").text.startswith("No match")
    assert not any(m.startswith("bin.dat") for m in files.search_files(job, "alpha").matches)
    with pytest.raises(Refused) as caught:
        files.search_files(job, "")
    assert caught.value.code == "BAD_ARGS"

    (folder / "big.txt").write_text("hit\n" * (browse.SEARCH_MATCHES + 15))
    big = files.search_files(job, "hit", path="big.txt")
    assert len(big.matches) == browse.SEARCH_MATCHES and big.hidden == 15
    assert "15 more match(es) not shown" in big.text


def test_read_file_numbers_lines_pages_and_caps(tmp_path: Path):
    job = _open(tmp_path)
    folder = job.layout.folder
    (folder / "long.txt").write_text("".join(f"line {n}\n" for n in range(1, 1001)))
    first = files.read_file(job, "long.txt", limit=3)
    assert first.text.splitlines()[0] == "long.txt: lines 1-3 of 1000"
    assert "     1\tline 1" in first.text and "continue with offset=3" in first.text
    nxt = files.read_file(job, "long.txt", offset=3, limit=2)
    assert (nxt.first_line, nxt.last_line, nxt.total_lines) == (4, 5, 1000)
    default = files.read_file(job, "long.txt")
    assert default.last_line == browse.READ_LINES
    assert files.read_file(job, "long.txt", offset=5000).text.startswith("long.txt has 1000")

    (folder / "wide.txt").write_text("y" * 5000 + "\n")
    assert "line cut, 5000 characters" in files.read_file(job, "wide.txt").text
    (folder / "heavy.txt").write_text(("z" * 300 + "\n") * 1000)
    heavy = files.read_file(job, "heavy.txt", limit=1000)
    assert len(heavy.text) < browse.TEXT_BYTES + 2000 and "more line(s)" in heavy.text
    (folder / "blob.bin").write_bytes(bytes(range(256)) * 4)
    blob = files.read_file(job, "blob.bin")
    assert blob.kind == "binary" and "binary file" in blob.text and "\x01" not in blob.text
    _refused(files.read_file, job, "history")


def test_index_and_notes_are_discoverable_and_readable(tmp_path: Path):
    job = _open(tmp_path)
    notes.add_note(job, "the cortex looks thin here")
    job.views.save(tool="zoom", pictures=[(Image.new("RGB", (20, 10), (200, 10, 10)), None)],
                   captions=["A red test picture"])
    job.views.flush()
    assert "views.jsonl" in files.list_files(job).text
    index = files.read_file(job, "views.jsonl")
    assert '"seq": 1' in index.text or '"seq":1' in index.text
    assert "A red test picture" in index.text
    assert files.search_files(job, "red test", glob="views.jsonl").matches
    assert files.search_files(job, "cortex looks thin", glob="state.json").matches


def test_picture_files_answer_with_their_index_record(tmp_path: Path):
    job = _open(tmp_path)
    job.views.save(tool="zoom", pictures=[(Image.new("RGB", (20, 10), (200, 10, 10)), None)],
                   captions=["A red test picture"])
    job.views.flush()
    record = job.views.lookup(1)
    assert record is not None and record.path
    reply = files.read_file(job, f"{record.path}/view.jpg")
    assert reply.kind == "picture" and reply.picture == record.seq
    assert "Picture 1" in reply.text and "A red test picture" in reply.text
    assert "call zoom with picture 1" in reply.text
    assert "20 x 10 pixels" in reply.text
    assert "JFIF" not in reply.text  # no pixels, no bytes

    listed = files.list_files(job, record.path).text
    assert "view.jpg  picture" in listed and "view.json  text" in listed
    assert files.read_file(job, f"{record.path}/view.json").kind == "text"

    stray = job.layout.folder / "stray.png"
    Image.new("RGB", (4, 4)).save(stray)
    assert "no entry in the picture index" in files.read_file(job, "stray.png").text

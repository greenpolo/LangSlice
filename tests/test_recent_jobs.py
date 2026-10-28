"""The recent jobs list (``~/.langslice/recent_jobs.jsonl``) and the
``logs/events.jsonl`` argument cap."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from langslice.core.spec import JobSpec
from langslice.job import recent
from langslice.job.job import Job
from tests.test_linear_engine import _ctx, _make_stack


def test_every_open_adds_the_job_to_the_list_in_a_private_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv(recent.RECENT_ENV, raising=False)
    images = tmp_path / "stack"
    images.mkdir()
    _make_stack(images, n=2)
    spec = JobSpec(image_folder=str(images), model="fake-model", preprocess="none")
    Job.open(spec, _ctx(spec))
    Job.open(JobSpec.from_dict({**spec.to_dict(), "resume": True}), _ctx(spec))

    listed = home / ".langslice" / "recent_jobs.jsonl"
    entries = [json.loads(line) for line in listed.read_text().splitlines()]
    assert len(entries) == 2
    assert entries[0]["folder"] == str(images / "langslice")
    assert entries[0]["image_folder"] == str(images)
    assert entries[0]["atlas"] == spec.atlas and entries[0]["at"]
    assert recent.read(listed) == entries


def test_the_list_can_be_moved_or_switched_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    elsewhere = tmp_path / "lists" / "recent.jsonl"
    monkeypatch.setenv(recent.RECENT_ENV, str(elsewhere))
    recent.remember(tmp_path / "a" / "langslice", tmp_path / "a", "allen_mouse_25um")
    assert [entry["folder"] for entry in recent.read()] == [str(tmp_path / "a" / "langslice")]
    monkeypatch.setenv(recent.RECENT_ENV, "off")
    assert recent.recent_file() is None
    recent.remember(tmp_path / "b", None, None)  # nothing, no error
    assert len(recent.read(elsewhere)) == 1


def test_a_long_list_keeps_the_newest_line_of_each_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    listed = tmp_path / "recent.jsonl"
    monkeypatch.setenv(recent.RECENT_ENV, str(listed))
    monkeypatch.setattr(recent, "KEEP_BYTES", 1)  # every append compacts
    monkeypatch.setattr(recent, "KEEP_JOBS", 3)
    for round_ in range(3):
        for name in "abcd":
            recent.remember(tmp_path / name, None, f"atlas{round_}")
    recent.remember(tmp_path / "b", None, "atlas3")
    entries = recent.read(listed)
    assert [Path(entry["folder"]).name for entry in entries] == ["c", "d", "b"]
    assert [entry["atlas"] for entry in entries] == ["atlas2", "atlas2", "atlas3"]


def test_event_arguments_are_capped():
    from langslice.doors.tools.toolbox import EVENT_ARGS_BYTES, _event_args

    small = {"sections": ["s1.png"]}
    assert _event_args(small) == {"args": small}
    big = {"pairs": [{"atlas": [index, index], "section": [index, index]}
                     for index in range(500)]}
    capped = _event_args(big)
    assert capped["args_truncated"] is True and isinstance(capped["args"], str)
    assert len(capped["args"].encode()) <= EVENT_ARGS_BYTES

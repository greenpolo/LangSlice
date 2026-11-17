"""A coding agent's words from Claude Code's transcript
(``langslice.hosts.gui.transcripts``): the transcript found by its
``start_job`` call, tailed as it grows, and shown by the watcher for a job
folder without a trace."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from langslice.hosts.gui import transcripts
from langslice.hosts.gui.transcripts import TranscriptWords, folder_key
from langslice.hosts.gui.watch import JobWatcher

NOW = time.time()


def _turn(*blocks: dict[str, Any], at: str = "2026-01-02T03:04:05.000Z") -> dict[str, Any]:
    return {"type": "assistant", "timestamp": at, "isSidechain": False,
            "message": {"model": "claude-opus-4-1-20250805", "content": list(blocks)}}


def _start(folder: Path) -> dict[str, Any]:
    return {"type": "tool_use", "name": "mcp__plugin_langslice_langslice__start_job",
            "input": {"image_folder": str(folder)}}


def _line(record: dict[str, Any]) -> str:
    """One line as Claude Code writes it: compact JSON."""
    return json.dumps(record, separators=(",", ":"))


def _write(path: Path, *records: dict[str, Any], end: str = "\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(_line(r) for r in records) + end)


def _job(images: Path) -> Path:
    """A job folder whose first tool call was made now."""
    folder = images / "langslice"
    moment = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    _write(folder / "logs" / "events.jsonl",
           {"kind": "tool_start", "name": "look", "execution_id": "e1", "at": moment,
            "door": "mcp", "target_ids": []})
    return folder


def _words(root: Path | None) -> TranscriptWords:
    ticks = iter(range(0, 10_000, 10))
    return TranscriptWords(root, clock=lambda: float(next(ticks)))


def test_names_and_times() -> None:
    assert transcripts.model_name("claude-opus-4-1-20250805") == "Claude Opus 4.1"
    assert transcripts.model_name("claude-fable-5-2") == "Claude Fable 5.2"
    assert transcripts.model_name("gpt-5") == "gpt-5"
    assert folder_key("C:\\Data\\Brain1\\") == folder_key("c:/data/brain1")
    assert transcripts.local_time("not a time") == "not a time"
    local = transcripts.local_time("2026-01-02T03:04:05.678Z")
    assert len(local) == len("2026-01-02T03:04:05") and "Z" not in local


def test_the_transcript_is_found_by_its_start_job_and_tailed(tmp_path: Path) -> None:
    images = tmp_path / "brain" / "images"
    folder = _job(images)
    root = tmp_path / "projects"
    other = root / "p1" / "aaaa.jsonl"
    ours = root / "p2" / "bbbb" / "subagents" / "agent-1.jsonl"
    _write(other, _turn(_start(tmp_path / "elsewhere")),
           _turn({"type": "text", "text": "Not this job."}))
    _write(ours, {"type": "api-request-blob", "blob": "start_job " + "x" * 50},
           _turn({"type": "text", "text": "Before the job."}),
           _turn({"type": "thinking", "thinking": "", "signature": "s"}, _start(images)),
           _turn({"type": "thinking", "thinking": "The pitch looks tilted.", "signature": "s"}),
           _turn({"type": "text", "text": "Placing the sections."}))
    words = _words(root)
    records = words.read(folder, None, NOW)
    assert [r["kind"] for r in records] == ["session", "model", "model"]
    assert records[0]["model"] == "Claude Opus 4.1" and records[0]["agent"] == "Claude Code"
    assert records[1]["thought"] == "The pitch looks tilted."
    assert records[2]["text"] == "Placing the sections."
    assert words.read(folder, None, NOW) == []
    half = _line(_turn({"type": "text", "text": "Fitting."}))
    _write(ours, {"type": "user"}, end="\n" + half[:30])
    assert words.read(folder, None, NOW) == []
    with ours.open("a", encoding="utf-8", newline="") as handle:
        handle.write(half[30:] + "\n")
    assert [r.get("text") for r in words.read(folder, None, NOW)] == ["Fitting."]


def test_an_old_transcript_is_not_searched_and_off_reads_nothing(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    images = tmp_path / "images"
    folder = _job(images)
    root = tmp_path / "projects"
    old = root / "p" / "old.jsonl"
    _write(old, _turn(_start(images)), _turn({"type": "text", "text": "Long ago."}))
    stale = NOW - 10 * 24 * 3600
    os.utime(old, (stale, stale))
    assert _words(root).read(folder, None, NOW) == []
    os.utime(old, (NOW, NOW))
    assert _words(root).read(tmp_path / "none" / "langslice", None, NOW) == []
    assert _words(None).read(folder, None, NOW) == []
    monkeypatch.setenv(transcripts.ENV, "off")
    assert transcripts.projects_root() is None
    monkeypatch.setenv(transcripts.ENV, str(root))
    assert transcripts.projects_root() == root


def test_the_watcher_shows_a_coding_agents_words(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    images = tmp_path / "images"
    folder = _job(images)
    root = tmp_path / "projects"
    monkeypatch.setenv(transcripts.ENV, str(root))
    _write(root / "p" / "s.jsonl", _turn(_start(images)),
           _turn({"type": "text", "text": "Reading the stack."}))
    watcher = JobWatcher(folder)
    watcher.poll(force=True)
    snap = watcher.snapshot()
    assert [i["text"] for i in snap["items"] if i["kind"] == "agent"] == ["Reading the stack."]
    assert snap["header"]["model"] == "Claude Opus 4.1"
    assert "replay" not in snap["header"]

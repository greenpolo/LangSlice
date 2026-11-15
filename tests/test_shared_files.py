"""Files and processes on every system: retried replaces and locks, LF job
files, the GPU queue, detached runs and their cancelling."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import threading
import time

import pytest

from langslice.core import shared_files
from langslice.core.fits.deformable import engines
from langslice.doors.agent_cli import background
from langslice.job import index, lock, recent
from langslice.job.layout import JobLayout
from langslice.job.state_file import write_json_atomic


def test_a_permission_error_is_tried_again_then_raised(monkeypatch):
    calls = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError(13, "in use")
        return "done"

    assert shared_files.retried(flaky) == "done" and len(calls) == 3

    monkeypatch.setattr(shared_files, "RETRY_S", 0.05)

    def denied() -> None:
        raise PermissionError(13, "denied")

    started = time.monotonic()
    with pytest.raises(PermissionError):
        shared_files.retried(denied)
    assert time.monotonic() - started < 2


def test_a_replace_waits_for_a_reader_to_let_go(tmp_path, monkeypatch):
    real = os.replace
    tries = []

    def busy_once(source, target):
        tries.append(1)
        if len(tries) == 1:
            raise PermissionError(13, "the target is open")
        real(source, target)

    monkeypatch.setattr(shared_files.os, "replace", busy_once)
    write_json_atomic(str(tmp_path / "state.json"), {"a": 1})
    saved = (tmp_path / "state.json").read_text(encoding="utf-8")
    assert json.loads(saved) == {"a": 1}
    assert len(tries) == 2


def test_a_tree_with_read_only_files_is_removed(tmp_path):
    folder = tmp_path / "env" / "lib"
    folder.mkdir(parents=True)
    frozen = folder / "frozen.pyc"
    frozen.write_bytes(b"x")
    frozen.chmod(stat.S_IREAD)
    shared_files.remove_tree(tmp_path / "env")
    assert not (tmp_path / "env").exists()


def test_job_files_are_written_with_lf_line_ends(tmp_path, monkeypatch):
    write_json_atomic(str(tmp_path / "state.json"), {"a": [1, 2]})
    assert b"\r\n" not in (tmp_path / "state.json").read_bytes()

    listed = tmp_path / "recent.jsonl"
    monkeypatch.setenv(recent.RECENT_ENV, str(listed))
    recent.remember(tmp_path, tmp_path, "test")
    assert listed.read_bytes().endswith(b"}\n")
    assert b"\r\n" not in listed.read_bytes()

    entry = index.register(tmp_path / "jobs", "0123456789ab", tmp_path)
    assert b"\r\n" not in entry.read_bytes()


class _FlakyLock:
    """A file lock refused *refusals* times (another process deleting it)."""

    def __init__(self, refusals: int) -> None:
        self.refusals = refusals
        self.held = False

    def acquire(self, timeout: float | None = None) -> None:
        if self.refusals > 0:
            self.refusals -= 1
            raise PermissionError(13, "delete pending")
        self.held = True

    def release(self) -> None:
        self.held = False


def test_a_lock_file_refused_for_a_moment_is_taken(tmp_path, monkeypatch):
    folder_lock = lock.FolderLock(tmp_path)
    flaky = _FlakyLock(refusals=2)
    monkeypatch.setattr(folder_lock, "_file_lock", lambda: flaky)
    with folder_lock.held():
        assert flaky.held
    assert not flaky.held and not folder_lock._unusable


def test_only_an_unwritable_folder_leaves_the_lock_off(tmp_path, monkeypatch):
    monkeypatch.setattr(shared_files, "RETRY_S", 0.05)
    folder_lock = lock.FolderLock(tmp_path)
    refused = _FlakyLock(refusals=10**6)
    monkeypatch.setattr(folder_lock, "_file_lock", lambda: refused)
    with folder_lock.held():
        pass
    assert not folder_lock._unusable  # the folder is writable: tried again

    monkeypatch.setattr(lock, "writable", lambda folder, image_folder: False)
    with folder_lock.held():
        pass
    assert folder_lock._unusable


def test_gpu_fits_take_turns_on_every_system(tmp_path, monkeypatch):
    monkeypatch.setattr(engines, "GPU_LOCK", tmp_path / "gpu.lock")
    waited: list[float] = []

    def second_fit() -> None:
        with engines.gpu_turn("cuda") as turn:
            waited.append(turn.waited_s)

    with engines.gpu_turn("cuda") as first:
        assert first.waited_s < 1
        other = threading.Thread(target=second_fit)
        other.start()
        time.sleep(0.4)
    other.join(timeout=10)
    assert waited and waited[0] >= 0.3
    with engines.gpu_turn("cpu") as cpu:
        assert cpu.waited_s == 0


#: A background run's process: it says it runs, then waits to be cancelled.
_RUN = """
import sys, time
from pathlib import Path
from langslice.doors.agent_cli import background
from langslice.job.layout import JobLayout
folder = Path(sys.argv[1])
background.begin(JobLayout(folder), sys.argv[2])
try:
    (folder / "ready").write_text("yes", encoding="utf-8")
    time.sleep(60)
except KeyboardInterrupt:
    (folder / "cleaned_up").write_text("yes", encoding="utf-8")
"""


def test_a_cancelled_run_ends(tmp_path):
    """Cancelling ends the run; on Windows through its own cleanup (a
    KeyboardInterrupt), as SIGTERM cannot be caught there."""
    layout = JobLayout(tmp_path)
    run = subprocess.Popen(
        [sys.executable, "-c", _RUN, str(tmp_path), "run-1"],
        stdin=subprocess.DEVNULL,
        creationflags=background.DETACHED.creationflags,
        start_new_session=background.DETACHED.start_new_session,
    )
    deadline = time.monotonic() + 60
    while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    background.cancel(layout, "run-1")
    assert run.wait(timeout=30) is not None
    if background.WINDOWS:
        cleaned = (tmp_path / "cleaned_up").read_text(encoding="utf-8")
        assert cleaned == "yes"
        assert not background.cancel_path(layout, "run-1").exists()

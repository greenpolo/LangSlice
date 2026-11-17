"""Background runs of the agent CLI: ``--background``, ``runs [ID]``, ``wait [ID]``.

A long verb (``Verb.long`` in :mod:`langslice.tools.catalog`: the fits, the
traces, ``export_maps``) answers at once with a run id when given
``--background``: the same command is started again as a detached process
(:data:`CHILD_COMMAND`) that runs the verb and saves its envelope. Each run is
one record, ``logs/runs/<id>.json`` in the job folder (``id``, ``verb``,
``arguments``, ``state`` running/finished/lost, ``started_at``, ``pid``, and
once finished ``finished_at``, ``exit`` and the ``envelope``), with the
process's stderr in ``logs/runs/<id>.log``. A run whose process is gone without
an answer reads as ``lost``.

Where a process may outlive the command that started it
(:func:`detaching`), a long verb called without ``--background`` runs the
same way: the call starts the run and waits for it, at most :data:`WAIT_S`,
so a harness that ends the command (Codex ends commands its agent has left
running) ends only the waiting; the run goes on and ``wait`` collects it.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langslice.job.folder.layout import JobLayout
from langslice.job.state_file import write_json_atomic

CHILD_COMMAND: list[str] = [
    sys.executable,
    "-m",
    "langslice.doors.agent_cli.main",
]
START_GRACE_S = 120.0
POLL_S = 0.25
WAIT_S = 30.0
DETACH_ENV = "LANGSLICE_DETACH"


def detaching() -> bool:
    """Whether a long verb called without ``--background`` runs detached
    here (:data:`DETACH_ENV`)."""
    value = os.environ.get(DETACH_ENV, "").strip().lower()
    if value in ("0", "false", "no", "off"):
        return False
    if value in ("1", "true", "yes", "on"):
        return True
    return not os.environ.get("CLAUDECODE")


def runs_dir(layout: JobLayout) -> Path:
    return layout.logs_dir / "runs"


def record_path(layout: JobLayout, run_id: str) -> Path:
    return runs_dir(layout) / f"{run_id}.json"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write(layout: JobLayout, record: dict[str, Any]) -> None:
    runs_dir(layout).mkdir(parents=True, exist_ok=True)
    write_json_atomic(str(record_path(layout, record["id"])), record)


def _read(layout: JobLayout, run_id: str) -> dict[str, Any] | None:
    try:
        data = json.loads(
            record_path(layout, run_id).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


WINDOWS = os.name == "nt"
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_ERROR_ACCESS_DENIED = 5


def _kernel32() -> Any:
    import ctypes

    return ctypes.WinDLL("kernel32", use_last_error=False)


def _alive_windows(pid: int) -> bool:
    """Whether process *pid* runs, on Windows: psutil when installed, else
    ``OpenProcess`` + ``GetExitCodeProcess`` (a process that exists but is
    not ours to open, access denied, counts as running)."""
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        return bool(psutil.pid_exists(pid))
    import ctypes

    kernel = _kernel32()
    handle = kernel.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return kernel.GetLastError() == _ERROR_ACCESS_DENIED
    try:
        code = ctypes.c_ulong(0)
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        return code.value == _STILL_ACTIVE
    finally:
        kernel.CloseHandle(handle)


def _alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    if WINDOWS:
        return _alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


_DETACHED_FLAGS = 0x00000200 | 0x08000000


@dataclass(frozen=True)
class Detached:
    """The ``subprocess.Popen`` options of a process that runs on its own.

    It runs past the command that started it: a session of its own on
    Linux and macOS; on Windows a process group and a windowless console
    of its own.
    """

    creationflags: int
    start_new_session: bool


_CREATION_FLAGS = _DETACHED_FLAGS if WINDOWS else 0
DETACHED = Detached(
    creationflags=_CREATION_FLAGS, start_new_session=not WINDOWS
)


def start(
    layout: JobLayout, verb: str, arguments: dict[str, Any], *, verbose: bool
) -> str:
    """Start *verb* on the job in a detached process; its run id."""
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
    log = runs_dir(layout) / f"{run_id}.log"
    _write(
        layout,
        {
            "id": run_id,
            "verb": verb,
            "arguments": arguments,
            "state": "running",
            "started_at": _now(),
            "started": time.time(),
            "pid": None,
            "log": layout.relative(log),
        },
    )
    command = [
        *CHILD_COMMAND,
        str(layout.folder),
        verb,
        "--args",
        json.dumps(arguments),
        "--run-id",
        run_id,
        *(["--verbose"] if verbose else []),
    ]
    with log.open("ab") as out:
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=out,  # noqa: S603
            stderr=subprocess.STDOUT,
            creationflags=DETACHED.creationflags,
            start_new_session=DETACHED.start_new_session,
        )
    return run_id


def begin(layout: JobLayout, run_id: str) -> None:
    """In the background process: say it runs (its pid).

    On Windows it also watches for :func:`cancel`'s request.
    """
    record = _read(layout, run_id) or {
        "id": run_id,
        "started_at": _now(),
        "started": time.time(),
    }
    if record.get("state") in (None, "running"):
        _write(layout, {**record, "state": "running", "pid": os.getpid()})
    if WINDOWS:
        threading.Thread(
            target=_interrupt_on_request,
            args=(cancel_path(layout, run_id),),
            name="langslice-cancel",
            daemon=True,
        ).start()


def cancel_path(layout: JobLayout, run_id: str) -> Path:
    """Name the file :func:`cancel` writes to end a run on Windows.

    Parameters
    ----------
    layout
        The job folder's layout the run belongs to.
    run_id
        The background run's id.

    Returns
    -------
    Path
        ``logs/runs/<id>.cancel``.
    """
    return runs_dir(layout) / f"{run_id}.cancel"


def _interrupt_on_request(path: Path) -> None:
    """Raise ``KeyboardInterrupt`` in the main thread once *path* exists.

    The run then ends through its own cleanup, as Ctrl-C ends it.
    """
    while not path.exists():
        time.sleep(POLL_S)
    path.unlink(missing_ok=True)
    signal.raise_signal(signal.SIGINT)


def finish(
    layout: JobLayout, run_id: str, envelope: dict[str, Any], exit: int
) -> None:
    """In the background process: the run's answer."""
    record = _read(layout, run_id) or {"id": run_id}
    _write(
        layout,
        {
            **record,
            "state": "finished",
            "finished_at": _now(),
            "exit": exit,
            "envelope": envelope,
        },
    )


def read(layout: JobLayout, run_id: str) -> dict[str, Any] | None:
    """The run's record, ``state`` "lost" when its process is gone unanswered."""
    record = _read(layout, run_id)
    if record is None or record.get("state") != "running":
        return record
    pid = record.get("pid")
    if pid is None:
        if time.time() - float(record.get("started") or 0) > START_GRACE_S:
            return {**record, "state": "lost"}
        return record
    if not _alive(pid):
        again = _read(layout, run_id) or record
        return (
            again
            if again.get("state") != "running"
            else {**again, "state": "lost"}
        )
    return record


def brief(record: dict[str, Any]) -> dict[str, Any]:
    """A run in one line: id, verb, state, times, exit."""
    return {
        key: record[key]
        for key in ("id", "verb", "state", "started_at", "finished_at", "exit")
        if key in record
    }


def listing(layout: JobLayout) -> list[dict[str, Any]]:
    """Every run of the job, newest first (brief)."""
    folder = runs_dir(layout)
    if not folder.is_dir():
        return []
    found = [read(layout, path.stem) for path in folder.glob("*.json")]
    runs = [brief(record) for record in found if record is not None]
    return sorted(runs, key=lambda run: str(run.get("id")), reverse=True)


def latest(layout: JobLayout) -> str | None:
    runs = listing(layout)
    return str(runs[0]["id"]) if runs else None


CANCEL_GRACE_S = 10.0


def cancel(layout: JobLayout, run_id: str) -> None:
    """End the run's process (Ctrl-C on a call that waits for its own run).

    Linux and macOS: ``SIGTERM``. Windows: a request the run's own watcher
    turns into Ctrl-C (``KeyboardInterrupt``, its cleanup runs), then, if it
    still runs after :data:`CANCEL_GRACE_S`, its whole process tree ended.
    A console signal cannot reach it there: it has a console of its own.
    """
    record = _read(layout, run_id) or {}
    pid = record.get("pid")
    if (
        record.get("state") != "running"
        or not isinstance(pid, int)
        or pid <= 0
        or pid == os.getpid()
    ):
        return
    if WINDOWS:
        _cancel_windows(layout, run_id, pid)
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass


def _cancel_windows(layout: JobLayout, run_id: str, pid: int) -> None:
    cancel_path(layout, run_id).touch()
    deadline = time.monotonic() + CANCEL_GRACE_S
    while _alive(pid):
        if time.monotonic() >= deadline:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],  # noqa: S607
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
                creationflags=_DETACHED_FLAGS,
            )
            return
        time.sleep(POLL_S)


def wait(
    layout: JobLayout, run_id: str, timeout: float | None
) -> dict[str, Any] | None:
    """The run's record once it is no longer running (or at *timeout* seconds)."""
    deadline = (
        None if timeout is None else time.monotonic() + max(0.0, timeout)
    )
    while True:
        record = read(layout, run_id)
        if record is None or record.get("state") != "running":
            return record
        if deadline is not None and time.monotonic() >= deadline:
            return record
        time.sleep(POLL_S)

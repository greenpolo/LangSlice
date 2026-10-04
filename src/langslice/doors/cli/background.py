"""Background runs of the agent CLI: ``--background``, ``status ID``, ``wait``.

A long verb (``fit_deformable``, ``trace_borders``) answers at once with a
run id when given ``--background``: the same command is started again as a
detached process (:data:`CHILD_COMMAND`) that runs the verb and saves its
envelope. Each run is one record, ``logs/runs/<id>.json`` in the job folder
(``id``, ``verb``, ``arguments``, ``state`` running/finished/lost,
``started_at``, ``pid``, and once finished ``finished_at``, ``exit`` and the
``envelope``), with the process's stderr in ``logs/runs/<id>.log``. A run
whose process is gone without an answer reads as ``lost``.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langslice.job.layout import JobLayout
from langslice.linear.checkpoint import write_json_atomic

#: How a background run starts the CLI again (tests start a helper instead).
CHILD_COMMAND: list[str] = [sys.executable, "-m", "langslice"]
#: A run whose process has not reported its id after this long is lost.
START_GRACE_S = 120.0
#: How often ``wait`` looks.
POLL_S = 0.25


def runs_dir(layout: JobLayout) -> Path:
    return layout.logs_dir / "runs"


def record_path(layout: JobLayout, run_id: str) -> Path:
    return runs_dir(layout) / f"{run_id}.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write(layout: JobLayout, record: dict[str, Any]) -> None:
    runs_dir(layout).mkdir(parents=True, exist_ok=True)
    write_json_atomic(str(record_path(layout, record["id"])), record)


def _read(layout: JobLayout, run_id: str) -> dict[str, Any] | None:
    try:
        data = json.loads(record_path(layout, run_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":  # no cheap probe without extra packages: trust the record
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _detached() -> dict[str, Any]:
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
            subprocess, "DETACHED_PROCESS", 0)
        return {"creationflags": flags}
    return {"start_new_session": True}


def start(layout: JobLayout, verb: str, arguments: dict[str, Any], *, verbose: bool) -> str:
    """Start *verb* on the job in a detached process; its run id."""
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
    log = runs_dir(layout) / f"{run_id}.log"
    _write(layout, {"id": run_id, "verb": verb, "arguments": arguments, "state": "running",
                    "started_at": _now(), "started": time.time(), "pid": None,
                    "log": layout.relative(log)})
    command = [*CHILD_COMMAND, "job", str(layout.folder), verb, "--args",
               json.dumps(arguments), "--run-id", run_id, *(["--verbose"] if verbose else [])]
    with log.open("ab") as out:
        subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out,  # noqa: S603
                         stderr=subprocess.STDOUT, **_detached())
    return run_id


def begin(layout: JobLayout, run_id: str) -> None:
    """In the background process: say it runs (its pid)."""
    record = _read(layout, run_id) or {"id": run_id, "started_at": _now(),
                                        "started": time.time()}
    if record.get("state") in (None, "running"):
        _write(layout, {**record, "state": "running", "pid": os.getpid()})


def finish(layout: JobLayout, run_id: str, envelope: dict[str, Any], exit: int) -> None:
    """In the background process: the run's answer."""
    record = _read(layout, run_id) or {"id": run_id}
    _write(layout, {**record, "state": "finished", "finished_at": _now(), "exit": exit,
                    "envelope": envelope})


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
        # Look once more: it may have finished between the two reads.
        again = _read(layout, run_id) or record
        return again if again.get("state") != "running" else {**again, "state": "lost"}
    return record


def brief(record: dict[str, Any]) -> dict[str, Any]:
    """A run in one line: id, verb, state, times, exit."""
    return {key: record[key] for key in ("id", "verb", "state", "started_at", "finished_at",
                                         "exit") if key in record}


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


def wait(layout: JobLayout, run_id: str, timeout: float | None) -> dict[str, Any] | None:
    """The run's record once it is no longer running (or at *timeout* seconds)."""
    deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
    while True:
        record = read(layout, run_id)
        if record is None or record.get("state") != "running":
            return record
        if deadline is not None and time.monotonic() >= deadline:
            return record
        time.sleep(POLL_S)

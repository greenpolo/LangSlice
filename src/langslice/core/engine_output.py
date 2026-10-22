"""What the registration engines print, kept in a log file instead of the caller's stderr.

FireANTs' first fit in a process compiles a GPU helper (Triton calls the C
compiler, whose warnings go to the process's standard error), torch's
allocator prints its warnings there, and ITK's Elastix and SciPy print
theirs. A tool reply is the caller's answer, and an agent reads standard
error with it: :func:`engine_output` points file descriptor 2 at a log file
while an engine runs, so what the engine and the programs it starts print
lands in the log (the job folder's ``logs/engine.log``, chosen by the
caller) and not in the reply.

Descriptor 2 belongs to the whole process: while one engine run holds it,
anything another thread prints to standard error goes to that log too. Runs
that overlap share the first one's log; the last to finish puts standard
error back.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

#: Guards the shared redirection (its depth and the saved descriptor).
_LOCK = threading.Lock()
_HELD: dict[str, int | None] = {"depth": 0, "saved": None}


def _flush() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (AttributeError, OSError, ValueError):
            pass


@contextlib.contextmanager
def engine_output(path: str | Path | None, label: str = "") -> Iterator[None]:
    """Inside the block, file descriptor 2 writes to *path* (appended; its
    folder made when missing), headed by a line naming *label*. None, or a
    log that cannot be opened, leaves standard error as it is."""
    if path is None:
        yield
        return
    with _LOCK:
        if _HELD["depth"] == 0:
            try:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            except OSError:
                fd = -1
            if fd >= 0:
                _flush()
                try:
                    _HELD["saved"] = os.dup(2)
                    os.dup2(fd, 2)
                except OSError:
                    _HELD["saved"] = None
                finally:
                    os.close(fd)
        _HELD["depth"] = int(_HELD["depth"] or 0) + 1
        if _HELD["saved"] is not None and label:
            os.write(2, f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} {label}\n".encode())
    try:
        yield
    finally:
        with _LOCK:
            _HELD["depth"] = int(_HELD["depth"] or 1) - 1
            saved = _HELD["saved"]
            if _HELD["depth"] == 0 and saved is not None:
                _flush()
                os.dup2(saved, 2)
                os.close(saved)
                _HELD["saved"] = None


#: The longest first line of an engine's error a reply carries, in characters.
FAILURE_CHARS = 240


def failure_message(text: str, path: str | Path | None, label: str = "") -> str:
    """What a reply says of an engine that failed: the first line of its
    error *text* (at most :data:`FAILURE_CHARS` characters), and, when the
    whole error was appended to *path* (the job's ``logs/engine.log``, headed
    by *label*), that the log holds it. A GPU that ran out of memory is said
    in words: another fit held it, and the call can run again."""
    whole = str(text or "").strip() or "no error text"
    first = whole.splitlines()[0].strip()
    if len(first) > FAILURE_CHARS:
        first = first[:FAILURE_CHARS - 3].rstrip() + "..."
    kept = False
    if path is not None:
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as stream:
                stream.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} {label or 'failed'}\n"
                             f"{whole}\n")
            kept = True
        except OSError:
            kept = False
    words = first
    if "out of memory" in whole.lower():
        words += (". The GPU ran out of memory, most likely while another fit held it; "
                  "running the call again fits it from the start")
    if kept:
        words += ". logs/engine.log holds the whole error"
    return words + ("" if words.endswith(".") else ".")

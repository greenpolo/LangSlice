"""What a host-owned door records of its calls: the trace and the call log.

The MCP door and the agent CLI cannot see the model's words (the host owns
the conversation), so their trace records what the host was shown and what
it called, one JSON line per record (:class:`HostTrace`; images as
descriptors, never bytes), written only when ``LANGSLICE_TRACE_DIR`` is set
(:data:`TRACE_DIR_ENV`, the one switch LangSlice's own agent's full trace
uses too, :mod:`langslice.agent.trace`). The MCP door keeps one file per
session (``mcp_<folder>_<id>.jsonl``); the agent CLI runs a process per call,
so it appends to one file per job folder (``cli_<folder>_<digest>.jsonl``,
:func:`cli_trace`).

The agent CLI also logs every call in the job folder, whatever the
environment says (:func:`log_call`: ``logs/calls.jsonl``): the verb, its
arguments as given, the envelope's outcome and the artifacts' paths.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Directory for traces. Unset means no tracing at all.
TRACE_DIR_ENV = "LANGSLICE_TRACE_DIR"

#: The agent CLI's call log in the job folder.
CALLS_FILE = "calls.jsonl"


class HostTrace:
    """JSONL record of what a host was shown and what it called.

    The host's own words between calls never reach LangSlice, so unlike
    :class:`langslice.agent.trace.SessionTrace` there is no model record.
    Images are descriptors, never bytes. *name* fixes the file name (the
    agent CLI's one file per job); otherwise each trace is a new file,
    ``<prefix>_<folder name>_<id>.jsonl``.
    """

    def __init__(self, trace_dir: str | Path, folder: str, *, prefix: str = "mcp",
                 name: str | None = None) -> None:
        stem = Path(folder).name or "stack"
        self.path = Path(trace_dir) / (name or f"{prefix}_{stem}_{uuid.uuid4().hex[:8]}.jsonl")

    def write(self, kind: str, **fields: Any) -> None:
        record = {"kind": kind, **fields, "at": datetime.now().isoformat(timespec="seconds")}
        _append(self.path, record)


def cli_trace(job_folder: str | Path, trace_dir: str | None) -> HostTrace | None:
    """The agent CLI's trace of the job in *job_folder* (one file per job
    folder in *trace_dir*, the value of :data:`TRACE_DIR_ENV`), or None."""
    if not trace_dir:
        return None
    folder = Path(job_folder)
    digest = hashlib.sha256(str(folder.resolve()).encode()).hexdigest()[:8]
    images = folder.parent.name if folder.name == "langslice" else folder.name
    return HostTrace(trace_dir, str(folder), name=f"cli_{images or 'stack'}_{digest}.jsonl")


def log_call(logs_folder: Path, record: dict[str, Any]) -> None:
    """Append one agent-CLI call to ``logs/calls.jsonl`` (never raises)."""
    _append(Path(logs_folder) / CALLS_FILE,
            {"at": datetime.now().isoformat(timespec="seconds"), **record})


def _append(path: Path, record: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    except Exception:
        logger.warning("Could not write a record to %s", path, exc_info=True)


__all__ = ["CALLS_FILE", "TRACE_DIR_ENV", "HostTrace", "cli_trace", "log_call"]

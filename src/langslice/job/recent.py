"""The jobs opened on this computer, newest last: ``~/.langslice/recent_jobs.jsonl``.

:meth:`langslice.job.job.Job.open` appends one line per open (``folder``,
``image_folder``, ``atlas``, ``at``), so a viewer such as the LangSlice GUI
can list the jobs any door made or resumed. ``LANGSLICE_RECENT_JOBS``
(:data:`RECENT_ENV`) names another file, or ``off`` (also ``0``, ``false``,
``no``, ``none``, empty) for none; the test suite sets it off. Past
:data:`KEEP_BYTES` the file is rewritten with the newest line of each job
folder, the newest :data:`KEEP_JOBS` of them. Writing it never fails an open.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Another file for the list, or ``off`` for none.
RECENT_ENV = "LANGSLICE_RECENT_JOBS"
RECENT_FILE = "recent_jobs.jsonl"
#: Past this size the file is rewritten, one line per job folder.
KEEP_BYTES = 256 * 1024
#: The job folders a rewrite keeps, newest first.
KEEP_JOBS = 500
_OFF = frozenset({"", "0", "off", "false", "no", "none"})


def recent_file() -> Path | None:
    """The list's file, or None when it is off."""
    value = os.environ.get(RECENT_ENV)
    if value is None:
        return Path.home() / ".langslice" / RECENT_FILE
    if value.strip().lower() in _OFF:
        return None
    return Path(value).expanduser()


def remember(folder: str | os.PathLike[str], image_folder: str | os.PathLike[str] | None,
             atlas: str | None) -> None:
    """Append the job *folder* to the list (absolute paths, an ISO time)."""
    path = recent_file()
    if path is None:
        return
    record = {
        "folder": os.path.abspath(os.fspath(folder)),
        "image_folder": (os.path.abspath(os.fspath(image_folder))
                         if image_folder is not None else None),
        "atlas": atlas,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if path.stat().st_size > KEEP_BYTES:
            _compact(path)
    except OSError:
        logger.warning("Could not add the job to %s", path, exc_info=True)


def read(path: Path | None = None) -> list[dict[str, Any]]:
    """The list's lines, oldest first; a line that does not parse is skipped."""
    path = path if path is not None else recent_file()
    if path is None:
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and isinstance(entry.get("folder"), str):
            out.append(entry)
    return out


def _compact(path: Path) -> None:
    """Rewrite *path* with the newest line of each of the newest
    :data:`KEEP_JOBS` job folders, oldest first."""
    newest: dict[str, dict[str, Any]] = {}
    for entry in read(path):
        newest.pop(entry["folder"], None)
        newest[entry["folder"]] = entry
    kept = list(newest.values())[-KEEP_JOBS:]
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(json.dumps(entry) + "\n" for entry in kept), encoding="utf-8")
    os.replace(temporary, path)

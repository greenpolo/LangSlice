"""Saved host jobs by id: a small index pointing at job folders.

A Claude job (``langslice claude prepare``, ABBA's Claude mode) is opened by
id (``start_job(job_id=...)``), so the id has to lead to a job folder.
``~/.langslice/jobs/<id>.json`` is that pointer: the job folder's absolute
path, plus the one thing that does not belong in a folder that may be
shared with colleagues, the host's loopback channel token. Owner-only on
POSIX. Everything else of the job lives in its folder (``job.json`` there,
under ``host``).

Before the job folder (phase 2), ``~/.langslice/jobs/<id>/`` WAS the job:
``job.json``, the checkpoint, the undo history and the results.
:func:`langslice.job.migrate.migrate_saved_job` moves such a directory into
the job folder next to its images on first open.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: The index entry's format. 1 (2026-10-03): ``job_id``, ``job_folder``,
#: ``host_channel``, ``created_at``.
INDEX_FORMAT_VERSION = 1
_JOB_ID = re.compile(r"[0-9a-f]{12}")


def default_root() -> Path:
    """``~/.langslice/jobs``."""
    return Path.home() / ".langslice" / "jobs"


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def check_id(job_id: str) -> str:
    if not isinstance(job_id, str) or _JOB_ID.fullmatch(job_id) is None:
        raise ValueError("Invalid LangSlice job id")
    return job_id


def entry_path(root: Path, job_id: str) -> Path:
    return Path(root) / f"{check_id(job_id)}.json"


def legacy_dir(root: Path, job_id: str) -> Path:
    """Where a phase-2 saved job lived (the whole job, not a pointer)."""
    return Path(root) / check_id(job_id)


def register(
    root: Path, job_id: str, folder: Path, *, host_channel: dict[str, Any] | None = None,
    created_at: str | None = None,
) -> Path:
    """Write the id's entry, pointing at *folder*; owner-only. Returns its path."""
    path = entry_path(root, job_id)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    record = {
        "format_version": INDEX_FORMAT_VERSION, "job_id": job_id,
        "job_folder": str(Path(os.path.abspath(folder))),
        "host_channel": host_channel,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        os.chmod(temporary, 0o600)
        json.dump(record, handle, indent=2)
    os.replace(temporary, path)
    return path


def lookup(root: Path, job_id: str) -> dict[str, Any] | None:
    """The id's entry, or None when the index has none."""
    path = entry_path(root, job_id)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if (not isinstance(record, dict) or record.get("job_id") != job_id
            or not isinstance(record.get("job_folder"), str)):
        raise ValueError("Unsupported or mismatched LangSlice job file")
    version = record.get("format_version")
    if not isinstance(version, int) or version > INDEX_FORMAT_VERSION:
        raise ValueError(
            f"This saved job was written by a newer LangSlice (format {version!r}). "
            "Update LangSlice to open it.")
    return record

"""Job folders kept outside their image folder, by id: a small index.

When an image folder cannot be written, its job folder is
``~/.langslice/jobs/<id>/`` (:func:`langslice.job.layout.locate_job_folder`),
the id being the image folder's path hashed (:func:`folder_id`), and
``~/.langslice/jobs/<id>.json`` points at it and says why
(``fallback``). Owner-only on POSIX. Everything else of the job lives in its
folder.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: The index entry's format: ``job_id``, ``job_folder``, ``created_at`` and
#: ``fallback``.
INDEX_FORMAT_VERSION = 1
_JOB_ID = re.compile(r"[0-9a-f]{12}")


def default_root() -> Path:
    """``~/.langslice/jobs``."""
    return Path.home() / ".langslice" / "jobs"


def folder_id(image_folder: str | os.PathLike[str]) -> str:
    """A stable id for an image folder (its absolute path, hashed), so a job
    whose folder could not be written is found again on reopen."""
    import hashlib

    return hashlib.sha256(os.path.abspath(os.fspath(image_folder)).encode()).hexdigest()[:12]


def check_id(job_id: str) -> str:
    if not isinstance(job_id, str) or _JOB_ID.fullmatch(job_id) is None:
        raise ValueError("Invalid LangSlice job id")
    return job_id


def entry_path(root: Path, job_id: str) -> Path:
    return Path(root) / f"{check_id(job_id)}.json"


def register(
    root: Path, job_id: str, folder: Path, *,
    created_at: str | None = None, fallback: dict[str, Any] | None = None,
) -> Path:
    """Write the id's entry, pointing at *folder*; owner-only. Returns its path.

    *fallback* records why the job folder is here and not next to its images
    (``image_folder``, ``reason``): the image folder could not be written.
    """
    path = entry_path(root, job_id)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    record: dict[str, object] = {
        "format_version": INDEX_FORMAT_VERSION, "job_id": job_id,
        "job_folder": str(Path(os.path.abspath(folder))),
        "created_at": created_at or datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if fallback is not None:
        record["fallback"] = fallback
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

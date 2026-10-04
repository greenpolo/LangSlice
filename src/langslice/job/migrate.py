"""Opening an old layout upgrades it into the job folder.

Two old layouts exist, both from before the job folder (phase 2):

- **Beside the images** (``langslice linear run``, ``langslice mcp`` on a
  folder, an ABBA worker run): ``linear_state.json``, ``linear_undo.json``
  and ``linear_results.json`` in the image folder, with the deformation
  records under ``deformable/<section>/<key>/`` and the image corrections
  under ``nonlinear/<section>/<key>/`` beside them.
- **A saved Claude job** under ``~/.langslice/jobs/<id>/``: ``job.json``
  (format 1: the host's parameters, notes, channel), the same three state
  files, ``deformable/``, ``nonlinear/``, ``result.json`` and ``prompt.txt``.

:func:`migrate_beside_images` and :func:`migrate_saved_job` move them into
``<images>/langslice/`` (:mod:`langslice.job.layout`): the checkpoint to
``state.json``, the undo history to ``history/`` one file per step, the
records and corrections into their section folders, the results to
``exports/``. Every stored path is rewritten to its new place, relative to
the job folder, and the checkpoint is written at the current format (state
format 2; ``job.json`` format 1). Each migration is one line in
``logs/events.jsonl``. A newer format is refused before anything moves.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langslice.job import index
from langslice.job.checkpoint import (
    CHECKPOINT_FILENAME,
    FORMAT_KEY,
    STATE_FORMAT_VERSION,
    relative_to,
    state_paths,
    upgrade_state,
    write_json_atomic,
)
from langslice.job.history import History
from langslice.job.layout import (
    DEFORMABLE_DIR,
    IMAGE_CORRECTION_DIR,
    JobLayout,
    write_job_file,
)

logger = logging.getLogger(__name__)

#: The old layout's undo history (one rewritten file, format 1) and results.
LEGACY_UNDO_FILENAME = "linear_undo.json"
LEGACY_RESULTS_FILENAME = "linear_results.json"
LEGACY_UNDO_FORMAT_VERSION = 1
#: The old record and correction folders, and where each goes per section.
_LEGACY_FOLDERS = (("deformable", DEFORMABLE_DIR), ("nonlinear", IMAGE_CORRECTION_DIR))
#: The phase-2 saved job's own format.
LEGACY_SAVED_JOB_FORMAT = 1


def _sanitized(section_id: str) -> str:
    """The old folder name of a section (``RecordStore`` / ``registration_tool``)."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", Path(section_id).name)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_legacy_history(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(undo, redo)`` of a phase-2 ``linear_undo.json``, each a state's
    fields (unconverted paths); empty when unreadable or newer."""
    try:
        data = _load_json(path)
    except (OSError, ValueError):
        logger.warning("Old undo history %s unreadable; not migrated", path, exc_info=True)
        return [], []
    version = data.get(FORMAT_KEY) if isinstance(data, dict) else None
    if not isinstance(version, int) or version > LEGACY_UNDO_FORMAT_VERSION:
        logger.warning("Old undo history %s has format %r; not migrated", path, version)
        return [], []
    state_version = data.get("state_format_version", 0)
    try:
        return tuple(  # type: ignore[return-value]
            [upgrade_state({**entry, FORMAT_KEY: state_version})
             for entry in data.get(side) or [] if isinstance(entry, dict)]
            for side in ("undo", "redo"))
    except ValueError:
        logger.warning("Old undo history %s unreadable; not migrated", path, exc_info=True)
        return [], []


def _section_ids(*states: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for data in states:
        for row in data.get("slices") or []:
            if isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"] not in ids:
                ids.append(row["id"])
    return ids


def _move_folders(base: Path, layout: JobLayout, ids: list[str]) -> dict[str, str]:
    """Move ``deformable/`` and ``nonlinear/`` into the section folders.

    Returns old absolute folder -> new absolute folder, per moved key folder.
    A key folder already present in the job folder is kept (the same key is
    the same content) and the old one removed.
    """
    by_name = {_sanitized(section_id): section_id for section_id in ids}
    moves: dict[str, str] = {}
    for old_name, new_name in _LEGACY_FOLDERS:
        root = base / old_name
        if not root.is_dir():
            continue
        for section_folder in sorted(root.iterdir()):
            if not section_folder.is_dir():
                continue
            section_id = by_name.get(section_folder.name, section_folder.name)
            target_root = layout.section_dir(section_id) / new_name
            target_root.mkdir(parents=True, exist_ok=True)
            for key_folder in sorted(section_folder.iterdir()):
                target = target_root / key_folder.name
                if target.exists():
                    shutil.rmtree(key_folder) if key_folder.is_dir() else key_folder.unlink()
                else:
                    shutil.move(str(key_folder), str(target))
                for spelled in dict.fromkeys((str(key_folder.absolute()),
                                              os.path.realpath(key_folder))):
                    moves[spelled] = str(target)
            _remove_if_empty(section_folder)
        _remove_if_empty(root)
    return moves


def _remove_if_empty(folder: Path) -> None:
    try:
        folder.rmdir()
    except OSError:
        pass


def _converter(moves: dict[str, str], layout: JobLayout) -> Callable[[str], str]:
    """Old stored path -> its new place, relative to the job folder."""
    inside = relative_to(layout.folder)
    ordered = sorted(moves.items(), key=lambda item: len(item[0]), reverse=True)

    def convert(value: str) -> str:
        if not os.path.isabs(value):
            return value
        for old, new in ordered:
            if value == old or value.startswith(old + os.sep):
                return inside(new + value[len(old):])
        return inside(value)

    return convert


def _rewrite_correction_results(layout: JobLayout, convert: Callable[[str], str]) -> None:
    """The saved replies of moved image corrections, pointing at their new place."""
    for path in layout.sections_dir.glob(f"*/{IMAGE_CORRECTION_DIR}/*/**/result.json"):
        try:
            data = _load_json(path)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        fixed = state_paths({"slices": [{"image_correction": data}]}, convert)
        write_json_atomic(str(path), fixed["slices"][0]["image_correction"])


def _move_state_files(
    base: Path, layout: JobLayout, *, source: str, extra: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Move one old layout's state files and folders from *base*; the report."""
    state_path = base / CHECKPOINT_FILENAME
    undo_path = base / LEGACY_UNDO_FILENAME
    results_path = base / LEGACY_RESULTS_FILENAME
    state = None
    if state_path.exists():
        raw = _load_json(state_path)
        if not isinstance(raw, dict):
            raise ValueError(f"{state_path} does not hold a stack state")
        state = upgrade_state(raw)  # refuses a newer format before anything moves
    undo, redo = read_legacy_history(undo_path) if undo_path.exists() else ([], [])
    layout.ensure()
    ids = _section_ids(*([state] if state else []), *undo, *redo)
    moves = _move_folders(base, layout, ids)
    convert = _converter(moves, layout)
    if state is not None:
        write_json_atomic(str(layout.state_file),
                          {FORMAT_KEY: STATE_FORMAT_VERSION, **state_paths(state, convert)})
        state_path.unlink()
    if undo or redo:
        History(layout.history_dir).save([state_paths(entry, convert) for entry in undo],
                                          [state_paths(entry, convert) for entry in redo])
    if undo_path.exists():
        undo_path.unlink()
    if results_path.exists():
        try:
            results = _load_json(results_path)
            if isinstance(results, dict):
                results = state_paths(results, convert)
            layout.exports_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(str(layout.results_file), results)
            results_path.unlink()
        except (OSError, ValueError):
            logger.warning("Old results %s not migrated", results_path, exc_info=True)
    for target, old in (extra or {}).items():
        if old.exists():
            destination = layout.folder / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), str(destination))
    _rewrite_correction_results(layout, convert)
    report = {"source": source, "from": str(base), "state": state is not None,
              "undo_steps": len(undo), "redo_steps": len(redo), "folders_moved": len(moves)}
    layout.log_event("migrated", **report)
    logger.info("Migrated the old LangSlice files in %s into %s", base, layout.folder)
    return report


def has_legacy_files(folder: Path) -> bool:
    """Whether *folder* holds a phase-2 checkpoint or undo history."""
    return (folder / CHECKPOINT_FILENAME).exists() or (folder / LEGACY_UNDO_FILENAME).exists()


def migrate_beside_images(layout: JobLayout) -> dict[str, Any] | None:
    """Move the old files beside the images into the job folder.

    Nothing to do (None) when the image folder holds no old checkpoint or
    history, or when the job folder already holds a checkpoint: the job
    folder is then the job, and the old files are left where they are
    (logged).
    """
    base = layout.image_folder
    if base is None or not has_legacy_files(base):
        return None
    if layout.state_file.exists():
        logger.warning("%s holds old LangSlice files, but %s already holds a job; "
                       "the old files are left as they are", base, layout.folder)
        return None
    return _move_state_files(base, layout, source="beside the images")


def migrate_saved_job(root: Path, job_id: str) -> dict[str, Any]:
    """Move a phase-2 saved job (``<root>/<id>/``) into the job folder next to
    its images and index it there; returns the new index entry.

    Refused (``ValueError``, nothing moved) when the job file is not the old
    format or when that job folder already holds another job's checkpoint.
    """
    old = index.legacy_dir(root, job_id)
    record = _load_json(old / "job.json")
    if (not isinstance(record, dict) or record.get(FORMAT_KEY) != LEGACY_SAVED_JOB_FORMAT
            or record.get("job_id") != job_id):
        raise ValueError("Unsupported or mismatched LangSlice job file")
    kind = record.get("kind", "host")
    if kind == "folder":
        image_folder = (record.get("spec") or {}).get("image_folder")
    else:
        image_folder = (record.get("params") or {}).get("image_folder")
    if not isinstance(image_folder, str) or not image_folder:
        raise ValueError(f"Saved job {job_id} names no image folder")
    layout = JobLayout.for_images(Path(image_folder).expanduser())
    if layout.state_file.exists() and (old / CHECKPOINT_FILENAME).exists():
        raise ValueError(
            f"Saved job {job_id} cannot move into {layout.folder}: that folder already "
            "holds a LangSlice job. Move it away and open the saved job again.")
    _move_state_files(old, layout, source=f"saved job {job_id}", extra={
        "prompt.txt": old / "prompt.txt",
        "exports/result.json": old / "result.json",
    })
    host = {key: record[key] for key in ("kind", "params", "notes", "trace_dir")
            if key in record}
    host.setdefault("kind", kind)
    fields: dict[str, Any] = {"job_id": job_id, "host": host}
    if kind == "folder" and isinstance(record.get("spec"), dict):
        fields["spec"] = record["spec"]
    if isinstance(record.get("created_at"), str):
        fields["created_at"] = record["created_at"]
    write_job_file(layout, **fields)
    index.register(root, job_id, layout.folder, host_channel=record.get("host_channel"),
                   created_at=record.get("created_at"))
    (old / "job.json").unlink()
    _remove_if_empty(old)
    if old.exists():
        logger.warning("Left unknown files in the old saved job %s", old)
    entry = index.lookup(root, job_id)
    if entry is None:  # pragma: no cover - just written
        raise ValueError(f"Saved job {job_id} could not be indexed")
    return entry

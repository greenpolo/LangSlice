"""The border trace folders of format 4, moved to their format-5 name.

A section's image-model border trace was kept in
``sections/<name>/image_correction/`` until state format 5 and is kept in
``sections/<name>/border_trace/`` since. :func:`move_trace_folders` moves
the folders of a job opened to write, under its write lock, and writes the
paths naming them again in the files besides the state that store them;
the state's own paths follow through :func:`moved_path`, which the
migration ``state_4_to_5`` (:mod:`langslice.job.migrations`) applies on
read, so a job read without writing finds its traces where they stand.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from langslice.job.layout import BORDER_TRACE_DIR, JobLayout

logger = logging.getLogger(__name__)

#: A section folder's border trace folder before format 5.
OLD_TRACE_DIR = "image_correction"
#: A path, or a JSON text, naming a file in a section's border trace
#: folder before format 5: ``sections/<name>/image_correction/...``.
OLD_TRACE_PATH = re.compile(rf'(^|["/])(sections/[^/"]+/){OLD_TRACE_DIR}/')
#: The files besides the state that store such paths, by pattern in the
#: job folder: the pictures' records and the sections' registrations.
PATH_FILES = (
    "views.jsonl",
    "views/*/view.json",
    "sections/*/views/*/view.json",
    "sections/*/registrations.jsonl",
)


def moved_path(path: str, layout: JobLayout) -> str:
    """Return a stored path in its section's moved border trace folder.

    Parameters
    ----------
    path
        A path a state stores, relative to the job folder or absolute.
    layout
        The job folder.

    Returns
    -------
    str
        *path* with ``sections/<name>/image_correction/`` as
        ``sections/<name>/border_trace/`` when the folder has moved (the
        old one is gone, the new one is there); else *path*.
    """
    moved = OLD_TRACE_PATH.sub(rf"\1\2{BORDER_TRACE_DIR}/", path)
    if moved == path:
        return path
    if layout.resolve(path).exists() or not layout.resolve(moved).exists():
        return path
    return moved


def move_trace_folders(layout: JobLayout) -> tuple[str, ...]:
    """Move each section's border trace folder to its format-5 name.

    ``sections/<name>/image_correction`` becomes
    ``sections/<name>/border_trace``, and the paths naming it are written
    again in the trace folder's own JSON files and in :data:`PATH_FILES`.
    A folder whose new name is taken, or that cannot be renamed, stays
    where it is (logged). The caller holds the job's write lock; the state
    is read afterwards, through :func:`state_4_to_5`.

    Parameters
    ----------
    layout
        The job folder.

    Returns
    -------
    tuple of str
        The section folders whose border trace folder moved.
    """
    moved: list[str] = []
    for old in sorted(layout.folder.glob(f"sections/*/{OLD_TRACE_DIR}")):
        new = old.with_name(BORDER_TRACE_DIR)
        if new.exists():
            logger.warning("%s stays where it is: %s exists", old, new.name)
            continue
        try:
            old.rename(new)
        except OSError:  # a folder that cannot be renamed is read where it is
            logger.warning("%s stays where it is", old, exc_info=True)
            continue
        moved.append(old.parent.name)
        for path in sorted(new.rglob("*.json")):
            _write_moved_paths(path)
    if moved:
        for pattern in PATH_FILES:
            for path in sorted(layout.folder.glob(pattern)):
                _write_moved_paths(path)
    return tuple(moved)


def _write_moved_paths(path: Path) -> None:
    """Write a JSON file again with its trace folder paths moved."""
    text = path.read_text(encoding="utf-8")
    moved = OLD_TRACE_PATH.sub(rf"\1\2{BORDER_TRACE_DIR}/", text)
    if moved == text:
        return
    temporary = path.with_name(f"{path.name}.moving")
    temporary.write_text(moved, encoding="utf-8")
    temporary.replace(path)

"""The state checkpoint, ``state.json``: the job's record of truth.

Every write saves the full :class:`StackState` there, atomically (temp file
+ rename) so a crash mid-write cannot leave a truncated file. Resume = load
the checkpoint and re-seed the agent with the state it had. Several writers
may share one job folder; they take turns under its write lock
(:meth:`langslice.job.job.Job.writing`).

The file carries :data:`STATE_FORMAT_VERSION` under ``format_version`` next
to the state's own fields. A checkpoint of any other format is refused,
never guessed at: a newer one asks for a newer LangSlice, an older one (a
pre-release LangSlice's) for a new job.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from typing import Any

from langslice.core.state import StackState

#: The key a job-folder file carries its format version under.
FORMAT_KEY = "format_version"
#: The checkpoint's format. Every path on a section record (a deformation's
#: ``record``, an image correction's ``artifact_dir`` and ``artifact_paths``)
#: is relative to the job folder (:func:`state_paths`), and every section
#: carries its own cutting angles (``SliceState.cutting_angles_deg``). A
#: state whose sections share one angle also writes it as the stack's
#: ``cutting_angles_deg``; one whose sections differ (a registration
#: supplied per section) writes the stack's as null.
STATE_FORMAT_VERSION = 3


def write_json_atomic(path: str, data: Any, *, indent: int | None = 2) -> None:
    """Write *data* as JSON to *path* through a temporary file and a rename."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            if indent is None:
                json.dump(data, handle, separators=(",", ":"))
            else:
                json.dump(data, handle, indent=indent)
        # mkstemp creates 0600; a job file is an ordinary output file.
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def write_checkpoint(state: StackState, path: str) -> None:
    """Write *state* to *path* atomically, versioned."""
    write_json_atomic(path, {FORMAT_KEY: STATE_FORMAT_VERSION, **state.to_dict()})


def state_paths(
    data: dict[str, Any], convert: Callable[[str], str],
) -> dict[str, Any]:
    """*data* (a state's fields) with every stored path passed through *convert*.

    The paths a state holds: each section's deformation ``record`` and its
    image correction's ``artifact_dir`` and ``artifact_paths``. Everything
    else is returned as it is (the input is not changed).
    """
    slices = []
    for row in data.get("slices") or []:
        if not isinstance(row, dict):
            slices.append(row)
            continue
        row = dict(row)
        held = row.get("deformation")
        if isinstance(held, dict) and isinstance(held.get("record"), str):
            row["deformation"] = {**held, "record": convert(held["record"])}
        correction = row.get("image_correction")
        if isinstance(correction, dict):
            correction = dict(correction)
            if isinstance(correction.get("artifact_dir"), str):
                correction["artifact_dir"] = convert(correction["artifact_dir"])
            paths = correction.get("artifact_paths")
            if isinstance(paths, dict):
                correction["artifact_paths"] = {
                    key: convert(value) if isinstance(value, str) else value
                    for key, value in paths.items()}
            row["image_correction"] = correction
        slices.append(row)
    return {**data, "slices": slices} if "slices" in data else dict(data)


def relative_to(root: str | os.PathLike[str]) -> Callable[[str], str]:
    """A path converter: absolute paths inside *root* become relative to it
    (POSIX separators); every other path is kept as it is."""
    bases = list(dict.fromkeys((os.path.abspath(root), os.path.realpath(root))))

    def convert(value: str) -> str:
        if not os.path.isabs(value):
            return value
        for base in bases:
            for spelled in dict.fromkeys((os.path.abspath(value), os.path.realpath(value))):
                try:
                    inside = os.path.commonpath([spelled, base]) == base
                except ValueError:  # different Windows drives have no common path
                    continue
                if inside:
                    return os.path.relpath(spelled, base).replace(os.sep, "/")
        return value

    return convert


def current_state(data: dict[str, Any]) -> dict[str, Any]:
    """A checkpoint's state fields, without the version key; ``ValueError``
    for any format but :data:`STATE_FORMAT_VERSION`."""
    version = data.get(FORMAT_KEY, 0)
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise ValueError(f"Unreadable checkpoint format_version {version!r}")
    if version > STATE_FORMAT_VERSION:
        raise ValueError(
            f"This checkpoint was written by a newer LangSlice (format {version}; this "
            f"version reads format {STATE_FORMAT_VERSION}). Update LangSlice to open it.")
    if version < STATE_FORMAT_VERSION:
        raise ValueError(
            f"This checkpoint was made by an older pre-release LangSlice (format {version}; "
            f"this version reads format {STATE_FORMAT_VERSION}). Start a new job: open it "
            "fresh (`--fresh`, resume=False) or remove the job folder.")
    return {key: value for key, value in data.items() if key != FORMAT_KEY}


def read_checkpoint(path: str) -> dict[str, Any] | None:
    """A checkpoint's state fields (:func:`current_state`); ``None`` if
    there is no file."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a stack state")
    return current_state(data)


def load_checkpoint(path: str) -> StackState | None:
    """Load a checkpoint, or ``None`` if there isn't one at *path*."""
    data = read_checkpoint(path)
    return None if data is None else StackState.from_dict(data)

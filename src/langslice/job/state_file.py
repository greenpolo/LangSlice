"""The state checkpoint, ``state.json``: the job's record of truth.

This module is the one place a :class:`~langslice.core.stack.stack.Stack`
and its sections become JSON and come back (:func:`stack_to_json`,
:func:`stack_from_json`); each section is one row, its parts in their JSON
form (:mod:`langslice.core.stack.rows`). Every write saves the full stack
there, atomically (temp file + rename) so a crash mid-write cannot leave a
truncated file. Resume = load
the checkpoint and re-seed the agent with the state it had. Several writers
may share one job folder; they take turns under its write lock
(:meth:`langslice.job.job.Job.writing`).

The file carries :data:`STATE_FORMAT_VERSION` under ``format_version`` next to
the state's own fields. A checkpoint of an older format is read through the
job's upgrade, the table of migrations
(:class:`langslice.job.migrations.FolderMigration`); any other format is
refused, never guessed at: a newer one asks for a newer LangSlice, an older
one the table does not read (a pre-release LangSlice's) for a new job.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from langslice.core.jsonable import Json, JsonObject, JsonValue, to_jsonable
from langslice.core.stack.json_reads import (
    flag_at,
    integer_at,
    list_at,
    object_at,
    objects_at,
    text_at,
    texts_at,
)
from langslice.core.stack.rows import (
    ANGLES_KEY,
    ROW_KEYS,
    section_from_json,
    section_to_json,
)
from langslice.core.stack.section import FLAT, Section
from langslice.core.stack.stack import Stack

#: A reader of a state's fields of its format (the first argument) that
#: returns them as the current format.
Upgrade = Callable[[int, dict[str, Any]], JsonObject]

#: The key a job-folder file carries its format version under.
FORMAT_KEY = "format_version"
#: The checkpoint's format. Every path on a section record (a deformation's
#: ``record``, a border trace's ``artifact_dir`` and ``artifact_paths``)
#: is relative to the job folder (:func:`state_paths`), and every section
#: carries its own cutting angles (``Section.angles``). A
#: state whose sections share one angle also writes it as the stack's
#: ``cutting_angles_deg``; one whose sections differ (a registration
#: supplied per section) writes the stack's as null. Every section's made
#: images are entries (``Section.images``) naming files under the job
#: folder; the stack has no ``appearance``. A section's border trace is
#: under ``border_trace``, its files in ``sections/<name>/border_trace/``.
STATE_FORMAT_VERSION = 5


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


#: The stack's own keys, in the order the state file writes them.
STACK_KEYS = (
    "image_folder", "atlas", "plane", "interval_mm", "thickness_mm", "spec",
    "interval_breaks", "notes", "slices", "submitted", "debrief",
)  # fmt: skip
#: The row keys a row leaves out while they are empty: most jobs have
#: none, so their state reads as it always has.
OPTIONAL_ROW_KEYS = ("layers", "masks", "images")


def stack_to_json(state: Stack) -> dict[str, JsonValue]:
    """Return a stack in its JSON form: the checkpoint's and the result's.

    The cutting angles are written once, on the stack
    (``"cutting_angles_deg": {"pitch", "yaw"}``), when every section shares
    them, so a single-angle state reads as it always has; when they differ
    (state format 3), the stack's is null and each section row carries its
    own.

    Parameters
    ----------
    state
        The stack written.

    Returns
    -------
    dict
        :data:`STACK_KEYS` in order, then ``cutting_angles_deg``.
    """
    rows = [section_to_json(section) for section in state.slices]
    angles: JsonValue = None
    if not state.mixed_angles:
        angles = {"pitch": FLAT[0], "yaw": FLAT[1]}
        if rows:
            angles = rows[0][ANGLES_KEY]
        for row in rows:
            del row[ANGLES_KEY]
    for row in rows:
        for key in OPTIONAL_ROW_KEYS:
            if not row[key]:
                del row[key]
    data: dict[str, JsonValue] = {
        "image_folder": state.image_folder,
        "atlas": state.atlas,
        "plane": state.plane,
        "interval_mm": state.interval_mm,
        "thickness_mm": state.thickness_mm,
        "spec": to_jsonable(state.spec),
        "interval_breaks": list[JsonValue](state.interval_breaks),
        "notes": list[JsonValue](state.notes),
        "slices": list[JsonValue](rows),
        "submitted": state.submitted,
        "debrief": state.debrief,
        ANGLES_KEY: angles,
    }
    return data


def _rows(data: JsonObject) -> tuple[JsonObject, ...]:
    """Return a state's section rows; an absent or null list is none."""
    if data.get("slices") is None:
        return ()
    return objects_at(data, "slices")


def _unknown_keys(data: JsonObject) -> list[str]:
    """Return the keys of a state, and of its rows, that it cannot hold."""
    unknown = {key for key in data if key not in (*STACK_KEYS, ANGLES_KEY)}
    for row in _rows(data):
        unknown |= {f"slices.{key}" for key in row if key not in ROW_KEYS}
    return sorted(unknown)


def _optional_number(data: JsonObject, key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number or null, not {value!r}")
    return value


def stack_from_json(data: JsonObject) -> Stack:
    """Return the stack a state's JSON form holds.

    A section row without its own angles carries the stack's
    (``cutting_angles_deg`` on the stack, flat without), which is how a
    single-angle state is written.

    Parameters
    ----------
    data
        A form :func:`stack_to_json` writes, without its format version.

    Returns
    -------
    Stack

    Raises
    ------
    ValueError
        Naming any unknown key (a section row's as ``slices.key``), or a
        key that holds a value of the wrong type.
    """
    unknown = _unknown_keys(data)
    if unknown:
        raise ValueError(f"Unknown stack state key(s) {unknown}")
    shared = data.get(ANGLES_KEY)
    sections: list[Section] = []
    for row in _rows(data):
        if not isinstance(row.get(ANGLES_KEY), Mapping) and isinstance(
            shared, Mapping
        ):
            row = {**row, ANGLES_KEY: shared}
        sections.append(section_from_json(row))
    defaults = Stack()
    spec: JsonObject = object_at(data, "spec") if "spec" in data else {}
    notes = list(texts_at(data, "notes")) if "notes" in data else []
    submitted = flag_at(data, "submitted") if "submitted" in data else False
    return Stack(
        image_folder=_text_or(data, "image_folder", defaults.image_folder),
        atlas=_text_or(data, "atlas", defaults.atlas),
        plane=_text_or(data, "plane", defaults.plane),
        interval_mm=_optional_number(data, "interval_mm"),
        thickness_mm=_optional_number(data, "thickness_mm"),
        spec=spec,
        interval_breaks=[
            integer_at({"interval_breaks": value}, "interval_breaks")
            for value in _items(data, "interval_breaks")
        ],
        notes=notes,
        slices=sections,
        submitted=submitted,
        debrief=_text_or(data, "debrief", defaults.debrief),
    )


def _text_or(data: JsonObject, key: str, default: str) -> str:
    """Return a state's text under *key*, or *default* without it."""
    if key not in data:
        return default
    return text_at(data, key)


def _items(data: JsonObject, key: str) -> Sequence[Json]:
    """Return a state's list under *key*; an absent one is empty."""
    if key not in data:
        return ()
    return list_at(data, key)


def serialized_mixed_angles(data: JsonObject) -> bool:
    """Return whether a state's JSON form has sections of differing angles.

    Parameters
    ----------
    data
        A form :func:`stack_to_json` writes (a checkpoint's fields at
        format 3 or 4).

    Returns
    -------
    bool
        True when the stack's ``cutting_angles_deg`` is written null, as
        it is exactly then.
    """
    return ANGLES_KEY in data and data[ANGLES_KEY] is None


def write_checkpoint(state: Stack, path: str) -> None:
    """Write *state* to *path* atomically, versioned."""
    write_json_atomic(path, {FORMAT_KEY: STATE_FORMAT_VERSION, **stack_to_json(state)})


def state_paths(
    data: dict[str, Any], convert: Callable[[str], str],
) -> dict[str, Any]:
    """*data* (a state's fields) with every stored path passed through *convert*.

    The paths a state holds: each section's deformation ``record`` and its
    border trace's ``artifact_dir`` and ``artifact_paths``. Everything
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
        correction = row.get("border_trace")
        if isinstance(correction, dict):
            correction = dict(correction)
            if isinstance(correction.get("artifact_dir"), str):
                correction["artifact_dir"] = convert(correction["artifact_dir"])
            paths = correction.get("artifact_paths")
            if isinstance(paths, dict):
                correction["artifact_paths"] = {
                    key: convert(value) if isinstance(value, str) else value
                    for key, value in paths.items()}
            row["border_trace"] = correction
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


def older_format(version: int) -> ValueError:
    """Return the refusal of an older checkpoint no migration reads.

    Parameters
    ----------
    version
        The checkpoint's format.

    Returns
    -------
    ValueError
        Its message asks for a new job.
    """
    return ValueError(
        "This checkpoint was made by an older pre-release LangSlice "
        f"(format {version}; this version reads format "
        f"{STATE_FORMAT_VERSION}). Start a new job: open it fresh "
        "(`--no-resume`, resume=False) or remove the job folder."
    )


def current_state(data: dict[str, Any], upgrade: Upgrade | None = None) -> dict[str, Any]:
    """A checkpoint's state fields, without the version key, read through
    *upgrade* (the job's migrations, which refuse an older format they do
    not read); ``ValueError`` without it for any format but
    :data:`STATE_FORMAT_VERSION`, and for a newer one."""
    version = data.get(FORMAT_KEY, 0)
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise ValueError(f"Unreadable checkpoint format_version {version!r}")
    if version > STATE_FORMAT_VERSION:
        raise ValueError(
            f"This checkpoint was written by a newer LangSlice (format {version}; this "
            f"version reads format {STATE_FORMAT_VERSION}). Update LangSlice to open it.")
    fields = {key: data[key] for key in data if key != FORMAT_KEY}
    if upgrade is not None:
        return dict(upgrade(version, fields))
    if version != STATE_FORMAT_VERSION:
        raise older_format(version)
    return fields


def read_checkpoint(path: str, upgrade: Upgrade | None = None) -> dict[str, Any] | None:
    """A checkpoint's state fields (:func:`current_state`, an older one
    through *upgrade*); ``None`` if there is no file."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a stack state")
    return current_state(data, upgrade)

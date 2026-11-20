"""The state checkpoint, ``state.json``: the job's record of truth.

This module is the one place a :class:`~langslice.core.stack.stack.Stack`
becomes JSON and comes back (:func:`stack_to_json`,
:func:`stack_from_json`); the stack and its sections declare their own
JSON form (:mod:`langslice.core.stack.part`). Every write saves the full
stack there, atomically (temp file + rename) so a crash mid-write cannot
leave a truncated file. Resume = load the checkpoint and re-seed the agent
with the state it had. Several writers may share one job folder; they take
turns under its write lock (:meth:`langslice.job.job.Job.writing`).

The file carries :data:`STATE_FORMAT_VERSION` under ``format_version`` next
to the state's own fields. A checkpoint of format 5 is read as format 6
(:func:`state_5_to_6`: every section gets its position's source); an older
one was made by a pre-release LangSlice and is refused, as is a newer one,
which asks for a newer LangSlice.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple, cast

from langslice.core.jsonable import JsonObject, JsonValue, to_jsonable
from langslice.core.shared_files import write_json_atomic
from langslice.core.stack.section import FLAT, INPUT, WRITTEN
from langslice.core.stack.stack import Stack

FORMAT_KEY = "format_version"
STATE_FORMAT_VERSION = 6
OLDEST_FORMAT = 5
ANGLES_KEY = "cutting_angles_deg"
OPTIONAL_ROW_KEYS = ("layers", "masks", "images")


def json_object(text: str) -> JsonObject | None:
    """Return the JSON object a text holds; None for any other JSON value.

    Parameters
    ----------
    text
        One JSON value written as text.

    Returns
    -------
    JsonObject or None
        The object.

    Raises
    ------
    ValueError
        When the text is not JSON.
    """
    data = cast("object", json.loads(text))
    if not isinstance(data, dict):
        return None
    return cast("JsonObject", data)


def stack_to_json(state: Stack) -> dict[str, JsonValue]:
    """Return a stack in its JSON form: the checkpoint's and the result's.

    The cutting angles are written once, on the stack
    (``"cutting_angles_deg": {"pitch", "yaw"}``), when every section shares
    them; when they differ the stack's is null and each section row carries
    its own. A section row leaves out its empty ``layers``, ``masks`` and
    ``images``.

    Parameters
    ----------
    state
        The stack written.

    Returns
    -------
    dict
        The stack's fields in declared order, then ``cutting_angles_deg``.
    """
    data = state.to_json()
    rows = cast("list[dict[str, JsonValue]]", data["slices"])
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
    data[ANGLES_KEY] = angles
    return data


def stack_fields(state: Stack) -> dict[str, JsonValue]:
    """Return the stack's own fields in their JSON form, without its sections.

    Parameters
    ----------
    state
        The stack read.

    Returns
    -------
    dict
        The stack's fields but ``slices``.
    """
    return state.model_dump(mode="json", exclude={"slices"}, fallback=to_jsonable)


def set_stack_fields(state: Stack, fields: JsonObject) -> None:
    """Put some of the stack's own fields back, from their JSON form.

    Parameters
    ----------
    state
        The stack written, in place.
    fields
        Keys of the stack's fields but ``slices``, as :func:`stack_fields`
        gives them.
    """
    read = stack_from_json(fields)
    for key in fields:
        setattr(state, key, getattr(read, key))


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
        Naming any unknown key, or a key that holds a value of the wrong
        type.
    """
    shared = data.get(ANGLES_KEY)
    fields: dict[str, object] = {key: value for key, value in data.items() if key != ANGLES_KEY}
    rows = fields.get("slices")
    if isinstance(shared, Mapping) and isinstance(rows, list):
        fields["slices"] = [
            {**row, ANGLES_KEY: shared}
            if isinstance(row, Mapping) and not isinstance(row.get(ANGLES_KEY), Mapping)
            else row
            for row in cast("list[object]", rows)
        ]
    return Stack.from_json(fields)


def write_checkpoint(state: Stack, path: str) -> None:
    """Write *state* to *path* atomically, versioned."""
    write_json_atomic(path, {FORMAT_KEY: STATE_FORMAT_VERSION, **stack_to_json(state)})


def state_paths(
    data: dict[str, Any],
    convert: Callable[[str], str],
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
                    for key, value in paths.items()
                }
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
                except ValueError:
                    continue
                if inside:
                    return os.path.relpath(spelled, base).replace(os.sep, "/")
        return value

    return convert


def older_format(version: int) -> ValueError:
    """Return the refusal of a checkpoint older than the oldest format read.

    Parameters
    ----------
    version
        The checkpoint's format.

    Returns
    -------
    ValueError
        Its message says a pre-release LangSlice made it and it cannot be
        opened.
    """
    return ValueError(
        "This job folder was made by a pre-release LangSlice (state format "
        f"{version}; this version reads formats {OLDEST_FORMAT} and "
        f"{STATE_FORMAT_VERSION}) and cannot be opened. Start a new job in "
        "another folder."
    )


def state_5_to_6(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a format-5 state's fields (or a history line's ``slices``) as format 6.

    Every section gets who set its position (``position_source``): the host
    when the job's ``inputs.positions`` names it, else a write. A history
    line's section without a position keeps none.

    Parameters
    ----------
    data
        The fields at format 5; not changed.

    Returns
    -------
    dict

    Raises
    ------
    ValueError
        For a whole state with a section that has no position.
    """
    fields = dict(data)
    rows = fields.get("slices")
    if not rows:
        return fields
    spec = fields.get("spec")
    inputs = spec.get("inputs") if isinstance(spec, Mapping) else None
    positions = inputs.get("positions") if isinstance(inputs, Mapping) else None
    supplied = {str(name) for name in positions} if isinstance(positions, Mapping) else set()
    migrated: list[dict[str, Any]] = []
    for row in rows:
        section = {key: value for key, value in row.items() if key != "position_source"}
        if section.get("position_mm") is None:
            if spec is not None:
                raise ValueError(f"section {section.get('id')} has no position (state format 5)")
            section.pop("position_mm", None)
        else:
            section["position_source"] = INPUT if str(section.get("id")) in supplied else WRITTEN
        migrated.append(section)
    fields["slices"] = migrated
    return fields


def upgraded_row(version: int, row: JsonObject) -> JsonObject:
    """Return a history line's section row at the current state format.

    Parameters
    ----------
    version
        The line's ``format_version``.
    row
        The line's section row.

    Returns
    -------
    JsonObject
        *row* itself when it is at the current format.
    """
    if version == STATE_FORMAT_VERSION:
        return row
    return state_5_to_6({"slices": [row]})["slices"][0]


class Checkpoint(NamedTuple):
    """Hold a checkpoint's state fields at the current format.

    Parameters
    ----------
    fields
        The state's fields, without the format version.
    upgraded
        Whether the file was of an older format.
    """

    fields: dict[str, Any]
    upgraded: bool


def checkpoint_of(data: Mapping[str, Any]) -> Checkpoint:
    """Return a state file's JSON as the current format.

    Parameters
    ----------
    data
        The file's JSON, with its format version.

    Returns
    -------
    Checkpoint

    Raises
    ------
    ValueError
        For a format older than :data:`OLDEST_FORMAT` (a pre-release
        LangSlice's), a newer one, or an unreadable version.
    """
    version = data.get(FORMAT_KEY, 0)
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise ValueError(f"Unreadable checkpoint format_version {version!r}")
    if version > STATE_FORMAT_VERSION:
        raise ValueError(
            f"This checkpoint was written by a newer LangSlice (format {version}; this "
            f"version reads format {STATE_FORMAT_VERSION}). Update LangSlice to open it."
        )
    if version < OLDEST_FORMAT:
        raise older_format(version)
    fields = {key: data[key] for key in data if key != FORMAT_KEY}
    if version < STATE_FORMAT_VERSION:
        return Checkpoint(state_5_to_6(fields), True)
    return Checkpoint(fields, False)


def viewer_stack(data: Mapping[str, Any]) -> Stack:
    """Return the stack a ``state.json`` holds, read as a viewer reads it.

    A format-5 file is read as format 6 in memory; nothing is written.

    Parameters
    ----------
    data
        The file's JSON, with its format version.

    Returns
    -------
    Stack

    Raises
    ------
    ValueError
        For a format that is not read, or a state that does not load.
    """
    return stack_from_json(checkpoint_of(data).fields)


def read_checkpoint(path: str) -> Checkpoint | None:
    """Return a checkpoint's state fields (:func:`checkpoint_of`); None if there is no file."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a stack state")
    return checkpoint_of(cast("dict[str, Any]", data))

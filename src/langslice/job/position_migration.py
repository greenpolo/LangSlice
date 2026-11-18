"""The migration that gives every section of an older state a position.

From state format 6 a section's position is always a number and says who
set it (``position_source``). :func:`state_5_to_6` reads a format-5
state: a section without a position gets its initial position (the job's
start and spacing, :func:`~langslice.core.stack.stack.initial_positions`)
and ``initial``; a placed one is ``input`` when the job's
``inputs.positions`` names it, else ``written``. A history line's section
without a position takes the one its checkpoint was given, else keeps the
section's current position.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from langslice.core.jsonable import JsonObject, JsonValue, to_jsonable
from langslice.core.stack.json_reads import (
    integer_at,
    number_at,
    object_at,
    objects_at,
)
from langslice.core.stack.section import INITIAL, INPUT, WRITTEN
from langslice.core.stack.stack import initial_positions

if TYPE_CHECKING:
    from langslice.job.migrations import FolderMigration

Fields = dict[str, JsonValue]


def state_5_to_6(data: Fields, folder: FolderMigration) -> Fields:
    """Return a format-5 state's fields as format 6 (the table's row).

    Parameters
    ----------
    data, folder
        The fields at format 5, not changed, and the job folder's
        migration, which keeps the initial positions a checkpoint was
        given for its history lines.

    Returns
    -------
    dict
    """
    fields = dict(data)
    if not fields.get("slices"):
        return fields
    rows = list(objects_at(fields, "slices"))
    spec = _spec(fields)
    unplaced = any(row.get("position_mm") is None for row in rows)
    if spec is not None and unplaced:
        _remember_initial(fields, spec, rows, folder)
    supplied = _supplied_positions(spec)
    migrated: list[JsonValue] = []
    for row in rows:
        section = {k: v for k, v in row.items() if k != "position_source"}
        name = str(section.get("id"))
        if section.get("position_mm") is not None:
            source = INPUT if name in supplied else WRITTEN
            section["position_source"] = source
        elif name in folder.initial:
            section["position_mm"] = folder.initial[name]
            section["position_source"] = INITIAL
        else:
            section.pop("position_mm", None)
        migrated.append(to_jsonable(section))
    fields["slices"] = migrated
    return fields


def _spec(fields: JsonObject) -> JsonObject | None:
    """Return a whole state's settings; None for a history line's row."""
    if fields.get("spec") is None:
        return None
    return object_at(fields, "spec")


def _remember_initial(
    fields: JsonObject,
    spec: JsonObject,
    rows: list[JsonObject],
    folder: FolderMigration,
) -> None:
    """Keep the initial position of each section without one."""
    ordered = sorted(rows, key=lambda row: integer_at(row, "index_corrected"))
    position = spec.get("position")
    start = None
    if isinstance(position, dict):
        start = _optional_number(cast("JsonObject", position), "start_mm")
    spacing = _optional_number(fields, "interval_mm")
    bounds = folder.position_range(fields)
    places = initial_positions(len(ordered), bounds, start, spacing)
    for row, place in zip(ordered, places, strict=True):
        if row.get("position_mm") is None:
            folder.initial[str(row.get("id"))] = round(place, 4)


def _optional_number(data: JsonObject, key: str) -> float | None:
    """Return the number under *key*; None when it is absent or null."""
    if data.get(key) is None:
        return None
    return number_at(data, key)


def _supplied_positions(spec: JsonObject | None) -> set[str]:
    """Return the sections the settings' ``inputs.positions`` names."""
    inputs = None if spec is None else spec.get("inputs")
    if not isinstance(inputs, dict):
        return set()
    positions = cast("JsonObject", inputs).get("positions")
    if not isinstance(positions, dict):
        return set()
    return {str(name) for name in cast("JsonObject", positions)}

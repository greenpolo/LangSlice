"""Undo and redo: one step of the job's history, either way."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langslice.job.job import Job


@dataclass(frozen=True)
class Stepped:
    """What :func:`undo` or :func:`redo` did."""

    #: False when there was no step to take (nothing changed).
    done: bool
    #: The steps left on that side afterwards (undo: undo steps; redo: redo).
    depth: int = 0
    #: What the step was: ``{"sections": the sections it touched, "fields":
    #: what changed}`` (:func:`describe_step`).
    step: dict[str, Any] = field(default_factory=dict)


#: The section fields each tool group writes, in the order they are told
#: apart: a placement change clears the deformation too, so it comes first.
_PLACEMENT = ("position_mm", "index_corrected")
_LINEAR = ("flip", "rotation_deg", "transform")
_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("placement", _PLACEMENT), ("linear", _LINEAR),
    ("damage", ("damaged_regions", "damage_note")), ("landmarks", ("landmarks",)),
    ("deformation", ("deformation",)),
)


def describe_step(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
    """The step that took state *old* to state *new* (both ``to_dict``
    snapshots): the sections it touched and the field groups it changed. A
    deformation that a placement or transform change cleared is part of that
    change."""
    held = {row["id"]: row for row in old.get("slices", [])}
    sections: list[str] = []
    groups: list[str] = []
    for row in new.get("slices", []):
        before = held.get(row["id"])
        if before is None:
            continue
        changed = [name for name, keys in _FIELDS if any(before.get(k) != row.get(k)
                                                          for k in keys)]
        if changed:
            sections.append(row["id"])
            groups += [name for name in changed if name not in groups]
    if old.get("interval_breaks") != new.get("interval_breaks") and "placement" not in groups:
        groups.append("placement")
    if old.get("cutting_angles_deg") != new.get("cutting_angles_deg") and "placement" not in groups:
        groups.append("placement")
    appearance = old.get("appearance") != new.get("appearance")
    if appearance and not groups:
        groups.append("appearance")
    if old.get("notes") != new.get("notes") and not groups:
        groups = ["notes"]
    return {"sections": sections, "fields": groups}


def undo(job: Job) -> Stepped:
    """Restore the state before the last write: one step back."""
    before = job.snapshot()
    if not job.undo():
        return Stepped(done=False, depth=len(job.undo_stack))
    return Stepped(done=True, depth=len(job.undo_stack), step=describe_step(job.snapshot(), before))


def redo(job: Job) -> Stepped:
    """Re-apply the write :func:`undo` reversed."""
    before = job.snapshot()
    if not job.redo():
        return Stepped(done=False, depth=len(job.redo_stack))
    return Stepped(done=True, depth=len(job.redo_stack), step=describe_step(before, job.snapshot()))

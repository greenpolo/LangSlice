"""Undo and redo: steps of the job's history, either way (one by default)."""

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
    #: What the steps were together: ``{"sections": the sections they
    #: touched, "fields": what changed}`` (:func:`describe_step`).
    step: dict[str, Any] = field(default_factory=dict)
    #: How many steps were taken (fewer than asked when the history held fewer).
    steps: int = 0


#: The section fields each tool group writes, in the order they are told
#: apart: a placement change clears the deformation too, so it comes first.
_PLACEMENT = ("position_mm", "index_corrected")
_LINEAR = ("flip", "rotation_deg", "transform")
_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("placement", _PLACEMENT), ("linear", _LINEAR),
    ("damage", ("damaged_regions", "damage_note")), ("landmarks", ("landmarks",)),
    ("deformation", ("deformation",)), ("images", ("images",)), ("masks", ("masks",)),
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
    if old.get("notes") != new.get("notes") and not groups:
        groups = ["notes"]
    return {"sections": sections, "fields": groups}


def _steps(job: Job, step: Any, steps: int) -> tuple[dict[str, Any], int]:
    """Take up to *steps* history steps with *step* (``job.undo`` or
    ``job.redo``) under one hold of the job's lock: the state before them
    and how many were taken."""
    with job.writing():
        before = job.snapshot()
        taken = 0
        while taken < max(1, int(steps)) and step():
            taken += 1
    return before, taken


def undo(job: Job, steps: int = 1) -> Stepped:
    """Restore the state before the last write: *steps* steps back (as many
    as the history holds when it holds fewer)."""
    before, taken = _steps(job, job.undo, steps)
    if not taken:
        return Stepped(done=False, depth=len(job.undo_stack))
    return Stepped(done=True, depth=len(job.undo_stack),
                   step=describe_step(job.snapshot(), before), steps=taken)


def redo(job: Job, steps: int = 1) -> Stepped:
    """Re-apply the writes :func:`undo` reversed, *steps* of them."""
    before, taken = _steps(job, job.redo, steps)
    if not taken:
        return Stepped(done=False, depth=len(job.redo_stack))
    return Stepped(done=True, depth=len(job.redo_stack),
                   step=describe_step(before, job.snapshot()), steps=taken)

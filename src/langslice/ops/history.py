"""Undo and redo: one step of the job's history, either way."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langslice.core.state import StackState
    from langslice.job.job import Job


@dataclass(frozen=True)
class Stepped:
    """What :func:`undo` or :func:`redo` did."""

    #: False when there was no step to take (nothing changed).
    done: bool
    #: Sections whose position the step moved.
    moved: list[str] = field(default_factory=list)
    #: The steps left on that side afterwards (undo: undo steps; redo: redo).
    depth: int = 0
    #: What the step was: ``{"tool": the tool that wrote it (None when the
    #: change fits none), "sections": the sections it touched, "fields":
    #: what changed}`` (:func:`describe_step`).
    step: dict[str, Any] = field(default_factory=dict)


def moved_positions(before: Mapping[str, Any], state: StackState) -> list[str]:
    """Sections whose position differs between *before* (a snapshot) and *state*."""
    held = {row["id"]: row.get("position_mm") for row in before.get("slices", [])}
    return [record.id for record in state.slices if held.get(record.id) != record.position_mm]


#: The section fields each tool group writes, in the order they are told
#: apart: a placement change clears the deformation too, so it comes first.
_PLACEMENT = ("position_mm", "index_corrected", "position_source")
_LINEAR = ("flip", "rotation_deg", "transform")
_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("placement", _PLACEMENT), ("linear", _LINEAR),
    ("damage", ("damaged_regions", "damage_note")), ("landmarks", ("landmarks",)),
    ("deformation", ("deformation",)),
)


def describe_step(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
    """The step that took state *old* to state *new* (both ``to_dict``
    snapshots): the tool it was, the sections it touched and the field groups
    it changed. The tool is read off what changed (a step stores no tool
    name, so it is the same after a reload and for background work): a
    placement change is ``position_sections``, a transform ``elastix_affine``
    or ``interactive_transform`` by its record, damage ``mark_damage``, pairs
    ``set_landmarks``, a deformation the method that made it, appearance the
    channel tools and a note ``note``. A change that fits none has ``tool``
    None. A deformation that a placement or transform change cleared is part
    of that change."""
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
    tool: str | None = None
    if "placement" in groups:
        tool = "position_sections"
    elif "linear" in groups:
        kinds = {((row.get("transform") or {}).get("kind")) for row in new.get("slices", [])
                 if row["id"] in sections}
        tool = "elastix_affine" if kinds == {"elastix"} else "interactive_transform"
    elif "damage" in groups:
        tool = "mark_damage"
    elif "landmarks" in groups:
        tool = "set_landmarks"
    elif "deformation" in groups:
        methods = {_method(row, held[row["id"]]) for row in new.get("slices", [])
                   if row["id"] in sections}
        tool = methods.pop() if len(methods) == 1 else None
    appearance = old.get("appearance") != new.get("appearance")
    if appearance and not groups:
        keys = {key for key in {*old.get("appearance", {}), *new.get("appearance", {})}
                if old.get("appearance", {}).get(key) != new.get("appearance", {}).get(key)}
        tool = ("set_preprocessed_channel_properties" if keys == {"preprocessed"}
                else "set_channel_properties")
        groups.append("appearance")
    if old.get("notes") != new.get("notes") and not groups:
        tool, groups = "note", ["notes"]
    return {"tool": tool, "sections": sections, "fields": groups}


def _method(row: Mapping[str, Any], earlier: Mapping[str, Any]) -> str | None:
    """The tool that made the deformation *row* holds (else the one it
    replaced or lost), from its last step: a landmark application names its
    method, a trace fit carries its trace, any other fit is ``ants_syn``."""
    for record in (row.get("deformation"), earlier.get("deformation")):
        steps = (record or {}).get("steps") or []
        if not steps:
            continue
        step = steps[-1]
        if step.get("method"):
            return str(step["method"])
        if step.get("trace"):
            return "trace_borders"
        if step.get("engine") == "ants":
            return "ants_syn"
    return None


def undo(job: Job) -> Stepped:
    """Restore the state before the last write: one step back."""
    before = job.snapshot()
    if not job.undo():
        return Stepped(done=False, depth=len(job.undo_stack))
    return Stepped(done=True, moved=moved_positions(before, job.state),
                   depth=len(job.undo_stack), step=describe_step(job.snapshot(), before))


def redo(job: Job) -> Stepped:
    """Re-apply the write :func:`undo` reversed."""
    before = job.snapshot()
    if not job.redo():
        return Stepped(done=False, depth=len(job.redo_stack))
    return Stepped(done=True, moved=moved_positions(before, job.state),
                   depth=len(job.redo_stack), step=describe_step(before, job.snapshot()))

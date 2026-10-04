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


def moved_positions(before: Mapping[str, Any], state: StackState) -> list[str]:
    """Sections whose position differs between *before* (a snapshot) and *state*."""
    held = {row["id"]: row.get("position_mm") for row in before.get("slices", [])}
    return [record.id for record in state.slices if held.get(record.id) != record.position_mm]


def undo(job: Job) -> Stepped:
    """Restore the state before the last write: one step back."""
    before = job.snapshot()
    if not job.undo():
        return Stepped(done=False, depth=len(job.undo_stack))
    return Stepped(done=True, moved=moved_positions(before, job.state),
                   depth=len(job.undo_stack))


def redo(job: Job) -> Stepped:
    """Re-apply the write :func:`undo` reversed."""
    before = job.snapshot()
    if not job.redo():
        return Stepped(done=False, depth=len(job.redo_stack))
    return Stepped(done=True, moved=moved_positions(before, job.state),
                   depth=len(job.redo_stack))

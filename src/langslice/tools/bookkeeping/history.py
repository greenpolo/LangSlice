"""Undo and redo: steps of the job's history, either way (one by default).

:func:`undo` and :func:`redo` take the steps; the tools' functions
(``undo_tool``, ``redo_tool``) answer with what the steps changed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langslice.core.refusal import RefusalCode, Refused

if TYPE_CHECKING:
    from langslice.job.job import Job
    from langslice.tools.run import Run


@dataclass(frozen=True)
class HistoryStep:
    """Hold what history steps changed together (:func:`describe_step`)."""

    sections: list[str] = field(default_factory=list[str])
    fields: list[str] = field(default_factory=list[str])


@dataclass(frozen=True)
class Stepped:
    """What :func:`undo` or :func:`redo` did."""

    done: bool
    depth: int = 0
    step: HistoryStep = field(default_factory=HistoryStep)
    steps: int = 0


_PLACEMENT = ("position_mm", "index_corrected")
_LINEAR = ("flip", "rotation_deg", "transform")
_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("placement", _PLACEMENT),
    ("linear", _LINEAR),
    ("damage", ("damaged_regions", "damage_note")),
    ("landmarks", ("landmarks",)),
    ("deformation", ("deformation",)),
    ("images", ("images",)),
    ("masks", ("masks",)),
)


def describe_step(old: Mapping[str, Any], new: Mapping[str, Any]) -> HistoryStep:
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
        changed = [name for name, keys in _FIELDS if any(before.get(k) != row.get(k) for k in keys)]
        if changed:
            sections.append(row["id"])
            groups += [name for name in changed if name not in groups]
    if old.get("interval_breaks") != new.get("interval_breaks") and "placement" not in groups:
        groups.append("placement")
    if old.get("cutting_angles_deg") != new.get("cutting_angles_deg") and "placement" not in groups:
        groups.append("placement")
    if old.get("notes") != new.get("notes") and not groups:
        groups = ["notes"]
    return HistoryStep(sections=sections, fields=groups)


def _steps(job: Job, step: Any, steps: int) -> tuple[dict[str, Any], int]:
    """Take up to *steps* history steps with *step* (``job.undo.undo`` or
    ``job.undo.redo``) under one hold of the job's lock: the state before them
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
    before, taken = _steps(job, job.undo.undo, steps)
    if not taken:
        return Stepped(done=False, depth=len(job.undo.steps))
    return Stepped(
        done=True,
        depth=len(job.undo.steps),
        step=describe_step(job.snapshot(), before),
        steps=taken,
    )


def redo(job: Job, steps: int = 1) -> Stepped:
    """Re-apply the writes :func:`undo` reversed, *steps* of them."""
    before, taken = _steps(job, job.undo.redo, steps)
    if not taken:
        return Stepped(done=False, depth=len(job.undo.redo_steps))
    return Stepped(
        done=True,
        depth=len(job.undo.redo_steps),
        step=describe_step(before, job.snapshot()),
        steps=taken,
    )


def undo_tool(run: Run, *, steps: int = 1) -> dict[str, Any]:
    """Run the ``undo`` tool: take *steps* steps back.

    Parameters
    ----------
    run
        The run the tool is called in.
    steps
        The declared argument as the model sent it
        (``tools/descriptions/bookkeeping.py``).

    Returns
    -------
    dict
        The reply: what the steps undid and the rows they changed
        (:func:`_stepped_reply`).
    """
    count = _step_count(steps)
    step = undo(run.job, count)
    if not step.done:
        raise Refused(
            RefusalCode.PRECONDITION,
            "Nothing to undo: the job's history holds no earlier step.",
        )
    return _stepped_reply(run, "undone", "undo_depth", step, count)


def redo_tool(run: Run, *, steps: int = 1) -> dict[str, Any]:
    """Run the ``redo`` tool: take *steps* undone steps forward again.

    Parameters
    ----------
    run
        The run the tool is called in.
    steps
        The declared argument as the model sent it
        (``tools/descriptions/bookkeeping.py``).

    Returns
    -------
    dict
        The reply: what the steps redid and the rows they changed
        (:func:`_stepped_reply`).
    """
    count = _step_count(steps)
    step = redo(run.job, count)
    if not step.done:
        raise Refused(
            RefusalCode.PRECONDITION,
            "Nothing to redo: no step is undone since the last write "
            "(a write clears what was undone).",
        )
    return _stepped_reply(run, "redone", "redo_depth", step, count)


def _step_count(steps: object) -> int:
    """Return the tools' *steps* as a count; ``BAD_ARGUMENT`` for anything but a
    whole number from 1.
    """
    try:
        count = int(steps) if isinstance(steps, (int, float)) else 0
    except (OverflowError, ValueError):
        count = 0
    if count < 1 or count != steps:
        raise Refused(RefusalCode.BAD_ARGUMENT, "steps is a whole number, 1 or more.")
    return count


def _stepped_reply(
    run: Run,
    word: str,
    depth_key: str,
    step: Stepped,
    count: int,
) -> dict[str, Any]:
    """Return the reply to an undo or redo: the steps' sections and field
    groups under *word*, the depth under *depth_key*, past one step asked
    how many were taken, the rows those sections changed and, after a
    placement change, the cutting angles and the interval breaks.
    """
    info = step.step
    what = {"sections": info.sections, "fields": info.fields}
    result: dict[str, Any] = {
        "status": "ok",
        word: what,
        depth_key: step.depth,
    }
    if count > 1:
        result["steps"] = step.steps
        if step.steps < count:
            result["message"] = (
                f"{count} steps were asked and the history held "
                f"{step.steps}: all {step.steps} were taken."
            )
    if info.sections:
        rows = run.changed(info.sections)
        result["changed"] = rows["changed"]
        if "placement" in info.fields:
            result["cutting_angles_deg"] = rows["cutting_angles_deg"]
            result["interval_breaks"] = rows["interval_breaks"]
    return result

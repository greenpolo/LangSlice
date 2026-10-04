"""Where sections sit: positions along the slicing axis, and the stack's cutting angles."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langslice.linear.job import Job
    from langslice.linear.workspace import Workspace


@dataclass(frozen=True)
class PositionsWritten:
    """What :func:`set_positions` wrote, in the order asked."""

    #: ``(section id, position mm)`` as written (clamped).
    written: list[tuple[str, float]] = field(default_factory=list)
    #: ``(section id, requested mm, written mm)`` for each value moved into range.
    clamped: list[tuple[str, float, float]] = field(default_factory=list)
    #: References that name no section.
    unknown: list[str] = field(default_factory=list)
    #: The atlas range the values were clamped into, ``(low, high)`` mm.
    position_range: tuple[float, float] = (0.0, 0.0)

    @property
    def touched(self) -> list[str]:
        return [section_id for section_id, _value in self.written]


def set_positions(
    job: Job, workspace: Workspace, positions: Iterable[tuple[object, float]],
) -> PositionsWritten:
    """Write each ``(filename or corrected index, millimetres)`` pair.

    A value outside the atlas range (``workspace.position_range``) is
    clamped into it and reported. The same section twice is written twice
    (the last value stays). One undo step for the whole call; nothing is
    committed when nothing was written.
    """
    low, high = workspace.position_range
    before = job.snapshot()
    written: list[tuple[str, float]] = []
    clamped: list[tuple[str, float, float]] = []
    unknown: list[str] = []
    for ref, requested in positions:
        record = job.state.resolve(ref)
        if record is None:
            unknown.append(str(ref))
            continue
        value = min(high, max(low, requested))
        if value != requested:
            clamped.append((record.id, requested, value))
        record.position_mm = value
        written.append((record.id, value))
    if written:
        job.commit(before)
    return PositionsWritten(written=written, clamped=clamped, unknown=unknown,
                            position_range=(low, high))


def set_cutting_angles(job: Job, workspace: Workspace, pitch_deg: float, yaw_deg: float) -> None:
    """Set the stack-wide cutting angles (degrees); one undo step.

    Every render cached at the old angles is dropped.
    """
    before = job.snapshot()
    job.state.cutting_angles_deg = {"pitch": float(pitch_deg), "yaw": float(yaw_deg)}
    workspace.render_cache.clear()
    job.commit(before)

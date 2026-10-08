"""The ``status`` verb: the stack as it stands, as data.

Writes nothing and takes no undo step. The pictures are ``look``'s
(:mod:`langslice.ops.look`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langslice.core.state import break_ids
from langslice.core.status import stack_angles_entry, status_rows

if TYPE_CHECKING:
    from langslice.job.job import Job


@dataclass(frozen=True)
class StackStatus:
    """The stack as it stands: one row per section in corrected order
    (:func:`langslice.core.status.status_rows`), the cutting angles and the
    interval breaks (the filenames of the sections after each gap).
    ``cutting_angles_deg`` is the stack's ``{"pitch", "yaw"}``, or ``"per
    section"`` when the sections' angles differ (each row then carries its
    own)."""

    rows: list[dict[str, Any]]
    cutting_angles_deg: dict[str, float] | str
    interval_breaks: list[str]


def status(job: Job) -> StackStatus:
    """The status table (``status``). A section the user locked (its
    in-plane alignment done, ``inputs.locked``) carries ``locked: true``,
    one the user gave a damage note (``inputs.damaged``: a note, not damage)
    ``damage_by_user: true``; the others carry neither."""
    state = job.state
    rows = status_rows(state)
    for row in rows:
        if row["id"] in job.locked:
            row["locked"] = True
        if row["id"] in job.host_damaged:
            row["damage_by_user"] = True
    return StackStatus(rows=rows, cutting_angles_deg=stack_angles_entry(state),
                       interval_breaks=break_ids(state))

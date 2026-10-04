"""Corrected order: which section follows which, by filename."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langslice.ops.refusal import Refused

if TYPE_CHECKING:
    from langslice.core.state import SliceState
    from langslice.job.job import Job


@dataclass(frozen=True)
class Reordered:
    """What :func:`reorder` moved: the ids whose corrected index changed."""

    moved: list[str] = field(default_factory=list)

    @property
    def touched(self) -> list[str]:
        return list(self.moved)


def renumber(order: Sequence[SliceState]) -> list[str]:
    """Give *order* corrected indices 0..n-1; the ids whose index changed.

    Only ``index_corrected`` moves. Positions and transforms stay where they
    are; the submit gates are what hold order and position together. No
    undo step: a building block for :func:`reorder`.
    """
    moved: list[str] = []
    for index, record in enumerate(order):
        if record.index_corrected != index:
            moved.append(record.id)
        record.index_corrected = index
    return moved


def reorder(job: Job, slices: Sequence[str], after: str = "start") -> Reordered:
    """Move *slices* (filenames) as one block, in that order, after *after*.

    *after* is a filename outside the block, or ``"start"`` to put it first.
    Unlisted sections keep their relative order. Filenames only: a reorder
    changes corrected indices. One undo step; :class:`Refused` (nothing
    written) for unknown or repeated names or a bad anchor.
    """
    if (not isinstance(slices, list) or not slices
            or any(not isinstance(item, str) for item in slices)
            or not isinstance(after, str)):
        raise Refused("BAD_ARGS")
    state = job.state
    ids = {record.id for record in state.slices}
    selected = set(slices)
    unknown = sorted(selected - ids)
    if unknown:
        raise Refused("UNKNOWN_SLICE_IDS", unknown=unknown)
    if len(selected) != len(slices):
        raise Refused("DUPLICATE_SLICE_IDS")
    if after != "start" and after not in ids:
        raise Refused("UNKNOWN_SLICE_IDS", unknown=[after])
    if after in selected:
        raise Refused("BAD_ARGS", message="after must name a section outside slices")
    records = {record.id: record for record in state.slices}
    remaining = [record for record in state.in_order() if record.id not in selected]
    index = 0 if after == "start" else remaining.index(records[after]) + 1
    ordered = remaining[:index] + [records[name] for name in slices] + remaining[index:]
    before = job.snapshot()
    moved = renumber(ordered)
    job.commit(before)
    return Reordered(moved=moved)

"""Damage flags: sections whose outline cannot be trusted for an affine fit."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langslice.job.job import Job


@dataclass(frozen=True)
class DamageMarked:
    """What :func:`mark_damaged` set and cleared."""

    marked: list[str] = field(default_factory=list)
    unmarked: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    #: ``{"id", "error": "DAMAGE_SET_BY_USER"}``: a host flag cannot be cleared.
    rejected: list[dict[str, str]] = field(default_factory=list)

    @property
    def touched(self) -> list[str]:
        return [*self.marked, *self.unmarked]


def mark_damaged(job: Job, entries: Iterable[Mapping[str, Any]]) -> DamageMarked:
    """Set or clear the flag per ``{"id", "damaged"?: bool = True, "note"?}`` entry.

    Clearing drops the note too. A flag the host set cannot be cleared. One
    undo step, always taken.
    """
    state = job.state
    before = job.snapshot()
    marked: list[str] = []
    unmarked: list[str] = []
    unknown: list[str] = []
    rejected: list[dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        record = state.resolve(entry.get("id", ""))
        if record is None:
            unknown.append(str(entry.get("id", "")))
            continue
        if record.id in job.host_damaged and not entry.get("damaged", True):
            rejected.append({"id": record.id, "error": "DAMAGE_SET_BY_USER"})
            continue
        record.damaged = entry.get("damaged", True)
        record.damage_note = str(entry.get("note", "")).strip() if record.damaged else ""
        (marked if record.damaged else unmarked).append(record.id)
    job.commit(before)
    return DamageMarked(marked=marked, unmarked=unmarked, unknown=unknown, rejected=rejected)

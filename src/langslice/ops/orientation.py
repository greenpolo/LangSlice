"""Orientation: each section's flip and quarter turn, recorded as data."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langslice.linear.job import Job

#: The rotations a section may carry, degrees.
ROTATIONS: tuple[int, ...] = (0, 90, 180, 270)


@dataclass(frozen=True)
class Oriented:
    """What :func:`orient_sections` did."""

    #: Sections the call reached (written, or set to what they already had).
    applied: list[str] = field(default_factory=list)
    #: Sections whose orientation changed and whose transform was dropped.
    cleared_transforms: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    #: ``{"id", "error", ...}`` per refused part: ``LOCKED`` (the whole
    #: entry), ``FLIP_DISABLED`` or ``BAD_ROTATION`` (that key only).
    rejected: list[dict[str, Any]] = field(default_factory=list)

    @property
    def touched(self) -> list[str]:
        return list(self.applied)


def orient_sections(job: Job, entries: Iterable[Mapping[str, Any]]) -> Oriented:
    """Set flip and/or rotation per ``{"id", "flip"?, "rotate_deg"?}`` entry.

    Rotation is applied first, then the flip (the renderer's order). A
    missing or None key leaves that correction as it is. A section whose
    orientation changes loses its transform: a transform describes the
    section after its orientation. The host's locked sections are refused,
    and so is a flip when the job's spec turns flipping off. One undo step,
    always taken.
    """
    state = job.state
    before = job.snapshot()
    applied: list[str] = []
    unknown: list[str] = []
    rejected: list[dict[str, Any]] = []
    cleared: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        record = state.resolve(entry.get("id", ""))
        if record is None:
            unknown.append(str(entry.get("id", "")))
            continue
        if record.id in job.locked:
            rejected.append({"id": record.id, "error": "LOCKED"})
            continue
        was = (record.flip, record.rotation_deg)
        if "flip" in entry and entry["flip"] is not None:
            if not job.spec.transform.flip:
                rejected.append({"id": record.id, "error": "FLIP_DISABLED"})
            else:
                record.flip = bool(entry["flip"])
        if "rotate_deg" in entry and entry["rotate_deg"] is not None:
            try:
                rotation = int(entry["rotate_deg"]) % 360
            except (TypeError, ValueError):
                rotation = -1
            if rotation not in ROTATIONS:
                rejected.append({"id": record.id, "error": "BAD_ROTATION",
                                 "allowed": list(ROTATIONS)})
            else:
                record.rotation_deg = rotation
        if (record.flip, record.rotation_deg) != was and record.transform is not None:
            record.transform = None
            cleared.append(record.id)
        applied.append(record.id)
    job.commit(before)
    return Oriented(applied=applied, cleared_transforms=cleared, unknown=unknown,
                    rejected=rejected)

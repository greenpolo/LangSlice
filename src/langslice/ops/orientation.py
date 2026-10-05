"""Orientation: each section's flip and quarter turn, recorded as data."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langslice.core.state import ROTATIONS

if TYPE_CHECKING:
    from PIL import Image

    from langslice.core.display import DisplayOptions
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job


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
    #: With display options: the first sections reached (up to
    #: :data:`langslice.ops.views.MAX_VIEW_SLICES`) as they now stand, one
    #: picture each (:func:`langslice.ops.views.view_slices`), and
    #: ``{"id", "message"}`` per picture that failed (the write stands).
    pictures: list[Image.Image] = field(default_factory=list)
    render_failed: list[dict[str, str]] = field(default_factory=list)

    @property
    def touched(self) -> list[str]:
        return list(self.applied)


def orient_sections(
    job: Job, entries: Iterable[Mapping[str, Any]], *,
    workspace: Workspace | None = None, options: DisplayOptions | None = None,
) -> Oriented:
    """Set flip and/or rotation per ``{"id", "flip"?, "rotate_deg"?}`` entry.

    Rotation is applied first, then the flip (the renderer's order). A
    missing or None key leaves that correction as it is. A section whose
    orientation changes loses its transform: a transform describes the
    section after its orientation. The host's locked sections are refused,
    and so is a flip when the job's spec turns flipping off. One undo step,
    always taken. With *workspace* and *options*, the sections reached are
    pictured as they now stand.
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
    pictures: list[Image.Image] = []
    failed: list[dict[str, str]] = []
    if workspace is not None and options is not None:
        from langslice.ops.views import MAX_VIEW_SLICES, view_slices

        records = [record for name in applied[:MAX_VIEW_SLICES]
                   if (record := state.by_id(name)) is not None]
        shown = view_slices(job, workspace, records, options, keep_going=True)
        pictures, failed = shown.pictures, shown.failed
    return Oriented(applied=applied, cleared_transforms=cleared, unknown=unknown,
                    rejected=rejected, pictures=pictures, render_failed=failed)

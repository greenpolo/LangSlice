"""Damage marks: the atlas regions a section is missing, which every fit leaves out.

A section is damaged when it has marked regions (``SliceState.damaged_regions``)
or a mark that names none (``SliceState.damage_marked``: the host's
``inputs.damaged`` ``{filename: note}``, or a :func:`mark_damaged` flag).
:func:`mark_damage` sets a section's regions and note; the fits and the
image model's trace read them (:func:`langslice.core.damage.exclusions`).
The host's marks stay: the agent may add regions to them, never remove them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from PIL import Image

from langslice.core.damage import DamagePictureError, damage_picture, normalized_entries
from langslice.ops.refusal import Refused

if TYPE_CHECKING:
    from langslice.core.display import DisplayOptions
    from langslice.core.state import SliceState
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job


# --- mark_damage: by region -----------------------------------------------------------


@dataclass(frozen=True)
class DamageRegions:
    """What :func:`mark_damage` did to one section."""

    id: str
    #: The section's marked regions after the call.
    regions: list[str] = field(default_factory=list)
    #: The section's note after the call.
    note: str = ""
    #: Whether the section is still damaged (regions, or the host's mark).
    damaged: bool = False
    #: Whether the host's mark (which the agent cannot remove) is on it.
    by_user: bool = False
    #: Whether anything changed (False: the same mark again, no undo step).
    written: bool = False
    #: With display options and marked regions: the picture
    #: (:func:`langslice.core.damage.damage_picture`).
    pictures: list[Image.Image] = field(default_factory=list)
    #: ``{"error", "message"}`` when the picture could not be drawn (the
    #: write stands).
    render_failed: dict[str, str] | None = None

    @property
    def touched(self) -> list[str]:
        return [self.id] if self.written else []


def _user_note(job: Job, record: SliceState) -> str:
    """The note the host gave with its mark ("" for none, or no mark)."""
    marks = (job.spec.inputs or {}).get("damaged") or {}
    return str(marks.get(record.id) or "").strip() if record.id in job.host_damaged else ""


def _joined_note(user: str, agent: str) -> str:
    """The user's note, then the agent's (each once)."""
    parts = [text for text in (user, agent) if text]
    return "; ".join(dict.fromkeys(parts))


def mark_damage(
    job: Job,
    workspace: Workspace,
    section: object,
    regions: Sequence[str],
    note: str = "",
    *,
    options: DisplayOptions | None = None,
) -> DamageRegions:
    """Set *section*'s marked regions to *regions*, with *note*; one undo step.

    *regions* are atlas region entries (acronyms or ids, ``"CTX:left"`` for
    one side), checked against the atlas and stored normalized
    (:func:`langslice.core.damage.normalized_entries`). They replace the
    section's regions; empty clears the agent's marks (regions, its note and
    a :func:`mark_damaged` flag). A mark the host set stays: its note is
    kept first, the agent's after it, and clearing leaves the host's mark
    and note. The same mark again writes nothing (``written`` False).

    With *options* (its size and border style), the result carries the
    picture of the marked regions on the section and on the atlas
    (:func:`langslice.core.damage.damage_picture`); a picture that fails is
    ``render_failed``, the write standing.

    Refused (nothing written): ``UNKNOWN_SLICE_IDS``; ``BAD_ARGS`` (regions
    not a list, or a side that does not exist); ``UNKNOWN_REGIONS``;
    ``NO_SIDES`` (a side on a sagittal stack).
    """
    from langslice.core.atlas.sides import has_sides
    from langslice.core.deformable.atlas_images import resolve_entries

    if isinstance(regions, (str, bytes)) or not isinstance(regions, Sequence):
        raise Refused("BAD_ARGS", message="regions must be a list of acronyms or ids")
    try:
        names = normalized_entries(regions)
    except ValueError as exc:
        raise Refused("BAD_ARGS", message=str(exc)) from exc
    if names:
        try:
            resolve_entries(workspace.atlas, names)
        except ValueError as exc:
            raise Refused("UNKNOWN_REGIONS", message=str(exc)) from exc
        if job.state.plane == "sagittal" and has_sides(names):
            raise Refused("NO_SIDES", message="A sagittal section lies within one "
                          "hemisphere, so a region cannot be limited to one side.")
    with job.writing():
        state = job.state
        record = state.resolve(section)
        if record is None:
            raise Refused("UNKNOWN_SLICE_IDS", unknown=[str(section)])
        by_user = record.id in job.host_damaged
        user = _user_note(job, record)
        agent = str(note or "").strip() if names else ""
        after = (names, by_user, _joined_note(user, agent) if by_user else agent)
        written = after != (list(record.damaged_regions), record.damage_marked,
                            record.damage_note)
        if written:
            before = job.snapshot()
            record.damaged_regions, record.damage_marked, record.damage_note = (
                list(after[0]), after[1], after[2])
            job.commit(before)
        result = {"id": record.id, "regions": list(record.damaged_regions),
                  "note": record.damage_note, "damaged": record.damaged,
                  "by_user": by_user, "written": written}
    drawn: list[Image.Image] = []
    failed: dict[str, str] | None = None
    if options is not None and record.damaged_regions:
        try:
            drawn = [damage_picture(
                workspace, state, record, store=job.deformations,
                long_edge=options.long_edge, border_color=options.border_color,
                border_thickness=options.border_thickness)]
        except DamagePictureError as exc:
            failed = {"error": exc.code, "message": str(exc)}
        except Exception as exc:  # noqa: BLE001 - the write stands; say why
            failed = {"error": getattr(exc, "code", "RENDER_FAILED"), "message": str(exc)}
    return DamageRegions(**result, pictures=drawn, render_failed=failed)


# --- mark_damaged: the older flag -----------------------------------------------------


@dataclass(frozen=True)
class DamageMarked:
    """What :func:`mark_damaged` set and cleared."""

    marked: list[str] = field(default_factory=list)
    unmarked: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    #: ``{"id", "error": "DAMAGE_SET_BY_USER"}``: a host mark cannot be cleared.
    rejected: list[dict[str, str]] = field(default_factory=list)

    @property
    def touched(self) -> list[str]:
        return [*self.marked, *self.unmarked]


def mark_damaged(job: Job, entries: Iterable[Mapping[str, Any]]) -> DamageMarked:
    """Set or clear a mark per ``{"id", "damaged"?: bool = True, "note"?}`` entry.

    Setting is a mark that names no regions (``damage_marked``, the note
    replacing the section's); clearing removes the section's whole mark,
    its regions and note too. A mark the host set cannot be cleared. One
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
        damaged = bool(entry.get("damaged", True))
        if record.id in job.host_damaged and not damaged:
            rejected.append({"id": record.id, "error": "DAMAGE_SET_BY_USER"})
            continue
        record.damage_marked = damaged
        record.damage_note = str(entry.get("note", "")).strip() if damaged else ""
        if not damaged:
            record.damaged_regions = []
        (marked if damaged else unmarked).append(record.id)
    job.commit(before)
    return DamageMarked(marked=marked, unmarked=unmarked, unknown=unknown, rejected=rejected)

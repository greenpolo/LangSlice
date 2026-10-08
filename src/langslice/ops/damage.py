"""Damage marks: the atlas regions a section is missing, which every fit leaves out.

A section is damaged exactly when it has marked regions
(``SliceState.damaged_regions``). :func:`mark_damage` sets a section's
regions and note; the fits and the image model's trace read them
(:func:`langslice.core.damage.exclusions`). A host's note
(``inputs.damaged``, ``{filename: note}``) is the section's
``damage_note`` alone: it stays first in the note, and the agent marks the
regions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from PIL import Image

from langslice.core.damage import DamagePictureError, damage_picture, normalized_entries
from langslice.ops.refusal import Refused, unknown_sections

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
    #: Whether the section is damaged after the call (it has marked regions).
    damaged: bool = False
    #: Whether the note starts with the host's note (``inputs.damaged``).
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
    section's regions; empty clears them and the agent's note. A note the
    host gave stays first, the agent's after it, and clearing leaves the
    host's note. The same mark again writes nothing (``written`` False).

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
            raise unknown_sections(state, [str(section)])
        by_user = record.id in job.host_damaged
        user = _user_note(job, record)
        agent = str(note or "").strip() if names else ""
        after = (names, _joined_note(user, agent) if by_user else agent)
        written = after != (list(record.damaged_regions), record.damage_note)
        if written:
            before = job.snapshot()
            record.damaged_regions, record.damage_note = list(after[0]), after[1]
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

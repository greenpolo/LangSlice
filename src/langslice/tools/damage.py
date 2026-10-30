"""Damage marks: the atlas regions a section is missing, which every fit leaves out.

A section is damaged exactly when it has marked regions
(``Section.damaged_regions``). :func:`mark_damage` sets a section's regions
and note; the fits and the image model's trace read them
(:func:`langslice.core.sections.damage.exclusions`). A host's note
(``inputs.damaged``, ``{filename: note}``) is the section's ``damage_note``
alone: it stays first in the note, and the agent marks the regions. The
tool's function (``mark_damage_tool``) adds its row and the picture of the
marked regions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from PIL import Image

from langslice.core.pictures import drawn as pictures_drawn
from langslice.core.pictures.media_keys import TOOL_MEDIA_PARTS_KEY
from langslice.core.pictures.options import default_options
from langslice.core.pictures.sizes import picture_edge
from langslice.core.refusal import RefusalCode, Refused
from langslice.core.sections.damage import (
    damage_picture,
    normalized_entries,
)
from langslice.tools.refusal import check_frozen, unknown_sections
from langslice.tools.run import as_list

if TYPE_CHECKING:
    from langslice.core.pictures.options import DisplayOptions
    from langslice.core.stack.section import Section
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job
    from langslice.tools.run import Run


# --- mark_damage: by region --------------------------------------------------


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
    #: (:func:`langslice.core.sections.damage.damage_picture`).
    pictures: list[Image.Image] = field(default_factory=list)
    #: ``{"error", "message"}`` when the picture could not be drawn (the
    #: write stands).
    render_failed: dict[str, str] | None = None

    @property
    def touched(self) -> list[str]:
        return [self.id] if self.written else []


def _user_note(job: Job, record: Section) -> str:
    """The note the host gave with its mark ("" for none, or no mark)."""
    marks = (job.spec.inputs or {}).get("damaged") or {}
    return (
        str(marks.get(record.id) or "").strip()
        if record.id in job.host_damaged
        else ""
    )


def _joined_note(user: str, agent: str) -> str:
    """The user's note, then the agent's (each once)."""
    parts = [text for text in (user, agent) if text]
    return "; ".join(dict.fromkeys(parts))


def _check_sides(
    job: Job, workspace: Workspace, record: Section, names: Sequence[str]
) -> None:
    """Refuse a one-sided entry whose side the section's current placement does
    not have (the picture's check, made before the write); a section without a
    usable placement is not checked."""
    from langslice.core.atlas.sides import split_side
    from langslice.core.fits.deformable.plane_images import placement_left
    from langslice.core.fits.deformation import fit_grid

    try:
        placement = fit_grid(job.state, workspace, record).placement
    except ValueError:
        return
    atlas = workspace.atlas  # pyright: ignore[reportAny]  # BrainGlobe's, untyped
    left = placement_left(atlas, placement, names)
    pair = getattr(left, "names", None)
    for name in names:
        side = split_side(name)[1]
        if side is not None and pair is not None and side not in pair:
            runs = (
                "vertically" if side in ("top", "bottom") else "horizontally"
            )
            raise Refused(
                RefusalCode.WRONG_SIDE_PAIR,
                message=(
                    f"On this section the midline runs {runs}, so its halves are "
                    f"{pair[0]!r} and {pair[1]!r}, not {side!r}: name one of those "
                    f'(e.g. "{split_side(name)[0]}:{pair[0]}").'
                ),
            )


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

    *regions* are atlas region entries (acronyms or ids, ``"CTX:left"`` for one
    side), checked against the atlas and stored normalized
    (:func:`langslice.core.sections.damage.normalized_entries`). They replace
    the section's regions; empty clears them and the agent's note. A note the
    host gave stays first, the agent's after it, and clearing leaves the host's
    note. The same mark again writes nothing (``written`` False).

    With *options* (its size and border style), the result carries the picture
    of the marked regions on the section and on the atlas
    (:func:`langslice.core.sections.damage.damage_picture`); a picture that
    fails is ``render_failed``, the write standing.

    Refused (nothing written): ``UNKNOWN_SLICE_IDS``; ``SECTION_BUSY``; ``BAD_ARGS`` (regions
    not a list, or a side that does not exist); ``UNKNOWN_REGIONS``;
    ``NO_SIDES`` (a side on a sagittal stack); ``WRONG_SIDE_PAIR`` (a side the
    section's placement does not have: ``left``/``right`` where its midline runs
    horizontally, ``top``/``bottom`` where it runs vertically), and the other
    side refusals of :func:`langslice.core.atlas.sides.native_left`.
    """
    from langslice.core.atlas.sides import has_sides
    from langslice.core.fits.deformable.plane_images import resolve_entries

    if isinstance(regions, (str, bytes)) or not isinstance(regions, Sequence):
        raise Refused(
            RefusalCode.BAD_ARGS,
            message="regions must be a list of acronyms or ids",
        )
    try:
        names = normalized_entries(regions)
    except ValueError as exc:
        raise Refused(RefusalCode.BAD_ARGS, message=str(exc)) from exc
    if names:
        try:
            resolve_entries(workspace.atlas, names)
        except ValueError as exc:
            raise Refused(
                RefusalCode.UNKNOWN_REGIONS, message=str(exc)
            ) from exc
        if job.state.plane == "sagittal" and has_sides(names):
            raise Refused(
                RefusalCode.NO_SIDES,
                message="A sagittal section lies within one "
                "hemisphere, so a region cannot be limited to one side.",
            )
    with job.writing():
        state = job.state
        record = state.resolve(section)
        if record is None:
            raise unknown_sections(state, [str(section)])
        check_frozen(
            job, [record.id]
        )  # no frozen property: refused only while work runs on it
        if has_sides(names):
            _check_sides(job, workspace, record, names)
        by_user = record.id in job.host_damaged
        user = _user_note(job, record)
        agent = str(note or "").strip() if names else ""
        after = (names, _joined_note(user, agent) if by_user else agent)
        written = after != (list(record.damaged_regions), record.damage_note)
        if written:
            before = job.snapshot()
            record.damaged_regions, record.damage_note = (
                list(after[0]),
                after[1],
            )
            job.commit(before)
        result = {
            "id": record.id,
            "regions": list(record.damaged_regions),
            "note": record.damage_note,
            "damaged": record.damaged,
            "by_user": by_user,
            "written": written,
        }
    drawn: list[Image.Image] = []
    failed: dict[str, str] | None = None
    if options is not None and record.damaged_regions:
        try:
            drawn = [
                damage_picture(
                    workspace,
                    state,
                    record,
                    store=job.deformations,
                    long_edge=options.long_edge,
                    border_color=options.border_color,
                    border_thickness=options.border_thickness,
                )
            ]
        except Exception as exc:  # noqa: BLE001 - the write stands; say why
            failed = {
                "error": getattr(exc, "code", RefusalCode.RENDER_FAILED),
                "message": str(exc),
            }
    return DamageRegions(**result, pictures=drawn, render_failed=failed)


# --- the tool ----------------------------------------------------------------


def mark_damage_tool(
    run: Run, section: str, regions: list[str], note: str
) -> dict[str, Any]:
    """Run the ``mark_damage`` tool: write, answer, picture.

    Parameters
    ----------
    run
        The run the tool belongs to.
    section
        The section's filename.
    regions
        The atlas regions it lost; empty clears them.
    note
        What the damage is.

    Returns
    -------
    dict
        The reply: the section's regions and note after the call, the row
        the write changed, and the picture of the marked regions on the
        section and on the atlas.
    """
    edge = picture_edge(run.workspace)
    options = default_options("overlay", long_edge=edge)
    done = mark_damage(
        run.job,
        run.workspace,
        section,
        as_list(regions),
        note,
        options=options,
    )

    result: dict[str, Any] = {
        "status": "ok",
        "id": done.id,
        "regions": done.regions,
        "note": done.note,
        "damaged": done.damaged,
        "written": done.written,
    }
    if done.render_failed:
        result["render_failed"] = done.render_failed
    if done.touched:
        result["changed"] = run.changed(done.touched)["changed"]
    if done.pictures:
        caption = (
            f"{done.id}: marked damage regions "
            + ", ".join(done.regions)
            + " shaded on the section under its current registration (left) "
            "and on the atlas (right)"
        )
        for picture in done.pictures:
            pictures_drawn.annotate(picture, caption=caption)
        result[TOOL_MEDIA_PARTS_KEY] = list(done.pictures)
    return result

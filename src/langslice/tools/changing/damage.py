"""Damage marks: the atlas regions a section is missing, which every fit leaves out.

A section is damaged exactly when it has marked regions
(``Section.damaged_regions``). :func:`mark_damage` sets a section's regions
and note; the fits and the image model's trace read them
(:func:`langslice.core.sections.damage.exclusions`). A host's note
(``inputs.damaged``, ``{filename: note}``) is the section's ``damage_note``
alone: it stays first in the note, and the agent marks the regions. A host
may also supply the regions themselves and freeze them (the ``damage``
property): then a change of them is refused. The
tool's function (``mark_damage_tool``) adds its row and the picture of the
marked regions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from PIL import Image

from langslice.core.atlas.sides import has_sides
from langslice.core.failures import IMAGE_FAILURES
from langslice.core.pictures import drawn as pictures_drawn
from langslice.core.pictures.media_keys import TOOL_MEDIA_PARTS_KEY
from langslice.core.pictures.options import default_options
from langslice.core.pictures.sizes import picture_edge
from langslice.core.refusal import RefusalCode, Refused
from langslice.core.sections.damage import (
    checked_entries,
    damage_picture,
    side_pair_refusal,
)
from langslice.tools.refusal import check_frozen, named_section
from langslice.tools.run import as_list

if TYPE_CHECKING:
    from langslice.core.pictures.options import DisplayOptions
    from langslice.core.stack.section import Section
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job
    from langslice.tools.run import Run


@dataclass(frozen=True)
class DamageRegions:
    """What :func:`mark_damage` did to one section."""

    id: str
    regions: list[str] = field(default_factory=list)
    note: str = ""
    damaged: bool = False
    by_user: bool = False
    written: bool = False
    pictures: list[Image.Image] = field(default_factory=list)
    render_failed: dict[str, str] | None = None

    @property
    def touched(self) -> list[str]:
        return [self.id] if self.written else []


def _user_note(job: Job, record: Section) -> str:
    """The note the host gave with its mark ("" for none, or no mark)."""
    marks = job.spec.inputs.get("damaged") or {}
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

    *regions* are atlas region entries (acronyms or ids, ``"CTX:left"`` for one
    side), checked against the atlas and stored normalized
    (:func:`langslice.core.sections.damage.checked_entries`). They replace
    the section's regions; empty clears them and the agent's note. A note the
    host gave stays first, the agent's after it, and clearing leaves the host's
    note. The same mark again writes nothing (``written`` False).

    With *options* (its size and border style), the result carries the picture
    of the marked regions on the section and on the atlas
    (:func:`langslice.core.sections.damage.damage_picture`); a picture that
    fails is ``render_failed``, the write standing.

    Refused (nothing written): ``NOT_FOUND`` (a section, a region); ``BUSY``;
    ``FROZEN`` (other regions on a section whose ``damage`` the job froze);
    ``BAD_ARGUMENT`` (regions not a list, or a side that does not exist);
    ``PRECONDITION`` (a side on a sagittal stack, or a side the
    section's placement does not have: ``left``/``right`` where its midline runs
    horizontally, ``top``/``bottom`` where it runs vertically), and the other
    side refusals of :func:`langslice.core.atlas.sides.native_left`.
    """
    if isinstance(regions, (str, bytes)) or not isinstance(regions, Sequence):
        raise Refused(
            RefusalCode.BAD_ARGUMENT,
            message="regions must be a list of acronyms or ids",
        )
    atlas = cast("object", workspace.atlas)
    names = checked_entries(atlas, regions, job.state.plane)

    with job.writing():
        state = job.state
        record = named_section(state, section)
        check_frozen(job, [record.id])
        if has_sides(names):
            sides = side_pair_refusal(state, workspace, record, names)
            if sides is not None:
                raise sides
        if names != list(record.damaged_regions):
            check_frozen(job, [record.id], "damage")
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
        except Refused as refusal:
            failed = {"error": refusal.code, "message": str(refusal)}
        except IMAGE_FAILURES as exc:
            failed = {
                "error": RefusalCode.IO_ERROR,
                "message": f"{exc}; check the section file opens, then try again.",
            }
    return DamageRegions(**result, pictures=drawn, render_failed=failed)


def mark_damage_tool(run: Run, section: str, regions: list[str], note: str) -> dict[str, Any]:
    """Run the ``mark_damage`` tool: write, answer, picture.

    Parameters
    ----------
    run
        The run the tool is called in.
    section, regions, note
        The declared arguments as the model sent them
        (``tools/descriptions/changing.py``).

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

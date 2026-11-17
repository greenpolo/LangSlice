"""The ``review_comments`` tool: what a person marked in the LangSlice window.

A person flipping through the sections in the window can draw a box on a
part that looks wrong and type a short comment (:mod:`langslice.job.review`,
``review.json``). :func:`review_pictures` reads them and draws, for each
section with an open comment, the section's overlay as the person saw it
with each box outlined and numbered; :func:`review_comments_tool` is the
tool. Writes no state and takes no undo step.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, cast

from langslice.core.jsonable import Json, JsonObject
from langslice.core.pictures import look as looks
from langslice.core.pictures.drawn import collecting
from langslice.core.pictures.review_boxes import Box, draw_boxes
from langslice.job.review import Comment, read_comments, revised
from langslice.tools.look import (
    Looked,
    not_shown,
    save_pictures,
    split_shown,
)
from langslice.tools.run import pictured

if TYPE_CHECKING:
    from PIL import Image

    from langslice.core.pictures.layers import PictureNote
    from langslice.core.stack.stack import Stack
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job
    from langslice.tools.run import Run

REVIEW_TOOL = "review_comments"
OPEN = "open"
REVISED = "revised"


@dataclass(frozen=True)
class ReviewRow:
    """One comment as the tool answers it."""

    id: str
    section: str
    comment: str
    box: list[float]
    state: str
    picture: int | None = None
    box_px: list[int] | None = None


@dataclass(frozen=True)
class Reviewed:
    """What :func:`review_pictures` found and drew."""

    rows: list[ReviewRow] = field(default_factory=list[ReviewRow])
    looked: Looked = field(default_factory=Looked)


@dataclass(frozen=True)
class _Picture:
    """One section's overlay with its boxes outlined, not yet saved."""

    picture: looks.LookPicture
    boxes: dict[str, list[int]]


def review_pictures(
    job: Job, workspace: Workspace, *, long_edge: int | None = None
) -> Reviewed:
    """Read the job's review comments and draw the open ones.

    Parameters
    ----------
    job
        The job whose ``review.json`` is read.
    workspace
        The job's workspace, which the pictures are drawn from.
    long_edge
        The pictures' long edge (None: the run's size).

    Returns
    -------
    Reviewed
        Each comment with its state (:data:`OPEN`: the section is as it was
        when the person commented; :data:`REVISED`: it has changed since),
        and per section with open comments the overlay (``look`` mode
        ``overlay``; ``section`` for a section with no position) with its
        boxes outlined, numbered by comment, saved like a ``look`` picture.
        At most ``Job.picture_cap`` pictures; the rest ``not_shown``.
    """
    state = job.state
    comments = read_comments(job.layout)
    open_by_section: dict[str, list[Comment]] = {}
    for comment in comments:
        if not revised(comment, state):
            open_by_section.setdefault(comment.section, []).append(comment)
    rows = [_row(c, c.section in open_by_section) for c in comments]
    if not open_by_section:
        return Reviewed(rows=rows)
    with collecting() as notes:
        pictures = [
            _outlined(job, workspace, state, section, marked, long_edge)
            for section, marked in open_by_section.items()
        ]
    return _saved(job, workspace, rows, pictures, notes)


def review_comments_tool(run: Run) -> dict[str, Json | list[Image.Image]]:
    """Run the ``review_comments`` tool.

    Parameters
    ----------
    run
        The run the tool is called in.

    Returns
    -------
    dict
        ``status``, ``open`` (how many comments are open), ``comments``
        (each :class:`ReviewRow`'s fields) and the pictures.
    """
    reviewed = review_pictures(run.job, run.workspace)
    count = sum(1 for row in reviewed.rows if row.state == OPEN)
    reply: dict[str, Json | list[Image.Image]] = {
        "status": "ok",
        "open": count,
        "comments": [_listed(row) for row in reviewed.rows],
    }
    if not reviewed.rows:
        reply["message"] = "No one has left a review comment on this job."
        return reply
    return pictured(reply, reviewed.looked)


def _outlined(
    job: Job,
    workspace: Workspace,
    state: Stack,
    section: str,
    marked: list[Comment],
    long_edge: int | None,
) -> _Picture:
    record = state.by_id(section)
    assert record is not None
    mode = "overlay" if record.position_mm is not None else "section"
    request = looks.LookRequest(
        mode=mode, sections=(section,), long_edge=long_edge
    )
    picture = looks.look(workspace, state, request, store=job.deformations)[0]
    width, height = cast("list[int]", picture.recipe["shown"])
    boxes = [(f"#{c.id}", _corners(c.box)) for c in marked]
    pixels = draw_boxes(picture.image, (width, height), boxes)
    ids = [c.id for c in marked]
    return _Picture(picture, dict(zip(ids, pixels, strict=True)))


def _saved(
    job: Job,
    workspace: Workspace,
    rows: list[ReviewRow],
    pictures: list[_Picture],
    notes: list[tuple[Image.Image, PictureNote]],
) -> Reviewed:
    kept, rest = cast(
        "tuple[list[_Picture], list[_Picture]]", split_shown(job, pictures)
    )
    looks_kept = [p.picture for p in kept]
    entries = save_pictures(job, workspace, REVIEW_TOOL, looks_kept, notes, {})
    numbers: dict[str, tuple[int, list[int]]] = {}
    for item, entry in zip(kept, entries, strict=True):
        for comment_id, pixels in item.boxes.items():
            numbers[comment_id] = (cast("int", entry["id"]), pixels)
    numbered_rows = [_numbered(row, numbers.get(row.id)) for row in rows]
    looked = Looked(
        pictures=[p.image for p in looks_kept],
        entries=entries,
        not_shown=not_shown([p.picture for p in rest], "overlay"),
    )
    return Reviewed(rows=numbered_rows, looked=looked)


def _numbered(
    row: ReviewRow, number: tuple[int, list[int]] | None
) -> ReviewRow:
    if number is None:
        return row
    return replace(row, picture=number[0], box_px=number[1])


def _row(comment: Comment, is_open: bool) -> ReviewRow:
    state = OPEN if is_open else REVISED
    return ReviewRow(
        id=comment.id,
        section=comment.section,
        comment=comment.text,
        box=comment.box,
        state=state,
    )


def _listed(row: ReviewRow) -> JsonObject:
    fields = cast("dict[str, Json]", asdict(row))
    return {k: v for k, v in fields.items() if v is not None}


def _corners(box: list[float]) -> Box:
    return cast("Box", tuple(box))

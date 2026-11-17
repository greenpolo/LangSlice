"""Review comments: boxes a person drew on sections in the LangSlice window.

A person flips through a job's sections in the window, draws a box on a
part that is misaligned and types a short comment. The comments live in
``review.json`` in the job folder, beside the state but not in it: they are
the person's notes on the registration, so no undo step takes them back and
no write of the agent's changes them. The window is their only writer; the
``review_comments`` tool reads them for the agent.

Each comment holds the section (its filename), the box as fractions of the
section's overlay picture (``[x0, y0, x1, y1]``, the picture ``look`` mode
``overlay`` draws, without its caption band; mode ``section`` for a
section with no position), the text, when it was written and the section's
key then (:func:`section_key`). A comment is *open* while the section's key
is the one it was written on, and *revised* once the section has changed
since: the agent has worked on it.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from langslice.core.pictures.look import snapshot
from langslice.job.state_file import write_json_atomic

if TYPE_CHECKING:
    from langslice.core.stack.stack import Stack
    from langslice.job.layout import JobLayout

__all__ = [
    "MAX_TEXT",
    "Comment",
    "ReviewError",
    "add_comment",
    "read_comments",
    "remove_comment",
    "revised",
    "section_key",
]

MAX_TEXT = 500
KEY_DIGITS = 16
MIN_SIDE = 0.005
BAD_BOX = "A box is four numbers: [x0, y0, x1, y1]."

_WRITING = threading.Lock()


class ReviewError(ValueError):
    """A comment that cannot be kept, with the reason in words."""


@dataclass(frozen=True)
class Comment:
    """One review comment, as ``review.json`` holds it."""

    id: str
    section: str
    box: list[float]
    text: str
    at: str
    key: str


def section_key(state: Stack, section_id: str) -> str:
    """Digest everything a picture of one section depends on.

    Parameters
    ----------
    state
        The job's stack.
    section_id
        The section's filename.

    Returns
    -------
    str
        :data:`KEY_DIGITS` hex digits of the section's look snapshot: it
        changes exactly when what an overlay of the section depends on
        changes.
    """
    text = json.dumps(snapshot(state, [section_id]), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:KEY_DIGITS]


def revised(comment: Comment, state: Stack) -> bool:
    """Say whether a comment's section has changed since it was written.

    Parameters
    ----------
    comment
        The comment, with the section key it was written on.
    state
        The job's stack as it stands.

    Returns
    -------
    bool
        True when the section's key is no longer the comment's, or the
        section is gone.
    """
    if state.by_id(comment.section) is None:
        return True
    return section_key(state, comment.section) != comment.key


def read_comments(layout: JobLayout) -> list[Comment]:
    """Read the job's review comments, oldest first.

    Parameters
    ----------
    layout
        The job folder.

    Returns
    -------
    list of Comment
        Every comment ``review.json`` holds; none when it is missing or
        does not read, and an entry that does not read is left out.
    """
    try:
        text = layout.review_file.read_text(encoding="utf-8")
        data = cast("object", json.loads(text))
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    entries = cast("dict[str, object]", data).get("comments")
    if not isinstance(entries, list):
        return []
    comments: list[Comment] = []
    for entry in cast("list[object]", entries):
        comment = _comment(entry)
        if comment is not None:
            comments.append(comment)
    return comments


def add_comment(
    layout: JobLayout,
    state: Stack,
    section: str,
    box: object,
    text: object,
) -> Comment:
    """Keep a new comment on a section.

    Parameters
    ----------
    layout
        The job folder.
    state
        The job's stack as it stands (the section's key is read from it).
    section
        The section's filename.
    box
        ``[x0, y0, x1, y1]``, fractions of the overlay picture; the corners
        in either order.
    text
        What the person wrote.

    Returns
    -------
    Comment
        The comment as kept.

    Raises
    ------
    ReviewError
        When the section is not the job's, the box is not four fractions
        or is too small, or the text is empty.
    """
    if state.by_id(section) is None:
        raise ReviewError(f"This job has no section {section}.")
    corners = _box(box)
    words = " ".join(str(text or "").split())
    if not words:
        raise ReviewError("A comment needs a few words.")
    with _WRITING:
        comments = read_comments(layout)
        numbers = [int(c.id) for c in comments if c.id.isdigit()]
        comment = Comment(
            id=str(1 + max(numbers, default=0)),
            section=section,
            box=corners,
            text=words[:MAX_TEXT],
            at=datetime.now(UTC).isoformat(timespec="seconds"),
            key=section_key(state, section),
        )
        _write(layout, [*comments, comment])
    return comment


def remove_comment(layout: JobLayout, comment_id: str) -> bool:
    """Remove one comment.

    Parameters
    ----------
    layout
        The job folder.
    comment_id
        The comment's id.

    Returns
    -------
    bool
        Whether a comment had that id.
    """
    with _WRITING:
        comments = read_comments(layout)
        kept = [c for c in comments if c.id != str(comment_id)]
        if len(kept) == len(comments):
            return False
        _write(layout, kept)
    return True


def _write(layout: JobLayout, comments: list[Comment]) -> None:
    data = {"comments": [asdict(c) for c in comments]}
    write_json_atomic(str(layout.review_file), data)


def _fraction(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReviewError(BAD_BOX)
    return min(max(float(value), 0.0), 1.0)


def _box(value: object) -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise ReviewError(BAD_BOX)
    numbers = [_fraction(v) for v in cast("list[object]", value)]
    if len(numbers) != 4:
        raise ReviewError(BAD_BOX)
    x0, x1 = sorted((numbers[0], numbers[2]))
    y0, y1 = sorted((numbers[1], numbers[3]))
    if x1 - x0 < MIN_SIDE or y1 - y0 < MIN_SIDE:
        raise ReviewError("The box is too small; drag out a larger one.")
    return [round(v, 4) for v in (x0, y0, x1, y1)]


def _comment(entry: object) -> Comment | None:
    if not isinstance(entry, dict):
        return None
    data = cast("dict[str, object]", entry)
    try:
        corners = _box(data.get("box"))
    except ReviewError:
        return None
    names = ("id", "section", "text", "at", "key")
    fields = [data.get(name) for name in names]
    if not all(isinstance(value, str) for value in fields):
        return None
    comment_id, section, text, at, key = cast("list[str]", fields)
    return Comment(
        id=comment_id, section=section, box=corners, text=text, at=at, key=key
    )

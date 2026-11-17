"""The review comments as the viewer shows them.

:class:`CommentWatch` reads a job's ``review.json``
(:mod:`langslice.job.review`) again when the file or the job's state
changes, and gives each comment with whether its section has changed since
it was written (``revised``). The snapshot carries them; the page draws
each as a box and its words over the section's picture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from langslice.job.folder.layout import REVIEW_FILE, JobLayout
from langslice.job.review import read_comments, revised

if TYPE_CHECKING:
    from langslice.core.stack.stack import Stack


@dataclass(frozen=True)
class CommentRow:
    """One comment as the page gets it."""

    id: str
    section: str
    box: list[float]
    text: str
    at: str
    revised: bool


@dataclass
class CommentWatch:
    """A job folder's review comments, read again only when they may differ."""

    folder: Path
    _stamp: tuple[tuple[int, int] | None, int] | None = None
    _rows: list[CommentRow] = field(default_factory=list[CommentRow])

    def rows(self, state: Stack | None, version: int) -> list[CommentRow]:
        """Return the comments, each with whether it is revised.

        Parameters
        ----------
        state
            The job's stack as the viewer last read it; None while it
            reads none (no comment is revised then).
        version
            The viewer's count of state changes: a new one reads the
            comments' states again.

        Returns
        -------
        list of CommentRow
            Every comment, oldest first.
        """
        stamp = (_file_stamp(self.folder / REVIEW_FILE), version)
        if stamp != self._stamp:
            comments = read_comments(JobLayout(self.folder))
            self._rows = [
                CommentRow(
                    id=c.id,
                    section=c.section,
                    box=c.box,
                    text=c.text,
                    at=c.at,
                    revised=state is not None and revised(c, state),
                )
                for c in comments
            ]
            self._stamp = stamp
        return self._rows


def _file_stamp(path: Path) -> tuple[int, int] | None:
    try:
        info = path.stat()
    except OSError:
        return None
    return (info.st_mtime_ns, info.st_size)

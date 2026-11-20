"""The undo and redo steps on disk: ``history/undo.json``.

The file holds ``format_version`` (:data:`UNDO_FORMAT_VERSION`),
``next_step`` (the number the job's next write gets) and the ``undo`` and
``redo`` steps, oldest first, and is rewritten after each write. A step
(:class:`UndoStep`) names, for each section the write changed, the ``seq``
of the line of the section's history it held before, the stack's own
fields the write changed, as they were, and for the run's notes only the
change (:class:`NotesBefore`: how many to keep and what follows them), so
the file grows with what each write changed, not with the notes held; undo
puts those back. There is no cap on the number of steps.

A file that does not read, or a newer LangSlice's, is left as it is and
undo covers the session.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ConfigDict, ValidationError

from langslice.core.jsonable import Json
from langslice.core.shared_files import write_json_atomic
from langslice.core.stack.part import Shape, Whole
from langslice.job.folder.layout import JobLayout
from langslice.job.state_file import FORMAT_KEY, json_object

logger = logging.getLogger(__name__)

UNDO_FORMAT_VERSION = 1


class NotesBefore(Shape):
    """Hold what puts the run's notes back: the first *keep*, then *then*.

    Parameters
    ----------
    keep
        How many of the notes, from the first, the step left as they were.
    then
        The notes that followed those before the step.
    """

    model_config = ConfigDict(frozen=True)

    keep: Whole
    then: tuple[str, ...] = ()

    @classmethod
    def between(cls, before: Sequence[str], after: Sequence[str]) -> NotesBefore | None:
        """Return what puts *after* back to *before*; None when they match.

        Parameters
        ----------
        before, after
            The notes before and after a step.

        Returns
        -------
        NotesBefore or None
            The notes they share from the first, and what followed in
            *before*.
        """
        if list(before) == list(after):
            return None
        keep = 0
        for old, new in zip(before, after, strict=False):
            if old != new:
                break
            keep += 1
        return cls(keep=keep, then=tuple(before[keep:]))

    def put_back(self, notes: Sequence[str]) -> list[str]:
        """Return *notes* as they were before the step.

        Parameters
        ----------
        notes
            The notes as they stand.

        Returns
        -------
        list of str
            The first :attr:`keep` of them, then :attr:`then`.
        """
        return [*notes[: self.keep], *self.then]

    def before(self, later: NotesBefore) -> NotesBefore:
        """Return what puts back *later* (a newer step's) and then this.

        Parameters
        ----------
        later
            The change of the step after this one.

        Returns
        -------
        NotesBefore
            The two as one.
        """
        if self.keep <= later.keep:
            return self
        shared = later.then[: self.keep - later.keep]
        return NotesBefore(keep=later.keep, then=(*shared, *self.then))


class UndoStep(Shape):
    """Hold one undo (or redo) step: what puts the state back before it.

    Parameters
    ----------
    step, by, at
        The write's number, the tool that made it and its local time.
    sections
        For each section it changed, the ``seq`` of the line before it.
    stack_before
        The stack's own fields it changed, as they were, but its notes.
    notes_before
        What puts the run's notes back; None when it left them as they were.
    """

    model_config = ConfigDict(frozen=True)

    step: Whole
    by: str
    at: str | None = None
    sections: dict[str, Whole]
    stack_before: dict[str, Any]
    notes_before: NotesBefore | None = None


def combined_steps(steps: Sequence[UndoStep]) -> UndoStep:
    """Return one step that undoes *steps* together.

    Parameters
    ----------
    steps
        Consecutive steps, oldest first; at least one.

    Returns
    -------
    UndoStep
        The first step's number, tool and time; for each section and
        stack field, what the earliest step that changed it held before;
        the notes' changes put back newest first.
    """
    sections: dict[str, int] = {}
    stack: dict[str, Json] = {}
    for step in steps:
        for section_id, seq in step.sections.items():
            sections.setdefault(section_id, seq)
        for key, value in step.stack_before.items():
            stack.setdefault(key, value)
    notes: NotesBefore | None = None
    for step in reversed(steps):
        change = step.notes_before
        if change is not None:
            notes = change if notes is None else change.before(notes)
    first = steps[0]
    return UndoStep(
        step=first.step,
        by=first.by,
        at=first.at,
        sections=sections,
        stack_before=stack,
        notes_before=notes,
    )


@dataclass
class UndoLists:
    """Hold the undo and redo steps, oldest first, and the next write."""

    undo: list[UndoStep]
    redo: list[UndoStep]
    next_step: int = 1
    notice: str | None = None


class UndoDocument(Shape):
    """Hold ``history/undo.json`` as it is written."""

    format_version: Whole = UNDO_FORMAT_VERSION
    next_step: Whole
    undo: list[UndoStep]
    redo: list[UndoStep]


class UndoFile:
    """Read and write ``history/undo.json`` (see the module text).

    Parameters
    ----------
    layout
        The job folder.
    """

    def __init__(self, layout: JobLayout) -> None:
        self.layout = layout
        self.path = layout.undo_file
        self.writable = True

    def load(self) -> UndoLists:
        """Return the steps on disk; none without a readable file (logged).

        Returns
        -------
        UndoLists
            The steps, oldest first, the next write's number and the
            notice of a file that cannot be used.
        """
        self.writable = True
        try:
            data = json_object(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return UndoLists([], [])
        except (OSError, ValueError) as exc:
            return self._unreadable(f"it does not read ({exc})")
        version = None if data is None else data.get(FORMAT_KEY)
        if data is None or version != UNDO_FORMAT_VERSION:
            return self._unreadable(f"its format is {version!r}, not {UNDO_FORMAT_VERSION}")
        try:
            file = UndoDocument.from_json(data)
        except ValidationError as exc:
            return self._unreadable(f"a step does not read ({exc})")
        return UndoLists(file.undo, file.redo, file.next_step)

    def save(self, lists: UndoLists) -> None:
        """Write the steps; nothing while the file on disk cannot be read.

        Parameters
        ----------
        lists
            The steps and the next write's number.
        """
        if not self.writable:
            return
        document = UndoDocument(next_step=lists.next_step, undo=lists.undo, redo=lists.redo)
        write_json_atomic(str(self.path), document.to_json(), indent=None)

    def _unreadable(self, why: str) -> UndoLists:
        self.writable = False
        notice = _told(
            f"The undo history {self.path} cannot be used: {why}. It is "
            "left untouched; undo covers this session's steps only."
        )
        return UndoLists([], [], notice=notice)


def _told(notice: str) -> str:
    """Log *notice* as a warning and return it."""
    logger.warning("%s", notice)
    return notice

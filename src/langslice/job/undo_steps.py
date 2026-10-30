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
undo covers the session; an older LangSlice's step files (``index.json``,
``step-*.json``) are ignored, with one warning per process, and left on
disk.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from langslice.core.jsonable import Json, JsonObject, JsonValue, to_jsonable
from langslice.core.stack.json_reads import (
    as_object,
    integer_at,
    list_at,
    text_at,
)
from langslice.job.layout import JobLayout
from langslice.job.state_file import (
    FORMAT_KEY,
    json_object,
    write_json_atomic,
)

logger = logging.getLogger(__name__)

#: The undo file's format: ``undo`` and ``redo`` steps, ``next_step``.
UNDO_FORMAT_VERSION = 1
#: The step files of an older LangSlice's ``history/``.
_OLD_INDEX, _OLD_STEP_PREFIX = "index.json", "step-"
#: The undo files whose folder's older step files this process told of.
_TOLD_OLDER: set[Path] = set()


def older_undo_steps(layout: JobLayout) -> bool:
    """Return whether an older LangSlice's undo steps are in the job folder.

    Parameters
    ----------
    layout
        The job folder.

    Returns
    -------
    bool
        True while ``history/`` holds ``index.json`` or a ``step-*.json``.
    """
    folder = layout.undo_file.parent
    return folder.is_dir() and any(
        path.name == _OLD_INDEX or path.name.startswith(_OLD_STEP_PREFIX)
        for path in folder.iterdir()
    )


@dataclass(frozen=True)
class NotesBefore:
    """Hold what puts the run's notes back: the first *keep*, then *then*.

    Parameters
    ----------
    keep
        How many of the notes, from the first, the step left as they were.
    then
        The notes that followed those before the step.
    """

    keep: int
    then: tuple[str, ...] = ()

    @classmethod
    def between(
        cls, before: Sequence[str], after: Sequence[str]
    ) -> NotesBefore | None:
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
        return cls(keep, tuple(before[keep:]))

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
        # this step kept notes the later one replaced: they are its then
        shared = later.then[: self.keep - later.keep]
        return NotesBefore(later.keep, (*shared, *self.then))


@dataclass(frozen=True)
class UndoStep:
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

    step: int
    by: str
    at: str | None
    sections: Mapping[str, int]
    stack_before: JsonObject
    notes_before: NotesBefore | None = None

    @classmethod
    def from_json(cls, data: Json) -> UndoStep:
        """Return the step a JSON object holds.

        Parameters
        ----------
        data
            One step of ``undo.json``.

        Returns
        -------
        UndoStep
            The step.

        Raises
        ------
        ValueError
            When it is not a step.
        """
        fields = as_object("step", data)
        sections = as_object("sections", fields.get("sections"))
        at = fields.get("at")
        if not isinstance(at, str):
            at = None
        notes: NotesBefore | None = None
        if fields.get("notes_before") is not None:
            change = as_object("notes_before", fields.get("notes_before"))
            then = list_at(change, "then")
            notes = NotesBefore(
                integer_at(change, "keep"), tuple(str(note) for note in then)
            )
        return cls(
            step=integer_at(fields, "step"),
            by=text_at(fields, "by"),
            at=at,
            sections={key: integer_at(sections, key) for key in sections},
            stack_before=as_object("stack_before", fields.get("stack_before")),
            notes_before=notes,
        )

    def to_json(self) -> dict[str, JsonValue]:
        """Return the step as ``undo.json`` holds it.

        Returns
        -------
        dict
            ``step``, ``by``, ``at``, ``sections``, ``stack_before`` and
            ``notes_before`` (``keep`` and ``then``, or null).
        """
        notes: dict[str, JsonValue] | None = None
        if self.notes_before is not None:
            notes = {
                "keep": self.notes_before.keep,
                "then": list[JsonValue](self.notes_before.then),
            }
        line: dict[str, JsonValue] = {
            "step": self.step,
            "by": self.by,
            "at": self.at,
            "sections": dict[str, JsonValue](self.sections),
            "stack_before": to_jsonable(self.stack_before),
            "notes_before": notes,
        }
        return line


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
    #: What a reader is told of the steps on disk: why they cannot be
    #: used, or that an older LangSlice's are ignored; None otherwise.
    notice: str | None = None


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
        #: False while the file on disk cannot be read: :meth:`save` leaves
        #: it. Kept here, not returned, because the save that reads it comes
        #: many writes after the load that set it.
        self.writable = True

    def load(self) -> UndoLists:
        """Return the steps on disk; none without a readable file (logged).

        Returns
        -------
        UndoLists
            The steps, oldest first, the next write's number and the
            notice of a file that cannot be used or of an older
            LangSlice's steps (once per process).
        """
        self.writable = True
        try:
            data = json_object(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            if self.path in _TOLD_OLDER or not older_undo_steps(self.layout):
                return UndoLists([], [])
            _TOLD_OLDER.add(self.path)
            notice = _told(
                "The undo history of an older LangSlice in "
                f"{self.path.parent} is ignored; undo covers this session."
            )
            return UndoLists([], [], notice=notice)
        except (OSError, ValueError) as exc:
            return self._unreadable(f"it does not read ({exc})")
        version = None if data is None else data.get(FORMAT_KEY)
        if data is None or version != UNDO_FORMAT_VERSION:
            return self._unreadable(
                f"its format is {version!r}, not {UNDO_FORMAT_VERSION}"
            )
        try:
            undo = [UndoStep.from_json(item) for item in list_at(data, "undo")]
            redo = [UndoStep.from_json(item) for item in list_at(data, "redo")]
            next_step = integer_at(data, "next_step")
        except ValueError as exc:
            return self._unreadable(f"a step does not read ({exc})")
        return UndoLists(undo, redo, next_step)

    def save(self, lists: UndoLists) -> None:
        """Write the steps; nothing while the file on disk cannot be read.

        Parameters
        ----------
        lists
            The steps and the next write's number.
        """
        if not self.writable:
            return
        write_json_atomic(
            str(self.path),
            {
                FORMAT_KEY: UNDO_FORMAT_VERSION,
                "next_step": lists.next_step,
                "undo": [step.to_json() for step in lists.undo],
                "redo": [step.to_json() for step in lists.redo],
            },
            indent=None,
        )

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

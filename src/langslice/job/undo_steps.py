"""The undo and redo steps on disk: ``history/undo.json``.

The file holds ``format_version`` (:data:`UNDO_FORMAT_VERSION`),
``next_step`` (the number the job's next write gets) and the ``undo`` and
``redo`` steps, oldest first, and is rewritten after each write. A step
(:class:`UndoStep`) names, for each section the write changed, the ``seq``
of the line of the section's history it held before, and the stack's own
fields the write changed, as they were; undo puts those back. There is no
cap on the number of steps.

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
        The stack's own fields it changed, as they were.
    """

    step: int
    by: str
    at: str | None
    sections: Mapping[str, int]
    stack_before: JsonObject

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
        return cls(
            step=integer_at(fields, "step"),
            by=text_at(fields, "by"),
            at=at,
            sections={key: integer_at(sections, key) for key in sections},
            stack_before=as_object("stack_before", fields.get("stack_before")),
        )

    def to_json(self) -> dict[str, JsonValue]:
        """Return the step as ``undo.json`` holds it.

        Returns
        -------
        dict
            ``step``, ``by``, ``at``, ``sections`` and ``stack_before``.
        """
        line: dict[str, JsonValue] = {
            "step": self.step,
            "by": self.by,
            "at": self.at,
            "sections": dict[str, JsonValue](self.sections),
            "stack_before": to_jsonable(self.stack_before),
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
        stack field, what the earliest step that changed it held before.
    """
    sections: dict[str, int] = {}
    stack: dict[str, Json] = {}
    for step in steps:
        for section_id, seq in step.sections.items():
            sections.setdefault(section_id, seq)
        for key, value in step.stack_before.items():
            stack.setdefault(key, value)
    first = steps[0]
    return UndoStep(
        step=first.step,
        by=first.by,
        at=first.at,
        sections=sections,
        stack_before=stack,
    )


@dataclass
class UndoLists:
    """Hold the undo and redo steps, oldest first, and the next write."""

    undo: list[UndoStep]
    redo: list[UndoStep]
    next_step: int = 1


class UndoFile:
    """Read and write ``history/undo.json`` (see the module text).

    Parameters
    ----------
    layout
        The job folder.
    """

    def __init__(self, layout: JobLayout) -> None:
        self.path = layout.undo_file
        #: What a reader is told about the steps on disk: why they cannot
        #: be used, or that an older LangSlice's are ignored; None otherwise.
        self.notice: str | None = None
        #: False while the file on disk cannot be read: :meth:`save` leaves it.
        self.writable = True

    def load(self) -> UndoLists:
        """Return the steps on disk; none without a readable file (logged).

        Returns
        -------
        UndoLists
            The steps, oldest first, and the next write's number.
        """
        self.notice, self.writable = None, True
        try:
            data = json_object(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            if self.path not in _TOLD_OLDER and self._older_history():
                _TOLD_OLDER.add(self.path)
                self._tell(
                    "The undo history of an older LangSlice in "
                    f"{self.path.parent} is ignored; undo covers this session."
                )
            return UndoLists([], [])
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

    def _older_history(self) -> bool:
        folder = self.path.parent
        return folder.is_dir() and any(
            path.name == _OLD_INDEX or path.name.startswith(_OLD_STEP_PREFIX)
            for path in folder.iterdir()
        )

    def _unreadable(self, why: str) -> UndoLists:
        self.writable = False
        self._tell(
            f"The undo history {self.path} cannot be used: {why}. It is left "
            "untouched; undo covers this session's steps only."
        )
        return UndoLists([], [])

    def _tell(self, notice: str) -> None:
        self.notice = notice
        logger.warning("%s", notice)

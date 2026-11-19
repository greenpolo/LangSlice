"""The undo steps: what each write left, and how to take it back.

**Undo.** One pattern: ``before = job.snapshot()`` before a write, then
``job.commit(before)`` once it succeeded: a line in the history of each
section the write changed, one undo step naming the lines they held before
(:mod:`langslice.job.steps.history`), then the checkpoint. Undo and redo put
those lines back (:meth:`UndoStore.undo`, :meth:`UndoStore.redo`), each
appending lines of its own; a resumed job reads its steps back, a fresh one
starts without any. There is no cap on the number of steps.

:class:`UndoStore` is the job's ``undo``. It holds the steps, writes the
history lines they name and keeps the journal of the registrations the writes
numbered; the live sync with the files, the lock and the checkpoint are the
job's (:mod:`langslice.job.job`).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager

from langslice.core.jsonable import Json, JsonObject
from langslice.core.stack.section import Section
from langslice.core.stack.stack import Stack
from langslice.job.checkpoint import Stamp, file_stamp
from langslice.job.folder.lock import FolderLock
from langslice.job.state_file import (
    set_stack_fields,
    stack_fields,
    stack_from_json,
    stack_to_json,
    upgraded_row,
)
from langslice.job.steps.history import (
    HOST,
    Row,
    SectionHistories,
    now,
    registration_key,
    section_row,
    set_head,
)
from langslice.job.steps.undo_steps import (
    NotesBefore,
    UndoFile,
    UndoLists,
    UndoStep,
    combined_steps,
)

logger = logging.getLogger(__name__)

NOTES = "notes"


class UndoStore:
    """The undo and redo steps of one job.

    Parameters
    ----------
    state
        The stack the job registers (changed in place, never replaced).
    histories
        The section histories the steps name lines of.
    lock
        The job folder's write lock.
    file
        ``history/undo.json``.
    persist
        False: nothing is written (a dry run).
    actor
        Returns the tool this thread runs, ``script`` for none.
    writing
        Holds the job's write lock around a step taken back or again
        (``Job.writing``).
    checkpoint
        Writes the state (``Checkpoints.write``).
    lists
        The steps already read from *file*; None starts without any.
    seen
        The stamp of *file* taken before it was read; None takes it now.
    """

    def __init__(
        self,
        state: Stack,
        histories: SectionHistories,
        lock: FolderLock,
        file: UndoFile,
        *,
        persist: bool,
        actor: Callable[[], str],
        writing: Callable[[], AbstractContextManager[object]],
        checkpoint: Callable[[], None],
        lists: UndoLists | None = None,
        seen: Stamp = None,
    ) -> None:
        self.steps: list[UndoStep] = [] if lists is None else lists.undo
        self.redo_steps: list[UndoStep] = [] if lists is None else lists.redo
        self.next_step = 1 if lists is None else lists.next_step
        self.pushed = 0
        self.registered: list[tuple[int, str, int]] = []
        self.stamp: Stamp = seen if seen is not None else file_stamp(str(file.path))
        self._state = state
        self._histories = histories
        self._lock = lock
        self._file = file
        self._persist = persist
        self._actor = actor
        self._writing = writing
        self._checkpoint = checkpoint

    @property
    def path(self) -> str:
        """The undo and redo steps, ``history/undo.json``."""
        return str(self._file.path)

    def registered_since(self, mark: int) -> dict[str, int]:
        """The registration each section holds after this thread's writes
        since ``len(undo.registered)`` was *mark*, by section."""
        me = threading.get_ident()
        return {
            section: number for thread, section, number in self.registered[mark:] if thread == me
        }

    def push(self, step: UndoStep) -> None:
        """Make *step* the latest step; the redo side is cleared."""
        self.steps.append(step)
        self.redo_steps.clear()
        self.pushed += 1

    def amend(self, before: JsonObject, by: str) -> bool:
        """Record a change that belongs to the last write.

        The history lines are appended and the sections joined to the last
        step. Call it inside the job's write lock.

        Parameters
        ----------
        before
            The state before the change, as ``Job.snapshot`` gives it.
        by
            The tool that made it.

        Returns
        -------
        bool
            False when there is no step to join: the lines are appended and
            nothing else changes.
        """
        if not self.steps:
            self.record(before, by)
            return False
        top = self.steps[-1]
        step = self.record(before, by, step=top.step)
        self.steps[-1] = combined_steps([top, step])
        return True

    def merge(self, mark: int) -> None:
        """Make the undo steps recorded since :attr:`pushed` was *mark* one
        step: the state before the first of them. A tool whose call writes
        through several operations is still one undo step. Call it inside
        the job's write lock."""
        count = min(self.pushed - mark, len(self.steps))
        if count < 2:
            return
        merged = combined_steps(self.steps[-count:])
        del self.steps[-count:]
        self.steps.append(merged)
        self.pushed -= count - 1
        self.save()

    def undo(self) -> bool:
        """Take the last step back; False when there is none."""
        with self._writing():
            if not self.steps:
                return False
            self.redo_steps.append(self._take(self.steps.pop()))
            self._end_step()
            return True

    def redo(self) -> bool:
        """Take the last undone step again; False when there is none."""
        with self._writing():
            if not self.redo_steps:
                return False
            self.steps.append(self._take(self.redo_steps.pop()))
            self._end_step()
            return True

    def _end_step(self) -> None:
        """Save the undo steps, then checkpoint the state."""
        self.save()
        self._checkpoint()

    def _take(self, step: UndoStep) -> UndoStep:
        """Put back the lines and stack fields *step* names; return the
        step that puts the state back as it was before.
        """
        before = stack_to_json(self._state)
        for section_id, seq in step.sections.items():
            section = self._state.by_id(section_id)
            row = self._histories.row(section_id, seq)
            if section is None or row is None:
                logger.warning(
                    "Undo: line %s of %s is not in its history; the section stays as it is",
                    seq,
                    section_id,
                )
                continue
            set_head(section, self._current_row(row))
        set_stack_fields(self._state, step.stack_before)
        if step.notes_before is not None:
            self._state.notes = step.notes_before.put_back(self._state.notes)
        back = self.record(before, self._actor(), also=step)
        return back.model_copy(update={"step": step.step, "by": step.by, "at": step.at})

    def _current_row(self, row: Row) -> JsonObject:
        """Return a line's section row at the current state format."""
        return upgraded_row(row.format_version, row.section)

    def record(
        self,
        before: JsonObject,
        by: str,
        *,
        step: int | None = None,
        also: UndoStep | None = None,
    ) -> UndoStep:
        """Append a history line for each section changed since *before*
        (and each *also* names), delete the record folders no line names,
        and return the undo step that puts *before* back.

        *step* joins the lines to that write (:meth:`amend`); *also* is the
        step being taken back: its sections and stack fields are named in the
        step returned even where they did not change.
        """
        if step is None:
            step = self.next_step
            self.next_step += 1
        then_state = stack_from_json(before)
        now_notes = self._state.notes
        named: Mapping[str, int] = {} if also is None else also.sections
        sections: dict[str, int] = {}
        for section in self._state.slices:
            seq = self._record_section(
                then_state,
                section,
                step=step,
                by=by,
                named=section.id in named,
            )
            if seq is not None:
                sections[section.id] = seq
        self._collect_records(sections)
        return UndoStep(
            step=step,
            by=by,
            at=now(),
            sections=sections,
            stack_before=self._stack_before(before, also),
            notes_before=NotesBefore.between(then_state.notes, now_notes),
        )

    def _stack_before(self, before: JsonObject, also: UndoStep | None) -> dict[str, Json]:
        """Return the stack fields but the notes that differ from *before*
        (and those *also* names), as *before* holds them.
        """
        fields = stack_fields(self._state)
        del fields[NOTES]
        changed = {key: before.get(key) for key in fields if before.get(key) != fields[key]}
        if also is not None:
            changed.update({key: before.get(key) for key in also.stack_before})
        return changed

    def _collect_records(self, section_ids: Iterable[str]) -> None:
        """Delete the record folders of these sections that no line names."""
        if not self._persist:
            return
        for section_id in section_ids:
            section = self._state.by_id(section_id)
            if section is not None:
                self._histories.collect_records(section)

    def _record_section(
        self,
        then_state: Stack,
        section: Section,
        *,
        step: int,
        by: str,
        named: bool,
    ) -> int | None:
        """Append a section's lines for one write; return the ``seq`` of the
        line it held before, or None when the write left it as it was.
        """
        old_section = then_state.by_id(section.id)
        if old_section is None:
            return None
        row_then, row_now = section_row(old_section), section_row(section)
        if row_then == row_now and not named:
            return None
        if not self._persist:
            rows = self._histories.rows(section.id)
            return rows[-1].seq if rows else 0
        replaced = registration_key(then_state, old_section) != (
            registration_key(self._state, section)
        )
        head = self._line_before(then_state, old_section, step, replaced)
        if row_now != row_then:
            self._line_after(section, step=step, by=by, replaced=replaced)
        return head.seq

    def _line_before(
        self,
        then_state: Stack,
        old_section: Section,
        step: int,
        replaced: bool,
    ) -> Row:
        """Return the line a section held before a write.

        It is the section's last line, unless no line holds its state then
        (an edit from outside, a history that starts here) or the write
        replaces a registration never numbered: then a ``host`` line of
        that state is appended first, numbering its registration when it
        is replaced.
        """
        rows = self._histories.rows(old_section.id)
        head = rows[-1] if rows else None
        key_then = registration_key(then_state, old_section)
        unnumbered = (
            replaced
            and key_then is not None
            and self._histories.number(old_section.id, key_then) is None
        )
        row_then = section_row(old_section)
        if head is not None and head.section == row_then and not unnumbered:
            return head
        return self._histories.append(
            then_state, old_section, step=step, by=HOST, numbered=replaced
        )

    def _line_after(self, section: Section, *, step: int, by: str, replaced: bool) -> None:
        """Append the line a write left a section with.

        A registration it newly numbered is noted in :attr:`registered`.
        """
        line = self._histories.append(self._state, section, step=step, by=by, numbered=replaced)
        if replaced and line.registration is not None:
            self.registered.append((threading.get_ident(), section.id, line.registration))

    def number_registration(self, section: Section) -> int | None:
        """Return the number of the registration a section holds.

        One never numbered is numbered now, by a ``host`` line. Call it
        inside the job's write lock.

        Parameters
        ----------
        section
            The section as it stands.

        Returns
        -------
        int or None
            None while it holds none (or for a job that writes nothing).
        """
        number = self._histories.registration_number(self._state, section)
        if number is not None or not self._persist:
            return number
        if registration_key(self._state, section) is None:
            return None
        last_write = max(self.next_step - 1, 0)
        line = self._histories.append(self._state, section, step=last_write, by=HOST, numbered=True)
        return line.registration

    def record_opening(self) -> None:
        """Append each section's opening line: by ``host``, write 0."""
        if not self._persist:
            return
        with self._lock.held():
            for section in self._state.slices:
                self._histories.append(self._state, section, step=0, by=HOST, numbered=False)

    def load(self) -> None:
        """Read the undo steps on disk into the store (none unreadable)."""
        lists = self._file.load()
        self.steps, self.redo_steps = lists.undo, lists.redo
        self.next_step = max(self.next_step, lists.next_step)

    def save(self) -> None:
        """Write the steps, under the lock."""
        if not self._persist:
            return
        with self._lock.held():
            self._file.save(UndoLists(self.steps, self.redo_steps, self.next_step))
        self.stamp = file_stamp(self.path)

    def reload_if_changed(self) -> bool:
        """Read the steps again when ``undo.json`` changed on disk since this
        store last read or wrote it; whether it did."""
        stamp = file_stamp(self.path)
        if stamp == self.stamp:
            return False
        self.load()
        self.stamp = stamp
        return True

    def mark_seen(self) -> None:
        """Count ``undo.json`` as it stands as read."""
        self.stamp = file_stamp(self.path)

"""Undo steps, and the live sync that keeps the job with its files.

**Undo.** One pattern: ``before = job.snapshot()`` before a write, then
``job.commit(before)`` once it succeeded: a line in the history of each
section the write changed, one undo step naming the lines they held before
(:mod:`langslice.job.history`), then the checkpoint. Undo and redo put those
lines back (:meth:`JobUndo.undo`, :meth:`JobUndo.redo`), each appending lines
of its own; a resumed job reads its steps back, a fresh one starts without
any. There is no cap on the number of steps.

**Live shared editing.** :meth:`JobUndo.sync` notices that the state file
changed on disk since this job last read or wrote it (a script edited it, or
another job on the same folder wrote it) and reloads it before the next
operation instead of overwriting it. A plain edit becomes one undo step;
when ``history/undo.json`` changed with it (another job wrote both), its
steps are read back instead.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

from langslice.core.jsonable import Json, JsonObject
from langslice.core.stack.json_reads import objects_at
from langslice.core.stack.section import Section
from langslice.core.stack.stack import Stack
from langslice.job.checkpoint import JobCheckpoint, file_stamp, state_from_file
from langslice.job.history import (
    HOST,
    SCRIPT,
    Row,
    now,
    registration_key,
    section_row,
    set_head,
)
from langslice.job.state_file import (
    STATE_FORMAT_VERSION,
    read_checkpoint,
    set_stack_fields,
    stack_fields,
    stack_from_json,
    stack_to_json,
)
from langslice.job.undo_steps import (
    NotesBefore,
    UndoLists,
    UndoStep,
    combined_steps,
)

logger = logging.getLogger(__name__)

#: The stack field a step keeps as a change (:class:`NotesBefore`).
NOTES = "notes"


class JobUndo(JobCheckpoint):
    """Write under the job's lock, one undo step at a time, and sync."""

    @contextlib.contextmanager
    def writing(self) -> Iterator[dict[str, Any] | None]:
        """Hold the job folder's write lock and bring the state up to date.

        Every write goes lock -> :meth:`sync` (reload what another writer
        saved) -> apply -> :meth:`commit` -> unlock, so a running agent, CLI
        calls and scripts on one folder never overwrite each other. Yields
        what :meth:`sync` returned (the state as held before a reload, or
        None). Reentrant in this thread. A long operation computes outside
        it, from the state it read, and enters it only to apply: after the
        sync it checks that each section's inputs are unchanged and refuses
        a section whose inputs moved (``STALE_INPUT``). The section records
        are new objects after a reload: resolve them by id inside the block.
        A thread running the job's background work gives way to a lock holder
        that waits for that work (:mod:`langslice.job.background`:
        ``LockYielded``).
        """
        with self.lock.held(self._lock_abort()):
            yield self.sync()

    def _lock_abort(self) -> Callable[[], bool] | None:
        """Return what a wait for the lock asks whether to give up.

        Nothing at this layer; the layer that runs background work answers
        its ``gives_way`` in one of that work's threads.
        """
        return None

    @contextlib.contextmanager
    def reading(self) -> Iterator[None]:
        """For a read: the state brought up to date without waiting for
        another process's writer. This process's other threads are kept out
        (their writes are short); when another process holds the job folder's
        lock (a long fit applying, a plugin tool that holds it while it
        computes), the state is read as that process last saved it, and the
        next locked :meth:`sync` puts the job's own back first."""
        with self.lock.threads():
            with self.lock.held_if_free() as free:
                if free:
                    self.sync()
            if not free:
                self._peek()
            yield

    def _peek(self) -> None:
        """Read the state file as it stands, without the lock (:meth:`reading`)."""
        stamp = file_stamp(self.checkpoint_path)
        if stamp in (self._state_stamp, self._peek_stamp):
            return
        try:
            data = read_checkpoint(self.checkpoint_path, self.upgrade)
        except (OSError, ValueError):
            return
        if data is None:
            return
        held = stack_to_json(self.state)
        loaded = stack_to_json(state_from_file(data, self.checkpoint_path))
        self._peek_stamp = stamp
        if loaded == held:
            return
        if self._peeked is None:
            self._peeked = held
        self.state.replace_with(stack_from_json(loaded))
        self._notify()

    def snapshot(self) -> dict[str, Any]:
        """The state as it stands: take it before a write, hand it to :meth:`commit`."""
        return stack_to_json(self.state)

    def commit(self, before: dict[str, Any]) -> None:
        """Record one write, then checkpoint.

        A history line for each section the write changed, one undo step,
        the redo side cleared. Call it inside :meth:`writing`, with *before*
        taken there.

        Parameters
        ----------
        before
            The state before the write, as :meth:`snapshot` gives it.
        """
        with self.lock.held():
            self._push(self._record(before, self.actor))
            self._end_step()

    def amend(self, before: JsonObject) -> None:
        """Record a change that belongs to the last write, then checkpoint.

        The deformations a write made stale are cleared after it: their
        history lines are appended and their sections joined to the
        write's undo step. Call it inside :meth:`writing`.

        Parameters
        ----------
        before
            The state before the change, as :meth:`snapshot` gives it.
        """
        with self.lock.held():
            if not self.undo_stack:
                # No write to join (an opened state that held a stale
                # deformation): its lines are written and, like the opening
                # lines, no step undoes them.
                self._record(before, self.actor)
                self.checkpoint()
                return
            top = self.undo_stack[-1]
            step = self._record(before, self.actor, step=top.step)
            self.undo_stack[-1] = combined_steps([top, step])
            self._end_step()

    def _end_step(self) -> None:
        """Save the undo steps, then checkpoint the state."""
        self._save_undo()
        self.checkpoint()

    def _push(self, step: UndoStep) -> None:
        self.undo_stack.append(step)
        self.redo_stack.clear()
        self.pushed += 1

    def merge_steps(self, mark: int) -> None:
        """Make the undo steps recorded since :attr:`pushed` was *mark* one
        step: the state before the first of them. A tool whose call writes
        through several operations is still one undo step. Call it inside
        :meth:`writing`."""
        count = min(self.pushed - mark, len(self.undo_stack))
        if count < 2:
            return
        merged = combined_steps(self.undo_stack[-count:])
        del self.undo_stack[-count:]
        self.undo_stack.append(merged)
        # the steps since an earlier mark are still counted right
        self.pushed -= count - 1
        self._save_undo()

    def undo(self) -> bool:
        """Take the last step back; False when there is none."""
        with self.writing():
            if not self.undo_stack:
                return False
            self.redo_stack.append(self._take(self.undo_stack.pop()))
            self._end_step()
            return True

    def redo(self) -> bool:
        """Take the last undone step again; False when there is none."""
        with self.writing():
            if not self.redo_stack:
                return False
            self.undo_stack.append(self._take(self.redo_stack.pop()))
            self._end_step()
            return True

    def _take(self, step: UndoStep) -> UndoStep:
        """Put back the lines and stack fields *step* names; return the
        step that puts the state back as it was before.
        """
        before = stack_to_json(self.state)
        for section_id, seq in step.sections.items():
            section = self.state.by_id(section_id)
            row = self.histories.row(section_id, seq)
            if section is None or row is None:
                logger.warning(
                    "Undo: line %s of %s is not in its history; the "
                    "section stays as it is",
                    seq,
                    section_id,
                )
                continue
            set_head(section, self._current_row(row))
        set_stack_fields(self.state, step.stack_before)
        if step.notes_before is not None:
            self.state.notes = step.notes_before.put_back(self.state.notes)
        # also=step: the step that takes this one back names every section
        # and stack field this one names, changed or not, so redo puts back
        # all that undo did.
        back = self._record(before, self.actor, also=step)
        return dataclasses.replace(
            back, step=step.step, by=step.by, at=step.at
        )

    def _current_row(self, row: Row) -> JsonObject:
        """Return a line's section row at the current state format."""
        if row.format_version == STATE_FORMAT_VERSION or self.upgrade is None:
            return row.section
        upgraded = self.upgrade(row.format_version, {"slices": [row.section]})
        return objects_at(upgraded, "slices")[0]

    def _record(
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
        step being taken back (:meth:`_take`): its sections and stack fields
        are named in the step returned even where they did not change.
        """
        if step is None:
            step = self.next_step
            self.next_step += 1
        then_state = stack_from_json(before)
        now_notes = self.state.notes
        named: Mapping[str, int] = {} if also is None else also.sections
        sections: dict[str, int] = {}
        for section in self.state.slices:
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

    def _stack_before(
        self, before: JsonObject, also: UndoStep | None
    ) -> dict[str, Json]:
        """Return the stack fields but the notes that differ from *before*
        (and those *also* names), as *before* holds them.
        """
        fields = stack_fields(self.state)
        del fields[NOTES]  # a step keeps the notes' change (NotesBefore)
        changed = {
            key: before.get(key)
            for key in fields
            if before.get(key) != fields[key]
        }
        if also is not None:
            changed.update({key: before.get(key) for key in also.stack_before})
        return changed

    def _collect_records(self, section_ids: Iterable[str]) -> None:
        """Delete the record folders of these sections that no line names."""
        if not self.persist:
            return
        for section_id in section_ids:
            section = self.state.by_id(section_id)
            if section is not None:
                self.histories.collect_records(section)

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
        if not self.persist:
            rows = self.histories.rows(section.id)
            return rows[-1].seq if rows else 0
        replaced = registration_key(then_state, old_section) != (
            registration_key(self.state, section)
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
        rows = self.histories.rows(old_section.id)
        head = rows[-1] if rows else None
        key_then = registration_key(then_state, old_section)
        unnumbered = (
            replaced
            and key_then is not None
            and self.histories.number(old_section.id, key_then) is None
        )
        row_then = section_row(old_section)
        if head is not None and head.section == row_then and not unnumbered:
            return head
        return self.histories.append(
            then_state, old_section, step=step, by=HOST, numbered=replaced
        )

    def _line_after(
        self, section: Section, *, step: int, by: str, replaced: bool
    ) -> None:
        """Append the line a write left a section with.

        A registration it newly numbered is noted in :attr:`registered`.
        """
        line = self.histories.append(
            self.state, section, step=step, by=by, numbered=replaced
        )
        if replaced and line.registration is not None:
            self.registered.append(
                (threading.get_ident(), section.id, line.registration)
            )

    def number_registration(self, section: Section) -> int | None:
        """Return the number of the registration a section holds.

        One never numbered is numbered now, by a ``host`` line. Call it
        inside :meth:`writing`.

        Parameters
        ----------
        section
            The section as it stands.

        Returns
        -------
        int or None
            None while it holds none (or for a job that writes nothing).
        """
        number = self.histories.registration_number(self.state, section)
        if number is not None or not self.persist:
            return number
        if registration_key(self.state, section) is None:
            return None
        last_write = max(self.next_step - 1, 0)  # 0: none yet, the opening
        line = self.histories.append(
            self.state, section, step=last_write, by=HOST, numbered=True
        )
        return line.registration

    def record_opening(self) -> None:
        """Append each section's opening line: by ``host``, write 0."""
        if not self.persist:
            return
        with self.lock.held():
            for section in self.state.slices:
                self.histories.append(
                    self.state, section, step=0, by=HOST, numbered=False
                )

    def load_undo(self) -> None:
        """Read the undo steps on disk into the job (none unreadable)."""
        lists = self.undo_file.load()  # its notice is logged
        self.undo_stack, self.redo_stack = lists.undo, lists.redo
        self.next_step = max(self.next_step, lists.next_step)

    def _save_undo(self) -> None:
        if not self.persist:
            return
        with self.lock.held():
            self.undo_file.save(
                UndoLists(self.undo_stack, self.redo_stack, self.next_step)
            )
        self._undo_stamp = file_stamp(self.undo_path)

    def sync(self) -> dict[str, Any] | None:
        """Reload the state file if it changed on disk since this job last
        read or wrote it.

        Returns the state as this job held it before a reload, or ``None``
        when nothing was reloaded. An edit from outside (``undo.json``
        unchanged) becomes one undo step by ``script``; when ``undo.json``
        changed too, another job wrote both and its steps are read back
        instead. A file that does not parse (a script mid-write) is left
        alone and read at the next call; the job's own next write replaces it.
        """
        if (
            self._peeked is not None
        ):  # a read's view first gives way to the job's own state
            self.state.replace_with(stack_from_json(self._peeked))
            self._peeked, self._peek_stamp = None, None
        state_stamp = file_stamp(self.checkpoint_path)
        undo_stamp = file_stamp(self.undo_path)
        history_changed = undo_stamp != self._undo_stamp
        if history_changed:
            self.load_undo()
            self._undo_stamp = undo_stamp
        if state_stamp == self._state_stamp:
            return None
        try:
            data = read_checkpoint(self.checkpoint_path, self.upgrade)
        except (OSError, ValueError):
            logger.warning(
                "State file %s changed but does not read; keeping the job's state",
                self.checkpoint_path,
                exc_info=True,
            )
            return None
        self._state_stamp = state_stamp
        if data is None:  # deleted: the next checkpoint writes it again
            return None
        held = stack_to_json(self.state)
        loaded = stack_to_json(state_from_file(data, self.checkpoint_path))
        if loaded == held:
            return None
        self.state.replace_with(stack_from_json(loaded))
        if not history_changed:
            with self.lock.held():
                self._push(self._record(held, SCRIPT))
                self._save_undo()
        logger.info("Reloaded %s: it changed on disk", self.checkpoint_path)
        self._notify()
        return held

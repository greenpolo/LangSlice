"""Undo steps, and the live sync that keeps the job with its files.

**Undo.** One pattern: ``before = job.snapshot()`` before a write, then
``job.commit(before)`` once it succeeded (one undo step, then the
checkpoint). A step is the whole state, so undo restores everything on it,
the deformation references in the section records included (the records
themselves are content-addressed on disk and never change). The history is
the job folder's ``history/`` (:mod:`langslice.job.history`: an index and one
file per step), depth :data:`~langslice.job.history.UNDO_DEPTH`; a resumed
job reads it back, a fresh one starts empty.

**Live shared editing.** :meth:`JobUndo.sync` notices that the state file
changed on disk since this job last read or wrote it (a script edited it, or
another job on the same folder wrote it) and reloads it before the next
operation instead of overwriting it. A plain edit becomes one undo step;
when the history file changed with it (another job wrote both), that history
is read back instead.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterator
from typing import Any

from langslice.job.checkpoint import JobCheckpoint, file_stamp, state_from_file
from langslice.job.history import UNDO_DEPTH
from langslice.job.state_file import (
    read_checkpoint,
    stack_from_json,
    stack_to_json,
)

logger = logging.getLogger(__name__)


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
        """Record one undo step (*before*), clear the redo side, checkpoint,
        and list the registrations the step left (:meth:`list_registrations`).
        Call it inside :meth:`writing` (with *before* taken there)."""
        with self.lock.held():
            self._push(before)
            self._save_history()
            self.checkpoint()
            self.list_registrations(before)

    def _push(self, before: dict[str, Any]) -> None:
        self.undo_stack.append(before)
        del self.undo_stack[:-UNDO_DEPTH]
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
        first = self.undo_stack[-count]
        del self.undo_stack[-count:]
        self.undo_stack.append(first)
        self.pushed -= count - 1  # the steps since an earlier mark are still counted right
        self._save_history()

    def undo(self) -> bool:
        """Restore the state before the last step; False when there is none."""
        with self.writing():
            if not self.undo_stack:
                return False
            before = stack_to_json(self.state)
            self.redo_stack.append(before)
            self.state.replace_with(stack_from_json(self.undo_stack.pop()))
            self._save_history()
            self.checkpoint()
            self.list_registrations(before)
            return True

    def redo(self) -> bool:
        """Re-apply the step :meth:`undo` reversed; False when there is none."""
        with self.writing():
            if not self.redo_stack:
                return False
            before = stack_to_json(self.state)
            self.undo_stack.append(before)
            del self.undo_stack[:-UNDO_DEPTH]
            self.state.replace_with(stack_from_json(self.redo_stack.pop()))
            self._save_history()
            self.checkpoint()
            self.list_registrations(before)
            return True

    def _save_history(self) -> None:
        if not self.persist:
            return
        with self.lock.held():
            self.history.save(self.undo_stack, self.redo_stack)
        self._undo_stamp = file_stamp(self.undo_path)

    def sync(self) -> dict[str, Any] | None:
        """Reload the state file if it changed on disk since this job last
        read or wrote it.

        Returns the state as this job held it before a reload, or ``None``
        when nothing was reloaded. An edit from outside (the history file
        unchanged) becomes one undo step; when the history file changed too,
        another job wrote both and its history is read back instead. A file
        that does not parse (a script mid-write) is left alone and read at
        the next call; the job's own next write replaces it.
        """
        if self._peeked is not None:  # a read's view first gives way to the job's own state
            self.state.replace_with(stack_from_json(self._peeked))
            self._peeked, self._peek_stamp = None, None
        state_stamp = file_stamp(self.checkpoint_path)
        undo_stamp = file_stamp(self.undo_path)
        history_changed = undo_stamp != self._undo_stamp
        if history_changed:
            self.undo_stack, self.redo_stack = self.history.load()
            self._undo_stamp = undo_stamp
        if state_stamp == self._state_stamp:
            return None
        try:
            data = read_checkpoint(self.checkpoint_path, self.upgrade)
        except (OSError, ValueError):
            logger.warning("State file %s changed but does not read; keeping the job's state",
                           self.checkpoint_path, exc_info=True)
            return None
        self._state_stamp = state_stamp
        if data is None:  # deleted: the next checkpoint writes it again
            return None
        held = stack_to_json(self.state)
        loaded = stack_to_json(state_from_file(data, self.checkpoint_path))
        if loaded == held:
            return None
        if not history_changed:
            self._push(held)
            self._save_history()
        self.state.replace_with(stack_from_json(loaded))
        logger.info("Reloaded %s: it changed on disk", self.checkpoint_path)
        self._notify()
        return held

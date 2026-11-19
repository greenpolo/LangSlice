"""The checkpoint: the state file, and what is written and told after it.

:class:`Checkpoints` is the job's ``checkpoints``
(:class:`langslice.job.job.Job`): it writes the state file, then
``registration.json`` and the landmark files as the lock's outermost hold
ends, and tells the observers. Every path the state stores (a deformation's
``record``, a border trace's artifacts) is relative to the job folder.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

from langslice.core.jsonable import JsonObject
from langslice.core.stack.stack import Stack
from langslice.job.exports import public_files as formats
from langslice.job.folder.layout import JobLayout
from langslice.job.folder.lock import FolderLock
from langslice.job.state_file import stack_from_json, write_checkpoint

if TYPE_CHECKING:
    from langslice.core.workspace import Workspace

logger = logging.getLogger(__name__)


Stamp = tuple[int, int, int] | None


def file_stamp(path: str) -> Stamp:
    """Return a file's identity on disk.

    Parameters
    ----------
    path
        The file whose identity is read.

    Returns
    -------
    Stamp
        Its inode, size and modification time; None when it cannot be read.
    """
    try:
        info = os.stat(path)
    except OSError:
        return None
    return (info.st_ino, info.st_size, info.st_mtime_ns)


def state_from_file(data: JsonObject, path: str | os.PathLike[str]) -> Stack:
    """Return the stack a state file holds.

    Parameters
    ----------
    data
        The file's JSON, at the current format.
    path
        The file it was read from, which an error names.

    Returns
    -------
    Stack

    Raises
    ------
    ValueError
        When the file holds unknown keys, naming the file and the keys.
    """
    try:
        return stack_from_json(data)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


class Checkpoints:
    """Write the state file, then tell the observers.

    Parameters
    ----------
    state
        The stack the job registers.
    layout
        The job folder and the paths in it.
    lock
        The job folder's write lock; the files derived from the state are
        written as its outermost hold ends.
    persist
        False: nothing is written (a dry run).
    seen
        The stamp of the state file taken before it was read; None takes it
        now.
    """

    def __init__(
        self,
        state: Stack,
        layout: JobLayout,
        lock: FolderLock,
        *,
        persist: bool,
        seen: Stamp = None,
    ) -> None:
        self.state = state
        self.workspace: Workspace | None = None
        self._layout = layout
        self.stamp: Stamp = seen if seen is not None else file_stamp(self.path)
        self._lock = lock
        self._persist = persist
        self._derived_due = False
        self._observers: list[Callable[[Stack], None]] = []
        lock.before_release.append(self._write_derived)

    @property
    def path(self) -> str:
        """The state checkpoint, ``<job folder>/state.json``."""
        return str(self._layout.state_file)

    def write(self) -> None:
        """Write the state (atomically, versioned), then tell every observer.

        Its public rendering, ``registration.json``
        (:func:`langslice.job.exports.public_files.write_registration`; a
        failure there
        is logged, never raised), and the landmark files follow once, as the
        outermost hold of the job folder's lock ends: a write that checkpoints
        several times in one hold (the write, then the deformations it made
        stale) maps the stack once, from the state it leaves, still under the
        lock."""
        from langslice.core.fits.landmarks import stale

        for section in self.state.slices:
            if section.landmarks is not None:
                pairs = tuple(
                    pair.model_copy(update={"stale": stale(self.state, section, pair)})
                    for pair in section.landmarks.pairs
                )
                section.landmarks = section.landmarks.model_copy(update={"pairs": pairs})
        if not self._persist:
            return
        with self._lock.held():
            write_checkpoint(self.state, self.path)
            self.stamp = file_stamp(self.path)
            self._derived_due = True
        self.notify()

    def _write_derived(self) -> None:
        """``registration.json`` and the landmark files of the state as it
        stands, when a checkpoint owes them (called by the lock, held)."""
        if not self._derived_due or not self._persist:
            return
        self._derived_due = False
        formats.write_registration(self._layout, self.state, self.workspace)
        formats.write_landmarks(self._layout, self.state)

    def notify(self) -> None:
        """Call every observer with the state."""
        for observer in list(self._observers):
            try:
                observer(self.state)
            except Exception:
                logger.exception("Job observer raised; ignoring")

    @contextlib.contextmanager
    def observe(self, fn: Callable[[Stack], None]) -> Iterator[None]:
        """Call *fn* with the state after every checkpoint and reload, while open."""
        self._observers.append(fn)
        try:
            yield
        finally:
            self._observers.remove(fn)

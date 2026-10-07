"""The job folder's write lock: one writer at a time, across processes.

Live shared editing (a running agent, CLI calls, scripts on one job folder)
needs every write to go lock -> sync (reload the state if it changed on
disk) -> apply -> commit -> unlock (:meth:`langslice.job.job.Job.writing`).
The lock is ``job.lock`` in the job folder, held through ``filelock``
(``fcntl`` on Linux and macOS, ``msvcrt`` on Windows; the operating system
releases it when a process dies). It is reentrant in the thread that holds
it, and two :class:`FolderLock` objects on one folder (two jobs in one
process) exclude each other like two processes do. Threads of one process
holding one :class:`FolderLock` (a job's background work beside the agent's
calls) also exclude each other through a thread lock taken first, which
holds even where the folder lock cannot be used. A waiting thread can be
told to give up instead (``held(abort=...)``: :class:`LockYielded`), which
is how a job's background work gives way to a thread that holds the lock
and waits for that work (:mod:`langslice.job.background`).

A folder that does not exist yet (a job made around a state, nothing
written) or cannot be written (a job read off read-only media) is used
without the lock: nothing can be written there anyway.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

LOCK_FILE = "job.lock"
#: How long a writer waits for another before giving up (:class:`JobBusy`).
#: Writes hold the lock for a commit; long fits compute outside it.
LOCK_TIMEOUT_S = 300.0
#: How often a waiting thread with an ``abort`` check asks it, in seconds.
ABORT_POLL_S = 0.05


class JobBusy(RuntimeError):
    """Another writer held the job folder's lock past the timeout."""


class LockYielded(RuntimeError):
    """A thread waiting for the lock gave up because its ``abort`` check said so."""


class FolderLock:
    """The write lock of one job folder (see the module text)."""

    def __init__(self, folder: Path, timeout: float = LOCK_TIMEOUT_S) -> None:
        self.path = Path(folder) / LOCK_FILE
        self.timeout = timeout
        self._lock: Any = None
        self._unusable = False
        #: The in-process lock taken before the folder's (reentrant).
        self._threads = threading.RLock()
        self._owner: int | None = None
        self._depth = 0

    def held_here(self) -> bool:
        """Whether the calling thread holds the lock."""
        return self._owner == threading.get_ident()

    @contextlib.contextmanager
    def held(self, abort: Callable[[], bool] | None = None) -> Iterator[None]:
        """Hold the lock for the block (reentrant in this thread).

        With *abort*, a thread that has to wait for another one asks it every
        :data:`ABORT_POLL_S` seconds and raises :class:`LockYielded` once it
        says True (nothing held)."""
        if abort is None or self.held_here():
            self._threads.acquire()
        else:
            while not self._threads.acquire(timeout=ABORT_POLL_S):
                if abort():
                    raise LockYielded(f"gave way to another writer of {self.path.parent}")
        self._owner = threading.get_ident()
        self._depth += 1
        try:
            with self._folder():
                yield
        finally:
            self._depth -= 1
            if self._depth == 0:
                self._owner = None
            self._threads.release()

    @contextlib.contextmanager
    def _folder(self) -> Iterator[None]:
        """The folder's file lock (none where it cannot be used)."""
        if self._unusable or not self.path.parent.is_dir():
            yield
            return
        if self._lock is None:
            from filelock import FileLock

            self._lock = FileLock(str(self.path), timeout=self.timeout)
        from filelock import Timeout

        try:
            self._lock.acquire()
        except Timeout as exc:
            raise JobBusy(f"{self.path.parent} is locked by another writer (waited "
                          f"{self.timeout:g} s)") from exc
        except PermissionError:
            logger.warning("Cannot lock %s (not writable); working without the lock",
                           self.path)
            self._unusable = True
            yield
            return
        try:
            yield
        finally:
            self._lock.release()

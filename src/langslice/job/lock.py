"""The job folder's write lock: one writer at a time, across processes.

Live shared editing (a running agent, CLI calls, scripts on one job folder)
needs every write to go lock -> sync (reload the state if it changed on
disk) -> apply -> commit -> unlock (:meth:`langslice.linear.job.Job.writing`).
The lock is ``job.lock`` in the job folder, held through ``filelock``
(``fcntl`` on Linux and macOS, ``msvcrt`` on Windows; the operating system
releases it when a process dies). It is reentrant in the thread that holds
it, and two :class:`FolderLock` objects on one folder (two jobs in one
process) exclude each other like two processes do.

A folder that does not exist yet (a job made around a state, nothing
written) or cannot be written (a job read off read-only media) is used
without the lock: nothing can be written there anyway.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

LOCK_FILE = "job.lock"
#: How long a writer waits for another before giving up (:class:`JobBusy`).
#: Writes hold the lock for a commit; long fits compute outside it.
LOCK_TIMEOUT_S = 300.0


class JobBusy(RuntimeError):
    """Another writer held the job folder's lock past the timeout."""


class FolderLock:
    """The write lock of one job folder (see the module text)."""

    def __init__(self, folder: Path, timeout: float = LOCK_TIMEOUT_S) -> None:
        self.path = Path(folder) / LOCK_FILE
        self.timeout = timeout
        self._lock: Any = None
        self._unusable = False

    @contextlib.contextmanager
    def held(self) -> Iterator[None]:
        """Hold the lock for the block (reentrant in this thread)."""
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

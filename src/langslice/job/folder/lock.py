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
calls) also exclude each other through a thread gate taken first, which
holds even where the folder lock cannot be used. The gate lets readers in
together (:meth:`FolderLock.shared`: looks, zooms and the other reads run side
by side) and a writer alone; a thread that reads and then has to write gives
its read up while it waits, so two such threads never wait on each other. A waiting thread can be
told to give up instead (``held(abort=...)``: :class:`LockYielded`), which
is how a job's background work gives way to a thread that holds the lock
and waits for that work (:mod:`langslice.job.background`).

A folder that does not exist yet (a job made around a state, nothing
written) or cannot be written (a job read off read-only media) is used
without the lock: nothing can be written there anyway. A lock file that
cannot be opened for a moment (Windows refuses one another process is
deleting) is tried again before the folder is probed for writing.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

from filelock import BaseFileLock, FileLock, Timeout

from langslice.core.shared_files import retried
from langslice.job.folder.layout import writable

logger = logging.getLogger(__name__)

LOCK_FILE = "job.lock"
LOCK_TIMEOUT_S = 300.0
ABORT_POLL_S = 0.05


class JobBusy(RuntimeError):
    """Another writer held the job folder's lock past the timeout."""


class LockYielded(RuntimeError):
    """A thread waiting for the lock gave up because its ``abort`` check said so."""


class _Gate:
    """Many readers or one writer among the threads of one process.

    Reentrant both ways: the writer may read and write again, a reader may
    read again. A reader that asks to write lets its reads go while it waits
    and takes them back after, so readers that all ask to write still get
    their turns one by one. Readers come in whenever no thread writes; a
    waiting writer does not hold new readers back, as reads are short.
    """

    def __init__(self) -> None:
        self._changed = threading.Condition(threading.Lock())
        self._writer: int | None = None
        self._writes = 0
        self._reads: dict[int, int] = {}

    def _free_for(self, me: int) -> bool:
        return self._writer in (None, me) and not any(
            count for thread, count in self._reads.items() if thread != me
        )

    def write(self, timeout_s: float | None = None) -> bool:
        """Become the writer; False when *timeout_s* ran out first."""
        me = threading.get_ident()
        with self._changed:
            kept = self._reads.pop(me, 0)
            if not self._changed.wait_for(lambda: self._free_for(me), timeout_s):
                if kept:
                    self._reads[me] = kept
                return False
            self._writer = me
            self._writes += 1
            if kept:
                self._reads[me] = kept
            return True

    def done_writing(self) -> None:
        """Let one write go; the gate opens when the writer's last one has."""
        with self._changed:
            self._writes -= 1
            if self._writes == 0:
                self._writer = None
            self._changed.notify_all()

    def read(self) -> None:
        """Become a reader, once no other thread writes."""
        me = threading.get_ident()
        with self._changed:
            self._changed.wait_for(lambda: self._writer in (None, me))
            self._reads[me] = self._reads.get(me, 0) + 1

    def done_reading(self) -> None:
        """Let one read go."""
        me = threading.get_ident()
        with self._changed:
            left = self._reads.get(me, 0) - 1
            if left > 0:
                self._reads[me] = left
            else:
                self._reads.pop(me, None)
            self._changed.notify_all()


class FolderLock:
    """The write lock of one job folder (see the module text)."""

    def __init__(self, folder: Path, timeout: float = LOCK_TIMEOUT_S) -> None:
        self.path = Path(folder) / LOCK_FILE
        self.timeout = timeout
        self._lock: BaseFileLock | None = None
        self._unusable = False
        self._gate = _Gate()
        self._owner: int | None = None
        self._depth = 0
        self.before_release: list[Callable[[], None]] = []

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
            self._gate.write()
        else:
            while not self._gate.write(timeout_s=ABORT_POLL_S):
                if abort():
                    raise LockYielded(f"gave way to another writer of {self.path.parent}")
        self._owner = threading.get_ident()
        self._depth += 1
        try:
            with self._folder():
                try:
                    yield
                finally:
                    if self._depth == 1:
                        self._released()
        finally:
            self._depth -= 1
            if self._depth == 0:
                self._owner = None
            self._gate.done_writing()

    @contextlib.contextmanager
    def shared(self) -> Iterator[None]:
        """Read beside this process's other readers for the block: only a
        writing thread is kept out (reentrant); the folder's lock is not taken."""
        self._gate.read()
        try:
            yield
        finally:
            self._gate.done_reading()

    @contextlib.contextmanager
    def alone_if_free(self) -> Iterator[bool]:
        """This process's other threads kept out for the block when none of
        them holds the gate now (yields True), else nothing (yields False):
        never waits for another thread. The folder's lock is not taken."""
        if not self._gate.write(timeout_s=0):
            yield False
            return
        try:
            yield True
        finally:
            self._gate.done_writing()

    @contextlib.contextmanager
    def held_if_free(self) -> Iterator[bool]:
        """The lock for the block when no other thread or process holds it
        (yields True), else nothing (yields False): never waits for another
        writer or reader. Where the folder lock cannot be used, the same as
        :meth:`held` when no other thread holds the gate."""
        with self.alone_if_free() as alone:
            if not alone:
                yield False
                return
            if self.held_here() or self._unusable or not self.path.parent.is_dir():
                with self.held():
                    yield True
                return
            lock = self._file_lock()
            try:
                retried(functools.partial(lock.acquire, timeout=0))
            except Timeout:
                yield False
                return
            except PermissionError:
                self._unusable = self._is_unwritable()
                with self.held():
                    yield True
                return
            try:
                with self.held():
                    yield True
            finally:
                lock.release()

    def _file_lock(self) -> BaseFileLock:
        if self._lock is None:
            self._lock = FileLock(str(self.path), timeout=self.timeout)
        return self._lock

    def _is_unwritable(self) -> bool:
        """Tell whether the folder refuses a new file.

        Only then is the lock left off for good; a lock file refused for
        longer than a retry is tried again at the next hold.
        """
        folder = self.path.parent
        return not writable(folder, folder)

    def _released(self) -> None:
        for callback in list(self.before_release):
            try:
                callback()
            except Exception:
                logger.exception("A lock release callback of %s failed", self.path.parent)

    @contextlib.contextmanager
    def _folder(self) -> Iterator[None]:
        """The folder's file lock (none where it cannot be used)."""
        if self._unusable or not self.path.parent.is_dir():
            yield
            return
        lock = self._file_lock()
        try:
            retried(lock.acquire)
        except Timeout as exc:
            raise JobBusy(
                f"{self.path.parent} is locked by another writer (waited {self.timeout:g} s); "
                "call again once it has finished"
            ) from exc
        except PermissionError:
            logger.warning("Cannot lock %s; working without the lock", self.path)
            self._unusable = self._is_unwritable()
            yield
            return
        try:
            yield
        finally:
            lock.release()

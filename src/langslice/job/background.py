"""Background work: a long operation that runs while the agent carries on.

A tool that takes minutes (``trace_borders``: an image-model call, then a
fit of what it drew) starts its work here and returns at once with the
work's id. The work runs on a thread of the job's :class:`BackgroundWork`
registry (:attr:`langslice.job.job.Job.background`): it first waits for the
image-model calls of its sections (:meth:`~langslice.job.job.Job.wait_image_job`,
the job's image-correction machinery, which records each reply on its
section), then runs its *land* callable, which writes its result as its own
undo step and says what happened. The registry saves the pictures *land*
drew among the job's pictures (their numbers are the work's ``pictures``)
and keeps a notice per finished work until it is asked for:

- :meth:`BackgroundWork.notices` pops the work finished since the last call
  (a door puts their notices at the head of its next reply);
- :meth:`BackgroundWork.running` lists the work still running (status);
- :meth:`BackgroundWork.wait_all` waits for every piece (submit).

The callable is the caller's (an operation's fit), so this module imports
no operation and no provider.

**Giving way.** ``submit`` runs under the job's write lock and waits for the
work, whose landing needs that lock. While a thread that holds the lock
waits here, a work thread that would have to wait for the lock gives up
instead (:class:`langslice.job.lock.LockYielded`, before it wrote
anything), and the waiting thread runs that work's landing itself.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langslice.core.layers import collecting, note_for
from langslice.job.lock import LockYielded

if TYPE_CHECKING:
    from PIL import Image

    from langslice.job.job import Job

logger = logging.getLogger(__name__)

RUNNING, DONE, FAILED = "running", "done", "failed"
#: A piece of work's states.
STATUSES = (RUNNING, DONE, FAILED)
#: Work threads one job runs at once. Each mostly waits for an image-model
#: call; the fits themselves run one at a time (``core.deformation``).
MAX_WORKERS = 8
#: Prefix of a work id (``w1``, ``w2``, ... in the order the job started them).
ID_PREFIX = "w"

_WORKER = threading.local()


@dataclass
class Landed:
    """What a work's *land* callable did: ``done`` (its result written, or
    nothing to write) or ``failed`` (nothing written), the notice's text and
    the pictures it drew (saved by the registry)."""

    status: str
    text: str
    pictures: list[Image.Image] = field(default_factory=list)
    #: Plain facts for a door or a script (the fit's row, the trace's record).
    result: dict[str, Any] = field(default_factory=dict)


@dataclass
class Work:
    """One piece of background work and, once finished, its notice."""

    id: str
    kind: str
    sections: tuple[str, ...]
    status: str = RUNNING
    #: Wall-clock seconds (``time.time()``) it started and finished.
    started: float = 0.0
    finished: float | None = None
    #: One paragraph for the agent, set when it finishes (:func:`notice_text`).
    notice: str = ""
    #: The numbers of the pictures it drew, in the job's picture index.
    pictures: list[int] = field(default_factory=list)
    #: The pictures themselves, for a door that sends them with the notice.
    images: list[Image.Image] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)
    #: Whether :meth:`BackgroundWork.notices` handed the notice out.
    delivered: bool = False

    def summary(self) -> dict[str, Any]:
        """The work as plain data (status, a script's view)."""
        return {"id": self.id, "kind": self.kind, "sections": list(self.sections),
                "status": self.status, "started": round(self.started, 3),
                **({"finished": round(self.finished, 3)} if self.finished is not None else {}),
                **({"notice": self.notice} if self.notice else {}),
                **({"pictures": list(self.pictures)} if self.pictures else {})}


def notice_text(work: Work, text: str) -> str:
    """``"<id> <kind> of <sections> finished|failed: <text> Pictures #n, #m."``"""
    sections = ", ".join(work.sections) or "the stack"
    verb = "finished" if work.status == DONE else "failed"
    pictures = (" Pictures " + ", ".join(f"#{seq}" for seq in work.pictures) + "."
                if work.pictures else "")
    return f"{work.id} {work.kind} of {sections} {verb}: {text.strip()}{pictures}"


def in_worker() -> bool:
    """Whether the calling thread is running a piece of background work."""
    return bool(getattr(_WORKER, "active", False))


@dataclass
class _Entry:
    work: Work
    land: Callable[[], Landed]
    wait_for: tuple[str, ...]
    future: Future[None] | None = None
    #: Taken by whichever thread runs the landing (the work thread, or a
    #: waiting lock holder after the work thread gave way).
    claimed: bool = False
    yielded: bool = False
    finished: threading.Event = field(default_factory=threading.Event)


class BackgroundWork:
    """The background work of one job (see the module text)."""

    def __init__(self, job: Job) -> None:
        self._job = job
        self._lock = threading.Lock()
        self._entries: dict[str, _Entry] = {}
        self._count = 0
        self._executor: ThreadPoolExecutor | None = None
        #: Set while a thread that holds the job's write lock waits here.
        self._draining = threading.Event()

    # --- starting ---------------------------------------------------------------------

    def start(
        self, kind: str, sections: Sequence[str], land: Callable[[], Landed], *,
        wait_for_images: bool = True,
    ) -> Work:
        """Start one piece of work and return it at once (status ``running``).

        On a work thread: wait for the running image-model call of each of
        *sections* (with *wait_for_images*), then run *land*, which writes
        its result (its own undo step) and says what happened. An exception
        from *land* finishes the work as ``failed`` with the exception's text.
        """
        with self._lock:
            self._count += 1
            work = Work(id=f"{ID_PREFIX}{self._count}", kind=kind,
                        sections=tuple(str(name) for name in sections), started=time.time())
            entry = _Entry(work=work, land=land,
                           wait_for=work.sections if wait_for_images else ())
            self._entries[work.id] = entry
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=MAX_WORKERS,
                                                    thread_name_prefix="background-work")
            entry.future = self._executor.submit(self._thread, entry)
        return work

    # --- reading ----------------------------------------------------------------------

    def get(self, work_id: str) -> Work | None:
        """The work named *work_id*, or None."""
        entry = self._entries.get(str(work_id))
        return entry.work if entry is not None else None

    def all(self) -> list[Work]:
        """Every piece of work this job started, oldest first."""
        with self._lock:
            return [entry.work for entry in self._entries.values()]

    def running(self) -> list[Work]:
        """The work still running, oldest first."""
        return [work for work in self.all() if work.status == RUNNING]

    def running_for(self, section: str, kind: str | None = None) -> Work | None:
        """The running work of *kind* (any kind when None) on *section*, or None."""
        for work in self.running():
            if section in work.sections and (kind is None or work.kind == kind):
                return work
        return None

    def notices(self) -> list[Work]:
        """The work finished since the last call, each handed out once."""
        with self._lock:
            ready = [entry.work for entry in self._entries.values()
                     if entry.work.status != RUNNING and not entry.work.delivered]
            for work in ready:
                work.delivered = True
        return ready

    # --- waiting ----------------------------------------------------------------------

    def wait_all(self, timeout: float | None = None) -> list[Work]:
        """Wait until every piece of work started so far has finished; return them.

        Called by a thread that holds the job's write lock (``submit``),
        work that would wait for the lock gives way and its landing runs on
        the calling thread (module text). *timeout* None waits as long as it
        takes; otherwise the work still running at the deadline stays running.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        holding = self._job.lock.held_here()
        if holding:
            self._draining.set()
        try:
            while True:
                with self._lock:
                    pending = [entry for entry in self._entries.values()
                               if not entry.finished.is_set()]
                if not pending:
                    break
                for entry in pending:
                    left = None if deadline is None else max(0.0, deadline - time.monotonic())
                    if entry.future is not None:
                        try:
                            entry.future.result(timeout=left)
                        except TimeoutError:
                            return self.all()
                    if entry.yielded and not entry.finished.is_set():
                        self._land(entry, retry=True)
                    if not entry.finished.wait(timeout=left):
                        return self.all()
        finally:
            if holding:
                self._draining.clear()
        return self.all()

    def wait_any(self, timeout: float | None = None, poll: float = 0.25) -> bool:
        """Wait until a piece of work running now has finished (at once when
        none runs); False when *timeout* passed first. For a caller that does
        not hold the job's write lock (the agent session between turns)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                pending = [entry for entry in self._entries.values()
                           if not entry.finished.is_set()]
            if not pending:
                return True
            if any(entry.finished.wait(poll / len(pending)) for entry in pending):
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False

    def gives_way(self) -> bool:
        """Whether a work thread waiting for the job's lock should give up now."""
        return self._draining.is_set()

    # --- the work thread --------------------------------------------------------------

    def _thread(self, entry: _Entry) -> None:
        _WORKER.active = True
        try:
            for section in entry.wait_for:
                self._job.wait_image_job(section, None)
            self._land(entry, retry=False)
        except LockYielded:
            entry.yielded = True
        except Exception as exc:  # the work's own failures are its notice
            logger.warning("Background work %s failed", entry.work.id, exc_info=True)
            self._finish(entry, Landed(status=FAILED, text=f"{type(exc).__name__}: {exc}"))
        finally:
            _WORKER.active = False

    def _land(self, entry: _Entry, *, retry: bool) -> None:
        """Run the entry's landing once (on whichever thread claims it) and finish it."""
        with self._lock:
            if entry.claimed and not retry:
                return
            entry.claimed = True
        if retry:
            for section in entry.wait_for:
                self._job.wait_image_job(section, None)
        try:
            with self._pictures(entry) as drawn:
                landed = entry.land()
                drawn.extend(landed.pictures)
        except LockYielded:
            with self._lock:
                entry.claimed = False
            raise
        except Exception as exc:
            logger.warning("Background work %s failed", entry.work.id, exc_info=True)
            landed = Landed(status=FAILED, text=f"{type(exc).__name__}: {exc}")
        self._finish(entry, landed)

    @contextmanager
    def _pictures(self, entry: _Entry) -> Iterator[list[Image.Image]]:
        """Collect what the core notes while the landing draws, then save the
        pictures it hands back among the job's pictures (their numbers on the
        work). Saving never fails the work."""
        drawn: list[Image.Image] = []
        with collecting() as notes:
            yield drawn
        if not drawn:
            return
        job = self._job
        try:
            names = job.views.save(
                tool=entry.work.kind, pictures=[(image, note_for(image, notes))
                                                for image in drawn],
                arguments={"work": entry.work.id, "sections": list(entry.work.sections)},
                atlas=job.workspace.atlas if job.workspace is not None else None)
            entry.work.pictures = [int(name.split("_", 1)[0]) for name in names]
        except Exception:
            logger.warning("Could not save the pictures of %s", entry.work.id, exc_info=True)
        entry.work.images = list(drawn)

    def _finish(self, entry: _Entry, landed: Landed) -> None:
        work = entry.work
        with self._lock:
            work.status = DONE if landed.status == DONE else FAILED
            work.result = dict(landed.result)
            work.finished = time.time()
            work.notice = notice_text(work, landed.text)
        entry.finished.set()

    def close(self) -> None:
        """Wait for every piece of work, then stop the work threads."""
        self.wait_all()
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True)

"""The job's background border traces: started, waited for, landed.

A border trace (``trace_borders``) calls the image model off the tool call's
thread. :class:`JobBorderTraces` runs each in a pool, and lands its result
on the section under the write lock, only while the section still holds
the running record for the same geometry.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from langslice.core.jsonable import JsonObject
from langslice.core.refusal import Refused
from langslice.core.stack.codec import trace_from_json
from langslice.core.stack.stack import Stack
from langslice.job.background import BackgroundWork, in_worker
from langslice.job.undo import JobUndo


def _job_result(
    section_id: str, fingerprint: str, future: Future[dict[str, Any]], timeout: float | None,
) -> dict[str, Any]:
    """A correction job's result; an exception becomes an error result.

    Raises ``TimeoutError`` when *timeout* passes first.
    """
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        if not future.done():  # the wait ran out; a job's own timeout is a result
            raise
        error: BaseException = future.exception() or TimeoutError()
        failure = Refused.from_exception(error).payload()
        return {"id": section_id, "geometry_fingerprint": fingerprint, **failure}
    except Exception as exc:  # the job records its own failures; this is a backstop
        failure = Refused.from_exception(exc).payload()
        return {"id": section_id, "geometry_fingerprint": fingerprint, **failure}


def _land(state: Stack, section_id: str, fingerprint: str, result: JsonObject) -> bool:
    """Record a finished correction on a section still waiting for it."""
    section = state.by_id(section_id)
    held = None if section is None else section.border_trace
    if (section is not None and held is not None and held.status == "running"
            and held.geometry_fingerprint == fingerprint):
        section.border_trace = trace_from_json(result)
        return True
    return False



class JobBorderTraces(JobUndo):
    """Run border traces in the background and land their results."""

    #: The job's background work (:attr:`background`), made on first use.
    _background: BackgroundWork | None = None

    @property
    def background(self) -> BackgroundWork:
        """The job's background work, made on first use.

        It waits for these border traces (:mod:`langslice.job.background`).
        """
        if self._background is None:
            self._background = BackgroundWork(self)
        return self._background

    def _lock_abort(self) -> Callable[[], bool] | None:
        """Return ``gives_way`` in a thread of the background work.

        Such a thread gives way to a lock holder that waits for that work
        (``LockYielded``); any other thread waits.
        """
        return self.background.gives_way if in_worker() else None

    def start_image_job(
        self, section_id: str, fingerprint: str, job: Callable[[], dict[str, Any]], *,
        workers: int,
    ) -> None:
        """Run one border trace in the background."""
        if self.image_executor is None:
            self.image_executor = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="image-correction"
            )
        self.image_jobs[section_id] = (fingerprint, self.image_executor.submit(job))

    def image_job_running(self, section_id: str, fingerprint: str) -> bool:
        running = self.image_jobs.get(section_id)
        return running is not None and running[0] == fingerprint and not running[1].done()

    def settle_border_traces(self) -> bool:
        """Wait for every running border trace, record its result, checkpoint.

        A result lands only on a section that still holds the running record
        for the same geometry; one undone or superseded meanwhile keeps what
        it has (the reply stays on disk and is reused at that geometry).

        Returns
        -------
        bool
            Whether any section changed (no undo step: a landing finishes
            the step that started it).
        """
        running = list(self.image_jobs.items())
        results = [
            (section_id, fingerprint,
             self.portable(_job_result(section_id, fingerprint, future, None)))
            for section_id, (fingerprint, future) in running]
        changed = False
        if results:
            # The calls ran outside the lock; they land under it, and only
            # then are forgotten: background work waiting for a call finds it
            # until its result is recorded (wait_image_job).
            with self.writing():
                for section_id, fingerprint, result in results:
                    changed |= _land(self.state, section_id, fingerprint,
                                     result)
                if changed:
                    self.checkpoint()
                for section_id, call in running:
                    if self.image_jobs.get(section_id) is call:
                        del self.image_jobs[section_id]
        if self.image_executor is not None:
            self.image_executor.shutdown(wait=True)
            self.image_executor = None
        return changed

    def wait_image_job(self, section_id: str, timeout: float | None) -> bool:
        """Wait up to *timeout* seconds (None: as long as it takes) for one
        section's running correction.

        Records its result the way :meth:`settle_border_traces` does
        (under the write lock, checkpointed, no undo step) and returns True
        once nothing is running for the section; False when the call is
        still running at the timeout. The call is forgotten only once its
        result is recorded, so a waiter that gives way under the lock
        (background work) leaves it for the next.
        """
        running = self.image_jobs.get(section_id)
        if running is None:
            return True
        fingerprint, future = running
        try:
            result = _job_result(section_id, fingerprint, future,
                                 None if timeout is None else max(0.0, timeout))
        except TimeoutError:
            return False
        with self.writing():
            if _land(self.state, section_id, fingerprint, self.portable(result)):
                self.checkpoint()
            if self.image_jobs.get(section_id) is running:
                del self.image_jobs[section_id]
        return True

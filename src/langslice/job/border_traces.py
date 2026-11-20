"""The job's background border traces: started, waited for, landed.

A border trace (``trace_borders``) calls the image model off the tool call's
thread. :class:`Traces` runs each in a pool, and lands its result on the
section under the write lock, only while the section still holds the running
record for the same geometry.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from langslice.core.jsonable import JsonObject
from langslice.core.refusal import Refused
from langslice.core.stack.stack import Stack
from langslice.core.stack.trace import BorderTrace
from langslice.job.state_file import relative_to, state_paths


def _reply(
    section_id: str,
    fingerprint: str,
    future: Future[dict[str, Any]],
    timeout: float | None,
) -> dict[str, Any]:
    """A trace call's result; an exception becomes an error result.

    Raises ``TimeoutError`` when *timeout* passes first.
    """
    error = future.exception(timeout=timeout)
    if error is None:
        return future.result()
    failure = Refused.from_exception(error).payload()
    return {"id": section_id, "geometry_fingerprint": fingerprint, **failure}


def _land(state: Stack, section_id: str, fingerprint: str, result: JsonObject) -> bool:
    """Record a finished trace on a section still waiting for it."""
    section = state.by_id(section_id)
    held = None if section is None else section.border_trace
    if (
        section is not None
        and held is not None
        and held.status == "running"
        and held.geometry_fingerprint == fingerprint
    ):
        section.border_trace = BorderTrace.from_json(result)
        return True
    return False


class Traces:
    """Run border traces in the background and land their results.

    Parameters
    ----------
    state
        The stack the job registers (changed in place, never replaced).
    folder
        The job folder, which the paths of a trace's record are relative to.
    writing
        Holds the job's write lock and brings the state up to date
        (``Job.writing``); a result lands inside it.
    checkpoint
        Writes the state (``Checkpoints.write``).
    """

    def __init__(
        self,
        state: Stack,
        folder: Path,
        *,
        writing: Callable[[], AbstractContextManager[object]],
        checkpoint: Callable[[], None],
    ) -> None:
        self._state = state
        self._folder = folder
        self._writing = writing
        self._checkpoint = checkpoint
        self._runs: dict[str, tuple[str, Future[dict[str, Any]]]] = {}
        self._executor: ThreadPoolExecutor | None = None

    def portable(self, trace: dict[str, Any]) -> dict[str, Any]:
        """A border trace's record with its paths relative to the job folder."""
        converted = state_paths({"slices": [{"border_trace": trace}]}, relative_to(self._folder))
        return converted["slices"][0]["border_trace"]

    def start(
        self,
        section_id: str,
        fingerprint: str,
        call: Callable[[], dict[str, Any]],
        *,
        workers: int,
    ) -> None:
        """Run one border trace's image-model call in the background."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="border-trace"
            )
        self._runs[section_id] = (fingerprint, self._executor.submit(call))

    def running(self, section_id: str, fingerprint: str) -> bool:
        """Whether the section's call at this geometry is still running."""
        run = self._runs.get(section_id)
        return run is not None and run[0] == fingerprint and not run[1].done()

    def pending(self) -> bool:
        """Whether any call is started and its result not yet recorded."""
        return bool(self._runs)

    def settle(self) -> bool:
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
        running = list(self._runs.items())
        results = [
            (
                section_id,
                fingerprint,
                self.portable(_reply(section_id, fingerprint, future, None)),
            )
            for section_id, (fingerprint, future) in running
        ]
        changed = False
        if results:
            with self._writing():
                for section_id, fingerprint, result in results:
                    changed |= _land(self._state, section_id, fingerprint, result)
                if changed:
                    self._checkpoint()
                for section_id, run in running:
                    if self._runs.get(section_id) is run:
                        del self._runs[section_id]
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        return changed

    def wait(self, section_id: str, timeout: float | None) -> bool:
        """Wait up to *timeout* seconds (None: as long as it takes) for one
        section's running trace call.

        Records its result the way :meth:`settle` does (under the write lock,
        checkpointed, no undo step) and returns True once nothing is running
        for the section; False when the call is still running at the timeout.
        The call is forgotten only once its result is recorded, so a waiter
        that gives way under the lock (background work) leaves it for the
        next.
        """
        run = self._runs.get(section_id)
        if run is None:
            return True
        fingerprint, future = run
        try:
            result = _reply(
                section_id, fingerprint, future, None if timeout is None else max(0.0, timeout)
            )
        except TimeoutError:
            return False
        with self._writing():
            if _land(self._state, section_id, fingerprint, self.portable(result)):
                self._checkpoint()
            if self._runs.get(section_id) is run:
                del self._runs[section_id]
        return True

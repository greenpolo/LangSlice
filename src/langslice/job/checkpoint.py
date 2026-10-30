"""The job's state, its files and the checkpoint that writes them.

:class:`JobCheckpoint` is the base of :class:`langslice.job.job.Job`: it
holds the stack, the spec and the job folder, writes the checkpoint (the
state, then ``registration.json`` and the landmark files as the lock's
outermost hold ends), tells the observers, and holds the section histories
and undo steps (:mod:`langslice.job.history`) that ``undo.py`` writes. Every
path the state stores (a deformation's ``record``, a border trace's
artifacts) is relative to the job folder.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langslice.core.fits import deformation
from langslice.core.jsonable import JsonObject
from langslice.core.stack.stack import Stack
from langslice.job import public_files as formats
from langslice.job.history import SCRIPT, SectionHistories
from langslice.job.host_inputs import host_damaged_ids
from langslice.job.layout import JobLayout
from langslice.job.lock import FolderLock
from langslice.job.pictures import ViewStore
from langslice.job.settings.spec import JobSpec
from langslice.job.state_file import (
    Upgrade,
    relative_to,
    stack_from_json,
    state_paths,
    write_checkpoint,
)
from langslice.job.undo_steps import UndoFile, UndoStep

if TYPE_CHECKING:
    from langslice.core.workspace import Workspace

logger = logging.getLogger(__name__)


#: A file's identity on disk: inode, size and modification time. An atomic
#: rewrite is a new inode, so even a same-size write within the clock's
#: resolution shows.
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
        The stack the file holds.

    Raises
    ------
    ValueError
        When the file holds unknown keys, naming the file and the keys.
    """
    try:
        return stack_from_json(data)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc



class JobCheckpoint:
    """Hold one job's state and files, and write its checkpoint.

    Parameters
    ----------
    state
        The stack the job registers.
    spec
        The job's settings form.
    layout
        The job folder and the paths in it.
    results_path
        Where the results JSON goes; None is the job folder's
        ``exports/linear_results.json``.
    seen
        The stamps of the state file and ``history/undo.json`` taken
        before they were read; None takes them now.
    """

    def __init__(
        self,
        state: Stack,
        spec: JobSpec,
        *,
        layout: JobLayout,
        results_path: str | None = None,
        seen: tuple[Stamp, Stamp] | None = None,
    ) -> None:
        self.state = state
        self.spec = spec
        self.layout = layout
        self.results_path = str(results_path or layout.results_file)
        #: Each section's history (``sections/<name>/history.jsonl``).
        self.histories = SectionHistories(layout)
        #: The undo and redo steps on disk (``history/undo.json``).
        self.undo_file = UndoFile(layout)
        #: The pictures the model was shown, saved with their layers.
        self.views: ViewStore = ViewStore(layout)
        #: ``registration.json`` and the landmark files are owed: the state
        #: was checkpointed in the lock hold that is still open (:meth:`checkpoint`).
        self._derived_due = False
        #: The job folder's write lock (:meth:`writing`), across processes.
        self.lock = FolderLock(layout.folder)
        #: Sections the host gave a damage note (``inputs.damaged``); the note
        #: stays first in the section's ``damage_note``.
        self.host_damaged = frozenset(host_damaged_ids(spec))
        #: The undo steps, oldest first; the last one is what ``undo`` takes.
        self.undo_stack: list[UndoStep] = []
        self.redo_stack: list[UndoStep] = []
        #: The number the job's next write gets (a line's ``step``).
        self.next_step = 1
        #: Undo steps recorded by this job so far (:meth:`merge_steps`).
        self.pushed: int = 0
        #: The most pictures one call of a picture verb shows, and the most sections
        #: one call of ``interactive_transform`` / ``elastix_affine`` names (the
        #: pictures spend the driving model's context). None: no cap, as at the
        #: agent CLI and the library; the agent tools and MCP set it
        #: (``build_tools``). The job's own ``transform.max_parallel`` is separate.
        self.picture_cap: int | None = None
        #: Called with the state after every checkpoint and every reload.
        self.observers: list[Callable[[Stack], None]] = []
        #: Image corrections still running, per section id: the geometry they
        #: were started at and their future. Nothing waits on them but
        #: ``settle_border_traces`` (submit, the end of a session) and a
        #: fit that reads a trace (``wait_image_job``).
        self.image_jobs: dict[str, tuple[str, Future[dict[str, Any]]]] = {}
        self.image_executor: ThreadPoolExecutor | None = None
        self._deformations: deformation.RecordStore | None = None
        #: False: nothing this job holds is written (the state, the history,
        #: the pictures); a dry run's job (the CLI's ``--dry-run``).
        self.persist = True
        self._workspace: Workspace | None = None
        #: How a state file and history line of an older format are read as
        #: the current format (the job's migrations, set by :meth:`open` and
        #: :meth:`load`; None refuses them).
        self.upgrade: Upgrade | None = None
        self._state_stamp, self._undo_stamp = seen or (
            file_stamp(self.checkpoint_path), file_stamp(self.undo_path))
        #: A read's view of a state another process saved while it held the
        #: lock (:meth:`reading`): the state this job held before it, put
        #: back by the next locked :meth:`sync`, and the file's stamp then.
        self._peeked: dict[str, Any] | None = None
        self._peek_stamp: Stamp = None
        #: The tool whose call this thread runs (:meth:`acting`).
        self._acting = threading.local()
        #: Each registration a write left, in order: (thread, section, number),
        #: for the doors to name in that call's reply (:meth:`registered_since`).
        self.registered: list[tuple[int, str, int]] = []
        self.wire_views()

    @property
    def lock(self) -> FolderLock:
        """The job folder's write lock; the files derived from the state are
        written as its outermost hold ends (:meth:`checkpoint`)."""
        return self._lock

    @lock.setter
    def lock(self, lock: FolderLock) -> None:
        self._lock = lock
        if self._write_derived not in lock.before_release:
            lock.before_release.append(self._write_derived)

    @property
    def workspace(self) -> Workspace | None:
        """The workspace the job's sections and atlas are read through (set by
        :meth:`open`, :meth:`load` and the tool door): every checkpoint rewrites
        ``registration.json`` through it
        (:func:`langslice.job.public_files.write_registration`), and its made
        section images are read from this job's folder."""
        return self._workspace

    @workspace.setter
    def workspace(self, workspace: Workspace | None) -> None:
        self._workspace = workspace
        if workspace is not None and getattr(workspace, "job_folder", None) is None:
            workspace.job_folder = str(self.layout.folder)

    def wire_views(self) -> None:
        """Number each saved picture's ``step`` with the undo history's depth
        (call again after replacing :attr:`views`)."""
        self.views.step_source = lambda: len(self.undo_stack)
        self.views.registration_source = self.registration_numbers

    @property
    def folder(self) -> Path:
        """The job folder."""
        return self.layout.folder

    @property
    def checkpoint_path(self) -> str:
        """The state checkpoint, ``<job folder>/state.json``."""
        return str(self.layout.state_file)

    @property
    def undo_path(self) -> str:
        """The undo and redo steps, ``history/undo.json``."""
        return str(self.undo_file.path)

    @property
    def engine_log(self) -> str | None:
        """Where the registration engines' own output goes while they fit
        (``logs/engine.log``, :mod:`langslice.core.fits.engine_log`); None for
        a job that persists nothing."""
        return str(self.layout.logs_dir / "engine.log") if self.persist else None

    def portable(self, correction: dict[str, Any]) -> dict[str, Any]:
        """A border trace's record with its paths relative to the job folder."""
        converted = state_paths({"slices": [{"border_trace": correction}]},
                                relative_to(self.folder))
        return converted["slices"][0]["border_trace"]

    def checkpoint(self) -> None:
        """Write the state (atomically, versioned), then tell every observer.

        Its public rendering, ``registration.json``
        (:func:`langslice.job.public_files.write_registration`; a failure there
        is logged, never raised), and the landmark files follow once, as the
        outermost hold of the job folder's lock ends: a write that checkpoints
        several times in one hold (the write, then the deformations it made
        stale) maps the stack once, from the state it leaves, still under the
        lock."""
        from langslice.core.fits.landmarks import stale

        for section in self.state.slices:
            if section.landmarks is not None:
                pairs = tuple(
                    replace(pair, stale=stale(self.state, section, pair))
                    for pair in section.landmarks.pairs)
                section.landmarks = replace(section.landmarks, pairs=pairs)
        if not self.persist:
            return
        with self.lock.held():
            write_checkpoint(self.state, self.checkpoint_path)
            self._state_stamp = file_stamp(self.checkpoint_path)
            self._derived_due = True
        self._notify()

    def _write_derived(self) -> None:
        """``registration.json`` and the landmark files of the state as it
        stands, when a checkpoint owes them (called by the lock, held)."""
        if not self._derived_due or not self.persist:
            return
        self._derived_due = False
        formats.write_registration(self.layout, self.state, self.workspace)
        formats.write_landmarks(self.layout, self.state)

    @contextlib.contextmanager
    def acting(self, tool: str) -> Iterator[None]:
        """This thread runs *tool*'s call for the block: the history lines its
        writes append are made by it."""
        outer = getattr(self._acting, "tool", None)
        self._acting.tool = tool
        try:
            yield
        finally:
            self._acting.tool = outer

    @property
    def actor(self) -> str:
        """The tool this thread runs (:meth:`acting`); ``script`` for none."""
        return getattr(self._acting, "tool", None) or SCRIPT

    def registration_numbers(self, ids: Sequence[str]) -> dict[str, int]:
        """The number of the registration each section of *ids* holds now,
        for the sections that hold a numbered one (a picture records them)."""
        out: dict[str, int] = {}
        for section_id in ids:
            record = self.state.by_id(section_id)
            number = (self.histories.registration_number(self.state, record)
                      if record is not None else None)
            if number is not None:
                out[section_id] = number
        return out

    def registered_since(self, mark: int) -> dict[str, int]:
        """The registration each section holds after this thread's writes
        since ``len(job.registered)`` was *mark*, by section."""
        me = threading.get_ident()
        return {section: number for thread, section, number in self.registered[mark:]
                if thread == me}

    def _notify(self) -> None:
        for observer in list(self.observers):
            try:
                observer(self.state)
            except Exception:
                logger.exception("Job observer raised; ignoring")

    @contextlib.contextmanager
    def observe(self, fn: Callable[[Stack], None]) -> Iterator[None]:
        """Call *fn* with the state after every checkpoint and reload, while open."""
        self.observers.append(fn)
        try:
            yield
        finally:
            self.observers.remove(fn)

    @property
    def deformations(self) -> deformation.RecordStore:
        """Fitted records, saved in each section's folder
        (``sections/<name>/deformable/<key>``; made on first use)."""
        if self._deformations is None:
            self._deformations = deformation.RecordStore(
                root=self.layout.sections_dir, folder_of=self.layout.deformable_dir)
        return self._deformations


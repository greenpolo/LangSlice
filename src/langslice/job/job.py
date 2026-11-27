"""The job: one stack's state and everything that keeps it consistent.

A :class:`Job` owns the :class:`~langslice.core.stack.stack.Stack` and the
:class:`~langslice.job.settings.spec.JobSpec`, its job folder
(:class:`langslice.job.folder.layout.JobLayout`: ``<images>/langslice/``), the
sections' frozen properties and the host's damage notes, undo/redo (persisted
in the job folder), the checkpoint and its observers (:meth:`Checkpoints.observe`), the
submit gates, the stale-deformation rule, the deformation record store, the
store of the pictures the model was shown
(:class:`langslice.job.pictures.ViewStore`), the transform-count cap, the
background border traces and the background work built on them
(:class:`langslice.job.background.BackgroundWork`). ``ingest``,
``apply_host_inputs`` and ``emit_results`` open and close it. It imports the
core and the job folder package (``langslice.job``) only: no agent framework,
no message types, no provider. The doors (the ADK toolbox, the MCP server) sit
on it; a script can drive it directly.

What a Job never holds: the model-delivery bookkeeping (which pictures a
model has received). That is the tool doors'
(:class:`langslice.tools.pipeline.ToolBox`).

A :class:`Job` holds its collaborators as named attributes, each a plain class
with its own state: ``checkpoints`` (:class:`~langslice.job.checkpoint.Checkpoints`:
the state file, the derived files, the observers), ``undo``
(:class:`~langslice.job.steps.undo.UndoStore`: the steps and the history lines
they name), ``histories`` (the sections' history files), ``traces``
(:class:`~langslice.job.border_traces.Traces`: the background border traces),
``background`` (:class:`~langslice.job.background.BackgroundWork`), ``views``
and ``deformations``. The Job itself holds the state, the spec, the lock and the
write protocol (``writing``, ``reading``, ``sync``, ``snapshot``, ``commit``,
``amend``, ``acting``), opening and the rules (the frozen and busy refusals, the
stale-deformation rule) and the results; ``host_inputs.py`` applies a host's
inputs to a new stack.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from collections.abc import Callable, Collection, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langslice.core.fits import deformation
from langslice.core.jsonable import JsonObject
from langslice.core.refusal import RefusalCode, Refused
from langslice.core.sections import images as section_images
from langslice.core.sections.discovery import discover_slices
from langslice.core.space import as_plane
from langslice.core.stack.section import Section
from langslice.core.stack.stack import Stack, initial_positions
from langslice.job.background import BackgroundWork, in_worker
from langslice.job.border_traces import Traces
from langslice.job.checkpoint import Checkpoints, Stamp, file_stamp, state_from_file
from langslice.job.folder import recent
from langslice.job.folder.layout import (
    JobLayout,
    check_owner,
    locate_job_folder,
    write_job_file,
)
from langslice.job.folder.lock import FolderLock
from langslice.job.host_inputs import (
    apply_host_inputs,
    changed_inputs,
    check_damaged_regions,
    host_damaged_ids,
    inputs_changed,
)
from langslice.job.pictures import DiscardedViews, ViewStore
from langslice.job.settings import frozen as freeze
from langslice.job.settings.spec import MAX_PARALLEL_TRANSFORMS, JobSpec
from langslice.job.state_file import (
    read_checkpoint,
    stack_from_json,
    stack_to_json,
    write_json_atomic,
)
from langslice.job.steps.history import SCRIPT, SectionHistories
from langslice.job.steps.undo import UndoStore
from langslice.job.steps.undo_steps import UndoFile, UndoLists

if TYPE_CHECKING:
    from langslice.core.workspace import Workspace

logger = logging.getLogger(__name__)


def ingest(spec: JobSpec, workspace: Workspace) -> Stack:
    """Discover the folder and build the stack state. Plain code, no model.

    Every section starts at its initial position, as ABBA places a new
    stack: ``position.start_mm`` plus its index times the spacing
    (``position.interval_um``), clamped to the atlas
    (:func:`~langslice.core.stack.stack.initial_positions`).
    """
    paths = discover_slices(workspace.image_folder)
    if not paths:
        raise ValueError(f"No slice images found in {workspace.image_folder}")

    pos_lo, pos_hi = workspace.position_range
    places = initial_positions(len(paths), (pos_lo, pos_hi), spec.start_mm, spec.interval_mm)
    state = Stack(
        image_folder=workspace.image_folder,
        atlas=spec.atlas,
        plane=as_plane(spec.plane),
        interval_mm=spec.interval_mm,
        thickness_mm=spec.thickness_mm,
        spec=spec.to_dict(),
        slices=[
            Section(
                id=os.path.basename(path),
                index_original=index,
                index_corrected=index,
                position_mm=round(places[index], 4),
            )
            for index, path in enumerate(paths)
        ],
    )
    state.notes.append(
        f"ingest: {len(paths)} sections, atlas {spec.atlas} ({spec.plane}) "
        f"spans {pos_lo:.2f}-{pos_hi:.2f} mm; initial positions "
        f"{places[0]:.2f} to {places[-1]:.2f} mm"
    )
    workspace.progress(f"[ingest] {len(paths)} sections from {workspace.image_folder}")
    return state


def _log_migrated(layout: JobLayout, upgraded: bool) -> None:
    """Log a ``migrated`` event when the job's files were read from an older format."""
    if upgraded:
        layout.log_event("migrated", migrations=["state_5_to_6"])


def emit_results(
    state: Stack,
    results_path: str,
    progress: Callable[[str], None] | None = None,
) -> Stack:
    """Write the results JSON — the same shape as the checkpoint, unversioned
    (its paths relative to the job folder, as the checkpoint's)."""
    write_json_atomic(results_path, stack_to_json(state))
    if progress is not None:
        progress(f"[emit] results -> {results_path}")
    return state


class Job:
    """One stack being registered: its state, its rules and its files.

    Build one with :meth:`open` (resume or ingest, then the first
    checkpoint), or around a state already in hand
    (``Job(state, spec, layout=...)``: no undo steps, nothing read or
    written until the first operation; the files as they stand on disk at
    that moment count as already seen, unless *seen* gives the stamps of
    the state file and ``history/undo.json`` taken before they were read).

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
    lock
        The job folder's write lock; None makes one.
    persist
        False: the job writes nothing at all.
    views
        The store of the pictures the model was shown; None makes one.
    workspace
        The workspace the job's sections and atlas are read through.
    undo_file
        ``history/undo.json`` as the opening read it; None makes one.
    undo_lists
        The undo steps that file held; None starts without any.
    """

    def __init__(
        self,
        state: Stack,
        spec: JobSpec,
        *,
        layout: JobLayout,
        results_path: str | None = None,
        seen: tuple[Stamp, Stamp] | None = None,
        lock: FolderLock | None = None,
        persist: bool = True,
        views: ViewStore | None = None,
        workspace: Workspace | None = None,
        undo_file: UndoFile | None = None,
        undo_lists: UndoLists | None = None,
    ) -> None:
        state_seen, undo_seen = seen if seen is not None else (None, None)
        self.state = state
        self.spec = spec
        self.layout = layout
        self.results_path = str(results_path or layout.results_file)
        self.persist = persist
        self.lock = lock if lock is not None else FolderLock(layout.folder)
        self.host_damaged = frozenset(host_damaged_ids(spec))
        self.picture_cap: int | None = None
        self.histories = SectionHistories(layout)
        self.checkpoints = Checkpoints(state, layout, self.lock, persist=persist, seen=state_seen)
        self.undo = UndoStore(
            state,
            self.histories,
            self.lock,
            undo_file if undo_file is not None else UndoFile(layout),
            persist=persist,
            actor=lambda: self.actor,
            writing=self.writing,
            checkpoint=self.checkpoints.write,
            lists=undo_lists,
            seen=undo_seen,
        )
        self.traces = Traces(
            state,
            layout.folder,
            writing=self.writing,
            checkpoint=self.checkpoints.write,
        )
        self.views: ViewStore = views if views is not None else ViewStore(layout)
        self.views.step_source = lambda: len(self.undo.steps)
        self.views.registration_source = self._registration_numbers
        self.background = BackgroundWork(
            self.lock,
            self.traces,
            self.undo,
            self.views,
            acting=self.acting,
            workspace=lambda: self.workspace,
        )
        self._deformations: deformation.RecordStore | None = None
        self._acting = threading.local()
        self._peeked: dict[str, Any] | None = None
        self._peek_stamp: Stamp = None
        if workspace is not None:
            self.workspace = workspace

    @classmethod
    def open(
        cls,
        spec: JobSpec,
        workspace: Workspace,
        *,
        folder: str | os.PathLike[str] | None = None,
        results_path: str | None = None,
    ) -> Job:
        """Open the job folder (*folder*; default
        :func:`~langslice.job.folder.layout.locate_job_folder`:
        ``spec.job_dir``, else
        ``<images>/langslice``, else, when that cannot be written,
        ``~/.langslice/jobs/<id>/``).

        A folder holding the job of another image folder is refused
        (``ValueError``). ``job.json`` gets this spec, then the checkpoint
        is resumed (``spec.resume``) or the folder ingested, and the first
        checkpoint written; the open is a line of ``logs/events.jsonl`` and
        of the recent jobs list (:mod:`langslice.job.folder.recent`). A resumed
        job
        keeps its undo steps; a fresh one starts without any (the steps an
        earlier job here left are emptied) and writes each section's opening
        history line (:mod:`langslice.job.steps.history`). A job folder or
        checkpoint
        of another format (a newer LangSlice's, or an older pre-release's) is
        refused (``ValueError``). A resume whose
        spec supplies other ``inputs`` than the checkpoint was made from is
        refused (:class:`InputsChanged`, naming the keys and how to start
        fresh) before anything is written: the old checkpoint would keep the
        old inputs and drop the new ones.
        """
        if folder is None:
            folder = locate_job_folder(
                workspace.image_folder, spec.job_dir, emit=workspace.progress
            )
        images = Path(os.path.abspath(workspace.image_folder))
        layout = JobLayout(Path(os.path.abspath(folder)), images)
        check_owner(layout)
        layout.ensure()
        lock = FolderLock(layout.folder)
        if workspace.job_folder is None:
            workspace.job_folder = str(layout.folder)
        with lock.held():
            state = None
            resumed = False
            upgraded = False
            if spec.resume:
                read = read_checkpoint(str(layout.state_file))
                if read is not None:
                    upgraded = read.upgraded
                state = None if read is None else state_from_file(read.fields, layout.state_file)
                changed = [] if state is None else changed_inputs(state.spec, spec)
                if changed:
                    raise inputs_changed(layout.folder, changed)
            write_job_file(layout, spec=spec.to_dict())
            undo_file = UndoFile(layout)
            lists = undo_file.load()
            if state is not None:
                resumed = True
                workspace.progress(f"[ingest] resuming from {layout.state_file}")
                state.spec = spec.to_dict()
                state.image_folder = str(images)
                state.submitted = False
            else:
                state = ingest(spec, workspace)
                apply_host_inputs(state, spec)
                check_damaged_regions(state, spec, workspace)
                lists.undo, lists.redo = [], []
            if lists.notice is not None:
                workspace.progress(f"[job] {lists.notice}")
            section_images.ensure_opening(workspace, layout, state)
            job = cls(
                state,
                spec,
                layout=layout,
                results_path=results_path,
                lock=lock,
                workspace=workspace,
                undo_file=undo_file,
                undo_lists=lists,
            )
            if not resumed:
                job.undo.record_opening()
                if undo_file.path.exists():
                    job.undo.save()
            job.checkpoints.write()
        _log_migrated(layout, upgraded)
        layout.log_event("open", resumed=resumed)
        recent.remember(layout.folder, images, spec.atlas)
        return job

    @classmethod
    def load(
        cls,
        spec: JobSpec,
        workspace: Workspace,
        *,
        folder: str | os.PathLike[str],
        results_path: str | None = None,
        persist: bool = True,
    ) -> Job:
        """Open an existing job folder as it stands, writing nothing.

        The state is the folder's checkpoint as saved (``FileNotFoundError``
        without one; a newer one is refused, ``ValueError``), with its undo
        steps; ``job.json`` and the state file are not rewritten, so a job
        loaded beside a running agent changes nothing until its first write
        (the CLI and the library open a job per call this way). A folder
        holding another image folder's job is refused (``ValueError``).
        *persist* False: the job writes nothing at all (:attr:`persist`).
        """
        images = Path(os.path.abspath(workspace.image_folder))
        layout = JobLayout(Path(os.path.abspath(folder)), images)
        check_owner(layout)
        if workspace.job_folder is None:
            workspace.job_folder = str(layout.folder)
        undo_file = UndoFile(layout)
        seen = (
            file_stamp(str(layout.state_file)),
            file_stamp(str(undo_file.path)),
        )
        read = read_checkpoint(str(layout.state_file))
        if read is None:
            raise FileNotFoundError(f"No job state in {layout.folder}")
        state = state_from_file(read.fields, layout.state_file)
        job = cls(
            state,
            spec,
            layout=layout,
            results_path=results_path,
            seen=seen,
            persist=persist,
            views=None if persist else DiscardedViews(layout),
            workspace=workspace,
            undo_file=undo_file,
        )
        job.undo.load()
        if section_images.ensure_opening(workspace, layout, job.state) or read.upgraded:
            if persist and os.access(layout.folder, os.W_OK):
                job._write_upgrade_if_writable()
        return job

    def _write_upgrade_if_writable(self) -> None:
        """:meth:`_write_upgrade`, kept in memory where the folder refuses it.

        Windows' ``os.access`` reads neither a folder's permissions nor a
        file's read-only flag, so the refusal comes from the write itself.
        """
        try:
            self._write_upgrade()
        except PermissionError as exc:
            logger.warning(
                "%s cannot be written (%s); the job is upgraded in memory",
                self.layout.folder,
                exc.strerror,
            )

    def _write_upgrade(self) -> None:
        """Write the state at the current format, read again under
        the job's lock first: another process may have migrated (and written
        to) the job since it was read, and its state is the one kept. Nothing
        is written when that read needs no upgrade and every section has its
        opening image."""
        workspace = self.workspace
        assert workspace is not None
        with self.lock.held():
            read = read_checkpoint(self.checkpoints.path)
            if read is not None:
                self.state.replace_with(state_from_file(read.fields, self.checkpoints.path))
            self.undo.load()
            made = section_images.ensure_opening(workspace, self.layout, self.state)
            if read is None or read.upgraded or made:
                self.checkpoints.write()
            else:
                self.checkpoints.stamp = file_stamp(self.checkpoints.path)
                self.undo.mark_seen()
        _log_migrated(self.layout, read is not None and read.upgraded)

    @property
    def workspace(self) -> Workspace | None:
        """The workspace the job's sections and atlas are read through (set by
        :meth:`open`, :meth:`load` and the tool door): every checkpoint rewrites
        ``registration.json`` through it
        (:func:`langslice.job.exports.public_files.write_registration`), and
        its made
        section images are read from this job's folder."""
        return self.checkpoints.workspace

    @workspace.setter
    def workspace(self, workspace: Workspace | None) -> None:
        self.checkpoints.workspace = workspace
        if workspace is not None and workspace.job_folder is None:
            workspace.job_folder = str(self.layout.folder)

    @property
    def folder(self) -> Path:
        """The job folder."""
        return self.layout.folder

    @property
    def engine_log(self) -> str | None:
        """Where the registration engines' own output goes while they fit
        (``logs/engine.log``, :mod:`langslice.core.fits.engine_log`); None for
        a job that persists nothing."""
        return str(self.layout.logs_dir / "engine.log") if self.persist else None

    @property
    def deformations(self) -> deformation.RecordStore:
        """Fitted records, saved in each section's folder
        (``sections/<name>/deformable/<key>``; made on first use)."""
        if self._deformations is None:
            self._deformations = deformation.RecordStore(
                root=self.layout.sections_dir, folder_of=self.layout.deformable_dir
            )
        return self._deformations

    def _registration_numbers(self, ids: Sequence[str]) -> dict[str, int]:
        """The number of the registration each section of *ids* holds now,
        for the sections that hold a numbered one (a picture records them)."""
        out: dict[str, int] = {}
        for section_id in ids:
            record = self.state.by_id(section_id)
            number = (
                self.histories.registration_number(self.state, record)
                if record is not None
                else None
            )
            if number is not None:
                out[section_id] = number
        return out

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
        abort = self.background.gives_way if in_worker() else None
        with self.lock.held(abort):
            yield self.sync()

    @contextlib.contextmanager
    def reading(self) -> Iterator[None]:
        """For a read: the state brought up to date without waiting for
        another writer, then read beside this process's other reads (looks
        and zooms of one turn run side by side); only a thread that writes
        is kept out. When no other thread holds the job, it is synced first:
        when another process holds the job folder's lock (a long fit
        applying, a plugin tool that holds it while it computes) the state is
        read as that process last saved it, and the next locked :meth:`sync`
        puts the job's own back first. When another thread of this process
        reads or writes, the state held in memory is already its latest."""
        with self.lock.alone_if_free() as alone:
            if alone:
                with self.lock.held_if_free() as free:
                    if free:
                        self.sync()
                if not free:
                    self._peek()
        with self.lock.shared():
            yield

    def _peek(self) -> None:
        """Read the state file as it stands, without the lock (:meth:`reading`)."""
        stamp = file_stamp(self.checkpoints.path)
        if stamp in (self.checkpoints.stamp, self._peek_stamp):
            return
        try:
            read = read_checkpoint(self.checkpoints.path)
        except (OSError, ValueError):
            return
        if read is None:
            return
        held = stack_to_json(self.state)
        loaded = stack_to_json(state_from_file(read.fields, self.checkpoints.path))
        self._peek_stamp = stamp
        if loaded == held:
            return
        if self._peeked is None:
            self._peeked = held
        self.state.replace_with(stack_from_json(loaded))
        self.checkpoints.notify()

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
        if self._peeked is not None:
            self.state.replace_with(stack_from_json(self._peeked))
            self._peeked, self._peek_stamp = None, None
        state_stamp = file_stamp(self.checkpoints.path)
        history_changed = self.undo.reload_if_changed()
        if state_stamp == self.checkpoints.stamp:
            return None
        try:
            read = read_checkpoint(self.checkpoints.path)
        except (OSError, ValueError):
            logger.warning(
                "State file %s changed but does not read; keeping the job's state",
                self.checkpoints.path,
                exc_info=True,
            )
            return None
        self.checkpoints.stamp = state_stamp
        if read is None:
            return None
        held = stack_to_json(self.state)
        loaded = stack_to_json(state_from_file(read.fields, self.checkpoints.path))
        if loaded == held:
            return None
        self.state.replace_with(stack_from_json(loaded))
        if not history_changed:
            with self.lock.held():
                self.undo.push(self.undo.record(held, SCRIPT))
                self.undo.save()
        logger.info("Reloaded %s: it changed on disk", self.checkpoints.path)
        self.checkpoints.notify()
        return held

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
            self.undo.push(self.undo.record(before, self.actor))
            self.undo.save()
            self.checkpoints.write()

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
            if self.undo.amend(before, self.actor):
                self.undo.save()
            self.checkpoints.write()

    @property
    def transform_cap(self) -> int | None:
        """Below the maximum, the most sections one transform call may take."""
        cap = self.spec.transform.max_parallel
        return cap if cap < MAX_PARALLEL_TRANSFORMS else None

    def frozen_refusal(
        self,
        section_ids: Collection[str],
        *properties: str,
    ) -> Refused | None:
        """``FROZEN`` (``ids``, ``properties``, ``message``) when a write
        to any of *properties* of any of *section_ids* would change what the
        job froze there (the sections and the properties it names are the
        frozen ones), else None."""
        hit: dict[str, list[str]] = {}
        for section_id in section_ids:
            held = [name for name in properties if name in self.spec.frozen_for(section_id)]
            if held:
                hit[section_id] = held
        if not hit:
            return None
        named = [name for name in freeze.PROPERTIES if any(name in held for held in hit.values())]
        message = freeze.message(named, list(hit), len(self.state.slices))
        return Refused(
            RefusalCode.FROZEN,
            ids=list(hit),
            properties=named,
            message=message,
        )

    def busy_refusal(self, section_ids: Collection[str]) -> Refused | None:
        """:class:`Refused` ``BUSY`` (``ids``, ``work``, ``message``) when background
        work is running on any of *section_ids* (it will write them when it
        lands, from what they hold now), else None. The landing of that work
        itself is not refused."""
        busy = self.background.busy(section_ids)
        if not busy:
            return None
        works = list({work.id: work for work in busy.values()}.values())
        running = "; ".join(
            f"{work.kind} is running on {', '.join(work.sections)} (work {work.id})"
            for work in works
        )
        return Refused(
            RefusalCode.BUSY,
            ids=list(busy),
            work=[work.id for work in works],
            message=f"{running}. Nothing written: write to it once the work "
            "lands (its notice comes with a later reply).",
        )

    def clear_stale_deformations(self) -> list[str]:
        """Drop deformations whose linear placement changed; checkpoint; their ids.

        A deformation is fitted on top of one linear placement; a write that
        moves the position, orientation, cutting angles or transform makes it
        stale. Called after every write, it lands in that write's undo step
        (the step holds the state from before the write).
        """
        with self.writing():
            before = self.snapshot()
            cleared = deformation.clear_stale(self.state)
            if cleared:
                self.amend(before)
        return cleared

    def emit_results(self, progress: Callable[[str], None] | None = None) -> Stack:
        """Write the results JSON (:func:`emit_results`); every picture saved."""
        self.views.flush()
        return emit_results(self.state, self.results_path, progress)

    def close(self) -> None:
        """Finish the job's background work (:attr:`background`), then its
        background writes (the pictures); the job stays usable."""
        self.background.close()
        self.views.flush()

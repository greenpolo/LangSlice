"""The job: one stack's state and everything that keeps it consistent.

A :class:`Job` owns the :class:`~langslice.core.stack.stack.Stack` and the
:class:`~langslice.job.settings.spec.JobSpec`, its job folder
(:class:`langslice.job.layout.JobLayout`: ``<images>/langslice/``), the
sections' frozen properties and the host's damage notes, undo/redo (persisted
in the job folder), the checkpoint and its observers (:meth:`Job.observe`), the
submit gates, the stale-deformation rule, the deformation record store, the
store of the pictures the model was shown
(:class:`langslice.job.pictures.ViewStore`), the transform-count cap, the
background image-correction jobs and the background work built on them
(:class:`langslice.job.background.BackgroundWork`). ``ingest``,
``apply_host_inputs`` and ``emit_results`` open and close it. It imports the
core and the job folder package (``langslice.job``) only: no agent framework,
no message types, no provider. The doors (the ADK toolbox, the MCP server) sit
on it; a script can drive it directly.

What a Job never holds: the model-delivery bookkeeping (which pictures a
model has received). That is the tool doors'
(:class:`langslice.tools.pipeline.ToolBox`).

The class is assembled from one class per seam, each built on the one
below: ``checkpoint.py`` (the state, its files, the checkpoint and its
observers, the registrations each step leaves), ``undo.py`` (writes under
the lock, undo steps, the live sync with the files) and
``border_traces.py`` (the background border traces). This module adds the
opening, the rules (the submit gates, the frozen and busy refusals, the
stale-deformation rule) and the results; ``host_inputs.py`` applies a
host's inputs to a new stack.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Collection, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

from langslice.core.fits import deformation
from langslice.core.jsonable import JsonObject
from langslice.core.refusal import RefusalCode, Refused
from langslice.core.sections import images as section_images
from langslice.core.sections.discovery import discover_slices
from langslice.core.stack.section import Section
from langslice.core.stack.stack import Stack
from langslice.job import recent
from langslice.job.border_traces import JobBorderTraces
from langslice.job.checkpoint import file_stamp, state_from_file
from langslice.job.gates import submit_errors
from langslice.job.host_inputs import (
    apply_host_inputs,
    changed_inputs,
    check_damaged_regions,
    inputs_changed,
    opening_spec,
)
from langslice.job.layout import (
    JobLayout,
    check_owner,
    locate_job_folder,
    read_job_file,
    write_job_file,
)
from langslice.job.lock import FolderLock
from langslice.job.migrations import (
    FolderMigration,
    warn_removed_settings,
    without_removed_settings,
)
from langslice.job.pictures import DiscardedViews
from langslice.job.settings import frozen as freeze
from langslice.job.settings.spec import MAX_PARALLEL_TRANSFORMS, JobSpec
from langslice.job.state_file import (
    read_checkpoint,
    stack_to_json,
    write_json_atomic,
)
from langslice.job.undo_steps import UndoFile

if TYPE_CHECKING:
    from langslice.core.workspace import Workspace

logger = logging.getLogger(__name__)


def ingest(spec: JobSpec, workspace: Workspace) -> Stack:
    """Discover the folder and build the stack state. Plain code, no model."""
    paths = discover_slices(workspace.image_folder)
    if not paths:
        raise ValueError(f"No slice images found in {workspace.image_folder}")

    pos_lo, pos_hi = workspace.position_range
    state = Stack(
        image_folder=workspace.image_folder,
        atlas=spec.atlas,
        plane=spec.plane,
        interval_mm=spec.interval_mm,
        thickness_mm=spec.thickness_mm,
        spec=spec.to_dict(),
        slices=[
            Section(
                id=os.path.basename(path),
                index_original=index,
                index_corrected=index,
            )
            for index, path in enumerate(paths)
        ],
    )
    state.notes.append(
        f"ingest: {len(paths)} sections, atlas {spec.atlas} ({spec.plane}) "
        f"spans {pos_lo:.2f}-{pos_hi:.2f} mm"
    )
    workspace.progress(
        f"[ingest] {len(paths)} sections from {workspace.image_folder}"
    )
    return state


def _removed_settings(layout: JobLayout) -> tuple[str, ...]:
    """Return the removed settings ``job.json`` holds, each logged."""
    saved = _saved_settings(layout)
    if saved is None:
        return ()
    _kept, removed = without_removed_settings(saved)
    warn_removed_settings(removed)
    return removed


def _write_settings_without_removed(layout: JobLayout) -> tuple[str, ...]:
    """Write ``job.json`` without its removed settings; return their names."""
    saved = _saved_settings(layout)
    if saved is None:
        return ()
    kept, removed = without_removed_settings(saved)
    if removed:
        write_job_file(layout, spec=kept)
    return removed


def _saved_settings(layout: JobLayout) -> JsonObject | None:
    """Return the settings ``job.json`` holds; None without any."""
    record = read_job_file(layout)
    saved = None if record is None else record.get("spec")
    if not isinstance(saved, dict):
        return None
    return cast("JsonObject", saved)


def _log_migrated(
    layout: JobLayout,
    migration: FolderMigration,
    removed: Sequence[str],
) -> None:
    """Log a ``migrated`` event when a migration changed the job's files."""
    if migration.applied or migration.dropped or removed:
        layout.log_event(
            "migrated",
            migrations=list(migration.applied),
            dropped=dict(migration.dropped),
            settings=list(removed),
        )


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


class Job(JobBorderTraces):
    """One stack being registered: its state, its rules and its files.

    Build one with :meth:`open` (resume or ingest, then the first
    checkpoint), or around a state already in hand
    (``Job(state, spec, layout=...)``: no undo steps, nothing read or
    written until the first operation; the files as they stand on disk at
    that moment count as already seen, unless *seen* gives the stamps of
    the state file and ``history/undo.json`` taken before they were read).
    """

    @classmethod
    def open(
        cls,
        spec: JobSpec,
        workspace: Workspace,
        *,
        folder: str | os.PathLike[str] | None = None,
        results_path: str | None = None,
        frozen_given: bool = False,
    ) -> Job:
        """Open the job folder (*folder*; default
        :func:`~langslice.job.layout.locate_job_folder`: ``spec.job_dir``, else
        ``<images>/langslice``, else, when that cannot be written,
        ``~/.langslice/jobs/<id>/``).

        A folder holding the job of another image folder is refused
        (``ValueError``). ``job.json`` gets this spec, then the checkpoint
        is resumed (``spec.resume``) or the folder ingested, and the first
        checkpoint written; the open is a line of ``logs/events.jsonl`` and
        of the recent jobs list (:mod:`langslice.job.recent`). A resumed job
        keeps its undo steps; a fresh one starts without any (the steps an
        earlier job here left are emptied) and writes each section's opening
        history line (:mod:`langslice.job.history`). A job folder or checkpoint
        of another format (a newer LangSlice's, or an older pre-release's) is
        refused (``ValueError``). A resume whose
        spec supplies other ``inputs`` than the checkpoint was made from is
        refused (:class:`InputsChanged`, naming the keys and how to start
        fresh) before anything is written: the old checkpoint would keep the
        old inputs and drop the new ones.

        A job the ABBA connector prepared opens with its saved settings as
        the defaults of *spec* (``opening_spec``; *frozen_given*:
        *spec*'s freeze was given explicitly and replaces the saved one), and
        *workspace* is given the settings it opens with.
        """
        if folder is None:
            folder, _fallback = locate_job_folder(
                workspace.image_folder, spec.job_dir, emit=workspace.progress
            )
        images = Path(os.path.abspath(workspace.image_folder))
        layout = JobLayout(Path(os.path.abspath(folder)), images)
        check_owner(layout)
        layout.ensure()
        lock = FolderLock(layout.folder)
        if getattr(workspace, "job_folder", None) is None:
            workspace.job_folder = str(layout.folder)
        migration = FolderMigration(workspace, layout)
        with lock.held():  # read and first checkpoint as one write
            opened = opening_spec(
                layout.folder, spec, frozen_given=frozen_given
            )
            if opened is not spec:
                spec = opened
                workspace.settings = spec.image_settings()
            state = None
            resumed = False
            if spec.resume:
                data = read_checkpoint(str(layout.state_file), migration)
                if data is not None:
                    migration.seed(data)
                state = (
                    None
                    if data is None
                    else state_from_file(data, layout.state_file)
                )
                changed = (
                    [] if state is None else changed_inputs(state.spec, spec)
                )
                if changed:
                    raise inputs_changed(layout.folder, changed)
            removed = _removed_settings(layout)  # this spec replaces them
            write_job_file(layout, spec=spec.to_dict())
            undo_file = UndoFile(layout)
            lists = undo_file.load()
            if state is not None:
                resumed = True
                workspace.progress(
                    f"[ingest] resuming from {layout.state_file}"
                )
                state.spec = spec.to_dict()
                state.image_folder = str(
                    images
                )  # where they are now (a moved folder)
                state.submitted = False
            else:
                state = ingest(spec, workspace)
                apply_host_inputs(state, spec)
                check_damaged_regions(state, spec, workspace)
                lists.undo, lists.redo = [], []
            if lists.notice is not None:
                workspace.progress(f"[job] {lists.notice}")
            # The opening ``preprocessed`` image of every section without one,
            # outside undo: part of the job as it opens.
            section_images.ensure_opening(workspace, layout, state)
            job = cls(
                state,
                spec,
                layout=layout,
                results_path=results_path,
            )
            job.lock = lock
            job.workspace = workspace
            job.upgrade = migration
            job.undo_file = undo_file
            job.undo_stack, job.redo_stack = lists.undo, lists.redo
            job.next_step = lists.next_step
            if not resumed:
                job.record_opening()
                if undo_file.path.exists():
                    job._save_undo()  # a fresh job empties the steps it read, never another's
            job.checkpoint()
        _log_migrated(layout, migration, removed)
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
        if getattr(workspace, "job_folder", None) is None:
            workspace.job_folder = str(layout.folder)
        migration = FolderMigration(workspace, layout)
        undo_file = UndoFile(layout)
        # stamped before the unlocked reads: a write landing during them is
        # reloaded by the first sync, not overwritten by the first commit
        seen = (
            file_stamp(str(layout.state_file)),
            file_stamp(str(undo_file.path)),
        )
        data = read_checkpoint(str(layout.state_file), migration)
        if data is None:
            raise FileNotFoundError(f"No job state in {layout.folder}")
        removed = _removed_settings(layout)
        migration.seed(data)
        state = state_from_file(data, layout.state_file)
        job = cls(
            state,
            spec,
            layout=layout,
            results_path=results_path,
            seen=seen,
        )
        job.persist = persist
        job.workspace = workspace
        job.upgrade = migration
        job.undo_file = undo_file
        job.load_undo()
        if not persist:
            job.views = DiscardedViews(layout)
            job.wire_views()
        upgraded = migration.used or bool(removed)
        if (
            section_images.ensure_opening(workspace, layout, job.state)
            or upgraded
        ):
            # A job of an older format, or one without its opening images:
            # written once at the current format (a dry run, or a folder this
            # process cannot write, keeps it in memory).
            if persist and os.access(layout.folder, os.W_OK):
                job._write_upgrade_if_writable(migration)
        return job

    def _write_upgrade_if_writable(self, migration: FolderMigration) -> None:
        """:meth:`_write_upgrade`, kept in memory where the folder refuses it.

        Windows' ``os.access`` reads neither a folder's permissions nor a
        file's read-only flag, so the refusal comes from the write itself.
        """
        try:
            self._write_upgrade(migration)
        except PermissionError as exc:
            logger.warning(
                "%s cannot be written (%s); the job is upgraded in memory only",
                self.layout.folder,
                exc.strerror,
            )

    def _write_upgrade(self, migration: FolderMigration) -> None:
        """Write the state at the current format, read again under
        the job's lock first: another process may have migrated (and written
        to) the job since it was read, and its state is the one kept; and
        ``job.json`` without the settings LangSlice no longer has. Nothing
        is written when that read needs no upgrade and every section has its
        opening image."""
        assert self.workspace is not None
        with self.lock.held():
            removed = _write_settings_without_removed(self.layout)
            migration.used = False
            data = read_checkpoint(self.checkpoint_path, migration)
            if data is not None:
                migration.seed(data)
                self.state.replace_with(
                    state_from_file(data, self.checkpoint_path)
                )
            self.load_undo()
            made = section_images.ensure_opening(
                self.workspace, self.layout, self.state
            )
            if data is None or migration.used or made:
                self.checkpoint()
            else:
                self._state_stamp = file_stamp(self.checkpoint_path)
                self._undo_stamp = file_stamp(self.undo_path)
        _log_migrated(self.layout, migration, removed)

    @property
    def transform_cap(self) -> int | None:
        """Below the maximum, the most sections one transform call may take."""
        cap = self.spec.transform.max_parallel
        return cap if cap < MAX_PARALLEL_TRANSFORMS else None

    def submit_errors(self, breaks: list[int]) -> Refused | None:
        """Every submit gate that applies to this job, in order (:func:`submit_errors`)."""
        return submit_errors(self.state, self.spec, breaks)

    def frozen_for(self, section_id: str) -> frozenset[str]:
        """The properties frozen on one section (:meth:`JobSpec.frozen_for`)."""
        return self.spec.frozen_for(section_id)

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
            held = [
                name
                for name in properties
                if name in self.frozen_for(section_id)
            ]
            if held:
                hit[section_id] = held
        if not hit:
            return None
        named = [
            name
            for name in freeze.PROPERTIES
            if any(name in held for held in hit.values())
        ]
        message = freeze.message(named, list(hit), len(self.state.slices))
        return Refused(
            RefusalCode.FROZEN,
            ids=list(hit),
            properties=named,
            message=message,
        )

    def busy_refusal(self, section_ids: Collection[str]) -> Refused | None:
        """:class:`Refused` ``SECTION_BUSY`` (``ids``, ``work``, ``message``) when background
        work is running on any of *section_ids* (it will write them when it
        lands, from what they hold now), else None. The landing of that work
        itself is not refused."""
        busy = (
            self._background.busy(section_ids)
            if self._background is not None
            else {}
        )
        if not busy:
            return None
        works = list({work.id: work for work in busy.values()}.values())
        running = "; ".join(
            f"{work.kind} is running on {', '.join(work.sections)} (work {work.id})"
            for work in works
        )
        return Refused(
            RefusalCode.SECTION_BUSY,
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

    def emit_results(
        self, progress: Callable[[str], None] | None = None
    ) -> Stack:
        """Write the results JSON (:func:`emit_results`); every picture saved."""
        self.views.flush()
        return emit_results(self.state, self.results_path, progress)

    def close(self) -> None:
        """Finish the job's background work (:attr:`background`), then its
        background writes (the pictures); the job stays usable."""
        if self._background is not None:
            self._background.close()
        self.views.flush()

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
(:class:`langslice.doors.tools.toolbox.ToolBox`).

**Undo.** One pattern: ``before = job.snapshot()`` before a write, then
``job.commit(before)`` once it succeeded (one undo step, then the
checkpoint). A step is the whole state, so undo restores everything on it,
the deformation references in the section records included (the records
themselves are content-addressed on disk and never change). The history is
the job folder's ``history/`` (:mod:`langslice.job.history`: an index and one
file per step), depth :data:`UNDO_DEPTH`; a resumed job reads it back, a
fresh one starts empty.

**Paths.** Every path the state stores (a deformation's ``record``, an image
correction's artifacts) is relative to the job folder.

**Live shared editing.** :meth:`Job.sync` notices that the state file changed
on disk since this job last read or wrote it (a script edited it, or another
job on the same folder wrote it) and reloads it before the next operation
instead of overwriting it. A plain edit becomes one undo step; when the
history file changed with it (another job wrote both), that history is read
back instead.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import os
import threading
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from langslice.core.fits import deformation
from langslice.core.jsonable import Json, JsonObject, to_jsonable
from langslice.core.refusal import RefusalCode, Refused
from langslice.core.sections import images as section_images
from langslice.core.sections.discovery import discover_slices
from langslice.core.stack.affine import IDENTITY_PHYSICAL, StoredAffine
from langslice.core.stack.codec import affine_from_json, trace_from_json
from langslice.core.stack.section import ROTATIONS, Section
from langslice.core.stack.stack import Stack
from langslice.job import public_files as formats
from langslice.job import recent
from langslice.job import registrations as regs
from langslice.job.background import BackgroundWork, in_worker
from langslice.job.history import UNDO_DEPTH as UNDO_DEPTH
from langslice.job.history import History
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
from langslice.job.pictures import DiscardedViews, ViewStore
from langslice.job.registrations import Registrations
from langslice.job.settings import frozen as freeze
from langslice.job.settings.spec import (
    MAX_PARALLEL_TRANSFORMS,
    JobSpec,
    supplied_angles,
)
from langslice.job.state_file import (
    Upgrade,
    read_checkpoint,
    relative_to,
    stack_from_json,
    stack_to_json,
    state_paths,
    write_checkpoint,
    write_json_atomic,
)

if TYPE_CHECKING:
    from langslice.core.workspace import Workspace

from langslice.job.gates import submit_errors

logger = logging.getLogger(__name__)

#: The transform a section with a frozen affine and none supplied carries: the
#: host's snapshot is already registered in-plane, so the identity IS its
#: alignment. ``kind`` ``"host"`` counts as a transform at submit and is never
#: written back to the host.
HOST_TRANSFORM_KIND = "host"

#: A file's identity on disk: inode, size and modification time. An atomic
#: rewrite is a new inode, so even a same-size write within the clock's
#: resolution shows.
Stamp = tuple[int, int, int] | None


def _stamp(path: str) -> Stamp:
    try:
        info = os.stat(path)
    except OSError:
        return None
    return (info.st_ino, info.st_size, info.st_mtime_ns)


def _named(spec: JobSpec, key: str) -> set[str]:
    """The section filenames ``spec.inputs[key]`` names (a list, or a mapping's keys)."""
    return {str(name) for name in (spec.inputs or {}).get(key) or []}


def host_damaged_ids(spec: JobSpec) -> set[str]:
    """Sections the host gave a damage note (``inputs.damaged``); the note
    stays first in the section's ``damage_note``."""
    return _named(spec, "damaged")


def host_transform() -> StoredAffine:
    """The identity transform a section with a frozen affine and none supplied
    carries: the host's snapshot is already aligned in-plane."""
    return StoredAffine(
        kind=HOST_TRANSFORM_KIND, params=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        physical=IDENTITY_PHYSICAL, mirrored=False)


# --- open and close ----------------------------------------------------------


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
                id=os.path.basename(path), index_original=index, index_corrected=index
            )
            for index, path in enumerate(paths)
        ],
    )
    state.notes.append(
        f"ingest: {len(paths)} sections, atlas {spec.atlas} ({spec.plane}) "
        f"spans {pos_lo:.2f}-{pos_hi:.2f} mm"
    )
    workspace.progress(f"[ingest] {len(paths)} sections from {workspace.image_folder}")
    return state


def _section(state: Stack, key: str, name: object) -> Section:
    """The section *name* of ``inputs.<key>``; ``ValueError`` for one not here."""
    section = state.by_id(str(name))
    if section is None:
        raise ValueError(f"inputs.{key} names an unknown section: {name!r}")
    return section


def _host_affine(given: object) -> StoredAffine:
    """Return a copy of a host's transform dictionary as the stored affine."""
    return affine_from_json(copy.deepcopy(cast("JsonObject", given)))


def apply_host_inputs(state: Stack, spec: JobSpec) -> None:
    """Write the host's answers for the tasks that are switched off.

    Order arrives as a list of filenames, positions as a filename -> mm
    mapping, angles as ``{"pitch": deg, "yaw": deg}`` for the whole stack or
    a filename -> ``{"pitch": deg, "yaw": deg}`` mapping, each section's own
    plane kept as supplied
    (:func:`langslice.job.settings.spec.supplied_angles`; a section it does
    not name keeps the flat plane), orientation as a
    filename -> ``{"flip": bool, "rotation_deg": 0|90|180|270}`` mapping (a
    missing key leaves that part as it is), damage as a filename -> note
    mapping, and transforms as filename -> stored transform dictionaries.
    Anything the host supplies for a task that IS on is applied too — it is
    a starting point, not a constraint. A ``damaged`` note becomes the
    section's ``damage_note`` (shown in status, so the agent can mark the
    regions it names); it does not make the section damaged, and it stays
    first in the note whatever the agent marks. ``spec.frozen`` is the
    constraint (:mod:`langslice.job.settings.frozen`): it must name sections
    that are here, a frozen position must have been supplied, and a section
    with a frozen affine and no supplied transform carries the ``"host"``
    identity (:func:`host_transform`), because its snapshot is already
    aligned. The other "none" values (the flat plane, no flip and no turn,
    no deformation) are what a section starts with.
    """
    inputs = spec.inputs or {}

    order = inputs.get("order") or []
    if order:
        known = [state.by_id(str(name)) for name in order]
        missing = [str(name) for name, hit in zip(order, known, strict=True) if hit is None]
        if missing:
            raise ValueError(f"inputs.order names sections that are not here: {missing}")
        tail = [s for s in state.in_order() if s.id not in {str(name) for name in order}]
        for index, record in enumerate([r for r in known if r is not None] + tail):
            record.index_corrected = index
        state.notes.append(f"inputs: order set by the host ({len(order)} sections)")

    positions = inputs.get("positions") or {}
    if positions:
        applied = 0
        for name, value in positions.items():
            record = _section(state, "positions", cast("object", name))
            record.position_mm = float(value)
            applied += 1
        state.notes.append(f"inputs: {applied} position(s) set by the host")

    stack_angles, section_angles = supplied_angles(inputs.get("angles"))
    if stack_angles is not None:
        pitch, yaw = stack_angles
        state.set_angles((pitch, yaw))
        state.notes.append(
            f"inputs: cutting angles set by the host "
            f"(pitch {pitch:.2f}, yaw {yaw:.2f})"
        )
    if section_angles:
        for name, (pitch, yaw) in section_angles.items():
            record = _section(state, "angles", name)
            record.angles = (pitch, yaw)
        state.notes.append(
            f"inputs: cutting angles of {len(section_angles)} section(s) set by the host"
        )

    orientation = inputs.get("orientation") or {}
    if orientation:
        if not isinstance(orientation, dict):
            raise ValueError("inputs.orientation must map section filenames to "
                             "{flip, rotation_deg}")
        for name, value in orientation.items():
            record = _section(state, "orientation", cast("object", name))
            if not isinstance(value, dict) or set(value) - {"flip", "rotation_deg"}:
                raise ValueError(f"inputs.orientation[{name!r}] must be "
                                 "{flip: bool, rotation_deg: 0|90|180|270}")
            flip = value.get("flip")
            if flip is not None:
                if not isinstance(flip, bool):
                    raise ValueError(f"inputs.orientation[{name!r}].flip must be true or false")
                record.flip = flip
            rotation = value.get("rotation_deg")
            if rotation is not None:
                if isinstance(rotation, bool) or not isinstance(rotation, (int, float)) or (
                        rotation % 360 not in ROTATIONS):
                    raise ValueError(f"inputs.orientation[{name!r}].rotation_deg must be one "
                                     f"of {list(ROTATIONS)}")
                record.rotation_deg = int(rotation % 360)
        state.notes.append(f"inputs: {len(orientation)} orientation(s) set by the host")

    transforms = inputs.get("transforms") or {}
    if transforms:
        for name, value in transforms.items():
            record = _section(state, "transforms", cast("object", name))
            if not isinstance(value, dict):
                raise ValueError(f"inputs.transforms[{name!r}] must be a transform dictionary")
            record.transform = _host_affine(cast("object", value))
        state.notes.append(f"inputs: {len(transforms)} transform(s) set by the host")

    damaged = inputs.get("damaged") or {}
    if damaged:
        # A host's word on a section's damage (a note, no regions): the agent
        # reads it in status and marks the regions it names.
        for name, note in damaged.items():
            record = _section(state, "damaged", cast("object", name))
            record.damage_note = str(note or "")
        state.notes.append(f"inputs: {len(damaged)} damage note(s) from the host")

    frozen = spec.frozen
    if frozen:
        unknown = freeze.unknown_names(frozen, {record.id for record in state.slices})
        if unknown:
            raise ValueError(f"frozen names sections that are not here: {unknown}")
        held = freeze.sections_of(frozen, [record.id for record in state.in_order()])
        unplaced = [name for name, props in held.items()
                    if "position" in props and (rec := state.by_id(name)) is not None
                    and rec.position_mm is None]
        if unplaced:
            raise ValueError(
                f"frozen: the position of {unplaced} is frozen, and a position has no "
                "default: supply it in inputs.positions")
        for name, props in held.items():
            record = _section(state, "frozen", name)
            if "affine" in props and record.transform is None:
                record.transform = host_transform()
        state.notes.append(
            f"inputs: {len(held)} section(s) with a frozen property "
            "(the agent may not change it)")


class InputsChanged(ValueError):
    """A job folder's checkpoint was made from other supplied inputs
    (``JobSpec.inputs``) than the spec opening it: resuming would keep the
    old ones and drop the new without a word (:meth:`Job.open`)."""


#: How a caller starts over instead of resuming (the CLI flag and the spec field).
START_FRESH = "start a fresh job instead (`--fresh` on the command line, resume=False in a spec)"


def changed_inputs(saved_spec: Any, spec: JobSpec) -> list[str]:
    """The ``inputs`` keys, and ``frozen``, whose values differ between a
    checkpoint's spec (*saved_spec*, the dict it stores) and *spec*; none
    when the checkpoint records no spec inputs."""
    if not isinstance(saved_spec, dict) or "inputs" not in saved_spec:
        return []

    # a checkpoint's spec, read as JSON
    saved = cast("dict[str, object]", saved_spec)
    inputs = saved.get("inputs")
    saved_inputs: Mapping[str, Json] = {}
    if isinstance(inputs, dict):
        saved_inputs = cast("dict[str, Json]", inputs)
    old = to_jsonable(saved_inputs)
    new = to_jsonable(spec.inputs or {})
    keys = set(old) | set(new)
    changed = [key for key in keys if old.get(key) != new.get(key)]
    if freeze.clean(saved.get("frozen")) != freeze.clean(spec.frozen):
        changed.append("frozen")
    return sorted(changed)


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
    layout: JobLayout, migration: FolderMigration, removed: Sequence[str],
) -> None:
    """Log a ``migrated`` event when a migration changed what the job read."""
    if migration.applied or migration.dropped or removed:
        layout.log_event(
            "migrated", migrations=list(migration.applied),
            dropped=dict(migration.dropped), settings=list(removed))


def _state_from(data: JsonObject, path: str | os.PathLike[str]) -> Stack:
    """The stack state *path* holds; ``ValueError`` naming the file and the
    unknown keys when it holds any."""
    try:
        return stack_from_json(data)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def inputs_changed(folder: str | os.PathLike[str], changed: list[str]) -> InputsChanged:
    """The refusal for resuming the job in *folder* with other *changed* inputs."""
    return InputsChanged(
        f"The job in {folder} was made from other supplied inputs "
        f"({', '.join(changed)} differ); resuming it would ignore the new "
        f"ones. Open it with the inputs it was made from, or {START_FRESH}."
    )


def emit_results(
    state: Stack, results_path: str, progress: Callable[[str], None] | None = None,
) -> Stack:
    """Write the results JSON — the same shape as the checkpoint, unversioned
    (its paths relative to the job folder, as the checkpoint's)."""
    write_json_atomic(results_path, stack_to_json(state))
    if progress is not None:
        progress(f"[emit] results -> {results_path}")
    return state


# --- background image corrections ---------------------------------------------------


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
        failure = Refused(type(error).__name__, message=str(error))
        return {"id": section_id, "geometry_fingerprint": fingerprint, **failure.payload()}
    except Exception as exc:  # the job records its own failures; this is a backstop
        failure = Refused(type(exc).__name__, message=str(exc))
        return {"id": section_id, "geometry_fingerprint": fingerprint, **failure.payload()}


def _land(state: Stack, section_id: str, fingerprint: str, result: JsonObject) -> bool:
    """Record a finished correction on a section still waiting for it."""
    section = state.by_id(section_id)
    held = None if section is None else section.image_correction
    if (section is not None and held is not None and held.status == "running"
            and held.geometry_fingerprint == fingerprint):
        section.image_correction = trace_from_json(result)
        return True
    return False


# --- the job -----------------------------------------------------------------------


class Job:
    """One stack being registered: its state, its rules and its files.

    Build one with :meth:`open` (resume or ingest, then the first
    checkpoint), or around a state already in hand
    (``Job(state, spec, layout=...)``: an empty history, nothing read or
    written until the first operation; the files as they stand on disk at
    that moment count as already seen). *results_path* None is the job
    folder's ``exports/linear_results.json``.
    """

    def __init__(
        self,
        state: Stack,
        spec: JobSpec,
        *,
        layout: JobLayout,
        results_path: str | None = None,
        undo: list[dict[str, Any]] | None = None,
        redo: list[dict[str, Any]] | None = None,
        history: History | None = None,
    ) -> None:
        self.state = state
        self.spec = spec
        self.layout = layout
        self.results_path = str(results_path or layout.results_file)
        self.history = history or History(layout.history_dir)
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
        #: Whole states, oldest first; the last one is what ``undo`` restores.
        self.undo_stack: list[dict[str, Any]] = list(undo or [])[-UNDO_DEPTH:]
        self.redo_stack: list[dict[str, Any]] = list(redo or [])
        #: Undo steps recorded by this job so far (:meth:`merge_steps`).
        self.pushed = 0
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
        #: ``settle_image_corrections`` (submit, the end of a session) and a
        #: fit that reads a trace (``wait_image_job``).
        self.image_jobs: dict[str, tuple[str, Future[dict[str, Any]]]] = {}
        self.image_executor: ThreadPoolExecutor | None = None
        self._background: BackgroundWork | None = None
        self._deformations: deformation.RecordStore | None = None
        #: False: nothing this job holds is written (the state, the history,
        #: the pictures); a dry run's job (the CLI's ``--dry-run``).
        self.persist = True
        self._workspace: Workspace | None = None
        #: How a state file and history step of an older format are read as
        #: the current format (the job's migrations, set by :meth:`open` and
        #: :meth:`load`; None refuses them).
        self.upgrade: Upgrade | None = None
        self._state_stamp = _stamp(self.checkpoint_path)
        self._undo_stamp = _stamp(self.undo_path)
        #: A read's view of a state another process saved while it held the
        #: lock (:meth:`reading`): the state this job held before it, put
        #: back by the next locked :meth:`sync`, and the file's stamp then.
        self._peeked: dict[str, Any] | None = None
        self._peek_stamp: Stamp = None
        self._registrations: Registrations | None = None
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

    # --- opening ---------------------------------------------------------------

    @classmethod
    def open(
        cls, spec: JobSpec, workspace: Workspace, *, folder: str | os.PathLike[str] | None = None,
        results_path: str | None = None,
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
        keeps its undo history; a fresh one starts without one (a history
        left by an earlier job here is emptied). A job folder or checkpoint
        of another format (a newer LangSlice's, or an older pre-release's) is
        refused (``ValueError``). A resume whose
        spec supplies other ``inputs`` than the checkpoint was made from is
        refused (:class:`InputsChanged`, naming the keys and how to start
        fresh) before anything is written: the old checkpoint would keep the
        old inputs and drop the new ones.
        """
        if folder is None:
            folder, _fallback = locate_job_folder(workspace.image_folder, spec.job_dir,
                                                  emit=workspace.progress)
        images = Path(os.path.abspath(workspace.image_folder))
        layout = JobLayout(Path(os.path.abspath(folder)), images)
        check_owner(layout)
        layout.ensure()
        lock = FolderLock(layout.folder)
        if getattr(workspace, "job_folder", None) is None:
            workspace.job_folder = str(layout.folder)
        migration = FolderMigration(workspace, layout)
        with lock.held():  # read and first checkpoint as one write
            state = None
            resumed = False
            if spec.resume:
                data = read_checkpoint(str(layout.state_file), migration)
                if data is not None:
                    migration.seed(data)
                state = None if data is None else _state_from(data, layout.state_file)
                changed = [] if state is None else changed_inputs(state.spec, spec)
                if changed:
                    raise inputs_changed(layout.folder, changed)
            removed = _removed_settings(layout)  # this spec replaces them
            write_job_file(layout, spec=spec.to_dict())
            history = History(layout.history_dir, migration)
            undo: list[dict[str, Any]] = []
            redo: list[dict[str, Any]] = []
            if state is not None:
                resumed = True
                workspace.progress(f"[ingest] resuming from {layout.state_file}")
                state.spec = spec.to_dict()
                state.image_folder = str(images)  # where they are now (a moved folder)
                state.submitted = False
                undo, redo = history.load()
            else:
                state = ingest(spec, workspace)
                apply_host_inputs(state, spec)
                if history.exists():
                    history.load()  # read before it is emptied: never delete unread steps
            if history.problem is not None:
                workspace.progress(f"[job] {history.problem}")
            # The opening ``preprocessed`` image of every section without one,
            # outside undo: part of the job as it opens.
            section_images.ensure_opening(workspace, layout, state)
            job = cls(state, spec, layout=layout, results_path=results_path,
                      undo=undo, redo=redo, history=history)
            job.lock = lock
            job.workspace = workspace
            job.upgrade = migration
            if (not undo and not redo and history.exists()) or history.upgraded:
                # a fresh job empties a history it read, never another's; an
                # upgraded one is written again at the current format
                job._save_history()
            job.checkpoint()
        _log_migrated(layout, migration, removed)
        layout.log_event("open", resumed=resumed)
        recent.remember(layout.folder, images, spec.atlas)
        return job

    @classmethod
    def load(
        cls, spec: JobSpec, workspace: Workspace, *, folder: str | os.PathLike[str],
        results_path: str | None = None, persist: bool = True,
    ) -> Job:
        """Open an existing job folder as it stands, writing nothing.

        The state is the folder's checkpoint as saved (``FileNotFoundError``
        without one; a newer one is refused, ``ValueError``), with its undo
        history; ``job.json`` and the state file are not rewritten, so a job
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
        data = read_checkpoint(str(layout.state_file), migration)
        if data is None:
            raise FileNotFoundError(f"No job state in {layout.folder}")
        removed = _removed_settings(layout)
        migration.seed(data)
        history = History(layout.history_dir, migration)
        undo, redo = history.load()
        job = cls(_state_from(data, layout.state_file), spec, layout=layout,
                  results_path=results_path,
                  undo=undo, redo=redo, history=history)
        job.persist = persist
        job.workspace = workspace
        job.upgrade = migration
        if not persist:
            job.views = DiscardedViews(layout)
            job.wire_views()
        upgraded = migration.used or history.upgraded or bool(removed)
        if section_images.ensure_opening(workspace, layout, job.state) or upgraded:
            # A job of an older format, or one without its opening images:
            # written once at the current format (a dry run keeps it in
            # memory).
            if persist:
                job._write_upgrade(migration)
        return job

    def _write_upgrade(self, migration: FolderMigration) -> None:
        """Write the state and history at the current format, read again under
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
                self.state.replace_with(_state_from(data, self.checkpoint_path))
            self.undo_stack, self.redo_stack = self.history.load()
            made = section_images.ensure_opening(self.workspace, self.layout, self.state)
            if data is None or migration.used or made or self.history.upgraded:
                if migration.used or self.history.upgraded:
                    self._save_history()
                self.checkpoint()
            else:
                self._state_stamp = _stamp(self.checkpoint_path)
                self._undo_stamp = _stamp(self.undo_path)
        _log_migrated(self.layout, migration, removed)

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
        """The undo history's index (``history/index.json``)."""
        return str(self.history.index_path)

    @property
    def engine_log(self) -> str | None:
        """Where the registration engines' own output goes while they fit
        (``logs/engine.log``, :mod:`langslice.core.fits.engine_log`); None for
        a job that persists nothing."""
        return str(self.layout.logs_dir / "engine.log") if self.persist else None

    def portable(self, correction: dict[str, Any]) -> dict[str, Any]:
        """An image correction's record with its paths relative to the job folder."""
        converted = state_paths({"slices": [{"image_correction": correction}]},
                                relative_to(self.folder))
        return converted["slices"][0]["image_correction"]

    # --- checkpoint and observers ------------------------------------------------

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
            self._state_stamp = _stamp(self.checkpoint_path)
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
        with self.lock.held(self.background.gives_way if in_worker() else None):
            yield self.sync()

    @contextlib.contextmanager
    def reading(self) -> Iterator[None]:
        """For a read: the state brought up to date without waiting for
        another process's writer. This process's other threads are kept out
        (their writes are short); when another process holds the job folder's
        lock (a long fit applying, a plugin tool that holds it while it
        computes), the state is read as that process last saved it, and the
        next locked :meth:`sync` puts the job's own back first."""
        with self.lock.threads():
            with self.lock.held_if_free() as free:
                if free:
                    self.sync()
            if not free:
                self._peek()
            yield

    def _peek(self) -> None:
        """Read the state file as it stands, without the lock (:meth:`reading`)."""
        stamp = _stamp(self.checkpoint_path)
        if stamp in (self._state_stamp, self._peek_stamp):
            return
        try:
            data = read_checkpoint(self.checkpoint_path, self.upgrade)
        except (OSError, ValueError):
            return
        if data is None:
            return
        held = stack_to_json(self.state)
        loaded = stack_to_json(_state_from(data, self.checkpoint_path))
        self._peek_stamp = stamp
        if loaded == held:
            return
        if self._peeked is None:
            self._peeked = held
        self.state.replace_with(stack_from_json(loaded))
        self._notify()

    # --- the registrations each section has held -------------------------------------

    @property
    def registrations(self) -> Registrations:
        """Every registration each section has held (:mod:`langslice.job.registrations`)."""
        if self._registrations is None:
            self._registrations = Registrations(self.layout)
        return self._registrations

    @contextlib.contextmanager
    def acting(self, tool: str) -> Iterator[None]:
        """This thread runs *tool*'s call for the block: the registrations its
        writes leave are listed as made by it."""
        outer = getattr(self._acting, "tool", None)
        self._acting.tool = tool
        try:
            yield
        finally:
            self._acting.tool = outer

    def registration_numbers(self, ids: Sequence[str]) -> dict[str, int]:
        """The number of the registration each section of *ids* holds now,
        for the sections that hold a listed one (a picture records them)."""
        out: dict[str, int] = {}
        for section_id in ids:
            record = self.state.by_id(section_id)
            number = (self.registrations.number_of(self.state, record)
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

    def list_registrations(self, before: dict[str, Any]) -> None:
        """List each registration a step left (*before*: the state before it,
        as :meth:`snapshot` gives it), as made by the tool this thread runs
        (:meth:`acting`; ``script`` for none), and the one it replaced when
        that was never listed (a job older than the list). A deformation the
        step made stale is no part of a registration
        (:func:`langslice.job.registrations.value`). A failure is logged; the
        step stands."""
        if not self.persist:
            return
        store = self.registrations
        by = getattr(self._acting, "tool", None) or regs.SCRIPT
        try:
            old = stack_from_json(before)
            for section in self.state.slices:
                now = regs.value(self.state, section)
                prior = old.by_id(section.id)
                then = regs.value(old, prior) if prior is not None else None
                key = regs.digest(now) if now is not None else None
                if key == (regs.digest(then) if then is not None else None):
                    continue
                if then is not None and prior is not None and store.find(
                        section.id, regs.digest(then)) is None:
                    store.add(section.id, then, position_mm=prior.position_mm,
                              angles=prior.angles, by=regs.made_by(then),
                              at=None)
                if now is not None:
                    entry = store.add(section.id, now, position_mm=section.position_mm,
                                      angles=section.angles, by=by,
                                      at=regs.now())
                    number = cast("int", entry["id"])  # an entry's number, from its file
                    self.registered.append((threading.get_ident(), section.id, number))
        except Exception:  # noqa: BLE001 - the step is written: a listing failure never undoes it
            logger.warning("Could not list the registrations a step left", exc_info=True)

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

    # --- undo ------------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """The state as it stands: take it before a write, hand it to :meth:`commit`."""
        return stack_to_json(self.state)

    def commit(self, before: dict[str, Any]) -> None:
        """Record one undo step (*before*), clear the redo side, checkpoint,
        and list the registrations the step left (:meth:`list_registrations`).
        Call it inside :meth:`writing` (with *before* taken there)."""
        with self.lock.held():
            self._push(before)
            self._save_history()
            self.checkpoint()
            self.list_registrations(before)

    def _push(self, before: dict[str, Any]) -> None:
        self.undo_stack.append(before)
        del self.undo_stack[:-UNDO_DEPTH]
        self.redo_stack.clear()
        self.pushed += 1

    def merge_steps(self, mark: int) -> None:
        """Make the undo steps recorded since :attr:`pushed` was *mark* one
        step: the state before the first of them. A tool whose call writes
        through several operations is still one undo step. Call it inside
        :meth:`writing`."""
        count = min(self.pushed - mark, len(self.undo_stack))
        if count < 2:
            return
        first = self.undo_stack[-count]
        del self.undo_stack[-count:]
        self.undo_stack.append(first)
        self.pushed -= count - 1  # the steps since an earlier mark are still counted right
        self._save_history()

    def undo(self) -> bool:
        """Restore the state before the last step; False when there is none."""
        with self.writing():
            if not self.undo_stack:
                return False
            before = stack_to_json(self.state)
            self.redo_stack.append(before)
            self.state.replace_with(stack_from_json(self.undo_stack.pop()))
            self._save_history()
            self.checkpoint()
            self.list_registrations(before)
            return True

    def redo(self) -> bool:
        """Re-apply the step :meth:`undo` reversed; False when there is none."""
        with self.writing():
            if not self.redo_stack:
                return False
            before = stack_to_json(self.state)
            self.undo_stack.append(before)
            del self.undo_stack[:-UNDO_DEPTH]
            self.state.replace_with(stack_from_json(self.redo_stack.pop()))
            self._save_history()
            self.checkpoint()
            self.list_registrations(before)
            return True

    def _save_history(self) -> None:
        if not self.persist:
            return
        with self.lock.held():
            self.history.save(self.undo_stack, self.redo_stack)
        self._undo_stamp = _stamp(self.undo_path)

    # --- live shared editing -------------------------------------------------------

    def sync(self) -> dict[str, Any] | None:
        """Reload the state file if it changed on disk since this job last
        read or wrote it.

        Returns the state as this job held it before a reload, or ``None``
        when nothing was reloaded. An edit from outside (the history file
        unchanged) becomes one undo step; when the history file changed too,
        another job wrote both and its history is read back instead. A file
        that does not parse (a script mid-write) is left alone and read at
        the next call; the job's own next write replaces it.
        """
        if self._peeked is not None:  # a read's view first gives way to the job's own state
            self.state.replace_with(stack_from_json(self._peeked))
            self._peeked, self._peek_stamp = None, None
        state_stamp = _stamp(self.checkpoint_path)
        undo_stamp = _stamp(self.undo_path)
        history_changed = undo_stamp != self._undo_stamp
        if history_changed:
            self.undo_stack, self.redo_stack = self.history.load()
            self._undo_stamp = undo_stamp
        if state_stamp == self._state_stamp:
            return None
        try:
            data = read_checkpoint(self.checkpoint_path, self.upgrade)
        except (OSError, ValueError):
            logger.warning("State file %s changed but does not read; keeping the job's state",
                           self.checkpoint_path, exc_info=True)
            return None
        self._state_stamp = state_stamp
        if data is None:  # deleted: the next checkpoint writes it again
            return None
        held = stack_to_json(self.state)
        loaded = stack_to_json(_state_from(data, self.checkpoint_path))
        if loaded == held:
            return None
        if not history_changed:
            self._push(held)
            self._save_history()
        self.state.replace_with(stack_from_json(loaded))
        logger.info("Reloaded %s: it changed on disk", self.checkpoint_path)
        self._notify()
        return held

    # --- rules ------------------------------------------------------------------------

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

    def frozen_refusal(self, section_ids: Collection[str], *properties: str,
                       ) -> Refused | None:
        """``FROZEN`` (``ids``, ``properties``, ``message``) when a write
        to any of *properties* of any of *section_ids* would change what the
        job froze there (the sections and the properties it names are the
        frozen ones), else None."""
        hit: dict[str, list[str]] = {}
        for section_id in section_ids:
            held = [name for name in properties if name in self.frozen_for(section_id)]
            if held:
                hit[section_id] = held
        if not hit:
            return None
        named = [name for name in freeze.PROPERTIES if any(name in held for held in hit.values())]
        message = freeze.message(named, list(hit), len(self.state.slices))
        return Refused(RefusalCode.FROZEN, ids=list(hit), properties=named,
                       message=message)

    def busy_refusal(self, section_ids: Collection[str]) -> Refused | None:
        """:class:`Refused` ``SECTION_BUSY`` (``ids``, ``work``, ``message``) when background
        work is running on any of *section_ids* (it will write them when it
        lands, from what they hold now), else None. The landing of that work
        itself is not refused."""
        busy = self._background.busy(section_ids) if self._background is not None else {}
        if not busy:
            return None
        works = list({work.id: work for work in busy.values()}.values())
        running = "; ".join(f"{work.kind} is running on {', '.join(work.sections)} (work {work.id})"
                            for work in works)
        return Refused(RefusalCode.SECTION_BUSY, ids=list(busy),
                       work=[work.id for work in works],
                       message=f"{running}. Nothing written: write to it once the work "
                       "lands (its notice comes with a later reply).")

    def clear_stale_deformations(self) -> list[str]:
        """Drop deformations whose linear placement changed; checkpoint; their ids.

        A deformation is fitted on top of one linear placement; a write that
        moves the position, orientation, cutting angles or transform makes it
        stale. Called after every write, it lands in that write's undo step
        (the step holds the state from before the write).
        """
        with self.writing():
            cleared = deformation.clear_stale(self.state)
            if cleared:
                self.checkpoint()
        return cleared

    @property
    def deformations(self) -> deformation.RecordStore:
        """Fitted records, saved in each section's folder
        (``sections/<name>/deformable/<key>``; made on first use)."""
        if self._deformations is None:
            self._deformations = deformation.RecordStore(
                root=self.layout.sections_dir, folder_of=self.layout.deformable_dir)
        return self._deformations

    @property
    def background(self) -> BackgroundWork:
        """The job's background work (:mod:`langslice.job.background`; made on first use)."""
        if self._background is None:
            self._background = BackgroundWork(self)
        return self._background

    # --- image corrections ----------------------------------------------------------

    def start_image_job(
        self, section_id: str, fingerprint: str, job: Callable[[], dict[str, Any]], *,
        workers: int,
    ) -> None:
        """Run one image correction in the background."""
        if self.image_executor is None:
            self.image_executor = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="image-correction"
            )
        self.image_jobs[section_id] = (fingerprint, self.image_executor.submit(job))

    def image_job_running(self, section_id: str, fingerprint: str) -> bool:
        running = self.image_jobs.get(section_id)
        return running is not None and running[0] == fingerprint and not running[1].done()

    def settle_image_corrections(self) -> bool:
        """Wait for every running correction, record its result, checkpoint.

        A result lands only on a section that still holds the running record
        for the same geometry; one undone or superseded meanwhile keeps what
        it has (the reply stays on disk and is reused at that geometry).
        Returns whether any section changed (no undo step: a landing finishes
        the step that started it).
        """
        results = [(section_id, fingerprint, self.portable(_job_result(
            section_id, fingerprint, future, None)))
            for section_id, (fingerprint, future) in list(self.image_jobs.items())]
        self.image_jobs.clear()
        if self.image_executor is not None:
            self.image_executor.shutdown(wait=True)
            self.image_executor = None
        if not results:
            return False
        changed = False
        with self.writing():  # the calls ran outside the lock; they land under it
            for section_id, fingerprint, result in results:
                changed |= _land(self.state, section_id, fingerprint, result)
            if changed:
                self.checkpoint()
        return changed

    def wait_image_job(self, section_id: str, timeout: float | None) -> bool:
        """Wait up to *timeout* seconds (None: as long as it takes) for one
        section's running correction.

        Records its result the way :meth:`settle_image_corrections` does
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

    # --- results ----------------------------------------------------------------------

    def emit_results(self, progress: Callable[[str], None] | None = None) -> Stack:
        """Write the results JSON (:func:`emit_results`); every picture saved."""
        self.views.flush()
        return emit_results(self.state, self.results_path, progress)

    def close(self) -> None:
        """Finish the job's background work (:attr:`background`), then its
        background writes (the pictures); the job stays usable."""
        if self._background is not None:
            self._background.close()
        self.views.flush()

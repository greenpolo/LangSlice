"""The job: one stack's state and everything that keeps it consistent.

The job layer of the layered core (``_local`` plan, phase 2). A :class:`Job`
owns the :class:`~langslice.linear.state.StackState` and the
:class:`~langslice.linear.spec.JobSpec`, the host's locked and damaged
sections, undo/redo (persisted next to the checkpoint), the checkpoint and its
observers, the submit gates, the stale-deformation rule, the deformation
record store, the transform-count cap and the background image-correction
jobs. ``ingest``, ``apply_host_inputs`` and ``emit_results`` open and close
it. It imports the core only: no agent framework, no message types, no
provider. The doors (the ADK toolbox, the MCP server) sit on it; a script
can drive it directly.

What a Job never holds: the look-before-commit gates (``view_placement``
before ``set_positions``, ``view_stack`` before ``submit``) and the
model-delivery bookkeeping (which pictures a model has received). Those are
the tool doors' (:class:`langslice.linear.toolbox.ToolBox`); a library call
is never gated.

**Undo.** One pattern: ``before = job.snapshot()`` before a write, then
``job.commit(before)`` once it succeeded (one undo step, then the
checkpoint). A step is the whole state, so undo restores everything on it,
the deformation references in the section records included (the records
themselves are content-addressed on disk and never change). The history is
the plain file :data:`UNDO_FILENAME` beside the checkpoint, versioned
(:data:`UNDO_FORMAT_VERSION`), depth :data:`UNDO_DEPTH`; a resumed job reads
it back, a fresh one starts empty.

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
import json
import logging
import os
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from langslice.linear import deformation
from langslice.linear.checkpoint import (
    FORMAT_KEY,
    STATE_FORMAT_VERSION,
    notify_observers,
    read_checkpoint,
    upgrade_state,
    write_checkpoint,
    write_json_atomic,
)
from langslice.linear.discovery import discover_slices
from langslice.linear.spec import MAX_PARALLEL_TRANSFORMS, JobSpec
from langslice.linear.state import IDENTITY_PARAMS, SliceState, StackState

if TYPE_CHECKING:
    from langslice.linear.workspace import Workspace

logger = logging.getLogger(__name__)

#: Undo steps kept (and saved). One tool call is one step, a batch included.
UNDO_DEPTH = 50
#: The undo history, beside the checkpoint.
UNDO_FILENAME = "linear_undo.json"
#: The history file's format. 1 (2026-10-03): ``undo`` and ``redo``, oldest
#: first, each a whole state at ``state_format_version``.
UNDO_FORMAT_VERSION = 1

#: A reported interval break must exceed this multiple of the median spacing.
INTERVAL_BREAK_MIN_RATIO = 1.5
#: With ``strict_interval``, every spacing within this fraction of the interval.
STRICT_INTERVAL_TOLERANCE = 0.10

#: The transform a locked section carries: the host's snapshot is already
#: registered in-plane, so the identity IS its alignment. ``kind`` ``"host"``
#: counts as a transform at submit and is never written back to the host.
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


def locked_ids(spec: JobSpec) -> set[str]:
    """Sections whose flip, rotation and transform the host locked."""
    return {str(name) for name in (spec.inputs or {}).get("locked") or []}


def host_damaged_ids(spec: JobSpec) -> set[str]:
    """Sections the host marked damaged; the agent cannot clear these flags."""
    return {str(name) for name in ((spec.inputs or {}).get("damaged") or {})}


def host_transform() -> dict[str, Any]:
    """The identity transform a locked section carries."""
    return {
        "kind": HOST_TRANSFORM_KIND,
        "params": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "physical": {**IDENTITY_PARAMS, "shear": 0.0, "pivot": [0.5, 0.5]},
        "mirrored": False,
    }


# --- open and close ----------------------------------------------------------


def ingest(spec: JobSpec, workspace: Workspace) -> StackState:
    """Discover the folder and build the stack state. Plain code, no model."""
    paths = discover_slices(workspace.image_folder)
    if not paths:
        raise ValueError(f"No slice images found in {workspace.image_folder}")

    pos_lo, pos_hi = workspace.position_range
    state = StackState(
        image_folder=workspace.image_folder,
        atlas=spec.atlas,
        plane=spec.plane,
        interval_mm=spec.interval_mm,
        thickness_mm=spec.thickness_mm,
        spec=spec.to_dict(),
        slices=[
            SliceState(
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


def apply_host_inputs(state: StackState, spec: JobSpec) -> None:
    """Write the host's answers for the tasks that are switched off.

    Order arrives as a list of filenames, positions as a filename -> mm
    mapping, angles as ``{"pitch": deg, "yaw": deg}``, damage as a filename
    -> note mapping, and transforms as filename -> stored transform dictionaries.
    Anything the host
    supplies for a task that IS on is applied too — it is a starting point,
    not a constraint. Two exceptions are constraints: ``damaged`` flags the
    agent cannot clear, and ``locked`` sections (a list of filenames) whose
    flip, rotation and transform the agent cannot change; a locked section
    without a supplied transform carries the ``"host"`` identity
    (:func:`host_transform`), because its snapshot is already aligned.
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
            record = state.by_id(str(name))
            if record is None:
                raise ValueError(f"inputs.positions names an unknown section: {name!r}")
            record.position_mm = float(value)
            applied += 1
        state.notes.append(f"inputs: {applied} position(s) set by the host")

    angles = inputs.get("angles") or {}
    if angles:
        state.cutting_angles_deg = {
            "pitch": float(angles.get("pitch", 0.0)),
            "yaw": float(angles.get("yaw", 0.0)),
        }
        state.notes.append(
            f"inputs: cutting angles set by the host "
            f"(pitch {state.pitch_deg:.2f}, yaw {state.yaw_deg:.2f})"
        )

    transforms = inputs.get("transforms") or {}
    if transforms:
        for name, value in transforms.items():
            record = state.by_id(str(name))
            if record is None:
                raise ValueError(f"inputs.transforms names an unknown section: {name!r}")
            if not isinstance(value, dict):
                raise ValueError(f"inputs.transforms[{name!r}] must be a transform dictionary")
            # Preserve complete historical mappings, including splines. The image
            # correction handoff explicitly refuses unsupported spline inputs.
            record.transform = copy.deepcopy(value)
        state.notes.append(f"inputs: {len(transforms)} transform(s) set by the host")

    damaged = inputs.get("damaged") or {}
    if damaged:
        # Damage is normally the agent's own classification; a host (or a
        # benchmark) may assert it up front so the automatic fits refuse the
        # section and it is aligned by hand.
        for name, note in damaged.items():
            record = state.by_id(str(name))
            if record is None:
                raise ValueError(f"inputs.damaged names an unknown section: {name!r}")
            record.damaged = True
            record.damage_note = str(note or "")
        state.notes.append(f"inputs: {len(damaged)} section(s) marked damaged by the host")

    locked = inputs.get("locked") or []
    if locked:
        if not isinstance(locked, (list, tuple)):
            raise ValueError("inputs.locked must be a list of section filenames")
        for name in locked:
            record = state.by_id(str(name))
            if record is None:
                raise ValueError(f"inputs.locked names an unknown section: {name!r}")
            if record.transform is None:
                record.transform = host_transform()
        state.notes.append(
            f"inputs: {len(locked)} section(s) locked by the host (in-plane alignment done)"
        )


def emit_results(
    state: StackState, results_path: str, progress: Callable[[str], None] | None = None,
) -> StackState:
    """Write the results JSON — the same shape as the checkpoint, unversioned."""
    os.makedirs(os.path.dirname(os.path.abspath(results_path)), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(state.to_dict(), handle, indent=2)
    if progress is not None:
        progress(f"[emit] results -> {results_path}")
    return state


# --- the undo history file ------------------------------------------------------


def undo_path_for(checkpoint_path: str) -> str:
    """The undo history beside *checkpoint_path*."""
    return os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), UNDO_FILENAME)


def read_history(path: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(undo, redo)`` from a history file, each upgraded; empty without one.

    An unreadable file is logged and read as empty: the history is a
    convenience, the checkpoint is the record.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return [], []
    except (OSError, ValueError):
        logger.warning("Undo history %s unreadable; starting without it", path, exc_info=True)
        return [], []
    version = data.get(FORMAT_KEY) if isinstance(data, dict) else None
    if not isinstance(version, int) or version > UNDO_FORMAT_VERSION:
        logger.warning("Undo history %s has format %r; starting without it", path, version)
        return [], []
    state_version = data.get("state_format_version", 0)

    def upgraded(entries: Any) -> list[dict[str, Any]]:
        return [upgrade_state({**entry, FORMAT_KEY: state_version})
                for entry in entries or [] if isinstance(entry, dict)]

    try:
        return upgraded(data.get("undo")), upgraded(data.get("redo"))
    except ValueError:
        logger.warning("Undo history %s unreadable; starting without it", path, exc_info=True)
        return [], []


def write_history(path: str, undo: list[dict[str, Any]], redo: list[dict[str, Any]]) -> None:
    """Write the history atomically (compact JSON: it is rewritten every step)."""
    write_json_atomic(path, {
        FORMAT_KEY: UNDO_FORMAT_VERSION,
        "state_format_version": STATE_FORMAT_VERSION,
        "depth": UNDO_DEPTH,
        "undo": undo,
        "redo": redo,
    }, indent=None)


# --- the submit gates ------------------------------------------------------------


def missing_positions(state: StackState) -> dict[str, Any] | None:
    """``None`` when every section has a position, else the rejection."""
    missing = [record.id for record in state.in_order() if record.position_mm is None]
    if not missing:
        return None
    return {
        "status": "error",
        "error": "MISSING_POSITIONS",
        "missing_ids": missing,
        "message": (
            f"{len(missing)} of {len(state.slices)} section(s) have no "
            "position; every section needs one, damaged ones included."
        ),
    }


def order_position_mismatch(state: StackState) -> dict[str, Any] | None:
    """Positions must run one way along the corrected order (either way)."""
    placed = [s for s in state.in_order() if s.position_mm is not None]
    if len(placed) < 2:
        return None
    first = float(placed[0].position_mm)  # type: ignore[arg-type]
    last = float(placed[-1].position_mm)  # type: ignore[arg-type]
    direction = 1.0 if last >= first else -1.0
    pairs: list[dict[str, Any]] = []
    for before, after in zip(placed, placed[1:], strict=False):
        delta = float(after.position_mm) - float(before.position_mm)  # type: ignore[arg-type]
        if delta * direction < 0:
            pairs.append(
                {
                    "before": before.id,
                    "after": after.id,
                    "before_position_mm": round(float(before.position_mm), 3),  # type: ignore[arg-type]
                    "after_position_mm": round(float(after.position_mm), 3),  # type: ignore[arg-type]
                }
            )
    if not pairs:
        return None
    trend = "increase" if direction > 0 else "decrease"
    return {
        "status": "error",
        "error": "ORDER_POSITION_MISMATCH",
        "pairs": pairs,
        "message": (
            f"Positions {trend} from {first:.3f} mm to {last:.3f} mm along the "
            f"corrected order, but {len(pairs)} neighbour pair(s) run the other "
            "way."
        ),
    }


def strict_interval_error(state: StackState, breaks: list[int]) -> dict[str, Any] | None:
    """Every consecutive spacing within 10% of the interval, no breaks."""
    if breaks:
        return {
            "status": "error",
            "error": "STRICT_INTERVAL",
            "reported_breaks": list(breaks),
            "message": (
                "strict_interval is on: interval_breaks must be empty, "
                f"{len(breaks)} were reported."
            ),
        }
    interval = float(state.interval_mm)
    if interval <= 0:
        return None
    placed = [s for s in state.in_order() if s.position_mm is not None]
    tolerance = interval * STRICT_INTERVAL_TOLERANCE
    failures: list[dict[str, Any]] = []
    for before, after in zip(placed, placed[1:], strict=False):
        spacing = abs(float(after.position_mm) - float(before.position_mm))  # type: ignore[arg-type]
        if abs(spacing - interval) > tolerance:
            failures.append(
                {
                    "between": [before.id, after.id],
                    "spacing_mm": round(spacing, 3),
                }
            )
    if not failures:
        return None
    return {
        "status": "error",
        "error": "STRICT_INTERVAL",
        "interval_mm": round(interval, 3),
        "tolerance_mm": round(tolerance, 3),
        "failures": failures,
        "message": (
            f"strict_interval is on: every consecutive spacing must be within "
            f"{STRICT_INTERVAL_TOLERANCE:.0%} of {interval:.3f} mm; "
            f"{len(failures)} pair(s) are not."
        ),
    }


def unsupported_breaks(state: StackState, breaks: list[int]) -> dict[str, Any] | None:
    """Check reported interval breaks against the spacing that was WRITTEN.

    A break at corrected index *i* claims the gap between *i-1* and *i* is
    larger than the rest of the stack's; the positions on the state make that
    claim checkable without an image.
    """
    if not breaks:
        return None
    ordered = state.in_order()
    deltas = [
        abs(float(b.position_mm) - float(a.position_mm))  # type: ignore[arg-type]
        for a, b in zip(ordered, ordered[1:], strict=False)
        if a.position_mm is not None and b.position_mm is not None
    ]
    if not deltas:
        return None
    median = float(sorted(deltas)[len(deltas) // 2])
    threshold = median * INTERVAL_BREAK_MIN_RATIO

    failures: list[dict[str, Any]] = []
    for index in sorted(set(breaks)):
        position = next(
            (i for i, r in enumerate(ordered) if r.index_corrected == index), None
        )
        if position is None or position == 0:
            failures.append(
                {
                    "index": index,
                    "error": "NOT_A_GAP",
                    "reason": (
                        f"corrected index {index} has no section before it; a "
                        "break index names the section AFTER the gap."
                    ),
                }
            )
            continue
        before, after = ordered[position - 1], ordered[position]
        if before.position_mm is None or after.position_mm is None:
            continue
        written = abs(float(after.position_mm) - float(before.position_mm))
        if written > threshold:
            continue
        failures.append(
            {
                "index": index,
                "error": "NOT_A_GAP",
                "written_interval_mm": round(written, 3),
                "median_interval_mm": round(median, 3),
                "between": [before.id, after.id],
            }
        )
    if not failures:
        return None
    return {
        "status": "error",
        "error": "INTERVAL_BREAKS_UNSUPPORTED",
        "failures": failures,
        "message": (
            f"{len(failures)} reported interval break(s) are not in the "
            "positions you wrote. A break index is accepted only where the "
            f"written interval exceeds {INTERVAL_BREAK_MIN_RATIO:g}x the "
            "stack's median written spacing."
        ),
    }


def missing_transforms(state: StackState) -> dict[str, Any] | None:
    """``None`` when every section carries a transform, else the rejection."""
    missing = [record.id for record in state.in_order() if record.transform is None]
    if not missing:
        return None
    return {
        "status": "error",
        "error": "MISSING_TRANSFORMS",
        "missing_ids": missing,
        "message": (
            f"{len(missing)} of {len(state.slices)} section(s) carry no "
            "transform; every section needs one, damaged ones included."
        ),
    }


def damaged_transform_error(state: StackState, spec: JobSpec) -> dict[str, Any] | None:
    """Damaged sections require an applied interactive correction, not identity."""
    failures = []
    identity = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    # A locked section was aligned by the user and cannot be changed here.
    locked = locked_ids(spec)
    for record in state.in_order():
        if not record.damaged or record.id in locked:
            continue
        transform = record.transform or {}
        reason = None
        if not transform:
            reason = "missing_transform"
        elif transform.get("kind") != "interactive":
            reason = "not_interactive"
        else:
            try:
                params = np.asarray(transform.get("params"), dtype=float)
                if params.shape != (6,) or not np.isfinite(params).all():
                    reason = "invalid_transform"
                elif transform.get("spline") is not None:
                    from langslice.landmark_warp import fit_spline

                    spline = transform["spline"]
                    fitted = fit_spline(spline)
                    is_identity = (
                        fitted.is_identity() if spline.get("backend") == "elastix" else
                        np.allclose(spline["source"], spline["target"], rtol=0, atol=1e-9)
                    )
                    if is_identity:
                        reason = "identity_transform"
                elif np.allclose(params, identity, rtol=0, atol=1e-9):
                    reason = "identity_transform"
            except (TypeError, ValueError, KeyError, RuntimeError, np.linalg.LinAlgError):
                reason = "invalid_transform"
        if reason:
            failures.append({"id": record.id, "reason": reason})
    if not failures:
        return None
    return {
        "status": "error",
        "error": "DAMAGED_REQUIRES_MANUAL_TRANSFORM",
        "failures": failures,
        "interactive_enabled": spec.transform.interactive,
        "message": (
            "Damaged sections require a non-identity interactive transform. "
            + (
                "Use adjust_transforms to align the surviving "
                "anatomy, inspect the returned overlays, then submit again."
                if spec.transform.interactive else
                "Interactive transform tools are disabled for this run; the host "
                "must enable interactive transforms to resolve these sections."
            )
        ),
    }


def missing_deformations(state: StackState) -> dict[str, Any] | None:
    """``None`` when every section carries a deformation, or a keep_linear
    reason, at its current linear placement; else the rejection."""
    missing: list[dict[str, str]] = []
    for record in state.in_order():
        held = record.deformation or {}
        if not held:
            missing.append({"id": record.id, "reason": "no deformation and no keep_linear reason"})
        elif held.get("linear_key") != deformation.linear_key(state, record):
            missing.append({"id": record.id, "reason": "made at a different linear placement"})
    if not missing:
        return None
    return {
        "status": "error",
        "error": "MISSING_DEFORMATIONS",
        "sections": missing,
        "message": (
            f"{len(missing)} of {len(state.slices)} section(s) carry no deformation at their "
            "current placement. Apply one fit_deformable result per section, or give "
            "fit_deformable keep_linear with a reason where the linear placement stands."
        ),
    }


def submit_errors(
    state: StackState, spec: JobSpec, breaks: list[int]
) -> dict[str, Any] | None:
    """Every gate that applies to this run, in order."""
    if spec.has("position"):
        refusal = (
            missing_positions(state)
            or order_position_mismatch(state)
            or (
                strict_interval_error(state, breaks)
                if spec.position.strict_interval
                else unsupported_breaks(state, breaks)
            )
        )
        if refusal is not None:
            return refusal
    if spec.has("transform"):
        refusal = damaged_transform_error(state, spec) or missing_transforms(state)
        if refusal is not None:
            return refusal
    if spec.has("nonlinear"):
        return missing_deformations(state)
    return None


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
        return {"id": section_id, "geometry_fingerprint": fingerprint,
                "status": "error", "error": type(error).__name__, "message": str(error)}
    except Exception as exc:  # the job records its own failures; this is a backstop
        return {"id": section_id, "geometry_fingerprint": fingerprint,
                "status": "error", "error": type(exc).__name__, "message": str(exc)}


def _land(state: StackState, section_id: str, fingerprint: str, result: dict[str, Any]) -> bool:
    """Record a finished correction on a section still waiting for it."""
    record = state.by_id(section_id)
    held = (record.image_correction or {}) if record is not None else {}
    if (record is not None and held.get("status") == "running"
            and held.get("geometry_fingerprint") == fingerprint):
        record.image_correction = result
        return True
    return False


# --- the job -----------------------------------------------------------------------


class Job:
    """One stack being registered: its state, its rules and its files.

    Build one with :meth:`open` (resume or ingest, then the first
    checkpoint), or around a state already in hand (``Job(state, spec,
    checkpoint_path=..., results_path=...)``: an empty history, nothing read
    or written until the first operation; the files as they stand on disk at
    that moment count as already seen).
    """

    def __init__(
        self,
        state: StackState,
        spec: JobSpec,
        *,
        checkpoint_path: str,
        results_path: str,
        undo: list[dict[str, Any]] | None = None,
        redo: list[dict[str, Any]] | None = None,
    ) -> None:
        self.state = state
        self.spec = spec
        self.checkpoint_path = str(checkpoint_path)
        self.results_path = str(results_path)
        #: Sections whose flip, rotation and transform the host locked.
        self.locked = frozenset(locked_ids(spec))
        #: Sections the host marked damaged; their flags cannot be cleared.
        self.host_damaged = frozenset(host_damaged_ids(spec))
        #: Whole states, oldest first; the last one is what ``undo`` restores.
        self.undo_stack: list[dict[str, Any]] = list(undo or [])[-UNDO_DEPTH:]
        self.redo_stack: list[dict[str, Any]] = list(redo or [])
        #: Called with the state after every checkpoint and every reload.
        self.observers: list[Callable[[StackState], None]] = []
        #: Image corrections still running, per section id: the geometry they
        #: were started at and their future. Nothing waits on them but
        #: ``settle_image_corrections`` (submit, the end of a session) and a
        #: fit that reads a trace (``wait_image_job``).
        self.image_jobs: dict[str, tuple[str, Future[dict[str, Any]]]] = {}
        self.image_executor: ThreadPoolExecutor | None = None
        self._deformations: deformation.RecordStore | None = None
        self._state_stamp = _stamp(self.checkpoint_path)
        self._undo_stamp = _stamp(self.undo_path)

    # --- opening ---------------------------------------------------------------

    @classmethod
    def open(
        cls, spec: JobSpec, workspace: Workspace, *, checkpoint_path: str, results_path: str,
    ) -> Job:
        """Resume the checkpoint (``spec.resume``) or ingest the folder; checkpoint.

        A resumed job keeps its undo history; a fresh one starts without one
        (a history file left by an earlier job here is emptied).
        """
        state = None
        if spec.resume:
            data = read_checkpoint(checkpoint_path)
            state = None if data is None else StackState.from_dict(data)
        undo: list[dict[str, Any]] = []
        redo: list[dict[str, Any]] = []
        if state is not None:
            workspace.progress(f"[ingest] resuming from {checkpoint_path}")
            state.spec = spec.to_dict()
            state.submitted = False
            undo, redo = read_history(undo_path_for(checkpoint_path))
        else:
            state = ingest(spec, workspace)
            apply_host_inputs(state, spec)
        job = cls(state, spec, checkpoint_path=checkpoint_path, results_path=results_path,
                  undo=undo, redo=redo)
        if not undo and not redo and os.path.exists(job.undo_path):
            job._save_history()
        job.checkpoint()
        return job

    @property
    def undo_path(self) -> str:
        return undo_path_for(self.checkpoint_path)

    # --- checkpoint and observers ------------------------------------------------

    def checkpoint(self) -> None:
        """Write the state (atomically, versioned), then tell every observer."""
        write_checkpoint(self.state, self.checkpoint_path)
        self._state_stamp = _stamp(self.checkpoint_path)
        self._notify()

    def _notify(self) -> None:
        notify_observers(self.state)
        for observer in list(self.observers):
            try:
                observer(self.state)
            except Exception:
                logger.exception("Job observer raised; ignoring")

    @contextlib.contextmanager
    def observe(self, fn: Callable[[StackState], None]) -> Iterator[None]:
        """Call *fn* with the state after every checkpoint and reload, while open."""
        self.observers.append(fn)
        try:
            yield
        finally:
            self.observers.remove(fn)

    # --- undo ------------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """The state as it stands: take it before a write, hand it to :meth:`commit`."""
        return self.state.to_dict()

    def commit(self, before: dict[str, Any]) -> None:
        """Record one undo step (*before*), clear the redo side, checkpoint."""
        self._push(before)
        self._save_history()
        self.checkpoint()

    def _push(self, before: dict[str, Any]) -> None:
        self.undo_stack.append(before)
        del self.undo_stack[:-UNDO_DEPTH]
        self.redo_stack.clear()

    def undo(self) -> bool:
        """Restore the state before the last step; False when there is none."""
        self.sync()
        if not self.undo_stack:
            return False
        self.redo_stack.append(self.state.to_dict())
        self.state.restore(self.undo_stack.pop())
        self._save_history()
        self.checkpoint()
        return True

    def redo(self) -> bool:
        """Re-apply the step :meth:`undo` reversed; False when there is none."""
        self.sync()
        if not self.redo_stack:
            return False
        self.undo_stack.append(self.state.to_dict())
        del self.undo_stack[:-UNDO_DEPTH]
        self.state.restore(self.redo_stack.pop())
        self._save_history()
        self.checkpoint()
        return True

    def _save_history(self) -> None:
        write_history(self.undo_path, self.undo_stack, self.redo_stack)
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
        state_stamp = _stamp(self.checkpoint_path)
        undo_stamp = _stamp(self.undo_path)
        history_changed = undo_stamp != self._undo_stamp
        if history_changed:
            self.undo_stack, self.redo_stack = read_history(self.undo_path)
            self._undo_stamp = undo_stamp
        if state_stamp == self._state_stamp:
            return None
        try:
            data = read_checkpoint(self.checkpoint_path)
        except (OSError, ValueError):
            logger.warning("State file %s changed but does not read; keeping the job's state",
                           self.checkpoint_path, exc_info=True)
            return None
        self._state_stamp = state_stamp
        if data is None:  # deleted: the next checkpoint writes it again
            return None
        held = self.state.to_dict()
        loaded = StackState.from_dict(data).to_dict()
        if loaded == held:
            return None
        if not history_changed:
            self._push(held)
            self._save_history()
        self.state.restore(loaded)
        logger.info("Reloaded %s: it changed on disk", self.checkpoint_path)
        self._notify()
        return held

    # --- rules ------------------------------------------------------------------------

    @property
    def transform_cap(self) -> int | None:
        """Below the maximum, the most sections one transform call may take."""
        cap = self.spec.transform.max_parallel
        return cap if cap < MAX_PARALLEL_TRANSFORMS else None

    def over_cap(self, requested: int) -> dict[str, Any] | None:
        """The refusal for a transform call naming more sections than allowed."""
        cap = self.transform_cap
        if cap is None or requested <= cap:
            return None
        return {
            "status": "error",
            "error": "TOO_MANY_SECTIONS",
            "max_sections": cap,
            "requested": requested,
        }

    def submit_errors(self, breaks: list[int]) -> dict[str, Any] | None:
        """Every submit gate that applies to this job, in order (:func:`submit_errors`)."""
        return submit_errors(self.state, self.spec, breaks)

    def clear_stale_deformations(self) -> list[str]:
        """Drop deformations whose linear placement changed; checkpoint; their ids.

        A deformation is fitted on top of one linear placement; a write that
        moves the position, orientation, cutting angles or transform makes it
        stale. Called after every write, it lands in that write's undo step
        (the step holds the state from before the write).
        """
        cleared = deformation.clear_stale(self.state)
        if cleared:
            self.checkpoint()
        return cleared

    @property
    def deformations(self) -> deformation.RecordStore:
        """Fitted records, saved under the results folder (made on first use)."""
        if self._deformations is None:
            self._deformations = deformation.RecordStore(
                root=Path(self.results_path).parent / "deformable")
        return self._deformations

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
        changed = False
        for section_id, (fingerprint, future) in list(self.image_jobs.items()):
            changed |= _land(self.state, section_id, fingerprint, _job_result(
                section_id, fingerprint, future, None))
        self.image_jobs.clear()
        if self.image_executor is not None:
            self.image_executor.shutdown(wait=True)
            self.image_executor = None
        if changed:
            self.checkpoint()
        return changed

    def wait_image_job(self, section_id: str, timeout: float) -> bool:
        """Wait up to *timeout* seconds for one section's running correction.

        Records its result the way :meth:`settle_image_corrections` does
        (without the checkpoint) and returns True once nothing is running for
        the section; False when the call is still running at the timeout.
        """
        running = self.image_jobs.get(section_id)
        if running is None:
            return True
        fingerprint, future = running
        try:
            result = _job_result(section_id, fingerprint, future, max(0.0, timeout))
        except TimeoutError:
            return False
        del self.image_jobs[section_id]
        _land(self.state, section_id, fingerprint, result)
        return True

    def missing_image_corrections(
        self, fingerprint: Callable[[str], str],
    ) -> list[dict[str, str]]:
        """Sections without a completed image correction at their current geometry.

        *fingerprint* gives a section's current geometry fingerprint (the
        image-correction bridge's, ``registration_tool.correction_fingerprint``,
        which the caller binds: it lives with the provider code). Running
        corrections are settled first.
        """
        self.settle_image_corrections()
        pending: list[dict[str, str]] = []
        for record in self.state.in_order():
            result = record.image_correction or {}
            try:
                current = fingerprint(record.id)
            except (OSError, ValueError) as exc:
                pending.append({"id": record.id, "reason": str(exc)})
                continue
            if result.get("status") != "ok":
                pending.append({"id": record.id, "reason": "No completed image correction"})
            elif result.get("geometry_fingerprint") != current:
                pending.append({
                    "id": record.id, "reason": "Placement changed since image correction",
                })
        return pending

    # --- results ----------------------------------------------------------------------

    def emit_results(self, progress: Callable[[str], None] | None = None) -> StackState:
        """Write the results JSON (:func:`emit_results`)."""
        return emit_results(self.state, self.results_path, progress)

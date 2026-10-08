"""The job: one stack's state and everything that keeps it consistent.

A :class:`Job` owns the :class:`~langslice.core.state.StackState` and the
:class:`~langslice.core.spec.JobSpec`, its job folder
(:class:`langslice.job.layout.JobLayout`: ``<images>/langslice/``), the
host's locked and damaged sections, undo/redo (persisted in the job folder),
the checkpoint and its observers (:meth:`Job.observe`), the submit gates,
the stale-deformation rule, the deformation record store, the store of the
pictures the model was shown (:class:`langslice.job.views.ViewStore`), the
transform-count cap, the background image-correction jobs and the background
work built on them (:class:`langslice.job.background.BackgroundWork`). ``ingest``,
``apply_host_inputs`` and ``emit_results`` open and close it. It imports the
core and the job folder package (``langslice.job``) only: no agent
framework, no message types, no provider. The doors (the ADK toolbox, the
MCP server) sit on it; a script can drive it directly.

What a Job never holds: the look-before-commit gates (``look`` before
``position_sections``, a positioning look before ``submit``) and the
model-delivery bookkeeping (which pictures a model has received). Those are
the tool doors' (:class:`langslice.doors.tools.toolbox.ToolBox`); a library call
is never gated.

**Undo.** One pattern: ``before = job.snapshot()`` before a write, then
``job.commit(before)`` once it succeeded (one undo step, then the
checkpoint). A step is the whole state, so undo restores everything on it,
the deformation references in the section records included (the records
themselves are content-addressed on disk and never change). The history is
the job folder's ``history/`` (:mod:`langslice.job.history`: an index and one
file per step), depth :data:`UNDO_DEPTH`; a resumed job reads it back, a
fresh one starts empty.

**Lean.** A job whose spec's ``output_level`` is "lean" keeps the results
only: no pictures (:class:`~langslice.job.views.DiscardedViews`), no undo
history on disk (undo and redo still work in the process that made the
steps), no ``logs/``. The state, ``registration.json``, the maps, the
deformation records, the image-model traces and the exports are written as
in a full job.

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
import json
import logging
import os
from collections.abc import Callable, Collection, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from langslice.core import deformation
from langslice.core.discovery import discover_slices
from langslice.core.spec import MAX_PARALLEL_TRANSFORMS, JobSpec, supplied_angles
from langslice.core.state import IDENTITY_KNOBS, ROTATIONS, SliceState, StackState
from langslice.job import formats
from langslice.job.background import BackgroundWork, in_worker
from langslice.job.checkpoint import (
    read_checkpoint,
    relative_to,
    state_paths,
    write_checkpoint,
    write_json_atomic,
)
from langslice.job.history import UNDO_DEPTH as UNDO_DEPTH
from langslice.job.history import History
from langslice.job.layout import (
    JobLayout,
    check_owner,
    locate_job_folder,
    write_job_file,
)
from langslice.job.lock import FolderLock
from langslice.job.views import DiscardedViews, ViewStore

if TYPE_CHECKING:
    from langslice.core.workspace import Workspace

logger = logging.getLogger(__name__)

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


def _named(spec: JobSpec, key: str) -> set[str]:
    """The section filenames ``spec.inputs[key]`` names (a list, or a mapping's keys)."""
    return {str(name) for name in (spec.inputs or {}).get(key) or []}


def locked_ids(spec: JobSpec) -> set[str]:
    """Sections whose flip, rotation and transform the host locked."""
    return _named(spec, "locked")


def keep_warp_ids(spec: JobSpec) -> set[str]:
    """Sections whose own deformation in the host the agent may not replace."""
    return _named(spec, "keep_warp")


def nonlinear_skip_ids(spec: JobSpec) -> set[str]:
    """Sections the user left out of the Nonlinear task."""
    return _named(spec, "nonlinear_skip")


def nonlinear_exempt_ids(spec: JobSpec) -> set[str]:
    """Sections the host kept out of the Nonlinear task (``keep_warp`` and
    ``nonlinear_skip``): its verbs refuse them and its submit gates skip them."""
    return keep_warp_ids(spec) | nonlinear_skip_ids(spec)


#: Why Nonlinear refuses a section the host kept out of it: code, message.
KEEPS_HOST_WARP = ("KEEPS_HOST_WARP", "This section carries the user's own deformation in the "
                   "host, which the user did not let the agent overwrite.")
NONLINEAR_SKIPPED = ("NONLINEAR_SKIPPED", "The user chose not to have this section aligned "
                     "first, so the Nonlinear task leaves it alone.")


def host_damaged_ids(spec: JobSpec) -> set[str]:
    """Sections the host gave a damage note (``inputs.damaged``); the note
    stays first in the section's ``damage_note``."""
    return _named(spec, "damaged")


def host_transform() -> dict[str, Any]:
    """The identity transform a locked section carries."""
    return {
        "kind": HOST_TRANSFORM_KIND,
        "params": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "physical": {**IDENTITY_KNOBS, "shear": 0.0, "pivot": [0.5, 0.5]},
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


def _section(state: StackState, key: str, name: Any) -> SliceState:
    """The section *name* of ``inputs.<key>``; ``ValueError`` for one not here."""
    record = state.by_id(str(name))
    if record is None:
        raise ValueError(f"inputs.{key} names an unknown section: {name!r}")
    return record


def apply_host_inputs(state: StackState, spec: JobSpec) -> None:
    """Write the host's answers for the tasks that are switched off.

    Order arrives as a list of filenames, positions as a filename -> mm
    mapping, angles as ``{"pitch": deg, "yaw": deg}`` for the whole stack or
    a filename -> ``{"pitch": deg, "yaw": deg}`` mapping, each section's own
    plane kept as supplied (:func:`langslice.core.spec.supplied_angles`; a
    section it does not name keeps the flat plane), orientation as a
    filename -> ``{"flip": bool, "rotation_deg": 0|90|180|270}`` mapping (a
    missing key leaves that part as it is), damage as a filename -> note
    mapping, and transforms as filename -> stored transform dictionaries.
    Anything the host supplies for a task that IS on is applied too — it is
    a starting point, not a constraint. A ``damaged`` note becomes the
    section's ``damage_note`` (shown in status, so the agent can mark the
    regions it names); it does not make the section damaged, and it stays
    first in the note whatever the agent marks. ``locked`` sections (a
    list of filenames) are a constraint: the agent cannot change their flip,
    rotation or transform; a locked section
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
            record = _section(state, "positions", name)
            record.position_mm = float(value)
            applied += 1
        state.notes.append(f"inputs: {applied} position(s) set by the host")

    stack_angles, section_angles = supplied_angles(inputs.get("angles"))
    if stack_angles is not None:
        pitch, yaw = stack_angles
        state.cutting_angles_deg = {"pitch": pitch, "yaw": yaw}
        state.notes.append(
            f"inputs: cutting angles set by the host "
            f"(pitch {pitch:.2f}, yaw {yaw:.2f})"
        )
    if section_angles:
        for name, (pitch, yaw) in section_angles.items():
            record = _section(state, "angles", name)
            record.cutting_angles_deg = {"pitch": pitch, "yaw": yaw}
        state.notes.append(
            f"inputs: cutting angles of {len(section_angles)} section(s) set by the host"
        )

    orientation = inputs.get("orientation") or {}
    if orientation:
        if not isinstance(orientation, dict):
            raise ValueError("inputs.orientation must map section filenames to "
                             "{flip, rotation_deg}")
        for name, value in orientation.items():
            record = _section(state, "orientation", name)
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
            record = _section(state, "transforms", name)
            if not isinstance(value, dict):
                raise ValueError(f"inputs.transforms[{name!r}] must be a transform dictionary")
            record.transform = copy.deepcopy(value)
        state.notes.append(f"inputs: {len(transforms)} transform(s) set by the host")

    damaged = inputs.get("damaged") or {}
    if damaged:
        # A host's word on a section's damage (a note, no regions): the agent
        # reads it in status and marks the regions it names.
        for name, note in damaged.items():
            record = _section(state, "damaged", name)
            record.damage_note = str(note or "")
        state.notes.append(f"inputs: {len(damaged)} damage note(s) from the host")

    locked = inputs.get("locked") or []
    if locked:
        if not isinstance(locked, (list, tuple)):
            raise ValueError("inputs.locked must be a list of section filenames")
        for name in locked:
            record = _section(state, "locked", name)
            if record.transform is None:
                record.transform = host_transform()
        state.notes.append(
            f"inputs: {len(locked)} section(s) locked by the host (in-plane alignment done)"
        )

    for key, note in (("keep_warp", "keep their own deformation in the host"),
                      ("nonlinear_skip", "left out of the Nonlinear task by the host")):
        names = inputs.get(key) or []
        if not names:
            continue
        if not isinstance(names, (list, tuple)):
            raise ValueError(f"inputs.{key} must be a list of section filenames")
        for name in names:
            _section(state, key, name)
        state.notes.append(f"inputs: {len(names)} section(s) {note}")


#: ``SliceState.position_source`` of a starting position LangSlice gave a
#: section at ingest (:func:`default_positions`): the submit gate
#: ``MISSING_POSITIONS`` refuses a section still there.
DEFAULT_POSITION = "default"


def default_positions(state: StackState, workspace: Workspace) -> list[str]:
    """Give every section without a position a starting one, marked
    :data:`DEFAULT_POSITION`; return their ids.

    Stack order is the order the images were found (``index_original``).
    With no position supplied, the sections are spread across the atlas's
    range in that order: at the stack's interval, centred, when the stack
    fits in the range at it, else evenly over the whole range. Between
    supplied positions a section gets the linear interpolation by stack
    order; beyond them, the nearest supplied one continued at the stack's
    interval (the median supplied step when no interval is set) in the
    direction the supplied ones run. Every starting position lies in the
    atlas's range.
    """
    ordered = sorted(state.slices, key=lambda record: record.index_original)
    missing = [index for index, record in enumerate(ordered) if record.position_mm is None]
    if not missing:
        return []
    low, high = (float(value) for value in workspace.position_range)
    count = len(ordered)
    interval = float(state.interval_mm)
    known = [(index, float(record.position_mm)) for index, record in enumerate(ordered)
             if record.position_mm is not None]
    values: dict[int, float] = {}
    if not known:
        if interval > 0 and interval * (count - 1) <= high - low:
            first = (low + high) / 2 - interval * (count - 1) / 2
            values = {index: first + interval * index for index in range(count)}
        else:
            values = {index: low + (high - low) * (index + 0.5) / count for index in range(count)}
    else:
        steps = [(b - a) / (j - i) for (i, a), (j, b) in zip(known, known[1:], strict=False)]
        trend = float(np.median(steps)) if steps else 0.0
        step = interval if interval > 0 else abs(trend)
        step = step if trend >= 0 else -step
        for index in missing:
            before = [(i, v) for i, v in known if i < index]
            after = [(i, v) for i, v in known if i > index]
            if before and after:
                (i, a), (j, b) = before[-1], after[0]
                values[index] = a + (b - a) * (index - i) / (j - i)
            elif before:
                i, a = before[-1]
                values[index] = a + step * (index - i)
            else:
                j, b = after[0]
                values[index] = b - step * (j - index)
    placed: list[str] = []
    for index in missing:
        record = ordered[index]
        record.position_mm = round(min(high, max(low, values[index])), 4)
        record.position_source = DEFAULT_POSITION
        placed.append(record.id)
    state.notes.append(f"ingest: {len(placed)} section(s) given evenly spaced starting "
                       "positions")
    return placed


def clear_default_marks(before: dict[str, Any], state: StackState) -> list[str]:
    """Drop the :data:`DEFAULT_POSITION` mark of every section whose position
    differs from *before* (a state's dict); return their ids. A written
    position is the writer's own, whatever wrote it."""
    held = {row.get("id"): row.get("position_mm") for row in before.get("slices", [])}
    cleared: list[str] = []
    for record in state.slices:
        if record.position_source == DEFAULT_POSITION and (
                record.id not in held or held[record.id] != record.position_mm):
            record.position_source = ""
            cleared.append(record.id)
    return cleared


class InputsChanged(ValueError):
    """A job folder's checkpoint was made from other supplied inputs
    (``JobSpec.inputs``) than the spec opening it: resuming would keep the
    old ones and drop the new without a word (:meth:`Job.open`)."""


#: How a caller starts over instead of resuming (the CLI flag and the spec field).
START_FRESH = "start a fresh job instead (`--fresh` on the command line, resume=False in a spec)"


def changed_inputs(saved_spec: Any, spec: JobSpec) -> list[str]:
    """The ``inputs`` keys whose values differ between a checkpoint's spec
    (*saved_spec*, the dict it stores) and *spec*; none when the checkpoint
    records no spec inputs."""
    if not isinstance(saved_spec, dict) or "inputs" not in saved_spec:
        return []

    def plain(value: Any) -> dict[str, Any]:
        return json.loads(json.dumps(value or {}, sort_keys=True, default=str))

    old, new = plain(saved_spec.get("inputs")), plain(spec.inputs)
    return sorted(key for key in set(old) | set(new) if old.get(key) != new.get(key))


def inputs_changed(folder: str | os.PathLike[str], changed: list[str]) -> InputsChanged:
    """The refusal for resuming the job in *folder* with other *changed* inputs."""
    return InputsChanged(
        f"The job in {folder} was made from other supplied inputs "
        f"({', '.join(changed)} differ); resuming it would ignore the new "
        f"ones. Open it with the inputs it was made from, or {START_FRESH}."
    )


def emit_results(
    state: StackState, results_path: str, progress: Callable[[str], None] | None = None,
) -> StackState:
    """Write the results JSON — the same shape as the checkpoint, unversioned
    (its paths relative to the job folder, as the checkpoint's)."""
    write_json_atomic(results_path, state.to_dict())
    if progress is not None:
        progress(f"[emit] results -> {results_path}")
    return state


# --- the submit gates ------------------------------------------------------------


def missing_positions(state: StackState) -> dict[str, Any] | None:
    """``None`` when every section has a position of its own, else the
    rejection: a section without one, or still at the starting position
    LangSlice gave it (:data:`DEFAULT_POSITION`, listed in ``at_default``)."""
    ordered = state.in_order()
    missing = [record.id for record in ordered if record.position_mm is None
               or record.position_source == DEFAULT_POSITION]
    if not missing:
        return None
    at_default = [record.id for record in ordered if record.position_mm is not None
                  and record.position_source == DEFAULT_POSITION]
    return {
        "status": "error",
        "error": "MISSING_POSITIONS",
        "missing_ids": missing,
        **({"at_default": at_default} if at_default else {}),
        "message": (
            f"{len(missing)} of {len(state.slices)} section(s) have no position of their "
            "own" + (f" ({len(at_default)} still at the evenly spaced starting position "
                     "they were given)" if at_default else "")
            + "; every section needs one, damaged ones included."
        ),
    }


def strict_interval_error(state: StackState, breaks: list[int]) -> dict[str, Any] | None:
    """Every consecutive spacing within 10% of the interval, no breaks."""
    if breaks:
        return {
            "status": "error",
            "error": "STRICT_INTERVAL",
            "reported_breaks": [record.id for record in state.in_order()
                                if record.index_corrected in set(breaks)],
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

    A break at the section of corrected index *i* (``submit`` names it by
    filename) claims the gap between *i-1* and *i* is larger than the rest
    of the stack's; the positions on the state make that
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
                    **({"id": ordered[position].id} if position is not None else {}),
                    "error": "NOT_A_GAP",
                    "reason": (
                        "no section comes before it in the stack's order; a break "
                        "names the section AFTER the gap."
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
                "id": after.id,
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
            "positions you wrote. A break is accepted only where the "
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


def missing_deformations(
    state: StackState, exempt: frozenset[str] | set[str] = frozenset(),
    left_linear: Collection[str] = (),
) -> dict[str, Any] | None:
    """``None`` when every section carries a deformation, or a keep_linear
    reason, at its current linear placement; else the rejection. The
    sections in *exempt* (kept out of Nonlinear by the host,
    :func:`nonlinear_exempt_ids`) need neither, and the sections in
    *left_linear* (``submit``'s ``left_linear``: left without a deformation,
    each with its reason) count as covered."""
    missing: list[dict[str, str]] = []
    covered = set(exempt) | {str(name) for name in left_linear}
    for record in state.in_order():
        if record.id in covered:
            continue
        held = record.deformation or {}
        if not held:
            missing.append({"id": record.id, "reason": "no deformation, and not in left_linear"})
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
            "current placement. Fit a deformation on each, or list it in submit's "
            "left_linear with the reason its linear placement stands."
        ),
    }


def submit_errors(
    state: StackState, spec: JobSpec, breaks: list[int], left_linear: Collection[str] = (),
) -> dict[str, Any] | None:
    """Every gate that applies to this run, in order. *left_linear*: the
    sections ``submit`` leaves without a deformation (:func:`missing_deformations`)."""
    if spec.has("position"):
        refusal = (
            missing_positions(state)
            or (
                strict_interval_error(state, breaks)
                if spec.position.strict_interval
                else unsupported_breaks(state, breaks)
            )
        )
        if refusal is not None:
            return refusal
    if spec.has("transform"):
        refusal = missing_transforms(state)
        if refusal is not None:
            return refusal
    if spec.has("nonlinear"):
        return missing_deformations(state, nonlinear_exempt_ids(spec), left_linear)
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
    checkpoint), or around a state already in hand
    (``Job(state, spec, layout=...)``: an empty history, nothing read or
    written until the first operation; the files as they stand on disk at
    that moment count as already seen). *results_path* None is the job
    folder's ``exports/linear_results.json``.
    """

    def __init__(
        self,
        state: StackState,
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
        #: The job folder keeps the results only (``JobSpec.output_level``
        #: "lean"): no pictures, no undo history on disk, no event log.
        self.lean = spec.lean
        #: The pictures the model was shown, saved with their layers (none
        #: in a lean job).
        self.views = DiscardedViews(layout) if self.lean else ViewStore(layout)
        #: The job folder's write lock (:meth:`writing`), across processes.
        self.lock = FolderLock(layout.folder)
        #: Sections whose flip, rotation and transform the host locked.
        self.locked = frozenset(locked_ids(spec))
        #: Sections whose own deformation in the host stays (``ants_syn``
        #: refuses them, ``KEEPS_HOST_WARP``).
        self.keep_warp = frozenset(keep_warp_ids(spec))
        #: Sections the user left out of Nonlinear (``ants_syn`` and
        #: ``trace_borders`` refuse them, ``NONLINEAR_SKIPPED``).
        self.nonlinear_skip = frozenset(nonlinear_skip_ids(spec))
        #: Sections the host gave a damage note (``inputs.damaged``); the note
        #: stays first in the section's ``damage_note``.
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
        self._background: BackgroundWork | None = None
        self._deformations: deformation.RecordStore | None = None
        #: False: nothing this job holds is written (the state, the history,
        #: the pictures); a dry run's job (the CLI's ``--dry-run``).
        self.persist = True
        #: The workspace the job's sections and atlas are read through
        #: (set by :meth:`open`, :meth:`load` and the tool door): every
        #: checkpoint rewrites ``registration.json`` through it
        #: (:func:`langslice.job.formats.write_registration`).
        self.workspace: Workspace | None = None
        self._state_stamp = _stamp(self.checkpoint_path)
        self._undo_stamp = _stamp(self.undo_path)
        self.wire_views()

    def wire_views(self) -> None:
        """Number each saved picture's ``step`` with the undo history's depth
        (call again after replacing :attr:`views`)."""
        self.views.step_source = lambda: len(self.undo_stack)

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
        checkpoint written. A resumed job
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
        layout.ensure(lean=spec.lean)
        lock = FolderLock(layout.folder)
        with lock.held():  # read and first checkpoint as one write
            state = None
            resumed = False
            if spec.resume:
                data = read_checkpoint(str(layout.state_file))
                state = None if data is None else StackState.from_dict(data)
                changed = [] if state is None else changed_inputs(state.spec, spec)
                if changed:
                    raise inputs_changed(layout.folder, changed)
            write_job_file(layout, spec=spec.to_dict())
            history = History(layout.history_dir)
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
                if spec.has("position"):
                    default_positions(state, workspace)
                if history.exists():
                    history.load()  # read before it is emptied: never delete unread steps
            if history.problem is not None:
                workspace.progress(f"[job] {history.problem}")
            job = cls(state, spec, layout=layout, results_path=results_path,
                      undo=undo, redo=redo, history=history)
            job.lock = lock
            job.workspace = workspace
            if not undo and not redo and history.exists():
                job._save_history()  # a fresh job empties a history it read; never another
            job.checkpoint()
        if not spec.lean:
            layout.log_event("open", resumed=resumed)
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
        data = read_checkpoint(str(layout.state_file))
        if data is None:
            raise FileNotFoundError(f"No job state in {layout.folder}")
        history = History(layout.history_dir)
        undo, redo = history.load()
        job = cls(StackState.from_dict(data), spec, layout=layout, results_path=results_path,
                  undo=undo, redo=redo, history=history)
        job.persist = persist
        job.workspace = workspace
        if not persist:
            job.views = DiscardedViews(layout)
            job.wire_views()
        return job

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

    def portable(self, correction: dict[str, Any]) -> dict[str, Any]:
        """An image correction's record with its paths relative to the job folder."""
        converted = state_paths({"slices": [{"image_correction": correction}]},
                                relative_to(self.folder))
        return converted["slices"][0]["image_correction"]

    # --- checkpoint and observers ------------------------------------------------

    def checkpoint(self) -> None:
        """Write the state (atomically, versioned) and its public rendering,
        ``registration.json`` (:func:`langslice.job.formats.write_registration`;
        a failure there is logged, never raised), then tell every observer."""
        if not self.persist:
            return
        with self.lock.held():
            write_checkpoint(self.state, self.checkpoint_path)
            self._state_stamp = _stamp(self.checkpoint_path)
            formats.write_registration(self.layout, self.state, self.workspace)
        self._notify()

    @contextlib.contextmanager
    def writing(self) -> Iterator[dict[str, Any] | None]:
        """Hold the job folder's write lock and bring the state up to date.

        Every write goes lock -> :meth:`sync` (reload what another writer
        saved) -> apply -> :meth:`commit` -> unlock, so a running agent, CLI
        calls and scripts on one folder never overwrite each other. Yields
        what :meth:`sync` returned (the state as held before a reload, or
        None). Reentrant in this thread. A long operation computes outside
        it, from the state it read, and enters it only to apply: after the
        sync it checks that each section's inputs are unchanged
        (``ops.inputs``) and refuses a section whose inputs moved
        (``STALE_INPUT``). The section records are new objects after a
        reload: resolve them by id inside the block. A thread running the
        job's background work gives way to a lock holder that waits for that
        work (:mod:`langslice.job.background`: ``LockYielded``).
        """
        with self.lock.held(self.background.gives_way if in_worker() else None):
            yield self.sync()

    def _notify(self) -> None:
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
        """Record one undo step (*before*), clear the redo side, checkpoint.
        Call it inside :meth:`writing` (with *before* taken there). A section
        whose position the step changed loses its :data:`DEFAULT_POSITION`
        mark (:func:`clear_default_marks`)."""
        with self.lock.held():
            clear_default_marks(before, self.state)
            self._push(before)
            self._save_history()
            self.checkpoint()

    def _push(self, before: dict[str, Any]) -> None:
        self.undo_stack.append(before)
        del self.undo_stack[:-UNDO_DEPTH]
        self.redo_stack.clear()

    def undo(self) -> bool:
        """Restore the state before the last step; False when there is none."""
        with self.writing():
            if not self.undo_stack:
                return False
            self.redo_stack.append(self.state.to_dict())
            self.state.restore(self.undo_stack.pop())
            self._save_history()
            self.checkpoint()
            return True

    def redo(self) -> bool:
        """Re-apply the step :meth:`undo` reversed; False when there is none."""
        with self.writing():
            if not self.redo_stack:
                return False
            self.undo_stack.append(self.state.to_dict())
            del self.undo_stack[:-UNDO_DEPTH]
            self.state.restore(self.redo_stack.pop())
            self._save_history()
            self.checkpoint()
            return True

    def _save_history(self) -> None:
        if not self.persist or self.lean:  # a lean job's undo lives in this process only
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
        state_stamp = _stamp(self.checkpoint_path)
        undo_stamp = _stamp(self.undo_path)
        history_changed = undo_stamp != self._undo_stamp
        if history_changed:
            self.undo_stack, self.redo_stack = self.history.load()
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

    def submit_errors(
        self, breaks: list[int], left_linear: Collection[str] = (),
    ) -> dict[str, Any] | None:
        """Every submit gate that applies to this job, in order (:func:`submit_errors`)."""
        return submit_errors(self.state, self.spec, breaks, left_linear)

    def nonlinear_refusal(self, section_id: str) -> tuple[str, str] | None:
        """``(code, message)`` when the host kept *section_id* out of the
        Nonlinear task (:data:`KEEPS_HOST_WARP`, :data:`NONLINEAR_SKIPPED`),
        else None."""
        if section_id in self.keep_warp:
            return KEEPS_HOST_WARP
        if section_id in self.nonlinear_skip:
            return NONLINEAR_SKIPPED
        return None

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

    def emit_results(self, progress: Callable[[str], None] | None = None) -> StackState:
        """Write the results JSON (:func:`emit_results`); every picture saved."""
        self.views.flush()
        return emit_results(self.state, self.results_path, progress)

    def close(self) -> None:
        """Finish the job's background work (:attr:`background`), then its
        background writes (the pictures); the job stays usable."""
        if self._background is not None:
            self._background.close()
        self.views.flush()

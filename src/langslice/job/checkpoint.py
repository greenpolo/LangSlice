"""JSON checkpoint for a linear run — the record of truth.

Every write saves the full :class:`StackState` here, atomically (temp file +
rename) so a crash mid-write cannot leave a truncated file. Resume = load the
checkpoint and re-seed the agent with the state it had. The job
(:class:`langslice.job.job.Job`) is the one writer during a run.

The file carries :data:`STATE_FORMAT_VERSION` under ``format_version`` next to
the state's own fields. A checkpoint without it predates versioning (version
0); :func:`upgrade_state` converts older versions (version 2 made the paths
on the section records relative to the job folder; version 3 gave every
section its own cutting angles, :func:`section_angles`) and refuses a file
from a newer LangSlice, never guessing at it.

Every checkpoint funnels through :func:`save_checkpoint` (or the job's
:meth:`~langslice.job.job.Job.checkpoint`, which calls
:func:`write_checkpoint` and :func:`notify_observers`), which makes it the one
place a host can watch a run live: :func:`observe_checkpoints` registers a
callback fired with the state after every write, which is what the ABBA
mirror (:mod:`langslice.hosts.integrations.abba_linear`) attaches to.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from collections.abc import Callable, Iterator
from typing import Any

from langslice.core.state import StackState

logger = logging.getLogger(__name__)

#: The checkpoint of the old layout, beside the images (before the job
#: folder; :mod:`langslice.job.migrate` moves it to ``langslice/state.json``).
CHECKPOINT_FILENAME = "linear_state.json"

#: The key a job-folder file carries its format version under.
FORMAT_KEY = "format_version"
#: The checkpoint's format. 1 (2026-10-03): the first versioned checkpoint,
#: the same fields as the unversioned ones before it. 2 (2026-10-03, the job
#: folder): every path on a section record (a deformation's ``record``, an
#: image correction's ``artifact_dir`` and ``artifact_paths``) is relative to
#: the job folder (:func:`state_paths`). 3 (2026-10-04): cutting angles are
#: per section (``SliceState.cutting_angles_deg``). A state whose sections
#: share one angle is written exactly as before (the stack's
#: ``cutting_angles_deg``, nothing on the rows); one whose sections differ
#: (a registration supplied per section) writes the stack's as null and each
#: row's own. An older LangSlice refuses format 3, so it never reads a
#: per-section state as flat; older states load with every section carrying
#: the stack's angle (:func:`section_angles`).
STATE_FORMAT_VERSION = 3

#: Called with the state after every atomic write. A list, not a single slot,
#: so nested runs (tests, a resumed session) can each hold their own observer
#: without clobbering another's.
_observers: list[Callable[[StackState], None]] = []


def default_checkpoint_path(image_folder: str) -> str:
    """``<image_folder>/langslice/state.json``: the job folder's checkpoint."""
    from langslice.job.layout import JobLayout

    return str(JobLayout.for_images(image_folder).state_file)


def write_json_atomic(path: str, data: Any, *, indent: int | None = 2) -> None:
    """Write *data* as JSON to *path* through a temporary file and a rename."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            if indent is None:
                json.dump(data, handle, separators=(",", ":"))
            else:
                json.dump(data, handle, indent=indent)
        # mkstemp creates 0600; a job file is an ordinary output file.
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def write_checkpoint(state: StackState, path: str) -> None:
    """Write *state* to *path* atomically, versioned; observers are not told."""
    write_json_atomic(path, {FORMAT_KEY: STATE_FORMAT_VERSION, **state.to_dict()})


def notify_observers(state: StackState) -> None:
    """Call every observer with *state*; one that raises is logged and skipped."""
    for observer in list(_observers):
        try:
            observer(state)
        except Exception:
            logger.exception("Checkpoint observer raised; ignoring")


def save_checkpoint(state: StackState, path: str) -> None:
    """Write *state* to *path* atomically, then notify any observers.

    An observer that raises is logged and skipped — a display glitch (the
    ABBA mirror hiccuping on a JPype call) must never break the agent's
    write, which has already happened by the time observers run.
    """
    write_checkpoint(state, path)
    notify_observers(state)


@contextlib.contextmanager
def observe_checkpoints(fn: Callable[[StackState], None]) -> Iterator[None]:
    """Call *fn* with the state after every :func:`save_checkpoint` write.

    A host wanting a live view of a run (the ABBA mirror) wraps its run in
    this; *fn* fires inline, on the writer's thread, right after the atomic
    replace — see :func:`save_checkpoint` for what happens if it raises.
    """
    _observers.append(fn)
    try:
        yield
    finally:
        _observers.remove(fn)


def state_paths(
    data: dict[str, Any], convert: Callable[[str], str],
) -> dict[str, Any]:
    """*data* (a state's fields) with every stored path passed through *convert*.

    The paths a state holds: each section's deformation ``record`` and its
    image correction's ``artifact_dir`` and ``artifact_paths``. Everything
    else is returned as it is (the input is not changed).
    """
    slices = []
    for row in data.get("slices") or []:
        if not isinstance(row, dict):
            slices.append(row)
            continue
        row = dict(row)
        held = row.get("deformation")
        if isinstance(held, dict) and isinstance(held.get("record"), str):
            row["deformation"] = {**held, "record": convert(held["record"])}
        correction = row.get("image_correction")
        if isinstance(correction, dict):
            correction = dict(correction)
            if isinstance(correction.get("artifact_dir"), str):
                correction["artifact_dir"] = convert(correction["artifact_dir"])
            paths = correction.get("artifact_paths")
            if isinstance(paths, dict):
                correction["artifact_paths"] = {
                    key: convert(value) if isinstance(value, str) else value
                    for key, value in paths.items()}
            row["image_correction"] = correction
        slices.append(row)
    return {**data, "slices": slices} if "slices" in data else dict(data)


def section_angles(data: dict[str, Any]) -> dict[str, Any]:
    """*data* (a state's fields before format 3) with every section row
    carrying the stack's cutting angles (``cutting_angles_deg``; flat when
    the stack has none). A row that already has its own keeps it; the input
    is not changed."""
    stack = data.get("cutting_angles_deg")
    angles = dict(stack) if isinstance(stack, dict) else {"pitch": 0.0, "yaw": 0.0}
    if "slices" not in data:
        return dict(data)
    rows = [
        {**row, "cutting_angles_deg": dict(angles)}
        if isinstance(row, dict) and not isinstance(row.get("cutting_angles_deg"), dict)
        else row
        for row in data.get("slices") or []
    ]
    return {**data, "slices": rows}


def relative_to(root: str | os.PathLike[str]) -> Callable[[str], str]:
    """A path converter: absolute paths inside *root* become relative to it
    (POSIX separators); every other path is kept as it is."""
    bases = list(dict.fromkeys((os.path.abspath(root), os.path.realpath(root))))

    def convert(value: str) -> str:
        if not os.path.isabs(value):
            return value
        for base in bases:
            for spelled in dict.fromkeys((os.path.abspath(value), os.path.realpath(value))):
                if os.path.commonpath([spelled, base]) == base:
                    return os.path.relpath(spelled, base).replace(os.sep, "/")
        return value

    return convert


def upgrade_state(
    data: dict[str, Any], root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """A checkpoint's fields at the current format, without the version key.

    Unversioned (version 0) checkpoints have the version-1 fields already.
    Version 1 stored absolute paths: those inside *root* (the job folder)
    become relative to it, the rest stay as they are (the job folder's
    migration, :mod:`langslice.job.migrate`, moves the old folders first).
    Before version 3 the cutting angles were the stack's alone: every
    section is given them (:func:`section_angles`).
    Raises ``ValueError`` for a version this LangSlice does not know.
    """
    version = data.get(FORMAT_KEY, 0)
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise ValueError(f"Unreadable checkpoint format_version {version!r}")
    if version > STATE_FORMAT_VERSION:
        raise ValueError(
            f"This checkpoint was written by a newer LangSlice (format {version}; this "
            f"version reads up to {STATE_FORMAT_VERSION}). Update LangSlice to open it.")
    fields = {key: value for key, value in data.items() if key != FORMAT_KEY}
    if version < 2 and root is not None:
        fields = state_paths(fields, relative_to(root))
    if version < 3:
        fields = section_angles(fields)
    return fields


def read_checkpoint(path: str, root: str | os.PathLike[str] | None = None) -> dict[str, Any] | None:
    """A checkpoint's state fields, upgraded; ``None`` if there is no file.

    *root* is the job folder its paths are relative to (default: the
    checkpoint's own folder, where the job folder keeps it).
    """
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a stack state")
    return upgrade_state(data, root if root is not None else os.path.dirname(
        os.path.abspath(path)))


def load_checkpoint(path: str) -> StackState | None:
    """Load a checkpoint, or ``None`` if there isn't one at *path*."""
    data = read_checkpoint(path)
    return None if data is None else StackState.from_dict(data)

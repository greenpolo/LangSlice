"""Where a job's files live: the job folder beside the images, and its names.

One stack, one job folder: ``<images>/langslice/``. Every host puts it next
to the images it hands LangSlice (the CLI's folder, a Claude job's folder,
the snapshots ABBA exports), so a job travels with its images and is found
without an id. Inside::

    job.lock             the write lock (:mod:`langslice.job.lock`)
    job.json             settings (the JobSpec) + format_version; a saved
                         host job's own fields under "host"
    state.json           the state checkpoint (StackState, versioned)
    registration.json    its public rendering (:mod:`langslice.job.formats`)
    history/             undo/redo: index.json + one file per step
    sections/<name>/     per section (<name>: the image filename's stem)
        deformable/<key>/          applied deformation records
        image_correction/<key>/    trace_borders calls and their attempts
        views/<seq>_<tool>_<mode>/ pictures of this section the model saw
        coords.tif, labels.tif, labels_fiji.tif, labels.csv, tissue.png,
        residual.tif, maps.json    the section's maps (at submit and by
                                   ``export_maps``)
    views/<seq>_<tool>_<mode>/     pictures of several sections (stack level)
    views.jsonl          append-only index of every saved picture
    views.seq            the picture and call numbers handed out (+ views.seq.lock)
    exports/             linear_results.json, a host's result.json,
                         quicknii.json and visualign.json
    logs/                events.jsonl (opens), runs/ (the agent CLI's
                         background runs), calls.jsonl (the agent CLI's calls)
    prompt.txt           a saved Claude job's copy prompt
    AGENTS.md, CLAUDE.md the reference card for coding agents
    BRIEF.md             the agent CLI's brief

Every path a job file stores is relative to the job folder
(:meth:`JobLayout.relative`), so the folder can move with its images.

:func:`locate_job_folder` picks the folder: ``JobSpec.job_dir`` when set
(exactly there), else ``<images>/langslice``, else, when the image folder
cannot be written, ``~/.langslice/jobs/<id>/`` (said once, recorded in the
id's index entry). :func:`check_owner` refuses a folder holding another
image folder's job.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: The job folder's name, next to the images.
JOB_DIRNAME = "langslice"
JOB_FILE = "job.json"
STATE_FILE = "state.json"
HISTORY_DIR = "history"
SECTIONS_DIR = "sections"
VIEWS_DIR = "views"
VIEWS_INDEX = "views.jsonl"
EXPORTS_DIR = "exports"
LOGS_DIR = "logs"
EVENTS_FILE = "events.jsonl"
PROMPT_FILE = "prompt.txt"
#: Under ``exports/``: the run's result (the state's shape, unversioned) and
#: a host job's final result (state + host updates).
RESULTS_FILE = "linear_results.json"
HOST_RESULT_FILE = "result.json"
#: Under ``sections/<name>/``.
DEFORMABLE_DIR = "deformable"
IMAGE_CORRECTION_DIR = "image_correction"
SECTION_VIEWS_DIR = "views"
#: The public rendering of the state (:mod:`langslice.job.formats`, which
#: also names each section's map files).
REGISTRATION_FILE = "registration.json"

#: The job folder's format, carried by ``job.json``: this layout.
JOB_FORMAT_VERSION = 1
FORMAT_KEY = "format_version"

_UNSAFE = re.compile(r"[\x00-\x1f/\\:*?\"<>|]")


def job_folder_for(image_folder: str | os.PathLike[str]) -> Path:
    """``<image_folder>/langslice`` (absolute)."""
    return Path(os.path.abspath(os.fspath(image_folder))) / JOB_DIRNAME


def writable(folder: Path, image_folder: Path) -> bool:
    """Whether the job folder can be made and written, without making it.

    An existing folder is probed with a temporary file; a missing one asks
    whether *image_folder* (its parent) can take a new folder.
    """
    if folder.exists():
        try:
            handle, probe = tempfile.mkstemp(dir=folder, prefix=".write-probe-")
            os.close(handle)
            os.unlink(probe)
            return True
        except OSError:
            return False
    return os.access(image_folder, os.W_OK | os.X_OK)


def locate_job_folder(
    image_folder: str | os.PathLike[str],
    job_dir: str | os.PathLike[str] | None = None,
    *,
    emit: Callable[[str], None] | None = None,
    root: Path | None = None,
    job_id: str | None = None,
    register: bool = True,
) -> tuple[Path, dict[str, Any] | None]:
    """``(job folder, fallback)`` for a job on *image_folder*.

    *job_dir* given: exactly there (``JobSpec.job_dir``, ``--job-dir``).
    Otherwise ``<images>/langslice`` (:func:`job_folder_for`), unless that
    cannot be created or written (a read-only image folder): then
    ``<root>/<id>/`` (``~/.langslice/jobs/``), the id *job_id* or one fixed
    by the image folder's path (:func:`langslice.job.index.folder_id`, so a
    reopen finds it), said once through *emit* and, with *register*,
    recorded in the id's index entry. *fallback* is ``{"image_folder",
    "reason"}`` then, None otherwise.
    """
    from langslice.job import index

    if job_dir:
        return Path(os.path.abspath(os.path.expanduser(os.fspath(job_dir)))), None
    images = Path(os.path.abspath(os.fspath(image_folder)))
    folder = job_folder_for(images)
    if writable(folder, images):
        return folder, None
    root = Path(root) if root is not None else index.default_root()
    job_id = job_id or index.folder_id(images)
    target = root / job_id
    fallback = {"image_folder": str(images),
                "reason": f"{folder} cannot be created or written"}
    if emit is not None:
        emit(f"[job] {folder} cannot be created or written; the job folder is {target} "
             f"(job id {job_id})")
    if register:
        held = index.lookup(root, job_id)
        if held is None or held.get("job_folder") != str(target) or "fallback" not in held:
            index.register(root, job_id, target, fallback=fallback)
    return target, fallback


#: ``job.json``'s ``image_folder`` when the job folder is the default one,
#: ``<images>/langslice``: its parent. Stored relative, so the job folder
#: moves (or is renamed) with its images. An explicit job folder
#: (``--job-dir``) stores the absolute path it needs.
IMAGES_ARE_PARENT = ".."


def stored_image_folder(layout: JobLayout) -> str | None:
    """What ``job.json`` stores as the image folder of *layout*."""
    if layout.image_folder is None:
        return None
    if layout.folder == job_folder_for(layout.image_folder):
        return IMAGES_ARE_PARENT
    return str(layout.image_folder)


def held_image_folder(folder: Path, held: dict[str, Any]) -> Path | None:
    """The image folder a ``job.json`` record *held* (of the job folder
    *folder*) names, as an absolute path, or None when it names none.

    A relative value is under the job folder (:data:`IMAGES_ARE_PARENT`: its
    parent). An absolute one is taken as it is, except a default job folder
    (named ``langslice``) whose stored folder no longer exists, written
    before the relative form: its parent, where its images moved with it.
    """
    value = held.get("image_folder")
    if not isinstance(value, str) or not value:
        value = (held.get("spec") or {}).get("image_folder") if isinstance(
            held.get("spec"), dict) else None
        if not isinstance(value, str) or not value:
            return None
    path = Path(value)
    if not path.is_absolute():
        return Path(os.path.normpath(os.path.join(os.path.abspath(folder), value)))
    if not path.is_dir() and Path(folder).name == JOB_DIRNAME:
        return Path(os.path.abspath(folder)).parent
    return path


def check_owner(layout: JobLayout) -> None:
    """Refuse a job folder that holds the job of ANOTHER image folder.

    Two jobs never share a folder silently; the same image folder's job is
    continued. The default job folder's images are its parent wherever it
    moves (:data:`IMAGES_ARE_PARENT`). An explicit folder whose image folder
    no longer exists is taken over by the images it is opened with (they
    moved: ``langslice job NEW_IMAGES init --job-dir FOLDER``). Raises
    ``ValueError``.
    """
    held = read_job_file(layout)
    if held is None or layout.image_folder is None:
        return
    owner = held_image_folder(layout.folder, held)
    if owner is None or not owner.is_dir():
        return
    if os.path.realpath(owner) != os.path.realpath(layout.image_folder):
        raise ValueError(
            f"{layout.folder} already holds the job of {owner}, not of {layout.image_folder}. "
            "Choose another job folder (--job-dir) for this one.")


def _safe(name: str) -> str:
    cleaned = _UNSAFE.sub("_", name).strip().lstrip(".")
    return cleaned or "_"


def section_dirname(section_id: str, siblings: list[str] | tuple[str, ...] = ()) -> str:
    """A section's folder name: its image filename's stem.

    The whole filename when another image among *siblings* shares the stem
    (``s1.png`` and ``s1.tif``), so two sections never share a folder.
    Characters a file system refuses become ``_``.
    """
    name = Path(section_id).name
    stem = Path(name).stem
    clash = sum(1 for other in siblings if Path(Path(other).name).stem == stem) > 1
    return _safe(name if clash else stem)


@dataclass
class JobLayout:
    """The job folder of one stack and the names inside it."""

    folder: Path
    #: The images this job registers; section folder names come from them.
    image_folder: Path | None = None
    _names: dict[str, str] | None = field(default=None, repr=False)

    @classmethod
    def for_images(cls, image_folder: str | os.PathLike[str]) -> JobLayout:
        """The job folder next to *image_folder*."""
        return cls(job_folder_for(image_folder), Path(os.path.abspath(os.fspath(image_folder))))

    @property
    def job_file(self) -> Path:
        return self.folder / JOB_FILE

    @property
    def state_file(self) -> Path:
        return self.folder / STATE_FILE

    @property
    def history_dir(self) -> Path:
        return self.folder / HISTORY_DIR

    @property
    def sections_dir(self) -> Path:
        return self.folder / SECTIONS_DIR

    @property
    def views_dir(self) -> Path:
        return self.folder / VIEWS_DIR

    @property
    def views_index(self) -> Path:
        return self.folder / VIEWS_INDEX

    @property
    def exports_dir(self) -> Path:
        return self.folder / EXPORTS_DIR

    @property
    def logs_dir(self) -> Path:
        return self.folder / LOGS_DIR

    @property
    def results_file(self) -> Path:
        return self.exports_dir / RESULTS_FILE

    @property
    def host_result_file(self) -> Path:
        return self.exports_dir / HOST_RESULT_FILE

    @property
    def prompt_file(self) -> Path:
        return self.folder / PROMPT_FILE

    # --- sections ---------------------------------------------------------------

    def _siblings(self) -> list[str]:
        if self.image_folder is None or not self.image_folder.is_dir():
            return []
        from langslice.core.discovery import discover_slices

        return [os.path.basename(path) for path in discover_slices(str(self.image_folder))]

    def section_name(self, section_id: str) -> str:
        """The folder name of *section_id* under ``sections/``."""
        if self._names is None:
            siblings = self._siblings()
            self._names = {name: section_dirname(name, siblings) for name in siblings}
        name = self._names.get(Path(section_id).name)
        return name if name is not None else section_dirname(section_id)

    def section_dir(self, section_id: str) -> Path:
        return self.sections_dir / self.section_name(section_id)

    def deformable_dir(self, section_id: str) -> Path:
        """Where a section's applied deformation records live (one per key)."""
        return self.section_dir(section_id) / DEFORMABLE_DIR

    def image_correction_dir(self, section_id: str) -> Path:
        """Where a section's ``trace_borders`` calls live (one per call key)."""
        return self.section_dir(section_id) / IMAGE_CORRECTION_DIR

    def section_views_dir(self, section_id: str) -> Path:
        return self.section_dir(section_id) / SECTION_VIEWS_DIR

    # --- paths in job files ------------------------------------------------------

    def relative(self, path: str | os.PathLike[str]) -> str:
        """*path* as job files store it: relative to the job folder (POSIX
        separators) when inside it; an outside path stays absolute."""
        from langslice.job.checkpoint import relative_to

        return relative_to(self.folder)(os.path.abspath(os.fspath(path)))

    def resolve(self, ref: str | os.PathLike[str]) -> Path:
        """A stored path as a real one: relative ones are under the job folder."""
        path = Path(os.fspath(ref))
        return path if path.is_absolute() else self.folder / path

    def ensure(self, *, lean: bool = False) -> None:
        """Create the job folder and its fixed subfolders (*lean*: those a
        lean job writes, without ``history/``, ``views/`` and ``logs/``)."""
        folders = ((self.folder, self.sections_dir, self.exports_dir) if lean else
                   (self.folder, self.history_dir, self.sections_dir, self.views_dir,
                    self.exports_dir, self.logs_dir))
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)

    def log_event(self, kind: str, **fields: Any) -> None:
        """Append one line to ``logs/events.jsonl``; a failure is not fatal."""
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            record = {"kind": kind, **fields,
                      "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            with (self.logs_dir / EVENTS_FILE).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        except OSError:
            pass


# --- job.json --------------------------------------------------------------------


def read_job_file(layout: JobLayout) -> dict[str, Any] | None:
    """``job.json``, or None when the folder has none.

    Raises ``ValueError`` for a folder written by a newer LangSlice: it is
    refused, never guessed at.
    """
    try:
        data = json.loads(layout.job_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(data, dict):
        raise ValueError(f"{layout.job_file} does not hold a LangSlice job")
    version = data.get(FORMAT_KEY)
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError(f"{layout.job_file} has an unreadable format_version {version!r}")
    if version > JOB_FORMAT_VERSION:
        raise ValueError(
            f"This job folder was written by a newer LangSlice (format {version}; this "
            f"version reads up to {JOB_FORMAT_VERSION}). Update LangSlice to open it.")
    return data


def write_job_file(layout: JobLayout, **fields: Any) -> dict[str, Any]:
    """Write ``job.json``: *fields* over what it holds (``created_at`` and any
    field not given are kept), at the current format version."""
    from langslice.job.checkpoint import write_json_atomic

    held = read_job_file(layout) or {}
    record = {**held, **fields, FORMAT_KEY: JOB_FORMAT_VERSION}
    record.setdefault("created_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    if layout.image_folder is not None:
        record["image_folder"] = stored_image_folder(layout)
    write_json_atomic(str(layout.job_file), dict(sorted(record.items())))
    return record

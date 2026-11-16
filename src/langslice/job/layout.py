"""Where a job's files live: the job folder beside the images, and its names.

One stack, one job folder: ``<images>/langslice/``. Every host puts it next
to the images it hands LangSlice (the CLI's folder, the snapshots ABBA
exports), so a job travels with its images and is found
without an id. Inside::

    job.lock             the write lock (:mod:`langslice.job.lock`)
    job.json             settings (the JobSpec) + format_version; a saved
                         host job's own fields under "host"
    state.json           the state checkpoint (Stack, versioned)
    registration.json    its public rendering (public_files.py)
    review.json          the review comments a person drew in the LangSlice
                         window (review.py; not state, no undo step)
    history/undo.json    the undo and redo steps (undo_steps.py)
    sections/<name>/     per section (<name>: the image filename's stem)
        history.jsonl              every write that changed it (history.py)
        deformable/<key>/          applied deformation records
        border_trace/<key>/        trace_borders calls and their attempts
        views/<seq>_<tool>_<mode>/ pictures of this section the model saw
        coords.tif, labels.tif, labels_fiji.tif, labels.csv,
        residual.tif, maps.json    the section's maps (at submit and by
                                   ``export_maps``)
    views/<seq>_<tool>_<mode>/     pictures of several sections (stack level)
    views.jsonl          append-only index of every saved picture
    views.seq            the picture and call numbers handed out (+ views.seq.lock)
    exports/             linear_results.json, spend.json (the agent run's spend),
                         quicknii.json and visualign.json
    logs/                events.jsonl (opens, plugins, every tool call's
                         tool_start / tool_end on every door), runs/ (the agent
                         CLI's background runs), calls.jsonl (the agent CLI's calls),
                         calls/ (each call's envelope), launches/ (the exact
                         launch prompts of `langslice-job FOLDER launch`)
    trace/               agent transcripts: LangSlice's own agent's trace (also
                         when --trace-dir sends it elsewhere), and a coding
                         agent's session copied in by its harness plugin
                         (langslice.doors.commands.transcripts)
    env/                 the job's own Python environment, made by
                         `langslice-job FOLDER env`
                         (langslice.doors.agent_cli.environment)
    AGENTS.md, CLAUDE.md the reference card for coding agents
    TOOLS.md             every verb's description and arguments (by `launch`)

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
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: The job folder's name, next to the images.
JOB_DIRNAME = "langslice"
JOB_FILE = "job.json"
STATE_FILE = "state.json"
HISTORY_DIR = "history"
#: Under ``history/``: the undo and redo steps.
UNDO_FILE = "undo.json"
#: Under ``sections/<name>/``: every write that changed the section.
SECTION_HISTORY_FILE = "history.jsonl"
SECTIONS_DIR = "sections"
VIEWS_DIR = "views"
VIEWS_INDEX = "views.jsonl"
EXPORTS_DIR = "exports"
LOGS_DIR = "logs"
EVENTS_FILE = "events.jsonl"
#: Under ``logs/``: the launch prompts ``langslice-job FOLDER launch`` printed,
#: ``<six-digit number>_<role>.md`` (the exact text).
LAUNCHES_DIR = "launches"
#: Agent transcripts: LangSlice's own agent's trace, a coding agent's session.
TRACE_DIR = "trace"
#: The job's own Python environment (``langslice-job FOLDER env``).
ENV_DIR = "env"
#: Where tool plugins keep their files: ``plugins/<tool>/`` is a plugin tool's
#: only place to write (:mod:`langslice.job.plugin_store`).
PLUGINS_DIR = "plugins"
#: Under ``exports/``: the run's result (the state's shape, unversioned) and
#: a host job's final result (state + host updates).
RESULTS_FILE = "linear_results.json"
#: Under ``exports/``: what the agent run spent (tokens, usage-window share, dollar
#: estimate), written when the session ends, for the operator.
SPEND_FILE = "spend.json"
#: Under ``sections/<name>/``.
DEFORMABLE_DIR = "deformable"
MASKS_DIR = "masks"
#: A section's made images (``<name>.png``, ``<name>.2.png``...; never deleted).
IMAGES_DIR = "images"
BORDER_TRACE_DIR = "border_trace"
SECTION_VIEWS_DIR = "views"
#: The public rendering of the state (:mod:`langslice.job.public_files`, which
#: also names each section's map files).
REGISTRATION_FILE = "registration.json"
#: The review comments a person left in the LangSlice window
#: (:mod:`langslice.job.review`).
REVIEW_FILE = "review.json"

#: The job folder's format, carried by ``job.json``: this layout.
JOB_FORMAT_VERSION = 1
FORMAT_KEY = "format_version"

_UNSAFE = re.compile(r"[\x00-\x1f/\\:*?\"<>|]")
#: The device names Windows refuses as a file or folder name, in any case and
#: before any dot: ``aux`` and ``aux.png`` alike.
_RESERVED = re.compile(r"CON|PRN|AUX|NUL|COM[0-9¹²³]|LPT[0-9¹²³]",
                       re.IGNORECASE)


def job_folder_for(image_folder: str | os.PathLike[str]) -> Path:
    """``<image_folder>/langslice`` (absolute)."""
    return Path(os.path.abspath(os.fspath(image_folder))) / JOB_DIRNAME


def writable(folder: Path, image_folder: Path) -> bool:
    """Whether the job folder can be made and written, without making it.

    An existing folder is probed with a temporary file; a missing one asks
    whether *image_folder* (its parent) can take a new folder.
    """
    parent = folder if folder.exists() else image_folder
    try:
        handle, probe = tempfile.mkstemp(dir=parent, prefix=".write-probe-")
        os.close(handle)
        os.unlink(probe)
        return True
    except OSError:
        return False


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
    parent); an absolute one is taken as it is. Without one, the stored
    spec's ``image_folder``.
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
    return path


def check_owner(layout: JobLayout) -> None:
    """Refuse a job folder that holds the job of ANOTHER image folder.

    Two jobs never share a folder silently; the same image folder's job is
    continued. The default job folder's images are its parent wherever it
    moves (:data:`IMAGES_ARE_PARENT`). An explicit folder whose image folder
    no longer exists is taken over by the images it is opened with (they
    moved: ``langslice-job NEW_IMAGES init --job-dir FOLDER``). Raises
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


def _windows_name(name: str) -> str:
    """Make *name* (a :func:`_safe` one) a name Windows takes.

    A trailing dot or space becomes ``_``, and a device name gets ``_``
    after it (``aux_.png``).
    """
    if name.endswith((".", " ")):  # Windows drops a trailing dot or space
        name = name.rstrip(". ") + "_"
    head, dot, rest = name.partition(".")
    if _RESERVED.fullmatch(head.rstrip(" ")):
        return f"{head}_{dot}{rest}"
    return name


def _distinct(order: list[str], bases: dict[str, str]) -> dict[str, str]:
    """Give each name its base as a folder, distinct ignoring case.

    A base an earlier name in *order* has gets the first ``_2``, ``_3``,
    ... that is no other name's base.
    """
    every_base = {base.casefold() for base in bases.values()}
    assigned: set[str] = set()
    folders: dict[str, str] = {}
    for name in order:
        folder = bases[name]
        n = 1
        while folder.casefold() in assigned:
            n += 1
            folder = f"{bases[name]}_{n}"
            if folder.casefold() in every_base:
                folder = bases[name]
        assigned.add(folder.casefold())
        folders[name] = folder
    return folders


def section_dirnames(siblings: list[str] | tuple[str, ...]) -> dict[str, str]:
    """Name the folder of every section among the images of one stack.

    A section's folder is its image filename's stem, or the whole filename
    when another image shares the stem ignoring case (``s1.png`` and
    ``s1.tif``). Characters a file system refuses become ``_``, a trailing
    dot or space becomes ``_``, and a name Windows reserves for a device
    (``CON``, ``aux``, ``nul.png``, ``COM1``) gets ``_`` after it. Two
    folders that would still be equal ignoring case (``S1.png`` and
    ``s1.png``, ``a?b`` and ``a_b``) are told apart: a filename none of
    these rules changed comes first, then sorted order; the first keeps
    the name, the next ones get ``_2``, ``_3``, ...

    Parameters
    ----------
    siblings
        The section filenames (or paths) of the stack.

    Returns
    -------
    dict of str to str
        Each filename's folder name; no two equal ignoring case.
    """
    names = sorted({Path(sibling).name for sibling in siblings})
    stems = Counter(Path(name).stem.casefold() for name in names)
    raw: dict[str, str] = {}
    for name in names:
        stem = Path(name).stem
        if stems[stem.casefold()] > 1:
            raw[name] = name
        else:
            raw[name] = stem

    bases = {name: _windows_name(_safe(text)) for name, text in raw.items()}
    order = sorted(names, key=lambda name: (bases[name] != raw[name], name))
    return _distinct(order, bases)


def section_dirname(section_id: str, siblings: list[str] | tuple[str, ...] = ()) -> str:
    """A section's folder name among *siblings* (:func:`section_dirnames`).

    A section that is not among *siblings* is named as if it were the stack's
    only image.
    """
    name = Path(section_id).name
    folders = section_dirnames(siblings)
    if name in folders:
        return folders[name]
    return section_dirnames([name])[name]


@dataclass
class JobLayout:
    """The job folder of one stack and the names inside it."""

    folder: Path
    #: The images this job registers; section folder names come from them.
    image_folder: Path | None = None
    _names: dict[str, str] | None = field(default=None, repr=False)

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
    def undo_file(self) -> Path:
        """The undo and redo steps, ``history/undo.json``."""
        return self.history_dir / UNDO_FILE

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
    def launches_dir(self) -> Path:
        return self.logs_dir / LAUNCHES_DIR

    @property
    def results_file(self) -> Path:
        return self.exports_dir / RESULTS_FILE

    @property
    def spend_file(self) -> Path:
        return self.exports_dir / SPEND_FILE

    @property
    def trace_dir(self) -> Path:
        return self.folder / TRACE_DIR

    @property
    def review_file(self) -> Path:
        """Return the review comments' file (:mod:`langslice.job.review`).

        Returns
        -------
        Path
            ``review.json`` in the job folder.
        """
        return self.folder / REVIEW_FILE

    @property
    def plugins_dir(self) -> Path:
        return self.folder / PLUGINS_DIR

    def plugin_dir(self, tool: str) -> Path:
        """The folder of plugin tool *tool* (``plugins/<tool>/``; made on use)."""
        return self.plugins_dir / tool

    # --- sections ---------------------------------------------------------------

    def _siblings(self) -> list[str]:
        if self.image_folder is None or not self.image_folder.is_dir():
            return []
        from langslice.core.sections.discovery import discover_slices

        return [os.path.basename(path) for path in discover_slices(str(self.image_folder))]

    def section_name(self, section_id: str) -> str:
        """The folder name of *section_id* under ``sections/``."""
        if self._names is None:
            self._names = section_dirnames(self._siblings())
        name = self._names.get(Path(section_id).name)
        return name if name is not None else section_dirname(section_id)

    def section_dir(self, section_id: str) -> Path:
        return self.sections_dir / self.section_name(section_id)

    def section_history(self, section_id: str) -> Path:
        """Return the path of a section's history.

        Parameters
        ----------
        section_id
            The section's filename.

        Returns
        -------
        Path
            Its ``history.jsonl``, appended to by every write.
        """
        return self.section_dir(section_id) / SECTION_HISTORY_FILE

    def deformable_dir(self, section_id: str) -> Path:
        """Where a section's applied deformation records live (one per key)."""
        return self.section_dir(section_id) / DEFORMABLE_DIR

    def masks_dir(self, section_id: str) -> Path:
        """Where a section's masks live (``<name>.npz``; never deleted)."""
        return self.section_dir(section_id) / MASKS_DIR

    def images_dir(self, section_id: str) -> Path:
        """Where a section's made images live (``<name>.png``; never deleted)."""
        return self.section_dir(section_id) / IMAGES_DIR

    def border_trace_dir(self, section_id: str) -> Path:
        """Return the folder of a section's ``trace_borders`` calls.

        Parameters
        ----------
        section_id
            The section's filename.

        Returns
        -------
        Path
            The folder; one subfolder per call key.
        """
        return self.section_dir(section_id) / BORDER_TRACE_DIR

    def section_views_dir(self, section_id: str) -> Path:
        return self.section_dir(section_id) / SECTION_VIEWS_DIR

    # --- paths in job files ------------------------------------------------------

    def relative(self, path: str | os.PathLike[str]) -> str:
        """*path* as job files store it: relative to the job folder (POSIX
        separators) when inside it; an outside path stays absolute."""
        from langslice.job.state_file import relative_to

        return relative_to(self.folder)(os.path.abspath(os.fspath(path)))

    def resolve(self, ref: str | os.PathLike[str]) -> Path:
        """A stored path as a real one: relative ones are under the job folder."""
        path = Path(os.fspath(ref))
        return path if path.is_absolute() else self.folder / path

    def ensure(self) -> None:
        """Create the job folder and its fixed subfolders."""
        for folder in (self.folder, self.history_dir, self.sections_dir, self.views_dir,
                       self.exports_dir, self.logs_dir):
            folder.mkdir(parents=True, exist_ok=True)

    def log_event(self, kind: str, **fields: Any) -> None:
        """Append one line to ``logs/events.jsonl``; a failure is not fatal."""
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            record = {"kind": kind, **fields,
                      "at": datetime.now(UTC).isoformat(timespec="seconds")}
            with (self.logs_dir / EVENTS_FILE).open("a", encoding="utf-8",
                newline="\n") as handle:
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
    from langslice.job.state_file import write_json_atomic

    held = read_job_file(layout) or {}
    record = {**held, **fields, FORMAT_KEY: JOB_FORMAT_VERSION}
    record.setdefault("created_at", datetime.now(UTC).isoformat(timespec="seconds"))
    if layout.image_folder is not None:
        record["image_folder"] = stored_image_folder(layout)
    write_json_atomic(str(layout.job_file), dict(sorted(record.items())))
    return record

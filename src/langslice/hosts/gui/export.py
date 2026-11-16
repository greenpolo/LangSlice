"""The window's Export menu: a job's registration as another program's file.

Three formats (:data:`FORMATS`), each written by the code the library uses:
``abba``, ``exports/<image folder>.abba``, an ABBA project file
(``tools/abba_export.py``); ``quicknii`` and ``visualign``,
``exports/quicknii.json`` and ``exports/visualign.json`` with the section
images they name, both written together (``export_stack``, what ``submit``
and ``export_maps`` write).

An export runs on a thread of its own (an ABBA file with warps takes
minutes); :class:`Exports` keeps each job's last export, which the page
polls. The export opens the job as the library does and writes only under
its ``exports/`` folder and ``registration.json``; the state is never
changed. :func:`reveal` opens the exports folder in the system's file
browser.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

import langslice
from langslice.core.refusal import RefusalCode, Refused
from langslice.job.public_files import QUICKNII_FILE, VISUALIGN_FILE

#: The formats the menu offers, with the label the page shows.
FORMATS = {
    "abba": "ABBA project (.abba)",
    "quicknii": "QuickNII (.json)",
    "visualign": "VisuAlign (.json)",
}
#: The file each QUINT format is written to.
QUINT_FILES = {"quicknii": QUICKNII_FILE, "visualign": VISUALIGN_FILE}


class ExportError(ValueError):
    """An export request that cannot be started."""


class ExportStatus(StrEnum):
    """Where a job's last export is."""

    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class ExportState:
    """One job's last export, as the page reads it.

    Attributes
    ----------
    state
        Idle, running, done or failed.
    format
        The format asked for.
    path
        The file written.
    message
        What was written, or why it was not.
    skipped
        How many sections were left out.
    started, seconds
        When it started (seconds since the epoch) and how long it took.
    """

    state: ExportStatus = ExportStatus.IDLE
    format: str = ""
    path: str = ""
    message: str = ""
    skipped: int = 0
    started: float = 0.0
    seconds: float = 0.0


@dataclass(frozen=True)
class Written:
    """What one export wrote: the file, in words, and the sections left out."""

    path: str
    message: str
    skipped: int


#: Writes one export of the job in a folder.
Runner = Callable[[Path, str], Written]


def _quint(folder: Path, kind: str) -> Written:
    """Write the job's QuickNII and VisuAlign files; name *kind*'s."""
    with langslice.open_job(folder) as job:
        files = [Path(path) for path in job.export_stack()]
    named = [path for path in files if path.name == QUINT_FILES[kind]]
    if not named:
        raise Refused(
            RefusalCode.NOTHING_EXPORTED, "no section has a placement to write"
        )
    count = sum(1 for path in files if path.suffix == ".png")
    return Written(str(named[0]), f"{count} sections", 0)


def run_export(folder: Path, kind: str) -> Written:
    """Write one export of the job in *folder*.

    Parameters
    ----------
    folder
        The job folder.
    kind
        One of :data:`FORMATS`.

    Returns
    -------
    Written
        The file, what it holds, and how many sections were left out.
    """
    if kind in QUINT_FILES:
        return _quint(folder, kind)
    with langslice.open_job(folder) as job:
        done = job.export_abba()
    warped = f", {len(done.warped)} with their warp" if done.warped else ""
    message = f"{len(done.sections)} sections{warped}"
    return Written(done.path, message, len(done.skipped))


class Exports:
    """Each listed job's last export (see the module text).

    Parameters
    ----------
    run
        What writes one export (:func:`run_export`; tests pass their own).
    """

    def __init__(self, run: Runner = run_export) -> None:
        self._run = run
        self._lock = threading.Lock()
        self._states: dict[Path, ExportState] = {}

    def state(self, folder: Path) -> ExportState:
        """Return the job's last export.

        Parameters
        ----------
        folder
            The job folder.

        Returns
        -------
        ExportState
            Idle when nothing was exported in this window.
        """
        with self._lock:
            return self._states.get(Path(folder), ExportState())

    def start(self, folder: Path, kind: str) -> ExportState:
        """Start an export of *kind* for the job in *folder*.

        Parameters
        ----------
        folder
            The job folder.
        kind
            One of :data:`FORMATS`.

        Returns
        -------
        ExportState
            The export, running.

        Raises
        ------
        ExportError
            For a format the menu does not offer, or while an export of
            the job is running.
        """
        if kind not in FORMATS:
            raise ExportError(f"LangSlice cannot export {kind!r}")
        folder = Path(folder)
        with self._lock:
            last = self._states.get(folder)
            if last is not None and last.state == ExportStatus.RUNNING:
                raise ExportError(
                    "An export of this job is still being written."
                )
            running = ExportState(
                state=ExportStatus.RUNNING, format=kind, started=time.time()
            )
            self._states[folder] = running
        threading.Thread(
            target=self._work,
            args=(folder, kind),
            daemon=True,
            name=f"langslice-export-{kind}",
        ).start()
        return running

    def _work(self, folder: Path, kind: str) -> None:
        started = time.perf_counter()
        try:
            written = self._run(folder, kind)
            done = ExportState(
                state=ExportStatus.DONE,
                format=kind,
                path=written.path,
                message=written.message,
                skipped=written.skipped,
            )
        except Exception as exc:  # noqa: BLE001 - the page says every failure
            message = str(exc) or type(exc).__name__
            done = ExportState(
                state=ExportStatus.FAILED, format=kind, message=message
            )
        seconds = round(time.perf_counter() - started, 1)
        with self._lock:
            began = self._states[folder].started
            self._states[folder] = replace(
                done, started=began, seconds=seconds
            )


def reveal(folder: Path) -> None:
    """Open the job's ``exports/`` folder in the system's file browser.

    Parameters
    ----------
    folder
        The job folder.

    Raises
    ------
    ExportError
        When the job has no exports folder yet.
    """
    target = Path(folder) / "exports"
    if not target.is_dir():
        raise ExportError("Nothing has been exported from this job yet.")
    if sys.platform == "win32":
        os.startfile(target)  # noqa: S606 - the job's folder, for the person
        return
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.Popen(  # noqa: S603
        [opener, str(target)], stdin=subprocess.DEVNULL
    )

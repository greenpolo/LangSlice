"""The window's Export menu: a job's registration as another program's file.

Three formats (:data:`FORMATS`), each written by the code the library uses:
``abba``, ``exports/<image folder>.abba``, an ABBA project file
(``tools/bookkeeping/abba_export.py``); ``quicknii`` and ``visualign``,
``exports/quicknii.json`` and ``exports/visualign.json`` with the section
images they name, both written together (``export_stack``, what ``submit``
and ``export_maps`` write).

An export runs on a thread of its own (an ABBA file with warps takes
minutes); :class:`Exports` keeps each job's last export, which the page
polls. The export opens the job as the library does and writes only under
its ``exports/`` folder and ``registration.json``; the state is never
changed. :func:`reveal` opens the exports folder in the system's file
browser.

``MIN_LANDMARKS`` is the fewest landmarks the menu writes a warp with:
fewer follow LangSlice's warp less closely than leaving it out (on 331CTX
the warps move tissue 95 um at the median; 100 landmarks miss them by
45 um, 50 by 83 um, 16 to 36 by 120 to 140 um).
"""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

import langslice
from langslice.core.failures import IMAGE_FAILURES
from langslice.core.maps.abba import MAX_POINTS
from langslice.core.refusal import RefusalCode, Refused
from langslice.job.exports.public_files import QUICKNII_FILE, VISUALIGN_FILE

FORMATS = {
    "abba": "ABBA project (.abba)",
    "abba_linear": "ABBA project, linear only (.abba)",
    "quicknii": "QuickNII (.json)",
    "visualign": "VisuAlign (.json)",
}
MIN_LANDMARKS = 100
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


Runner = Callable[[Path, str, int], Written]


def _quint(folder: Path, kind: str) -> Written:
    """Write the job's QuickNII and VisuAlign files; name *kind*'s."""
    with langslice.open_job(folder) as job:
        files = [Path(path) for path in job.export_stack()]
    named = [path for path in files if path.name == QUINT_FILES[kind]]
    if not named:
        raise Refused(RefusalCode.NOTHING_EXPORTED, "no section has a placement to write")
    count = sum(1 for path in files if path.suffix == ".png")
    return Written(str(named[0]), f"{count} sections", 0)


def run_export(folder: Path, kind: str, landmarks: int) -> Written:
    """Write one export of the job in *folder*.

    Parameters
    ----------
    folder
        The job folder.
    kind
        One of :data:`FORMATS`.
    landmarks
        An ABBA project's most landmark pairs per warp.

    Returns
    -------
    Written
        The file, what it holds, and how many sections were left out.
    """
    if kind in QUINT_FILES:
        return _quint(folder, kind)
    warps = kind != "abba_linear"
    with langslice.open_job(folder) as job:
        done = job.export_abba(warps=warps, landmarks=landmarks)
    message = f"{len(done.sections)} sections"
    if done.warp_error_um:
        medians = [row["median"] for row in done.warp_error_um.values()]
        typical = statistics.median(medians)
        message += (
            f", {len(done.warped)} with their warp (at most {landmarks}"
            f" landmarks, {typical:.0f} µm from LangSlice's at the median)"
        )
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

    def start(self, folder: Path, kind: str, *, landmarks: int = MAX_POINTS) -> ExportState:
        """Start an export of *kind* for the job in *folder*.

        Parameters
        ----------
        folder
            The job folder.
        kind
            One of :data:`FORMATS`.
        landmarks
            An ABBA project's most landmark pairs per warp.

        Returns
        -------
        ExportState
            The export, running.

        Raises
        ------
        ExportError
            For a format the menu does not offer, a landmark count out
            of range, or while an export of the job is running.
        """
        if kind not in FORMATS:
            raise ExportError(f"LangSlice cannot export {kind!r}")
        if not MIN_LANDMARKS <= landmarks <= MAX_POINTS:
            raise ExportError(f"A warp takes {MIN_LANDMARKS} to {MAX_POINTS} landmarks.")
        folder = Path(folder)
        with self._lock:
            last = self._states.get(folder)
            if last is not None and last.state == ExportStatus.RUNNING:
                raise ExportError("An export of this job is still being written.")
            running = ExportState(state=ExportStatus.RUNNING, format=kind, started=time.time())
            self._states[folder] = running
        threading.Thread(
            target=self._work,
            args=(folder, kind, landmarks),
            daemon=True,
            name=f"langslice-export-{kind}",
        ).start()
        return running

    def _work(self, folder: Path, kind: str, landmarks: int) -> None:
        started = time.perf_counter()
        try:
            written = self._run(folder, kind, landmarks)
            done = ExportState(
                state=ExportStatus.DONE,
                format=kind,
                path=written.path,
                message=written.message,
                skipped=written.skipped,
            )
        except Refused as refusal:
            done = ExportState(state=ExportStatus.FAILED, format=kind, message=str(refusal))
        except IMAGE_FAILURES as exc:
            message = str(exc) or type(exc).__name__
            done = ExportState(state=ExportStatus.FAILED, format=kind, message=message)
        seconds = round(time.perf_counter() - started, 1)
        with self._lock:
            began = self._states[folder].started
            self._states[folder] = replace(done, started=began, seconds=seconds)


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
        os.startfile(target)
        return
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.Popen([opener, str(target)], stdin=subprocess.DEVNULL)

"""``abba.follow``: a job's checkpoints streamed into ABBA as rows.

The Fiji connector starts one follower per job it attached to ABBA and
keeps the request open while ABBA is. The follower changes nothing of the
job: every :data:`POLL_S` it looks at ``state.json`` (size and
modification time) and, when it changed, reads the state as last saved
(never waiting for a writer) and sends one ``checkpoint`` event of ABBA
rows, so the writes that land between two looks are one ABBA step. The
first checkpoint is the job's whole state measured against ABBA with this
job's own steps removed (:func:`baseline_rows`), so it brings ABBA in line
with the job. The first look after a checkpoint that finds the state as it
was sends ``idle`` with the state file's modification time
(``state_mtime_ns``): everything up to that state has been sent, so a
connector that has applied the events before it shows the job as saved
then. Before the job has a checkpoint (prepared, not opened) it sends
``waiting`` once and keeps looking. It ends when *stop* is set: the
connector closed the request's input.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, cast

from langslice.doors.abba.checkpoints import (
    Emit,
    HostCheckpoints,
    baseline_rows,
    host_facts,
)
from langslice.doors.abba.contract import AbbaFollowRequest
from langslice.doors.jobs import Opened, open_folder
from langslice.job.checkpoint import file_stamp
from langslice.job.layout import JobLayout, read_job_file
from langslice.job.state_file import stack_to_json

if TYPE_CHECKING:
    from langslice.job.job import Job

logger = logging.getLogger(__name__)

#: Seconds between two looks at the job's state file: short enough that
#: ABBA follows a run as it goes, long enough to cost nothing.
POLL_S = 0.5
#: Why a folder cannot be followed.
NOT_PREPARED = (
    "{folder} holds no job prepared from ABBA (job.json has no host.abba); "
    "choose Register > LangSlice in ABBA to make one."
)


def follow(
    request: AbbaFollowRequest, emit: Emit, stop: threading.Event
) -> None:
    """Send the job's checkpoints to ABBA until *stop* is set.

    Parameters
    ----------
    request
        The job folder to follow.
    emit
        Where each ``checkpoint``, ``log`` and ``waiting`` event goes.
    stop
        Set when the connector closed the request: the follower ends.

    Raises
    ------
    ValueError
        When the folder holds no job prepared from ABBA, or its sections
        differ in cutting angle (ABBA shows one angle for the stack).
    """
    folder = Path(request.job_folder).expanduser().resolve(strict=True)
    layout = JobLayout(folder)
    _abba_angles(layout)
    if not _opened(layout, emit, stop):
        return
    logger.info("Following the job in %s", layout.folder)
    opened = open_folder(layout.folder, persist=False)
    try:
        _send_changes(opened, layout, emit, stop)
    finally:
        opened.close()


def _opened(layout: JobLayout, emit: Emit, stop: threading.Event) -> bool:
    """Wait for the job's first checkpoint; False when stopped first."""
    if layout.state_file.exists():
        return True
    emit({"kind": "waiting"})
    while not layout.state_file.exists():
        if stop.wait(POLL_S):
            return False
    return True


def _send_changes(
    opened: Opened, layout: JobLayout, emit: Emit, stop: threading.Event
) -> None:
    """Send a checkpoint whenever the saved state changed, until *stop*."""
    job = opened.job
    tracker = HostCheckpoints(host_facts(opened.spec), emit)
    tracker.attach(job, opened.ctx)
    try:
        _follow(job, tracker, layout, stop)
    finally:
        tracker.close()


def _follow(
    job: Job,
    tracker: HostCheckpoints,
    layout: JobLayout,
    stop: threading.Event,
) -> None:
    """Check the job's state every :data:`POLL_S` until *stop*."""
    seen = None
    told = None
    while not stop.is_set():
        stamp = file_stamp(str(layout.state_file))
        if stamp != seen:
            seen = stamp
            with job.reading():
                current = stack_to_json(job.state)
                if tracker.previous is None:
                    angles = _abba_angles(layout)
                    tracker.previous = baseline_rows(current, angles)
                if current != tracker.previous:
                    tracker(job.state)
        elif stamp is not None and stamp != told:
            told = stamp
            tracker.emit({"kind": "idle", "state_mtime_ns": stamp[2]})
        if stop.wait(POLL_S):
            return
        _abba_angles(layout)  # the job is still ABBA's


def _abba_angles(layout: JobLayout) -> dict[str, float]:
    """ABBA's cutting angles when the job was prepared (``host.abba``)."""
    record = read_job_file(layout)
    angles: object = None
    if record is not None:
        host: object = record.get("host")
        if isinstance(host, dict):
            abba = cast("dict[str, object]", host).get("abba")
            if isinstance(abba, dict):
                angles = cast("dict[str, object]", abba).get("angles_deg")
    if not isinstance(angles, dict):
        raise ValueError(NOT_PREPARED.format(folder=layout.folder))
    values = cast("dict[str, float]", angles)
    return {
        "pitch_deg": float(values["pitch_deg"]),
        "yaw_deg": float(values["yaw_deg"]),
    }

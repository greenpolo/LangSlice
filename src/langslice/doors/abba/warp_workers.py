"""Warp conversions for ABBA, run in processes of their own.

One applied deformation becomes an ABBA landmark row in 10 to 80 s
(:func:`langslice.core.maps.abba.warp_world_landmarks`), so the follower
(:class:`langslice.doors.abba.checkpoints.HostCheckpoints`) converts the
sections a checkpoint changed side by side: at most :data:`MAX_CONVERSIONS`
spawned processes, each started with its share of the machine's threads
(:func:`thread_share`).
"""

from __future__ import annotations

import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from langslice.core.fits.deformable.record import DeformableRecord
from langslice.core.maps.abba import warp_world_landmarks
from langslice.core.maps.section_maps import SectionFrame

#: At most this many conversions run at once, one process each.
MAX_CONVERSIONS = 4
#: The thread-count variables of the numerical libraries a conversion uses.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class WarpInput:
    """Hold what one conversion needs, read from the job in the follower."""

    #: The section's filename.
    section: str
    #: Its frame, built from the snapshot's pixels.
    frame: SectionFrame
    #: Its applied deformation.
    record: DeformableRecord
    #: Its six stored numbers.
    params: tuple[float, ...]
    #: The snapshot's width and height in pixels.
    size: tuple[int, int]
    #: The snapshot's micrometres per pixel.
    calibration: float


@dataclass(frozen=True)
class ConvertedWarp:
    """Hold one conversion's landmark pairs, accuracy and cost."""

    #: Section points as the affine step places them, as two rows (x, y).
    source_mm: list[list[float]]
    #: Where the deformation takes them, as two rows.
    target_mm: list[list[float]]
    #: The atlas-side error of ABBA's spline: maximum, 99th and 95th
    #: percentiles and median, millimetres.
    max_error_mm: float
    p99_error_mm: float
    p95_error_mm: float
    median_error_mm: float
    #: Where the error was measured (``atlas overlap`` or ``whole
    #: snapshot``).
    region: str
    #: How many pairs.
    points: int
    #: How long the conversion took.
    seconds: float
    #: Its process's peak resident memory, None where unknown.
    peak_mb: float | None


def thread_share(workers: int) -> int:
    """Return the threads each of *workers* processes may use.

    Parameters
    ----------
    workers
        How many conversions run at once.

    Returns
    -------
    int
        The machine's cores shared out, at least one.
    """
    return max(1, (os.cpu_count() or 1) // max(1, workers))


@contextmanager
def started_with_threads(threads: int) -> Generator[None, None, None]:
    """Give processes started inside the block *threads* threads each.

    A spawned process copies the environment when it starts, before it
    loads numpy, so the numerical libraries read the limit; this
    process's own libraries are loaded already and keep theirs.

    Parameters
    ----------
    threads
        The threads a started process's numerical libraries may use.

    Yields
    ------
    None
        While the limit is in the environment.
    """
    before = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    os.environ.update(dict.fromkeys(THREAD_VARIABLES, str(threads)))
    try:
        yield
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def convert_warp(item: WarpInput) -> ConvertedWarp:
    """Convert one applied deformation to ABBA landmark pairs.

    Parameters
    ----------
    item
        The section's frame, deformation and snapshot.

    Returns
    -------
    ConvertedWarp
        The pairs, their measured accuracy and what the conversion cost.
    """
    started = time.perf_counter()
    landmarks = warp_world_landmarks(
        item.frame,
        item.record,
        item.params,
        size=item.size,
        pixel_size_um=item.calibration,
    )
    atlas = landmarks.accuracy.atlas
    return ConvertedWarp(
        source_mm=cast("list[list[float]]", landmarks.source_mm.T.tolist()),
        target_mm=cast("list[list[float]]", landmarks.target_mm.T.tolist()),
        max_error_mm=atlas.max_mm,
        p99_error_mm=atlas.p99_mm,
        p95_error_mm=atlas.p95_mm,
        median_error_mm=atlas.median_mm,
        region=landmarks.region,
        points=len(landmarks.source_mm),
        seconds=time.perf_counter() - started,
        peak_mb=_peak_mb(),
    )


def _peak_mb() -> float | None:
    """Return the process's peak resident memory in MB, None off Linux."""
    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in status.splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024  # kilobytes
    return None

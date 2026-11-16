"""Warp conversions for ABBA, run in processes of their own.

One applied deformation becomes an ABBA landmark row in 10 to 80 s
(:func:`langslice.core.maps.abba.warp_world_landmarks`), so the follower
(:class:`langslice.doors.abba.checkpoints.HostCheckpoints`) converts the
sections a checkpoint changed side by side: at most :data:`MAX_CONVERSIONS`
spawned processes, each started with its share of the machine's threads
(:func:`langslice.core.worker_threads.thread_share`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from langslice.core.fits.deformable.record import DeformableRecord
from langslice.core.maps.abba import warp_world_landmarks
from langslice.core.maps.section_maps import SectionFrame

#: At most this many conversions run at once, one process each.
MAX_CONVERSIONS = 4


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

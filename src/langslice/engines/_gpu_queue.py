"""The GPU's one queue, as the engine processes take it: a lock on one file.

Every fit on the machine that runs on the GPU takes the same file, so jobs
running at once never fill the GPU together: FireANTs and SPOmiAlign in
LangSlice's own process (the core's ``gpu_turn``), and STalign and emlddmm
in theirs. The file is
``LANGSLICE_GPU_LOCK``, else ``~/.cache/langslice-stalign/gpu.lock``.

Each engine's ``engine.py`` runs under the engine's own interpreter, which
need not have LangSlice or ``filelock`` installed, so this module imports the
standard library only and is put on the path by the engine itself. It takes
the lock as ``filelock`` does, so the two wait for each other: ``fcntl.flock``
on POSIX, and on Windows ``msvcrt.locking`` of the file's first byte.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

#: The machine's one GPU queue; LangSlice's own GPU fits take the same file.
GPU_LOCK = Path(
    os.environ.get("LANGSLICE_GPU_LOCK")
    or Path.home() / ".cache" / "langslice-stalign" / "gpu.lock"
)
#: How long a wait on Windows sleeps between asks: ``filelock``'s own poll.
POLL_S = 0.05
#: The start of the name torch gives a GPU device (``cuda``, ``cuda:0``).
GPU_DEVICE = "cuda"

if sys.platform == "win32":
    import msvcrt

    def _take(fd: int) -> None:
        while True:
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except OSError:  # held by another process: ask again
                time.sleep(POLL_S)
                continue
            return

    def _give(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _take(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX)

    def _give(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


@dataclass
class GpuTurn:
    """Hold how long a fit waited for its turn on the GPU."""

    waited_s: float = 0.0


@contextlib.contextmanager
def gpu_turn(device: object, *, path: Path = GPU_LOCK) -> Generator[GpuTurn]:
    """Wait for the GPU's queue, then hold it until the block ends.

    Parameters
    ----------
    device
        The torch device the fit runs on, or its name; a device whose name
        does not start with ``cuda`` takes no lock.
    path
        The lock file every GPU fit on the machine takes.

    Yields
    ------
    GpuTurn
        How long the turn was waited for, in seconds.
    """
    turn = GpuTurn()
    if not str(device).startswith(GPU_DEVICE):
        yield turn
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    try:
        waiting_since = time.monotonic()
        _take(fd)
        turn.waited_s = time.monotonic() - waiting_since
        try:
            yield turn
        finally:
            _give(fd)
    finally:
        os.close(fd)

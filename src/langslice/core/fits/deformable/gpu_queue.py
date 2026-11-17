"""The machine's GPU queue: how many fits run on the GPU at once.

Every process on the machine that fits on the GPU (FireANTs here, and
STalign, emlddmm and SPOmiAlign through the plugin API) takes a slot of one
queue, a set of file locks, so the fits of several jobs, agents and threads
share the GPU without filling it: :func:`gpu_turn`. How many slots there are
is ``core.hardware.gpu_slots``. A fit that still runs out of GPU memory (a
process outside the queue holds it) waits and runs again; :func:`out_of_memory`
tells that error.
"""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock, Timeout

from langslice.core.hardware import gpu_slots

GPU_LOCK = Path(
    os.environ.get("LANGSLICE_GPU_LOCK")
    or Path.home() / ".cache" / "langslice-stalign" / "gpu.lock"
)
OOM_RETRIES = 4
OOM_WAIT_S = 5.0


@dataclass
class GpuTurn:
    """How long a fit waited for its turn on the GPU (:func:`gpu_turn`)."""

    waited_s: float = 0.0


GPU_SLOT_POLL_S = 0.2


def _slot_file(slot: int) -> Path:
    if slot == 0:
        return GPU_LOCK
    return GPU_LOCK.with_name(f"{GPU_LOCK.stem}.{slot}{GPU_LOCK.suffix}")


def _free_slot(slots: int) -> FileLock | None:
    for slot in range(slots):
        lock = FileLock(str(_slot_file(slot)))
        try:
            lock.acquire(timeout=0)
        except (Timeout, PermissionError):
            continue
        return lock
    return None


GPU_DEVICE = "cuda"


def on_gpu(device: str) -> bool:
    """Tell whether *device* is a GPU.

    Parameters
    ----------
    device
        A torch device name: ``cpu``, ``cuda`` or ``cuda:1``.

    Returns
    -------
    bool
        True for ``cuda`` and every numbered GPU.
    """
    return device.startswith(GPU_DEVICE)


@contextlib.contextmanager
def gpu_turn(device: str) -> Generator[GpuTurn, None, None]:
    """Take a slot of the GPU queue for a fit on a GPU.

    As many fits run on the GPU at once, across processes and threads, as
    ``core.hardware.gpu_slots`` gives (``LANGSLICE_GPU_SLOTS``, else the
    GPU's memory over ``GIB_PER_GPU_FIT``); a fit beyond them waits for a
    free slot.

    Parameters
    ----------
    device
        The torch device the fit runs on; a fit on the CPU takes no slot.

    Yields
    ------
    GpuTurn
        How long the fit waited for its slot.
    """
    turn = GpuTurn()
    if not on_gpu(device):
        yield turn
        return
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    lock = _free_slot(gpu_slots())
    while lock is None:
        time.sleep(GPU_SLOT_POLL_S)
        lock = _free_slot(gpu_slots())
    turn.waited_s = time.perf_counter() - started
    try:
        yield turn
    finally:
        lock.release()


GPU_ALLOCATION_FAILURES = (
    "CUDNN_STATUS_INTERNAL_ERROR",
    "CUBLAS_STATUS_ALLOC_FAILED",
)


def out_of_memory(exc: BaseException) -> bool:
    """Tell whether an error is the GPU running out of memory.

    Parameters
    ----------
    exc
        The error a fit raised.

    Returns
    -------
    bool
        True for torch's ``OutOfMemoryError``, a kernel's generic CUDA
        error saying so, and cuDNN's and cuBLAS's own allocation failures.
    """
    text = str(exc)
    if type(exc).__name__ == "OutOfMemoryError" or "out of memory" in text:
        return True
    return any(failure in text for failure in GPU_ALLOCATION_FAILURES)

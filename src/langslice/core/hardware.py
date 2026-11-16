"""The machine LangSlice runs on, and how many sections to work on at once.

A per-section step that runs a multithreaded filter (ITK's N4,
Elastix) keeps a fixed thread count per call, because the count changes a
multithreaded filter's last bits. What scales with the machine is how many
sections run at once: :func:`section_workers` divides the processor's cores
by the threads one call takes. :func:`map_sections` runs a step over the
sections on that many threads, in order; a library that holds Python's lock
while it runs (SimpleITK, ITK's Python wrapping) runs in worker processes
instead (``process_pool``, ``kept_process_pool``).
``LANGSLICE_CPU_CORES`` sets the core count by hand (``1`` runs every section
in turn, in this process).

A fit on the GPU still takes the machine's GPU queue
(:func:`langslice.core.fits.deformable.engines.gpu_turn`) one at a time; the
work around it (reading the section, the pictures) runs in parallel.
"""

from __future__ import annotations

import contextlib
import contextvars
import ctypes
import logging
import multiprocessing
import os
import sys
import threading
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Protocol, TypeVar

#: Sets the core count LangSlice divides among sections (an integer).
CORES_VARIABLE = "LANGSLICE_CPU_CORES"
#: A gibibyte, in bytes.
GIB = 2**30

logger = logging.getLogger(__name__)

_T = TypeVar("_T")
_R = TypeVar("_R")


@dataclass(frozen=True)
class Hardware:
    """The machine's processor, memory and GPU (:func:`detect`)."""

    #: Logical cores this process may run on.
    cores: int
    #: Physical memory, in GiB (0 when it cannot be read).
    memory_gib: float
    #: The first GPU's name, or None without one torch sees.
    gpu: str | None = None
    #: The first GPU's memory, in GiB (0 without one).
    gpu_memory_gib: float = 0.0

    def describe(self) -> str:
        """Say the machine in one line.

        Returns
        -------
        str
            Its cores, memory and GPU.
        """
        words = f"{self.cores} cores, {self.memory_gib:.0f} GiB memory"
        if self.gpu is None:
            return f"{words}, no GPU"
        return f"{words}, {self.gpu} ({self.gpu_memory_gib:.0f} GiB)"


def cpu_cores() -> int:
    """Count the logical cores this process may run on.

    Returns
    -------
    int
        ``LANGSLICE_CPU_CORES`` when it is set to a positive integer, else
        the cores the process's affinity allows (all of them on Windows).
    """
    given = os.environ.get(CORES_VARIABLE, "").strip()
    if given.isdigit() and int(given) > 0:
        return int(given)
    if sys.platform != "win32" and hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


class _MemoryStatus(ctypes.Structure):
    length: int
    total_physical: int
    _fields_ = [  # noqa: RUF012 - ctypes reads the class attribute
        ("length", ctypes.c_ulong),
        ("load", ctypes.c_ulong),
        ("total_physical", ctypes.c_ulonglong),
        ("free_physical", ctypes.c_ulonglong),
        ("total_page_file", ctypes.c_ulonglong),
        ("free_page_file", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("free_virtual", ctypes.c_ulonglong),
        ("free_extended_virtual", ctypes.c_ulonglong),
    ]


def _windows_memory() -> float:
    status = _MemoryStatus()
    status.length = ctypes.sizeof(_MemoryStatus)
    kernel = ctypes.windll.kernel32  # pyright: ignore[reportAttributeAccessIssue] - Windows only
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):  # pyright: ignore[reportUnknownMemberType]
        return 0.0
    return int(status.total_physical) / GIB


def memory_gib() -> float:
    """Measure the machine's physical memory.

    Returns
    -------
    float
        The memory in GiB; 0 when it cannot be read.
    """
    try:
        if sys.platform == "win32":
            return _windows_memory()
        pages = os.sysconf("SC_PHYS_PAGES")
        return pages * os.sysconf("SC_PAGE_SIZE") / GIB
    except (AttributeError, OSError, ValueError):
        return 0.0


def _first_gpu() -> tuple[str | None, float]:
    try:
        import torch  # local: slow to import, needed only to name the GPU

        if not torch.cuda.is_available():
            return None, 0.0
        name = torch.cuda.get_device_name(0)
        _free, total = torch.cuda.mem_get_info(0)
    except Exception:  # noqa: BLE001 - no usable torch is no GPU
        return None, 0.0
    return name, total / GIB


def detect(*, gpu: bool = True) -> Hardware:
    """Find the machine LangSlice runs on.

    Parameters
    ----------
    gpu
        Ask torch for the first GPU (importing torch, which is slow the
        first time); False leaves the GPU out.

    Returns
    -------
    Hardware
        Its cores, memory and GPU.
    """
    name, memory = _first_gpu() if gpu else (None, 0.0)
    return Hardware(
        cores=cpu_cores(),
        memory_gib=round(memory_gib(), 1),
        gpu=name,
        gpu_memory_gib=round(memory, 1),
    )


def section_workers(tasks: int, threads_each: int) -> int:
    """Divide the cores among sections.

    Parameters
    ----------
    tasks
        The sections (or other tasks) to work on.
    threads_each
        The threads one of them keeps busy.

    Returns
    -------
    int
        How many to work on at once: the cores over *threads_each*, at most
        *tasks*, at least one.
    """
    if tasks <= 1:
        return 1
    return max(1, min(tasks, cpu_cores() // max(1, threads_each)))


def map_sections(
    step: Callable[[_T], _R], items: Sequence[_T], threads_each: int
) -> list[_R]:
    """Run *step* on each item, :func:`section_workers` at once, on threads.

    Each step runs in a copy of the caller's context, so what the call
    collects in context variables (the pictures a tool drew) sees every
    section. The first exception any step raised is raised once every step
    has finished (no step is left running behind the caller).

    Parameters
    ----------
    step
        The work on one item.
    items
        The items, in order.
    threads_each
        The threads one step keeps busy.

    Returns
    -------
    list
        Each item's result, in order.
    """
    workers = section_workers(len(items), threads_each)
    if workers <= 1:
        return [step(item) for item in items]
    with ThreadPoolExecutor(workers, thread_name_prefix="section") as pool:
        futures = [
            pool.submit(contextvars.copy_context().run, step, item)
            for item in items
        ]
    return [future.result() for future in futures]


def _spawned_pool(
    workers: int, *, initializer: Callable[[], None] | None = None
) -> ProcessPoolExecutor | None:
    try:
        return ProcessPoolExecutor(
            workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initializer,
        )
    except (OSError, ValueError, NotImplementedError) as exc:
        logger.warning("no worker processes (%s): working here", exc)
        return None


@contextlib.contextmanager
def process_pool(
    tasks: int, *, threads_each: int = 1
) -> Generator[ProcessPoolExecutor | None, None, None]:
    """Start worker processes for one call's tasks.

    The processes are spawned, so a task is a module-level function and its
    arguments are pickled; they end with the block.

    Parameters
    ----------
    tasks
        The tasks the call will hand them.
    threads_each
        The threads one task keeps busy (:func:`section_workers`).

    Yields
    ------
    ProcessPoolExecutor or None
        The pool; None when one process would do, or when processes cannot
        be started here (the caller then runs the tasks itself).
    """
    workers = section_workers(tasks, threads_each)
    pool = _spawned_pool(workers) if workers > 1 else None
    if pool is None:
        yield None
        return
    with pool:
        yield pool


def map_processes(
    task: Callable[[_T], _R],
    items: Sequence[_T],
    pool: ProcessPoolExecutor | None,
) -> list[_R]:
    """Run *task* on each item in *pool*'s processes.

    Parameters
    ----------
    task
        A module-level function of one item.
    items
        The items, in order.
    pool
        The processes (:func:`process_pool`); None runs the items here, and
        so does a pool that broke (a process that died).

    Returns
    -------
    list
        Each item's result, in order.
    """
    if pool is None or len(items) <= 1:
        return [task(item) for item in items]
    try:
        return list(pool.map(task, items))
    except BrokenProcessPool as exc:
        logger.warning("worker processes failed (%s): working here", exc)
        return [task(item) for item in items]


_KEPT_POOLS: dict[str, ProcessPoolExecutor] = {}
_KEPT_POOLS_LOCK = threading.Lock()


def kept_process_pool(
    name: str,
    *,
    threads_each: int,
    initializer: Callable[[], None] | None = None,
) -> ProcessPoolExecutor | None:
    """Return the process's kept worker pool *name*, starting it on first use.

    A kept pool lives as long as the process, so its workers load a slow
    library once (*initializer*) for every later call.

    Parameters
    ----------
    name
        The pool's name; one pool per name.
    threads_each
        The threads one task keeps busy: the pool has the cores over that.
    initializer
        A module-level function each worker runs first.

    Returns
    -------
    ProcessPoolExecutor or None
        The pool; None when one worker would do, or when processes cannot be
        started here.
    """
    workers = cpu_cores() // max(1, threads_each)
    if workers <= 1:
        return None
    with _KEPT_POOLS_LOCK:
        pool = _KEPT_POOLS.get(name)
        if pool is None:
            pool = _spawned_pool(workers, initializer=initializer)
            if pool is not None:
                _KEPT_POOLS[name] = pool
        return pool


def drop_process_pool(name: str, pool: ProcessPoolExecutor) -> None:
    """Forget the kept pool *name* when it is still *pool* (it broke).

    Parameters
    ----------
    name
        The pool's name (:func:`kept_process_pool`).
    pool
        The pool the caller found broken.
    """
    with _KEPT_POOLS_LOCK:
        if _KEPT_POOLS.get(name) is pool:
            del _KEPT_POOLS[name]
    pool.shutdown(wait=False, cancel_futures=True)


class Threader(Protocol):
    """A library's global default thread count (ITK's and SimpleITK's)."""

    def GetGlobalDefaultNumberOfThreads(self, /) -> int:  # noqa: N802 - ITK's name
        """Return the count."""
        ...

    def SetGlobalDefaultNumberOfThreads(self, count: int, /) -> None:  # noqa: N802 - ITK's name
        """Set the count.

        Parameters
        ----------
        count
            The threads every component built from now on runs on.
        """
        ...


_THREAD_COUNTS: dict[int, list[int]] = {}
_THREAD_COUNTS_LOCK = threading.Lock()


@contextlib.contextmanager
def global_threads(
    threader: Threader, count: int
) -> Generator[None, None, None]:
    """Hold a library's global thread count at *count* while any call runs.

    Several sections' calls on threads overlap; the count before the first
    comes back after the last, so no call sees another's restore.

    Parameters
    ----------
    threader
        The library's thread settings (ITK's ``MultiThreaderBase``,
        SimpleITK's ``ProcessObject``).
    count
        The threads every call runs on.

    Yields
    ------
    None
        While the count is held.
    """
    key = id(threader)
    with _THREAD_COUNTS_LOCK:
        counts = _THREAD_COUNTS.get(key)
        if counts is None:
            before = int(threader.GetGlobalDefaultNumberOfThreads())
            counts = _THREAD_COUNTS[key] = [0, before]
            threader.SetGlobalDefaultNumberOfThreads(count)
        counts[0] += 1
    try:
        yield
    finally:
        with _THREAD_COUNTS_LOCK:
            counts[0] -= 1
            if counts[0] == 0:
                threader.SetGlobalDefaultNumberOfThreads(counts[1])
                del _THREAD_COUNTS[key]


def simpleitk_threader() -> Threader:
    """Return SimpleITK's thread settings, for :func:`global_threads`.

    Some of its filters ignore their own thread count and run on the
    global one.

    Returns
    -------
    Threader
        ``SimpleITK.ProcessObject``.
    """
    import SimpleITK as sitk  # local: slow to import, needed only to filter

    return sitk.ProcessObject

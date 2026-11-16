"""Thread limits for worker processes that run numerical work side by side.

A spawned process copies the environment when it starts, before it loads
numpy, so the thread counts set here are the ones its numerical libraries
use; this process's own libraries are loaded already and keep theirs.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager

#: The thread-count variables of the numerical libraries a conversion uses.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def thread_share(workers: int) -> int:
    """Return the threads each of *workers* processes may use.

    Parameters
    ----------
    workers
        How many processes run at once.

    Returns
    -------
    int
        The machine's cores shared out, at least one.
    """
    return max(1, (os.cpu_count() or 1) // max(1, workers))


@contextmanager
def started_with_threads(threads: int) -> Generator[None, None, None]:
    """Give processes started inside the block *threads* threads each.

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

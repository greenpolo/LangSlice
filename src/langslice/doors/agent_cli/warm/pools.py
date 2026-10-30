"""Stop the threads loading left behind, before the warm process forks.

The one module of the warm process that reaches into other libraries'
private state (``README.md`` lists it with the versions it was written
against).
"""

from __future__ import annotations

import sys
from typing import Any


def quiet_pools() -> None:
    """Stop the threads loading left, which a forked child would find gone.

    Dask's pool (it reads BrainGlobe's arrays), zarr's event loop and
    LangSlice's sampling threads (each makes them again on its next use),
    and fsspec's event loop with the S3 sessions on it (BrainGlobe's version
    check), whose closing at a child's exit would wait for that loop.
    """
    import weakref

    oblique = sys.modules.get("langslice.core.geometry.oblique")
    if oblique is not None:
        for pool in list(oblique._sample_pools.values()):
            pool.shutdown(wait=True)
        oblique._sample_pools.clear()
    zarr_sync = sys.modules.get("zarr.core.sync")
    if zarr_sync is not None and zarr_sync.loop[0] is not None:
        zarr_sync.cleanup_resources()

    threaded: Any = sys.modules.get("dask.threaded")
    if threaded is not None:
        pool = getattr(threaded, "default_pool", None)
        if pool is not None:
            pool.shutdown(wait=True)
        threaded.default_pool = None
        getattr(threaded, "pools", {}).clear()
    asyn = sys.modules.get("fsspec.asyn")
    if asyn is None:
        return
    registry = getattr(weakref.finalize, "_registry", {})
    for finalizer, info in list(registry.items()):
        if getattr(info.func, "__name__", "") == "close_session":
            finalizer()
    loop, thread = asyn.loop[0], asyn.iothread[0]
    if loop is not None and thread is not None and thread.is_alive():
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        executor = getattr(loop, "_default_executor", None)
        if executor is not None:
            executor.shutdown(wait=True)
        asyn.reset_lock()

# The warm process

Infrastructure, not science: it makes `langslice-job` calls fast by loading
the libraries and the job's atlas once per job folder and forking a child
per call. A call it answers behaves as a cold call does. The package
docstring (`__init__.py`) describes the client, the server and the child;
`docs/agents/cli.md`, "The warm process", is the user's view.

## Private state it touches

Stopping threads before a fork and ending a child as Python ends reach into
private names. Each is written against the versions below; a new version of
one of them is checked against this list.

| Module | Name | Library, version |
|---|---|---|
| `pools.py` | `langslice.core.geometry.oblique._sample_pools` | LangSlice |
| `pools.py` | `zarr.core.sync.loop`, `cleanup_resources` | zarr 3.1.6 |
| `pools.py` | `dask.threaded.default_pool`, `pools` | dask 2026.7.1 |
| `pools.py` | `fsspec.asyn.loop`, `iothread`, `reset_lock` | fsspec 2026.7.0 |
| `pools.py` | `weakref.finalize._registry`, an event loop's `_default_executor` | Python 3.11 |
| `child.py` | `threading._shutdown`, `atexit._run_exitfuncs` | Python 3.11 |
| `script.py` | `site._init_pathinfo`, `site.addpackage` | Python 3.11 |

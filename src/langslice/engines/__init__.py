"""In-tree engines: the STalign, SPOmiAlign and emlddmm adapters, loaded as tool plugins.

Each engine is a package of ``plugin.py`` (the adapter, run in LangSlice's process, importing
only :mod:`langslice.plugin_api`) and ``engine.py`` (the method itself, run as a subprocess
under the engine's own interpreter; nothing of it is imported here). A job names them in
``JobSpec.plugins`` exactly as any tool plugin::

    langslice.engines.stalign.plugin:tools
    langslice.engines.spomialign.plugin:tools
    langslice.engines.emlddmm.plugin:tools

The three load together in one job (the tool names do not clash) and re-import from a copy of
this package with no ``PYTHONPATH``. The engines' environments and sources stay outside
LangSlice's venv; a missing one is refused at tool-call time with ``ENGINE_UNAVAILABLE``, never
at load time. The code was moved here from SliceBench ``comparators/plugins/<name>/`` (its
history is in that repository).
"""

"""In-tree engines: the STalign, SPOmiAlign and emlddmm adapters, loaded as tool plugins.

Each engine is a package of ``plugin.py`` (the adapter, run in LangSlice's process, importing
only :mod:`langslice.plugin_api` and the engines' own modules) and ``engine.py`` (the method's
calls). A job names them in ``JobSpec.plugins`` exactly as any tool plugin::

    langslice.engines.stalign.plugin:tools
    langslice.engines.spomialign.plugin:tools
    langslice.engines.emlddmm.plugin:tools

The three load together in one job (the tool names do not clash) and re-import from a copy of
this package with no ``PYTHONPATH``. STalign and emlddmm run in LangSlice's process, imported
when a tool runs: STalign from the extra ``langslice[engines]``, emlddmm vendored as
``langslice_vendor.emlddmm``. SPOmiAlign's ``engine.py`` runs as a subprocess under its own
environment's interpreter, which stays outside LangSlice's venv. A missing engine is refused
at tool-call time with ``ENGINE_MISSING``, never at load time.
"""

"""In-tree engines: the DeepSlice, STalign, SPOmiAlign and emlddmm adapters.

Each engine is a package loaded as a tool plugin, its ``plugin.py`` the
adapter, run in LangSlice's process and importing only
:mod:`langslice.plugin_api`. A job names them in ``JobSpec.plugins``
exactly as any tool plugin::

    langslice.engines.deepslice.plugin:tools
    langslice.engines.stalign.plugin:tools
    langslice.engines.spomialign.plugin:tools
    langslice.engines.emlddmm.plugin:tools

They load together in one job (the tool names do not clash) and re-import
from a copy of this package with no ``PYTHONPATH``. DeepSlice runs in
LangSlice's process, imported only when its tool runs, from the
``langslice[engines]`` extra. The others run their ``engine.py`` as a
subprocess under the engine's own interpreter, and their environments and
sources stay outside LangSlice's venv; nothing of them is imported here. A
missing engine is refused at tool-call time with ``ENGINE_MISSING``, never
at load time. The code was moved here from SliceBench
``comparators/plugins/<name>/`` (its history is in that repository).
"""

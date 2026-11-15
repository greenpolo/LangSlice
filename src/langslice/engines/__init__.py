"""In-tree engines: DeepSlice, Brainways, STalign, SPOmiAlign, emlddmm.

Each engine is a package loaded as a tool plugin, its ``plugin.py`` the
adapter, run in LangSlice's process and importing only
:mod:`langslice.plugin_api`. A job names them in ``JobSpec.plugins``
exactly as any tool plugin::

    langslice.engines.deepslice.plugin:tools
    langslice.engines.brainways.plugin:tools
    langslice.engines.stalign.plugin:tools
    langslice.engines.spomialign.plugin:tools
    langslice.engines.emlddmm.plugin:tools

They load together in one job (the tool names do not clash) and re-import
from a copy of this package with no ``PYTHONPATH``. DeepSlice
(``deepslice_place``, mouse coronal jobs), Brainways (``brainways_ap``, rat
coronal jobs) and SPOmiAlign run in LangSlice's own process: the part of
DeepSlice and Brainways the tool uses is vendored beside its adapter
(DeepSlice 1.2.8 under the MIT licence, on TensorFlow from the
``langslice[engines]`` extra; Brainways 0.1.16.3 under the GPL-3.0, on
torch), each with its ``LICENSE``; SPOmiAlign runs on RoMa (``romatch``,
the same extra), imported only when a call runs. STalign and emlddmm run
their ``engine.py`` as a subprocess under the engine's own interpreter;
their environments and sources stay outside LangSlice's venv. A missing
engine is refused at tool-call time with ``ENGINE_MISSING``, never at
load time. The code was moved here from SliceBench
``comparators/plugins/<name>/`` (its history is in that repository).
"""

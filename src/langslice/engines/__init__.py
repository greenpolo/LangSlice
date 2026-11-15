"""In-tree engines: DeepSlice, Brainways, STalign, SPOmiAlign, emlddmm.

Each engine is a package loaded as a tool plugin, its ``plugin.py`` the
adapter, run in LangSlice's process and importing only
:mod:`langslice.plugin_api` and the engines' own modules. A job names them
in ``JobSpec.plugins`` exactly as any tool plugin::

    langslice.engines.deepslice.plugin:tools
    langslice.engines.brainways.plugin:tools
    langslice.engines.stalign.plugin:tools
    langslice.engines.spomialign.plugin:tools
    langslice.engines.emlddmm.plugin:tools

They load together in one job (the tool names do not clash) and re-import
from a copy of this package with no ``PYTHONPATH``. All five run in
LangSlice's own process, each method imported only when a call runs:
DeepSlice (``deepslice_place``, mouse coronal jobs) and Brainways
(``brainways_ap``, rat coronal jobs), the part each tool uses vendored
beside its adapter with its ``LICENSE`` (DeepSlice 1.2.8 under the MIT
licence, on TensorFlow from the ``langslice[engines]`` extra; Brainways
0.1.16.3 under the GPL-3.0, on torch); SPOmiAlign on RoMa (``romatch``, the
same extra); STalign (GPL-3.0) and emlddmm (Apache-2.0) vendored whole as
``langslice_vendor.stalign`` and ``langslice_vendor.emlddmm``, on torch. A
call whose engine is not installed (DeepSlice or SPOmiAlign without the
extra) is refused at tool-call time with ``ENGINE_MISSING``, never at load
time.
"""

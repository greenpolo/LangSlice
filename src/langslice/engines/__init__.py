"""In-tree engines: DeepSlice, Brainways, STalign, SPOmiAlign, emlddmm.

Each engine is a package whose ``plugin.py`` is the adapter: it binds the
engine's tools to their handlers and runs them in LangSlice's process. The
engines are first-party code, one layer above the plugin machinery
(:mod:`langslice.plugins`), whose tool-plugin API they use. Their tools are
tools of every job and are declared once, as rows of the tool table
(:data:`langslice.job.settings.tool_table.TOOLS`; the engines' own
:class:`~langslice.core.plugins.ToolSpec` is built from the row), which
names each engine by its reference
(:data:`langslice.job.settings.tool_table.ENGINES`); no job names them::

    langslice.engines.deepslice.plugin:tools
    langslice.engines.brainways.plugin:tools
    langslice.engines.stalign.plugin:tools
    langslice.engines.spomialign.plugin:tools
    langslice.engines.emlddmm.plugin:tools

An engine is loaded for a run that has one of its tools;
``deepslice_place`` and ``brainways_ap`` start off and a job's ``enabled``
switches them on. All five run in
LangSlice's own process, each method imported only when a call runs:
DeepSlice (``deepslice_place``, mouse coronal jobs) and Brainways
(``brainways_ap``, rat coronal jobs), the part each tool uses vendored
beside its adapter with its ``LICENSE`` (DeepSlice 1.2.8 under the MIT
licence, on TensorFlow from the ``langslice[engines]`` extra; Brainways
0.1.16.3 under the GPL-3.0, on torch); SPOmiAlign on RoMa (``romatch``, the
same extra); STalign (GPL-3.0) and emlddmm (Apache-2.0) vendored whole as
``langslice._vendor.stalign`` and ``langslice._vendor.emlddmm``, on torch. A
call whose engine is not installed (DeepSlice or SPOmiAlign without the
extra) is refused at tool-call time with ``ENGINE_UNAVAILABLE``, never at load
time.
"""

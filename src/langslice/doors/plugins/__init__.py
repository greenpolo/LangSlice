"""Tool plugins: the public API (``docs/plugins.md``), importable as ``langslice.plugin_api``.

A plugin is a Python module whose factory returns a list of :class:`ToolSpec`;
this package loads it, gives its handlers a :class:`ToolContext` and makes each
tool a first-class tool of the job's toolbox. ``PLUGIN_API_VERSION`` is the
API's version. Importing the package registers the loader the core calls for
the plugins a ``JobSpec`` names.
"""

from __future__ import annotations

from langslice.core.plugins import (
    PLUGIN_API_VERSION,
    Layer,
    PluginError,
    ToolError,
    ToolResult,
    ToolSpec,
)
from langslice.doors.plugins import loader as _loader
from langslice.doors.plugins.context import (
    AffineView,
    AtlasView,
    FitFrame,
    LayerRef,
    SectionRaster,
    SectionView,
    ToolContext,
    TransformProposal,
    WriteResult,
)
from langslice.doors.plugins.loader import as_refs, load

_loader.install()

__all__ = [
    "PLUGIN_API_VERSION", "AffineView", "AtlasView", "FitFrame", "Layer", "LayerRef",
    "PluginError", "SectionRaster", "SectionView", "ToolContext", "ToolError", "ToolResult",
    "ToolSpec",
    "TransformProposal", "WriteResult", "as_refs", "load",
]

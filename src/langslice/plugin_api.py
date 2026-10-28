"""The tool plugin API: what a plugin module imports (``docs/developers/plugins.md``).

A plugin adds tools to a job's toolbox. It declares each tool with a
:class:`ToolSpec` (a stub whose signature and docstring are the tool's
declaration, a handler, and what the tool reads and writes) and returns them
from a factory function; the handler gets a :class:`ToolContext` to read the
job and to write through. Only the names here are the API; ``PLUGIN_API_VERSION``
changes when they do incompatibly.
"""

from langslice.doors.plugins import (
    PLUGIN_API_VERSION,
    AffineView,
    AtlasPlane,
    AtlasView,
    DeformationProposal,
    FitFrame,
    Layer,
    LayerRef,
    PluginError,
    SectionRaster,
    SectionView,
    ToolContext,
    ToolError,
    ToolResult,
    ToolSpec,
    TransformProposal,
    WriteResult,
    flag_words,
)

__all__ = [
    "PLUGIN_API_VERSION", "AffineView", "AtlasPlane", "AtlasView", "DeformationProposal",
    "FitFrame", "Layer",
    "LayerRef",
    "PluginError", "SectionRaster", "SectionView", "ToolContext", "ToolError", "ToolResult",
    "ToolSpec",
    "TransformProposal", "WriteResult", "flag_words",
]

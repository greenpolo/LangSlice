"""The tool plugin API: what a plugin module imports (``docs/developers/plugins.md``).

A plugin adds tools to a job's toolbox. It declares each tool with a
:class:`ToolSpec` (a stub whose signature, and docstring or ``description``,
are the tool's declaration, a handler, and what the tool reads and writes) and returns them
from a factory function; the handler gets a :class:`ToolContext` to read the
job and to write through; a handler refuses a call with :class:`ToolError`
and a :class:`RefusalCode` (or a code of its own). Only the names here are
the API; ``PLUGIN_API_VERSION`` changes when they do incompatibly.
"""

from langslice.core.failures import ENGINE_FAILURES
from langslice.core.fits.deformable.gpu_queue import gpu_turn, out_of_memory
from langslice.core.fits.deformation import flag_words
from langslice.core.hardware import map_sections
from langslice.core.plugins import (
    PLUGIN_API_VERSION,
    Layer,
    PluginError,
    ToolError,
    ToolResult,
    ToolSpec,
)
from langslice.core.refusal import RefusalCode
from langslice.job.settings.tools_on import ToolsOn
from langslice.plugins.context import ToolContext
from langslice.plugins.frames import AtlasPlane, FitFrame
from langslice.plugins.proposals import DeformationProposal, TransformProposal
from langslice.plugins.reads import AffineView, AtlasView, LayerRef, SectionRaster, SectionView
from langslice.plugins.writes import WriteResult

__all__ = [
    "ENGINE_FAILURES",
    "PLUGIN_API_VERSION",
    "AffineView",
    "AtlasPlane",
    "AtlasView",
    "DeformationProposal",
    "FitFrame",
    "Layer",
    "LayerRef",
    "PluginError",
    "RefusalCode",
    "SectionRaster",
    "SectionView",
    "ToolContext",
    "ToolError",
    "ToolResult",
    "ToolSpec",
    "ToolsOn",
    "TransformProposal",
    "WriteResult",
    "flag_words",
    "gpu_turn",
    "map_sections",
    "out_of_memory",
]

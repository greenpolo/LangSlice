"""STalign as LangSlice tools: the adapter, run in LangSlice's process.

It is a tool of every job (its row of the tool table names
``langslice.engines.stalign.plugin:tools``). Options (``plugin_config``):
``device`` (``auto``: the GPU at float32 when there is one, else the CPU at
float64) and ``threads``; any other key is refused. STalign (GPL-3.0, vendored
as ``langslice._vendor.stalign``) runs in LangSlice's process: ``engine.py``
imports it when a tool runs. Its declaration is its rows of the tool table
(``langslice.plugins.declared``).

Two tools:

- ``stalign_lddmm``: STalign's 2-D fit, one piece of background work per section, building on
  the section's current registration (a landmark warp included);
- ``stalign_plane_search``: STalign 2.0's 3-D plane search, proposing cutting angles and
  positions, nothing written.

Inputs, as every engine takes them: ``section_image`` (a section image by name, default
``preprocessed``), ``atlas_image`` (default ``nissl``, resolved by the job's atlas; the reply
names the image used) and ``mask`` (STalign takes none: it sorts the section itself, and its final
sorting is saved with the deformation). The reply carries ``settings`` and, per section,
``inputs`` and ``estimated``.

Frames. The target is the section image (by name) on the fit grid, grey, scaled so
the slide (median of the image border) is 0 and the tissue (median inside the placed atlas
outline) is 1, whatever the stain's polarity, area-averaged to 10 um. Target micrometres are
atlas micrometres at the section's linear scale. The source is the atlas image on its native
plane, area-averaged to 50 um (the published atlas voxel), I / mean|I| plus (I - mean I)^2 (the
published atlas recipe); on a section that holds a deformation, the atlas resampled onto the
fit grid through it (``FitFrame.applied_map``), so the bend starts on top of a landmark warp.
Marked damage (with the native margin) is painted as atlas background.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langslice.engines import _blocks
from langslice.engines.stalign.declarations import (
    lddmm_description,
    plane_search_description,
    stalign_lddmm,
    stalign_plane_search,
)
from langslice.engines.stalign.landing import lddmm_tool
from langslice.engines.stalign.plane_search import plane_search_tool
from langslice.engines.stalign.settings import Options
from langslice.plugin_api import (
    ToolContext,
    ToolResult,
    ToolSpec,
)
from langslice.plugins.declared import engine_tool


def tools(config: Mapping[str, object] | None = None) -> list[ToolSpec]:
    """Return the plugin's tools.

    Parameters
    ----------
    config
        The plugin's entry of ``plugin_config``; None gives the defaults.

    Returns
    -------
    list of ToolSpec
        ``stalign_lddmm`` and ``stalign_plane_search``.

    Raises
    ------
    PluginError
        For an option the plugin does not take.
    """
    machine = Options.from_config(config).machine
    meta = {
        "method": "STalign 2.0.0",
        "paper": "Clifton et al. 2023, Nat Commun",
        "commit": "8a84bf1f",
        "engine": "langslice.engines.stalign",
    }

    def lddmm(
        ctx: ToolContext,
        sections: list[str],
        section_image: str = "preprocessed",
        atlas_image: str = "nissl",
        mask: str = "none",
        kernel_um: float = 500.0,
        fit: str = "warp",
        displacement_um: float = 94.0,
        sorting: dict[str, Any] | None = None,
        iterations: int = 2000,
        view: bool = True,
    ) -> ToolResult:
        _blocks.refuse_mask("stalign_lddmm", mask, "sorts the section's pixels itself")
        return lddmm_tool(
            ctx,
            machine,
            sections,
            section_image,
            atlas_image,
            kernel_um,
            fit,
            displacement_um,
            dict(sorting or {}),
            iterations,
            view,
        )

    def plane(
        ctx: ToolContext,
        sections: list[str],
        section_image: str = "preprocessed",
        atlas_image: str = "nissl",
        view: bool = True,
    ) -> ToolResult:
        return plane_search_tool(ctx, machine, sections, section_image, atlas_image, view)

    return [
        engine_tool(
            "stalign_lddmm",
            stalign_lddmm,
            lddmm,
            description=lddmm_description,
            version="1",
            meta=meta,
        ),
        engine_tool(
            "stalign_plane_search",
            stalign_plane_search,
            plane,
            description=plane_search_description,
            version="1",
            meta=meta,
        ),
    ]


__all__ = ["tools"]

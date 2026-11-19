"""SPOmiAlign as a LangSlice tool: the adapter, run in LangSlice's process.

It is a tool of every job (its row of the tool table names
``langslice.engines.spomialign.plugin:tools``). Option (``plugin_config``):
``device`` (the torch device: ``auto``, the first GPU when torch sees one,
else the CPU; ``cpu``, ``cuda`` or a numbered GPU); any other key is refused.
The method itself runs in this process, in ``engine.py`` and ``matching.py``,
on RoMa (``romatch``, LangSlice's ``engines`` extra), imported only when a call
runs; a call without it is refused ``ENGINE_UNAVAILABLE``. Its declaration is
its row of the tool table (``langslice.plugins.declared``).

Each call puts the section on SPOmiAlign's canvas, runs the engine on (atlas
image, section), and maps the answer back: the linear part through
``transform_from_matrix``, the full map through ``deformation_from_map``.
Canvas: for rigid, affine and affine_warp the atlas plane's grid, upsampled
to about :data:`CANVAS_EDGE` px, with the section resampled onto it through
its current transform (as the published user's hand-picked plate and
rotation did); for warp on a section that holds a deformation, the fit grid,
with the atlas resampled onto it through that deformation
(``FitFrame.applied_map``), so the warp starts on top of a landmark warp, as
``stalign_lddmm`` does. A new transform clears a deformation, so rigid,
affine and affine_warp are refused on a section that holds one
(``PRECONDITION``).
"""

from __future__ import annotations

from collections.abc import Mapping

from langslice.engines.spomialign import matching
from langslice.engines.spomialign.declarations import spomialign, spomialign_description
from langslice.engines.spomialign.landing import run_call
from langslice.engines.spomialign.settings import MISSING, NISSL, Options
from langslice.plugin_api import (
    RefusalCode,
    ToolContext,
    ToolError,
    ToolResult,
    ToolSpec,
)
from langslice.plugins.declared import engine_tool


def tools(config: Mapping[str, object] | None = None) -> list[ToolSpec]:
    """Return the plugin's tool.

    Parameters
    ----------
    config
        The plugin's entry of ``plugin_config``; None gives the defaults.

    Returns
    -------
    list of ToolSpec
        ``spomialign``.

    Raises
    ------
    PluginError
        For an option the plugin does not take.
    """
    options = Options.from_config(config)

    def handler(
        ctx: ToolContext,
        sections: list[str],
        fit: str,
        rigid: bool = False,
        section_image: str = "preprocessed",
        atlas_image: str = NISSL,
        mask: str = "none",
        smoothness_um: float = 150.0,
        matches: int = 5000,
        view: bool = True,
    ) -> ToolResult:
        if not matching.is_installed():
            raise ToolError(RefusalCode.ENGINE_UNAVAILABLE, MISSING)
        return run_call(
            ctx,
            options,
            sections,
            fit,
            rigid,
            section_image,
            atlas_image,
            mask,
            smoothness_um,
            matches,
            view,
        )

    return [
        engine_tool(
            "spomialign",
            spomialign,
            handler,
            description=spomialign_description,
            version="2",
            meta={
                "method": "SPOmiAlign",
                "paper": "Wang et al. 2026, Brief Bioinform",
                "commit": "c7f8b7a",
                "engine": "langslice.engines.spomialign",
            },
        )
    ]

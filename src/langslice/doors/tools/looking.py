"""The looking tools at the tool door: each name wired to its tool function.

Each tool's body is a function in its module under :mod:`langslice.tools`
that takes the run first and the model's arguments by name. Two take the
model's many options as one value, which this module builds from the
arguments: ``look`` a :class:`~langslice.core.pictures.look.LookRequest`
(``_look``) and ``make_image`` an
:class:`~langslice.tools.images.ImageRecipe` (``_make_image``).
``look``, ``zoom``, ``grep_atlas_view`` and ``make_image`` save their own
pictures: their replies carry ``pictures`` with the numbers, and the tool
door sends the pictures without saving them again.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

from langslice.core.atlas.levels import OVERLAY_DEPTH
from langslice.core.pictures.look import LookRequest
from langslice.tools import atlas, files, status, zoom
from langslice.tools import images as image_tools
from langslice.tools import look as tools_look
from langslice.tools.run import Run, as_list

#: The looking tools whose function takes the model's arguments as declared.
_TOOLS: dict[str, Callable[..., object]] = {
    "zoom": zoom.zoom,
    "grep_atlas": atlas.grep_atlas,
    "grep_atlas_view": atlas.grep_atlas_view,
    "status": status.status,
    "list_files": files.list_files,
    "search_files": files.search_files,
    "read_file": files.read_file,
}


def _look(
    run: Run,
    mode: str,
    sections: list[str],
    images: list[str],
    *,
    positions_mm: list[float] | None = None,
    atlas_layers: list[str] | None = None,
    atlas_opacity: float = 0.5,
    warp: str = "applied",
    resolution: int = 0,
    layers: list[str] | None = None,
    depth: int = OVERLAY_DEPTH,
    masks: list[str] | None = None,
    registrations: dict[str, int] | None = None,
    sheet: bool = False,
) -> object:
    """Build ``look``'s request from the model's arguments and run the tool.

    The arguments are ``look``'s declared ones (the description says what
    each does); a list argument may come as one item, and ``resolution``
    becomes the request's ``long_edge`` as asked, which the tool clamps.
    """
    request = LookRequest(
        mode=str(mode or "").strip().lower(),
        sections=tuple(as_list(sections)),
        positions_mm=tuple(as_list(positions_mm)),
        images=tuple(as_list(images)),
        atlas_layers=tuple(as_list(atlas_layers)) or None,
        atlas_opacity=atlas_opacity,
        warp=str(warp or "applied").strip().lower(),
        long_edge=resolution,
        layers=tuple(as_list(layers)),
        depth=depth,
        masks=tuple(as_list(masks)),
    )
    return tools_look.look(
        run, request, registrations=registrations, sheet=sheet
    )


def _make_image(
    run: Run,
    name: str,
    sections: list[str],
    weights: dict[str, float],
    clahe_clip: float,
    clahe_tiles: int,
    n4: bool,
    denoise: bool,
    contrast: list[float],
    gamma: float,
    invert: bool,
    delete: bool,
    *,
    view: bool = True,
) -> object:
    """Build ``make_image``'s recipe from the model's arguments; run the tool.

    The arguments are ``make_image``'s declared ones (the description says
    what each does); the tool checks the recipe's values.
    """
    recipe = image_tools.ImageRecipe(
        weights=weights,
        clahe_clip=clahe_clip,
        clahe_tiles=clahe_tiles,
        n4=n4,
        denoise=denoise,
        contrast=contrast,
        gamma=gamma,
        invert=invert,
    )
    return image_tools.make_image(
        run, name, sections, recipe, delete=delete, view=view
    )


def bodies(run: Run) -> dict[str, Callable[..., object]]:
    """Return the looking tools' bodies over *run*, by tool name.

    Parameters
    ----------
    run
        The run every tool body works over.

    Returns
    -------
    dict
        Each tool's function with *run* bound, taking the model's
        arguments by name.
    """
    wired = {"look": _look, "make_image": _make_image, **_TOOLS}
    return {name: functools.partial(tool, run) for name, tool in wired.items()}

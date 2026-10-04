"""Picture sizes: the long edge of every picture the agent is shown, per level.

Split out of ``linear/render.py`` (layered refactor, phase 3d). The working
frame fits are computed on is :data:`langslice.core.sections.PREVIEW_LONG_EDGE`,
never one of these.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langslice.linear.workspace import Workspace

#: Long edge, in pixels, of every picture the agent is SHOWN, per
#: ``JobSpec.image_resolution``: ``(opening, later)``. *Opening* is each tile
#: of the opening strips (every section, every atlas section beneath it or in
#: the atlas reference; :mod:`langslice.linear.opening`);
#: *later* is each picture a tool returns (each panel of a multi-panel
#: picture). "auto" opens at 256 and lets the agent pass ``resolution`` per
#: call (:data:`MIN_RESOLUTION` up to the driver model's own largest image,
#: which the door passes in), 512 when it does not. The only other
#: bound is the source: nothing is upsampled past the pixels it is drawn from
#: (a section's working copy, the atlas plane at its own voxel size), so a
#: small snapshot stays small. Never a working frame (:data:`PREVIEW_LONG_EDGE`)
#: and never the image model's input.
PICTURE_EDGES: dict[str, tuple[int, int]] = {
    "low": (256, 512),
    "medium": (384, 768),
    "high": (512, 1024),
    "auto": (256, 512),
}
#: The level that lets the agent choose each picture's size.
AUTO_RESOLUTION = "auto"
#: Smallest long edge the agent may ask for at "auto". The largest is the
#: driver model's own maximum image (the door knows the model:
#: ``view_options.view_edge_limit``; the MCP host is Claude,
#: ``opening.CLAUDE_MAX_VIEW_EDGE``). A value outside is clamped and the
#: reply says so.
MIN_RESOLUTION = 128

#: Default image/pair batch size. Separate positioning comparisons may return
#: two references per pair (up to eight images); retained context is governed
#: by the session's image-retention policy, not this per-tool batch size.
MAX_IMAGES_PER_CALL = 4


def resolution_level(ctx: Workspace) -> str:
    """This run's ``image_resolution`` ("low" when the spec has none)."""
    level = str(getattr(getattr(ctx, "spec", None), "image_resolution", "low") or "low")
    return level if level in PICTURE_EDGES else "low"


def opening_edge(ctx: Workspace) -> int:
    """Long edge of each opening-strip tile at this run's level."""
    return PICTURE_EDGES[resolution_level(ctx)][0]


def picture_edge(ctx: Workspace, requested: int | None = None) -> int:
    """Long edge of each later picture: the level's, or *requested* at "auto".

    *requested* must already be clamped (:data:`MIN_RESOLUTION` up to the
    driver model's maximum; :func:`langslice.linear.view_options.parse_view`
    does it); every level but
    "auto" ignores it.
    """
    level = resolution_level(ctx)
    if level == AUTO_RESOLUTION and requested:
        return int(requested)
    return PICTURE_EDGES[level][1]

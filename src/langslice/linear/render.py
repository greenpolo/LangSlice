"""Moved: the section renders, captions, canvas, sheets and status table are
in :mod:`langslice.core` since the layered refactor's phase 3d (2026-10-04).

This module only re-exports them for the sibling repos that import it
(``SliceBench``: ``slicebench/adapters/langslice_geometry.py`` imports
``PREVIEW_LONG_EDGE``, ``canvas_geometry``, ``canvas_um_per_px`` and
``render_slice``). LangSlice itself imports the core modules directly:
:mod:`langslice.core.sections` (renders and their cache),
:mod:`langslice.core.captions` (captions, fonts, scale bar),
:mod:`langslice.core.canvas` (the physical canvas),
:mod:`langslice.core.sheets` (stack sheets), :mod:`langslice.core.status`
(the status table) and :mod:`langslice.core.sizes` (picture sizes).
"""

from __future__ import annotations

from langslice.core.canvas import (  # noqa: F401
    CHECKER_TILES,
    MARKER_ATLAS,
    MARKER_CONNECTOR,
    MARKER_SECTION,
    OUTLINE_LAYERS,
    REGION_CONTEXT_ALPHA,
    VIEW_MODES,
    WORKING_MARGIN,
    CanvasGeometry,
    PanelFrame,
    atlas_mask_canvas,
    canvas_geometry,
    estimate_um_per_px,
    line_coverage,
    normalize_border_style,
    physical_overlay,
    physical_views,
    pivot_on_canvas,
    placement_matrices,
    region_polys,
    regions_left,
    template_canvas,
    zoom_box,
)
from langslice.core.captions import (  # noqa: F401
    CAPTION_PX,
    caption,
    scale_bar_px,
    wrap_caption,
)
from langslice.core.sections import (  # noqa: F401
    PREVIEW_LONG_EDGE,
    canvas_um_per_px,
    fine_detail,
    render_cache_key,
    render_slice,
    rescale_section_matrix,
    shown_section,
)
from langslice.core.sheets import (  # noqa: F401
    SHEET_MAX_LONG_EDGE,
    beside,
    grid,
    reference_slice_picture,
    spacing_plot,
    stack_pictures,
    stack_sheet,
    stacked,
)
from langslice.core.sizes import (  # noqa: F401
    AUTO_RESOLUTION,
    MAX_IMAGES_PER_CALL,
    MIN_RESOLUTION,
    PICTURE_EDGES,
    opening_edge,
    picture_edge,
    resolution_level,
)
from langslice.core.status import (  # noqa: F401
    compact_rows,
    slice_flags,
    status_rows,
    status_text,
)

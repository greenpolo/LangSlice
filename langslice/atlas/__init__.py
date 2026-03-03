"""LangSlice Atlas module — BrainGlobe atlas access as plain Python functions."""

from langslice.atlas.core import (
    canonicalize_atlas_name,
    DEFAULT_ATLAS_NAME,
    get_atlas_info,
    get_boundary_slice,
    get_composite_slice,
    get_position_range_mm,
    get_reference_slice,
    index_to_position_mm,
    list_available_atlases,
    list_downloaded_atlases,
    load_atlas,
    position_mm_to_index,
)

__all__ = [
    "load_atlas",
    "DEFAULT_ATLAS_NAME",
    "canonicalize_atlas_name",
    "position_mm_to_index",
    "index_to_position_mm",
    "get_position_range_mm",
    "get_reference_slice",
    "get_boundary_slice",
    "get_composite_slice",
    "get_atlas_info",
    "list_available_atlases",
    "list_downloaded_atlases",
]

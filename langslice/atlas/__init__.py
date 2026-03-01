"""LangSlice Atlas module — BrainGlobe atlas access as plain Python functions."""

from langslice.atlas.core import (
    ap_mm_to_index,
    canonicalize_atlas_name,
    DEFAULT_ATLAS_NAME,
    get_ap_range,
    get_atlas_info,
    get_boundary_slice,
    get_composite_slice,
    get_origin_index,
    get_reference_slice,
    list_available_atlases,
    list_downloaded_atlases,
    load_atlas,
)

__all__ = [
    "load_atlas",
    "DEFAULT_ATLAS_NAME",
    "canonicalize_atlas_name",
    "get_origin_index",
    "ap_mm_to_index",
    "get_ap_range",
    "get_reference_slice",
    "get_boundary_slice",
    "get_composite_slice",
    "get_atlas_info",
    "list_available_atlases",
    "list_downloaded_atlases",
]

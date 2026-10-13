"""BrainGlobe atlas access as plain functions: loading, plane positions, slices.

A deliberately small public surface: what LangSlice itself reads (the atlas,
its plane positions and the reference plate); region maps and borders are in
:mod:`langslice.core.atlas.render`, region levels in
:mod:`langslice.core.atlas.levels`.
"""

from langslice.core.atlas.core import (
    DEFAULT_ATLAS_NAME,
    canonicalize_atlas_name,
    get_position_range_mm,
    get_reference_slice,
    index_to_position_mm,
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
]

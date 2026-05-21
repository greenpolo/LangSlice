"""Backward-compatible shim to shared adaptive curriculum bins."""

from langslice_training.adaptive.curriculum.bins import *  # noqa: F401,F403
from langslice_training.adaptive.curriculum.bins import (  # noqa: F401
    clear_plane_extent_cache,
    compute_section_bins,
)

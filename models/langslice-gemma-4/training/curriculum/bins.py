"""Backward-compatible shim — curriculum.bins is now adaptive.curriculum.bins."""
from adaptive.curriculum.bins import *  # noqa: F401,F403
from adaptive.curriculum.bins import clear_plane_extent_cache, compute_section_bins  # noqa: F401

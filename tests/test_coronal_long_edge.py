"""Test the in-plane long-edge helper (coronal alias kept for back-compat)."""
from unittest.mock import MagicMock

import numpy as np

from langslice_harness.atlas.core import get_coronal_long_edge, get_in_plane_long_edge


def _mock_atlas(shape: tuple[int, int, int]) -> MagicMock:
    """Build a minimal atlas mock with AP/DV/ML on axes 0/1/2 (Allen-style)."""
    atlas = MagicMock()
    atlas.reference = np.zeros(shape, dtype=np.uint8)
    atlas.orientation = "asl"  # anterior, superior (top=dorsal), left
    atlas.resolution = (25.0, 25.0, 25.0)
    return atlas


def test_coronal_long_edge_returns_max_of_dv_ml():
    atlas = _mock_atlas((100, 320, 528))  # AP=100, DV=320, ML=528
    assert get_coronal_long_edge(atlas) == 528


def test_coronal_long_edge_tall_atlas():
    atlas = _mock_atlas((200, 600, 400))  # DV=600 > ML=400
    assert get_coronal_long_edge(atlas) == 600


def test_in_plane_long_edge_sagittal_picks_ap_or_dv():
    atlas = _mock_atlas((100, 320, 528))  # AP=100, DV=320, ML=528
    # Sagittal cuts along ML; in-plane axes are AP=100 and DV=320 → long edge 320
    assert get_in_plane_long_edge(atlas, plane="sagittal") == 320


def test_in_plane_long_edge_horizontal_picks_ap_or_ml():
    atlas = _mock_atlas((100, 320, 528))  # AP=100, DV=320, ML=528
    # Horizontal cuts along DV; in-plane axes are AP=100 and ML=528 → long edge 528
    assert get_in_plane_long_edge(atlas, plane="horizontal") == 528

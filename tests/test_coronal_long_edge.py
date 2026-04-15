"""Test the get_coronal_long_edge helper."""
from unittest.mock import MagicMock

import numpy as np

from langslice.atlas.core import get_coronal_long_edge


def test_coronal_long_edge_returns_max_of_dv_ml():
    atlas = MagicMock()
    atlas.reference = np.zeros((100, 320, 528), dtype=np.uint8)  # AP=100, DV=320, ML=528
    assert get_coronal_long_edge(atlas) == 528


def test_coronal_long_edge_tall_atlas():
    atlas = MagicMock()
    atlas.reference = np.zeros((200, 600, 400), dtype=np.uint8)  # DV > ML
    assert get_coronal_long_edge(atlas) == 600

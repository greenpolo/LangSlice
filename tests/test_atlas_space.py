"""Checks for brainglobe-space atlas orientation adapter."""

from collections.abc import Sequence

import numpy as np

from langslice.atlas.core import get_position_range_mm, index_to_position_mm, position_mm_to_index


class _FakeAtlas:
    atlas_name: str
    orientation: str
    reference: np.ndarray
    annotation: np.ndarray
    resolution: Sequence[float]
    metadata: dict[str, object]

    def __init__(self, orientation: str) -> None:
        self.atlas_name = f"fake_{orientation}"
        self.orientation = orientation
        self.reference = np.zeros((4, 5, 6), dtype=np.uint16)
        self.annotation = np.zeros((4, 5, 6), dtype=np.uint32)
        self.resolution = (100.0, 100.0, 100.0)
        self.metadata = {}


def test_atlas_space_adapter_asr_behavior() -> None:
    atlas_asr = _FakeAtlas("asr")
    assert position_mm_to_index(atlas_asr, 0.2) == 2
    assert abs(index_to_position_mm(atlas_asr, 2) - 0.2) < 1e-12
    min_pos, max_pos = get_position_range_mm(atlas_asr)
    assert abs(min_pos - 0.0) < 1e-12
    assert abs(max_pos - 0.3) < 1e-12


def test_atlas_space_adapter_rejects_unsupported_ap_orientation() -> None:
    atlas_psr = _FakeAtlas("psr")
    try:
        _ = position_mm_to_index(atlas_psr, 0.1)
    except ValueError as exc:
        assert "not supported" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported AP orientation")

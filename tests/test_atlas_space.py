"""Atlas orientation adapter: plane axes, position <-> index, position range
and reference-slice orientation, on small fake atlases and one real Allen atlas."""

from collections.abc import Sequence

import numpy as np
import pytest

from langslice.core.atlas.core import (
    get_position_range_mm,
    get_reference_slice,
    index_to_position_mm,
    load_atlas,
    position_mm_to_index,
)
from langslice.core.space import atlas_space_context, slice_axis_index


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
        self.template = np.zeros((4, 5, 6), dtype=np.uint16)
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


@pytest.mark.parametrize("orientation", ["asr", "air", "asl"])
def test_each_plane_slices_along_its_own_axis(orientation: str) -> None:
    ctx = atlas_space_context(_FakeAtlas(orientation))
    assert slice_axis_index(ctx, "coronal") == ctx.ap_axis_index
    assert slice_axis_index(ctx, "sagittal") == ctx.ml_axis_index
    assert slice_axis_index(ctx, "horizontal") == ctx.dv_axis_index


def test_position_range_follows_the_plane_axis() -> None:
    atlas = _FakeAtlas("asr")  # shape (4, 5, 6) at 0.1 mm: AP, DV, ML
    assert get_position_range_mm(atlas, plane="sagittal")[1] == pytest.approx(0.5)
    assert get_position_range_mm(atlas, plane="horizontal")[1] == pytest.approx(0.4)


class _TinyAsrAtlas:
    atlas_name = "tiny_asr"
    orientation = "asr"
    resolution = (1000.0, 1000.0, 1000.0)
    metadata = {}

    def __init__(self) -> None:
        self.template = np.zeros((5, 3, 4), dtype=np.uint8)
        for ap_idx in range(self.template.shape[0]):
            self.template[ap_idx, :, :] = ap_idx + 1
        self.annotation = self.template


@pytest.mark.parametrize(("plane", "size"), [("sagittal", (5, 3)), ("horizontal", (5, 4))])
def test_reference_slices_display_ap_left_to_right(plane: str, size: tuple[int, int]) -> None:
    img = get_reference_slice(_TinyAsrAtlas(), 0.0, plane=plane)
    assert img.size == size
    assert img.getpixel((0, 0)) < img.getpixel((4, 0))


@pytest.mark.slow
def test_real_allen_axes_ranges_and_round_trip() -> None:
    atlas = load_atlas("allen_mouse_25um")
    ctx = atlas_space_context(atlas)
    assert slice_axis_index(ctx, "coronal") == ctx.ap_axis_index
    lo, hi = get_position_range_mm(atlas)
    assert lo == 0.0
    assert hi > 10.0  # a mouse brain is ~13 mm AP
    assert 9.0 < get_position_range_mm(atlas, plane="sagittal")[1] < 13.0
    for mm in [0.0, 2.5, 5.0, 7.5]:
        back = index_to_position_mm(atlas, position_mm_to_index(atlas, mm))
        assert abs(back - mm) < 0.025  # within one voxel
    coronal = get_reference_slice(atlas, 5.0)
    assert coronal.size != get_reference_slice(atlas, 5.0, plane="sagittal").size

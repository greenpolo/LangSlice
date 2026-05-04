import numpy as np

from langslice_harness.atlas.core import (
    get_position_range_mm,
    get_reference_slice,
    index_to_position_mm,
    load_atlas,
    position_mm_to_index,
)
from langslice_harness.atlas.space import atlas_space_context, slice_axis_index


def test_coronal_axis_is_ap():
    atlas = load_atlas("allen_mouse_25um")
    ctx = atlas_space_context(atlas)
    assert slice_axis_index(ctx, "coronal") == ctx.ap_axis_index


def test_sagittal_axis_is_ml():
    atlas = load_atlas("allen_mouse_25um")
    ctx = atlas_space_context(atlas)
    assert slice_axis_index(ctx, "sagittal") == ctx.ml_axis_index


def test_horizontal_axis_is_dv():
    atlas = load_atlas("allen_mouse_25um")
    ctx = atlas_space_context(atlas)
    assert slice_axis_index(ctx, "horizontal") == ctx.dv_axis_index


def test_position_range_coronal_default():
    atlas = load_atlas("allen_mouse_25um")
    lo, hi = get_position_range_mm(atlas)
    assert lo == 0.0
    assert hi > 10.0  # mouse brain is ~13 mm AP


def test_position_range_sagittal():
    atlas = load_atlas("allen_mouse_25um")
    lo, hi = get_position_range_mm(atlas, plane="sagittal")
    assert lo == 0.0
    # Mouse brain ~11 mm ML total
    assert 9.0 < hi < 13.0


def test_position_roundtrip_coronal():
    atlas = load_atlas("allen_mouse_25um")
    for mm in [0.0, 2.5, 5.0, 7.5]:
        idx = position_mm_to_index(atlas, mm)
        back = index_to_position_mm(atlas, idx)
        assert abs(back - mm) < 0.025  # within one voxel


def test_reference_slice_shape_differs_by_plane():
    atlas = load_atlas("allen_mouse_25um")
    coronal = get_reference_slice(atlas, 5.0)
    sagittal = get_reference_slice(atlas, 5.0, plane="sagittal")
    horizontal = get_reference_slice(atlas, 5.0, plane="horizontal")
    # All three should be PIL Images with distinct (W, H) tuples.
    # Coronal is (ML, DV); sagittal is (AP, DV); horizontal is (ML, AP).
    assert coronal.size != sagittal.size
    assert coronal.size != horizontal.size


class _TinyAsrAtlas:
    atlas_name = "tiny_asr"
    orientation = "asr"
    resolution = (1000.0, 1000.0, 1000.0)
    metadata = {}

    def __init__(self) -> None:
        self.reference = np.zeros((5, 3, 4), dtype=np.uint8)
        for ap_idx in range(self.reference.shape[0]):
            self.reference[ap_idx, :, :] = ap_idx + 1
        self.annotation = self.reference


def test_sagittal_reference_slice_displays_ap_left_to_right():
    atlas = _TinyAsrAtlas()
    img = get_reference_slice(atlas, 0.0, plane="sagittal")

    assert img.size == (5, 3)
    assert img.getpixel((0, 0)) < img.getpixel((4, 0))


def test_horizontal_reference_slice_displays_ap_left_to_right():
    atlas = _TinyAsrAtlas()
    img = get_reference_slice(atlas, 0.0, plane="horizontal")

    assert img.size == (5, 4)
    assert img.getpixel((0, 0)) < img.getpixel((4, 0))

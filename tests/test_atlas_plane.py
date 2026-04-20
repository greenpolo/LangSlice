from langslice.atlas.space import atlas_space_context, slice_axis_index
from langslice.atlas.core import load_atlas


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

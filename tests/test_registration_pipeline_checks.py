"""Deterministic checks on shared orientation helpers.

`_despeckle_classified` and `generation_report` (paint-classification and
generated-vs-atlas diagnostics for the colormap workflow) were deleted with
that workflow; what remains here is the render-orientation helper, which is
unrelated to either border route or the colormap path."""

from __future__ import annotations


def test_orient_slice_to_axes_maps_horizontal_to_user_frame() -> None:
    from types import SimpleNamespace

    import numpy as np

    from langslice.core.space import atlas_space_context, native_slice_axes, orient_slice_to_axes

    atlas = SimpleNamespace(
        atlas_name="fake",
        orientation="asr",
        template=np.zeros((4, 5, 6)),
        resolution=(25.0, 25.0, 25.0),
    )
    ctx = atlas_space_context(atlas)
    assert native_slice_axes(ctx, "horizontal") == ("rl", "ap")
    assert native_slice_axes(ctx, "sagittal") == ("si", "ap")

    # horizontal render: rows right->left, cols anterior->posterior
    arr = np.arange(6 * 4).reshape(6, 4)  # (rl rows=6, ap cols=4)
    out = orient_slice_to_axes(arr, ctx, "horizontal", "ap,lr")
    assert out.shape == (4, 6)
    # anterior row of the output = first ap index; left col first.
    # native [rl, ap]; transpose -> [ap, rl]; rows ap ok; cols rl -> flip for lr
    assert out[0, 0] == arr[-1, 0]

    # no-op when requesting the native frame
    same = orient_slice_to_axes(arr, ctx, "horizontal", "rl,ap")
    assert (same == arr).all()

    import pytest as _pytest

    with _pytest.raises(ValueError):
        orient_slice_to_axes(arr, ctx, "horizontal", "si,ap")  # si not in-plane

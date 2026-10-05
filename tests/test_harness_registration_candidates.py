"""The atlas root mask and the border traces' working canvas."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image


def _make_slice(size: tuple[int, int] = (12, 8)) -> Image.Image:
    image = Image.new("RGB", size, color=(30, 40, 50))
    for y in range(size[1]):
        for x in range(size[0]):
            image.putpixel((x, y), (30 + x, 40 + y, 50))
    return image


def test_root_mask_produces_binary_alpha_at_target_size(monkeypatch):
    """`get_root_mask` slices annotation at the AP index for the requested
    plane, marks non-zero structure IDs as opaque (255) and zeros as
    transparent (0), and NEAREST-resizes to *target_size* so alpha stays
    binary (bilinear interpolation would halo the silhouette)."""
    from langslice.core.atlas import core as atlas_core

    # Annotation slab: top half has tissue (non-zero IDs), bottom half is bg.
    annotation = np.array(
        [
            [
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        ],
        dtype=np.int32,
    )
    atlas = SimpleNamespace(annotation=annotation)

    monkeypatch.setattr(
        atlas_core, "position_mm_to_index", lambda a, p, plane="coronal": 0
    )
    monkeypatch.setattr(atlas_core, "slice_axis_index", lambda ctx, plane: 0)
    monkeypatch.setattr(atlas_core, "atlas_space_context", lambda a: SimpleNamespace())
    monkeypatch.setattr(atlas_core, "orient_slice_for_display", lambda a, plane: a)

    target_size = (8, 8)  # (W, H) per PIL convention
    mask = atlas_core.get_root_mask(
        atlas, position_mm=0.0, target_size=target_size, plane="coronal"
    )

    assert mask.shape == (8, 8)  # numpy (H, W)
    assert mask.dtype == np.uint8
    unique_vals = set(np.unique(mask).tolist())
    assert unique_vals.issubset({0, 255})
    assert 0 in unique_vals and 255 in unique_vals
    # Top half opaque (was non-zero), bottom half transparent (was zero).
    assert (mask[0] == 255).all()
    assert (mask[-1] == 0).all()


def test_canvas_pad_grows_working_canvas_and_reports_offsets():
    from langslice.core.nonlinear.image_gen_registration import prepare_canvas

    canvas, unpadded, ox, oy, pad_px = prepare_canvas(
        _make_slice(),  # 12x8
        canvas_pad=0.25, native_canvas=False,
    )

    # pad = round(0.25 * 12) = 3 px per side -> 18x14 (aspect within range,
    # so no further clamp).
    assert pad_px == 3
    assert unpadded == (12, 8)
    assert canvas.size == (18, 14)
    assert (ox, oy) == (3.0, 3.0)


def test_extreme_aspect_ratio_clamps_into_the_supported_range():
    """gpt-image-2 accepts 1:3..3:1; a 4:1 strip black-pads down to 3:1."""
    from langslice.core.nonlinear.image_gen_registration import prepare_canvas

    canvas, unpadded, ox, oy, _pad_px = prepare_canvas(
        _make_slice((48, 12)), native_canvas=False, image_model="gpt-image-2",
    )

    assert canvas.size == (48, 16)  # ceil(48 / 3)
    assert (ox, oy) == (0.0, 2.0)  # centered pad
    assert unpadded == (48, 12)


def test_in_range_aspect_ratio_is_left_untouched():
    from langslice.core.nonlinear.image_gen_registration import prepare_canvas

    canvas, unpadded, ox, oy, _pad_px = prepare_canvas(
        _make_slice((24, 9)), native_canvas=False, image_model="gpt-image-2",
    )

    assert canvas.size == (24, 9)
    assert unpadded == (24, 9)
    assert (ox, oy) == (0.0, 0.0)

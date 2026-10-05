"""The working canvas is the image path's own output frame (2026-09-11)."""

from __future__ import annotations

import math

from PIL import Image

from langslice.core.nonlinear.image_frames import gemini_aspect_for, native_output_size
from langslice.core.nonlinear.image_gen_registration import prepare_canvas
from langslice.providers.images import _api_edit_size

_BUDGET = 1024 * 1536


def test_codex_frame_is_the_fixed_budget_at_the_canvas_aspect() -> None:
    # Measured 2026-09-11: 2197x1686 in -> 1431x1099 out.
    assert native_output_size(None, "openai-oauth", (2197, 1686)) == (1431, 1099)
    w, h = native_output_size(None, "openai-oauth", (1000, 3000)) or (0, 0)
    assert abs(w * h - _BUDGET) < 2000 and abs(w / h - 1 / 3) < 0.01


def test_api_frame_sits_on_the_16px_grid_and_is_requested_verbatim() -> None:
    w, h = native_output_size("gpt-image-2.5-sunburst", "openai-api", (2197, 1686)) or (0, 0)
    assert w % 16 == 0 and h % 16 == 0
    assert abs(w * h - _BUDGET) < _BUDGET * 0.03
    assert _api_edit_size(Image.new("RGB", (w, h))) == f"{w}x{h}"
    # A non-native canvas still gets the budget size at its aspect.
    assert _api_edit_size(Image.new("RGB", (2197, 1686))) == "1424x1104"


def test_gemini_frame_is_the_measured_fixed_frame_for_the_nearest_aspect() -> None:
    lite = "gemini-3.1-flash-lite-image"
    assert native_output_size(lite, "gemini-api", (2048, 1536)) == (1200, 896)
    assert native_output_size(lite, "gemini-api", (2048, 1400)) == (1264, 848)
    assert native_output_size(lite, "gemini-api", (1536, 2048)) == (896, 1200)
    flash = "gemini-3.1-flash-image"
    assert native_output_size(flash, "gemini-api", (2048, 1400), "512") == (624, 416)
    assert gemini_aspect_for(1.303) == "4:3" and gemini_aspect_for(0.75) == "3:4"


def test_unknown_lane_keeps_the_long_edge_rule() -> None:
    assert native_output_size(None, "none", (4000, 3000)) is None
    canvas, unpadded, ox, oy, pad = prepare_canvas(Image.new("RGB", (4000, 3000)), provider="none")
    assert canvas.size == (2048, 1536) and unpadded == (2048, 1536)
    assert (ox, oy, pad) == (0.0, 0.0, 0)


def test_native_canvas_is_exactly_the_output_frame_and_keeps_the_slide_box() -> None:
    image = Image.new("RGB", (3137, 2353), (20, 20, 20))
    canvas, unpadded, ox, oy, pad = prepare_canvas(
        image, canvas_pad=0.0365, provider="openai-oauth"
    )
    assert canvas.size == native_output_size(None, "openai-oauth", canvas.size)
    # The slide box scales with the canvas: its aspect is the slide's, and
    # origin + box stay inside the frame.
    assert math.isclose(unpadded[0] / unpadded[1], 3137 / 2353, rel_tol=0.01)
    assert ox >= 0 and oy >= 0 and ox + unpadded[0] <= canvas.width + 1
    assert oy + unpadded[1] <= canvas.height + 1
    assert pad == round(ox) or abs(pad - ox) <= 1


def test_gemini_canvas_is_padded_to_the_fixed_frame_never_cropped() -> None:
    image = Image.new("RGB", (3137, 2353), (20, 20, 20))
    canvas, unpadded, ox, oy, _ = prepare_canvas(
        image, image_model="gemini-3.1-flash-lite-image", provider="gemini-api"
    )
    assert canvas.size == (1200, 896)
    assert unpadded[0] <= 1200 and unpadded[1] <= 896
    assert ox + unpadded[0] <= 1200 + 1 and oy + unpadded[1] <= 896 + 1


def test_canvas_long_edge_overrides_the_native_frame() -> None:
    canvas, unpadded, *_ = prepare_canvas(
        Image.new("RGB", (3137, 2353)), provider="openai-oauth", canvas_long_edge=1024
    )
    assert canvas.width == 1024 or canvas.width == 1023
    assert unpadded == canvas.size

"""The working canvas is the image path's own output frame."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from langslice.core.nonlinear.image_frames import gemini_aspect_for, native_output_size
from langslice.core.nonlinear.image_gen_registration import prepare_canvas
from langslice.providers.images import _api_edit_size

_BUDGET = 1024 * 1536


def test_codex_frame_is_the_fixed_budget_at_the_canvas_aspect() -> None:
    # Measured: 2197x1686 in -> 1431x1099 out.
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


def _render_box(render_to_canvas: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """The render's pixel-edge corners on the canvas, in canvas pixel-edge numbers."""
    width, height = size
    corners = np.array([[-0.5, -0.5, 1], [width - 0.5, -0.5, 1],
                        [-0.5, height - 0.5, 1], [width - 0.5, height - 0.5, 1]]).T
    return (render_to_canvas @ corners)[:2].T + 0.5


def test_unknown_lane_keeps_the_long_edge_rule() -> None:
    assert native_output_size(None, "none", (4000, 3000)) is None
    canvas, render_to_canvas = prepare_canvas(Image.new("RGB", (4000, 3000)), provider="none")
    assert canvas.size == (2048, 1536)
    np.testing.assert_allclose(_render_box(render_to_canvas, (4000, 3000)),
                               [[0, 0], [2048, 0], [0, 1536], [2048, 1536]], atol=1e-9)


def test_native_canvas_is_exactly_the_output_frame_and_keeps_the_slide_box() -> None:
    image = Image.new("RGB", (3137, 2353), (20, 20, 20))
    canvas, render_to_canvas = prepare_canvas(image, provider="openai-oauth")
    # The frame of the long-edge layout (2048 x 1537), padded out to exactly.
    assert canvas.size == native_output_size(None, "openai-oauth", (2048, 1537))
    # The slide box scales with the canvas: its aspect is the slide's, and it
    # stays inside the frame, centred.
    box = _render_box(render_to_canvas, image.size)
    (x0, y0), (x1, y1) = box.min(axis=0), box.max(axis=0)
    assert math.isclose((x1 - x0) / (y1 - y0), 3137 / 2353, rel_tol=1e-9)
    assert x0 >= -1e-9 and y0 >= -1e-9 and x1 <= canvas.width + 1e-9
    assert y1 <= canvas.height + 1e-9
    assert math.isclose(x0 + x1, canvas.width) and math.isclose(y0 + y1, canvas.height)


def test_gemini_canvas_is_padded_to_the_fixed_frame_never_cropped() -> None:
    image = Image.new("RGB", (3137, 2353), (20, 20, 20))
    canvas, render_to_canvas = prepare_canvas(
        image, image_model="gemini-3.1-flash-lite-image", provider="gemini-api"
    )
    assert canvas.size == (1200, 896)
    box = _render_box(render_to_canvas, image.size)
    assert box.min() >= -1e-9
    assert box[:, 0].max() <= 1200 + 1e-9 and box[:, 1].max() <= 896 + 1e-9

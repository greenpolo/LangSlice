"""Hard boundary: with a known pixel size, atlas renders are placed at TRUE physical scale.

Fit-to-canvas is the fallback for an unknown pixel size only. It is not a
calibration: it fits the whole atlas plane, empty margins included, into a
tissue-cropped canvas, which on LSD_910 M01 drew the plate at 0.69x the tissue
width (2026-09-05) where true scale draws it at 1.05x. Removing the scale path
must fail here first.
"""

import numpy as np
from PIL import Image

from langslice.nonlinear import image_gen_helpers as helpers
from langslice.nonlinear import image_gen_registration as reg

WHY = "atlas renders must be placed at true physical scale when a pixel size is known"


def _content_bbox(canvas: Image.Image) -> tuple[int, int, int, int]:
    arr = np.asarray(canvas.convert("RGB"))
    ys, xs = np.nonzero(arr.any(axis=2))
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


class _Atlas:
    resolution = (25.0, 25.0, 25.0)


def test_fit_to_canvas_without_scale_fits_inside() -> None:
    canvas = (200, 120)
    for size in [(400, 100), (50, 40), (300, 300)]:
        out = reg._fit_to_canvas(Image.new("RGB", size, (255, 255, 255)), canvas)
        x0, y0, x1, y1 = _content_bbox(out)
        assert out.size == canvas
        assert (x1 - x0 == canvas[0]) or (y1 - y0 == canvas[1]), size
        assert x0 >= 0 and y0 >= 0 and x1 <= canvas[0] and y1 <= canvas[1]


def test_fit_to_canvas_with_scale_scales_exactly_and_centers_focus() -> None:
    out = reg._fit_to_canvas(Image.new("RGB", (100, 50), (255, 255, 255)), (400, 300), scale=2.0)
    x0, y0, x1, y1 = _content_bbox(out)
    assert (x1 - x0, y1 - y0) == (200, 100), WHY
    assert (x0 + x1) // 2 == 200 and (y0 + y1) // 2 == 150
    # focus at the image's left edge lands that edge on the canvas center
    out = reg._fit_to_canvas(
        Image.new("RGB", (100, 50), (255, 255, 255)), (400, 300), scale=2.0, focus=(0.0, 0.5)
    )
    x0, _, _, _ = _content_bbox(out)
    assert x0 == 200


def test_physical_scale_puts_anatomy_at_true_size(monkeypatch) -> None:
    # annotation plane 60 x 40 voxels at 25 um; render pre-upscaled 10x; canvas 5 um/px
    plane = np.ones((40, 60), dtype=np.int64)
    monkeypatch.setattr(helpers, "_annotation_slice", lambda *a, **k: plane)
    render = Image.new("RGB", (600, 400), (255, 255, 255))
    canvas_um = reg._canvas_um_per_px(2.5, Image.new("RGB", (4000, 2000)), (2000, 1000))
    assert canvas_um == 5.0
    scale = reg._physical_scale_for(render, _Atlas(), 5.0, "coronal", canvas_um)
    assert scale is not None and abs(scale - 0.5) < 1e-9, WHY
    placed = reg._fit_to_canvas(render, (2000, 1000), scale=scale)
    x0, _, x1, _ = _content_bbox(placed)
    assert abs((x1 - x0) - 60 * 25.0 / canvas_um) <= 1, WHY  # 300 px = 60 voxels x 25 um / 5 um/px


def test_no_pixel_size_means_fit_to_canvas() -> None:
    assert reg._canvas_um_per_px(None, Image.new("RGB", (10, 10)), (10, 10)) is None
    render = Image.new("RGB", (10, 10))
    assert reg._physical_scale_for(render, _Atlas(), 1.0, "coronal", None) is None

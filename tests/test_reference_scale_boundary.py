"""Hard boundary: atlas renders are FIT TO CANVAS, never physically scaled.

True-physical placement from a pixel size (2026-08-29 to 2026-09-05) drew the
atlas 20-30% larger than the shrunken tissue and the image model copied the
oversized plate instead of repainting the tissue (painting dice 0.52 vs 0.62
fit-to-canvas, visibly worse). Re-adding a scale path must fail here first.
"""

import inspect

import numpy as np
from PIL import Image

from langslice.nonlinear import image_gen_registration as reg

WHY = "atlas renders must stay fit-to-canvas; see nonlinear/CLAUDE.md (2026-09-05 regression)"


def _content_bbox(canvas: Image.Image) -> tuple[int, int, int, int]:
    arr = np.asarray(canvas.convert("RGB"))
    ys, xs = np.nonzero(arr.any(axis=2))
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def test_fit_to_canvas_never_overflows_or_underfills() -> None:
    canvas = (200, 120)
    for size in [(400, 100), (50, 40), (300, 300)]:
        out = reg._fit_to_canvas(Image.new("RGB", size, (255, 255, 255)), canvas)
        x0, y0, x1, y1 = _content_bbox(out)
        assert out.size == canvas
        touches_long_axis = (x1 - x0 == canvas[0]) or (y1 - y0 == canvas[1])
        assert touches_long_axis, (size, WHY)
        assert x0 >= 0 and y0 >= 0 and x1 <= canvas[0] and y1 <= canvas[1]  # never cropped


def test_no_physical_scale_plumbing_exists() -> None:
    params = inspect.signature(reg._fit_to_canvas).parameters
    assert set(params) == {"image", "size", "fill"}, WHY
    helpers = (
        "_physical_scale_for",
        "_pad_to_contain_atlas",
        "_canvas_um_per_px",
        "_anatomy_focus",
    )
    for name in helpers:
        assert not hasattr(reg, name), (name, WHY)

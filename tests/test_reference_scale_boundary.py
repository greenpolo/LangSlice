"""Hard boundary: the map Elastix registers sits where the model saw it.

Until 2026-09-12 this file guarded the opposite rule — atlas renders placed at
TRUE physical scale from a known pixel size, fit-to-canvas being only a
fallback. The April lineup removed physical placement from this path: measured
on LSD_910 M01 A_02, a plate drawn at the tissue's own size was COPIED by the
model on 7 of 8 draws instead of deformed, and the lineup that beat every
other configuration was benchmarked without it. Physical placement lives on in
``linear/``, where a person, not a model, reads the overlay.

What has to hold now: the model's Image 1 (its map, letterboxed into the
section's aspect) and the Elastix moving map (the same render fit to the
section canvas) are the SAME geometry. If they ever drift apart, every
boundary the model paints lands somewhere else for the fit.
"""

import numpy as np
from PIL import Image

from langslice.nonlinear import image_gen_registration as reg

WHY = "the Elastix moving map must sit in the geometry the model was shown"


def _content_bbox(canvas: Image.Image) -> tuple[int, int, int, int]:
    arr = np.asarray(canvas.convert("RGB"))
    ys, xs = np.nonzero(arr.any(axis=2))
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def test_fit_to_canvas_fits_inside_and_centers() -> None:
    canvas = (200, 120)
    for size in [(400, 100), (50, 40), (300, 300)]:
        out = reg._fit_to_canvas(Image.new("RGB", size, (255, 255, 255)), canvas)
        x0, y0, x1, y1 = _content_bbox(out)
        assert out.size == canvas
        assert (x1 - x0 == canvas[0]) or (y1 - y0 == canvas[1]), size
        assert x0 >= 0 and y0 >= 0 and x1 <= canvas[0] and y1 <= canvas[1]
        assert abs((x0 + x1) // 2 - canvas[0] // 2) <= 1
        assert abs((y0 + y1) // 2 - canvas[1] // 2) <= 1


def test_fit_to_canvas_never_stretches() -> None:
    render = Image.new("RGB", (300, 100), (255, 255, 255))
    out = reg._fit_to_canvas(render, (400, 400))
    x0, y0, x1, y1 = _content_bbox(out)
    assert abs((x1 - x0) / (y1 - y0) - 3.0) < 0.05, WHY


def test_fit_to_canvas_keeps_a_label_map_flat() -> None:
    """NEAREST by default: a blended boundary classifies as a third region."""
    plate = Image.new("RGB", (40, 30), (255, 0, 0))
    plate.paste(Image.new("RGB", (10, 30), (0, 255, 0)), (20, 0))
    out = reg._fit_to_canvas(plate, (400, 300))
    colors = {tuple(c) for c in np.asarray(out).reshape(-1, 3)}
    assert colors <= {(0, 0, 0), (255, 0, 0), (0, 255, 0)}


def test_the_model_frame_and_the_elastix_frame_are_the_same_geometry() -> None:
    """Letterbox-then-resize (what the model saw) == fit-to-canvas (the fit)."""
    plate = Image.new("RGB", (456, 320), (0, 0, 0))
    plate.paste(Image.new("RGB", (300, 200), (255, 255, 255)), (78, 60))
    canvas = (2048, 1536)

    shown = reg.letterbox_to_aspect(
        reg.upscale_to_min_long_edge(plate, Image.Resampling.NEAREST),
        canvas[0] / canvas[1],
    ).resize(canvas, Image.Resampling.NEAREST)
    registered = reg._fit_to_canvas(plate, canvas)

    a = _content_bbox(shown)
    b = _content_bbox(registered)
    assert max(abs(x - y) for x, y in zip(a, b, strict=True)) <= 2, WHY

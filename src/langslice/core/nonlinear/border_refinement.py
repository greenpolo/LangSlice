"""The yellow lines of a border trace: drawn onto a section, and read back off a reply.

:func:`smooth_border_overlay` draws a placed atlas plane's family borders
on the section (route "supplied"'s Image 1 or 2), :func:`border_overlay`
draws a line mask on it, and :func:`extract_thinned_lines` reads the
model's yellow lines off a raw reply onto the canvas. Nothing here calls a
model or fits anything; the fit of the lines is ``fit_deformable``'s
(:mod:`langslice.core.deformation`).
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from langslice.core.atlas.render import placed_border_coverage

__all__ = [
    "border_overlay",
    "extract_thinned_lines",
    "smooth_border_overlay",
    "thin",
    "yellow_mask",
]


def yellow_mask(rgb: np.ndarray) -> np.ndarray:
    """Extract saturated yellow, allowing the model's orange/green color drift."""
    hsv = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    hue, sat, val = (hsv[..., i] for i in range(3))
    return (hue >= 18) & (hue <= 42) & (sat >= 90) & (val >= 110)


def thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen skeletonization, retaining connected one-pixel boundaries."""
    img = np.asarray(mask, dtype=bool).astype(np.uint8)
    while True:
        removed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            p2, p3, p4 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:]
            p5, p6, p7 = p[2:, 2:], p[2:, 1:-1], p[2:, :-2]
            p8, p9 = p[1:-1, :-2], p[:-2, :-2]
            ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            neighbours = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = np.zeros_like(img)
            for i in range(8):
                transitions += ((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.uint8)
            if step == 0:
                corner = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                corner = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            drop = (img == 1) & (neighbours >= 2) & (neighbours <= 6) & (transitions == 1)
            drop &= corner
            if drop.any():
                img[drop] = 0
                removed = True
        if not removed:
            return img.astype(bool)


def smooth_border_overlay(
    image: Image.Image,
    labels: np.ndarray,
    atlas_to_image: np.ndarray,
    *,
    width_px: float = 2.0,
    smoothing_px: float = 0.8,
    supersample: int = 3,
) -> Image.Image:
    """Placed atlas region boundaries as smooth, single, antialiased yellow lines.

    *labels* is an atlas-resolution region map and *atlas_to_image* the 3x3
    (or 2x3) map from its pixel centres to *image*'s. A nearest-neighbour warp
    of the label map magnifies the atlas grid into a staircase; instead each
    region's indicator is blurred by *smoothing_px* atlas pixels, warped
    bilinearly onto a *supersample*-times finer grid, and every fine pixel
    takes the region that covers it most. Boundaries between neighbouring
    fine pixels are one shared line (tracing each region separately draws a
    shared edge twice, one atlas pixel apart), widened to *width_px* image
    pixels and area-averaged back down, so the edges are antialiased. The
    coverage itself is :func:`langslice.core.atlas.render.placed_border_coverage`,
    shared with the deformable fit's border images.
    """
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    coverage = placed_border_coverage(
        labels, atlas_to_image, image.size, width_px=width_px,
        smoothing_px=smoothing_px, supersample=supersample,
    )
    alpha = coverage[..., None]
    blended = base * (1.0 - alpha) + np.array([255.0, 255.0, 0.0]) * alpha
    return Image.fromarray(np.rint(blended).astype(np.uint8))


def border_overlay(image: Image.Image, mask: np.ndarray, width: int = 1) -> Image.Image:
    """Replace only boundary pixels on the original photograph."""
    pixels = np.array(image.convert("RGB"))
    drawn = np.asarray(mask, dtype=bool)
    if width > 1:
        drawn = cv2.dilate(drawn.astype(np.uint8), np.ones((width, width), np.uint8)) > 0
    pixels[drawn] = (255, 255, 0)
    return Image.fromarray(pixels)


def extract_thinned_lines(raw: Image.Image, canvas_size: tuple[int, int]) -> np.ndarray:
    """Boolean thinned yellow-line mask from a raw model reply, on *canvas_size*.

    Every trace's line extraction, and route "atlas"'s pass 2 input (pass
    1's lines redrawn on the clean tissue at the same canvas). Extracts
    before any interpolation, so tissue colors cannot blend into yellow;
    crops back to the canvas aspect first: a lane with a fixed output frame
    answers at its own aspect, ours letterboxed inside it, so crop back
    rather than stretch.
    """
    from langslice.core.nonlinear.image_gen_registration import crop_to_aspect

    raw_mask = Image.fromarray(yellow_mask(np.asarray(raw.convert("RGB"))))
    framed = crop_to_aspect(raw_mask, canvas_size[0] / canvas_size[1])
    return thin(np.asarray(framed.resize(canvas_size, Image.Resampling.NEAREST)))

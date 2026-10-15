"""Model-facing canvas geometry: the working canvas (:func:`prepare_canvas`),
the section turned upright on it, and the aspect-ratio helpers."""

from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image

from langslice.core.nonlinear.image_frames import aspect_ratio_limits, native_output_size
from langslice.core.nonlinear.upright import Upright

#: Relative aspect-ratio difference above which a returned image is treated as
#: letterboxed inside a different frame and cropped back. Lanes with fixed
#: output frames answer at the nearest legal aspect, a percent or two off.
_ASPECT_TOLERANCE = 0.005

_MAX_LONG_EDGE = 2048


def crop_to_aspect(
    image: Image.Image, aspect: float, tolerance: float = _ASPECT_TOLERANCE
) -> Image.Image:
    """Center-crop *image* to the given width/height ratio, never stretch.

    The model is asked for the frame it was given; a lane with fixed output
    frames returns the nearest legal one instead, with our frame letterboxed
    inside it. Cropping the excess back off puts every painted boundary where
    the model put it — resampling the whole thing to the frame would slide
    them all off the tissue.
    """
    width, height = image.size
    if abs((width / height) / aspect - 1.0) <= tolerance:
        return image
    if width / height > aspect:
        new_width = round(height * aspect)
        left = (width - new_width) // 2
        return image.crop((left, 0, left + new_width, height))
    new_height = round(width / aspect)
    top = (height - new_height) // 2
    return image.crop((0, top, width, top + new_height))


def _background_color(image: Image.Image) -> tuple[int, int, int]:
    """Median color of the image's border ring — the slide background.

    Padding in this color (instead of black) keeps the margin visually
    continuous with the image's own background, for both the image model
    and the downstream foreground mask.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    ring = max(2, int(round(0.02 * max(rgb.shape[:2]))))
    edges = np.concatenate(
        [
            rgb[:ring].reshape(-1, 3),
            rgb[-ring:].reshape(-1, 3),
            rgb[:, :ring].reshape(-1, 3),
            rgb[:, -ring:].reshape(-1, 3),
        ]
    )
    r, g, b = (int(v) for v in np.median(edges, axis=0))
    return (r, g, b)


def prepare_canvas(
    image: Image.Image,
    *,
    image_model: str | None = None,
    provider: str | None = None,
    upright: Upright | None = None,
) -> tuple[Image.Image, np.ndarray]:
    """The working canvas: the section render turned upright, scaled, padded
    and aspect-snapped in ONE resample.

    Returns ``(canvas, render_to_canvas)``, the 3x3 from render pixel centres
    to canvas pixel centres (integers at pixel centres, OpenCV's convention).

    *upright* (:func:`~langslice.core.nonlinear.upright.upright`) turns the
    render so its ``up`` points straight up and, with ``through``, shifts it
    so that point lies on the canvas's vertical centre line; the canvas holds
    the whole turned render (never a crop) and pads the rest with the
    render's own background colour. Without it the render stands as it is,
    centred. A turn is a rotation and a shift only, never a mirror.

    The aspect-ratio clamp pads the short axis (pad only, never crop): in
    edit mode the model paints on ITS canvas, and resampling a mismatched
    ratio back onto the slice would silently undo the pixel alignment.

    The canvas is the frame the image path returns
    (:func:`image_frames.native_output_size`) when it has one: the layout is
    worked out at the long-edge rule, scaled to fit that frame, and padded
    out to it exactly, so the model edits on the output's own pixel grid: an
    input the model must rescale to its output is a pixel the output cannot
    carry, paid for twice (input tokens in, a resample out). The render is
    resampled ONCE, straight to the canvas (:func:`_resample`).
    """
    stance = upright or Upright()
    fill = _background_color(image)
    limits = aspect_ratio_limits(image_model, provider)
    width, height = image.size
    turn = stance.rotation()
    # Edge coordinates (pixel i spans [i, i + 1]) about the render's centre.
    centre = np.array([width / 2.0, height / 2.0])
    corners = np.array([[0.0, 0.0], [width, 0.0], [0.0, height], [width, height]]) - centre
    turned = corners @ turn.T
    xs, ys = turned[:, 0], turned[:, 1]
    if stance.through is not None:
        mid = float((turn @ (np.asarray(stance.through, dtype=np.float64) + 0.5 - centre))[0])
        half = float(np.abs(xs - mid).max())
        left, box_w = mid - half, 2.0 * half
    else:
        left, box_w = float(xs.min()), float(xs.max() - xs.min())
    top, box_h = float(ys.min()), float(ys.max() - ys.min())

    def snapped(scale: float) -> tuple[int, int]:
        """The canvas holding the turned render at *scale*, aspect-clamped."""
        cw, ch = _cover(box_w * scale), _cover(box_h * scale)
        if limits:
            if cw / ch > limits[1]:
                ch = math.ceil(cw / limits[1])
            elif cw / ch < limits[0]:
                cw = math.ceil(ch * limits[0])
        return cw, ch

    scale = min(1.0, _MAX_LONG_EDGE / max(box_w, box_h))
    size = snapped(scale)
    frame = native_output_size(image_model, provider, size)
    if frame is not None:
        # Fit to the frame and pad out to it exactly (never crop): the request IS the frame.
        scale *= min(frame[0] / size[0], frame[1] / size[1])
        size = frame
    offset = np.array([(size[0] - box_w * scale) / 2.0 - scale * left,
                       (size[1] - box_h * scale) / 2.0 - scale * top])
    # canvas_edge = scale * turn @ (render_edge - centre) + offset; centres are edges - 0.5.
    linear = scale * turn
    shift = linear @ (0.5 - centre) + offset - 0.5
    render_to_canvas = np.array([[*linear[0], shift[0]], [*linear[1], shift[1]], [0.0, 0.0, 1.0]])
    return _resample(image, render_to_canvas, size, fill, scale), render_to_canvas


def _cover(length: float) -> int:
    """Whole pixels covering *length* (a float's last-bit excess ignored)."""
    return max(1, math.ceil(length - 1e-6))


def _resample(
    image: Image.Image, matrix: np.ndarray, size: tuple[int, int],
    fill: tuple[int, int, int], scale: float,
) -> Image.Image:
    """*image* drawn through *matrix* (3x3, pixel centres) onto a *size* canvas
    filled with *fill*, interpolated once (Lanczos).

    Shrinking is anti-aliased first, on the source grid, by the Lanczos-3
    kernel stretched to the output's pixel spacing (:func:`_lanczos_kernel`):
    the filter PIL's Lanczos resize applies, so a section shrunk without a
    turn looks as PIL's resize draws it, and fine tissue texture does not
    alias. An exact copy (no turn, no resize, a whole-pixel shift) is
    pasted, not interpolated."""
    rgb = image.convert("RGB")
    whole = np.allclose(matrix[:2, :2], np.eye(2)) and np.allclose(
        matrix[:2, 2], np.round(matrix[:2, 2]))
    if whole:
        canvas = Image.new("RGB", size, fill)
        canvas.paste(rgb, (int(round(matrix[0, 2])), int(round(matrix[1, 2]))))
        return canvas
    pixels = np.asarray(rgb, dtype=np.float32)
    if scale < 1.0:
        kernel = _lanczos_kernel(scale)
        pixels = cv2.sepFilter2D(pixels, -1, kernel, kernel, borderType=cv2.BORDER_REFLECT)
    drawn = cv2.warpAffine(
        pixels, matrix[:2], size, flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(float(v) for v in fill),
    )
    return Image.fromarray(np.clip(np.rint(drawn), 0, 255).astype(np.uint8))


def _lanczos_kernel(scale: float, lobes: int = 3) -> np.ndarray:
    """The Lanczos low-pass for shrinking by *scale* (< 1), sampled on the
    source grid: ``lanczos(x * scale) * scale``, normalized to sum 1."""
    radius = int(math.ceil(lobes / scale))
    x = np.arange(-radius, radius + 1) * scale
    kernel = np.sinc(x) * np.sinc(x / lobes)
    kernel[np.abs(x) >= lobes] = 0.0
    return (kernel / kernel.sum()).astype(np.float32)

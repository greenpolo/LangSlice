"""Model-facing canvas geometry: the working canvas (:func:`prepare_canvas`)
and the aspect-ratio helpers."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from langslice.core.nonlinear.image_frames import aspect_ratio_limits, native_output_size

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
) -> tuple[Image.Image, tuple[int, int], float, float]:
    """Downsample, pad, and aspect-snap the slice into the working canvas.

    Returns ``(slice_image, unpadded_size, origin_x, origin_y)``.
    Padding uses the slice's own background color, and the aspect-ratio
    clamp pads the short axis (pad only, never crop): in edit mode the
    model paints on ITS canvas, and resampling a mismatched ratio back
    onto the slice would silently undo the pixel alignment.

    The canvas is the frame the image path returns
    (:func:`image_frames.native_output_size`) when it has one: the layout is
    worked out at the long-edge rule, scaled to fit that frame, and padded
    out to it exactly, so the model edits on the output's own pixel grid: an
    input the model must rescale to its output is a pixel the output cannot
    carry, paid for twice (input tokens in, a resample out). The slice is
    resampled ONCE, straight from the original to its final size.
    """
    fill = _background_color(image)
    limits = aspect_ratio_limits(image_model, provider)

    def layout(scale: float) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        """(slice size, snapped canvas size, slice offset) at *scale*."""
        tw, th = _target_size_for_slice(image)
        tw, th = max(1, round(tw * scale)), max(1, round(th * scale))
        cw, ch = tw, th
        if limits:
            if cw / ch > limits[1]:
                ch = math.ceil(cw / limits[1])
            elif cw / ch < limits[0]:
                cw = math.ceil(ch * limits[0])
        offset = ((cw - tw) // 2, (ch - th) // 2)
        return (tw, th), (cw, ch), offset

    scale = 1.0
    _, canvas0, _ = layout(1.0)
    frame = native_output_size(image_model, provider, canvas0)
    if frame is not None:
        scale = min(frame[0] / canvas0[0], frame[1] / canvas0[1])
    target_size, canvas_size, offset = layout(scale)
    if frame is not None:
        # Pad out to the frame exactly (never crop): the request IS the frame.
        extra = ((frame[0] - canvas_size[0]) // 2, (frame[1] - canvas_size[1]) // 2)
        canvas_size = frame
        offset = (offset[0] + extra[0], offset[1] + extra[1])

    slice_image = _resize_if_needed(image, target_size)
    if canvas_size != target_size:
        canvas = Image.new("RGB", canvas_size, fill)
        canvas.paste(slice_image, offset)
        slice_image = canvas
    return slice_image, target_size, float(offset[0]), float(offset[1])


def _target_size_for_slice(image: Image.Image) -> tuple[int, int]:
    width, height = image.size
    long_edge = max(width, height)
    limit = _MAX_LONG_EDGE
    if long_edge <= limit:
        return width, height

    scale = limit / float(long_edge)
    return int(width * scale), int(height * scale)


def _resize_if_needed(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    if image.size == size:
        return image.convert("RGB")
    return image.convert("RGB").resize(size, resample=Image.Resampling.LANCZOS)

"""Model-facing canvas geometry and the outlined atlas template.

The geometry both border traces share: the working canvas
(:func:`prepare_canvas`), the aspect-ratio helpers, and the outlined
grayscale atlas plate route "atlas" shows the model instead of a placement
(:func:`outlined_atlas_template`).
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from PIL import Image

from langslice.core.atlas.render import annotation_slice
from langslice.core.nonlinear.image_frames import aspect_ratio_limits, native_output_size
from langslice.core.nonlinear.image_gen_helpers import (
    _extract_borders_from_classified,
    _merge_classified,
    line_width_px,
)
from langslice.core.space import Plane

#: Long edge every model-facing atlas render is NEAREST/LANCZOS-upscaled to at
#: least. The atlas is coarse (a 25um coronal plate is ~456px across); below
#: this the thin bands and small nuclei the model has to place stop being
#: legible.
MODEL_MAP_MIN_LONG_EDGE = 1024

#: Relative aspect-ratio difference above which a returned image is treated as
#: letterboxed inside a different frame and cropped back. Lanes with fixed
#: output frames answer at the nearest legal aspect, a percent or two off.
_ASPECT_TOLERANCE = 0.005

_MAX_LONG_EDGE = 2048


def letterbox_to_aspect(
    image: Image.Image, aspect: float, fill: tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    """Center *image* on a *fill* canvas of the given width/height ratio.

    Grows one axis, never scales or crops: the atlas renders keep their own
    geometry and still arrive in the section's frame, which is what lets the
    prompt tell the model to answer in that one shared frame.
    """
    width, height = image.size
    if width / height < aspect:
        canvas_size = (round(height * aspect), height)
    else:
        canvas_size = (width, round(width / aspect))
    canvas = Image.new("RGB", canvas_size, fill)
    canvas.paste(
        image.convert("RGB"),
        ((canvas_size[0] - width) // 2, (canvas_size[1] - height) // 2),
    )
    return canvas


def upscale_to_min_long_edge(
    image: Image.Image,
    resample: Image.Resampling,
    min_long_edge: int = MODEL_MAP_MIN_LONG_EDGE,
) -> Image.Image:
    """Enlarge *image* until its long edge reaches *min_long_edge*; never shrink."""
    factor = max(1.0, min_long_edge / max(image.size))
    if factor == 1.0:
        return image
    return image.resize(
        (round(image.width * factor), round(image.height * factor)), resample
    )


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
    canvas_pad: float = 0.0,
    image_model: str | None = None,
    provider: str | None = None,
    native_canvas: bool = True,
    canvas_long_edge: int | None = None,
    size_tier: str | None = None,
) -> tuple[Image.Image, tuple[int, int], float, float, int]:
    """Downsample, pad, and aspect-snap the slice into the working canvas.

    Returns ``(slice_image, unpadded_size, origin_x, origin_y, pad_px)``.
    Padding uses the slice's own background color, and the aspect-ratio
    clamp pads the short axis (pad only, never crop): in edit mode the
    model paints on ITS canvas, and resampling a mismatched ratio back
    onto the slice would silently undo the pixel alignment.

    ``native_canvas`` (the default) sizes the canvas to the frame the image
    path returns (:func:`image_frames.native_output_size`): the layout is
    worked out at the long-edge rule, scaled to fit that frame, and padded
    out to it exactly, so the model edits on the output's own pixel grid: an
    input the model must rescale to its output is a pixel the output cannot
    carry, paid for twice (input tokens in, a resample out). The slice is
    resampled ONCE, straight from the original to its final size.
    ``canvas_long_edge`` instead pins the long edge (the model is shown a
    smaller canvas and its output is resampled DOWN onto it).
    """
    fill = _background_color(image)
    limits = aspect_ratio_limits(image_model, provider)

    def layout(scale: float) -> tuple[tuple[int, int], int, tuple[int, int], tuple[int, int]]:
        """(slice size, pad px, snapped canvas size, slice offset) at *scale*."""
        tw, th = _target_size_for_slice(image, canvas_long_edge)
        tw, th = max(1, round(tw * scale)), max(1, round(th * scale))
        pad = int(round(float(canvas_pad) * max(tw, th))) if canvas_pad else 0
        cw, ch = tw + 2 * pad, th + 2 * pad
        if limits:
            if cw / ch > limits[1]:
                ch = math.ceil(cw / limits[1])
            elif cw / ch < limits[0]:
                cw = math.ceil(ch * limits[0])
        offset = ((cw - tw) // 2, (ch - th) // 2)
        return (tw, th), pad, (cw, ch), offset

    scale = 1.0
    frame: tuple[int, int] | None = None
    if native_canvas and canvas_long_edge is None:
        _, _, canvas0, _ = layout(1.0)
        frame = native_output_size(image_model, provider, canvas0, size_tier)
        if frame is not None:
            scale = min(frame[0] / canvas0[0], frame[1] / canvas0[1])
    target_size, pad_px, canvas_size, offset = layout(scale)
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
    return slice_image, target_size, float(offset[0]), float(offset[1]), pad_px


def _target_size_for_slice(
    image: Image.Image, max_long_edge: int | None = None
) -> tuple[int, int]:
    width, height = image.size
    long_edge = max(width, height)
    limit = max_long_edge or _MAX_LONG_EDGE
    if long_edge <= limit:
        return width, height

    scale = limit / float(long_edge)
    return int(width * scale), int(height * scale)


def _resize_if_needed(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    if image.size == size:
        return image.convert("RGB")
    return image.convert("RGB").resize(size, resample=Image.Resampling.LANCZOS)


def _orient_pil(
    image: Image.Image, atlas: Any, plane: Plane, image_axes: str | None,
    atlas_mirror_lr: bool = False,
) -> Image.Image:
    """Rotate/flip an atlas render into the user's image frame (no-op if unset)."""
    if image_axes:
        from langslice.core.space import atlas_space_context, orient_slice_to_axes

        arr = orient_slice_to_axes(np.asarray(image), atlas_space_context(atlas), plane, image_axes)
        image = Image.fromarray(arr)
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if atlas_mirror_lr else image


def _model_facing_template(
    atlas: Any,
    position_mm: float,
    plane: Plane,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
) -> Image.Image:
    """The grayscale atlas template of one plane, contrast-normalized for the model.

    ``get_reference_slice`` normalizes by the volume's brightest voxel, which
    leaves a typical plate dim; the model reads structure off this image, so
    it is restretched linearly on the 99.5th percentile of the plate's own
    tissue.
    """
    from langslice.core.atlas import get_reference_slice

    gray = np.asarray(
        get_reference_slice(
            atlas, position_mm, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg
        ),
        dtype=np.float32,
    )
    tissue = gray[gray > 0]
    peak = float(np.percentile(tissue, 99.5)) if tissue.size else 0.0
    scaled = np.clip(255.0 * gray / max(peak, 1.0), 0, 255).astype(np.uint8)
    return Image.fromarray(scaled, mode="L").convert("RGB")


def _overlay_borders(
    base_image: Image.Image, borders: np.ndarray, line_px: int = 1
) -> Image.Image:
    """Draw atlas-region borders over a base image.

    Yellow core on a black rim: readable on violet Nissl, white brightfield,
    and dark fluorescence alike (plain cyan vanished on cyan-tinted Nissl).
    ``line_px`` grows the yellow core before the rim is drawn, so a render at
    a different working resolution than a 2048px canvas still
    gets a legible (not hairline-thin, not bloated) line — see
    :func:`~langslice.core.nonlinear.image_gen_helpers.line_width_px`.
    """
    overlay_rgb = np.asarray(base_image.convert("RGB"), dtype=np.uint8).copy()
    core = np.asarray(borders) > 0
    kernel = np.ones((3, 3), np.uint8)
    if line_px > 1:
        core = cv2.dilate(core.astype(np.uint8), kernel, iterations=line_px - 1) > 0
    rim = cv2.dilate(core.astype(np.uint8), kernel) > 0
    overlay_rgb[rim] = (0, 0, 0)
    overlay_rgb[core] = (255, 255, 0)
    return Image.fromarray(overlay_rgb, mode="RGB")


def outlined_atlas_template(
    atlas: Any,
    position_mm: float,
    plane: Plane,
    *,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    image_axes: str | None = None,
    atlas_mirror_lr: bool = False,
    section_aspect: float | None = None,
    native_labels: np.ndarray | None = None,
) -> Image.Image:
    """The grayscale atlas plate with thin yellow family borders — route "atlas"'s
    only atlas-facing input.

    Oriented by ``image_axes`` then explicit ``atlas_mirror_lr``, never inferred
    from a silhouette. ``native_labels`` is the already-oriented annotation
    plane when the caller has it (the oblique resample is the expensive step);
    otherwise it is sampled and oriented here. Labels are NEAREST-upscaled to
    the frame the plate is LANCZOS-upscaled to, then letterboxed to the
    section's aspect.
    """
    template = _orient_pil(
        _model_facing_template(atlas, position_mm, plane, pitch_deg, yaw_deg),
        atlas, plane, image_axes, atlas_mirror_lr,
    )
    if native_labels is None:
        native_labels = annotation_slice(
            atlas, position_mm, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg,
        )
        if image_axes:
            from langslice.core.space import atlas_space_context, orient_slice_to_axes

            native_labels = orient_slice_to_axes(
                native_labels, atlas_space_context(atlas), plane, image_axes
            )
        if atlas_mirror_lr:
            native_labels = np.fliplr(native_labels)
    if native_labels.shape != (template.height, template.width):
        raise ValueError("Native atlas labels and grayscale template must share a pixel frame")

    grown = upscale_to_min_long_edge(template, Image.Resampling.LANCZOS)
    if grown.size != template.size:
        labels_up = np.asarray(
            Image.fromarray(native_labels.astype(np.int32), mode="I").resize(
                grown.size, Image.Resampling.NEAREST
            ),
            dtype=np.int64,
        )
    else:
        labels_up = native_labels
    borders = _extract_borders_from_classified(_merge_classified(labels_up, atlas))
    outlined = _overlay_borders(grown, borders, line_width_px(max(grown.size)))
    if section_aspect is not None:
        outlined = letterbox_to_aspect(outlined, section_aspect)
    return outlined

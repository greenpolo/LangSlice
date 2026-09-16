"""Harness candidate pipeline for dense image-gen registration."""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image

from langslice.agent_trace import image_part_from_pil, json_part, runtime_event
from langslice.atlas import load_atlas
from langslice.atlas.recolor import color_lut, use_palette
from langslice.image_prep import foreground_mask
from langslice.nonlinear.image_gen_helpers import (
    _classified_to_rgb,
    _classify_pixels_to_region_ids,
    _compute_deformation_field,
    _despeckle_classified,
    _elastix_report,
    _extract_borders_from_classified,
    _extract_visualign_markers,
    _generate_colored_region_slice,
    _merge_classified,
    _mm2_per_pixel,
    _region_label,
    _region_ledger,
    _register_region_maps,
    _registration_rgb,
    _run_elastix_april_borders,
    _run_inverse_warp_for_slice,
    _warp_classified_labels,
    generation_report,
)
from langslice.nonlinear.model_prompts import (
    aspect_ratio_limits,
    base_segmentation_prompt,
    native_output_size,
)
from langslice.nonlinear.prior import build_silhouette_prior
from langslice.nonlinear.providers import (
    SegmentationGenerationRequest,
    generate_warped_segmentation_image,
)
from langslice.nonlinear.types import (
    Deformation,
    GeneratedSegmentation,
    RegistrationAnnotationSession,
    RegistrationCandidate,
)
from langslice.providers.registry import canonical_provider
from langslice.space import Plane

_MAX_LONG_EDGE = 2048

#: Which Elastix stage fits the atlas to the painting. ``"rgb"`` is the
#: measured default (joint RGB at family granularity, affine + B-spline, NCC);
#: ``"april-borders"`` is the April 2026 stage: B-spline only on single-pixel
#: border images, mean squares, grid 64, bending 10 (an experiment arm).
ElastixStage = Literal["rgb", "april-borders"]

#: Long edge every model-facing atlas render is NEAREST-upscaled to at least.
#: The atlas is coarse (a 25um coronal plate is ~456px across); below this the
#: thin bands and small nuclei the model has to place stop being legible.
MODEL_MAP_MIN_LONG_EDGE = 1024

#: Relative aspect-ratio difference above which a returned image is treated as
#: letterboxed inside a different frame and cropped back. Lanes with fixed
#: output frames answer at the nearest legal aspect, a percent or two off.
_ASPECT_TOLERANCE = 0.005

#: Off-palette foreground fraction above which a draw is called translucent
#: and dropped from the vote. Clean draws measure 0.01-0.03, translucent ones
#: 0.10-0.17, so the gate sits between the two populations.
_MAX_OFF_PALETTE = 0.08


def _off_palette_fraction(model_output_rgb: np.ndarray, classified: np.ndarray) -> float:
    """Share of a painting's foreground that classifies as background.

    The translucency detector. A "translucent" draw half-preserves the
    tissue texture instead of painting flat atlas color, and those blended
    pixels land far from every palette color, so the classifier calls them
    background. Measured over 3 slices x 8 draws: 0.01-0.03 for clean
    paintings, 0.10-0.17 for translucent ones. Foreground is the
    classifier's own darkness cut (max channel >= 20), and *classified* must
    be the raw classification — before the preserved-background mask (a
    preserved white slide is foreground by this test) and before despeckle.
    """
    foreground = model_output_rgb.max(axis=2) >= 20
    if not foreground.any():
        return 0.0
    return float((classified[foreground] == 0).mean())


def _majority_vote_classified(classified_draws: list[np.ndarray]) -> np.ndarray:
    """Per-pixel majority region id across independently classified draws.

    Ties go to the LOWEST draw index (first pass wins, later passes need a
    strictly larger count to displace it), so with exactly TWO kept draws
    every disagreement is a 1-1 tie and the result is the first kept draw —
    ask for three or more (odd is best) to get a real vote. Measured on the
    hand-registered slices: voting K draws beats the mean single draw by
    0.04-0.08 family dice and lands near the best draw of the set.
    """
    stack = np.stack(classified_draws)
    best = np.zeros_like(stack[0])
    best_count = np.zeros(stack.shape[1:], dtype=int)
    # ponytail: K passes over a K-deep stack (O(K^2) per pixel). Fine for the
    # handful of draws this is used with; a per-pixel bincount would win only
    # well past K=16.
    for draw in stack:
        count = (stack == draw).sum(axis=0)
        take = count > best_count
        best = np.where(take, draw, best)
        best_count = np.where(take, count, best_count)
    return best


def _pinned_registration_palette():
    """Force the atlas render style registration is built on.

    The lineup fixes what the model is shown — ONE flat, undelineated region
    map — so the style must not float with the environment: a delineated or
    family-collapsed render would change the map without changing what the
    classifier expects of it. The styles in ``atlas.recolor`` stay for
    renders people look at.
    """
    return use_palette("family")


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
    quality: str | None = None,
) -> tuple[Image.Image, tuple[int, int], float, float, int]:
    """Downsample, pad, and aspect-snap the slice into the working canvas.

    Returns ``(slice_image, unpadded_size, origin_x, origin_y, pad_px)``.
    Padding uses the slice's own background color, and the aspect-ratio
    clamp pads the short axis (pad only, never crop): in edit mode the
    model paints on ITS canvas, and resampling a mismatched ratio back
    onto the slice would silently undo the pixel alignment.

    ``native_canvas`` (the default) sizes the canvas to the frame the image
    path returns (:func:`model_prompts.native_output_size`): the layout is
    worked out at the long-edge rule, scaled to fit that frame, and padded
    out to it exactly, so the model edits on the output's own pixel grid.
    Nash 2026-09-11: every input the model must rescale to its output is a
    pixel the output cannot carry, paid for twice (input tokens in, a
    resample out). The slice is resampled ONCE, straight from the original
    to its final size. ``canvas_long_edge`` instead pins the long edge (an
    experiment knob: the model is shown a smaller canvas and its output is
    resampled DOWN onto it).
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
        frame = native_output_size(image_model, provider, canvas0, quality)
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


def _onto_canvas(draw: Image.Image, size: tuple[int, int], slack_px: int = 2) -> Image.Image:
    """A model draw on the canvas grid.

    A native canvas comes back at its own size, give or take a pixel of the
    lane's rounding: that is pasted (centered, cropped/padded), never
    resampled. Anything further off is resampled onto the canvas.
    """
    draw = draw.convert("RGB")
    if draw.size == size:
        return draw
    dw, dh = abs(draw.width - size[0]), abs(draw.height - size[1])
    if dw <= slack_px and dh <= slack_px:
        out = Image.new("RGB", size, (0, 0, 0))
        out.paste(draw, ((size[0] - draw.width) // 2, (size[1] - draw.height) // 2))
        return out
    return draw.resize(size, resample=Image.Resampling.LANCZOS)


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
        from langslice.space import atlas_space_context, orient_slice_to_axes

        arr = orient_slice_to_axes(np.asarray(image), atlas_space_context(atlas), plane, image_axes)
        image = Image.fromarray(arr)
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if atlas_mirror_lr else image


def _canvas_from_native_matrix(
    native_size: tuple[int, int], canvas_size: tuple[int, int]
) -> np.ndarray:
    """Exact pixel-center transform used by _fit_to_canvas, including rounded sizes."""
    scale = min(canvas_size[0] / native_size[0], canvas_size[1] / native_size[1])
    new = tuple(max(1, round(length * scale)) for length in native_size)
    sx, sy = new[0] / native_size[0], new[1] / native_size[1]
    ox, oy = (canvas_size[0] - new[0]) // 2, (canvas_size[1] - new[1]) // 2
    return np.array([[sx, 0, ox + (sx - 1) / 2],
                     [0, sy, oy + (sy - 1) / 2], [0, 0, 1]], dtype=np.float64)


def _native_coordinate_map(
    field: np.ndarray, native_size: tuple[int, int], prealign_matrix: np.ndarray | None
) -> np.ndarray:
    """Compose output→prealigned canvas→canonical canvas→native atlas pullback."""
    h, w = field.shape[:2]
    placement = np.eye(3)
    if prealign_matrix is not None:
        placement[:2] = prealign_matrix
    inverse = np.linalg.inv(placement @ _canvas_from_native_matrix(native_size, (w, h)))
    yy, xx = np.indices((h, w), dtype=np.float64)
    displaced = np.stack((xx, yy), axis=-1) + field
    return displaced @ inverse[:2, :2].T + inverse[:2, 2]


def _gather_native_labels(labels: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """Nearest integer gather; atlas IDs never pass through floating-point images."""
    finite = np.asarray(np.isfinite(coordinates).all(axis=-1), dtype=bool)
    safe = np.where(finite[..., None], coordinates, 0)
    indices = np.floor(safe + 0.5).astype(np.int64)
    x, y = indices[..., 0], indices[..., 1]
    valid = finite & (x >= 0) & (y >= 0) & (x < labels.shape[1]) & (y < labels.shape[0])
    result = np.zeros(coordinates.shape[:2], dtype=labels.dtype)
    result[valid] = labels[y[valid], x[valid]]
    return result


def _fit_to_canvas(
    image: Image.Image,
    size: tuple[int, int],
    fill: tuple[int, int, int] = (0, 0, 0),
    resample: Image.Resampling = Image.Resampling.NEAREST,
) -> Image.Image:
    """Uniform-scale *image* onto a *size* canvas, center-padded with *fill*.

    Never stretches: an atlas render squeezed into the histology's aspect
    ratio deforms the very anatomy the registration measures against. This
    is the same geometry the model was shown (its map letterboxed to the
    section's aspect), expressed at the section's pixel size. The default
    NEAREST is for label maps, where any blending invents colors that
    classify as third regions.
    """
    scale = min(size[0] / image.width, size[1] / image.height)
    new = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.convert("RGB").resize(new, resample=resample)
    canvas = Image.new("RGB", size, fill)
    canvas.paste(resized, ((size[0] - new[0]) // 2, (size[1] - new[1]) // 2))
    return canvas


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
    it is restretched on the 99.5th percentile of the plate's own tissue
    (a linear rescale, exactly what the April benchmark sent).
    """
    from langslice.atlas import get_reference_slice

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


def _overlay_borders(base_image: Image.Image, borders: np.ndarray) -> Image.Image:
    """Draw atlas-region borders over a base image.

    Yellow core on a 1px black rim: readable on violet Nissl, white
    brightfield, and dark fluorescence alike (plain cyan vanished on
    cyan-tinted Nissl).
    """
    overlay = base_image.convert("RGB").copy()
    overlay_rgb = np.asarray(overlay, dtype=np.uint8).copy()
    border_mask = np.asarray(borders) > 0
    rim = cv2.dilate(border_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    overlay_rgb[rim] = (0, 0, 0)
    overlay_rgb[border_mask] = (255, 255, 0)
    return Image.fromarray(overlay_rgb, mode="RGB")


def _review_renders(
    slice_image: Image.Image,
    warped_classified: np.ndarray,
    warped_families: np.ndarray,
    atlas: Any,
    *,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[Image.Image | None, Image.Image | None]:
    """The two human-review renders: annotated overlay, and its checkerboard.

    Purely presentational, and purely additive — ``warped_border_overlay.png``
    and friends are untouched. Both are best-effort: a render failure must
    never cost a caller its registration, so this returns ``(None, None)``
    and says so on the progress channel instead of raising.
    """
    from langslice.nonlinear import render

    try:
        lut = color_lut(atlas)
        structures = getattr(atlas, "structures", None)
        names = {
            int(uid): _region_label(structures, int(uid))
            for uid in np.unique(warped_classified)
            if int(uid) != 0
        }
        overlay = render.region_overlay(
            slice_image, warped_classified, lut=lut, names=names, families=warped_families
        )
        plain = render.region_overlay(
            slice_image,
            warped_classified,
            lut=lut,
            families=warped_families,
            show_labels=False,  # a label sliced in half by a tile edge reads as breakage
        )
        return overlay, render.checkerboard(slice_image, plain, tiles=10, seam_opacity=0.15)
    except Exception as exc:  # pragma: no cover - debug artifacts only
        if on_progress:
            on_progress(f"Image-gen registration: review renders skipped ({exc})")
        return None, None


#: Clamp for the silhouette prealign scales: wide enough for real
#: inter-animal size spread, tight enough that a fragment or hemibrain
#: painting cannot shrink the whole atlas onto itself.
_PREALIGN_SCALE_BOUNDS = (0.85, 1.2)


def _prealign_atlas_to_paint(
    atlas_classified: np.ndarray, generated_classified: np.ndarray
) -> tuple[np.ndarray, np.ndarray | None]:
    """Axis-aligned moments placement of the atlas map onto the painted map.

    A real brain differs from the atlas average in size, and when the placed
    atlas starts inside the painting, the uncovered rim gives mean-squares
    ZERO gradient (samples land on flat moving background), so neither
    Elastix stage can grow it — measured on the LSD_910 hand-registered
    benchmark: 4.9% of truth tissue left uncovered, all of it an outward
    rim. Matching the two silhouettes' centroids and axis spreads closes the
    rim by construction (GT dice 0.928 -> 0.950, uncovered 4.9% -> 0.1%).
    Scale and translation only: rotations stay with Elastix, and reflections
    are forbidden — a mirrored fit scores the same silhouette IoU on a
    near-symmetric section and lands anatomy on the wrong hemispheres.
    Returns (aligned map, 2x3 matrix) — the matrix must also place every
    other map that will be warped through the resulting transform.
    """
    src = atlas_classified != 0
    dst = generated_classified != 0
    if not src.any() or not dst.any():
        return atlas_classified, None
    sy, sx = np.nonzero(src)
    dy, dx = np.nonzero(dst)
    lo, hi = _PREALIGN_SCALE_BOUNDS
    scx = float(np.clip(dx.std() / max(sx.std(), 1e-6), lo, hi))
    scy = float(np.clip(dy.std() / max(sy.std(), 1e-6), lo, hi))
    matrix = np.array(
        [
            [scx, 0.0, dx.mean() - scx * sx.mean()],
            [0.0, scy, dy.mean() - scy * sy.mean()],
        ]
    )
    return _warp_labels_affine(atlas_classified, matrix), matrix


def _warp_labels_affine(labels: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Affine-warp an integer label map without blending labels.

    cv2 cannot warp integer label dtypes; float64 holds every Allen id
    exactly and NEAREST keeps labels unblended.
    """
    h, w = labels.shape
    warped = cv2.warpAffine(
        labels.astype(np.float64),
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    return warped.astype(labels.dtype)


def _leaf_overlay_render(
    slice_image: Image.Image,
    atlas: Any,
    position_mm: float,
    plane: Plane,
    image_axes: str | None,
    result_transform: Any,
    *,
    prealign_matrix: np.ndarray | None = None,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    on_progress: Callable[[str], None] | None = None,
    atlas_mirror_lr: bool = False,
    warped_labels: np.ndarray | None = None,
) -> tuple[Image.Image | None, np.ndarray | None]:
    """Leaf-level review render: the RAW annotation warped through the fit.

    Returns ``(overlay, warped_leaf_ids)``; the id map is what landmark-level
    evaluation against hand registrations reads, so it is saved alongside.

    Color classification collapses every set of same-colored regions (all
    fiber tracts, quantized families) into one label, hiding their internal
    boundaries. Warping the annotation ids directly shows every parcellation
    on the slice, so fine-structure damage from the warp is visible. Best
    effort, like the other review renders.
    """
    from langslice.nonlinear import render
    from langslice.nonlinear.image_gen_helpers import _annotation_slice
    from langslice.space import atlas_space_context

    try:
        if warped_labels is not None:
            structures = getattr(atlas, "structures", None)
            names = {
                int(uid): _region_label(structures, int(uid))
                for uid in np.unique(warped_labels) if int(uid) != 0
            }
            return render.region_overlay(
                slice_image, warped_labels, lut=color_lut(atlas), names=names,
                families=_merge_classified(warped_labels, atlas), fill_alpha=0.15,
            ), warped_labels
        context = atlas_space_context(atlas)
        leaf = _annotation_slice(
            atlas, position_mm, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg,
            blackout=False,  # a reviewer checks the ventricles; keep them in this render
        )
        if image_axes:
            from langslice.space import orient_slice_to_axes

            leaf = orient_slice_to_axes(leaf, context, plane, image_axes)
        if atlas_mirror_lr:
            leaf = np.fliplr(leaf)
        # Same frame as the Elastix moving render: uniform scale (physical
        # when known), centered — a stretched leaf map against a letterboxed
        # transform inflates every annotation off the tissue.
        leaf_img = Image.fromarray(leaf.astype(np.int32), mode="I")
        scale = min(
            slice_image.size[0] / leaf_img.width,
            slice_image.size[1] / leaf_img.height,
        )
        new_size = (
            max(1, round(leaf_img.width * scale)),
            max(1, round(leaf_img.height * scale)),
        )
        scaled = np.asarray(
            leaf_img.resize(new_size, Image.Resampling.NEAREST), dtype=np.int64
        )
        leaf_resized = np.zeros((slice_image.size[1], slice_image.size[0]), dtype=np.int64)
        ox = (slice_image.size[0] - new_size[0]) // 2
        oy = (slice_image.size[1] - new_size[1]) // 2
        src_x0, src_y0 = max(0, -ox), max(0, -oy)
        dst_x0, dst_y0 = max(0, ox), max(0, oy)
        w = min(new_size[0] - src_x0, slice_image.size[0] - dst_x0)
        h = min(new_size[1] - src_y0, slice_image.size[1] - dst_y0)
        leaf_resized[dst_y0 : dst_y0 + h, dst_x0 : dst_x0 + w] = scaled[
            src_y0 : src_y0 + h, src_x0 : src_x0 + w
        ]
        if prealign_matrix is not None:
            # The transform was fit against the PREALIGNED moving frame;
            # everything warped through it must share that placement.
            leaf_resized = _warp_labels_affine(leaf_resized, prealign_matrix)
        warped_leaf = _warp_classified_labels(leaf_resized, result_transform)
        structures = getattr(atlas, "structures", None)
        names = {
            int(uid): _region_label(structures, int(uid))
            for uid in np.unique(warped_leaf)
            if int(uid) != 0
        }
        overlay = render.region_overlay(
            slice_image,
            warped_leaf,
            lut=color_lut(atlas),
            names=names,
            families=_merge_classified(warped_leaf, atlas),
            fill_alpha=0.15,
        )
        return overlay, warped_leaf
    except Exception as exc:  # pragma: no cover - debug artifacts only
        if on_progress:
            on_progress(f"Image-gen registration: leaf overlay skipped ({exc})")
        return None, None


def _save_debug_artifacts(
    artifact_dir: Path,
    *,
    generated_segmentation: Image.Image,
    generated_draws: list[Image.Image] | None = None,
    warped_atlas: Image.Image,
    warped_border_overlay: Image.Image,
    input_colored_regions: Image.Image,
    input_reference_images: list[Image.Image],
    input_slice: Image.Image,
    input_prior: Image.Image | None = None,
    generated_border_overlay: Image.Image | None = None,
    slice_warped_to_atlas: Image.Image | None = None,
    slice_atlas_border_overlay: Image.Image | None = None,
    region_overlay: Image.Image | None = None,
    leaf_overlay: Image.Image | None = None,
    checkerboard: Image.Image | None = None,
) -> dict[str, str]:
    """Persist registration artifacts to disk and return absolute paths."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    def _save(name: str, image: Image.Image) -> None:
        out_path = artifact_dir / name
        # Preserve alpha for RGBA inputs so the atlas-space slice texture keeps
        # its root-mask silhouette. Everything else gets flattened to RGB for
        # consistent on-disk format across forward-pipeline artifacts.
        if image.mode == "RGBA":
            image.save(out_path)
        else:
            image.convert("RGB").save(out_path)
        paths[name] = str(out_path.resolve())

    _save("generated_segmentation.png", generated_segmentation)
    # Raw provider draws behind a vote (only written when there is more than
    # one — a single draw IS generated_segmentation.png). Index matches the
    # kept/dropped indices in the candidate metadata.
    for i, draw in enumerate(generated_draws or []):
        _save(f"generated_segmentation_draw{i}.png", draw)
    _save("warped_atlas.png", warped_atlas)
    _save("warped_border_overlay.png", warped_border_overlay)
    # Image 1 exactly as the model received it: the colored atlas map it was
    # asked to edit, keeping its historical name so downstream tooling works.
    _save("input_colored_regions.png", input_colored_regions)
    # Images 2..N in prompt order. The first is the grayscale template; the
    # second is the section, which also keeps its historical name.
    for i, ref in enumerate(input_reference_images):
        _save(f"input_reference_{i + 2}.png", ref)
    _save("input_slice.png", input_slice)
    # The silhouette prior, when provider="none" made it the painting.
    if input_prior is not None:
        _save("input_prior.png", input_prior)
    if generated_border_overlay is not None:
        _save("generated_border_overlay.png", generated_border_overlay)
    if slice_warped_to_atlas is not None:
        _save("slice_warped_to_atlas.png", slice_warped_to_atlas)
    if slice_atlas_border_overlay is not None:
        _save("slice_atlas_border_overlay.png", slice_atlas_border_overlay)
    if region_overlay is not None:
        _save("region_overlay.png", region_overlay)
    if leaf_overlay is not None:
        _save("leaf_overlay.png", leaf_overlay)
    if checkerboard is not None:
        _save("checkerboard.png", checkerboard)
    return paths


def _emit_trace(
    on_trace: Callable[[dict[str, object]], None] | None,
    *,
    candidate_id: str,
    metadata: dict[str, Any],
    generated_segmentation: Image.Image,
    warped_atlas: Image.Image,
    warped_border_overlay: Image.Image,
) -> None:
    if on_trace is None:
        return

    on_trace(
        runtime_event(
            stage="registration",
            title="Image-gen registration candidate generated",
            summary=f"Candidate {candidate_id} generated with {metadata['n_markers']} markers",
            parts=[
                image_part_from_pil(generated_segmentation, label="Generated segmentation"),
                image_part_from_pil(warped_atlas, label="Warped atlas"),
                image_part_from_pil(warped_border_overlay, label="Warped border overlay"),
                json_part(metadata, label="Candidate metadata"),
            ],
            metadata=metadata,
        )
    )


def generate_registration_candidate(
    image: Image.Image,
    *,
    atlas_name: str,
    position_mm: float,
    plane: Plane = "coronal",
    provider: str = "google",
    image_model: str | None = None,
    image_prompt: str | None = None,
    generated_image: Image.Image | None = None,
    image_axes: str | None = None,
    canvas_pad: float = 0.0,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    draws: int = 1,
    max_off_palette: float = _MAX_OFF_PALETTE,
    deformation: Deformation = "bspline",
    previous_candidate_id: str | None = None,
    candidate_id: str | None = None,
    debug_dir: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
    openai_image_route: str = "images",
    review_model: str | None = None,
    thinking_level: str | None = None,
    native_canvas: bool = True,
    canvas_long_edge: int | None = None,
    elastix: ElastixStage = "rgb",
    registration_mode: Literal["borders", "colormap"] = "borders",
    initial_atlas_to_slice: Sequence[Sequence[float]] | np.ndarray | None = None,
    initial_alignment_source: str = "supplied",
    atlas_mirror_lr: bool = False,
) -> RegistrationCandidate:
    """Generate one dense registration candidate from a histology slice.

    The default ``registration_mode="borders"`` corrects yellow atlas lines
    on the histology, using the unmarked histology as the second image.
    A supplied ``initial_atlas_to_slice`` maps native atlas pixel centers
    (after ``image_axes`` orientation and optional ``atlas_mirror_lr``) to
    original section pixels. Without that placement, a color-map generation
    and atlas fit establish it before the border correction. Raw model lines
    and the fitted atlas remain separate review artifacts.

    ``registration_mode="colormap"`` runs only the legacy initial stage:
    the model edits a colored atlas map with grayscale atlas and histology
    references, then Elastix fits the atlas labels to the generated map.
    Multiple ``draws`` and the off-palette vote gate apply only to this mode.

    ``provider="none"`` calls no model. The border route retains supplied
    placement, or builds a silhouette placement when none is supplied.
    ``pitch_deg`` and ``yaw_deg`` reslice every atlas input on the requested
    cutting plane. Both modes return integer warped labels and an output
    canvas-to-native-atlas coordinate map alongside registration artifacts.
    """
    with _pinned_registration_palette():
        kwargs: dict[str, Any] = dict(
            atlas_name=atlas_name,
            position_mm=position_mm,
            plane=plane,
            provider=provider,
            image_model=image_model,
            image_prompt=image_prompt,
            generated_image=generated_image,
            image_axes=image_axes,
            canvas_pad=canvas_pad,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            draws=draws,
            max_off_palette=max_off_palette,
            deformation=deformation,
            previous_candidate_id=previous_candidate_id,
            candidate_id=candidate_id,
            debug_dir=debug_dir,
            on_progress=on_progress,
            on_trace=on_trace,
            openai_image_route=openai_image_route,
            review_model=review_model,
            thinking_level=thinking_level,
            native_canvas=native_canvas,
            canvas_long_edge=canvas_long_edge,
            elastix=elastix,
            atlas_mirror_lr=atlas_mirror_lr,
        )
        if registration_mode == "borders":
            if draws != 1:
                raise ValueError("Border refinement supports exactly one draw; set draws=1")
            from langslice.nonlinear.border_registration import (
                generate_border_registration_candidate,
            )

            for key in ("draws", "max_off_palette", "elastix"):
                kwargs.pop(key)
            return generate_border_registration_candidate(
                image, **kwargs,
                initial_atlas_to_slice=initial_atlas_to_slice,
                initial_alignment_source=initial_alignment_source,
            )
        if registration_mode != "colormap":
            raise ValueError(f"Unknown registration mode: {registration_mode}")
        if initial_atlas_to_slice is not None:
            raise ValueError("A supplied alignment requires registration_mode='borders'")
        return _generate_registration_candidate(image, **kwargs)


def _generate_registration_candidate(
    image: Image.Image,
    *,
    atlas_name: str,
    position_mm: float,
    plane: Plane = "coronal",
    provider: str = "google",
    image_model: str | None = None,
    image_prompt: str | None = None,
    generated_image: Image.Image | None = None,
    image_axes: str | None = None,
    canvas_pad: float = 0.0,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    draws: int = 1,
    max_off_palette: float = _MAX_OFF_PALETTE,
    deformation: Deformation = "bspline",
    extra_reference_images: list[Image.Image] | None = None,
    reference_images_override: list[Image.Image] | None = None,
    previous_candidate_id: str | None = None,
    candidate_id: str | None = None,
    debug_dir: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
    openai_image_route: str = "images",
    review_model: str | None = None,
    thinking_level: str | None = None,
    native_canvas: bool = True,
    canvas_long_edge: int | None = None,
    elastix: ElastixStage = "rgb",
    atlas_mirror_lr: bool = False,
) -> RegistrationCandidate:
    candidate_id = candidate_id or f"candidate-{uuid.uuid4().hex[:12]}"
    original_width, original_height = image.size

    if on_progress:
        on_progress("Image-gen registration: loading atlas and preparing inputs...")
    atlas = load_atlas(atlas_name)

    slice_image, unpadded_size, origin_x, origin_y, pad_px = prepare_canvas(
        image, canvas_pad=canvas_pad, image_model=image_model, provider=provider,
        native_canvas=native_canvas, canvas_long_edge=canvas_long_edge, quality=thinking_level,
    )
    target_size = slice_image.size

    # The atlas map of this plane, drawn pixel-exact at the atlas's own
    # geometry: this is BOTH the image the model edits and the moving image
    # Elastix registers, so the two can never disagree about where a
    # boundary is. Oriented into the user's image frame first.
    map_native = _orient_pil(
        _generate_colored_region_slice(
            atlas, position_mm, None, plane=plane, smooth=False,
            pitch_deg=pitch_deg, yaw_deg=yaw_deg,
        ),
        atlas, plane, image_axes, atlas_mirror_lr,
    )
    from langslice.nonlinear.image_gen_helpers import _annotation_slice

    native_labels = _annotation_slice(
        atlas, position_mm, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg,
        blackout=False,
    )
    if image_axes:
        from langslice.space import atlas_space_context, orient_slice_to_axes

        native_labels = orient_slice_to_axes(
            native_labels, atlas_space_context(atlas), plane, image_axes
        )
    if atlas_mirror_lr:
        native_labels = np.fliplr(native_labels)
    if native_labels.shape != (map_native.height, map_native.width):
        raise ValueError("Native atlas labels and region map must share a pixel frame")
    section_aspect = target_size[0] / target_size[1]
    # Model-facing Image 1: the map, NEAREST-upscaled to a legible size and
    # letterboxed to the section's aspect, so it and the section share one
    # frame — which is what the prompt pins the answer to. No physical
    # rescaling, no cropping, and the anatomy keeps the atlas's proportions.
    colored_regions = letterbox_to_aspect(
        upscale_to_min_long_edge(map_native, Image.Resampling.NEAREST), section_aspect
    )
    canvas_image = colored_regions

    # No model in the loop: the silhouette prior IS the painting. Everything
    # downstream (Elastix, markers, overlays, report, exports) runs on it
    # exactly as it runs on a generated one.
    model_free = canonical_provider(provider) == "none"
    prior_image: Image.Image | None = None
    prior_metadata: dict[str, Any] = {}
    if model_free and generated_image is None:
        if on_progress:
            on_progress("Image-gen registration: placing the silhouette prior...")
        prior_image, prior_metadata = build_silhouette_prior(
            (
                slice_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if atlas_mirror_lr else slice_image
            ),
            atlas=atlas,
            position_mm=position_mm,
            plane=plane,
            image_axes=image_axes,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
        )
        if atlas_mirror_lr:
            prior_image = prior_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if on_progress:
            on_progress(
                "Image-gen registration: prior placed, tissue IoU "
                f"{prior_metadata['tissue_iou']:.3f} signs {prior_metadata['sign_pattern']}"
            )

    # Images 2..N, built only when a model is actually going to see them:
    # the grayscale template of the same plane (same upscale and letterbox),
    # then the section. Experimental: a caller may replace the references
    # entirely (its own prompt names them) or append extra ones (e.g. a
    # worked example: another section and its correct painting).
    reference_images: list[Image.Image] = []
    if generated_image is None and not model_free:
        if reference_images_override is not None:
            reference_images = [im.convert("RGB") for im in reference_images_override]
        else:
            atlas_template = letterbox_to_aspect(
                upscale_to_min_long_edge(
                    _orient_pil(
                        _model_facing_template(
                            atlas, position_mm, plane, pitch_deg, yaw_deg
                        ),
                        atlas, plane, image_axes, atlas_mirror_lr,
                    ),
                    Image.Resampling.LANCZOS,
                ),
                section_aspect,
            )
            reference_images = [atlas_template, slice_image]
        reference_images += list(extra_reference_images or [])

    prompt = image_prompt or base_segmentation_prompt(plane, provider)
    request_metadata: dict[str, Any] = {
        "workflow": "image_gen_registration",
        "candidate_id": candidate_id,
        "atlas_name": atlas_name,
        "position_mm": float(position_mm),
        "plane": plane,
        "target_size": list(target_size),
        "original_size": [original_width, original_height],
    }
    if previous_candidate_id is not None:
        request_metadata["previous_candidate_id"] = previous_candidate_id
    if image_prompt is not None:
        request_metadata["image_prompt"] = image_prompt

    draws = max(1, int(draws))
    if generated_image is not None:
        # The painting was made elsewhere (an offline re-derivation, a
        # benchmark draw); only the downstream pipeline runs here.
        generated = GeneratedSegmentation(
            image=generated_image.convert("RGB"),
            provider=provider,
            model=image_model or "unknown",
            route="external_image",
            metadata=dict(request_metadata),
        )
        draw_images = [generated.image]
    elif model_free:
        assert prior_image is not None
        generated = GeneratedSegmentation(
            image=prior_image,
            provider="none",
            model="none",
            route="silhouette_prior",
            metadata=dict(request_metadata),
        )
        draw_images = [generated.image]
    else:
        if on_progress:
            on_progress("Image-gen registration: generating warped atlas image...")
        # One request, sampled *draws* times: independent paintings of the same
        # inputs, voted per pixel below.
        generation_request = SegmentationGenerationRequest(
            reference_images=reference_images,
            slice_image=canvas_image,
            prompt=prompt,
            provider=provider,
            model=image_model,
            review_model=review_model,
            openai_image_route=openai_image_route,
            thinking_level=thinking_level,
            metadata=request_metadata,
        )
        generated = generate_warped_segmentation_image(generation_request)
        draw_images = [generated.image]
        for draw_index in range(1, draws):
            if on_progress:
                on_progress(f"Image-gen registration: draw {draw_index + 1}/{draws}...")
            draw_images.append(
                generate_warped_segmentation_image(generation_request).image
            )

    # Elastix side: the same pixel-exact map the model edited, in the same
    # geometry (uniform scale, centered), expressed at the section's pixel
    # size — the old full-canvas anisotropic stretch fabricated a large
    # distortion the affine stage had to undo before doing real work, and
    # when it under-corrected the warped atlas landed outside the slice.
    atlas_colored_at_target = _fit_to_canvas(map_native, target_size)
    atlas_target_rgb = np.asarray(atlas_colored_at_target, dtype=np.uint8)

    # Classify each draw: registration runs on the CLEANED map (exact palette
    # colors on black), so color drift cannot poison the per-channel metric.
    # Every draw goes through exactly this path, and only then are they voted
    # on. Nothing is masked out by comparison with the input: the model edits
    # the ATLAS MAP, so pixels it left untouched are regions it had no reason
    # to move, not background.
    classified_draws: list[np.ndarray] = []
    off_palette_fractions: list[float] = []
    for draw in draw_images:
        # The frame the model answered in is the frame it was shown; a lane
        # with fixed output frames returns the nearest legal aspect instead,
        # ours letterboxed inside it, so crop back rather than stretch.
        cropped = crop_to_aspect(draw.convert("RGB"), section_aspect)
        draw_rgb = np.asarray(_onto_canvas(cropped, target_size), dtype=np.uint8)
        classified = _classify_pixels_to_region_ids(
            draw_rgb, atlas, position_mm, plane=plane,
            pitch_deg=pitch_deg, yaw_deg=yaw_deg,
            paint=True,
        )
        off_palette_fractions.append(_off_palette_fraction(draw_rgb, classified))
        classified_draws.append(_despeckle_classified(classified))

    # Translucency gate, then the vote. Dropping every draw would leave
    # nothing to register, so an all-translucent set votes on itself.
    kept = [i for i, frac in enumerate(off_palette_fractions) if frac <= max_off_palette]
    if not kept:
        kept = list(range(len(classified_draws)))
    dropped = [i for i in range(len(classified_draws)) if i not in kept]
    if len(kept) == 1:
        generated_classified = classified_draws[kept[0]]
        painting = draw_images[kept[0]]
    else:
        generated_classified = _majority_vote_classified(
            [classified_draws[i] for i in kept]
        )
        # The voted ids repainted in atlas colors on black: from here down
        # (Elastix target, ledger, overlays, artifacts) there is one painting.
        painting = Image.fromarray(
            _classified_to_rgb(generated_classified, atlas), mode="RGB"
        )
    if on_progress and len(classified_draws) > 1:
        on_progress(
            f"Image-gen registration: voted {len(kept)}/{len(classified_draws)} draws"
            + (f", dropped {dropped} as translucent" if dropped else "")
        )

    atlas_pre_classified = _classify_pixels_to_region_ids(
        atlas_target_rgb,
        atlas,
        position_mm,
        plane=plane,
        off_palette_background=False,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
    )
    atlas_pre_classified, prealign_matrix = _prealign_atlas_to_paint(
        atlas_pre_classified, generated_classified
    )
    # Merged-granularity RGB kept for the review renders only; the
    # registration itself runs on one-hot family channels.
    cleaned_output_rgb = _registration_rgb(generated_classified, atlas)

    if on_progress:
        on_progress("Image-gen registration: registering region maps...")
    # The metric samples only the model's segmented tissue: omitted (absent)
    # structures never constrain the fit, per the segmentation-only contract.
    if elastix == "april-borders":
        result_transform, elastix_elapsed = _run_elastix_april_borders(
            _extract_borders_from_classified(generated_classified),
            _extract_borders_from_classified(atlas_pre_classified),
        )
    else:
        result_transform, elastix_elapsed = _register_region_maps(
            atlas_pre_classified,
            generated_classified,
            atlas,
            fixed_mask=generated_classified != 0,
            deformation=deformation,
        )

    if on_progress:
        on_progress("Image-gen registration: warping atlas and extracting borders...")
    # Warp the LABEL map (nearest-neighbor), never the RGB: linear color
    # blending at boundaries re-classifies to arbitrary third regions.
    warped_classified = _warp_classified_labels(atlas_pre_classified, result_transform)
    warped_atlas_rgb = _classified_to_rgb(warped_classified, atlas)
    warped_atlas_img = Image.fromarray(warped_atlas_rgb, mode="RGB")
    # Overlay borders at the merged family granularity registration runs at;
    # the report and ledger keep the full palette.
    warped_families = _merge_classified(warped_classified, atlas)
    warped_borders = _extract_borders_from_classified(warped_families)
    warped_border_overlay = _overlay_borders(slice_image, warped_borders)

    atlas_classified = atlas_pre_classified
    tissue_mask = foreground_mask(slice_image)
    if tissue_mask is not None and tissue_mask.shape != warped_classified.shape:
        tissue_mask = (
            np.asarray(
                Image.fromarray(tissue_mask.astype(np.uint8) * 255).resize(
                    target_size, resample=Image.Resampling.NEAREST
                )
            )
            > 0
        )
    deformation_field = _compute_deformation_field(
        result_transform, cv2.cvtColor(atlas_target_rgb, cv2.COLOR_RGB2GRAY)
    )
    if (
        deformation_field is None
        or deformation_field.shape != (target_size[1], target_size[0], 2)
        or not np.isfinite(deformation_field).all()
    ):
        raise RuntimeError("Registration did not produce a finite atlas deformation field")
    atlas_coordinate_map = _native_coordinate_map(
        deformation_field, map_native.size, prealign_matrix
    )
    warped_labels = _gather_native_labels(native_labels, atlas_coordinate_map)
    elastix_report = _elastix_report(
        atlas_classified=atlas_classified,
        warped_classified=warped_classified,
        structures=getattr(atlas, "structures", None),
        deformation_field=deformation_field,
        tissue_mask=tissue_mask,
    )
    region_ledger = _region_ledger(
        atlas_classified=atlas_classified,
        warped_classified=warped_classified,
        structures=getattr(atlas, "structures", None),
        mm2_per_px=_mm2_per_pixel(atlas, target_size, plane=plane),
    )
    # Human/benchmark diagnostics on the GENERATED map (pre-Elastix). Never
    # sent to any model; flags name only topology-level no-nos.
    gen_report = generation_report(
        generated_classified,
        atlas_classified,
        atlas,
        tissue_mask=tissue_mask,
        structures=getattr(atlas, "structures", None),
    )
    if on_progress:
        flags = [f.get("code") for f in gen_report.get("flags", [])]
        on_progress(
            "Image-gen registration: generation flags: "
            + (", ".join(str(f) for f in flags) if flags else "none")
        )

    # Generated borders: the model's region prediction (no Elastix warp)
    # overlaid on the slice, for inspecting the output independent of Elastix.
    generated_borders = _extract_borders_from_classified(
        _merge_classified(generated_classified, atlas)
    )
    generated_border_overlay = _overlay_borders(slice_image, generated_borders)

    # Inverse warp: deform the histology slice into atlas space (slice -> atlas),
    # then overlay atlas-space region borders so callers have a complementary
    # view to the forward warped-atlas-on-slice overlay.
    if on_progress:
        on_progress("Image-gen registration: computing inverse warp (slice -> atlas)...")
    slice_rgb_array = np.asarray(slice_image.convert("RGB"), dtype=np.uint8)
    forward_fixed_gray = cv2.cvtColor(cleaned_output_rgb, cv2.COLOR_RGB2GRAY)
    try:
        warped_slice_to_atlas_rgb, _inverse_transform = _run_inverse_warp_for_slice(
            slice_rgb_array,
            forward_fixed_gray=forward_fixed_gray,
            forward_result_transform=result_transform,
            deformation=deformation,
        )
        # Mask the warped slice to the atlas root silhouette so the 3D viewer
        # can render it as a brain-shaped sheet at the AP position instead of
        # a rectangular slab. NEAREST resize keeps alpha binary (0 or 255).
        # The mask is derived from the oriented Elastix-side render so its
        # frame matches everything else in the pipeline.
        root_mask = (
            np.asarray(atlas_colored_at_target).sum(axis=2) > 0
        ).astype(np.uint8) * 255
        warped_slice_to_atlas_img: Image.Image | None = Image.fromarray(
            np.dstack([warped_slice_to_atlas_rgb, root_mask]), mode="RGBA"
        )
        atlas_borders = _extract_borders_from_classified(
            _merge_classified(atlas_classified, atlas)
        )
        # Border overlay stays RGB — it's consumed by the 2D Split/Overlay
        # views, which composite against a solid panel background.
        slice_atlas_border_overlay: Image.Image | None = _overlay_borders(
            Image.fromarray(warped_slice_to_atlas_rgb, mode="RGB"), atlas_borders
        )
        inverse_warp_status: str = "ok"
    except Exception as exc:
        warped_slice_to_atlas_img = None
        slice_atlas_border_overlay = None
        inverse_warp_status = f"failed: {type(exc).__name__}: {exc}"
        if on_progress:
            on_progress(f"Image-gen registration: inverse warp skipped ({inverse_warp_status})")

    scale_to_slice = float(original_width) / float(unpadded_size[0])
    markers = _extract_visualign_markers(
        deformation_field,
        scale_to_slice=scale_to_slice,
        origin_px=(origin_x, origin_y),
    )

    session_metadata: dict[str, Any] = {
        "registration_mode": "colormap",
        "native_atlas_size": list(map_native.size),
        "atlas_mirror_lr": atlas_mirror_lr,
        "visualign_markers": markers,
        "n_markers": len(markers),
        "elastix_elapsed_s": round(float(elastix_elapsed), 2),
        "target_size": list(target_size),
        "unpadded_size": list(unpadded_size),
        "canvas_native": bool(native_canvas and canvas_long_edge is None),
        "elastix_stage": elastix,
        "canvas_pad": float(canvas_pad),
        "pad_px": pad_px,
        "canvas_origin_px": [origin_x, origin_y],
        "scale_to_slice": scale_to_slice,
        "provider": generated.provider,
        "model": generated.model,
        "model_name": generated.model,
        "route": generated.route,
        "candidate_id": candidate_id,
        "atlas_name": atlas_name,
        "position_mm": float(position_mm),
        "plane": plane,
        "inverse_warp_status": inverse_warp_status,
        "elastix": elastix_report,
        "generation": gen_report,
        "prior": prior_metadata,
    }
    if previous_candidate_id is not None:
        session_metadata["previous_candidate_id"] = previous_candidate_id
    if image_prompt is not None:
        session_metadata["image_prompt"] = image_prompt

    session = RegistrationAnnotationSession(
        workflow="image_gen_registration",
        target_count=0,
        metadata=session_metadata,
    )

    candidate_metadata: dict[str, Any] = {
        "registration_mode": "colormap",
        "native_atlas_size": list(map_native.size),
        "atlas_mirror_lr": atlas_mirror_lr,
        "workflow": "image_gen_registration",
        "candidate_id": candidate_id,
        "atlas_name": atlas_name,
        "position_mm": float(position_mm),
        "plane": plane,
        "target_size": list(target_size),
        "unpadded_size": list(unpadded_size),
        "canvas_native": bool(native_canvas and canvas_long_edge is None),
        "elastix_stage": elastix,
        "canvas_pad": float(canvas_pad),
        "pad_px": pad_px,
        "canvas_origin_px": [origin_x, origin_y],
        "original_size": [original_width, original_height],
        "inverse_warp_status": inverse_warp_status,
        "elastix": elastix_report,
        "generation": gen_report,
        "deformation": deformation,
        # Empty unless provider="none" put the silhouette placement in the
        # model's place; then it records how well that placement's outline
        # matched the tissue.
        "prior": prior_metadata,
        "provider": generated.provider,
        # Which draws the registered painting came from. Indices match the
        # generated_segmentation_draw{i}.png artifacts.
        "draws": {
            "n": len(classified_draws),
            "kept": kept,
            "dropped": dropped,
            "off_palette_fraction": [round(f, 4) for f in off_palette_fractions],
            "max_off_palette": float(max_off_palette),
        },
        "generated": {
            "provider": generated.provider,
            "model": generated.model,
            "route": generated.route,
            "revised_prompt": generated.revised_prompt,
            "metadata": dict(generated.metadata),
        },
    }
    if previous_candidate_id is not None:
        candidate_metadata["previous_candidate_id"] = previous_candidate_id
    if image_prompt is not None:
        candidate_metadata["image_prompt"] = image_prompt

    saved_artifact_paths: dict[str, str] = {}
    if debug_dir is not None:
        review_overlay, review_checkerboard = _review_renders(
            slice_image, warped_classified, warped_families, atlas, on_progress=on_progress
        )
        leaf_overlay, warped_leaf_ids = _leaf_overlay_render(
            slice_image, atlas, position_mm, plane, image_axes, result_transform,
            prealign_matrix=prealign_matrix,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            on_progress=on_progress,
            atlas_mirror_lr=atlas_mirror_lr,
            warped_labels=warped_labels,
        )
        candidate_dir = Path(debug_dir) / "registration" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        if warped_leaf_ids is not None:
            # Leaf ids of the warped atlas, in canvas pixels: the artifact
            # landmark scoring reads (which structures landed where).
            np.savez_compressed(
                candidate_dir / "warped_leaf_ids.npz", ids=warped_leaf_ids
            )
            saved_artifact_paths["warped_leaf_ids.npz"] = str(
                (candidate_dir / "warped_leaf_ids.npz").resolve()
            )
        ledger_path = candidate_dir / "region_ledger.json"
        ledger_path.write_text(json.dumps(region_ledger, indent=1))
        saved_artifact_paths["region_ledger.json"] = str(ledger_path.resolve())
        report_path = candidate_dir / "elastix_report.json"
        report_path.write_text(json.dumps(elastix_report, indent=1))
        saved_artifact_paths["elastix_report.json"] = str(report_path.resolve())
        gen_report_path = candidate_dir / "generation_report.json"
        gen_report_path.write_text(json.dumps(gen_report, indent=1))
        saved_artifact_paths["generation_report.json"] = str(gen_report_path.resolve())
        saved_artifact_paths |= _save_debug_artifacts(
            Path(debug_dir) / "registration" / candidate_id,
            generated_segmentation=painting,
            generated_draws=draw_images if len(draw_images) > 1 else None,
            warped_atlas=warped_atlas_img,
            warped_border_overlay=warped_border_overlay,
            input_colored_regions=colored_regions,
            input_reference_images=reference_images,
            input_slice=slice_image,
            input_prior=prior_image,
            generated_border_overlay=generated_border_overlay,
            slice_warped_to_atlas=warped_slice_to_atlas_img,
            slice_atlas_border_overlay=slice_atlas_border_overlay,
            region_overlay=review_overlay,
            checkerboard=review_checkerboard,
            leaf_overlay=leaf_overlay,
        )

    # Surface artifact paths in metadata so the register CLI can return them in
    # its JSON payload (forward + inverse pair).
    artifact_paths: dict[str, str | None] = {
        "warped_atlas_path": saved_artifact_paths.get("warped_atlas.png"),
        "warped_border_overlay_path": saved_artifact_paths.get("warped_border_overlay.png"),
        "generated_segmentation_path": saved_artifact_paths.get(
            "generated_segmentation.png"
        ),
        "generated_border_overlay_path": saved_artifact_paths.get(
            "generated_border_overlay.png"
        ),
        "slice_warped_to_atlas_path": saved_artifact_paths.get(
            "slice_warped_to_atlas.png"
        ),
        "slice_atlas_border_overlay_path": saved_artifact_paths.get(
            "slice_atlas_border_overlay.png"
        ),
        "region_ledger_path": saved_artifact_paths.get("region_ledger.json"),
        "region_overlay_path": saved_artifact_paths.get("region_overlay.png"),
        "checkerboard_path": saved_artifact_paths.get("checkerboard.png"),
        "leaf_overlay_path": saved_artifact_paths.get("leaf_overlay.png"),
    }
    session_metadata["artifact_paths"] = dict(artifact_paths)
    for key, value in artifact_paths.items():
        if value is not None:
            session_metadata[key] = value
    candidate_metadata["artifact_paths"] = dict(artifact_paths)
    for key, value in artifact_paths.items():
        if value is not None:
            candidate_metadata[key] = value

    trace_metadata = {
        **session_metadata,
        "generated_provider": generated.provider,
        "generated_model": generated.model,
        "generated_route": generated.route,
    }
    _emit_trace(
        on_trace,
        candidate_id=candidate_id,
        metadata=trace_metadata,
        generated_segmentation=painting,
        warped_atlas=warped_atlas_img,
        warped_border_overlay=warped_border_overlay,
    )

    if on_progress:
        on_progress(
            f"Image-gen registration complete: {len(markers)} markers, "
            f"{float(elastix_elapsed):.1f}s Elastix"
        )

    return RegistrationCandidate(
        candidate_id=candidate_id,
        generated_segmentation=painting,
        warped_atlas=warped_atlas_img,
        warped_border_overlay=warped_border_overlay,
        markers=markers,
        annotation_session=session,
        metadata=candidate_metadata,
        warped_labels=warped_labels,
        atlas_coordinate_map=atlas_coordinate_map,
    )

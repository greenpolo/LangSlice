"""Harness candidate pipeline for dense image-gen registration."""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from langslice.agent_trace import image_part_from_pil, json_part, runtime_event
from langslice.atlas import get_position_range_mm, load_atlas
from langslice.atlas.recolor import Palette, color_lut, use_palette
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
    _run_inverse_warp_for_slice,
    _warp_classified_labels,
    generation_report,
)
from langslice.nonlinear.model_prompts import (
    aspect_ratio_limits,
    base_segmentation_prompt,
    prior_refinement_prompt,
)
from langslice.nonlinear.prior import build_silhouette_prior
from langslice.nonlinear.providers import (
    SegmentationGenerationRequest,
    generate_warped_segmentation_image,
)
from langslice.nonlinear.types import (
    Deformation,
    GeneratedSegmentation,
    Init,
    RegistrationAnnotationSession,
    RegistrationCandidate,
)
from langslice.providers.registry import canonical_provider
from langslice.space import Plane

_MAX_LONG_EDGE = 2048

#: AP offsets (mm) of the model-facing atlas references around the estimated
#: position, anterior → posterior. ±0.125mm is human placement error — the
#: three maps bracket where the section can actually be, and the model reads
#: the tissue against the whole band rather than one possibly-off plane.
REFERENCE_OFFSETS_MM = (-0.125, 0.0, +0.125)

#: Max summed RGB delta for a generated pixel to count as "untouched" input.
#: Comfortably above resampling noise, well below the distance to the
#: nearest palette colors (light grays sit ~138 from white).
_PRESERVED_PIXEL_TOL = 45

#: Off-palette foreground fraction above which a draw is called translucent
#: and dropped from the vote. Clean draws measure 0.01-0.03, translucent ones
#: 0.10-0.17, so the gate sits between the two populations.
_MAX_OFF_PALETTE = 0.08


#: Benchmark hook: force the preserved-background mask off. Production
#: decides per call — ``init="silhouette"`` sends a label map as the canvas,
#: where unchanged pixels are valid paint rather than preserved background —
#: so this flag only exists for offline arms that need it off on the
#: ordinary path.
PRESERVED_BACKGROUND_MASKING = True


def _preserved_background_mask(
    model_output_rgb: np.ndarray, canvas_image: Image.Image
) -> np.ndarray:
    """Pixels the edit left untouched, against the canvas that was sent.

    The generation is an in-place edit, so anything still (nearly) identical
    to the input is unchanged. On a SECTION canvas that means unpainted, and
    the caller zeroes it — without that, a preserved white background
    classifies as the atlas root color (white in the Allen LUT) and poisons
    the registration. On a PRIOR canvas the same pixels are paint the model
    chose to keep, so the caller only records the fraction.
    """
    slice_rgb = np.asarray(canvas_image.convert("RGB"), dtype=int)
    delta = np.abs(model_output_rgb.astype(int) - slice_rgb).sum(axis=2)
    return delta <= _PRESERVED_PIXEL_TOL


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
) -> tuple[Image.Image, tuple[int, int], float, float, int]:
    """Downsample, pad, and aspect-snap the slice into the working canvas.

    Returns ``(slice_image, unpadded_size, origin_x, origin_y, pad_px)``.
    Padding uses the slice's own background color, and the aspect-ratio
    clamp pads the short axis (pad only, never crop): in edit mode the
    model paints on ITS canvas, and resampling a mismatched ratio back
    onto the slice would silently undo the pixel alignment.
    """
    target_size = _target_size_for_slice(image)
    slice_image = _resize_if_needed(image, target_size)
    fill = _background_color(slice_image)

    # Canvas padding: a fragment or hemibrain image cannot hold its own
    # complete anatomy, so the working canvas grows by a margin on every
    # side and the generated map may legitimately extend past the original
    # image bounds. Marker coordinates are mapped back to the original frame
    # (and may fall outside it).
    unpadded_size = target_size
    pad_px = int(round(float(canvas_pad) * max(target_size))) if canvas_pad else 0
    if pad_px > 0:
        padded = Image.new(
            "RGB", (target_size[0] + 2 * pad_px, target_size[1] + 2 * pad_px), fill
        )
        padded.paste(slice_image, (pad_px, pad_px))
        slice_image = padded
        target_size = slice_image.size
    origin_x = float(pad_px)
    origin_y = float(pad_px)

    limits = aspect_ratio_limits(image_model, provider)
    if limits:
        width, height = target_size
        current = width / height
        new_w, new_h = width, height
        if current > limits[1]:
            new_h = math.ceil(width / limits[1])
        elif current < limits[0]:
            new_w = math.ceil(height * limits[0])
        if (new_w, new_h) != (width, height):
            snapped = Image.new("RGB", (new_w, new_h), fill)
            offset = ((new_w - width) // 2, (new_h - height) // 2)
            snapped.paste(slice_image, offset)
            slice_image = snapped
            origin_x += offset[0]
            origin_y += offset[1]

    return slice_image, unpadded_size, origin_x, origin_y, pad_px


def _target_size_for_slice(image: Image.Image) -> tuple[int, int]:
    width, height = image.size
    long_edge = max(width, height)
    if long_edge <= _MAX_LONG_EDGE:
        return width, height

    scale = _MAX_LONG_EDGE / float(long_edge)
    return int(width * scale), int(height * scale)


def _resize_if_needed(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    if image.size == size:
        return image.convert("RGB")
    return image.convert("RGB").resize(size, resample=Image.Resampling.LANCZOS)


def _orient_pil(
    image: Image.Image, atlas: Any, plane: Plane, image_axes: str | None
) -> Image.Image:
    """Rotate/flip an atlas render into the user's image frame (no-op if unset)."""
    if not image_axes:
        return image
    from langslice.space import atlas_space_context, orient_slice_to_axes

    arr = orient_slice_to_axes(np.asarray(image), atlas_space_context(atlas), plane, image_axes)
    return Image.fromarray(arr)


#: An image with no more distinct colors than this is a flat region map, not
#: a photograph: resample it NEAREST so every pixel stays an exact palette
#: color. Anything richer (the grayscale template) is a continuous-tone image
#: and reads as blocks unless it is resampled smoothly.
_FLAT_MAP_MAX_COLORS = 128


def _fit_to_canvas(
    image: Image.Image,
    size: tuple[int, int],
    fill: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Uniform-scale *image* to fit INSIDE a *size* canvas, centered on *fill*.

    Never stretches: an atlas render squeezed into the histology's aspect
    ratio deforms the very anatomy the prompt calls authoritative. And never
    anything but fit-to-canvas: this is the one place every atlas render
    (model-facing maps, the Elastix moving image, the leaf review render)
    is placed, and it deliberately takes no scale. True-physical placement
    from a pixel size was tried here (2026-08-29 to 2026-09-05) and drew the
    atlas 20-30%% larger than shrunken tissue; the image model then copied
    the oversized plate instead of repainting the tissue (painting dice 0.52
    vs 0.62 fit-to-canvas, and visibly worse). ``tests/test_reference_scale_
    boundary.py`` pins this.
    """
    scale = min(size[0] / image.width, size[1] / image.height)
    new = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    rgb = image.convert("RGB")
    resample = (
        Image.Resampling.NEAREST
        if rgb.getcolors(_FLAT_MAP_MAX_COLORS) is not None
        else Image.Resampling.LANCZOS
    )
    resized = rgb.resize(new, resample=resample)
    canvas = Image.new("RGB", size, fill)
    canvas.paste(resized, ((size[0] - new[0]) // 2, (size[1] - new[1]) // 2))
    return canvas


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
        context = atlas_space_context(atlas)
        leaf = _annotation_slice(
            atlas, position_mm, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg,
            blackout=False,  # a reviewer checks the ventricles; keep them in this render
        )
        if image_axes:
            from langslice.space import orient_slice_to_axes

            leaf = orient_slice_to_axes(leaf, context, plane, image_axes)
        # Same frame as the Elastix moving render: uniform fit-to-canvas,
        # centered — a stretched leaf map against a letterboxed transform
        # inflates every annotation off the tissue.
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
    input_reference_images: list[Image.Image],
    input_extra_images: list[Image.Image] | None = None,
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
    # The model-facing reference set, in prompt order (Image 2..N). The
    # CENTER one keeps the historical input_colored_regions.png name so
    # downstream tooling keeps working.
    center = len(input_reference_images) // 2
    for i, ref in enumerate(input_reference_images):
        name = (
            "input_colored_regions.png"
            if i == center
            else f"input_reference_{i + 2}.png"
        )
        _save(name, ref)
    for i, ref in enumerate(input_extra_images or []):
        _save(f"input_exemplar_{i + 1}.png", ref)
    _save("input_slice.png", input_slice)
    # The prior canvas (init="silhouette"): Image 1 as the model received it,
    # and — with provider="none" — the painting itself.
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
    pixel_size_um: float | None = None,
    canvas_pad: float = 0.0,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    draws: int = 1,
    max_off_palette: float = _MAX_OFF_PALETTE,
    deformation: Deformation = "bspline",
    init: Init = "atlas",
    previous_candidate_id: str | None = None,
    candidate_id: str | None = None,
    debug_dir: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
    openai_image_route: str = "images",
    review_model: str | None = None,
    thinking_level: str | None = None,
    palette: Palette | None = None,
) -> RegistrationCandidate:
    """Generate one dense registration candidate from a histology slice.

    ``pitch_deg``/``yaw_deg`` are the block's cutting angles (see
    ``langslice.oblique``): every atlas render this call makes is resliced on
    that plane instead of taken flat off the voxel grid. Zero — a flat
    plane — is the default because nothing upstream fits them yet, but they
    are the single largest lever measured on the LSD_910 hand registrations,
    whose block was cut at 4 degrees: fit-only family dice 0.93 -> 0.96 and
    boundary p95 34px -> 9px over 33 slices, dwarfing every fit-side knob.

    ``pixel_size_um`` is recorded in the candidate metadata and NEVER scales
    an atlas render: every render is fit to the canvas. True-physical
    placement was tried (2026-08-29 to 2026-09-05) and was a regression on
    the model side — LSD_910 tissue runs ~10-14%% smaller than the Allen
    average, the true-scale plate overflowed the tissue by 20-30%%, and the
    image model copied the oversized plate instead of repainting the tissue
    (painting dice 0.52 vs 0.62, visibly worse). The fit side never needed
    it either (0.912 fit-to-canvas vs 0.908 physical on the 33-slice panel).

    ``palette`` overrides the process-wide atlas render style for this call
    (``"family"``, ``"leaf-borders"`` or ``"family-flat"``, see
    ``atlas.recolor``); ``None``
    keeps whatever ``LANGSLICE_ATLAS_PALETTE`` says. It changes only the
    model-facing region map; colors, the Elastix pair and everything
    classified from it are the same either way.

    ``draws`` > 1 asks the provider for that many independent paintings of
    the same request and registers their per-pixel majority vote (see
    :func:`_majority_vote_classified`); draws whose off-palette foreground
    fraction exceeds ``max_off_palette`` are dropped as translucent first
    (:func:`_off_palette_fraction`), unless that would drop them all.
    ``deformation="affine"`` fits the affine stage alone, without the
    B-spline stage.

    ``init="silhouette"`` builds the silhouette prior first — the atlas
    plane placed on this section's own outline by a moments fit and painted
    like an atlas reference (:mod:`langslice.nonlinear.prior`) — and sends
    THAT as the canvas to edit, with the section itself as the only
    reference image and the boundary-correction prompt. With
    ``provider="none"`` no model is called at all and the prior IS the
    painting: the model-free backbone, which with ``deformation="affine"``
    runs the whole downstream chain (Elastix, markers, overlays, report) on
    the placement alone. Measured on the LSD_910 hand registrations: the
    placement scores 0.82 mean family dice, better than every image-model
    configuration, and as a canvas it lifts the painting floor from ~0.5 to
    ~0.8.
    """
    with use_palette(palette):
        return _generate_registration_candidate(
            image,
            atlas_name=atlas_name,
            position_mm=position_mm,
            plane=plane,
            provider=provider,
            image_model=image_model,
            image_prompt=image_prompt,
            generated_image=generated_image,
            image_axes=image_axes,
            pixel_size_um=pixel_size_um,
            canvas_pad=canvas_pad,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            draws=draws,
            max_off_palette=max_off_palette,
            deformation=deformation,
            init=init,
            previous_candidate_id=previous_candidate_id,
            candidate_id=candidate_id,
            debug_dir=debug_dir,
            on_progress=on_progress,
            on_trace=on_trace,
            openai_image_route=openai_image_route,
            review_model=review_model,
            thinking_level=thinking_level,
        )


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
    pixel_size_um: float | None = None,
    canvas_pad: float = 0.0,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    draws: int = 1,
    max_off_palette: float = _MAX_OFF_PALETTE,
    deformation: Deformation = "bspline",
    init: Init = "atlas",
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
) -> RegistrationCandidate:
    candidate_id = candidate_id or f"candidate-{uuid.uuid4().hex[:12]}"
    original_width, original_height = image.size

    if on_progress:
        on_progress("Image-gen registration: loading atlas and preparing inputs...")
    atlas = load_atlas(atlas_name)

    slice_image, unpadded_size, origin_x, origin_y, pad_px = prepare_canvas(
        image, canvas_pad=canvas_pad, image_model=image_model, provider=provider
    )
    target_size = slice_image.size

    # The silhouette prior: the atlas plane placed on this section's own
    # outline and painted like an atlas reference (see nonlinear.prior). It
    # is the canvas the model edits under init="silhouette", and the painting
    # itself under provider="none".
    model_free = canonical_provider(provider) == "none"
    prior_image: Image.Image | None = None
    prior_metadata: dict[str, Any] = {}
    if init == "silhouette":
        if on_progress:
            on_progress("Image-gen registration: placing the silhouette prior...")
        prior_image, prior_metadata = build_silhouette_prior(
            slice_image,
            atlas=atlas,
            position_mm=position_mm,
            plane=plane,
            image_axes=image_axes,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
        )
        if on_progress:
            on_progress(
                "Image-gen registration: prior placed, tissue IoU "
                f"{prior_metadata['tissue_iou']:.3f} signs {prior_metadata['sign_pattern']}"
            )
    elif model_free and generated_image is None:
        raise ValueError(
            'provider="none" has nothing to paint with: it registers the '
            'silhouette prior, so it needs init="silhouette".'
        )
    # What the model is handed as Image 1. Everything measured against "what
    # was sent" (the preserved mask) reads this; the overlays stay on the
    # section, which is what a human checks the fit against.
    canvas_image = prior_image if prior_image is not None else slice_image
    # Model-facing atlas references: THREE colored region maps bracketing the
    # estimated depth at human-placement error (REFERENCE_OFFSETS_MM, ±125um),
    # anterior → posterior. The model sees how the anatomy evolves through the
    # position's own uncertainty band and matches the tissue against it,
    # instead of trusting one possibly-off plane. Each is oriented into the
    # user's image frame, then uniform-scaled and letterboxed to the histology
    # canvas — NEAREST-crisp thin bands, never an anisotropic stretch.
    lo_mm, hi_mm = get_position_range_mm(atlas, plane=plane)
    reference_positions = [
        min(max(position_mm + off, lo_mm), hi_mm) for off in REFERENCE_OFFSETS_MM
    ]
    reference_natives = [
        _orient_pil(
            _generate_colored_region_slice(
                atlas, pos, None, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg
            ),
            atlas, plane, image_axes,
        )
        for pos in reference_positions
    ]
    # Fit-to-canvas, always. `pixel_size_um` is recorded in the metadata for
    # downstream consumers and never scales a render — see _fit_to_canvas.
    reference_images = [_fit_to_canvas(native, target_size) for native in reference_natives]
    # Experimental: images the PROMPT describes beyond the atlas maps (e.g. a
    # worked example: another section and its correct painting). Delivered
    # after the atlas maps, in the order given; the caller's prompt names them.
    atlas_reference_images = reference_images
    if init == "silhouette" and reference_images_override is None:
        # Image 1 is the prior; Image 2 is the section whose anatomy the
        # prior's boundaries have to be moved onto. The atlas maps are not
        # sent — the prior already carries them, in this section's frame.
        reference_images_override = [slice_image]
    if reference_images_override is not None:
        # Experimental: the caller supplies Images 2..N itself (its prompt
        # names them); the atlas maps still feed the Elastix side.
        def _center_on_canvas(im: Image.Image) -> Image.Image:
            if im.size == slice_image.size:
                return im
            out = Image.new("RGB", slice_image.size, (0, 0, 0))
            offset = ((slice_image.width - im.width) // 2, (slice_image.height - im.height) // 2)
            out.paste(im.convert("RGB"), offset)
            return out
        reference_images = [_center_on_canvas(im) for im in reference_images_override]
        extra_reference_images = list(reference_images)
    else:
        reference_images = reference_images + list(extra_reference_images or [])

    prompt = image_prompt or (
        prior_refinement_prompt(plane)
        if init == "silhouette"
        else base_segmentation_prompt(plane, image_model)
    )
    request_metadata: dict[str, Any] = {
        "workflow": "image_gen_registration",
        "candidate_id": candidate_id,
        "atlas_name": atlas_name,
        "position_mm": float(position_mm),
        "plane": plane,
        "target_size": list(target_size),
        "original_size": [original_width, original_height],
        "pixel_size_um": pixel_size_um,
    }
    if previous_candidate_id is not None:
        request_metadata["previous_candidate_id"] = previous_candidate_id
    if image_prompt is not None:
        request_metadata["image_prompt"] = image_prompt

    draws = max(1, int(draws))
    if generated_image is not None:
        # The image came from an external conversation (the router session);
        # only the downstream pipeline runs here.
        generated = GeneratedSegmentation(
            image=generated_image.convert("RGB"),
            provider=provider,
            model=image_model or "unknown",
            route="router_session",
            metadata=dict(request_metadata),
        )
        draw_images = [generated.image]
    elif model_free:
        # No model in the loop: the prior IS the painting. Everything
        # downstream (Elastix, markers, overlays, report, exports) runs on it
        # exactly as it runs on a generated one.
        generated = GeneratedSegmentation(
            image=canvas_image,
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

    # Elastix side: the pixel-exact NEAREST render, never the smoothed one —
    # this is the image classified back to region ids and warped. Placed in
    # the SAME frame as the model-facing references (oriented, uniform
    # fit-to-canvas scale) — the old
    # full-canvas anisotropic stretch fabricated a large distortion the
    # affine stage had to undo before doing real work, and when it
    # under-corrected the warped atlas landed outside the slice.
    elastix_native = _orient_pil(
        _generate_colored_region_slice(
            atlas, position_mm, None, plane=plane, smooth=False,
            pitch_deg=pitch_deg, yaw_deg=yaw_deg,
        ),
        atlas, plane, image_axes,
    )
    atlas_colored_at_target = _fit_to_canvas(elastix_native, target_size)
    atlas_target_rgb = np.asarray(atlas_colored_at_target, dtype=np.uint8)

    # Classify the raw model output first: registration runs on the CLEANED
    # map (exact palette colors on black), so the preserved background and any
    # color drift cannot poison the per-channel metric. Every draw goes
    # through exactly this path, and only then are they voted on.
    # Unchanged pixels are unpainted BACKGROUND only when the canvas was the
    # section. On a prior canvas they are paint the model chose to keep, and
    # zeroing them would erase every boundary it got right first time.
    mask_preserved = PRESERVED_BACKGROUND_MASKING and prior_image is None
    classified_draws: list[np.ndarray] = []
    off_palette_fractions: list[float] = []
    preserved_fractions: list[float] = []
    for draw in draw_images:
        draw_rgb = np.asarray(
            draw.convert("RGB").resize(target_size, resample=Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )
        classified = _classify_pixels_to_region_ids(
            draw_rgb, atlas, position_mm, plane=plane,
            pitch_deg=pitch_deg, yaw_deg=yaw_deg,
            paint=True,
        )
        off_palette_fractions.append(_off_palette_fraction(draw_rgb, classified))
        preserved_mask = _preserved_background_mask(draw_rgb, canvas_image)
        if mask_preserved:
            classified[preserved_mask] = 0
        classified_draws.append(_despeckle_classified(classified))
        preserved_fractions.append(float(preserved_mask.mean()))

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
    preserved_fraction = float(np.mean([preserved_fractions[i] for i in kept]))
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
        preserved_fraction=preserved_fraction,
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
        "visualign_markers": markers,
        "n_markers": len(markers),
        "elastix_elapsed_s": round(float(elastix_elapsed), 2),
        "target_size": list(target_size),
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
        "init": init,
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
        "workflow": "image_gen_registration",
        "candidate_id": candidate_id,
        "atlas_name": atlas_name,
        "position_mm": float(position_mm),
        "plane": plane,
        "target_size": list(target_size),
        "canvas_pad": float(canvas_pad),
        "pad_px": pad_px,
        "canvas_origin_px": [origin_x, origin_y],
        "original_size": [original_width, original_height],
        "inverse_warp_status": inverse_warp_status,
        "elastix": elastix_report,
        "generation": gen_report,
        "deformation": deformation,
        # What the painting started from: "atlas" (a blank section canvas) or
        # "silhouette" (the placed prior), and how well the placement's
        # silhouette matched the tissue.
        "init": init,
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
        )
        candidate_dir = Path(debug_dir) / "registration" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        if warped_leaf_ids is not None:
            # Leaf ids of the warped atlas, in canvas pixels: the artifact
            # landmark scoring reads (which structures landed where).
            np.savez_compressed(
                candidate_dir / "warped_leaf_ids.npz", ids=warped_leaf_ids.astype(np.int32)
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
            input_reference_images=atlas_reference_images,
            input_extra_images=list(extra_reference_images or []),
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
    )

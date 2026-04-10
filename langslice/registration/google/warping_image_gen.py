"""Colored-segmentation registration workflow.

This workflow uses a Gemini image-generation model to produce a colored
segmentation of histology tissue guided by atlas reference images, then
extracts a dense deformation field via itk-elastix B-spline registration.

Pipeline:
    1. Generate three atlas input images at the target AP position: colored
       region map, grayscale reference, and the histology slice itself.
    2. Send all three images with the v13 prompt to Gemini image-gen.  The
       model warps the colored atlas regions to match the histology anatomy.
    3. Run itk-elastix B-spline registration on grayscale versions of the
       atlas colored regions (moving) and model output (fixed) to recover
       the dense deformation field.
    4. Warp the atlas RGB through the recovered transform.
    5. Classify warped atlas pixels and extract borders for visualization.
    6. Extract VisuAlign-compatible markers from the B-spline control
       points.
"""

from __future__ import annotations

import importlib
import logging
import os
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d

import langslice.registration.common as _agents
from langslice.agent_trace import image_part_from_pil, json_part, runtime_event
from langslice.atlas import (
    get_reference_slice,
    load_atlas,
    position_mm_to_index,
)
from langslice.registration.google.landmarks_image_gen import (
    _build_image_gen_config,
    _extract_generated_image,
    _image_to_typed_part,
    _species_from_atlas_name,
    _upscale_to_min_long_edge,
)
from langslice.registration.types import RegistrationAnnotationSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Atlas image generation helpers
# ---------------------------------------------------------------------------


# Ventricles and fluid-filled spaces should be black (matching histology holes).
_VENTRICLE_KEYWORDS = {"ventricle", "aqueduct", "central canal", "choroid", "subependymal"}


def _is_ventricle(structures: Any, uid: int) -> bool:
    """Return True if the structure is a ventricle or fluid-filled space."""
    try:
        name = str(structures[uid]["name"]).lower()
        return any(kw in name for kw in _VENTRICLE_KEYWORDS)
    except Exception:
        return False


def _generate_colored_region_slice(
    atlas: Any,
    position_mm: float,
    target_size: tuple[int, int] | None = None,
) -> Image.Image:
    """Render the annotation slice as an RGB image with atlas-defined colors.

    Each region ID is filled with its ``rgb_triplet`` from the atlas
    structures table.  Ventricles and fluid-filled spaces are rendered
    as black to match histology holes.  When *target_size* is given the
    result is resized using NEAREST interpolation to preserve exact colors.
    """
    idx = position_mm_to_index(atlas, position_mm)
    annotation_slice = np.asarray(atlas.annotation[idx, :, :])
    h, w = annotation_slice.shape

    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    structures = getattr(atlas, "structures", None)

    unique_ids = np.unique(annotation_slice)
    for uid in unique_ids:
        uid_int = int(uid)
        if uid_int == 0:
            continue
        # Ventricles stay black (0, 0, 0) to match histology holes
        if structures is not None and _is_ventricle(structures, uid_int):
            continue
        color = (128, 128, 128)  # fallback gray
        if structures is not None:
            try:
                triplet = structures[uid_int]["rgb_triplet"]
                if isinstance(triplet, (list, tuple)) and len(triplet) >= 3:
                    color = (int(triplet[0]), int(triplet[1]), int(triplet[2]))
            except Exception:
                pass
        mask = annotation_slice == uid_int
        rgb[mask] = color

    img = Image.fromarray(rgb, mode="RGB")
    if target_size is not None:
        img = img.resize(target_size, resample=Image.Resampling.NEAREST)
    return img


# ---------------------------------------------------------------------------
# Pixel classification
# ---------------------------------------------------------------------------


def _classify_pixels_to_region_ids(
    model_output_rgb: np.ndarray,
    atlas: Any,
    position_mm: float,
) -> np.ndarray:
    """Classify each pixel in *model_output_rgb* to the nearest atlas region ID.

    Builds a color-to-region-ID lookup from the atlas structures table, then
    assigns each pixel to the region whose ``rgb_triplet`` is closest in
    Euclidean RGB distance.  Pixels with brightness below 20 are classified
    as background (region ID 0).

    Returns a 2D ``int32`` array of region IDs with the same spatial
    dimensions as *model_output_rgb*.
    """
    idx = position_mm_to_index(atlas, position_mm)
    annotation_slice = np.asarray(atlas.annotation[idx, :, :])
    structures = getattr(atlas, "structures", None)

    # Build color -> region_id lookup
    color_to_id: dict[tuple[int, int, int], int] = {}
    unique_ids = np.unique(annotation_slice)
    for uid in unique_ids:
        uid_int = int(uid)
        if uid_int == 0:
            continue
        if structures is not None:
            try:
                triplet = structures[uid_int]["rgb_triplet"]
                if isinstance(triplet, (list, tuple)) and len(triplet) >= 3:
                    color = (int(triplet[0]), int(triplet[1]), int(triplet[2]))
                    color_to_id[color] = uid_int
            except Exception:
                pass

    if not color_to_id:
        logger.warning("No atlas structure colors found; returning all-zero classification")
        return np.zeros(model_output_rgb.shape[:2], dtype=np.int32)

    # Build arrays for vectorized nearest-color lookup
    palette_colors = np.array(list(color_to_id.keys()), dtype=np.float32)  # (K, 3)
    palette_ids = np.array(list(color_to_id.values()), dtype=np.int32)  # (K,)

    h, w = model_output_rgb.shape[:2]
    pixels = model_output_rgb.reshape(-1, 3).astype(np.float32)  # (N, 3)

    # Background mask: brightness < 20
    brightness = np.max(pixels, axis=1)
    bg_mask = brightness < 20.0

    # Compute squared distances to each palette color
    # Using broadcasting: (N, 1, 3) - (1, K, 3) -> (N, K, 3) -> sum -> (N, K)
    diff = pixels[:, np.newaxis, :] - palette_colors[np.newaxis, :, :]
    sq_dist = np.sum(diff * diff, axis=2)
    nearest_idx = np.argmin(sq_dist, axis=1)

    classified = palette_ids[nearest_idx]
    classified[bg_mask] = 0

    return classified.reshape(h, w)


# ---------------------------------------------------------------------------
# Border extraction from classified maps
# ---------------------------------------------------------------------------


def _extract_borders_from_classified(classified_2d: np.ndarray) -> np.ndarray:
    """Extract smoothed region borders from a classified region-ID map.

    For each unique region ID, contours are found via ``cv2.findContours``,
    smoothed with ``gaussian_filter1d(sigma=3.0)``, and drawn with
    anti-aliased lines.

    Returns a grayscale ``uint8`` array (0 or 255) with the same spatial
    dimensions as *classified_2d*.
    """
    h, w = classified_2d.shape
    canvas = np.zeros((h, w), dtype=np.uint8)

    unique_ids = np.unique(classified_2d)
    for uid in unique_ids:
        uid_int = int(uid)
        if uid_int == 0:
            continue
        region_mask = (classified_2d == uid_int).astype(np.uint8) * 255
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        smoothed_contours = []
        for contour in contours:
            if len(contour) < 5:
                smoothed_contours.append(contour)
                continue
            pts = contour.squeeze(axis=1).astype(np.float64)
            pts[:, 0] = gaussian_filter1d(pts[:, 0], sigma=3.0, mode="wrap")
            pts[:, 1] = gaussian_filter1d(pts[:, 1], sigma=3.0, mode="wrap")
            smoothed = pts.astype(np.int32).reshape(-1, 1, 2)
            smoothed_contours.append(smoothed)
        cv2.drawContours(canvas, smoothed_contours, -1, 255, 1, lineType=cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# Elastix B-spline registration
# ---------------------------------------------------------------------------


def _run_elastix_registration(
    fixed_gray: np.ndarray,
    moving_gray: np.ndarray,
) -> tuple[Any, float]:
    """Run itk-elastix B-spline registration.

    *fixed_gray* is the target image (model output) and *moving_gray* is
    the source image (atlas) to be warped.  The returned transform maps
    from fixed (model) space to moving (atlas) space, so
    ``transformix(atlas_image, transform)`` warps the atlas into the
    model's deformed shape.

    Configuration:
        - FinalGridSpacingInPhysicalUnits: 64
        - Metric: AdvancedMeanSquares + TransformBendingEnergyPenalty (weight 10.0)
        - MaximumNumberOfIterations: 512
        - NumberOfResolutions: 4

    Returns ``(result_transform, elapsed_seconds)``.
    """
    import itk

    t0 = time.perf_counter()

    fixed_image = itk.image_from_array(fixed_gray.astype(np.float32))
    moving_image = itk.image_from_array(moving_gray.astype(np.float32))

    parameter_object = itk.ParameterObject.New()  # type: ignore[attr-defined]
    parameter_map = parameter_object.GetDefaultParameterMap("bspline")

    parameter_map["FinalGridSpacingInPhysicalUnits"] = ("64",)
    parameter_map["Metric"] = ("AdvancedMeanSquares", "TransformBendingEnergyPenalty")
    parameter_map["Metric0Weight"] = ("1.0",)
    parameter_map["Metric1Weight"] = ("10.0",)
    parameter_map["MaximumNumberOfIterations"] = ("512",)
    parameter_map["NumberOfResolutions"] = ("4",)

    parameter_object.AddParameterMap(parameter_map)

    result_image, result_transform = itk.elastix_registration_method(  # type: ignore[attr-defined]
        fixed_image,
        moving_image,
        parameter_object=parameter_object,
        log_to_console=False,
    )

    elapsed = time.perf_counter() - t0
    logger.info("Elastix B-spline registration completed in %.1fs", elapsed)
    return result_transform, elapsed


# ---------------------------------------------------------------------------
# Warp atlas RGB through Elastix transform
# ---------------------------------------------------------------------------


def _warp_atlas_rgb(
    atlas_rgb: np.ndarray,
    result_transform: Any,
) -> np.ndarray:
    """Warp the atlas RGB image through the Elastix B-spline transform.

    Each channel is warped independently via ``itk.transformix_filter`` and
    the results are recombined into an RGB array.
    """
    import itk

    warped_channels = []
    for ch in range(3):
        channel_image = itk.image_from_array(atlas_rgb[:, :, ch].astype(np.float32))
        warped = itk.transformix_filter(channel_image, result_transform)  # type: ignore[attr-defined]
        warped_channels.append(itk.array_from_image(warped).astype(np.uint8))

    return np.stack(warped_channels, axis=-1)


# ---------------------------------------------------------------------------
# VisuAlign marker extraction from B-spline control points
# ---------------------------------------------------------------------------


def _extract_visualign_markers(
    result_transform: Any,
    scale_to_slice: float,
    image_width: int,
    image_height: int,
) -> list[list[float]]:
    """Read B-spline control points and convert to VisuAlign ``[ox, oy, nx, ny]`` markers.

    Each marker encodes:
        - ``(ox, oy)``: original atlas control point position
        - ``(nx, ny)``: displaced position after the B-spline warp

    Positions are scaled from registration-image coordinates to
    original-slice pixel coordinates using *scale_to_slice*.  Points
    outside image bounds are filtered out.
    """
    # Read the last transform parameter map from the result
    n_maps = result_transform.GetNumberOfParameterMaps()
    if n_maps == 0:
        logger.warning("No parameter maps in Elastix transform; returning empty markers")
        return []

    param_map = result_transform.GetParameterMap(n_maps - 1)

    grid_size_str = param_map.get("GridSize", ())
    grid_origin_str = param_map.get("GridOrigin", ())
    grid_spacing_str = param_map.get("GridSpacing", ())
    transform_params_str = param_map.get("TransformParameters", ())

    if not grid_size_str or not grid_origin_str or not grid_spacing_str:
        logger.warning("Missing B-spline grid parameters; returning empty markers")
        return []

    grid_size = [int(s) for s in grid_size_str]
    grid_origin = [float(s) for s in grid_origin_str]
    grid_spacing = [float(s) for s in grid_spacing_str]
    transform_params = [float(s) for s in transform_params_str]

    if len(grid_size) < 2:
        logger.warning("Grid size has fewer than 2 dimensions; returning empty markers")
        return []

    nx_grid, ny_grid = grid_size[0], grid_size[1]
    n_control_points = nx_grid * ny_grid

    if len(transform_params) < 2 * n_control_points:
        logger.warning(
            "Transform parameters count (%d) < expected (%d); returning empty markers",
            len(transform_params),
            2 * n_control_points,
        )
        return []

    markers: list[list[float]] = []
    for j in range(ny_grid):
        for i in range(nx_grid):
            cp_idx = j * nx_grid + i
            # Atlas (original) position of this control point
            ox = (grid_origin[0] + i * grid_spacing[0]) * scale_to_slice
            oy = (grid_origin[1] + j * grid_spacing[1]) * scale_to_slice
            # Displacement from transform parameters
            dx = transform_params[cp_idx] * scale_to_slice
            dy = transform_params[n_control_points + cp_idx] * scale_to_slice
            nx_pos = ox + dx
            ny_pos = oy + dy

            # Filter out points outside image bounds
            if ox < 0 or ox >= image_width or oy < 0 or oy >= image_height:
                continue
            if nx_pos < 0 or nx_pos >= image_width or ny_pos < 0 or ny_pos >= image_height:
                continue

            markers.append([ox, oy, nx_pos, ny_pos])

    logger.info(
        "Extracted %d VisuAlign markers from %d control points", len(markers), n_control_points
    )
    return markers


# ---------------------------------------------------------------------------
# v13 prompt text (locked 2026-04-03)
# ---------------------------------------------------------------------------

_SEGMENTATION_PROMPT = (
    "Deform the colored region map (Image 1) to match the shape of the "
    "real brain tissue in Image 3.\n"
    "\n"
    "IMAGE 1: A colored brain atlas region map. Each anatomical region "
    "is a unique solid color.\n"
    "IMAGE 2: A grayscale atlas reference showing the same brain anatomy.\n"
    "IMAGE 3: A real histology photograph of a mouse brain coronal section.\n"
    "\n"
    "Warp each colored region in Image 1 so it aligns with the "
    "corresponding anatomy visible in Image 3. The atlas is symmetric "
    "and idealized; the real tissue is asymmetric, stretched, and may "
    "have tears or damage. Use Image 2 to identify how atlas structures "
    "correspond to features in the histology.\n"
    "\n"
    "Preserve the exact colors from Image 1. Where tissue is damaged "
    "or missing in Image 3, fill in the expected atlas regions. Reflect "
    "the natural left-right asymmetry of this individual brain section.\n"
    "\n"
    "Output only the deformed colored regions on a black background."
)


# ---------------------------------------------------------------------------
# Gemini image-gen request
# ---------------------------------------------------------------------------


def _request_warped_segmentation(
    client: Any,
    *,
    model_name: str,
    thinking_level: str,
    colored_regions: Image.Image,
    reference_slice: Image.Image,
    slice_image: Image.Image,
    on_progress: Callable[[str], None] | None = None,
) -> Image.Image:
    """Send three images to Gemini and return the model-generated colored segmentation.

    Raises ``RuntimeError`` if the model does not return an image.
    """
    types_mod = importlib.import_module("google.genai.types")
    content_cls = types_mod.Content
    part_cls = types_mod.Part

    contents = [
        content_cls(
            role="user",
            parts=[
                _image_to_typed_part(colored_regions),
                _image_to_typed_part(reference_slice),
                _image_to_typed_part(slice_image),
                part_cls.from_text(text=_SEGMENTATION_PROMPT),
            ],
        )
    ]

    config = _build_image_gen_config(
        model_name=model_name,
        thinking_level=thinking_level,
    )

    response = _agents._retry_generate_stream(
        client,
        model=model_name,
        contents=contents,
        config=config,
        request_label="Colored segmentation Gemini pass",
        on_progress=on_progress,
    )

    model_output = _extract_generated_image(response)
    if model_output is None:
        raise RuntimeError(
            "Colored segmentation pass did not return an image -- "
            "this workflow requires the model to generate a warped region map"
        )

    logger.info("Received model output image: %dx%d", model_output.width, model_output.height)
    return model_output


# ---------------------------------------------------------------------------
# Local registration: colored images → grayscale → Elastix
# ---------------------------------------------------------------------------


def _register_colored_images(
    atlas_rgb: np.ndarray,
    model_output_rgb: np.ndarray,
) -> tuple[Any, float]:
    """Register atlas to model output using grayscale of the colored images.

    The model output is the fixed (target) image and the atlas is the
    moving (source) image.  This means ``transformix(atlas, transform)``
    warps the atlas into the model's deformed shape.

    Returns ``(result_transform, elapsed)``.
    """
    atlas_gray = cv2.cvtColor(atlas_rgb, cv2.COLOR_RGB2GRAY)
    model_gray = cv2.cvtColor(model_output_rgb, cv2.COLOR_RGB2GRAY)

    # fixed=model (target), moving=atlas (source to warp)
    return _run_elastix_registration(model_gray, atlas_gray)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def _estimate_correspondences_colored_segmentation(
    client: Any,
    *,
    prepared: _agents._PreparedRegistrationInputs,
    atlas_name: str,
    position_mm: float,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
    on_annotation_session: Callable[[RegistrationAnnotationSession], None] | None = None,
) -> list[dict[str, object]]:
    """Run the colored-segmentation registration workflow.

    Sends three atlas/slice images to a Gemini image-gen model with the
    v13 prompt, then recovers a dense deformation field via itk-elastix
    B-spline registration on grayscale versions of the colored images.

    Returns an empty correspondence list (the dense transform is attached
    to the annotation session metadata instead of sparse point pairs).
    """
    species = _species_from_atlas_name(atlas_name)
    atlas = load_atlas(atlas_name)

    # ------------------------------------------------------------------
    # 1. Determine target dimensions — cap at 2K long edge
    # ------------------------------------------------------------------
    slice_w, slice_h = prepared.slice_image_size
    max_long_edge = 2048
    long_edge = max(slice_w, slice_h)
    if long_edge > max_long_edge:
        scale_factor = max_long_edge / long_edge
        target_size = (int(slice_w * scale_factor), int(slice_h * scale_factor))
    else:
        target_size = (slice_w, slice_h)
    logger.info(
        "Target registration size: %dx%d (from slice %dx%d)",
        target_size[0], target_size[1], slice_w, slice_h,
    )

    # ------------------------------------------------------------------
    # 2. Generate atlas images at native aspect ratio, slice at 2K
    # ------------------------------------------------------------------
    if on_progress:
        on_progress("Colored segmentation: generating atlas input images...")

    colored_regions = _upscale_to_min_long_edge(
        _generate_colored_region_slice(atlas, position_mm, None)
    )
    reference_slice = _upscale_to_min_long_edge(
        get_reference_slice(atlas, position_mm).convert("RGB")
    )

    slice_image = prepared.slice_prep.image
    slice_w_prep, slice_h_prep = slice_image.size
    if max(slice_w_prep, slice_h_prep) > max_long_edge:
        sf = max_long_edge / max(slice_w_prep, slice_h_prep)
        slice_image = slice_image.resize(
            (int(slice_w_prep * sf), int(slice_h_prep * sf)),
            resample=Image.Resampling.LANCZOS,
        )

    _agents._emit_trace(
        on_trace,
        runtime_event(
            stage="registration",
            title="Colored segmentation inputs prepared",
            summary=(
                f"3 atlas images generated at AP {position_mm:.3f}mm, "
                f"target size {target_size[0]}x{target_size[1]}"
            ),
            parts=[
                image_part_from_pil(colored_regions, label="Colored region map"),
                image_part_from_pil(reference_slice, label="Grayscale reference"),
                image_part_from_pil(slice_image, label="Histology slice"),
            ],
            metadata={
                "workflow": "colored_segmentation",
                "position_mm": round(position_mm, 3),
                "target_size": list(target_size),
            },
        ),
    )

    if prepared.registration_dir:
        os.makedirs(prepared.registration_dir, exist_ok=True)
        colored_regions.save(
            os.path.join(prepared.registration_dir, "input_colored_regions.png")
        )
        reference_slice.convert("RGB").save(
            os.path.join(prepared.registration_dir, "input_reference.png")
        )
        slice_image.convert("RGB").save(
            os.path.join(prepared.registration_dir, "input_slice.png")
        )

    # ------------------------------------------------------------------
    # 3. Send images to Gemini image-gen model
    # ------------------------------------------------------------------
    if on_progress:
        on_progress("Colored segmentation: sending images to Gemini image-gen model...")

    _agents._emit_trace(
        on_trace,
        runtime_event(
            stage="registration",
            title="Colored segmentation Gemini request",
            summary="3 images + v13 prompt sent to Gemini image-gen model",
            metadata={
                "workflow": "colored_segmentation",
                "model": prepared.model_name,
                "thinking_level": prepared.thinking_level,
            },
        ),
    )

    model_output = _request_warped_segmentation(
        client,
        model_name=prepared.model_name,
        thinking_level=prepared.thinking_level,
        colored_regions=colored_regions,
        reference_slice=reference_slice,
        slice_image=slice_image,
        on_progress=on_progress,
    )

    _agents._emit_trace(
        on_trace,
        runtime_event(
            stage="registration",
            title="Colored segmentation model output",
            summary=f"Model generated {model_output.width}x{model_output.height} image",
            parts=[
                image_part_from_pil(model_output, label="Model-generated colored segmentation"),
            ],
            metadata={"workflow": "colored_segmentation"},
        ),
    )

    if prepared.registration_dir:
        model_output.save(
            os.path.join(prepared.registration_dir, "model_colored_output.png")
        )

    # Resize model output to match target dimensions for registration
    model_output_resized = model_output.resize(target_size, resample=Image.Resampling.LANCZOS)
    model_output_rgb = np.asarray(model_output_resized, dtype=np.uint8)

    # ------------------------------------------------------------------
    # 4. Register colored images via Elastix B-spline
    # ------------------------------------------------------------------
    if on_progress:
        on_progress("Colored segmentation: running Elastix B-spline registration...")

    # Generate atlas colored regions at target size for registration
    atlas_colored_at_target = _generate_colored_region_slice(
        atlas, position_mm, target_size
    )
    atlas_target_rgb = np.asarray(atlas_colored_at_target, dtype=np.uint8)

    result_transform, elastix_elapsed = _register_colored_images(
        atlas_target_rgb, model_output_rgb
    )

    _agents._emit_trace(
        on_trace,
        runtime_event(
            stage="registration",
            title="Elastix registration complete",
            summary=f"B-spline registration completed in {elastix_elapsed:.1f}s",
            metadata={
                "workflow": "colored_segmentation",
                "elastix_elapsed_s": round(elastix_elapsed, 2),
            },
        ),
    )

    if on_progress:
        on_progress(f"Colored segmentation: Elastix completed in {elastix_elapsed:.1f}s")

    # ------------------------------------------------------------------
    # 5. Warp atlas through transform and extract borders
    # ------------------------------------------------------------------
    if on_progress:
        on_progress("Colored segmentation: warping atlas through transform...")

    warped_atlas_rgb = _warp_atlas_rgb(atlas_target_rgb, result_transform)

    warped_atlas_img = Image.fromarray(warped_atlas_rgb, mode="RGB")
    if prepared.registration_dir:
        warped_atlas_img.save(
            os.path.join(prepared.registration_dir, "warped_atlas_rgb.png")
        )

    warped_classified = _classify_pixels_to_region_ids(warped_atlas_rgb, atlas, position_mm)
    warped_borders = _extract_borders_from_classified(warped_classified)
    if prepared.registration_dir:
        Image.fromarray(warped_borders, mode="L").save(
            os.path.join(prepared.registration_dir, "warped_borders.png")
        )

    # ------------------------------------------------------------------
    # 6. Extract VisuAlign markers from B-spline control points
    # ------------------------------------------------------------------
    if on_progress:
        on_progress("Colored segmentation: extracting VisuAlign markers...")

    scale_to_slice = float(slice_w) / float(target_size[0])

    markers = _extract_visualign_markers(
        result_transform,
        scale_to_slice=scale_to_slice,
        image_width=slice_w,
        image_height=slice_h,
    )

    _agents._emit_trace(
        on_trace,
        runtime_event(
            stage="registration",
            title="Colored segmentation results",
            summary=(
                f"Extracted {len(markers)} VisuAlign markers, "
                f"Elastix took {elastix_elapsed:.1f}s"
            ),
            parts=[
                image_part_from_pil(warped_atlas_img, label="Warped atlas RGB"),
                image_part_from_pil(
                    Image.fromarray(warped_borders, mode="L"),
                    label="Warped borders",
                ),
                image_part_from_pil(model_output, label="Model output"),
                json_part(
                    {
                        "n_markers": len(markers),
                        "elastix_elapsed_s": round(elastix_elapsed, 2),
                        "target_size": list(target_size),
                        "scale_to_slice": round(scale_to_slice, 4),
                    },
                    label="Registration summary",
                ),
            ],
            metadata={
                "workflow": "colored_segmentation",
                "n_markers": len(markers),
                "elastix_elapsed_s": round(elastix_elapsed, 2),
            },
        ),
    )

    # ------------------------------------------------------------------
    # 7. Build annotation session with dense transform metadata
    # ------------------------------------------------------------------
    session = RegistrationAnnotationSession(
        workflow="colored_segmentation",
        target_count=0,
        metadata={
            "visualign_markers": markers,
            "n_markers": len(markers),
            "elastix_elapsed_s": round(elastix_elapsed, 2),
            "target_size": list(target_size),
            "scale_to_slice": round(scale_to_slice, 4),
            "model_name": prepared.model_name,
            "position_mm": round(position_mm, 3),
            "atlas_name": atlas_name,
            "species": species,
        },
    )

    if on_annotation_session:
        on_annotation_session(session)

    if on_progress:
        on_progress(
            f"Colored segmentation complete: {len(markers)} markers, "
            f"{elastix_elapsed:.1f}s Elastix"
        )

    # Return empty list -- dense transforms are in the annotation session
    # metadata, not as sparse correspondence entries.
    return []

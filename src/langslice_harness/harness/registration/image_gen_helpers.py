"""Deterministic helpers for image-gen registration."""

from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d

from langslice_harness.atlas import position_mm_to_index

logger = logging.getLogger(__name__)


_VENTRICLE_KEYWORDS = {"ventricle", "central canal", "choroid", "subependymal"}


def _upscale_to_min_long_edge(img: Image.Image, min_long_edge: int = 1024) -> Image.Image:
    """Upscale *img* so its long edge is at least *min_long_edge* pixels."""
    width, height = img.size
    long_edge = max(width, height)
    if long_edge >= min_long_edge:
        return img
    scale = min_long_edge / long_edge
    return img.resize(
        (round(width * scale), round(height * scale)),
        resample=Image.Resampling.LANCZOS,
    )


def _is_ventricle(structures: Any, uid: int) -> bool:
    try:
        name = str(structures[uid]["name"]).lower()
        return any(keyword in name for keyword in _VENTRICLE_KEYWORDS)
    except Exception:
        return False


def _generate_colored_region_slice(
    atlas: Any,
    position_mm: float,
    target_size: tuple[int, int] | None = None,
) -> Image.Image:
    """Render an atlas annotation slice as a color-preserving RGB region map."""
    idx = position_mm_to_index(atlas, position_mm)
    annotation_slice = np.asarray(atlas.annotation[idx, :, :])
    height, width = annotation_slice.shape

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    structures = getattr(atlas, "structures", None)
    for uid in np.unique(annotation_slice):
        uid_int = int(uid)
        if uid_int == 0:
            continue
        if structures is not None and _is_ventricle(structures, uid_int):
            continue

        color = (128, 128, 128)
        if structures is not None:
            try:
                triplet = structures[uid_int]["rgb_triplet"]
                if isinstance(triplet, (list, tuple)) and len(triplet) >= 3:
                    color = (int(triplet[0]), int(triplet[1]), int(triplet[2]))
            except Exception:
                pass
        rgb[annotation_slice == uid_int] = color

    image = Image.fromarray(rgb, mode="RGB")
    if target_size is not None:
        image = image.resize(target_size, resample=Image.Resampling.NEAREST)
    return image


def _classify_pixels_to_region_ids(
    model_output_rgb: np.ndarray,
    atlas: Any,
    position_mm: float,
) -> np.ndarray:
    """Classify RGB pixels to the nearest atlas region color at *position_mm*."""
    idx = position_mm_to_index(atlas, position_mm)
    annotation_slice = np.asarray(atlas.annotation[idx, :, :])
    structures = getattr(atlas, "structures", None)

    color_to_id: dict[tuple[int, int, int], int] = {}
    for uid in np.unique(annotation_slice):
        uid_int = int(uid)
        if uid_int == 0 or structures is None:
            continue
        try:
            triplet = structures[uid_int]["rgb_triplet"]
            if isinstance(triplet, (list, tuple)) and len(triplet) >= 3:
                color_to_id[(int(triplet[0]), int(triplet[1]), int(triplet[2]))] = uid_int
        except Exception:
            pass

    if not color_to_id:
        logger.warning("No atlas structure colors found; returning all-zero classification")
        return np.zeros(model_output_rgb.shape[:2], dtype=np.int32)

    palette_colors = np.array(list(color_to_id.keys()), dtype=np.float32)
    palette_ids = np.array(list(color_to_id.values()), dtype=np.int32)

    height, width = model_output_rgb.shape[:2]
    pixels = model_output_rgb.reshape(-1, 3).astype(np.float32)
    background_mask = np.max(pixels, axis=1) < 20.0

    diff = pixels[:, np.newaxis, :] - palette_colors[np.newaxis, :, :]
    nearest_idx = np.argmin(np.sum(diff * diff, axis=2), axis=1)
    classified = palette_ids[nearest_idx]
    classified[background_mask] = 0
    return classified.reshape(height, width)


def _extract_borders_from_classified(classified_2d: np.ndarray) -> np.ndarray:
    """Extract anti-aliased, smoothed region borders from classified region IDs."""
    height, width = classified_2d.shape
    canvas = np.zeros((height, width), dtype=np.uint8)

    for uid in np.unique(classified_2d):
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
            smoothed_contours.append(pts.astype(np.int32).reshape(-1, 1, 2))
        cv2.drawContours(canvas, smoothed_contours, -1, 255, 1, lineType=cv2.LINE_AA)

    return canvas


def _run_elastix_registration(
    fixed_gray: np.ndarray,
    moving_gray: np.ndarray,
) -> tuple[Any, float]:
    """Run two-stage affine plus B-spline Elastix registration."""
    import itk

    start = time.perf_counter()
    fixed_image = itk.image_from_array(fixed_gray.astype(np.float32))
    moving_image = itk.image_from_array(moving_gray.astype(np.float32))

    parameter_object = itk.ParameterObject.New()  # type: ignore[attr-defined]

    affine_map = parameter_object.GetDefaultParameterMap("affine")
    affine_map["Metric"] = ("AdvancedNormalizedCorrelation",)
    affine_map["MaximumNumberOfIterations"] = ("512",)
    affine_map["NumberOfResolutions"] = ("4",)
    affine_map["AutomaticTransformInitialization"] = ("true",)
    affine_map["AutomaticTransformInitializationMethod"] = ("CenterOfGravity",)
    parameter_object.AddParameterMap(affine_map)

    bspline_map = parameter_object.GetDefaultParameterMap("bspline")
    bspline_map["FinalGridSpacingInPhysicalUnits"] = ("32",)
    bspline_map["Metric"] = ("AdvancedNormalizedCorrelation", "TransformBendingEnergyPenalty")
    bspline_map["Metric0Weight"] = ("1.0",)
    bspline_map["Metric1Weight"] = ("3.0",)
    bspline_map["MaximumNumberOfIterations"] = ("1024",)
    bspline_map["NumberOfResolutions"] = ("4",)
    parameter_object.AddParameterMap(bspline_map)

    _result_image, result_transform = itk.elastix_registration_method(  # type: ignore[attr-defined]
        fixed_image,
        moving_image,
        parameter_object=parameter_object,
        log_to_console=False,
    )

    elapsed = time.perf_counter() - start
    logger.info("Elastix affine+B-spline registration completed in %.1fs", elapsed)
    return result_transform, elapsed


def _register_colored_images(
    atlas_rgb: np.ndarray,
    model_output_rgb: np.ndarray,
) -> tuple[Any, float]:
    """Register atlas RGB regions to a model-generated RGB region map."""
    atlas_gray = cv2.cvtColor(atlas_rgb, cv2.COLOR_RGB2GRAY)
    model_gray = cv2.cvtColor(model_output_rgb, cv2.COLOR_RGB2GRAY)
    return _run_elastix_registration(model_gray, atlas_gray)


def _warp_atlas_rgb(atlas_rgb: np.ndarray, result_transform: Any) -> np.ndarray:
    """Warp an RGB atlas image through an Elastix transform."""
    import itk

    warped_channels = []
    for channel in range(3):
        channel_image = itk.image_from_array(atlas_rgb[:, :, channel].astype(np.float32))
        warped = itk.transformix_filter(channel_image, result_transform)  # type: ignore[attr-defined]
        warped_channels.append(itk.array_from_image(warped).astype(np.uint8))
    return np.stack(warped_channels, axis=-1)


def _extract_visualign_markers(
    result_transform: Any,
    scale_to_slice: float,
    image_width: int,
    image_height: int,
) -> list[list[float]]:
    """Convert B-spline control-point displacements to VisuAlign markers."""
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

    grid_size = [int(value) for value in grid_size_str]
    grid_origin = [float(value) for value in grid_origin_str]
    grid_spacing = [float(value) for value in grid_spacing_str]
    transform_params = [float(value) for value in transform_params_str]

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
    for y_idx in range(ny_grid):
        for x_idx in range(nx_grid):
            cp_idx = y_idx * nx_grid + x_idx
            ox = (grid_origin[0] + x_idx * grid_spacing[0]) * scale_to_slice
            oy = (grid_origin[1] + y_idx * grid_spacing[1]) * scale_to_slice
            dx = transform_params[cp_idx] * scale_to_slice
            dy = transform_params[n_control_points + cp_idx] * scale_to_slice
            nx_pos = ox + dx
            ny_pos = oy + dy

            if ox < 0 or ox >= image_width or oy < 0 or oy >= image_height:
                continue
            if nx_pos < 0 or nx_pos >= image_width or ny_pos < 0 or ny_pos >= image_height:
                continue
            markers.append([ox, oy, nx_pos, ny_pos])

    logger.info(
        "Extracted %d VisuAlign markers from %d control points",
        len(markers),
        n_control_points,
    )
    return markers


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

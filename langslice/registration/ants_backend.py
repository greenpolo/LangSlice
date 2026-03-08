"""ANTsPyX-backed affine registration helpers."""

from __future__ import annotations

import importlib
import inspect
import logging
import math
import os
import tempfile
from typing import Any, Callable

import numpy as np
from PIL import Image
from scipy import ndimage

from langslice.atlas import get_reference_slice, load_atlas
from langslice.registration.types import AffineResult, is_valid_affine_matrix

logger = logging.getLogger(__name__)


class RegistrationFailure(RuntimeError):
    """Raised when the ANTs backend cannot produce a usable affine."""


def _normalize_grayscale_array(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    lo = float(gray.min())
    hi = float(gray.max())
    if hi > lo:
        gray = (gray - lo) / (hi - lo)
    else:
        gray = np.zeros_like(gray, dtype=np.float32)
    return gray.astype(np.float32)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, count = ndimage.label(mask)
    if count <= 1:
        return mask.astype(bool)
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, count + 1))
    winner = int(np.argmax(sizes)) + 1
    return labeled == winner


def _refine_mask(mask: np.ndarray) -> np.ndarray:
    refined = mask.astype(bool)
    refined = ndimage.binary_fill_holes(refined)
    refined = ndimage.binary_opening(refined, structure=np.ones((3, 3), dtype=bool))
    refined = ndimage.binary_closing(refined, structure=np.ones((5, 5), dtype=bool))
    refined = _largest_component(refined)
    refined = ndimage.binary_fill_holes(refined)
    return refined.astype(bool)


def _score_mask(mask: np.ndarray) -> float:
    fraction = float(mask.mean())
    if fraction <= 0.01 or fraction >= 0.99:
        return -1e6

    center = np.array(mask.shape, dtype=np.float64) / 2.0
    centroid = np.array(ndimage.center_of_mass(mask), dtype=np.float64)
    if not np.isfinite(centroid).all():
        return -1e6

    norm = np.linalg.norm(center) or 1.0
    center_distance = float(np.linalg.norm(centroid - center) / norm)
    border_pixels = np.concatenate(
        [mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]]
    )
    border_fraction = float(border_pixels.mean()) if border_pixels.size else 0.0
    return -abs(fraction - 0.35) - center_distance - border_fraction


def _build_tissue_mask(image: Image.Image) -> np.ndarray:
    gray = _normalize_grayscale_array(image)
    q25, q75 = np.percentile(gray, [25.0, 75.0])
    threshold = float((q25 + q75) / 2.0)
    candidates = [
        _refine_mask(gray <= threshold),
        _refine_mask(gray >= threshold),
    ]
    return max(candidates, key=_score_mask)


def _build_atlas_mask(image: Image.Image) -> np.ndarray:
    gray = _normalize_grayscale_array(image)
    threshold = max(0.03, float(np.percentile(gray, 10.0)))
    mask = gray > threshold
    return _refine_mask(mask)


def _load_ants_module() -> Any:
    try:
        return importlib.import_module("ants")
    except ImportError as exc:
        raise RegistrationFailure(
            "ANTsPyX is not installed in this environment."
        ) from exc


def _call_supported(func: Any, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(**kwargs)

    accepted = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return func(**accepted)


def _to_ants_image(ants_mod: Any, array: np.ndarray) -> Any:
    image = ants_mod.from_numpy(array.astype(np.float32))
    if hasattr(image, "set_spacing"):
        image.set_spacing((1.0, 1.0))
    if hasattr(image, "set_origin"):
        image.set_origin((0.0, 0.0))
    if hasattr(image, "set_direction"):
        try:
            image.set_direction(np.eye(2, dtype=np.float64))
        except Exception:
            pass
    return image


def _apply_transform_to_points(
    ants_mod: Any,
    transform: Any,
    points: list[tuple[float, float]],
) -> np.ndarray:
    if hasattr(transform, "apply_to_point") and callable(transform.apply_to_point):
        return np.asarray(
            [transform.apply_to_point(point) for point in points],
            dtype=np.float64,
        )

    apply_point = getattr(ants_mod, "apply_ants_transform_to_point", None)
    if callable(apply_point):
        return np.asarray(
            [apply_point(transform=transform, point=point) for point in points],
            dtype=np.float64,
        )

    raise RegistrationFailure("ANTsPyX transform object does not expose point mapping.")


def _matrix_from_transform(ants_mod: Any, transform_path: str) -> np.ndarray:
    read_transform = getattr(ants_mod, "read_transform", None)
    if not callable(read_transform):
        raise RegistrationFailure("ANTsPyX does not expose read_transform.")

    transform = read_transform(transform_path)
    points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    mapped = _apply_transform_to_points(ants_mod, transform, points)
    origin = mapped[0]
    ex = mapped[1] - origin
    ey = mapped[2] - origin
    return np.array(
        [
            [ex[0], ey[0], origin[0]],
            [ex[1], ey[1], origin[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def estimate_affine_with_ants(
    image: Image.Image,
    *,
    atlas_name: str,
    position_mm: float,
    on_progress: Callable[[str], None] | None = None,
) -> AffineResult:
    """Run ANTsPyX affine registration against the atlas reference slice."""

    def _progress(message: str) -> None:
        if on_progress:
            on_progress(message)
        logger.info(message)

    ants_mod = _load_ants_module()
    atlas = load_atlas(atlas_name)
    atlas_reference = get_reference_slice(atlas, position_mm)

    moving_gray = _normalize_grayscale_array(image)
    fixed_gray = _normalize_grayscale_array(atlas_reference)
    moving_mask = _build_tissue_mask(image)
    fixed_mask = _build_atlas_mask(atlas_reference)

    _progress(
        "ANTsPyX affine registration: "
        f"moving={image.width}x{image.height}, fixed={atlas_reference.width}x{atlas_reference.height}"
    )

    fixed_image = _to_ants_image(ants_mod, fixed_gray)
    moving_image = _to_ants_image(ants_mod, moving_gray)
    fixed_mask_image = _to_ants_image(ants_mod, fixed_mask.astype(np.float32))
    moving_mask_image = _to_ants_image(ants_mod, moving_mask.astype(np.float32))

    initial_transform: str | None = None
    with tempfile.TemporaryDirectory(prefix="langslice_ants_") as tmpdir:
        initializer_path = os.path.join(tmpdir, "initializer.mat")
        affine_initializer = getattr(ants_mod, "affine_initializer", None)
        if callable(affine_initializer):
            init_result = _call_supported(
                affine_initializer,
                fixed=fixed_mask_image,
                moving=moving_mask_image,
                txfn=initializer_path,
                search_factor=20,
                radian_fraction=0.1,
                local_search_iterations=10,
            )
            if isinstance(init_result, str) and init_result:
                initial_transform = init_result
            elif os.path.exists(initializer_path):
                initial_transform = initializer_path

        registration = _call_supported(
            ants_mod.registration,
            fixed=fixed_image,
            moving=moving_image,
            type_of_transform="AffineFast",
            aff_metric="mattes",
            mask=fixed_mask_image,
            moving_mask=moving_mask_image,
            mask_all_stages=True,
            initial_transform=initial_transform,
        )

        if not isinstance(registration, dict):
            raise RegistrationFailure("ANTsPyX registration returned an unexpected result.")

        fwd_transforms = registration.get("fwdtransforms", [])
        if not isinstance(fwd_transforms, list) or not fwd_transforms:
            raise RegistrationFailure("ANTsPyX registration did not return a forward transform.")

        matrix = _matrix_from_transform(ants_mod, str(fwd_transforms[0]))
        if not is_valid_affine_matrix(matrix):
            raise RegistrationFailure("ANTsPyX produced a degenerate affine transform.")

        determinant = float(np.linalg.det(matrix[:2, :2]))
        if not math.isfinite(determinant) or abs(determinant) < 1e-8:
            raise RegistrationFailure("ANTsPyX produced a near-singular affine transform.")

        reasoning = (
            f"ANTsPyX AffineFast registration against atlas reference slice at {position_mm:.2f} mm. "
            "Initialized from tissue masks and refined on grayscale images with the Mattes metric."
        )
        provenance = {
            "atlas_name": atlas_name,
            "position_mm": position_mm,
            "used_initializer": initial_transform is not None,
            "forward_transform_count": len(fwd_transforms),
        }

        return AffineResult(
            matrix=matrix,
            source_size=(image.width, image.height),
            output_size=(atlas_reference.width, atlas_reference.height),
            backend="antspyx",
            reasoning=reasoning,
            provenance=provenance,
        )

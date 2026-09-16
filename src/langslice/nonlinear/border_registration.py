"""Atlas placement, boundary correction and fully composed export coordinates."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from langslice.atlas import load_atlas
from langslice.atlas.render import annotation_slice
from langslice.nonlinear.border_refinement import refine_borders
from langslice.nonlinear.image_gen_helpers import (
    _classified_to_rgb,
    _elastix_report,
    _grid_spacing_px,
)
from langslice.nonlinear.prior import place_plane_on_tissue_with_matrix, tissue_mask
from langslice.nonlinear.types import (
    Deformation,
    RegistrationAnnotationSession,
    RegistrationCandidate,
)
from langslice.providers.registry import canonical_provider
from langslice.space import Plane, atlas_space_context, orient_slice_to_axes


def pixel_center_map(
    source_size: tuple[int, int],
    resized_size: tuple[int, int],
    offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Map pixel centers through resize and padding, including rounding per axis."""
    sx, sy = resized_size[0] / source_size[0], resized_size[1] / source_size[1]
    return np.array([
        [sx, 0.0, offset[0] + (sx - 1.0) / 2.0],
        [0.0, sy, offset[1] + (sy - 1.0) / 2.0],
        [0.0, 0.0, 1.0],
    ])


def canonical_atlas_map(native_size: tuple[int, int], canvas_size: tuple[int, int]) -> np.ndarray:
    """Native oriented atlas pixels→centered, uniformly fit atlas canvas pixels."""
    scale = min(canvas_size[0] / native_size[0], canvas_size[1] / native_size[1])
    resized = tuple(max(1, round(v * scale)) for v in native_size)
    offset = ((canvas_size[0] - resized[0]) // 2, (canvas_size[1] - resized[1]) // 2)
    return pixel_center_map(native_size, (resized[0], resized[1]), offset)


def _affine(value: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[2], [0, 0, 1])
            or abs(np.linalg.det(matrix[:2, :2])) < 1e-12):
        raise ValueError("Initial atlas-to-slice placement must be a finite invertible 3x3 affine")
    return matrix.copy()


def composed_correspondences(
    field: np.ndarray,
    atlas_to_canvas: np.ndarray | None,
    slice_to_canvas: np.ndarray,
    native_to_canonical: np.ndarray,
    initial_coordinate_map: np.ndarray | None = None,
) -> tuple[list[list[float]], list[list[float]]]:
    """Sample slice→canonical atlas markers and slice→native atlas points.

    The residual field maps output canvas q to placed atlas q+d. Undo rough
    placement before converting native atlas pixels into the canonical frame.
    The latter frame, like the source points, is reported in unpadded original
    image units. Native correspondences retain the unambiguous atlas grid.
    """
    if field.ndim != 3 or field.shape[2] != 2 or not np.isfinite(field).all():
        raise ValueError("Expected a finite H×W×2 residual deformation field")
    height, width = field.shape[:2]
    if not height or not width:
        raise ValueError("Cannot export an empty deformation field")
    spacing = _grid_spacing_px(field.shape)
    ys = np.unique(np.append(np.arange(0, height, spacing), height - 1))
    xs = np.unique(np.append(np.arange(0, width, spacing), width - 1))
    canvas_to_native = (
        np.linalg.inv(_affine(atlas_to_canvas)) if atlas_to_canvas is not None else None
    )
    if initial_coordinate_map is not None:
        if (initial_coordinate_map.shape != field.shape
                or not np.isfinite(initial_coordinate_map).all()):
            raise ValueError("Initial atlas coordinate map must match the finite canvas field")
    elif canvas_to_native is None:
        raise ValueError("Initial placement needs an affine or a dense native atlas coordinate map")
    canvas_to_slice = np.linalg.inv(_affine(slice_to_canvas))
    canonical = _affine(native_to_canonical)
    markers, direct = [], []
    for y in ys:
        for x in xs:
            q = np.array([float(x), float(y), 1.0])
            moved = q.copy()
            moved[:2] += field[y, x]
            if initial_coordinate_map is not None:
                native = np.array([
                    float(map_coordinates(initial_coordinate_map[..., axis],
                                          [[moved[1]], [moved[0]]], order=1,
                                          mode="nearest", prefilter=False)[0])
                    for axis in range(2)
                ] + [1.0])
            else:
                assert canvas_to_native is not None
                native = canvas_to_native @ moved
            source = canvas_to_slice @ q
            target = canvas_to_slice @ canonical @ native
            markers.append([float(source[0]), float(source[1]), float(target[0]), float(target[1])])
            direct.append([float(source[0]), float(source[1]), float(native[0]), float(native[1])])
    if not np.isfinite(np.asarray(markers)).all():
        raise ValueError("Composed registration correspondences are not finite")
    return markers, direct


def composed_native_map(
    field: np.ndarray,
    atlas_to_canvas: np.ndarray | None,
    initial_coordinate_map: np.ndarray | None = None,
) -> np.ndarray:
    """Compose every residual sample with the initial native atlas mapping."""
    yy, xx = np.indices(field.shape[:2], dtype=np.float64)
    x, y = xx + field[..., 0], yy + field[..., 1]
    if initial_coordinate_map is not None:
        return np.stack([
            map_coordinates(initial_coordinate_map[..., axis], [y, x], order=1,
                            mode="nearest", prefilter=False)
            for axis in range(2)
        ], axis=-1)
    if atlas_to_canvas is None:
        raise ValueError("Initial atlas placement is missing")
    inverse = np.linalg.inv(_affine(atlas_to_canvas))
    return np.stack([
        inverse[axis, 0] * x + inverse[axis, 1] * y + inverse[axis, 2]
        for axis in range(2)
    ], axis=-1)


def residual_fit_report(
    rough_labels: np.ndarray,
    fitted_labels: np.ndarray,
    field: np.ndarray,
    structures: Any = None,
) -> dict[str, Any]:
    """Mechanical residual checks; neither initial placement nor model anatomy QC."""
    if (fitted_labels.shape != rough_labels.shape
            or not np.issubdtype(fitted_labels.dtype, np.integer)):
        raise ValueError("Fitted atlas labels must remain an integer map on the rough canvas")
    allowed = np.append(np.unique(rough_labels), 0)
    if not np.isin(np.unique(fitted_labels), allowed).all():
        raise ValueError("Border fitting introduced region identities absent from the rough atlas")
    if field.shape != (*rough_labels.shape, 2) or not np.isfinite(field).all():
        raise ValueError("Residual deformation must be finite and match the fitted label canvas")
    report = _elastix_report(
        atlas_classified=rough_labels, warped_classified=fitted_labels,
        structures=structures,
        deformation_field=field if min(rough_labels.shape) >= 2 else None,
    )
    original_count = int(np.count_nonzero(rough_labels))
    fitted_count = int(np.count_nonzero(fitted_labels))
    ratio = fitted_count / original_count if original_count else None
    if original_count and fitted_count == 0:
        report["codes"].append({"code": "EMPTY_WARP", "rough_foreground_px": original_count})
    elif ratio is not None and ratio < 0.2:
        report["codes"].append({"code": "ATLAS_COLLAPSED", "foreground_retained_fraction": ratio})
    report.update({
        "scope": "residual deformation only; initial-stage report is separate",
        "description": "Mechanical fit diagnostics, not an assessment of model anatomy",
        "finite_residual_field": True, "foreground_retained_fraction": ratio,
        "max_residual_displacement_px": float(np.linalg.norm(field, axis=2).max()),
    })
    return report


def generate_border_registration_candidate(
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
    atlas_mirror_lr: bool = False,
    initial_atlas_to_slice: Sequence[Sequence[float]] | np.ndarray | None = None,
    initial_alignment_source: str = "supplied",
    canvas_pad: float = 0.0,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
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
) -> RegistrationCandidate:
    """Correct rough atlas borders, retaining exact region ids and placement.

    ``initial_atlas_to_slice`` maps the annotation grid AFTER ``image_axes``
    and ``atlas_mirror_lr`` into original image pixel centers. It takes
    precedence over the automatic silhouette placement. No reflection is
    inferred from symmetric tissue.
    """
    from langslice.nonlinear.image_gen_registration import (
        generate_registration_candidate,
        prepare_canvas,
    )

    if on_progress:
        on_progress("Preparing rough atlas boundaries on the original section...")
    if (generated_image is not None and initial_atlas_to_slice is None
            and canonical_provider(provider) != "none"):
        raise ValueError("Border replay requires a supplied initial alignment")
    candidate_id = candidate_id or f"candidate-{uuid.uuid4().hex[:12]}"
    atlas = load_atlas(atlas_name)
    labels = annotation_slice(
        atlas, position_mm, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg
    )
    if image_axes:
        labels = orient_slice_to_axes(labels, atlas_space_context(atlas), plane, image_axes)
    if atlas_mirror_lr:
        labels = np.fliplr(labels)
    labels = np.array(labels)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Atlas annotation must be a two-dimensional integer label map")
    native_size = (labels.shape[1], labels.shape[0])
    canvas, unpadded, ox, oy, pad_px = prepare_canvas(
        image, canvas_pad=canvas_pad, image_model=image_model, provider=provider,
        native_canvas=native_canvas, canvas_long_edge=canvas_long_edge, quality=thinking_level,
    )
    original_to_canvas = pixel_center_map(image.size, unpadded, (ox, oy))
    canonical = canonical_atlas_map(native_size, canvas.size)
    prior: dict[str, Any]
    initial_coordinate_map = None
    first = None
    if initial_atlas_to_slice is not None:
        initial = _affine(initial_atlas_to_slice)
        atlas_to_canvas = original_to_canvas @ initial
        rough = cv2.warpAffine(
            labels.astype(np.float64), atlas_to_canvas[:2], canvas.size,
            flags=cv2.INTER_NEAREST, borderValue=0,
        ).astype(labels.dtype)
        prior = {"source": initial_alignment_source, "automatic": False}
    elif canonical_provider(provider) == "none":
        rough, signs, iou, atlas_to_canvas = place_plane_on_tissue_with_matrix(
            labels, tissue_mask(canvas, canvas.size)
        )
        initial = np.linalg.inv(original_to_canvas) @ atlas_to_canvas
        prior = {
            "source": "silhouette_moments", "automatic": True,
            "sign_pattern": list(signs), "silhouette_iou": float(iou),
        }
    else:
        # Standalone registration retains the two-call route: first generate
        # and fit the color atlas; then correct its boundaries on raw tissue.
        first = generate_registration_candidate(
            image, atlas_name=atlas_name, position_mm=position_mm, plane=plane,
            provider=provider, image_model=image_model, image_axes=image_axes,
            atlas_mirror_lr=atlas_mirror_lr, canvas_pad=canvas_pad,
            pitch_deg=pitch_deg, yaw_deg=yaw_deg, deformation=deformation,
            candidate_id=f"{candidate_id}-initial", debug_dir=debug_dir,
            on_progress=on_progress, on_trace=on_trace,
            openai_image_route=openai_image_route, review_model=review_model,
            thinking_level=thinking_level, native_canvas=native_canvas,
            canvas_long_edge=canvas_long_edge, registration_mode="colormap",
        )
        if first.warped_labels is None or first.atlas_coordinate_map is None:
            raise RuntimeError(
                "Initial color registration did not retain atlas labels and coordinates"
            )
        rough = first.warped_labels
        initial_coordinate_map = first.atlas_coordinate_map
        if (rough.shape != (canvas.height, canvas.width)
                or initial_coordinate_map.shape != (*rough.shape, 2)):
            raise RuntimeError(
                "Initial color registration and correction use different canvas frames"
            )
        for key, expected in (("target_size", list(canvas.size)),
                              ("unpadded_size", list(unpadded)),
                              ("canvas_origin_px", [ox, oy]),
                              ("native_atlas_size", list(native_size))):
            if first.metadata.get(key) != expected:
                raise RuntimeError(f"Initial color registration has inconsistent {key}")
        initial = None
        atlas_to_canvas = None
        prior = {"source": "colormap_registration", "automatic": True,
                 "candidate_id": first.candidate_id, "image_generation_calls": 1,
                 "coordinate_map_edge_mode": "nearest (clamp samples outside the initial canvas)"}
    if on_progress:
        on_progress("Correcting atlas boundaries and fitting the residual deformation...")
    result = refine_borders(
        canvas, rough, atlas, provider=provider, model=image_model, plane=plane,
        review_model=review_model, openai_image_route=openai_image_route,
        thinking_level=thinking_level, generated_image=generated_image,
        deformation=deformation, image_prompt=image_prompt,
    )
    fit_report = residual_fit_report(
        rough, result.fitted_labels, result.deformation_field, getattr(atlas, "structures", None)
    )
    markers, native_points = composed_correspondences(
        result.deformation_field, atlas_to_canvas, original_to_canvas, canonical,
        initial_coordinate_map=initial_coordinate_map,
    )
    final_native_map = composed_native_map(
        result.deformation_field, atlas_to_canvas, initial_coordinate_map
    )
    warped_atlas = Image.fromarray(_classified_to_rgb(result.fitted_labels, atlas))
    metadata: dict[str, Any] = {
        **result.metadata,
        "workflow": "border_refinement", "output_kind": "border_overlay",
        "candidate_id": candidate_id, "atlas_name": atlas_name,
        "position_mm": float(position_mm), "plane": plane,
        "pitch_deg": float(pitch_deg), "yaw_deg": float(yaw_deg),
        "image_axes": image_axes, "atlas_mirror_lr": bool(atlas_mirror_lr),
        "original_size": list(image.size), "target_size": list(canvas.size),
        "unpadded_size": list(unpadded), "canvas_origin_px": [ox, oy],
        "canvas_pad": float(canvas_pad), "pad_px": pad_px,
        "canvas_native": bool(native_canvas and canvas_long_edge is None),
        "native_atlas_size": list(native_size),
        "native_atlas_frame": "annotation_slice after image_axes and explicit atlas_mirror_lr",
        "initial_atlas_to_slice": initial.tolist() if initial is not None else None,
        "initial_alignment_source": prior["source"],
        "atlas_to_canvas": atlas_to_canvas.tolist() if atlas_to_canvas is not None else None,
        "slice_to_canvas": original_to_canvas.tolist(),
        "native_atlas_to_canonical_canvas": canonical.tolist(),
        "visualign_markers": markers, "n_markers": len(markers),
        "marker_frames": {
            "source": "original section pixel centers (may extend outside original image)",
            "target": "canonical letterboxed atlas canvas mapped to original image units",
            "composition": (
                "inverse(slice_to_canvas) @ native_atlas_to_canonical_canvas @ "
                "initial_native_map(canvas_point + residual_displacement)"
            ),
        },
        "slice_to_native_atlas_correspondences": native_points,
        "native_correspondence_columns": ["slice_x", "slice_y", "native_atlas_x", "native_atlas_y"],
        "prior": prior, "inverse_warp_status": "not_computed",
        "elastix_elapsed_s": float(result.elapsed),
        "elastix_stage": "border_residual",
        "elastix": fit_report,
        "initial_stage_elastix": first.metadata.get("elastix") if first is not None else None,
    }
    if previous_candidate_id is not None:
        metadata["previous_candidate_id"] = previous_candidate_id
    paths: dict[str, str | None] = {
        key: None for key in (
            "raw_correction_path", "rough_border_overlay_path", "corrected_border_overlay_path",
            "generated_segmentation_path", "generated_border_overlay_path",
            "warped_atlas_path", "warped_border_overlay_path",
            "slice_warped_to_atlas_path", "slice_atlas_border_overlay_path",
        )
    }
    if debug_dir is not None:
        directory = Path(debug_dir) / "registration" / candidate_id
        directory.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "input_slice.png": canvas,
            "rough_border_overlay.png": result.rough_border_overlay,
            "raw_correction.png": result.raw_model_image,
            "generated_segmentation.png": result.raw_model_image,
            "corrected_border_overlay.png": result.model_border_overlay,
            "generated_border_overlay.png": result.model_border_overlay,
            "warped_atlas.png": warped_atlas,
            "warped_border_overlay.png": result.fitted_border_overlay,
        }
        for filename, artifact in artifacts.items():
            artifact.save(directory / filename)
            key = f"{Path(filename).stem}_path"
            if key in paths:
                paths[key] = str((directory / filename).resolve())
        np.savez_compressed(directory / "rough_leaf_ids.npz", ids=rough)
        np.savez_compressed(directory / "warped_leaf_ids.npz", ids=result.fitted_labels)
        np.savez_compressed(directory / "residual_field.npz", field=result.deformation_field)
        np.savez_compressed(directory / "atlas_coordinate_map.npz", coordinates=final_native_map)
        if initial_coordinate_map is not None:
            np.savez_compressed(directory / "initial_atlas_coordinate_map.npz",
                                coordinates=initial_coordinate_map)
        Image.fromarray(result.model_border_mask.astype(np.uint8) * 255).save(
            directory / "corrected_border_mask.png"
        )
        (directory / "prompt.txt").write_text(str(result.metadata["prompt"]))
    metadata["artifact_paths"] = dict(paths)
    metadata.update({key: value for key, value in paths.items() if value is not None})
    if debug_dir is not None:
        directory = Path(debug_dir) / "registration" / candidate_id
        (directory / "meta.json").write_text(json.dumps(metadata, indent=2))
        (directory / "elastix_report.json").write_text(json.dumps(metadata["elastix"], indent=2))
    if on_trace is not None:
        on_trace({"stage": "registration", "title": "Atlas boundary correction complete",
                  "metadata": metadata})
    if on_progress:
        on_progress(f"Border registration complete: {len(markers)} composed correspondences")
    session = RegistrationAnnotationSession(
        workflow="border_refinement", target_count=0, metadata=dict(metadata)
    )
    return RegistrationCandidate(
        candidate_id=candidate_id, generated_segmentation=result.raw_model_image,
        warped_atlas=warped_atlas, warped_border_overlay=result.fitted_border_overlay,
        markers=markers, annotation_session=session, metadata=metadata,
        warped_labels=result.fitted_labels, atlas_coordinate_map=final_native_map,
    )

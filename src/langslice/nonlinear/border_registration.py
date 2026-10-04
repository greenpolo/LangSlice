"""Atlas placement, boundary correction and fully composed export coordinates."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image

from langslice.affine import pixel_center_map
from langslice.atlas import load_atlas
from langslice.atlas.render import annotation_slice
from langslice.deformable.settings import Engine
from langslice.nonlinear.border_fit import fit_border_lines, placement_on_canvas
from langslice.nonlinear.border_refinement import (
    border_overlay,
    extract_thinned_lines,
    refine_borders,
)
from langslice.nonlinear.image_gen_helpers import _classified_to_rgb, line_width_px
from langslice.nonlinear.prior import place_plane_on_tissue_with_matrix, tissue_mask
from langslice.nonlinear.prompts import pass1_atlas_prompt, pass2_atlas_prompt
from langslice.nonlinear.providers import (
    SegmentationGenerationRequest,
    generate_warped_segmentation_image,
)
from langslice.nonlinear.types import (
    Deformation,
    RegistrationAnnotationSession,
    RegistrationCandidate,
)
from langslice.providers.registry import canonical_provider
from langslice.space import Plane, atlas_space_context, orient_slice_to_axes

if TYPE_CHECKING:
    from langslice.providers.registry import ImageCall


def marker_spacing_px(shape: tuple[int, ...]) -> int:
    """Spacing of the exported correspondence grid: 1/36 of the long edge, 32 px at least.

    (The control-grid spacing of the retired Elastix residual fit, kept as
    the marker density so exports keep their size.)
    """
    return max(32, round(max(shape) / 36))


def native_to_oriented_map(
    native_shape: tuple[int, int], atlas: Any, plane: Plane,
    image_axes: str | None, atlas_mirror_lr: bool,
) -> np.ndarray:
    """3x3 map of native atlas-plane pixels to the oriented (and mirrored) grid.

    The orientation is right-angle turns and flips of the label array
    (:func:`langslice.space.orient_slice_to_axes`, then the explicit
    left-right mirror), so it is read off by orienting the pixel index grids
    themselves: the deformable fit works on the native plane, the canvas
    placement on the oriented one.
    """
    height, width = native_shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    if image_axes:
        context = atlas_space_context(atlas)
        xx = orient_slice_to_axes(xx, context, plane, image_axes)
        yy = orient_slice_to_axes(yy, context, plane, image_axes)
    if atlas_mirror_lr:
        xx, yy = np.fliplr(xx), np.fliplr(yy)
    if xx.shape[0] < 2 or xx.shape[1] < 2:
        raise ValueError("The atlas plane is too small to place")
    oriented_to_native = np.array([
        [xx[0, 1] - xx[0, 0], xx[1, 0] - xx[0, 0], xx[0, 0]],
        [yy[0, 1] - yy[0, 0], yy[1, 0] - yy[0, 0], yy[0, 0]],
        [0.0, 0.0, 1.0],
    ])
    return np.linalg.inv(oriented_to_native)


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
    atlas_to_canvas: np.ndarray,
    slice_to_canvas: np.ndarray,
    native_to_canonical: np.ndarray,
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
    spacing = marker_spacing_px(field.shape)
    ys = np.unique(np.append(np.arange(0, height, spacing), height - 1))
    xs = np.unique(np.append(np.arange(0, width, spacing), width - 1))
    canvas_to_native = np.linalg.inv(_affine(atlas_to_canvas))
    canvas_to_slice = np.linalg.inv(_affine(slice_to_canvas))
    canonical = _affine(native_to_canonical)
    markers, direct = [], []
    for y in ys:
        for x in xs:
            q = np.array([float(x), float(y), 1.0])
            moved = q.copy()
            moved[:2] += field[y, x]
            native = canvas_to_native @ moved
            source = canvas_to_slice @ q
            target = canvas_to_slice @ canonical @ native
            markers.append([float(source[0]), float(source[1]), float(target[0]), float(target[1])])
            direct.append([float(source[0]), float(source[1]), float(native[0]), float(native[1])])
    if not np.isfinite(np.asarray(markers)).all():
        raise ValueError("Composed registration correspondences are not finite")
    return markers, direct


def composed_native_map(field: np.ndarray, atlas_to_canvas: np.ndarray) -> np.ndarray:
    """Compose every residual sample with the initial native atlas placement."""
    yy, xx = np.indices(field.shape[:2], dtype=np.float64)
    x, y = xx + field[..., 0], yy + field[..., 1]
    inverse = np.linalg.inv(_affine(atlas_to_canvas))
    return np.stack([
        inverse[axis, 0] * x + inverse[axis, 1] * y + inverse[axis, 2]
        for axis in range(2)
    ], axis=-1)


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
    deformation: Deformation = "deformable",
    passes: int = 1,
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
    image_call: ImageCall | None = None,
    engine: Engine = "elastix",
) -> RegistrationCandidate:
    """Route to exactly one of two border-based placements, then correct and fit.

    ``initial_atlas_to_slice`` (route "supplied") takes precedence. Without
    it, ``provider="none"`` keeps its historical model-free diagnostic
    (silhouette placement, zero residual). Otherwise (route "atlas"): the
    rough placement is ALWAYS the local silhouette-moments fit — never a
    remote call, since there is no placement to correct — and one model call
    (``passes=2`` for two) draws/corrects boundaries on the clean tissue
    against the outlined atlas template before the SAME fit as "supplied"
    runs (``generated_image`` is reassigned to the model's output and passed
    to :func:`~langslice.nonlinear.border_refinement.refine_borders` below;
    the fit is not duplicated). The fit is the deformable package's
    (:func:`~langslice.nonlinear.border_fit.fit_border_lines`, Elastix by
    default, *engine* ``"ants"`` for the named-region mode);
    ``deformation="none"`` fits nothing (identity residual), as does the
    model-free diagnostic. A
    caller-supplied ``generated_image`` with no placement replays route
    "atlas" without any model call: it is treated as that route's own final
    output. No reflection is inferred from symmetric tissue;
    ``atlas_mirror_lr`` is the only source of it.
    *image_call* is the image model's edit (default: the transport adapter
    for *provider*).
    """
    if passes not in (1, 2):
        raise ValueError("passes must be 1 or 2")
    edit = image_call or generate_warped_segmentation_image
    if on_progress:
        on_progress("Preparing rough atlas boundaries on the original section...")
    candidate_id = candidate_id or f"candidate-{uuid.uuid4().hex[:12]}"
    atlas = load_atlas(atlas_name)
    labels = annotation_slice(
        atlas, position_mm, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg
    )
    native_shape = (int(np.shape(labels)[0]), int(np.shape(labels)[1]))
    if image_axes:
        labels = orient_slice_to_axes(labels, atlas_space_context(atlas), plane, image_axes)
    if atlas_mirror_lr:
        labels = np.fliplr(labels)
    labels = np.array(labels)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Atlas annotation must be a two-dimensional integer label map")
    native_size = (labels.shape[1], labels.shape[0])
    from langslice.nonlinear.image_gen_registration import outlined_atlas_template, prepare_canvas

    canvas, unpadded, ox, oy, pad_px = prepare_canvas(
        image, canvas_pad=canvas_pad, image_model=image_model, provider=provider,
        native_canvas=native_canvas, canvas_long_edge=canvas_long_edge, quality=thinking_level,
    )
    original_to_canvas = pixel_center_map(image.size, unpadded, (ox, oy))
    canonical = canonical_atlas_map(native_size, canvas.size)
    prior: dict[str, Any]
    atlas_route_model_calls = 0
    atlas_route_artifacts: dict[str, Image.Image] = {}
    atlas_route_prompts: dict[str, str] = {}
    atlas_route_transports: list[str | None] = []
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
        # Route "atlas": no supplied placement. One model call draws
        # boundaries on the clean tissue against the outlined atlas
        # template; an optional second corrects them against the template
        # plus the first attempt's lines redrawn on the tissue. A supplied
        # `generated_image` replays this route with no model call at all
        # (it IS this route's final output).
        outlined_atlas = outlined_atlas_template(
            atlas, position_mm, plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg,
            image_axes=image_axes, atlas_mirror_lr=atlas_mirror_lr,
            section_aspect=canvas.width / canvas.height, native_labels=labels,
        )
        atlas_route_artifacts["outlined_atlas.png"] = outlined_atlas
        # A caller's image_prompt replaces the pass-1 text on this route.
        atlas_route_prompts["pass1"] = image_prompt or pass1_atlas_prompt(plane, provider)
        if generated_image is not None:
            atlas_route_output = generated_image
        else:
            if on_progress:
                on_progress("Drawing atlas boundaries on the clean tissue (pass 1)...")
            reply = edit(SegmentationGenerationRequest(
                slice_image=canvas.convert("RGB"), reference_images=[outlined_atlas],
                prompt=atlas_route_prompts["pass1"],
                provider=provider, model=image_model, review_model=review_model,
                openai_image_route=openai_image_route, thinking_level=thinking_level,
                metadata={"workflow": "border_registration", "route": "atlas", "pass": 1},
            ))
            atlas_route_output = reply.image
            atlas_route_transports.append(reply.route)
            atlas_route_model_calls = 1
            if passes == 2:
                pass1_lines = extract_thinned_lines(atlas_route_output, canvas.size)
                if not pass1_lines.any():
                    raise ValueError(
                        "Pass 1 returned no usable yellow anatomical boundaries; "
                        "pass 2 has nothing to correct"
                    )
                if on_progress:
                    on_progress("Correcting atlas boundaries (pass 2)...")
                lines_on_tissue = border_overlay(
                    canvas, pass1_lines, line_width_px(max(canvas.size))
                )
                atlas_route_artifacts["pass1_raw_correction.png"] = atlas_route_output
                atlas_route_artifacts["pass1_lines_on_tissue.png"] = lines_on_tissue
                atlas_route_prompts["pass2"] = pass2_atlas_prompt(plane, provider)
                reply = edit(SegmentationGenerationRequest(
                    slice_image=canvas.convert("RGB"),
                    reference_images=[lines_on_tissue, outlined_atlas],
                    prompt=atlas_route_prompts["pass2"],
                    provider=provider, model=image_model, review_model=review_model,
                    openai_image_route=openai_image_route, thinking_level=thinking_level,
                    metadata={"workflow": "border_registration", "route": "atlas", "pass": 2},
                ))
                atlas_route_output = reply.image
                atlas_route_transports.append(reply.route)
                atlas_route_model_calls = 2
        generated_image = atlas_route_output
        rough, signs, iou, atlas_to_canvas = place_plane_on_tissue_with_matrix(
            labels, tissue_mask(canvas, canvas.size)
        )
        initial = np.linalg.inv(original_to_canvas) @ atlas_to_canvas
        prior = {
            "source": "silhouette_moments_atlas_route", "automatic": True,
            "sign_pattern": list(signs), "silhouette_iou": float(iou),
            "passes": passes, "atlas_route_model_calls": atlas_route_model_calls,
        }
    if on_progress:
        on_progress("Correcting atlas boundaries and fitting the residual deformation...")
    result = refine_borders(
        canvas, rough, atlas, provider=provider, model=image_model, plane=plane,
        review_model=review_model, openai_image_route=openai_image_route,
        thinking_level=thinking_level, generated_image=generated_image,
        image_prompt=image_prompt, image_call=image_call,
    )
    if prior["source"] == "silhouette_moments_atlas_route":
        # refine_borders sees `generated_image` as a replay either way (it
        # never calls the model itself on this route); record what actually
        # happened above instead of leaving its generic replay bookkeeping.
        result.metadata["model_called"] = atlas_route_model_calls > 0
        result.metadata.pop("replayed", None)
        result.metadata["atlas_route_model_calls"] = atlas_route_model_calls
        result.metadata["route"] = atlas_route_transports
        # refine_borders records the route-"supplied" prompt it would have
        # sent; on this route the prompts actually sent are the atlas ones.
        result.metadata["prompt"] = atlas_route_prompts["pass1"]
        if "pass2" in atlas_route_prompts:
            result.metadata["pass2_prompt"] = atlas_route_prompts["pass2"]
    record = None
    if deformation == "none" or result.metadata.get("model_free"):
        field = np.zeros((canvas.height, canvas.width, 2), dtype=np.float64)
        fitted_labels = rough.copy()
        fitted_overlay = result.rough_border_overlay.copy()
        fit_elapsed = 0.0
        fit_report: dict[str, Any] = {"fit": "none", "fit_skipped": True}
    else:
        placement = placement_on_canvas(
            atlas, atlas_to_canvas @ native_to_oriented_map(
                native_shape, atlas, plane, image_axes, atlas_mirror_lr),
            atlas_name=atlas_name, position_mm=position_mm, plane=plane,
            pitch_deg=pitch_deg, yaw_deg=yaw_deg, source=str(prior["source"]),
        )
        fitted = fit_border_lines(canvas, result.model_border_mask, atlas, placement,
                                  engine=engine)
        record = fitted.record
        field, fitted_labels = fitted.field_px, fitted.fitted_labels
        fitted_overlay, fit_elapsed, fit_report = (
            fitted.fitted_border_overlay, fitted.elapsed, fitted.metadata)
    markers, native_points = composed_correspondences(
        field, atlas_to_canvas, original_to_canvas, canonical,
    )
    final_native_map = composed_native_map(field, atlas_to_canvas)
    warped_atlas = Image.fromarray(_classified_to_rgb(fitted_labels, atlas))
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
        "passes": passes if prior["source"] == "silhouette_moments_atlas_route" else 1,
        "atlas_to_canvas": atlas_to_canvas.tolist() if atlas_to_canvas is not None else None,
        "slice_to_canvas": original_to_canvas.tolist(),
        "native_atlas_to_canonical_canvas": canonical.tolist(),
        "visualign_markers": markers, "n_markers": len(markers),
        "marker_frames": {
            "source": "original section pixel centers (may extend outside original image)",
            "target": "canonical letterboxed atlas canvas mapped to original image units",
            "composition": (
                "inverse(slice_to_canvas) @ native_atlas_to_canonical_canvas @ "
                "inverse(atlas_to_canvas) @ (canvas_point + residual_displacement)"
            ),
        },
        "slice_to_native_atlas_correspondences": native_points,
        "native_correspondence_columns": ["slice_x", "slice_y", "native_atlas_x", "native_atlas_y"],
        "prior": prior, "inverse_warp_status": "not_computed",
        "deformation": deformation, "fit_elapsed_s": float(fit_elapsed),
        "fit": fit_report,
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
            "warped_border_overlay.png": fitted_overlay,
            **atlas_route_artifacts,
        }
        for filename, artifact in artifacts.items():
            artifact.save(directory / filename)
            key = f"{Path(filename).stem}_path"
            if key in paths:
                paths[key] = str((directory / filename).resolve())
        np.savez_compressed(directory / "rough_leaf_ids.npz", ids=rough)
        np.savez_compressed(directory / "warped_leaf_ids.npz", ids=fitted_labels)
        np.savez_compressed(directory / "residual_field.npz", field=field)
        if record is not None:
            record.save(directory / "deformable")
        np.savez_compressed(directory / "atlas_coordinate_map.npz", coordinates=final_native_map)
        Image.fromarray(result.model_border_mask.astype(np.uint8) * 255).save(
            directory / "corrected_border_mask.png"
        )
        (directory / "prompt.txt").write_text(str(result.metadata["prompt"]))
        if "pass2_prompt" in result.metadata:
            (directory / "pass2_prompt.txt").write_text(str(result.metadata["pass2_prompt"]))
    metadata["artifact_paths"] = dict(paths)
    metadata.update({key: value for key, value in paths.items() if value is not None})
    if debug_dir is not None:
        directory = Path(debug_dir) / "registration" / candidate_id
        (directory / "meta.json").write_text(json.dumps(metadata, indent=2))
        (directory / "fit_report.json").write_text(json.dumps(metadata["fit"], indent=2))
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
        warped_atlas=warped_atlas, warped_border_overlay=fitted_overlay,
        markers=markers, annotation_session=session, metadata=metadata,
        warped_labels=fitted_labels, atlas_coordinate_map=final_native_map,
    )

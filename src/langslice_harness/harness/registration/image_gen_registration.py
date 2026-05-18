"""Harness candidate pipeline for dense image-gen registration."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from langslice_harness.agent_trace import image_part_from_pil, json_part, runtime_event
from langslice_harness.atlas import get_reference_slice, load_atlas
from langslice_harness.atlas.space import Plane
from langslice_harness.harness.registration.image_gen_helpers import (
    _SEGMENTATION_PROMPT,
    _build_atlas_root_mask,
    _classify_pixels_to_region_ids,
    _extract_borders_from_classified,
    _extract_visualign_markers,
    _generate_colored_region_slice,
    _register_colored_images,
    _run_inverse_warp_for_slice,
    _upscale_to_min_long_edge,
    _warp_atlas_rgb,
)
from langslice_harness.harness.registration.providers import (
    SegmentationGenerationRequest,
    generate_warped_segmentation_image,
)
from langslice_harness.harness.registration.types import RegistrationCandidate
from langslice_harness.registration.types import RegistrationAnnotationSession

_MAX_LONG_EDGE = 2048


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


def _segmentation_prompt(prompt_revision: str | None) -> str:
    if not prompt_revision:
        return _SEGMENTATION_PROMPT
    return (
        f"{_SEGMENTATION_PROMPT}\n\n"
        "Revision guidance:\n"
        f"{prompt_revision.strip()}"
    )


def _overlay_borders(base_image: Image.Image, borders: np.ndarray) -> Image.Image:
    """Draw atlas-region borders in cyan over a base image."""
    overlay = base_image.convert("RGB").copy()
    overlay_rgb = np.asarray(overlay, dtype=np.uint8).copy()
    border_mask = np.asarray(borders) > 0
    overlay_rgb[border_mask] = (0, 255, 255)
    return Image.fromarray(overlay_rgb, mode="RGB")


def _save_debug_artifacts(
    artifact_dir: Path,
    *,
    generated_segmentation: Image.Image,
    warped_atlas: Image.Image,
    warped_border_overlay: Image.Image,
    input_colored_regions: Image.Image,
    input_reference: Image.Image,
    input_slice: Image.Image,
    generated_border_overlay: Image.Image | None = None,
    slice_warped_to_atlas: Image.Image | None = None,
    slice_atlas_border_overlay: Image.Image | None = None,
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
    _save("warped_atlas.png", warped_atlas)
    _save("warped_border_overlay.png", warped_border_overlay)
    _save("input_colored_regions.png", input_colored_regions)
    _save("input_reference.png", input_reference)
    _save("input_slice.png", input_slice)
    if generated_border_overlay is not None:
        _save("generated_border_overlay.png", generated_border_overlay)
    if slice_warped_to_atlas is not None:
        _save("slice_warped_to_atlas.png", slice_warped_to_atlas)
    if slice_atlas_border_overlay is not None:
        _save("slice_atlas_border_overlay.png", slice_atlas_border_overlay)
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
    prompt_revision: str | None = None,
    previous_candidate_id: str | None = None,
    candidate_id: str | None = None,
    debug_dir: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
    openai_image_route: str = "images",
    review_model: str | None = None,
) -> RegistrationCandidate:
    """Generate one dense registration candidate from a histology slice."""

    candidate_id = candidate_id or f"candidate-{uuid.uuid4().hex[:12]}"
    original_width, original_height = image.size
    target_size = _target_size_for_slice(image)

    if on_progress:
        on_progress("Image-gen registration: loading atlas and preparing inputs...")
    atlas = load_atlas(atlas_name)

    colored_regions = _upscale_to_min_long_edge(
        _generate_colored_region_slice(atlas, position_mm, None, plane=plane)
    )
    reference_slice = _upscale_to_min_long_edge(
        get_reference_slice(atlas, position_mm, plane=plane).convert("RGB")
    )
    slice_image = _resize_if_needed(image, target_size)

    prompt = _segmentation_prompt(prompt_revision)
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
    if prompt_revision is not None:
        request_metadata["prompt_revision"] = prompt_revision

    if on_progress:
        on_progress("Image-gen registration: generating warped atlas image...")
    generated = generate_warped_segmentation_image(
        SegmentationGenerationRequest(
            colored_regions=colored_regions,
            reference_slice=reference_slice,
            slice_image=slice_image,
            prompt=prompt,
            provider=provider,
            model=image_model,
            review_model=review_model,
            openai_image_route=openai_image_route,
            metadata=request_metadata,
        )
    )

    model_output = generated.image.convert("RGB").resize(
        target_size,
        resample=Image.Resampling.LANCZOS,
    )
    model_output_rgb = np.asarray(model_output, dtype=np.uint8)
    atlas_colored_at_target = _generate_colored_region_slice(
        atlas, position_mm, target_size, plane=plane
    )
    atlas_target_rgb = np.asarray(atlas_colored_at_target, dtype=np.uint8)

    if on_progress:
        on_progress("Image-gen registration: registering colored images...")
    result_transform, elastix_elapsed = _register_colored_images(
        atlas_target_rgb,
        model_output_rgb,
    )

    if on_progress:
        on_progress("Image-gen registration: warping atlas and extracting borders...")
    warped_atlas_rgb = _warp_atlas_rgb(atlas_target_rgb, result_transform)
    warped_atlas_img = Image.fromarray(warped_atlas_rgb, mode="RGB")
    warped_classified = _classify_pixels_to_region_ids(
        warped_atlas_rgb, atlas, position_mm, plane=plane
    )
    warped_borders = _extract_borders_from_classified(warped_classified)
    warped_border_overlay = _overlay_borders(slice_image, warped_borders)

    # Generated borders: classify the raw image-gen RGB directly (no Elastix
    # warp) so the GUI can show the model's region prediction overlaid on the
    # slice without the deformation step intermediating. Useful for inspecting
    # the model's output independent of Elastix accuracy.
    generated_classified = _classify_pixels_to_region_ids(
        model_output_rgb, atlas, position_mm, plane=plane
    )
    generated_borders = _extract_borders_from_classified(generated_classified)
    generated_border_overlay = _overlay_borders(slice_image, generated_borders)

    # Inverse warp: deform the histology slice into atlas space (slice -> atlas),
    # then overlay atlas-space region borders so callers have a complementary
    # view to the forward warped-atlas-on-slice overlay.
    if on_progress:
        on_progress("Image-gen registration: computing inverse warp (slice -> atlas)...")
    slice_rgb_array = np.asarray(slice_image.convert("RGB"), dtype=np.uint8)
    forward_fixed_gray = cv2.cvtColor(model_output_rgb, cv2.COLOR_RGB2GRAY)
    try:
        warped_slice_to_atlas_rgb, _inverse_transform = _run_inverse_warp_for_slice(
            slice_rgb_array,
            forward_fixed_gray=forward_fixed_gray,
            forward_result_transform=result_transform,
        )
        # Mask the warped slice to the atlas root silhouette so the 3D viewer
        # can render it as a brain-shaped sheet at the AP position instead of
        # a rectangular slab. NEAREST resize keeps alpha binary (0 or 255).
        root_mask = _build_atlas_root_mask(
            atlas, position_mm, target_size, plane=plane
        )
        warped_slice_to_atlas_img: Image.Image | None = Image.fromarray(
            np.dstack([warped_slice_to_atlas_rgb, root_mask]), mode="RGBA"
        )
        atlas_classified = _classify_pixels_to_region_ids(
            atlas_target_rgb, atlas, position_mm, plane=plane
        )
        atlas_borders = _extract_borders_from_classified(atlas_classified)
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

    scale_to_slice = float(original_width) / float(target_size[0])
    markers = _extract_visualign_markers(
        result_transform,
        scale_to_slice=scale_to_slice,
        image_width=original_width,
        image_height=original_height,
    )

    session_metadata: dict[str, Any] = {
        "visualign_markers": markers,
        "n_markers": len(markers),
        "elastix_elapsed_s": round(float(elastix_elapsed), 2),
        "target_size": list(target_size),
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
    }
    if previous_candidate_id is not None:
        session_metadata["previous_candidate_id"] = previous_candidate_id
    if prompt_revision is not None:
        session_metadata["prompt_revision"] = prompt_revision

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
        "original_size": [original_width, original_height],
        "inverse_warp_status": inverse_warp_status,
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
    if prompt_revision is not None:
        candidate_metadata["prompt_revision"] = prompt_revision

    saved_artifact_paths: dict[str, str] = {}
    if debug_dir is not None:
        saved_artifact_paths = _save_debug_artifacts(
            Path(debug_dir) / "registration" / candidate_id,
            generated_segmentation=generated.image,
            warped_atlas=warped_atlas_img,
            warped_border_overlay=warped_border_overlay,
            input_colored_regions=colored_regions,
            input_reference=reference_slice,
            input_slice=slice_image,
            generated_border_overlay=generated_border_overlay,
            slice_warped_to_atlas=warped_slice_to_atlas_img,
            slice_atlas_border_overlay=slice_atlas_border_overlay,
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
        generated_segmentation=generated.image,
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
        generated_segmentation=generated.image,
        warped_atlas=warped_atlas_img,
        warped_border_overlay=warped_border_overlay,
        markers=markers,
        annotation_session=session,
        metadata=candidate_metadata,
    )

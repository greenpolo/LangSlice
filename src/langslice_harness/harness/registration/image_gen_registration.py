"""Harness candidate pipeline for dense image-gen registration."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from langslice_harness.agent_trace import image_part_from_pil, json_part, runtime_event
from langslice_harness.atlas import get_reference_slice, load_atlas
from langslice_harness.harness.registration.image_gen_helpers import (
    _SEGMENTATION_PROMPT,
    _classify_pixels_to_region_ids,
    _extract_borders_from_classified,
    _extract_visualign_markers,
    _generate_colored_region_slice,
    _register_colored_images,
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
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    generated_segmentation.convert("RGB").save(artifact_dir / "generated_segmentation.png")
    warped_atlas.convert("RGB").save(artifact_dir / "warped_atlas.png")
    warped_border_overlay.convert("RGB").save(artifact_dir / "warped_border_overlay.png")
    input_colored_regions.convert("RGB").save(artifact_dir / "input_colored_regions.png")
    input_reference.convert("RGB").save(artifact_dir / "input_reference.png")
    input_slice.convert("RGB").save(artifact_dir / "input_slice.png")


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
        _generate_colored_region_slice(atlas, position_mm, None)
    )
    reference_slice = _upscale_to_min_long_edge(
        get_reference_slice(atlas, position_mm).convert("RGB")
    )
    slice_image = _resize_if_needed(image, target_size)

    prompt = _segmentation_prompt(prompt_revision)
    request_metadata: dict[str, Any] = {
        "workflow": "image_gen_registration",
        "candidate_id": candidate_id,
        "atlas_name": atlas_name,
        "position_mm": float(position_mm),
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
    atlas_colored_at_target = _generate_colored_region_slice(atlas, position_mm, target_size)
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
    warped_classified = _classify_pixels_to_region_ids(warped_atlas_rgb, atlas, position_mm)
    warped_borders = _extract_borders_from_classified(warped_classified)
    warped_border_overlay = _overlay_borders(slice_image, warped_borders)

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
        "target_size": list(target_size),
        "original_size": [original_width, original_height],
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

    if debug_dir is not None:
        _save_debug_artifacts(
            Path(debug_dir) / "registration" / candidate_id,
            generated_segmentation=generated.image,
            warped_atlas=warped_atlas_img,
            warped_border_overlay=warped_border_overlay,
            input_colored_regions=colored_regions,
            input_reference=reference_slice,
            input_slice=slice_image,
        )

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

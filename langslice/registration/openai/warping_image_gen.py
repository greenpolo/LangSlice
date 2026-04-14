"""OpenAI colored-segmentation registration workflow (two-step).

Port of the Gemini colored-segmentation workflow from
``langslice.registration.google.warping_image_gen`` adapted for
OpenAI-compatible servers that lack combined reasoning + image generation.

The workflow splits into two calls:

1. **Reason (Chat Completions)**: Send the tissue photo, atlas reference,
   and colored regions to a vision model.  The model describes how each
   region must be deformed to match the real tissue anatomy.

2. **Generate (Images API)**: Send the colored regions image and the
   deformation description to ``images.edit()``.  The image model
   produces the warped colored segmentation.

Steps 3-7 (Elastix registration, warping, border extraction, VisuAlign
marker extraction) reuse the provider-agnostic helpers from the Gemini
implementation.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from collections.abc import Callable
from typing import Any

import numpy as np
from PIL import Image

import langslice.registration.common as _agents
from langslice.agent_trace import image_part_from_pil, json_part, runtime_event
from langslice.atlas import (
    get_reference_slice,
    load_atlas,
)
from langslice.estimation.openai.common import (
    _build_image_content,
    _build_text_content,
    _extract_text,
    _image_to_base64,
)
from langslice.openai_config import (
    get_openai_client,
    get_openai_image_client,
    get_openai_image_model,
    get_openai_model,
)
from langslice.registration.google.landmarks_image_gen import (
    _species_from_atlas_name,
    _upscale_to_min_long_edge,
)
from langslice.registration.google.warping_image_gen import (
    _SEGMENTATION_PROMPT,
    _classify_pixels_to_region_ids,
    _extract_borders_from_classified,
    _extract_visualign_markers,
    _generate_colored_region_slice,
    _register_colored_images,
    _warp_atlas_rgb,
)
from langslice.registration.types import RegistrationAnnotationSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reasoning prompt for the Chat Completions step
# ---------------------------------------------------------------------------

_REASONING_PROMPT = (
    "You are an expert neuroanatomist comparing a real histology brain "
    "section to an atlas reference.\n"
    "\n"
    "Image 1: A colored brain atlas region map where each region is a "
    "unique solid color.\n"
    "Image 2: A grayscale atlas reference showing the same anatomy.\n"
    "Image 3: A real histology photograph of a mouse brain coronal section.\n"
    "\n"
    "Describe in detail how each visible colored region in Image 1 needs "
    "to be deformed to match the anatomy in Image 3. Focus on:\n"
    "- Which regions need to stretch, shrink, or shift\n"
    "- Left-right asymmetries visible in the real tissue\n"
    "- Areas of tissue damage or missing tissue\n"
    "- How the overall brain outline differs from the atlas\n"
    "\n"
    "Be specific about spatial relationships and proportions."
)


# ---------------------------------------------------------------------------
# Two-step warped segmentation request
# ---------------------------------------------------------------------------


def _image_to_png_bytes(img: Image.Image) -> bytes:
    """Convert a PIL Image to PNG bytes for the Images API."""
    buf = io.BytesIO()
    prepared = img.convert("RGB") if img.mode != "RGB" else img
    prepared.save(buf, format="PNG")
    return buf.getvalue()


def _request_warped_segmentation(
    *,
    colored_regions: Image.Image,
    reference_slice: Image.Image,
    slice_image: Image.Image,
    on_progress: Callable[[str], None] | None = None,
) -> Image.Image:
    """Two-step reasoning + image-edit request for warped segmentation.

    Step 1: Chat Completions vision call to reason about deformations.
    Step 2: Images API edit call to apply the deformations to the colored
    regions image.

    Raises ``RuntimeError`` if the image model does not return an image.
    """
    # ------------------------------------------------------------------
    # Step 1: Reason about deformations via Chat Completions
    # ------------------------------------------------------------------
    if on_progress:
        on_progress("Colored segmentation: reasoning about deformations...")

    client = get_openai_client()
    model = get_openai_model()

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                _build_text_content("Image 1 — Colored atlas region map:"),
                _build_image_content(_image_to_base64(colored_regions)),
                _build_text_content("Image 2 — Grayscale atlas reference:"),
                _build_image_content(_image_to_base64(reference_slice)),
                _build_text_content("Image 3 — Real histology section:"),
                _build_image_content(_image_to_base64(slice_image)),
                _build_text_content(_REASONING_PROMPT),
            ],
        },
    ]

    reasoning_response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=4000,
    )

    description = _extract_text(reasoning_response)
    if not description:
        raise RuntimeError(
            "Reasoning step did not return text -- "
            "cannot proceed without deformation description"
        )

    logger.info(
        "Reasoning step returned %d characters of deformation description",
        len(description),
    )
    if on_progress:
        snippet = description[:120].replace("\n", " ")
        on_progress(f"Colored segmentation: reasoning complete: {snippet}...")

    # ------------------------------------------------------------------
    # Step 2: Generate warped segmentation via Images API edit
    # ------------------------------------------------------------------
    if on_progress:
        on_progress("Colored segmentation: generating warped image via Images API...")

    image_client = get_openai_image_client()
    image_model = get_openai_image_model()

    # The images.edit() API expects a file-like object for the image param
    colored_png_bytes = _image_to_png_bytes(colored_regions)
    colored_file = io.BytesIO(colored_png_bytes)
    colored_file.name = "colored_regions.png"

    edit_prompt = (
        f"{_SEGMENTATION_PROMPT}\n\n"
        f"Specific deformation guidance:\n{description}"
    )

    edit_response = image_client.images.edit(
        image=colored_file,
        prompt=edit_prompt,
        model=image_model,
        n=1,
        response_format="b64_json",
    )

    # Extract the generated image from the response
    if not edit_response.data:
        raise RuntimeError(
            "Images API edit did not return any images -- "
            "this workflow requires the model to generate a warped region map"
        )

    b64_data = edit_response.data[0].b64_json
    if not b64_data:
        raise RuntimeError(
            "Images API edit response missing b64_json data -- "
            "check that response_format='b64_json' is supported"
        )

    image_bytes = base64.b64decode(b64_data)
    model_output = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    logger.info("Received model output image: %dx%d", model_output.width, model_output.height)
    return model_output


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def _estimate_correspondences_colored_segmentation(
    *,
    prepared: _agents._PreparedRegistrationInputs,
    atlas_name: str,
    position_mm: float,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
    on_annotation_session: Callable[[RegistrationAnnotationSession], None] | None = None,
) -> list[dict[str, object]]:
    """Run the colored-segmentation registration workflow via OpenAI-compatible APIs.

    Uses a two-step approach: Chat Completions for reasoning about
    deformations, then Images API for generating the warped segmentation.
    Steps 4-7 (Elastix registration, warping, borders, markers) are
    identical to the Gemini version.

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
    # 3. Two-step: reason about deformations, then generate warped image
    # ------------------------------------------------------------------
    if on_progress:
        on_progress("Colored segmentation: sending images to OpenAI-compatible API...")

    _agents._emit_trace(
        on_trace,
        runtime_event(
            stage="registration",
            title="Colored segmentation OpenAI request",
            summary="Two-step: Chat Completions reasoning + Images API edit",
            metadata={
                "workflow": "colored_segmentation",
                "model": get_openai_model(),
                "image_model": get_openai_image_model(),
            },
        ),
    )

    model_output = _request_warped_segmentation(
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
    # 4. Register colored images via affine + B-spline
    # ------------------------------------------------------------------
    if on_progress:
        on_progress("Colored segmentation: running Elastix affine+B-spline registration...")

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
            "model_name": get_openai_model(),
            "image_model_name": get_openai_image_model(),
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

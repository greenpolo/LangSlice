"""OpenAI-compatible two-shot landmark registration workflow.

Port of the Gemini image-gen two-shot workflow from
``langslice.registration.google.landmarks_image_gen`` adapted for
OpenAI-compatible servers that lack combined reasoning + image generation.

The workflow splits each pass into two calls:

1. **Atlas pass**:
   - Step 1 (Chat Completions): Send the atlas image and ask the model to
     describe where to place landmarks (border + interior points).
   - Step 2 (Images API ``images.edit()``): Send the atlas image + the
     landmark description. The image model draws the numbered landmarks.

2. **Slice pass**:
   - Step 1 (Chat Completions): Send the annotated atlas + histology slice.
     Ask the model to describe where corresponding landmarks should go.
   - Step 2 (Images API ``images.edit()``): Send the slice image + the
     landmark description. The image model draws matching numbered landmarks.

Generated images are saved to the registration debug directory for visual
inspection.  Marker extraction from the generated images is not yet
implemented — this function currently returns an empty correspondence list.
"""

from __future__ import annotations

import io
import logging
import os
from collections.abc import Callable
from typing import Any

from PIL import Image

import langslice.registration.common as _agents
from langslice.agent_trace import image_part_from_pil, runtime_event
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
    _boost_exposure,
    _save_image_gen_artifacts,
    _species_from_atlas_name,
    _upscale_to_min_long_edge,
)
from langslice.registration.types import RegistrationAnnotationSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _image_to_png_bytes(img: Image.Image) -> bytes:
    """Convert a PIL Image to PNG bytes for the Images API."""
    buf = io.BytesIO()
    prepared = img.convert("RGB") if img.mode != "RGB" else img
    prepared.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Atlas annotation pass (two-step)
# ---------------------------------------------------------------------------


def _request_annotated_atlas(
    *,
    atlas_image: Image.Image,
    species: str,
    target_count: int,
    on_progress: Callable[[str], None] | None = None,
) -> Image.Image:
    """Two-step atlas annotation: Chat Completions reasoning + Images API edit.

    Step 1: Ask the chat model where to place landmarks on the atlas.
    Step 2: Ask the image model to draw the described landmarks on the atlas.

    Raises ``RuntimeError`` if the image model does not return an image.
    """
    atlas_upscaled = _upscale_to_min_long_edge(atlas_image)

    # ------------------------------------------------------------------
    # Step 1: Reason about landmark placement via Chat Completions
    # ------------------------------------------------------------------
    if on_progress:
        on_progress("Landmarks (atlas): reasoning about landmark positions...")

    client = get_openai_client()
    model = get_openai_model()

    border_count = max(1, target_count // 2)
    interior_count = max(1, target_count - border_count)

    reasoning_prompt = (
        f"This is a coronal section of a {species} brain anatomical atlas. "
        f"I need to place {border_count} landmark points on the outline/border "
        f"of the brain slice, and {interior_count} landmark points in the interior. "
        "Describe in detail exactly where each numbered landmark should be placed. "
        "For each point, specify:\n"
        "- The landmark number (1 through the total count)\n"
        "- Whether it is a border or interior point\n"
        "- A precise anatomical location description (e.g., 'dorsal midline near "
        "corpus callosum', 'left lateral cortex boundary at mid-height')\n\n"
        "Prioritize landmarks that are:\n"
        "- Evenly distributed across the entire atlas section\n"
        "- Visually distinct and anatomically identifiable\n"
        "- Easily matched in a real histology slide\n\n"
        "List all landmarks clearly and concisely."
    )

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                _build_image_content(_image_to_base64(atlas_upscaled)),
                _build_text_content(reasoning_prompt),
            ],
        },
    ]

    reasoning_response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=2000,
    )

    description = _extract_text(reasoning_response)
    if not description:
        raise RuntimeError(
            "Atlas landmark reasoning step did not return text -- "
            "cannot proceed without landmark position description"
        )

    logger.info(
        "Atlas reasoning step returned %d characters of landmark description",
        len(description),
    )
    if on_progress:
        snippet = description[:120].replace("\n", " ")
        on_progress(f"Landmarks (atlas): reasoning complete: {snippet}...")

    # ------------------------------------------------------------------
    # Step 2: Draw landmarks on atlas via Images API edit
    # ------------------------------------------------------------------
    if on_progress:
        on_progress("Landmarks (atlas): generating annotated atlas via Images API...")

    image_client = get_openai_image_client()
    image_model = get_openai_image_model()

    atlas_png_bytes = _image_to_png_bytes(atlas_upscaled)
    atlas_file = io.BytesIO(atlas_png_bytes)
    atlas_file.name = "atlas.png"

    draw_prompt = (
        f"Draw small numbered landmark annotation points on this {species} brain atlas image. "
        "Place the points exactly as described below. "
        "Label each point with its number. "
        "Make the points small so they do not obscure local features. "
        "Make no other edits to the image.\n\n"
        f"Landmark placement instructions:\n{description}"
    )

    edit_response = image_client.images.edit(
        image=atlas_file,
        prompt=draw_prompt,
        model=image_model,
        n=1,
        response_format="b64_json",
    )

    if not edit_response.data:
        raise RuntimeError(
            "Images API atlas annotation did not return any images -- "
            "this workflow requires the model to generate an annotated atlas"
        )

    import base64

    b64_data = edit_response.data[0].b64_json
    if not b64_data:
        raise RuntimeError(
            "Images API atlas annotation response missing b64_json data -- "
            "check that response_format='b64_json' is supported"
        )

    image_bytes = base64.b64decode(b64_data)
    annotated_atlas = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    logger.info(
        "Atlas annotation step produced %dx%d image",
        annotated_atlas.width, annotated_atlas.height,
    )
    return annotated_atlas


# ---------------------------------------------------------------------------
# Slice annotation pass (two-step)
# ---------------------------------------------------------------------------


def _request_annotated_slice(
    *,
    annotated_atlas: Image.Image,
    slice_image: Image.Image,
    species: str,
    on_progress: Callable[[str], None] | None = None,
) -> Image.Image | None:
    """Two-step slice annotation: Chat Completions reasoning + Images API edit.

    Step 1: Ask the chat model where corresponding landmarks should go on the
    slice by comparing the annotated atlas to the histology image.
    Step 2: Ask the image model to draw the described landmarks on the slice.

    Returns ``None`` if the image model does not return an image.
    """
    slice_boosted = _boost_exposure(slice_image)

    # ------------------------------------------------------------------
    # Step 1: Reason about corresponding landmark positions via Chat
    # ------------------------------------------------------------------
    if on_progress:
        on_progress("Landmarks (slice): reasoning about corresponding positions...")

    client = get_openai_client()
    model = get_openai_model()

    reasoning_prompt = (
        f"Image 1 is a {species} brain atlas annotated with numbered landmark points. "
        "Image 2 is a real histology photograph of a corresponding brain section.\n\n"
        "For each numbered landmark in Image 1, describe precisely where the "
        "corresponding landmark should be placed on Image 2 (the histology slice). "
        "Focus on matching the same anatomical location in the real tissue.\n\n"
        "Important: the histology slide may contain common microscopy artifacts "
        "such as air bubbles, tears, or tissue damage. Do not mistake these "
        "for anatomical structures. Place landmarks only on real brain anatomy.\n\n"
        "For each landmark number, give a clear spatial description of where to "
        "place the corresponding point on the histology image."
    )

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                _build_text_content("Image 1 — Annotated atlas:"),
                _build_image_content(_image_to_base64(annotated_atlas)),
                _build_text_content("Image 2 — Histology slice:"),
                _build_image_content(_image_to_base64(slice_boosted)),
                _build_text_content(reasoning_prompt),
            ],
        },
    ]

    reasoning_response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=2000,
    )

    description = _extract_text(reasoning_response)
    if not description:
        logger.warning(
            "Slice landmark reasoning step did not return text -- "
            "skipping image generation for slice pass"
        )
        return None

    logger.info(
        "Slice reasoning step returned %d characters of landmark description",
        len(description),
    )
    if on_progress:
        snippet = description[:120].replace("\n", " ")
        on_progress(f"Landmarks (slice): reasoning complete: {snippet}...")

    # ------------------------------------------------------------------
    # Step 2: Draw landmarks on slice via Images API edit
    # ------------------------------------------------------------------
    if on_progress:
        on_progress("Landmarks (slice): generating annotated slice via Images API...")

    image_client = get_openai_image_client()
    image_model = get_openai_image_model()

    slice_png_bytes = _image_to_png_bytes(slice_boosted)
    slice_file = io.BytesIO(slice_png_bytes)
    slice_file.name = "slice.png"

    draw_prompt = (
        f"Draw small numbered landmark annotation points on this {species} brain "
        "histology image. "
        "Place the points exactly as described below, matching the same anatomical "
        "locations as in the corresponding atlas. "
        "Preserve the landmark numbers so they match the atlas annotations. "
        "Make the points small so they do not obscure local features. "
        "Make no other edits to the image.\n\n"
        f"Landmark placement instructions:\n{description}"
    )

    edit_response = image_client.images.edit(
        image=slice_file,
        prompt=draw_prompt,
        model=image_model,
        n=1,
        response_format="b64_json",
    )

    if not edit_response.data:
        logger.warning("Images API slice annotation did not return any images")
        return None

    import base64

    b64_data = edit_response.data[0].b64_json
    if not b64_data:
        logger.warning("Images API slice annotation response missing b64_json data")
        return None

    image_bytes = base64.b64decode(b64_data)
    annotated_slice = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    logger.info(
        "Slice annotation step produced %dx%d image",
        annotated_slice.width, annotated_slice.height,
    )
    return annotated_slice


# ---------------------------------------------------------------------------
# Main two-shot orchestration
# ---------------------------------------------------------------------------


def _estimate_correspondences_image_gen_two_shot(
    *,
    prepared: _agents._PreparedRegistrationInputs,
    atlas_name: str,
    position_mm: float,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
    on_annotation_session: Callable[[RegistrationAnnotationSession], None] | None = None,
) -> list[dict[str, object]]:
    """Run the two-shot image-gen landmark workflow via OpenAI-compatible APIs.

    Each pass uses a two-step approach: Chat Completions for reasoning about
    where landmarks should go, then Images API to draw them on the image.
    Generated images are saved to the registration debug directory for visual
    inspection.

    Marker extraction from the generated images is not yet implemented --
    this function currently returns an empty correspondence list.
    """
    species = _species_from_atlas_name(atlas_name)
    atlas_image = prepared.atlas_prep.image
    slice_image = prepared.slice_prep.image

    # Save the exact images the model will see for pass 1.
    atlas_for_model = _upscale_to_min_long_edge(atlas_image)
    if prepared.registration_dir:
        os.makedirs(prepared.registration_dir, exist_ok=True)
        atlas_for_model.convert("RGB").save(
            os.path.join(prepared.registration_dir, "pass1_atlas_sent.png"), format="PNG"
        )
        _boost_exposure(slice_image).save(
            os.path.join(prepared.registration_dir, "pass2_slice_sent.png"), format="PNG"
        )

    # ------------------------------------------------------------------
    # Atlas pass — two-step: reason then draw landmarks on atlas
    # ------------------------------------------------------------------
    if on_progress:
        on_progress("Registration: atlas landmark annotation pass (OpenAI image-gen)...")

    _agents._emit_trace(
        on_trace,
        runtime_event(
            stage="registration",
            title="OpenAI image-gen atlas request",
            summary="Atlas image sent for two-step landmark annotation (Chat + Images API)",
            parts=[
                image_part_from_pil(atlas_image, label="Atlas image sent"),
            ],
            metadata={
                "workflow": "image_gen_two_shot",
                "provider": "openai",
                "pass": 1,
                "model": get_openai_model(),
                "image_model": get_openai_image_model(),
            },
        ),
    )

    annotated_atlas = _request_annotated_atlas(
        atlas_image=atlas_image,
        species=species,
        target_count=prepared.target_count,
        on_progress=on_progress,
    )

    logger.info(
        "Atlas pass complete: annotated atlas %dx%d",
        annotated_atlas.width, annotated_atlas.height,
    )

    _agents._emit_trace(
        on_trace,
        runtime_event(
            stage="registration",
            title="OpenAI image-gen atlas result",
            summary="Atlas pass produced annotated atlas image",
            parts=[
                image_part_from_pil(annotated_atlas, label="Annotated atlas (model-generated)"),
            ],
            metadata={
                "workflow": "image_gen_two_shot",
                "provider": "openai",
                "pass": 1,
            },
        ),
    )

    # ------------------------------------------------------------------
    # Slice pass — two-step: reason then draw corresponding landmarks
    # ------------------------------------------------------------------
    if on_progress:
        on_progress("Registration: slice landmark transfer pass (OpenAI image-gen)...")

    _agents._emit_trace(
        on_trace,
        runtime_event(
            stage="registration",
            title="OpenAI image-gen slice request",
            summary="Annotated atlas and histology slice sent for landmark transfer",
            parts=[
                image_part_from_pil(annotated_atlas, label="Annotated atlas sent"),
                image_part_from_pil(slice_image, label="Histology slice sent"),
            ],
            metadata={
                "workflow": "image_gen_two_shot",
                "provider": "openai",
                "pass": 2,
            },
        ),
    )

    annotated_slice = _request_annotated_slice(
        annotated_atlas=annotated_atlas,
        slice_image=slice_image,
        species=species,
        on_progress=on_progress,
    )

    logger.info(
        "Slice pass complete: annotated_slice=%s",
        f"{annotated_slice.width}x{annotated_slice.height}" if annotated_slice else "None",
    )

    if annotated_slice is not None:
        _agents._emit_trace(
            on_trace,
            runtime_event(
                stage="registration",
                title="OpenAI image-gen slice result",
                summary="Slice pass produced annotated slice image",
                parts=[
                    image_part_from_pil(
                        annotated_slice, label="Annotated slice (model-generated)"
                    ),
                ],
                metadata={
                    "workflow": "image_gen_two_shot",
                    "provider": "openai",
                    "pass": 2,
                },
            ),
        )

    # Save all model-generated images for visual inspection.
    _save_image_gen_artifacts(
        prepared.registration_dir,
        annotated_atlas=annotated_atlas,
        annotated_slice=annotated_slice,
    )

    if annotated_slice is None:
        raise RuntimeError(
            "OpenAI image-gen slice pass did not return an image -- "
            "this workflow requires the model to generate annotated images"
        )

    # TODO: extract marker positions from model-generated images and
    # return paired correspondences.  For now, return an empty list so
    # callers can inspect the saved images and iterate on model quality.
    return []

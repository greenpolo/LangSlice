"""Image-generation two-shot registration workflow.

This workflow is exclusively for Gemini image-generation models:

- ``gemini-3-pro-image-preview``
- ``gemini-3.1-flash-image-preview``

These models cannot use tools, structured JSON output, or system
instructions.  Instead, we use a two-pass strategy:

1. **Atlas pass** — the model draws numbered landmark annotations directly
   on the atlas image.
2. **Slice pass** — a second call receives the model-annotated atlas
   alongside the raw histology slice.  The model draws matching numbered
   landmarks on the slice.

Calls use the streaming API (``generate_content_stream``) with typed
``google.genai.types`` objects and ``response_modalities=["IMAGE", "TEXT"]``
so the model can emit reasoning text alongside generated images — matching
the behaviour observed in Google AI Studio.
"""

from __future__ import annotations

import importlib
import io
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PIL import Image

import langslice.registration.agents as _agents
from langslice.agent_trace import image_part_from_pil, runtime_event
from langslice.registration.types import RegistrationAnnotationSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _species_from_atlas_name(atlas_name: str) -> str:
    """Extract a readable species name from a BrainGlobe atlas identifier."""
    lower = atlas_name.lower()
    for species in ("mouse", "rat", "human", "zebrafish", "fish"):
        if species in lower:
            return species
    return "animal"


@dataclass(frozen=True)
class _GeneratedImagePayload:
    image: Image.Image
    data: bytes
    mime_type: str


def _upscale_to_min_long_edge(img: Image.Image, min_long_edge: int = 1024) -> Image.Image:
    """Upscale *img* so its long edge is at least *min_long_edge* pixels."""
    w, h = img.size
    long_edge = max(w, h)
    if long_edge >= min_long_edge:
        return img
    scale = min_long_edge / long_edge
    new_w = round(w * scale)
    new_h = round(h * scale)
    return img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)


def _boost_exposure(img: Image.Image, factor: float = 1.5) -> Image.Image:
    """Brighten *img* by *factor* to improve visibility for the model."""
    from PIL import ImageEnhance

    return ImageEnhance.Brightness(img.convert("RGB")).enhance(factor)


def _image_to_typed_part(img: Image.Image) -> Any:
    """Convert a PIL Image to a typed ``google.genai.types.Part``."""
    buf = io.BytesIO()
    prepared = img.convert("RGB") if img.mode != "RGB" else img
    prepared.save(buf, format="PNG")
    types_mod = importlib.import_module("google.genai.types")
    part_cls = types_mod.Part
    return part_cls.from_bytes(mime_type="image/png", data=buf.getvalue())


def _generated_payload_to_typed_part(payload: _GeneratedImagePayload) -> Any:
    types_mod = importlib.import_module("google.genai.types")
    return types_mod.Part.from_bytes(mime_type=payload.mime_type, data=payload.data)


def _build_image_gen_config(
    *,
    model_name: str,
    thinking_level: str = "HIGH",
) -> Any:
    """Build a typed ``GenerateContentConfig`` for image-gen models.

    Requests ``response_modalities=["IMAGE", "TEXT"]`` so the model can emit
    reasoning text alongside generated images (matching AI Studio behaviour),
    and enables image-model thinking when supported.

    Config mirrors AI Studio's exported code: ``image_size="1K"``.
    """
    types_mod = importlib.import_module("google.genai.types")
    generate_content_config_cls = types_mod.GenerateContentConfig
    image_config_cls = types_mod.ImageConfig

    kwargs: dict[str, Any] = {
        "response_modalities": ["IMAGE", "TEXT"],
        "image_config": image_config_cls(image_size="1K"),
    }

    vlm_config = importlib.import_module("langslice.vlm.config")
    supports_image_model_thinking = getattr(
        vlm_config,
        "supports_image_model_thinking",
        lambda _model_name: False,
    )
    if bool(supports_image_model_thinking(model_name)):
        thinking_config_cls = types_mod.ThinkingConfig
        level = str(thinking_level).strip().upper() or "HIGH"
        kwargs["thinking_config"] = thinking_config_cls(thinking_level=level)

    return generate_content_config_cls(**kwargs)


def _extract_generated_images(response: object) -> list[_GeneratedImagePayload]:
    """Collect generated image parts from a Gemini response.

    Preserves the original bytes and MIME type so later requests can reuse the
    exact returned image payload instead of re-encoding through PIL.
    """
    candidates = getattr(response, "candidates", None)
    if not isinstance(candidates, list) or not candidates:
        return []
    content = getattr(candidates[0], "content", None)
    images: list[_GeneratedImagePayload] = []
    for part in getattr(content, "parts", None) or []:
        inline = getattr(part, "inline_data", None)
        if inline is None:
            continue
        data = getattr(inline, "data", None)
        mime = getattr(inline, "mime_type", "")
        if isinstance(data, (bytes, bytearray)) and str(mime).startswith("image/"):
            images.append(
                _GeneratedImagePayload(
                    image=Image.open(io.BytesIO(data)).convert("RGB"),
                    data=bytes(data),
                    mime_type=str(mime),
                )
            )
    return images


def _extract_generated_image(response: object) -> Image.Image | None:
    """Pull the last generated image from a Gemini response.

    AI Studio exports for image-editing flows can include multiple generated
    images interleaved with text. The final image part has been the closest
    match to the visible result the user sees in AI Studio, so we keep the
    last image encountered.
    """
    images = _extract_generated_images(response)
    return images[-1].image if images else None


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------


def _save_image_gen_artifacts(
    registration_dir: str | None,
    *,
    annotated_atlas: Image.Image,
    annotated_slice: Image.Image | None = None,
    atlas_candidates: list[Image.Image] | None = None,
    slice_candidates: list[Image.Image] | None = None,
) -> None:
    """Save model-generated images to the debug directory for visual inspection."""
    if not registration_dir:
        return
    os.makedirs(registration_dir, exist_ok=True)
    annotated_atlas.save(os.path.join(registration_dir, "model_atlas_annotated.png"))
    if annotated_slice is not None:
        annotated_slice.save(os.path.join(registration_dir, "model_slice_annotated.png"))
    for idx, candidate in enumerate(atlas_candidates or [], start=1):
        candidate.save(os.path.join(registration_dir, f"model_atlas_candidate_{idx:02d}.png"))
    for idx, candidate in enumerate(slice_candidates or [], start=1):
        candidate.save(os.path.join(registration_dir, f"model_slice_candidate_{idx:02d}.png"))


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------


def _build_image_gen_atlas_request(
    *,
    atlas_prep: Any,
    species: str,
    target_count: int,
) -> list[Any]:
    """Build typed ``Content`` list for the atlas annotation pass.

    Image part comes **before** text, matching the AI Studio pattern.
    """
    types_mod = importlib.import_module("google.genai.types")
    content_cls = types_mod.Content
    part_cls = types_mod.Part

    border_count = max(1, target_count // 2)
    interior_count = max(1, target_count - border_count)
    prompt = (
        f"This is a coronal section of a {species} brain anatomical atlas. "
        "I'd like you to place landmark points as annotations on the atlas. "
        "Prioritize placing landmarks which are evenly distributed, visually "
        "distinct, and would be easily identifiable in a real brain section "
        "or easily identifiable if described in text. "
        f"Please place {border_count} points on the outline/border of the "
        f"brain slice atlas, and {interior_count} points in the interior of "
        "the brain slice atlas. Label each annotation with a number.\n\n"
        "Draw the point annotations directly on this image. Make no other "
        "edits to the image. Please ensure the points are small, such that "
        "they do not block local features."
    )
    atlas_image = _upscale_to_min_long_edge(atlas_prep.image)
    return [
        content_cls(
            role="user",
            parts=[
                _image_to_typed_part(atlas_image),
                part_cls.from_text(text=prompt),
            ],
        )
    ]


def _build_image_gen_slice_request(
    *,
    annotated_atlas: Image.Image | _GeneratedImagePayload,
    slice_prep: Any,
    species: str,
) -> list[Any]:
    """Build typed ``Content`` list for the slice annotation pass.

    AI Studio's exported second-pass example orders parts as slice image,
    annotated atlas, then prompt text.
    """
    types_mod = importlib.import_module("google.genai.types")
    content_cls = types_mod.Content
    part_cls = types_mod.Part

    prompt = (
        f"This is a {species} brain slice and a corresponding anatomical atlas. "
        "The anatomical atlas is annotated with landmark points. "
        "I'd like you to place corresponding point annotations on the "
        "histological slice in the same relative anatomical positions as "
        "the atlas points. Ensure the numbers of the annotations are preserved.\n\n"
        "Important: this is a real histology slide and may contain common "
        "microscopy artifacts such as air bubbles, tears, or tissue damage. "
        "Do not mistake these artifacts for anatomical structures. Place "
        "landmarks only on real brain anatomy.\n\n"
        "Output the histology slice as an image edited with the point "
        "annotations. Please ensure the points are small, such that they "
        "do not block local features."
    )
    slice_image = _boost_exposure(slice_prep.image)
    atlas_part = (
        _generated_payload_to_typed_part(annotated_atlas)
        if isinstance(annotated_atlas, _GeneratedImagePayload)
        else _image_to_typed_part(annotated_atlas)
    )
    return [
        content_cls(
            role="user",
            parts=[
                _image_to_typed_part(slice_image),
                atlas_part,
                part_cls.from_text(text=prompt),
            ],
        )
    ]


# ---------------------------------------------------------------------------
# Main two-shot orchestration
# ---------------------------------------------------------------------------


def _estimate_correspondences_image_gen_two_shot(
    client: Any,
    *,
    prepared: _agents._PreparedRegistrationInputs,
    atlas_name: str,
    position_mm: float,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
    on_annotation_session: Callable[[RegistrationAnnotationSession], None] | None = None,
) -> list[dict[str, object]]:
    """Run the two-shot image-gen landmark workflow.

    The model draws landmark annotations directly on the images.  Generated
    images are saved to the registration debug directory for visual inspection.

    Marker extraction from the generated images is not yet implemented —
    this function currently returns an empty correspondence list.
    """
    species = _species_from_atlas_name(atlas_name)
    atlas_image = prepared.atlas_prep.image
    slice_image = prepared.slice_prep.image

    # Save the exact images Gemini will see for pass 1 (atlas upscaled to 1K).
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
    # Atlas pass — model annotates the atlas image
    # ------------------------------------------------------------------
    if on_progress:
        on_progress("Registration: atlas landmark annotation pass (image-gen)...")
    _agents._emit_trace(
        on_trace,
        runtime_event(
            stage="registration",
            title="Image-gen atlas request",
            summary="Atlas image sent to Gemini for landmark annotation",
            parts=[
                image_part_from_pil(atlas_image, label="Atlas image sent to Gemini"),
            ],
            metadata={"workflow": "image_gen_two_shot", "pass": 1},
        ),
    )

    atlas_response = _agents._retry_generate_stream(
        client,
        model=prepared.model_name,
        contents=_build_image_gen_atlas_request(
            atlas_prep=prepared.atlas_prep,
            species=species,
            target_count=prepared.target_count,
        ),
        config=_build_image_gen_config(
            model_name=prepared.model_name,
            thinking_level=prepared.thinking_level,
        ),
        request_label="Registration image-gen atlas pass",
        on_progress=on_progress,
    )

    atlas_payloads = _extract_generated_images(atlas_response)
    if not atlas_payloads:
        raise RuntimeError(
            "Image-gen atlas pass did not return an image — "
            "this workflow requires the model to generate annotated images"
        )
    annotated_atlas_payload = atlas_payloads[-1]
    annotated_atlas = annotated_atlas_payload.image
    logger.info("Atlas pass: received %d generated image(s)", len(atlas_payloads))

    _agents._emit_trace(
        on_trace,
        runtime_event(
            stage="registration",
            title="Image-gen atlas result",
            summary=f"Atlas pass returned {len(atlas_payloads)} image(s)",
            parts=[
                image_part_from_pil(annotated_atlas, label="Annotated atlas (model-generated)"),
            ],
            metadata={
                "workflow": "image_gen_two_shot",
                "pass": 1,
                "image_count": len(atlas_payloads),
            },
        ),
    )

    # ------------------------------------------------------------------
    # Slice pass — model annotates the histology slice
    # ------------------------------------------------------------------
    if on_progress:
        on_progress("Registration: slice landmark transfer pass (image-gen)...")
    _agents._emit_trace(
        on_trace,
        runtime_event(
            stage="registration",
            title="Image-gen slice request",
            summary="Model-annotated atlas and slice sent to Gemini for transfer",
            parts=[
                image_part_from_pil(annotated_atlas, label="Annotated atlas sent"),
                image_part_from_pil(slice_image, label="Histology slice sent"),
            ],
            metadata={"workflow": "image_gen_two_shot", "pass": 2},
        ),
    )

    slice_response = _agents._retry_generate_stream(
        client,
        model=prepared.model_name,
        contents=_build_image_gen_slice_request(
            annotated_atlas=annotated_atlas_payload,
            slice_prep=prepared.slice_prep,
            species=species,
        ),
        config=_build_image_gen_config(
            model_name=prepared.model_name,
            thinking_level=prepared.thinking_level,
        ),
        request_label="Registration image-gen slice pass",
        on_progress=on_progress,
    )

    slice_payloads = _extract_generated_images(slice_response)
    annotated_slice = slice_payloads[-1].image if slice_payloads else None
    logger.info("Slice pass: received %d generated image(s)", len(slice_payloads))

    if annotated_slice is not None:
        _agents._emit_trace(
            on_trace,
            runtime_event(
                stage="registration",
                title="Image-gen slice result",
                summary=f"Slice pass returned {len(slice_payloads)} image(s)",
                parts=[
                    image_part_from_pil(annotated_slice, label="Annotated slice (model-generated)"),
                ],
                metadata={
                    "workflow": "image_gen_two_shot",
                    "pass": 2,
                    "image_count": len(slice_payloads),
                },
            ),
        )

    # Save all model-generated images for visual inspection.
    _save_image_gen_artifacts(
        prepared.registration_dir,
        annotated_atlas=annotated_atlas,
        annotated_slice=annotated_slice,
        atlas_candidates=[p.image for p in atlas_payloads],
        slice_candidates=[p.image for p in slice_payloads],
    )

    if annotated_slice is None:
        raise RuntimeError(
            "Image-gen slice pass did not return an image — "
            "this workflow requires the model to generate annotated images"
        )

    # TODO: extract marker positions from model-generated images and
    # return paired correspondences.  For now, return an empty list so
    # callers can inspect the saved images and iterate on model quality.
    return []

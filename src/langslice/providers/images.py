"""The image-edit transports: one request in, one edited image out.

``generate_warped_segmentation_image`` sends a
:class:`~langslice.core.nonlinear.types.SegmentationGenerationRequest` to its
provider: ``gemini-api`` (``generate_content`` with an image config),
``openai-api`` (the Images API edit endpoint) or ``openai-oauth`` (the Codex
``images/edits`` endpoint, :func:`langslice.providers.openai_oauth.edit_image`).
The slice image is always Image 1; the references follow in prompt order.
"""

from __future__ import annotations

import base64
import importlib
import io
from typing import Any, cast

from PIL import Image

from langslice.core.nonlinear.model_prompts import gemini_aspect_for
from langslice.core.nonlinear.types import GeneratedSegmentation, SegmentationGenerationRequest
from langslice.providers import vlm_config
from langslice.providers.openai_config import get_openai_image_client, get_openai_image_model
from langslice.providers.registry import GEMINI_DEFAULT_IMAGE_MODEL, canonical_provider

_IMAGE_QUALITIES = {"low", "medium", "high", "xhigh", "max"}


def _image_to_png_file(image: Image.Image, name: str) -> io.BytesIO:
    buffer = io.BytesIO()
    prepared = image.convert("RGB") if image.mode != "RGB" else image
    prepared.save(buffer, format="PNG")
    buffer.seek(0)
    buffer.name = name
    return buffer


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = _image_to_png_file(image, "image.png")
    return buffer.getvalue()


def _image_to_data_url(image: Image.Image) -> str:
    encoded = base64.b64encode(_image_to_png_bytes(image)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _response_parts(response: Any) -> list[Any]:
    parts = getattr(response, "parts", None)
    if parts:
        return list(parts)

    candidates = getattr(response, "candidates", None) or []
    if candidates:
        content = getattr(candidates[0], "content", None)
        candidate_parts = getattr(content, "parts", None) if content is not None else None
        if candidate_parts:
            return list(candidate_parts)

    return []


def _extract_last_inline_image(response: Any) -> Image.Image:
    parts = _response_parts(response)
    for part in reversed(parts):
        inline_data = getattr(part, "inline_data", None)
        image_bytes = getattr(inline_data, "data", None) if inline_data is not None else None
        if image_bytes:
            image = cast(Image.Image, Image.open(io.BytesIO(image_bytes)))
            image.load()
            return image.convert("RGB")

        as_image = getattr(part, "as_image", None)
        if callable(as_image):
            image = cast(Image.Image, as_image())
            if image is not None:
                if hasattr(image, "load"):
                    image.load()
                return image.convert("RGB")

    raise RuntimeError("Gemini image generation did not return an inline image")


def _extract_openai_image_b64(response: Any) -> str:
    data = getattr(response, "data", None) or []
    if not data:
        raise RuntimeError("OpenAI Images API did not return any image data")

    image_b64 = getattr(data[0], "b64_json", None)
    if not image_b64:
        raise RuntimeError("OpenAI Images API response missing b64_json")
    return image_b64


def _build_metadata(
    request: SegmentationGenerationRequest,
    *,
    provider: str,
    route: str,
) -> dict[str, Any]:
    metadata = dict(request.metadata)
    metadata.update({
        "provider": provider,
        "route": route,
        "request": {
            "provider": request.provider.lower(),
            "model": request.model,
            "thinking_level": request.thinking_level,
            "prompt": request.prompt,
        },
    })
    return metadata


#: ``thinking_level`` values that name a Gemini output resolution (the API's
#: own spellings). Flash-Lite Image serves 1K only; 512 is Flash Image, 2K/4K
#: are Flash Image / Pro Image tiers.
_GEMINI_IMAGE_SIZES = frozenset({"512", "512P", "512PX", "1K", "2K", "4K"})


def _gemini_image_config(
    request: SegmentationGenerationRequest,
    response_modalities: tuple[str, ...] = ("IMAGE",),
) -> Any:
    """The image config Gemini needs to return an aligned edit.

    Without it the model picks its own size and aspect (1K, whatever it
    likes), which the pipeline then stretches back onto the canvas. Pin the
    aspect to the nearest ratio the API accepts, and the resolution to the
    caller's tier when one is given. ``response_modalities`` defaults to IMAGE
    only: adding TEXT silently caps output at 1K on these models, and a
    single-image request has nothing to gain from it.
    """
    types_mod = importlib.import_module("google.genai.types")
    w, h = request.slice_image.size
    # The same table the canvas was framed with (native_output_size), so the
    # aspect asked for is the aspect the canvas already has.
    aspect = gemini_aspect_for(w / h, request.thinking_level)
    size = (request.thinking_level or "").upper()
    image_config = types_mod.ImageConfig(
        aspect_ratio=aspect,
        image_size=size if size in _GEMINI_IMAGE_SIZES else None,
    )
    return types_mod.GenerateContentConfig(
        response_modalities=list(response_modalities),
        image_config=image_config,
    )


def _generate_google_segmentation(request: SegmentationGenerationRequest) -> GeneratedSegmentation:
    model = request.model or GEMINI_DEFAULT_IMAGE_MODEL
    client = vlm_config.get_client()

    # Histology first: the edited base image leads, references follow. There
    # is no edit-vs-generate switch on this API: the prompt alone carries the
    # editing intent, and the config pins the output frame to the canvas.
    contents = [
        request.slice_image,
        *request.reference_images,
        request.prompt,
    ]
    config = _gemini_image_config(request)
    response = client.models.generate_content(  # type: ignore[attr-defined]
        model=model, contents=contents, config=config
    )
    try:
        image = _extract_last_inline_image(response)
    except Exception:  # noqa: BLE001 — an imageless reply (e.g. IMAGE_RECITATION); once more
        response = client.models.generate_content(  # type: ignore[attr-defined]
            model=model, contents=contents, config=config
        )
        image = _extract_last_inline_image(response)
    route = "google_genai"
    return GeneratedSegmentation(
        image=image,
        provider="gemini-api",
        model=model,
        route=route,
        metadata=_build_metadata(request, provider="gemini-api", route=route),
    )


def _generate_openai_images_segmentation(
    request: SegmentationGenerationRequest,
    *,
    provider: str,
) -> GeneratedSegmentation:
    model = request.model or get_openai_image_model()
    client = get_openai_image_client()
    # Histology first: the edited base image leads, references follow.
    image_files = [
        _image_to_png_file(request.slice_image, "slice_image.png"),
        *(
            _image_to_png_file(ref, f"atlas_reference_{i + 1}.png")
            for i, ref in enumerate(request.reference_images)
        ),
    ]

    quality = (request.thinking_level or "auto").lower()
    response = client.images.edit(  # type: ignore[attr-defined]
        model=model,
        image=image_files,
        prompt=request.prompt,
        quality=cast(Any, quality if quality in _IMAGE_QUALITIES else "auto"),
        size=_api_edit_size(request.slice_image),
    )

    image_b64 = _extract_openai_image_b64(response)
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes))
    image.load()
    route = "openai_images"
    metadata = _build_metadata(request, provider=provider, route=route)
    usage = getattr(response, "usage", None)
    if usage is not None:
        metadata["usage"] = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
    return GeneratedSegmentation(
        image=image.convert("RGB"),
        provider=provider,
        model=model,
        route=route,
        metadata=metadata,
    )


def _api_edit_size(canvas: Image.Image, budget_px: int = 1024 * 1536) -> str:
    """Legal ``WIDTHxHEIGHT`` for the images endpoint.

    A native canvas (``prepare_canvas``, the default) already sits on the
    16-px grid at the lane's budget, so it is requested verbatim and the
    output shares its pixel grid. Any other canvas gets the budget size at
    its aspect (multiples of 16), which the pipeline resamples back.
    """
    w, h = canvas.size
    if w % 16 == 0 and h % 16 == 0 and max(w, h) <= 3840 and 1 / 3 <= w / h <= 3:
        return f"{w}x{h}"
    s = (budget_px / (w * h)) ** 0.5
    return f"{round(w * s / 16) * 16}x{round(h * s / 16) * 16}"


def _generate_openai_oauth_segmentation(
    request: SegmentationGenerationRequest,
) -> GeneratedSegmentation:
    from langslice.providers import openai_oauth

    model = request.model or openai_oauth.DEFAULT_IMAGE_MODEL
    quality = (request.thinking_level or "high").lower()
    # The images/edits endpoint: the prompt reaches the image model verbatim.
    png_bytes = openai_oauth.edit_image(
        request.prompt,
        [
            _image_to_data_url(request.slice_image),
            *(_image_to_data_url(ref) for ref in request.reference_images),
        ],
        image_model=model,
        quality=quality if quality in _IMAGE_QUALITIES else "high",
    )
    image = Image.open(io.BytesIO(png_bytes))
    image.load()
    route = "openai_oauth_images_edit"
    return GeneratedSegmentation(
        image=image.convert("RGB"),
        provider="openai-oauth",
        model=model,
        route=route,
        metadata=_build_metadata(request, provider="openai-oauth", route=route),
    )


def generate_warped_segmentation_image(
    request: SegmentationGenerationRequest,
) -> GeneratedSegmentation:
    """Send *request* to its provider's image model; the edited image."""
    provider = canonical_provider(request.provider)
    if provider == "gemini-api":
        return _generate_google_segmentation(request)
    if provider == "openai-oauth":
        return _generate_openai_oauth_segmentation(request)
    if provider == "openai-api":
        return _generate_openai_images_segmentation(request, provider=provider)
    raise ValueError(f"Unknown provider: {request.provider}")

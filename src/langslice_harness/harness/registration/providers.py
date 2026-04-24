"""Provider-neutral image generation adapter for harness registration flows."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Any, cast

from PIL import Image

from langslice_harness import vlm_config
from langslice_harness.harness.registration.types import GeneratedSegmentation
from langslice_harness.openai_config import (
    get_openai_client,
    get_openai_image_client,
    get_openai_image_model,
    get_openai_model,
)

_VALID_REQUEST_ROUTES = {
    "google_genai",
    "openai_images",
    "openai_responses_image_generation",
}

_VALID_OPENAI_RESPONSES_PREFIXES = (
    "gpt-4o",
    "gpt-4.1",
    "o3",
    "gpt-5",
)


@dataclass
class SegmentationGenerationRequest:
    colored_regions: Image.Image
    reference_slice: Image.Image
    slice_image: Image.Image
    prompt: str
    provider: str = "google"
    model: str | None = None
    route: str | None = None
    review_model: str | None = None
    openai_image_route: str = "images"
    thinking_level: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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


def _extract_openai_responses_image(response: Any) -> tuple[Image.Image, str | None]:
    outputs = getattr(response, "output", None) or []
    for output in outputs:
        if getattr(output, "type", None) != "image_generation_call":
            continue

        result = getattr(output, "result", None)
        if not result:
            raise RuntimeError("OpenAI Responses image_generation call did not include result data")
        if isinstance(result, bytes):
            image_bytes = base64.b64decode(result)
        else:
            image_bytes = base64.b64decode(str(result))

        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        revised_prompt = getattr(output, "revised_prompt", None)
        return image.convert("RGB"), revised_prompt

    raise RuntimeError("OpenAI Responses API did not return an image_generation_call output")


def _build_metadata(
    request: SegmentationGenerationRequest,
    *,
    provider: str,
    route: str,
) -> dict[str, Any]:
    metadata = dict(request.metadata)
    metadata.update(
        {
            "provider": provider,
            "route": route,
            "request": {
                "provider": request.provider.lower(),
                "route": request.route.lower() if request.route else None,
                "openai_image_route": request.openai_image_route.lower(),
                "model": request.model,
                "review_model": request.review_model,
                "thinking_level": request.thinking_level,
                "prompt": request.prompt,
            },
        }
    )
    return metadata


def _validate_requested_route(request_route: str | None) -> None:
    if request_route is None:
        return
    normalized_route = request_route.lower()
    if normalized_route not in _VALID_REQUEST_ROUTES:
        raise ValueError(f"Unknown route: {request_route}")


def _validate_openai_responses_model(model_name: str) -> None:
    if model_name.startswith("gpt-image-"):
        raise ValueError("GPT Image models are not valid Responses model values")

    normalized_model = model_name.lower()
    if not normalized_model.startswith(_VALID_OPENAI_RESPONSES_PREFIXES):
        raise ValueError(
            "OpenAI Responses image_generation requires a text-capable OpenAI "
            "mainline model; pass review_model explicitly instead of relying on "
            f"the default {model_name!r}."
        )


def _generate_google_segmentation(request: SegmentationGenerationRequest) -> GeneratedSegmentation:
    model = request.model or vlm_config.MODEL_NAME
    client = vlm_config.get_client()

    contents = [
        request.colored_regions,
        request.reference_slice,
        request.slice_image,
        request.prompt,
    ]
    response = client.models.generate_content(  # type: ignore[attr-defined]
        model=model,
        contents=contents,
    )

    image = _extract_last_inline_image(response)
    route = "google_genai"
    return GeneratedSegmentation(
        image=image,
        provider="google",
        model=model,
        route=route,
        metadata=_build_metadata(request, provider="google", route=route),
    )


def _generate_openai_images_segmentation(
    request: SegmentationGenerationRequest,
    *,
    provider: str,
) -> GeneratedSegmentation:
    model = request.model or get_openai_image_model()
    client = get_openai_image_client()
    image_files = [
        _image_to_png_file(request.colored_regions, "colored_regions.png"),
        _image_to_png_file(request.reference_slice, "reference_slice.png"),
        _image_to_png_file(request.slice_image, "slice_image.png"),
    ]

    response = client.images.edit(  # type: ignore[attr-defined]
        model=model,
        image=image_files,
        prompt=request.prompt,
    )

    image_b64 = _extract_openai_image_b64(response)
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes))
    image.load()
    route = "openai_images"
    return GeneratedSegmentation(
        image=image.convert("RGB"),
        provider=provider,
        model=model,
        route=route,
        metadata=_build_metadata(request, provider=provider, route=route),
    )


def _generate_openai_responses_segmentation(
    request: SegmentationGenerationRequest,
) -> GeneratedSegmentation:
    mainline_model = request.review_model or get_openai_model()
    _validate_openai_responses_model(mainline_model)

    client = get_openai_client()
    contents = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": request.prompt},
                {"type": "input_image", "image_url": _image_to_data_url(request.colored_regions)},
                {"type": "input_image", "image_url": _image_to_data_url(request.reference_slice)},
                {"type": "input_image", "image_url": _image_to_data_url(request.slice_image)},
            ],
        }
    ]

    response = client.responses.create(  # type: ignore[attr-defined]
        model=mainline_model,
        input=cast(Any, contents),
        tools=[{"type": "image_generation", "action": "edit"}],
    )

    image, revised_prompt = _extract_openai_responses_image(response)
    route = "openai_responses_image_generation"
    return GeneratedSegmentation(
        image=image,
        provider="openai",
        model=mainline_model,
        route=route,
        revised_prompt=revised_prompt,
        metadata=_build_metadata(request, provider="openai", route=route),
    )


def generate_warped_segmentation_image(
    request: SegmentationGenerationRequest,
) -> GeneratedSegmentation:
    provider = request.provider.lower()
    _validate_requested_route(request.route)

    if provider == "google":
        return _generate_google_segmentation(request)

    if provider in {"openai", "flux", "openai-compatible", "openai_compatible"}:
        if provider == "openai":
            image_route = request.openai_image_route.lower()
            if image_route == "images":
                return _generate_openai_images_segmentation(request, provider=provider)
            if image_route == "responses":
                return _generate_openai_responses_segmentation(request)
            raise ValueError(f"Unknown openai_image_route: {request.openai_image_route}")

        return _generate_openai_images_segmentation(request, provider=provider)

    raise ValueError(f"Unknown provider: {request.provider}")

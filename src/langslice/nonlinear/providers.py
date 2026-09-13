"""Provider-neutral image generation adapter for harness registration flows."""

from __future__ import annotations

import base64
import importlib
import io
from dataclasses import dataclass, field
from typing import Any, cast

from PIL import Image

from langslice.nonlinear.model_prompts import gemini_aspect_for
from langslice.nonlinear.types import GeneratedSegmentation
from langslice.providers import vlm_config
from langslice.providers.openai_config import (
    get_openai_client,
    get_openai_image_client,
    get_openai_image_model,
    get_openai_model,
)
from langslice.providers.registry import canonical_provider

_VALID_REQUEST_ROUTES = {
    "google_genai",
    "openai_images",
    "openai_responses_image_generation",
    "openai_oauth_images_edit",
    "openai_oauth_image_generation",  # hosted-tool fallback path
    "chatgpt_responses_image_generation",  # legacy spelling of openai_oauth_image_generation
}

_IMAGE_QUALITIES = {"low", "medium", "high", "xhigh", "max"}


@dataclass
class SegmentationGenerationRequest:
    #: Model-facing atlas references, in prompt order: they follow the slice
    #: image as Image 2..N. The prompt describes what each one is; providers
    #: just deliver them in this order.
    reference_images: list[Image.Image]
    slice_image: Image.Image
    prompt: str
    provider: str = "google"
    model: str | None = None
    route: str | None = None
    review_model: str | None = None
    #: Task-level semantic: registration is always an EDIT of the slice image
    #: (pixel-aligned output). Each transport translates this its own way —
    #: the images endpoint IS an edit call, the Responses-based routes pass it
    #: as the image_generation tool's action.
    mode: str = "edit"
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


def _extract_inline_images(response: Any) -> list[Image.Image]:
    """EVERY inline image part of a Gemini response, in the order returned.

    :func:`_extract_last_inline_image` keeps the single-image contract (the
    last picture wins); this one is for prompts that ask for several pictures
    in one reply, where the order of the parts IS the order of the outputs.
    """
    images: list[Image.Image] = []
    for part in _response_parts(response):
        inline_data = getattr(part, "inline_data", None)
        image_bytes = getattr(inline_data, "data", None) if inline_data is not None else None
        if image_bytes:
            image = cast(Image.Image, Image.open(io.BytesIO(image_bytes)))
            image.load()
            images.append(image.convert("RGB"))
            continue

        # A text part has no picture to give; walking FORWARD over every part
        # means asking that question of text parts too, and ``as_image()`` is
        # not guaranteed to answer politely on one.
        if getattr(part, "text", None):
            continue
        as_image = getattr(part, "as_image", None)
        if callable(as_image):
            try:
                image = cast(Image.Image, as_image())
            except Exception:  # noqa: BLE001 - a part that holds no image
                continue
            if image is not None:
                if hasattr(image, "load"):
                    image.load()
                images.append(image.convert("RGB"))
    return images


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
    model = request.model or vlm_config.MODEL_NAME
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
        provider="google",
        model=model,
        route=route,
        metadata=_build_metadata(request, provider="google", route=route),
    )


def _generate_google_segmentation_images(
    request: SegmentationGenerationRequest,
) -> list[GeneratedSegmentation]:
    """Every image Gemini returns for one request, in reply order.

    Same contents, config and one retry as :func:`_generate_google_segmentation`;
    the retry fires only when the reply carried no picture at all (an imageless
    reply, e.g. IMAGE_RECITATION). A prompt that asks for two images gets two
    entries when the model obliges and one when it does not — the caller decides
    what a short reply means.
    """
    model = request.model or vlm_config.MODEL_NAME
    client = vlm_config.get_client()
    contents = [
        request.slice_image,
        *request.reference_images,
        request.prompt,
    ]
    # A reply carrying two pictures is a multi-part reply, and these models
    # narrate the pictures they emit, so IMAGE-only output is a plausible
    # reason a two-image request comes back as one blended picture. TEXT in
    # the modalities silently caps the image at 1K, so it is added only when
    # the run already asks for that tier or lower.
    modalities = (
        ("TEXT", "IMAGE")
        if (request.thinking_level or "").upper() in ("", "512", "512P", "512PX", "1K")
        else ("IMAGE",)
    )
    config = _gemini_image_config(request, modalities)
    response = client.models.generate_content(  # type: ignore[attr-defined]
        model=model, contents=contents, config=config
    )
    images = _extract_inline_images(response)
    if not images:
        response = client.models.generate_content(  # type: ignore[attr-defined]
            model=model, contents=contents, config=config
        )
        images = _extract_inline_images(response)
    if not images:
        raise RuntimeError("Gemini image generation did not return an inline image")

    route = "google_genai"
    metadata = _build_metadata(request, provider="google", route=route)
    return [
        GeneratedSegmentation(
            image=image,
            provider="google",
            model=model,
            route=route,
            metadata={
                **metadata,
                "image_index": index,
                "images_returned": len(images),
                "response_modalities": list(modalities),
            },
        )
        for index, image in enumerate(images)
    ]


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


def _generate_openai_responses_segmentation(
    request: SegmentationGenerationRequest,
) -> GeneratedSegmentation:
    mainline_model = request.review_model or get_openai_model()

    client = get_openai_client()
    contents = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": request.prompt},
                {"type": "input_image", "image_url": _image_to_data_url(request.slice_image)},
                *(
                    {"type": "input_image", "image_url": _image_to_data_url(ref)}
                    for ref in request.reference_images
                ),
            ],
        }
    ]

    reasoning_effort = (request.thinking_level or "medium").lower()

    response = client.responses.create(  # type: ignore[attr-defined]
        model=mainline_model,
        input=cast(Any, contents),
        tools=cast(Any, [{"type": "image_generation", "action": request.mode}]),
        reasoning=cast(Any, {"effort": reasoning_effort}),
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


def _generate_openai_oauth_segmentation(
    request: SegmentationGenerationRequest,
) -> GeneratedSegmentation:
    from langslice.providers import openai_oauth

    model = request.model or openai_oauth.DEFAULT_IMAGE_MODEL
    quality = (request.thinking_level or "high").lower()
    # Direct images/edits: one GPT model (the pilot) and one image model —
    # no server-side routing model rewriting the prompt in between.
    png_bytes = openai_oauth.edit_image(
        request.prompt,
        [
            _image_to_data_url(request.slice_image),
            *(_image_to_data_url(ref) for ref in request.reference_images),
        ],
        image_model=model,
        quality=quality if quality in _IMAGE_QUALITIES else "high",
    )
    revised_prompt = None  # nothing rewrites the prompt on this path

    image = Image.open(io.BytesIO(png_bytes))
    image.load()
    route = "openai_oauth_images_edit"
    return GeneratedSegmentation(
        image=image.convert("RGB"),
        provider="openai-oauth",
        model=model,
        route=route,
        revised_prompt=revised_prompt,
        metadata=_build_metadata(request, provider="openai-oauth", route=route),
    )


def generate_warped_segmentation_image(
    request: SegmentationGenerationRequest,
) -> GeneratedSegmentation:
    provider = canonical_provider(request.provider)
    _validate_requested_route(request.route)

    if provider == "gemini-api":
        return _generate_google_segmentation(request)

    if provider == "openai-oauth":
        return _generate_openai_oauth_segmentation(request)

    if provider == "openai-api":
        image_route = request.openai_image_route.lower()
        if image_route == "images":
            return _generate_openai_images_segmentation(request, provider=provider)
        if image_route == "responses":
            return _generate_openai_responses_segmentation(request)
        raise ValueError(f"Unknown openai_image_route: {request.openai_image_route}")

    raise ValueError(f"Unknown provider: {request.provider}")


def generate_warped_segmentation_images(
    request: SegmentationGenerationRequest,
) -> list[GeneratedSegmentation]:
    """All the images one request yields, in the order the provider returned them.

    Only the Gemini lane can hand back more than one picture per reply, so only
    it has its own multi-image path; every other lane returns the single image of
    :func:`generate_warped_segmentation_image` as a one-element list. Callers that
    asked for several outputs read the list length to learn how many arrived.
    """
    provider = canonical_provider(request.provider)
    _validate_requested_route(request.route)

    if provider == "gemini-api":
        return _generate_google_segmentation_images(request)

    return [generate_warped_segmentation_image(request)]

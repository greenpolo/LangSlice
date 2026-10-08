from __future__ import annotations

import base64
import io
from types import SimpleNamespace
from typing import Any, cast

import pytest
from PIL import Image

from langslice.core.nonlinear.types import GeneratedSegmentation


def _providers():
    from langslice.providers import images as providers

    return providers


def _make_image(color: tuple[int, int, int], size: tuple[int, int] = (8, 6)) -> Image.Image:
    return Image.new("RGB", size, color=color)


def _decode_image(image: Image.Image) -> tuple[int, int, tuple[int, int, int]]:
    pixel = image.getpixel((0, 0))
    if isinstance(pixel, int):
        pixel = (pixel, pixel, pixel)
    if not isinstance(pixel, tuple):
        raise AssertionError("expected RGB pixel")
    return image.size[0], image.size[1], cast(tuple[int, int, int], pixel)


class _FakeOpenAIImagesClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.images = SimpleNamespace(edit=self.edit)

    def edit(self, **kwargs):  # noqa: ANN003 - SDK-shaped fake
        self.calls.append(kwargs)
        image = Image.new("RGB", (9, 7), color=(12, 34, 56))
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return SimpleNamespace(
            data=[SimpleNamespace(b64_json=base64.b64encode(buf.getvalue()).decode("ascii"))]
        )


class _FakeGeminiPart:
    def __init__(self, *, text: str | None = None, image: Image.Image | None = None) -> None:
        self.text = text
        self.inline_data = None
        if image is not None:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            self.inline_data = SimpleNamespace(data=buf.getvalue())

    def as_image(self) -> Image.Image:
        assert self.inline_data is not None
        return Image.open(io.BytesIO(self.inline_data.data))


class _FakeGeminiResponse:
    def __init__(self, parts: list[_FakeGeminiPart]) -> None:
        self.parts = parts
        self.candidates = []


class _FakeGeminiModels:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs):  # noqa: ANN003 - SDK-shaped fake
        self.calls.append(kwargs)
        return _FakeGeminiResponse(
            [
                _FakeGeminiPart(text="ignore me"),
                _FakeGeminiPart(image=Image.new("RGB", (6, 4), color=(1, 2, 3))),
                _FakeGeminiPart(image=Image.new("RGB", (7, 5), color=(4, 5, 6))),
            ]
        )


class _FakeGeminiClient:
    def __init__(self) -> None:
        self.models = _FakeGeminiModels()


def test_openai_images_route_sends_the_edited_image_first(monkeypatch):
    providers = _providers()
    fake_client = _FakeOpenAIImagesClient()
    monkeypatch.setattr(providers, "get_openai_image_client", lambda: fake_client)
    monkeypatch.setattr(providers, "get_openai_image_model", lambda: "gpt-image-2.5-sunburst")

    request = providers.SegmentationGenerationRequest(
        reference_images=[_make_image((255, 0, 0)), _make_image((0, 255, 0))],
        slice_image=_make_image((0, 0, 255)),
        prompt="warp it",
        provider="openai-api",
    )

    result = providers.generate_warped_segmentation_image(request)

    assert isinstance(result, GeneratedSegmentation)
    assert result.provider == "openai-api"
    assert result.model == "gpt-image-2.5-sunburst"
    assert result.route == "openai_images"
    assert result.revised_prompt is None
    assert result.metadata["provider"] == "openai-api"
    assert result.metadata["request"]["prompt"] == "warp it"
    assert _decode_image(result.image) == (9, 7, (12, 34, 56))

    assert len(fake_client.calls) == 1
    call: dict[str, Any] = fake_client.calls[0]
    assert call["model"] == "gpt-image-2.5-sunburst"
    assert call["prompt"] == "warp it"
    image_files = cast(list[io.BytesIO], call["image"])
    assert len(image_files) == 3
    # Image 1 is the image being edited; the references follow in prompt order.
    assert [img.name for img in image_files] == [
        "slice_image.png",
        "atlas_reference_1.png",
        "atlas_reference_2.png",
    ]


def test_google_route_uses_last_inline_image_from_parts(monkeypatch):
    providers = _providers()
    fake_client = _FakeGeminiClient()
    monkeypatch.setattr(providers.vlm_config, "get_client", lambda: fake_client)

    request = providers.SegmentationGenerationRequest(
        reference_images=[_make_image((255, 0, 0)), _make_image((0, 255, 0))],
        slice_image=_make_image((0, 0, 255)),
        prompt="google it",
        provider="gemini-api",
        size_tier="high",
    )

    result = providers.generate_warped_segmentation_image(request)

    # No model named: the lane's image model, never a text model.
    assert result.provider == "gemini-api"
    assert result.model == "gemini-3.1-flash-image"
    assert result.route == "google_genai"
    assert _decode_image(result.image) == (7, 5, (4, 5, 6))
    assert result.metadata["provider"] == "gemini-api"

    assert len(fake_client.models.calls) == 1
    call: dict[str, Any] = fake_client.models.calls[0]
    assert call["model"] == "gemini-3.1-flash-image"
    contents = cast(list[Any], call["contents"])
    assert len(contents) == 4
    assert isinstance(contents[0], Image.Image)
    assert isinstance(contents[1], Image.Image)
    assert isinstance(contents[2], Image.Image)
    assert contents[3] == "google it"


def test_openai_api_provider_uses_the_images_route(monkeypatch):
    providers = _providers()
    fake_client = _FakeOpenAIImagesClient()
    monkeypatch.setattr(providers, "get_openai_image_client", lambda: fake_client)
    monkeypatch.setattr(providers, "get_openai_image_model", lambda: "gpt-image-2.5-sunburst")

    request = providers.SegmentationGenerationRequest(
        reference_images=[_make_image((255, 0, 0)), _make_image((0, 255, 0))],
        slice_image=_make_image((0, 0, 255)),
        prompt="compat",
        provider="openai-api",
        model="gpt-image-2.5-sunburst",
    )

    result = providers.generate_warped_segmentation_image(request)

    assert result.provider == "openai-api"  # canonical access-method name
    assert result.route == "openai_images"
    assert result.model == "gpt-image-2.5-sunburst"
    assert _decode_image(result.image) == (9, 7, (12, 34, 56))


def test_unknown_provider_raises_value_error():
    providers = _providers()
    request = providers.SegmentationGenerationRequest(
        reference_images=[_make_image((255, 0, 0)), _make_image((0, 255, 0))],
        slice_image=_make_image((0, 0, 255)),
        prompt="nope",
        provider="mystery",
    )

    with pytest.raises(ValueError, match="Unknown provider"):
        providers.generate_warped_segmentation_image(request)


class _FakeGeminiModelsWithParts:
    """Gemini models stub that replays a scripted list of responses."""

    def __init__(self, responses: list[_FakeGeminiResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs):  # noqa: ANN003 - SDK-shaped fake
        self.calls.append(kwargs)
        return self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]


def _gemini_client(responses: list[_FakeGeminiResponse]):
    client = _FakeGeminiClient()
    client.models = _FakeGeminiModelsWithParts(responses)
    return client


def _dual_request(providers):
    return providers.SegmentationGenerationRequest(
        reference_images=[_make_image((255, 0, 0)), _make_image((0, 255, 0))],
        slice_image=_make_image((0, 0, 255)),
        prompt="two please",
        provider="gemini-api",
        size_tier="1K",
    )


def test_single_image_google_route_keeps_image_only_modalities(monkeypatch):
    providers = _providers()
    fake_client = _gemini_client(
        [_FakeGeminiResponse([_FakeGeminiPart(image=Image.new("RGB", (4, 3), color=(2, 2, 2)))])]
    )
    monkeypatch.setattr(providers.vlm_config, "get_client", lambda: fake_client)

    providers.generate_warped_segmentation_image(_dual_request(providers))

    assert list(fake_client.models.calls[0]["config"].response_modalities) == ["IMAGE"]


def test_the_openai_api_image_client_defaults_to_the_openai_api(monkeypatch):
    """An OPENAI_API_KEY alone reaches the OpenAI API itself; no key is refused."""
    from langslice.providers import openai_config

    made: list[dict[str, Any]] = []

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            made.append(kwargs)

    monkeypatch.setattr(openai_config.importlib, "import_module",
                        lambda name: SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setattr(openai_config, "_image_client_instance", None)
    for name in ("OPENAI_IMAGE_API_KEY", "OPENAI_API_KEY", "OPENAI_IMAGE_BASE_URL",
                 "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        openai_config.get_openai_image_client()
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    openai_config.get_openai_image_client()
    assert made == [{"base_url": "https://api.openai.com/v1", "api_key": "key"}]
    monkeypatch.setattr(openai_config, "_image_client_instance", None)


def test_every_image_provider_resolves_to_an_image_model():
    from langslice.providers.registry import resolve_image_model

    assert resolve_image_model("gemini-api").model == "gemini-3.1-flash-image"
    assert resolve_image_model("openai-oauth").model == "gpt-image-2.5-sunburst"

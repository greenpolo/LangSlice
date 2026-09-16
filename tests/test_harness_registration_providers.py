from __future__ import annotations

import base64
import io
from types import SimpleNamespace
from typing import Any, cast

import pytest
from PIL import Image

from langslice.nonlinear.types import GeneratedSegmentation


def _providers():
    from langslice.nonlinear import providers

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


class _FakeOpenAIResponsesClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **kwargs):  # noqa: ANN003 - SDK-shaped fake
        self.calls.append(kwargs)
        image = Image.new("RGB", (11, 5), color=(90, 80, 70))
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return SimpleNamespace(
            output=[
                SimpleNamespace(type="text", result=None),
                SimpleNamespace(
                    type="image_generation_call",
                    result=base64.b64encode(buf.getvalue()).decode("ascii"),
                    revised_prompt="revised prompt",
                ),
            ]
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
    monkeypatch.setattr(providers, "get_openai_image_model", lambda: "gpt-image-2")

    request = providers.SegmentationGenerationRequest(
        reference_images=[_make_image((255, 0, 0)), _make_image((0, 255, 0))],
        slice_image=_make_image((0, 0, 255)),
        prompt="warp it",
        provider="openai",
    )

    result = providers.generate_warped_segmentation_image(request)

    assert isinstance(result, GeneratedSegmentation)
    assert result.provider == "openai-api"
    assert result.model == "gpt-image-2"
    assert result.route == "openai_images"
    assert result.revised_prompt is None
    assert result.metadata["provider"] == "openai-api"
    assert result.metadata["request"]["prompt"] == "warp it"
    assert _decode_image(result.image) == (9, 7, (12, 34, 56))

    assert len(fake_client.calls) == 1
    call: dict[str, Any] = fake_client.calls[0]
    assert call["model"] == "gpt-image-2"
    assert call["prompt"] == "warp it"
    image_files = cast(list[io.BytesIO], call["image"])
    assert len(image_files) == 3
    # Image 1 is the image being edited; the references follow in prompt order.
    assert [img.name for img in image_files] == [
        "slice_image.png",
        "atlas_reference_1.png",
        "atlas_reference_2.png",
    ]


def test_openai_responses_route_uses_image_generation_tool_and_revised_prompt(monkeypatch):
    providers = _providers()
    fake_client = _FakeOpenAIResponsesClient()
    monkeypatch.setattr(providers, "get_openai_client", lambda: fake_client)
    monkeypatch.setattr(providers, "get_openai_model", lambda: "gpt-4.1")

    request = providers.SegmentationGenerationRequest(
        reference_images=[_make_image((255, 0, 0)), _make_image((0, 255, 0))],
        slice_image=_make_image((0, 0, 255)),
        prompt="edit please",
        provider="openai",
        openai_image_route="responses",
        review_model="gpt-4.1",
    )

    result = providers.generate_warped_segmentation_image(request)

    assert result.route == "openai_responses_image_generation"
    assert result.revised_prompt == "revised prompt"
    assert _decode_image(result.image) == (11, 5, (90, 80, 70))

    assert len(fake_client.calls) == 1
    call: dict[str, Any] = fake_client.calls[0]
    assert call["model"] == "gpt-4.1"
    assert call["tools"] == [{"type": "image_generation", "action": "edit"}]

    input_payload = cast(list[dict[str, Any]], call["input"])
    content = cast(list[dict[str, Any]], input_payload[0]["content"])
    assert content[0] == {"type": "input_text", "text": "edit please"}
    image_parts = [part for part in content if part["type"] == "input_image"]
    assert len(image_parts) == 3
    for part in image_parts:
        assert part["image_url"].startswith("data:image/png;base64,")


def test_google_route_uses_last_inline_image_from_parts(monkeypatch):
    providers = _providers()
    fake_client = _FakeGeminiClient()
    monkeypatch.setattr(providers.vlm_config, "get_client", lambda: fake_client)
    monkeypatch.setattr(
        providers.vlm_config._runtime,
        "model_name",
        "gemini-3.1-flash-image-preview",
    )

    request = providers.SegmentationGenerationRequest(
        reference_images=[_make_image((255, 0, 0)), _make_image((0, 255, 0))],
        slice_image=_make_image((0, 0, 255)),
        prompt="google it",
        provider="google",
        thinking_level="high",
    )

    result = providers.generate_warped_segmentation_image(request)

    assert result.provider == "google"
    assert result.model == "gemini-3.1-flash-image-preview"
    assert result.route == "google_genai"
    assert _decode_image(result.image) == (7, 5, (4, 5, 6))
    assert result.metadata["provider"] == "google"

    assert len(fake_client.models.calls) == 1
    call: dict[str, Any] = fake_client.models.calls[0]
    assert call["model"] == "gemini-3.1-flash-image-preview"
    contents = cast(list[Any], call["contents"])
    assert len(contents) == 4
    assert isinstance(contents[0], Image.Image)
    assert isinstance(contents[1], Image.Image)
    assert isinstance(contents[2], Image.Image)
    assert contents[3] == "google it"


def test_openai_compatible_provider_uses_images_route_and_normalized_provider(monkeypatch):
    providers = _providers()
    fake_client = _FakeOpenAIImagesClient()
    monkeypatch.setattr(providers, "get_openai_image_client", lambda: fake_client)
    monkeypatch.setattr(providers, "get_openai_image_model", lambda: "gpt-image-2")

    request = providers.SegmentationGenerationRequest(
        reference_images=[_make_image((255, 0, 0)), _make_image((0, 255, 0))],
        slice_image=_make_image((0, 0, 255)),
        prompt="compat",
        provider="OpenAI-Compatible",
        model="gpt-image-2",
    )

    result = providers.generate_warped_segmentation_image(request)

    assert result.provider == "openai-api"  # canonical access-method name
    assert result.route == "openai_images"
    assert result.model == "gpt-image-2"
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


def test_unknown_openai_image_route_raises_value_error():
    providers = _providers()
    request = providers.SegmentationGenerationRequest(
        reference_images=[_make_image((255, 0, 0)), _make_image((0, 255, 0))],
        slice_image=_make_image((0, 0, 255)),
        prompt="nope",
        provider="openai",
        openai_image_route="bad-route",
    )

    with pytest.raises(ValueError, match="Unknown openai_image_route"):
        providers.generate_warped_segmentation_image(request)


def test_unknown_request_route_raises_value_error():
    providers = _providers()
    request = providers.SegmentationGenerationRequest(
        reference_images=[_make_image((255, 0, 0)), _make_image((0, 255, 0))],
        slice_image=_make_image((0, 0, 255)),
        prompt="nope",
        provider="google",
        route="bogus",
    )

    with pytest.raises(ValueError, match="Unknown route"):
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
        provider="google",
        thinking_level="1K",
    )


def test_google_multi_image_route_returns_every_image_in_reply_order(monkeypatch):
    providers = _providers()
    fake_client = _gemini_client(
        [
            _FakeGeminiResponse(
                [
                    _FakeGeminiPart(text="here you go"),
                    _FakeGeminiPart(image=Image.new("RGB", (6, 4), color=(1, 2, 3))),
                    _FakeGeminiPart(image=Image.new("RGB", (7, 5), color=(4, 5, 6))),
                ]
            )
        ]
    )
    monkeypatch.setattr(providers.vlm_config, "get_client", lambda: fake_client)
    monkeypatch.setattr(providers.vlm_config._runtime, "model_name", "gemini-3.1-flash-image")

    results = providers.generate_warped_segmentation_images(_dual_request(providers))

    assert [_decode_image(r.image) for r in results] == [(6, 4, (1, 2, 3)), (7, 5, (4, 5, 6))]
    assert [r.metadata["image_index"] for r in results] == [0, 1]
    assert {r.metadata["images_returned"] for r in results} == {2}
    assert {r.route for r in results} == {"google_genai"}
    assert len(fake_client.models.calls) == 1
    # The single-image function still takes the LAST image; nothing about it moved.
    assert _decode_image(
        providers.generate_warped_segmentation_image(_dual_request(providers)).image
    ) == (7, 5, (4, 5, 6))


def test_google_multi_image_route_returns_a_single_image_as_one_element(monkeypatch):
    providers = _providers()
    fake_client = _gemini_client(
        [_FakeGeminiResponse([_FakeGeminiPart(image=Image.new("RGB", (6, 4), color=(9, 9, 9)))])]
    )
    monkeypatch.setattr(providers.vlm_config, "get_client", lambda: fake_client)
    monkeypatch.setattr(providers.vlm_config._runtime, "model_name", "gemini-3.1-flash-image")

    results = providers.generate_warped_segmentation_images(_dual_request(providers))

    assert len(results) == 1
    assert _decode_image(results[0].image) == (6, 4, (9, 9, 9))


def test_google_multi_image_route_retries_once_on_an_imageless_reply(monkeypatch):
    providers = _providers()
    fake_client = _gemini_client(
        [
            _FakeGeminiResponse([_FakeGeminiPart(text="no picture this time")]),
            _FakeGeminiResponse(
                [
                    _FakeGeminiPart(image=Image.new("RGB", (3, 2), color=(7, 7, 7))),
                    _FakeGeminiPart(image=Image.new("RGB", (3, 2), color=(8, 8, 8))),
                ]
            ),
        ]
    )
    monkeypatch.setattr(providers.vlm_config, "get_client", lambda: fake_client)
    monkeypatch.setattr(providers.vlm_config._runtime, "model_name", "gemini-3.1-flash-image")

    results = providers.generate_warped_segmentation_images(_dual_request(providers))

    assert len(fake_client.models.calls) == 2
    assert [_decode_image(r.image)[2] for r in results] == [(7, 7, 7), (8, 8, 8)]


def test_google_multi_image_route_raises_when_no_image_ever_arrives(monkeypatch):
    providers = _providers()
    fake_client = _gemini_client([_FakeGeminiResponse([_FakeGeminiPart(text="still nothing")])])
    monkeypatch.setattr(providers.vlm_config, "get_client", lambda: fake_client)
    monkeypatch.setattr(providers.vlm_config._runtime, "model_name", "gemini-3.1-flash-image")

    with pytest.raises(RuntimeError, match="did not return an inline image"):
        providers.generate_warped_segmentation_images(_dual_request(providers))
    assert len(fake_client.models.calls) == 2


def test_non_google_provider_returns_its_single_image_as_a_one_element_list(monkeypatch):
    providers = _providers()
    fake_client = _FakeOpenAIImagesClient()
    monkeypatch.setattr(providers, "get_openai_image_client", lambda: fake_client)
    monkeypatch.setattr(providers, "get_openai_image_model", lambda: "gpt-image-2")

    request = providers.SegmentationGenerationRequest(
        reference_images=[_make_image((255, 0, 0)), _make_image((0, 255, 0))],
        slice_image=_make_image((0, 0, 255)),
        prompt="one only",
        provider="openai",
    )

    results = providers.generate_warped_segmentation_images(request)

    assert len(results) == 1
    assert results[0].route == "openai_images"
    assert _decode_image(results[0].image) == (9, 7, (12, 34, 56))


def test_google_multi_image_route_asks_for_text_and_image_at_1k_only(monkeypatch):
    """TEXT in the modalities lets a reply interleave several pictures; it also
    caps the image at 1K, so a higher tier keeps IMAGE-only."""
    providers = _providers()
    monkeypatch.setattr(providers.vlm_config._runtime, "model_name", "gemini-3.1-flash-image")

    seen = {}
    for tier, expected in (("1K", ["TEXT", "IMAGE"]), (None, ["TEXT", "IMAGE"]), ("4K", ["IMAGE"])):
        part = _FakeGeminiPart(image=Image.new("RGB", (4, 3), color=(1, 1, 1)))
        fake_client = _gemini_client([_FakeGeminiResponse([part])])
        monkeypatch.setattr(providers.vlm_config, "get_client", lambda c=fake_client: c)
        request = _dual_request(providers)
        request.thinking_level = tier
        results = providers.generate_warped_segmentation_images(request)
        config = fake_client.models.calls[0]["config"]
        seen[tier] = list(config.response_modalities)
        assert seen[tier] == expected
        assert results[0].metadata["response_modalities"] == expected


def test_single_image_google_route_keeps_image_only_modalities(monkeypatch):
    providers = _providers()
    fake_client = _gemini_client(
        [_FakeGeminiResponse([_FakeGeminiPart(image=Image.new("RGB", (4, 3), color=(2, 2, 2)))])]
    )
    monkeypatch.setattr(providers.vlm_config, "get_client", lambda: fake_client)
    monkeypatch.setattr(providers.vlm_config._runtime, "model_name", "gemini-3.1-flash-image")

    providers.generate_warped_segmentation_image(_dual_request(providers))

    assert list(fake_client.models.calls[0]["config"].response_modalities) == ["IMAGE"]

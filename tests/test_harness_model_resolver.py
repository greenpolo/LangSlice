import asyncio

import pytest
from google.adk.models import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from PIL import Image

from langslice_harness.harness.estimation import model_resolver
from langslice_harness.harness.estimation.runner import run_single_slice_session


class _FakeLiteLlm:
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs


def _count_inline_images(contents: list[types.Content]) -> int:
    count = 0
    for content in contents:
        for part in content.parts or []:
            inline = getattr(part, "inline_data", None)
            data = getattr(inline, "data", None) if inline is not None else None
            if data:
                count += 1
    return count


def _count_file_images(contents: list[types.Content]) -> int:
    count = 0
    for content in contents:
        for part in content.parts or []:
            file_data = getattr(part, "file_data", None)
            file_uri = getattr(file_data, "file_uri", None) if file_data else None
            if file_uri:
                count += 1
    return count


def test_resolve_litellm_proxy_model_uses_proxy_alias(monkeypatch):
    monkeypatch.setattr(model_resolver, "_load_litellm_class", lambda: _FakeLiteLlm)
    monkeypatch.setenv("LANGSLICE_LITELLM_PROXY_BASE", "http://127.0.0.1:4000/v1")
    monkeypatch.setenv("LANGSLICE_LITELLM_PROXY_KEY", "sk-local")

    model = model_resolver.resolve_adk_model("litellm-proxy:langslice-qwen36-plus")

    assert isinstance(model, _FakeLiteLlm)
    assert model.model == "openai/langslice-qwen36-plus"
    assert model.kwargs["api_base"] == "http://127.0.0.1:4000/v1"
    assert model.kwargs["api_key"] == "sk-local"

    # Transport-side effect: resolving LiteLLM proxy shorthands must stay inline (not file_api).
    captured_first_request: list[types.Content] = []

    class CaptureLiteLlm(BaseLlm):
        def __init__(self, model: str, **kwargs) -> None:
            del kwargs
            super().__init__(model=model)

        async def generate_content_async(self, llm_request, stream: bool = False):
            del stream
            contents = list(llm_request.contents or [])
            if not captured_first_request:
                captured_first_request.extend(contents)
            call = types.Part.from_function_call(
                name="fetch_atlas",
                args={"positions_mm": [4.0, 5.0, 6.0]},
            )
            yield LlmResponse(
                content=types.Content(role="model", parts=[call]),
                partial=False,
                turn_complete=True,
            )

    monkeypatch.setattr(model_resolver, "_load_litellm_class", lambda: CaptureLiteLlm)

    def _raise_if_called():
        raise AssertionError("LiteLLM proxy path should not create a Gemini client")

    monkeypatch.setattr("langslice_harness.vlm_config.get_client", _raise_if_called)

    asyncio.run(
        run_single_slice_session(
            image=Image.new("RGB", (256, 256), color=128),
            atlas_name="allen_mouse_25um",
            plane="coronal",
            model="litellm-proxy:langslice-qwen36-plus",
            max_iterations=1,
            max_retries=1,
            apply_clahe=False,
        )
    )

    assert _count_inline_images(captured_first_request) == 1
    assert _count_file_images(captured_first_request) == 0


def test_resolve_openrouter_model_uses_openrouter_key(monkeypatch):
    monkeypatch.setattr(model_resolver, "_load_litellm_class", lambda: _FakeLiteLlm)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")

    model = model_resolver.resolve_adk_model("openrouter:qwen/qwen3.6-plus")

    assert isinstance(model, _FakeLiteLlm)
    assert model.model == "openrouter/qwen/qwen3.6-plus"
    assert model.kwargs["api_key"] == "sk-or"


def test_resolve_plain_openai_model_uses_litellm_openai(monkeypatch):
    monkeypatch.setattr(model_resolver, "_load_litellm_class", lambda: _FakeLiteLlm)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    model = model_resolver.resolve_adk_model("gpt-4.1")

    assert isinstance(model, _FakeLiteLlm)
    assert model.model == "openai/gpt-4.1"
    assert model.kwargs["api_key"] == "sk-openai"
    assert model.kwargs["api_base"] == "https://api.openai.com/v1"


def test_resolve_ollama_model_uses_local_ollama_chat(monkeypatch):
    monkeypatch.setattr(model_resolver, "_load_litellm_class", lambda: _FakeLiteLlm)
    monkeypatch.setenv("LANGSLICE_OLLAMA_BASE", "http://localhost:11434")

    model = model_resolver.resolve_adk_model("ollama:gemma4:26b")

    assert isinstance(model, _FakeLiteLlm)
    assert model.model == "ollama_chat/gemma4:26b"
    assert model.kwargs["api_base"] == "http://localhost:11434"

    # Transport-side effect: resolving local Ollama shorthands must stay inline (not file_api).
    captured_first_request: list[types.Content] = []

    class CaptureOllamaLiteLlm(BaseLlm):
        def __init__(self, model: str, **kwargs) -> None:
            assert kwargs["api_base"] == "http://localhost:11434"
            super().__init__(model=model)

        async def generate_content_async(self, llm_request, stream: bool = False):
            del stream
            contents = list(llm_request.contents or [])
            if not captured_first_request:
                captured_first_request.extend(contents)
            call = types.Part.from_function_call(
                name="fetch_atlas",
                args={"positions_mm": [4.0, 5.0, 6.0]},
            )
            yield LlmResponse(
                content=types.Content(role="model", parts=[call]),
                partial=False,
                turn_complete=True,
            )

    monkeypatch.setattr(model_resolver, "_load_litellm_class", lambda: CaptureOllamaLiteLlm)

    def _raise_if_called():
        raise AssertionError("Ollama path should not create a Gemini client")

    monkeypatch.setattr("langslice_harness.vlm_config.get_client", _raise_if_called)

    asyncio.run(
        run_single_slice_session(
            image=Image.new("RGB", (256, 256), color=128),
            atlas_name="allen_mouse_25um",
            plane="coronal",
            model="ollama:gemma4:26b",
            max_iterations=1,
            max_retries=1,
            apply_clahe=False,
        )
    )

    assert _count_inline_images(captured_first_request) == 1
    assert _count_file_images(captured_first_request) == 0


def test_resolve_bare_local_model_tag_uses_ollama_chat(monkeypatch):
    monkeypatch.setattr(model_resolver, "_load_litellm_class", lambda: _FakeLiteLlm)
    monkeypatch.setenv("LANGSLICE_OLLAMA_BASE", "http://localhost:11434")

    model = model_resolver.resolve_adk_model("gemma4:31b")

    assert isinstance(model, _FakeLiteLlm)
    assert model.model == "ollama_chat/gemma4:31b"
    assert model.kwargs["api_base"] == "http://localhost:11434"


def test_resolve_ollama_model_can_disable_thinking(monkeypatch):
    monkeypatch.setattr(model_resolver, "_load_litellm_class", lambda: _FakeLiteLlm)
    monkeypatch.setenv("LANGSLICE_OLLAMA_THINK", "false")

    model = model_resolver.resolve_adk_model("ollama:gemma4:26b")

    assert isinstance(model, _FakeLiteLlm)
    assert model.model == "ollama_chat/gemma4:26b"
    assert model.kwargs["think"] is False


def test_resolve_ollama_model_rejects_invalid_thinking_flag(monkeypatch):
    monkeypatch.setattr(model_resolver, "_load_litellm_class", lambda: _FakeLiteLlm)
    monkeypatch.setenv("LANGSLICE_OLLAMA_THINK", "sometimes")

    with pytest.raises(ValueError, match="LANGSLICE_OLLAMA_THINK"):
        model_resolver.resolve_adk_model("ollama:gemma4:26b")


def test_resolve_hosted_gemma_model_uses_adk_gemma(monkeypatch):
    monkeypatch.setattr(model_resolver, "_load_adk_gemma_class", lambda: _FakeLiteLlm)

    model = model_resolver.resolve_adk_model("models/gemma-4-26b-a4b-it")

    assert isinstance(model, _FakeLiteLlm)
    assert model.model == "gemma-4-26b-a4b-it"
    assert model.kwargs == {}


def test_resolve_litellm_model_reports_missing_dependency(monkeypatch):
    def _raise_import_error():
        raise ImportError("LiteLLM support requires: pip install google-adk[extensions]")

    monkeypatch.setattr(model_resolver, "_load_litellm_class", _raise_import_error)

    with pytest.raises(RuntimeError, match="LiteLLM support requires"):
        model_resolver.resolve_adk_model("litellm-proxy:langslice-qwen36-plus")

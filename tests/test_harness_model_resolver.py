import pytest

from langslice.harness.estimation import model_resolver


class _FakeLiteLlm:
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs


def test_resolve_litellm_proxy_model_uses_proxy_alias(monkeypatch):
    monkeypatch.setattr(model_resolver, "_load_litellm_class", lambda: _FakeLiteLlm)
    monkeypatch.setenv("LANGSLICE_LITELLM_PROXY_BASE", "http://127.0.0.1:4000/v1")
    monkeypatch.setenv("LANGSLICE_LITELLM_PROXY_KEY", "sk-local")

    model = model_resolver.resolve_adk_model("litellm-proxy:langslice-qwen36-plus")

    assert isinstance(model, _FakeLiteLlm)
    assert model.model == "openai/langslice-qwen36-plus"
    assert model.kwargs["api_base"] == "http://127.0.0.1:4000/v1"
    assert model.kwargs["api_key"] == "sk-local"


def test_resolve_openrouter_model_uses_openrouter_key(monkeypatch):
    monkeypatch.setattr(model_resolver, "_load_litellm_class", lambda: _FakeLiteLlm)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")

    model = model_resolver.resolve_adk_model("openrouter:qwen/qwen3.6-plus")

    assert isinstance(model, _FakeLiteLlm)
    assert model.model == "openrouter/qwen/qwen3.6-plus"
    assert model.kwargs["api_key"] == "sk-or"


def test_resolve_ollama_model_uses_local_ollama_chat(monkeypatch):
    monkeypatch.setattr(model_resolver, "_load_litellm_class", lambda: _FakeLiteLlm)
    monkeypatch.setenv("LANGSLICE_OLLAMA_BASE", "http://localhost:11434")

    model = model_resolver.resolve_adk_model("ollama:gemma4:26b")

    assert isinstance(model, _FakeLiteLlm)
    assert model.model == "ollama_chat/gemma4:26b"
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

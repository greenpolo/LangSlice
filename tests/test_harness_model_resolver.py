"""Model string to ADK model: the litellm, ollama and hosted-Gemma routes."""

import sys

import pytest

from langslice.agent import model_resolver


class _FakeLiteLlm:
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs


_PROXY_ENV = {"LANGSLICE_LITELLM_PROXY_BASE": "http://127.0.0.1:4000/v1",
              "LANGSLICE_LITELLM_PROXY_KEY": "sk-local"}
_OLLAMA_ENV = {"LANGSLICE_OLLAMA_BASE": "http://localhost:11434"}


@pytest.mark.parametrize(("string", "env", "model_id", "kwargs"), [
    ("litellm-proxy:langslice-qwen36-plus", _PROXY_ENV, "openai/langslice-qwen36-plus",
     {"api_base": "http://127.0.0.1:4000/v1", "api_key": "sk-local"}),
    ("openrouter:qwen/qwen3.6-plus", {"OPENROUTER_API_KEY": "sk-or"},
     "openrouter/qwen/qwen3.6-plus", {"api_key": "sk-or"}),
    ("gpt-4.1", {"OPENAI_API_KEY": "sk-openai", "OPENAI_BASE_URL": "https://api.openai.com/v1"},
     "openai/gpt-4.1", {"api_key": "sk-openai", "api_base": "https://api.openai.com/v1"}),
    ("ollama:gemma4:26b", _OLLAMA_ENV, "ollama_chat/gemma4:26b",
     {"api_base": "http://localhost:11434"}),
    ("gemma4:31b", _OLLAMA_ENV, "ollama_chat/gemma4:31b", {"api_base": "http://localhost:11434"}),
    ("ollama:gemma4:26b", {"LANGSLICE_OLLAMA_THINK": "false"}, "ollama_chat/gemma4:26b",
     {"think": False}),
])
def test_litellm_routes(monkeypatch, string, env, model_id, kwargs):
    monkeypatch.setattr(model_resolver, "_load_litellm_class", lambda: _FakeLiteLlm)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    model = model_resolver.resolve_adk_model(string)
    assert isinstance(model, _FakeLiteLlm) and model.model == model_id
    assert {key: model.kwargs[key] for key in kwargs} == kwargs


def test_hosted_gemma_and_the_failures(monkeypatch):
    monkeypatch.setattr(model_resolver, "_load_adk_gemma_class", lambda: _FakeLiteLlm)
    model = model_resolver.resolve_adk_model("models/gemma-4-26b-a4b-it")
    assert model.model == "gemma-4-26b-a4b-it" and model.kwargs == {}

    monkeypatch.setattr(model_resolver, "_load_litellm_class", lambda: _FakeLiteLlm)
    monkeypatch.setenv("LANGSLICE_OLLAMA_THINK", "sometimes")
    with pytest.raises(ValueError, match="LANGSLICE_OLLAMA_THINK"):
        model_resolver.resolve_adk_model("ollama:gemma4:26b")

    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "google.adk.models.lite_llm", None)  # import fails
    with pytest.raises(RuntimeError, match="LiteLLM support requires"):
        model_resolver.resolve_adk_model("litellm-proxy:langslice-qwen36-plus")

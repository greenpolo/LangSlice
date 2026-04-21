"""Resolve external model strings into ADK model objects."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

_PROXY_PREFIX = "litellm-proxy:"
_OPENROUTER_PREFIX = "openrouter:"
_OLLAMA_PREFIX = "ollama:"
_MODELS_PREFIX = "models/"

_DEFAULT_PROXY_BASE = "http://127.0.0.1:4000/v1"
_DEFAULT_PROXY_KEY = "sk-langslice-local"
_DEFAULT_OLLAMA_BASE = "http://localhost:11434"
_ENV_OLLAMA_THINK = "LANGSLICE_OLLAMA_THINK"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    cleaned = value.strip()
    return cleaned or default


def _env_bool(name: str) -> bool | None:
    value = _env(name)
    if value is None:
        return None
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "on", "think"}:
        return True
    if lowered in {"0", "false", "no", "off", "nothink"}:
        return False
    raise ValueError(
        f"{name} must be a boolean value like true/false, yes/no, or think/nothink"
    )


def _load_litellm_class() -> Callable[..., object]:
    try:
        from google.adk.models.lite_llm import LiteLlm
    except ImportError as exc:
        raise _missing_litellm_error() from exc
    return LiteLlm


def _load_adk_gemma_class() -> Callable[..., object]:
    from google.adk.models import Gemma

    return Gemma


def _missing_litellm_error() -> RuntimeError:
    return RuntimeError(
        "LiteLLM support requires the litellm package. Install the project "
        "with LiteLLM support before using 'litellm-proxy:', 'openrouter:', "
        "or 'ollama:' model strings."
    )


def resolve_adk_model(model: str | object) -> str | object:
    """Return a model value suitable for ADK ``LlmAgent(model=...)``.

    Native Gemini model strings are returned unchanged. Open-model shorthands
    are converted to ADK's LiteLlm wrapper so the rest of the harness can keep
    accepting ``str | object`` without knowing provider-specific setup details.
    """
    if not isinstance(model, str):
        return model

    stripped = model.strip()
    lowered = stripped.lower()
    if lowered.startswith("gemma-") or lowered.startswith(f"{_MODELS_PREFIX}gemma-"):
        model_id = stripped
        if model_id.lower().startswith(_MODELS_PREFIX):
            model_id = model_id[len(_MODELS_PREFIX):]
        gemma_cls = _load_adk_gemma_class()
        return gemma_cls(model=model_id)

    if lowered.startswith(_PROXY_PREFIX):
        alias = stripped[len(_PROXY_PREFIX):].strip()
        if not alias:
            raise ValueError("litellm-proxy model strings require an alias after ':'")
        try:
            litellm_cls = _load_litellm_class()
        except ImportError as exc:
            raise _missing_litellm_error() from exc
        return litellm_cls(
            model=f"openai/{alias}",
            api_base=_env("LANGSLICE_LITELLM_PROXY_BASE", _DEFAULT_PROXY_BASE),
            api_key=_env("LANGSLICE_LITELLM_PROXY_KEY", _DEFAULT_PROXY_KEY),
        )

    if lowered.startswith(_OPENROUTER_PREFIX):
        model_id = stripped[len(_OPENROUTER_PREFIX):].strip()
        if not model_id:
            raise ValueError("openrouter model strings require a model id after ':'")
        try:
            litellm_cls = _load_litellm_class()
        except ImportError as exc:
            raise _missing_litellm_error() from exc
        kwargs: dict[str, Any] = {}
        key = _env("OPENROUTER_API_KEY")
        if key:
            kwargs["api_key"] = key
        return litellm_cls(model=f"openrouter/{model_id}", **kwargs)

    if lowered.startswith(_OLLAMA_PREFIX):
        model_id = stripped[len(_OLLAMA_PREFIX):].strip()
        if not model_id:
            raise ValueError("ollama model strings require a model tag after ':'")
        try:
            litellm_cls = _load_litellm_class()
        except ImportError as exc:
            raise _missing_litellm_error() from exc
        ollama_kwargs: dict[str, object] = {
            "api_base": _env("LANGSLICE_OLLAMA_BASE", _DEFAULT_OLLAMA_BASE),
        }
        think = _env_bool(_ENV_OLLAMA_THINK)
        if think is not None:
            ollama_kwargs["think"] = think
        return litellm_cls(
            model=f"ollama_chat/{model_id}",
            **ollama_kwargs,
        )

    return model

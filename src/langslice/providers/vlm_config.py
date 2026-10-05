"""The ``gemini-api`` client: one google-genai client, authenticated from the environment.

``LANGSLICE_GENAI_BACKEND`` picks the authentication (default ``ai_studio``,
or ``vertex_adc`` when ``GOOGLE_GENAI_USE_VERTEXAI`` is set):

- ``ai_studio``: ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``).
- ``vertex_api_key``: ``GOOGLE_CLOUD_API_KEY`` (or ``VERTEX_API_KEY``).
- ``vertex_adc``: application default credentials, ``GOOGLE_CLOUD_PROJECT``
  and ``GOOGLE_CLOUD_LOCATION`` (default ``us-central1``).
"""

from __future__ import annotations

import atexit
import importlib
import logging
import os
from collections.abc import Callable
from typing import Protocol, cast

logger = logging.getLogger(__name__)

_BACKEND_AI_STUDIO = "ai_studio"
_BACKEND_VERTEX_API_KEY = "vertex_api_key"
_BACKEND_VERTEX_ADC = "vertex_adc"
_VALID_BACKENDS = {_BACKEND_AI_STUDIO, _BACKEND_VERTEX_API_KEY, _BACKEND_VERTEX_ADC}


class _GenAIModelsProtocol(Protocol):
    def generate_content(self, *, model: str, contents: object, config: object) -> object: ...


class GenAIClientProtocol(Protocol):
    models: _GenAIModelsProtocol

    def close(self) -> None: ...


_client_instance: GenAIClientProtocol | None = None


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    return value.strip() or None


def _env_bool(name: str) -> bool:
    value = _env(name)
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def get_backend() -> str:
    """The authentication backend for the google-genai client."""
    backend = _env("LANGSLICE_GENAI_BACKEND")
    if backend is None:
        return _BACKEND_VERTEX_ADC if _env_bool("GOOGLE_GENAI_USE_VERTEXAI") \
            else _BACKEND_AI_STUDIO
    normalized = backend.lower()
    if normalized not in _VALID_BACKENDS:
        allowed = ", ".join(sorted(_VALID_BACKENDS))
        raise RuntimeError(
            f"Invalid LANGSLICE_GENAI_BACKEND='{backend}'. Expected one of: {allowed}."
        )
    return normalized


def get_api_key() -> str:
    """The API key of the selected backend."""
    backend = get_backend()
    if backend == _BACKEND_AI_STUDIO:
        key = _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")
        if key:
            return key
        raise RuntimeError(
            "AI Studio mode requires GEMINI_API_KEY (or GOOGLE_API_KEY). "
            "Set LANGSLICE_GENAI_BACKEND=ai_studio and configure one of those keys."
        )
    if backend == _BACKEND_VERTEX_API_KEY:
        key = _env("GOOGLE_CLOUD_API_KEY") or _env("VERTEX_API_KEY")
        if key:
            return key
        raise RuntimeError(
            "Vertex API-key mode requires GOOGLE_CLOUD_API_KEY (or VERTEX_API_KEY). "
            "Set LANGSLICE_GENAI_BACKEND=vertex_api_key and configure one of those keys."
        )
    raise RuntimeError(
        "Vertex ADC mode does not use an API key. "
        "Use get_client() with LANGSLICE_GENAI_BACKEND=vertex_adc."
    )


def _vertex_project() -> str:
    project = _env("GOOGLE_CLOUD_PROJECT")
    if project:
        return project
    raise RuntimeError("Vertex mode requires GOOGLE_CLOUD_PROJECT (GCP project id).")


def _vertex_location() -> str:
    return _env("GOOGLE_CLOUD_LOCATION") or "us-central1"


def close_client() -> None:
    """Close the cached client and release its transport."""
    global _client_instance
    if _client_instance is None:
        return
    close = getattr(_client_instance, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:
            logger.debug("Failed to close GenAI client cleanly: %s", exc)
    _client_instance = None


def get_client() -> GenAIClientProtocol:
    """The google-genai client for the selected backend, made on first use."""
    global _client_instance
    if _client_instance is not None:
        return _client_instance
    genai_module = importlib.import_module("google.genai")
    client_cls = cast(Callable[..., GenAIClientProtocol], genai_module.Client)
    backend = get_backend()
    if backend == _BACKEND_AI_STUDIO:
        _client_instance = client_cls(api_key=get_api_key())
    elif backend == _BACKEND_VERTEX_API_KEY:
        _client_instance = client_cls(vertexai=True, api_key=get_api_key())
    else:
        logger.info("Using Vertex ADC auth (project=%s, location=%s)",
                    _vertex_project(), _vertex_location())
        _client_instance = client_cls(vertexai=True, project=_vertex_project(),
                                      location=_vertex_location())
    return _client_instance


atexit.register(close_client)

"""The ``gemini-api`` client: one google-genai client, authenticated from the environment.

``LANGSLICE_GENAI_BACKEND`` picks the authentication (default ``ai_studio``,
or ``vertex_adc`` when ``GOOGLE_GENAI_USE_VERTEXAI`` is set):

- ``ai_studio``: ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``).
- ``vertex_api_key``: ``GOOGLE_CLOUD_API_KEY`` (or ``VERTEX_API_KEY``).
- ``vertex_adc``: application default credentials, ``GOOGLE_CLOUD_PROJECT``
  and ``GOOGLE_CLOUD_LOCATION`` (default ``us-central1``).
"""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Callable
from typing import Protocol, cast

from langslice.providers.client_cache import CachedClient
from langslice.providers.table import provider

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


def read_env(name: str) -> str | None:
    """Return an environment variable's value, stripped.

    Parameters
    ----------
    name
        The environment variable to read.

    Returns
    -------
    str or None
        Its value without surrounding blanks; None when it is unset or blank.
    """
    value = os.environ.get(name)
    if value is None:
        return None
    return value.strip() or None


def _env_bool(name: str) -> bool:
    value = read_env(name)
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def get_backend() -> str:
    """The authentication backend for the google-genai client."""
    backend = read_env("LANGSLICE_GENAI_BACKEND")
    if backend is None:
        return _BACKEND_VERTEX_ADC if _env_bool("GOOGLE_GENAI_USE_VERTEXAI") else _BACKEND_AI_STUDIO
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
        key = next((v for n in provider("gemini-api").key_env if (v := read_env(n))), None)
        if key:
            return key
        raise RuntimeError(
            "AI Studio mode requires GEMINI_API_KEY (or GOOGLE_API_KEY). "
            "Set LANGSLICE_GENAI_BACKEND=ai_studio and configure one of those keys."
        )
    if backend == _BACKEND_VERTEX_API_KEY:
        key = read_env("GOOGLE_CLOUD_API_KEY") or read_env("VERTEX_API_KEY")
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
    project = read_env("GOOGLE_CLOUD_PROJECT")
    if project:
        return project
    raise RuntimeError("Vertex mode requires GOOGLE_CLOUD_PROJECT (GCP project id).")


def _vertex_location() -> str:
    return read_env("GOOGLE_CLOUD_LOCATION") or "us-central1"


def _make_client() -> GenAIClientProtocol:
    genai_module = importlib.import_module("google.genai")
    client_cls = cast(Callable[..., GenAIClientProtocol], genai_module.Client)
    backend = get_backend()
    if backend == _BACKEND_AI_STUDIO:
        return client_cls(api_key=get_api_key())
    if backend == _BACKEND_VERTEX_API_KEY:
        return client_cls(vertexai=True, api_key=get_api_key())
    logger.info(
        "Using Vertex ADC auth (project=%s, location=%s)", _vertex_project(), _vertex_location()
    )
    return client_cls(vertexai=True, project=_vertex_project(), location=_vertex_location())


_CLIENT = CachedClient(_make_client, "GenAI client")


def get_client() -> GenAIClientProtocol:
    """The google-genai client for the selected backend, made on first use."""
    return _CLIENT.get()

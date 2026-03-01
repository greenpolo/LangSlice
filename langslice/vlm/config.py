"""VLM configuration - authentication and model settings."""

import importlib
import logging
import os
from typing import Callable, Protocol, cast

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-3-flash-preview"
THINKING_BUDGET = 8096

_BACKEND_AI_STUDIO = "ai_studio"
_BACKEND_VERTEX_API_KEY = "vertex_api_key"
_BACKEND_VERTEX_ADC = "vertex_adc"
_VALID_BACKENDS = {
    _BACKEND_AI_STUDIO,
    _BACKEND_VERTEX_API_KEY,
    _BACKEND_VERTEX_ADC,
}


class _GenAIModelsProtocol(Protocol):
    def generate_content(self, *, model: str, contents: object, config: object) -> object: ...


class GenAIClientProtocol(Protocol):
    models: _GenAIModelsProtocol


def _load_dotenv() -> None:
    try:
        dotenv_module = importlib.import_module("dotenv")
        load_dotenv = cast(Callable[[], bool], getattr(dotenv_module, "load_dotenv"))
        _ = load_dotenv()
    except ImportError:
        pass


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _env_bool(name: str) -> bool:
    value = _env(name)
    if value is None:
        return False
    return value.lower() in {"1", "true", "yes", "on"}


def get_backend() -> str:
    """Resolve authentication backend for google-genai client."""
    _load_dotenv()

    backend = _env("LANGSLICE_GENAI_BACKEND")
    if backend is None:
        if _env_bool("GOOGLE_GENAI_USE_VERTEXAI"):
            return _BACKEND_VERTEX_ADC
        return _BACKEND_AI_STUDIO

    normalized = backend.lower()
    if normalized not in _VALID_BACKENDS:
        allowed = ", ".join(sorted(_VALID_BACKENDS))
        raise RuntimeError(
            f"Invalid LANGSLICE_GENAI_BACKEND='{backend}'. Expected one of: {allowed}."
        )
    return normalized


def get_api_key() -> str:
    """Load API key for the selected backend mode."""
    _load_dotenv()
    backend = get_backend()

    if backend == _BACKEND_AI_STUDIO:
        key = _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")
        if key:
            return key
        raise RuntimeError(
            "AI Studio mode requires GEMINI_API_KEY (or GOOGLE_API_KEY). "
            + "Set LANGSLICE_GENAI_BACKEND=ai_studio and configure one of those keys."
        )

    if backend == _BACKEND_VERTEX_API_KEY:
        key = _env("GOOGLE_CLOUD_API_KEY") or _env("VERTEX_API_KEY")
        if key:
            return key
        raise RuntimeError(
            "Vertex API-key mode requires GOOGLE_CLOUD_API_KEY (or VERTEX_API_KEY). "
            + "Set LANGSLICE_GENAI_BACKEND=vertex_api_key and configure one of those keys."
        )

    raise RuntimeError(
        "Vertex ADC mode does not use an API key. "
        + "Use get_client() with LANGSLICE_GENAI_BACKEND=vertex_adc."
    )


def _vertex_project() -> str:
    project = _env("GOOGLE_CLOUD_PROJECT")
    if project:
        return project
    raise RuntimeError("Vertex mode requires GOOGLE_CLOUD_PROJECT (GCP project id).")


def _vertex_location() -> str:
    return _env("GOOGLE_CLOUD_LOCATION") or "us-central1"


def get_client() -> GenAIClientProtocol:
    """Create and return a configured GenAI client for the selected backend."""
    _load_dotenv()
    genai_module = importlib.import_module("google.genai")
    client_cls = cast(Callable[..., GenAIClientProtocol], getattr(genai_module, "Client"))
    backend = get_backend()

    if backend == _BACKEND_AI_STUDIO:
        return client_cls(api_key=get_api_key())

    if backend == _BACKEND_VERTEX_API_KEY:
        return client_cls(vertexai=True, api_key=get_api_key())

    logger.info(
        "Using Vertex ADC auth (project=%s, location=%s)",
        _vertex_project(),
        _vertex_location(),
    )
    return client_cls(
        vertexai=True,
        project=_vertex_project(),
        location=_vertex_location(),
    )

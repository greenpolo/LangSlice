"""VLM configuration - authentication and model settings."""

import atexit
import importlib
import logging
import os
from collections.abc import Callable
from typing import Any, Protocol, cast

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-3-flash-preview"
THINKING_LEVEL = "HIGH"
CODE_EXECUTION_ENABLED = True
TEMPERATURE: float = 1.0

REGISTRATION_WORKFLOW_IMAGE_GEN_TWO_SHOT = "image_gen_two_shot"
REGISTRATION_WORKFLOW_COLORED_SEGMENTATION = "colored_segmentation"
REGISTRATION_WORKFLOW_MULTIMODAL_TOOL_LOOP = "multimodal_tool_loop"

AVAILABLE_THINKING_LEVELS: list[tuple[str, str]] = [
    ("Minimal", "MINIMAL"),
    ("Low", "LOW"),
    ("Medium", "MEDIUM"),
    ("High", "HIGH"),
]

_ENV_COUNT_TOKENS = "LANGSLICE_GENAI_COUNT_TOKENS"
_ENV_AP_USE_FILE_API = "LANGSLICE_GENAI_AP_USE_FILE_API"
_ENV_AP_USE_CONTEXT_CACHE = "LANGSLICE_GENAI_AP_USE_CONTEXT_CACHE"
_ENV_AP_USE_INTERACTIONS = "LANGSLICE_GENAI_AP_USE_INTERACTIONS"
_ENV_AP_CACHE_TTL = "LANGSLICE_GENAI_AP_CACHE_TTL"
_ENV_FILE_POLL_TIMEOUT_S = "LANGSLICE_GENAI_FILE_POLL_TIMEOUT_S"
_ENV_TEMPERATURE = "LANGSLICE_GENAI_TEMPERATURE"

AVAILABLE_MODELS: list[str] = [
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-3-pro-image-preview",
    "gemini-3.1-flash-image-preview",
]

CODE_EXECUTION_MODELS: set[str] = {
    "gemini-3-flash-preview",
}

IMAGE_GENERATION_MODELS: set[str] = {
    "gemini-3-pro-image-preview",
    "gemini-3.1-flash-image-preview",
}

IMAGE_MODEL_THINKING_MODELS: set[str] = {
    "gemini-3.1-flash-image-preview",
}

REGISTRATION_WORKFLOW_LABELS: dict[str, str] = {
    REGISTRATION_WORKFLOW_COLORED_SEGMENTATION: "Colored Segmentation",
    REGISTRATION_WORKFLOW_IMAGE_GEN_TWO_SHOT: "Image Gen (2-Shot)",
    REGISTRATION_WORKFLOW_MULTIMODAL_TOOL_LOOP: "Tool Loop",
}


def set_model_name(name: str) -> None:
    """Set active model name at runtime for subsequent requests."""
    globals()["MODEL_NAME"] = name
    logger.info("Model changed to: %s", name)


def set_thinking_level(level: str) -> None:
    """Set active Gemini thinking level at runtime for subsequent requests."""
    normalized = str(level).strip().upper()
    valid_levels = {value for _label, value in AVAILABLE_THINKING_LEVELS}
    if normalized not in valid_levels:
        allowed = ", ".join(sorted(valid_levels))
        raise ValueError(f"Invalid thinking level {level!r}. Expected one of: {allowed}")
    globals()["THINKING_LEVEL"] = normalized
    logger.info("Thinking level changed to: %s", normalized)


def set_temperature(value: float) -> None:
    """Set active generation temperature at runtime for subsequent requests."""
    clamped = max(0.0, min(2.0, float(value)))
    globals()["TEMPERATURE"] = clamped
    logger.info("Temperature changed to: %.2f", clamped)


def set_code_execution_enabled(enabled: bool) -> None:
    """Set whether Gemini code execution should be enabled when supported."""
    globals()["CODE_EXECUTION_ENABLED"] = bool(enabled)
    logger.info("Code execution enabled: %s", bool(enabled))


def is_image_generation_model(model_name: str | None) -> bool:
    """Return True when *model_name* targets an image-generation Gemini model."""
    if model_name is None:
        return False
    return str(model_name).strip() in IMAGE_GENERATION_MODELS


def supports_image_model_thinking(model_name: str | None) -> bool:
    """Return True when the image-gen model supports thinking_config."""
    if model_name is None:
        return False
    return str(model_name).strip() in IMAGE_MODEL_THINKING_MODELS


def supports_code_execution(model_name: str | None) -> bool:
    """Return True when the selected model supports Gemini code execution."""
    if model_name is None:
        return False
    return str(model_name).strip() in CODE_EXECUTION_MODELS


def get_registration_workflow_options(model_name: str | None) -> list[tuple[str, str]]:
    """Return GUI-ready `(label, value)` registration workflow options."""
    if is_image_generation_model(model_name):
        return [
            (
                REGISTRATION_WORKFLOW_LABELS[REGISTRATION_WORKFLOW_COLORED_SEGMENTATION],
                REGISTRATION_WORKFLOW_COLORED_SEGMENTATION,
            ),
            (
                REGISTRATION_WORKFLOW_LABELS[REGISTRATION_WORKFLOW_IMAGE_GEN_TWO_SHOT],
                REGISTRATION_WORKFLOW_IMAGE_GEN_TWO_SHOT,
            ),
        ]
    return [
        (
            REGISTRATION_WORKFLOW_LABELS[REGISTRATION_WORKFLOW_MULTIMODAL_TOOL_LOOP],
            REGISTRATION_WORKFLOW_MULTIMODAL_TOOL_LOOP,
        ),
    ]


def default_registration_workflow(model_name: str | None) -> str:
    """Return the default registration workflow for the selected model."""
    options = get_registration_workflow_options(model_name)
    if not options:
        return REGISTRATION_WORKFLOW_MULTIMODAL_TOOL_LOOP
    return options[0][1]


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

    def generate_content_stream(
        self, *, model: str, contents: object, config: object
    ) -> object: ...

    def count_tokens(
        self, *, model: str, contents: object, config: object | None = None
    ) -> object: ...


class _GenAIFilesProtocol(Protocol):
    def upload(self, *, file: object, config: object | None = None) -> object: ...

    def get(self, *, name: str, config: object | None = None) -> object: ...

    def delete(self, *, name: str, config: object | None = None) -> object: ...


class _GenAICachesProtocol(Protocol):
    def create(self, *, model: str, config: object | None = None) -> object: ...

    def delete(self, *, name: str, config: object | None = None) -> object: ...


class _GenAIInteractionsProtocol(Protocol):
    def create(self, **kwargs: object) -> object: ...


class _GenAIBatchesProtocol(Protocol):
    def create(self, *, model: str, src: object, config: object | None = None) -> object: ...


class GenAIClientProtocol(Protocol):
    models: _GenAIModelsProtocol
    files: _GenAIFilesProtocol
    caches: _GenAICachesProtocol
    interactions: _GenAIInteractionsProtocol
    batches: _GenAIBatchesProtocol

    def close(self) -> None: ...


_client_instance: GenAIClientProtocol | None = None


def _load_dotenv() -> None:
    try:
        dotenv_module = importlib.import_module("dotenv")
        load_dotenv = cast(Callable[[], bool], dotenv_module.load_dotenv)
        _ = load_dotenv()
    except ImportError:
        pass


_load_dotenv()


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


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float env %s=%r; using default %.2f", name, value, default)
        return default


def get_backend() -> str:
    """Resolve authentication backend for google-genai client."""
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


def count_tokens_enabled() -> bool:
    return _env_bool(_ENV_COUNT_TOKENS)


def ap_use_file_api() -> bool:
    return _env_bool(_ENV_AP_USE_FILE_API)


def ap_use_context_cache() -> bool:
    return _env_bool(_ENV_AP_USE_CONTEXT_CACHE)


def ap_use_interactions() -> bool:
    return _env_bool(_ENV_AP_USE_INTERACTIONS)


def ap_cache_ttl() -> str:
    return _env(_ENV_AP_CACHE_TTL) or "3600s"


def file_poll_timeout_s() -> float:
    return _env_float(_ENV_FILE_POLL_TIMEOUT_S, 10.0)


def configured_temperature() -> float:
    return _env_float(_ENV_TEMPERATURE, TEMPERATURE)


TEMPERATURE = configured_temperature()


def supports_file_api() -> bool:
    return get_backend() == _BACKEND_AI_STUDIO


def supports_interactions_api() -> bool:
    return get_backend() == _BACKEND_AI_STUDIO


def supports_batch_api() -> bool:
    return get_backend() == _BACKEND_VERTEX_ADC


def feature_flags() -> dict[str, Any]:
    return {
        "count_tokens_enabled": count_tokens_enabled(),
        "ap_use_file_api": ap_use_file_api(),
        "ap_use_context_cache": ap_use_context_cache(),
        "ap_use_interactions": ap_use_interactions(),
        "supports_file_api": supports_file_api(),
        "supports_interactions_api": supports_interactions_api(),
        "supports_batch_api": supports_batch_api(),
        "ap_cache_ttl": ap_cache_ttl(),
        "file_poll_timeout_s": file_poll_timeout_s(),
        "temperature": configured_temperature(),
        "backend": get_backend(),
    }


def get_api_key() -> str:
    """Load API key for the selected backend mode."""
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


def close_client() -> None:
    """Close cached GenAI client and release transport resources."""
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
    """Create and return a configured GenAI client for the selected backend."""
    global _client_instance
    if _client_instance is not None:
        return _client_instance

    genai_module = importlib.import_module("google.genai")
    client_cls = cast(Callable[..., GenAIClientProtocol], genai_module.Client)
    backend = get_backend()

    if backend == _BACKEND_AI_STUDIO:
        _client_instance = client_cls(api_key=get_api_key())
        return _client_instance

    if backend == _BACKEND_VERTEX_API_KEY:
        _client_instance = client_cls(vertexai=True, api_key=get_api_key())
        return _client_instance

    logger.info(
        "Using Vertex ADC auth (project=%s, location=%s)",
        _vertex_project(),
        _vertex_location(),
    )
    _client_instance = client_cls(
        vertexai=True,
        project=_vertex_project(),
        location=_vertex_location(),
    )
    return _client_instance


def create_batch_client() -> GenAIClientProtocol:
    """Create a dedicated v1 Vertex client for Batch API usage."""
    if not supports_batch_api():
        raise RuntimeError("Batch API is only supported with Vertex backends.")

    genai_module = importlib.import_module("google.genai")
    types_module = importlib.import_module("google.genai.types")
    client_cls = cast(Callable[..., GenAIClientProtocol], genai_module.Client)
    http_options_cls = cast(Callable[..., object], types_module.HttpOptions)
    http_options = http_options_cls(api_version="v1")
    backend = get_backend()

    if backend == _BACKEND_VERTEX_API_KEY:
        return client_cls(vertexai=True, api_key=get_api_key(), http_options=http_options)

    return client_cls(
        vertexai=True,
        project=_vertex_project(),
        location=_vertex_location(),
        http_options=http_options,
    )


atexit.register(close_client)

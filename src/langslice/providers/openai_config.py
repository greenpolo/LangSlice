"""OpenAI-compatible client configuration — endpoint, model, and API key settings."""

import atexit
import importlib
import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_BASE_URL = "http://localhost:11434/v1"
_DEFAULT_API_KEY = "ollama"
_DEFAULT_MODEL = "gemma4:31b"
_DEFAULT_IMAGE_MODEL = "gpt-image-2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_dotenv() -> None:
    try:
        dotenv_module = importlib.import_module("dotenv")
        load_dotenv = cast(Callable[[], bool], dotenv_module.load_dotenv)
        _ = load_dotenv()
    except ImportError:
        pass


_load_dotenv()


def _env(name: str) -> str | None:
    """Return the value of *name* from the environment, stripping whitespace.

    Returns ``None`` when the variable is absent or blank.
    """
    value = os.environ.get(name)
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


# ---------------------------------------------------------------------------
# Accessors for model names
# ---------------------------------------------------------------------------


def get_openai_model() -> str:
    """Return the configured chat/tool model name."""
    return _env("OPENAI_MODEL") or _DEFAULT_MODEL


def get_openai_image_model() -> str:
    """Return the configured image generation model name."""
    return _env("OPENAI_IMAGE_MODEL") or _DEFAULT_IMAGE_MODEL


# ---------------------------------------------------------------------------
# Singleton client cache
# ---------------------------------------------------------------------------

_client_instance: "openai.OpenAI | None" = None  # type: ignore[name-defined]
_image_client_instance: "openai.OpenAI | None" = None  # type: ignore[name-defined]

if TYPE_CHECKING:
    import openai


def get_openai_client() -> "openai.OpenAI":
    """Return a singleton OpenAI-compatible client for chat/tool completions.

    Endpoint and key are read from the environment at first call:
    - ``OPENAI_BASE_URL``  (default: ``http://localhost:11434/v1``)
    - ``OPENAI_API_KEY``   (default: ``ollama``)
    """
    global _client_instance
    if _client_instance is not None:
        return _client_instance

    openai_module = importlib.import_module("openai")
    client_cls = cast(Callable[..., "openai.OpenAI"], openai_module.OpenAI)

    base_url = _env("OPENAI_BASE_URL") or _DEFAULT_BASE_URL
    api_key = _env("OPENAI_API_KEY") or _DEFAULT_API_KEY

    logger.info("Creating OpenAI-compatible text client: base_url=%s", base_url)
    _client_instance = client_cls(base_url=base_url, api_key=api_key)
    return _client_instance


def get_openai_image_client() -> "openai.OpenAI":
    """Return a singleton OpenAI-compatible client for image generation.

    Falls back to the text client's base URL and API key when the image-specific
    environment variables are not set:
    - ``OPENAI_IMAGE_BASE_URL``  (falls back to ``OPENAI_BASE_URL``)
    - ``OPENAI_IMAGE_API_KEY``   (falls back to ``OPENAI_API_KEY``)
    """
    global _image_client_instance
    if _image_client_instance is not None:
        return _image_client_instance

    openai_module = importlib.import_module("openai")
    client_cls = cast(Callable[..., "openai.OpenAI"], openai_module.OpenAI)

    base_url = _env("OPENAI_IMAGE_BASE_URL") or _env("OPENAI_BASE_URL") or _DEFAULT_BASE_URL
    api_key = _env("OPENAI_IMAGE_API_KEY") or _env("OPENAI_API_KEY") or _DEFAULT_API_KEY

    logger.info("Creating OpenAI-compatible image client: base_url=%s", base_url)
    _image_client_instance = client_cls(base_url=base_url, api_key=api_key)
    return _image_client_instance


def close_client() -> None:
    """Close cached OpenAI clients and release HTTP transport resources."""
    global _client_instance, _image_client_instance

    for name, instance in (
        ("text", _client_instance),
        ("image", _image_client_instance),
    ):
        if instance is None:
            continue
        close = getattr(instance, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                logger.debug("Failed to close OpenAI %s client cleanly: %s", name, exc)

    _client_instance = None
    _image_client_instance = None


atexit.register(close_client)

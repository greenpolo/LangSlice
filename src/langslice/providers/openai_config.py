"""The ``openai-api`` image client: endpoint, key and image model.

Endpoint and key come from the environment (loaded by
``doors.api.setup.load_credentials``): ``OPENAI_IMAGE_BASE_URL``, else
``OPENAI_BASE_URL``, else the OpenAI API itself; ``OPENAI_IMAGE_API_KEY``,
else ``OPENAI_API_KEY``. The image model is ``OPENAI_IMAGE_MODEL``, else
``gpt-image-2.5-sunburst`` (GPT Image 2.5, the most capable; ``gpt-image-2.5-flare``
is the fastest).
"""

from __future__ import annotations

import atexit
import importlib
import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import openai

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
#: The GPT Image 2.5 models a host offers on ``openai-api``, the default first.
OPENAI_IMAGE_MODELS: tuple[str, ...] = ("gpt-image-2.5-sunburst", "gpt-image-2.5-flare")
DEFAULT_IMAGE_MODEL = OPENAI_IMAGE_MODELS[0]


def _env(name: str) -> str | None:
    """The environment's *name*, stripped; None when absent or blank."""
    value = os.environ.get(name)
    if value is None:
        return None
    return value.strip() or None


def get_openai_image_model() -> str:
    """The image model ``openai-api`` calls when none is named."""
    return _env("OPENAI_IMAGE_MODEL") or DEFAULT_IMAGE_MODEL


_image_client_instance: openai.OpenAI | None = None


def get_openai_image_client() -> openai.OpenAI:
    """One OpenAI client for image edits, made on first use (see the module
    text for the endpoint and key). ``RuntimeError`` without a key."""
    global _image_client_instance
    if _image_client_instance is not None:
        return _image_client_instance
    api_key = _env("OPENAI_IMAGE_API_KEY") or _env("OPENAI_API_KEY")
    if api_key is None:
        raise RuntimeError("The openai-api image model needs OPENAI_API_KEY (or "
                           "OPENAI_IMAGE_API_KEY), in the environment, a .env file or "
                           "saved by LangSlice's setup.")
    base_url = _env("OPENAI_IMAGE_BASE_URL") or _env("OPENAI_BASE_URL") or DEFAULT_BASE_URL
    client_cls = cast(Callable[..., "openai.OpenAI"], importlib.import_module("openai").OpenAI)
    logger.info("Creating the OpenAI image client: base_url=%s", base_url)
    _image_client_instance = client_cls(base_url=base_url, api_key=api_key)
    return _image_client_instance


def close_client() -> None:
    """Close the cached client and release its HTTP transport."""
    global _image_client_instance
    if _image_client_instance is None:
        return
    close = getattr(_image_client_instance, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:
            logger.debug("Failed to close the OpenAI image client cleanly: %s", exc)
    _image_client_instance = None


atexit.register(close_client)

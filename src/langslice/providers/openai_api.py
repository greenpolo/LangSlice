"""The ``openai-api`` image client: endpoint, key and image model.

Endpoint and key come from the environment (loaded by
``doors.credentials.load_credentials``): ``OPENAI_IMAGE_BASE_URL``, else
``OPENAI_BASE_URL``, else the OpenAI API itself; ``OPENAI_IMAGE_API_KEY``, else
``OPENAI_API_KEY``. The image model is ``OPENAI_IMAGE_MODEL``, else
``gpt-image-2.5-sunburst`` (GPT Image 2.5, the most capable;
``gpt-image-2.5-flare`` is the fastest).
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from langslice.providers.client_cache import CachedClient
from langslice.providers.gemini_api import read_env
from langslice.providers.table import provider

if TYPE_CHECKING:
    import openai

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
_PROVIDER = provider("openai-api")


def get_openai_image_model() -> str:
    """The image model ``openai-api`` calls when none is named."""
    return _PROVIDER.image_model_or_default()


def _api_key() -> str | None:
    for name in (*_PROVIDER.image_key_env, *_PROVIDER.key_env):
        value = read_env(name)
        if value:
            return value
    return None


def _make_image_client() -> openai.OpenAI:
    api_key = _api_key()
    if api_key is None:
        raise RuntimeError(
            "The openai-api image model needs OPENAI_API_KEY (or "
            "OPENAI_IMAGE_API_KEY), in the environment, a .env file or "
            "saved by LangSlice's setup."
        )
    base_url = read_env("OPENAI_IMAGE_BASE_URL") or read_env("OPENAI_BASE_URL") or DEFAULT_BASE_URL
    client_cls = cast(Callable[..., "openai.OpenAI"], importlib.import_module("openai").OpenAI)
    logger.info("Creating the OpenAI image client: base_url=%s", base_url)
    return client_cls(base_url=base_url, api_key=api_key)


_IMAGE_CLIENT = CachedClient(_make_image_client, "OpenAI image client")


def get_openai_image_client() -> openai.OpenAI:
    """One OpenAI client for image edits, made on first use (see the module
    text for the endpoint and key). ``RuntimeError`` without a key."""
    return _IMAGE_CLIENT.get()

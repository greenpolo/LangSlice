"""Provider taxonomy.

A provider is an ACCESS METHOD — a vendor plus how you authenticate — never a
product name. Canonical names:

- ``gemini-api``: Gemini models via a Google API key.
- ``openai-api``: the OpenAI API (or a compatible endpoint,
  ``OPENAI_BASE_URL``) via an API key.
- ``anthropic-api``: Claude models via an Anthropic API key (agent model
  only; transport in ``providers/anthropic_api.py``).
- ``openai-oauth``: OpenAI via a ChatGPT-subscription OAuth login (``langslice
  login``; transport lives in ``providers/oauth/transport.py``).
- ``none``: no image model; FireANTs fits to the stain directly. There is
  nothing to authenticate for tracing, so it needs no transport module.

A new provider gets its canonical name in the provider-free table
:mod:`langslice.core.connections` (re-exported here, so the core compares names
without importing a provider) and its transport in this package; downstream
code compares canonical names only.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from langslice.core.connections import (
    CANONICAL_PROVIDERS,
    CUSTOM_PROVIDER,
    canonical_provider,
)
from langslice.core.fits.trace.request import ImageModel

if TYPE_CHECKING:
    from langslice.core.fits.trace.request import (
        BorderTraceRequest,
        GeneratedSegmentation,
    )

#: Agent models a host may offer on the ``openai-oauth`` lane (host dialogs
#: list these; any other ``openai-oauth/*`` string still works). Defined here,
#: not in ``openai_oauth.py``, so an offline status check can read them
#: without importing the ADK transport.
OPENAI_OAUTH_AGENT_MODELS: tuple[str, ...] = tuple(
    f"openai-oauth/{name}"
    for name in (
        "gpt-6.1-sol", "gpt-6-astra", "gpt-6-sol", "gpt-6-luna",
        "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
    )
)
#: Default agent/review model on the ``openai-oauth`` lane.
OPENAI_OAUTH_DEFAULT_AGENT_MODEL = "openai-oauth/gpt-6.1-sol"
#: Image models on the ``openai-oauth`` lane (GPT Image 2.5: ``sunburst``
#: the most capable, ``flare`` the fastest), and the default. The Codex
#: backend serves its own current image model whatever the request names.
OPENAI_OAUTH_IMAGE_MODELS: tuple[str, ...] = ("gpt-image-2.5-sunburst", "gpt-image-2.5-flare")
OPENAI_OAUTH_DEFAULT_IMAGE_MODEL = OPENAI_OAUTH_IMAGE_MODELS[0]
#: Image models on the ``gemini-api`` lane (Google's image-generation
#: models), and the default: the general-purpose one.
GEMINI_IMAGE_MODELS: tuple[str, ...] = (
    "gemini-3.1-flash-image", "gemini-3.1-flash-lite-image", "gemini-3-pro-image",
)
GEMINI_DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image"


def openai_oauth_credentials_path() -> Path:
    """The ``openai-oauth`` login file (``langslice login`` writes it, mode
    600): ``LANGSLICE_OPENAI_AUTH`` when set (another account for one
    process), else ``~/.langslice/openai_auth.json``. Read at call time."""
    override = os.environ.get("LANGSLICE_OPENAI_AUTH", "").strip()
    return Path(override).expanduser() if override else (
        Path.home() / ".langslice" / "openai_auth.json")


def default_image_model(provider: str) -> str | None:
    """The image model a provider uses when none is named (None for
    ``openai-api``: its transport reads ``OPENAI_IMAGE_MODEL``)."""
    canonical = canonical_provider(provider)
    if canonical == "openai-oauth":
        return OPENAI_OAUTH_DEFAULT_IMAGE_MODEL
    if canonical == "gemini-api":
        return GEMINI_DEFAULT_IMAGE_MODEL
    return None


def resolve_image_model(provider: str, model: str | None = None) -> ImageModel:
    """Resolve a provider name to its image-edit call.

    The transport (``langslice.providers.image_calls``) is imported when the
    call runs, not here, so resolving loads no model client. ``none`` and
    unknown names are refused: there is no image model to call.
    """
    canonical = canonical_provider(provider)
    if canonical == "none":
        raise ValueError("The image correction tool requires an image-model provider")
    if canonical == CUSTOM_PROVIDER:
        raise ValueError("This job's image model is the caller's own (provider 'custom'): "
                         "hand it to the library, langslice.open_job(..., image_model=...)")
    if canonical not in CANONICAL_PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")

    def call(request: BorderTraceRequest) -> GeneratedSegmentation:
        from langslice.providers.image_calls import (
            generate_warped_segmentation_image,
        )

        return generate_warped_segmentation_image(request)

    return ImageModel(provider=canonical, model=model or default_image_model(canonical), call=call)

__all__ = [
    "CANONICAL_PROVIDERS",
    "CUSTOM_PROVIDER",
    "GEMINI_DEFAULT_IMAGE_MODEL",
    "GEMINI_IMAGE_MODELS",
    "OPENAI_OAUTH_AGENT_MODELS",
    "OPENAI_OAUTH_DEFAULT_AGENT_MODEL",
    "OPENAI_OAUTH_DEFAULT_IMAGE_MODEL",
    "OPENAI_OAUTH_IMAGE_MODELS",
    "canonical_provider",
    "default_image_model",
    "openai_oauth_credentials_path",
    "resolve_image_model",
]

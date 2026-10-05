"""Provider taxonomy.

A provider is an ACCESS METHOD — a vendor plus how you authenticate — never a
product name. Canonical names:

- ``gemini-api``: Gemini models via a Google API key.
- ``openai-api``: the OpenAI API (or a compatible endpoint,
  ``OPENAI_BASE_URL``) via an API key.
- ``openai-oauth``: OpenAI via a ChatGPT-subscription OAuth login
  (``langslice login``; transport lives in ``providers/openai_oauth.py``).
- ``none``: no model at all. Registration's model-free backbone registers
  the silhouette prior itself (see ``core/nonlinear/prior.py``); there is nothing
  to authenticate, so it needs no transport module.

A new provider gets its canonical name in the provider-free table
:mod:`langslice.core.provider_names` (re-exported here, so the core compares
names without importing a provider) and its transport in this package;
downstream code compares canonical names only.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from langslice.core.provider_names import CANONICAL_PROVIDERS, CUSTOM_PROVIDER, canonical_provider

if TYPE_CHECKING:
    from langslice.core.nonlinear.types import GeneratedSegmentation, SegmentationGenerationRequest

#: Agent models a host may offer on the ``openai-oauth`` lane (host dialogs
#: list these; any other ``openai-oauth/*`` string still works). Defined here,
#: not in ``openai_oauth.py``, so an offline status check can read them
#: without importing the ADK transport.
OPENAI_OAUTH_AGENT_MODELS: tuple[str, ...] = tuple(
    f"openai-oauth/{name}"
    for name in (
        "gpt-6-astra", "gpt-6-sol", "gpt-6-luna",
        "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
    )
)
#: Default agent/review model on the ``openai-oauth`` lane.
OPENAI_OAUTH_DEFAULT_AGENT_MODEL = "openai-oauth/gpt-5.6-sol"
#: Image models on the ``openai-oauth`` lane, and the default.
OPENAI_OAUTH_IMAGE_MODELS: tuple[str, ...] = ("gpt-image-2",)
OPENAI_OAUTH_DEFAULT_IMAGE_MODEL = "gpt-image-2"
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


#: One image edit: the request in (prompt, images in order, provider, model),
#: one image out.
ImageCall = Callable[["SegmentationGenerationRequest"], "GeneratedSegmentation"]


@dataclass(frozen=True)
class ImageModel:
    """An image model as the operations receive it: resolved, never chosen there.

    ``provider`` is the canonical access method (it selects the prompt's
    GPT or Gemini wording and keys saved replies; ``"custom"`` for a model
    of the caller's own), ``model`` the image model (None: the provider's
    own default), ``call`` the one edit. A door (the toolbox binding, the
    engine, MCP, the CLI, a host plugin) builds this with
    :func:`resolve_image_model`; a test or a script passes its own ``call``.

    The rest makes it a model PROFILE (:mod:`langslice.providers.profiles`):
    ``prompt`` the base prompt written for this model (None: LangSlice's own
    for ``provider``, :func:`langslice.core.nonlinear.prompts.border_correction_tool_prompt`;
    ``{plane}`` in it becomes the section plane), ``photograph_first``
    the attachment order that prompt describes (True: the clean photograph
    is Image 1 and the placed borders Image 2; False: the other way round;
    None: the order of ``provider``'s own prompt), ``profile`` its name, and
    ``tested`` False for a prompt or model the project has not measured:
    every trace made with it is marked ``untested``.
    """

    provider: str
    model: str | None
    call: ImageCall
    prompt: str | None = None
    photograph_first: bool | None = None
    profile: str | None = None
    tested: bool = True


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
    """Resolve a provider name (any accepted spelling) to its image-edit call.

    The transport (``langslice.providers.images``) is imported when the
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

    def call(request: SegmentationGenerationRequest) -> GeneratedSegmentation:
        from langslice.providers.images import generate_warped_segmentation_image

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
    "ImageCall",
    "ImageModel",
    "canonical_provider",
    "default_image_model",
    "openai_oauth_credentials_path",
    "resolve_image_model",
]

"""Provider taxonomy.

A provider is an ACCESS METHOD — a vendor plus how you authenticate — never a
product name. Canonical names:

- ``gemini-api``: Gemini models via a Google API key.
- ``openai-api``: OpenAI-compatible endpoints via an API key (or a custom
  ``--endpoint``).
- ``openai-oauth``: OpenAI via a ChatGPT-subscription OAuth login
  (``langslice login``; transport lives in ``providers/openai_oauth.py``).
- ``none``: no model at all. Registration's model-free backbone registers
  the silhouette prior itself (see ``core/nonlinear/prior.py``); there is nothing
  to authenticate, so it needs no transport module.

Future providers (``anthropic-api``, ``openrouter-api``, ``qwen-api``, ...)
are added HERE and nowhere else; downstream code compares canonical names
only. Legacy spellings ("google", "openai", "chatgpt") resolve here so old
CLIs, saved configs, and sibling repos keep working.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langslice.core.nonlinear.types import GeneratedSegmentation, SegmentationGenerationRequest

CANONICAL_PROVIDERS = ("gemini-api", "openai-api", "openai-oauth", "none")

_ALIASES = {
    "google": "gemini-api",
    "openai": "openai-api",
    "chatgpt": "openai-oauth",
    "flux": "openai-api",
    "openai-compatible": "openai-api",
    "openai_compatible": "openai-api",
}


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


def canonical_provider(name: str) -> str:
    """Resolve any accepted provider spelling to its canonical name."""
    normalized = (name or "").strip().lower()
    return _ALIASES.get(normalized, normalized)


#: One image edit: the request in (prompt, images in order, provider, model),
#: one image out.
ImageCall = Callable[["SegmentationGenerationRequest"], "GeneratedSegmentation"]


@dataclass(frozen=True)
class ImageModel:
    """An image model as the operations receive it: resolved, never chosen there.

    ``provider`` is the canonical access method (it selects the prompt's
    GPT or Gemini wording and keys saved replies), ``model`` the image model
    (None: the provider's own default), ``call`` the one edit. A door (the
    toolbox binding, the engine, MCP, the CLI, a host plugin) builds this with
    :func:`resolve_image_model`; a test or a script passes its own ``call``.
    """

    provider: str
    model: str | None
    call: ImageCall


def default_image_model(provider: str) -> str | None:
    """The image model a provider uses when none is named (None: the transport's)."""
    if canonical_provider(provider) == "openai-oauth":
        return OPENAI_OAUTH_DEFAULT_IMAGE_MODEL
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
    if canonical not in CANONICAL_PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")

    def call(request: SegmentationGenerationRequest) -> GeneratedSegmentation:
        from langslice.providers.images import generate_warped_segmentation_image

        return generate_warped_segmentation_image(request)

    return ImageModel(provider=canonical, model=model or default_image_model(canonical), call=call)

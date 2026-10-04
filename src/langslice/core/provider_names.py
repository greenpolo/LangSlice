"""The provider names: canonical access methods and their accepted spellings.

Plain strings, no provider imported, so the core can compare names (the
GPT or Gemini prompt wording, a spec's validation) without the providers
layer. :mod:`langslice.providers.registry` re-exports these and documents
what each name means; a new provider's name and aliases go here, its
transport in the providers package.
"""

from __future__ import annotations

CANONICAL_PROVIDERS = ("gemini-api", "openai-api", "openai-oauth", "none")

#: A job whose image model the caller hands the library itself (an object or
#: function of its own, ``langslice.image_model``): accepted in a job spec,
#: never resolved from a name (nothing to log in to).
CUSTOM_PROVIDER = "custom"

_ALIASES = {
    "google": "gemini-api",
    "openai": "openai-api",
    "chatgpt": "openai-oauth",
    "flux": "openai-api",
    "openai-compatible": "openai-api",
    "openai_compatible": "openai-api",
}


def canonical_provider(name: str) -> str:
    """Resolve any accepted provider spelling to its canonical name."""
    normalized = (name or "").strip().lower()
    return _ALIASES.get(normalized, normalized)

"""The provider names: the canonical access methods.

Plain strings, no provider imported, so the core can compare names (the
GPT or Gemini prompt wording, a spec's validation) without the providers
layer. :mod:`langslice.providers.registry` re-exports these and documents
what each name means; a new provider's name goes here, its
transport in the providers package.
"""

from __future__ import annotations

CANONICAL_PROVIDERS = ("gemini-api", "openai-api", "openai-oauth", "none")

#: A job whose image model the caller hands the library itself (an object or
#: function of its own, ``langslice.image_model``): accepted in a job spec,
#: never resolved from a name (nothing to log in to).
CUSTOM_PROVIDER = "custom"

def canonical_provider(name: str) -> str:
    """A provider name, trimmed and lower-cased (canonical names only)."""
    return (name or "").strip().lower()

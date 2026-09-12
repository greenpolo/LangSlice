"""Provider taxonomy.

A provider is an ACCESS METHOD — a vendor plus how you authenticate — never a
product name. Canonical names:

- ``gemini-api``: Gemini models via a Google API key.
- ``openai-api``: OpenAI-compatible endpoints via an API key (or a custom
  ``--endpoint``).
- ``openai-oauth``: OpenAI via a ChatGPT-subscription OAuth login
  (``langslice login``; transport lives in ``providers/openai_oauth.py``).
- ``none``: no model at all. Registration's model-free backbone registers
  the silhouette prior itself (see ``nonlinear/prior.py``); there is nothing
  to authenticate, so it needs no transport module.

Future providers (``anthropic-api``, ``openrouter-api``, ``qwen-api``, ...)
are added HERE and nowhere else; downstream code compares canonical names
only. Legacy spellings ("google", "openai", "chatgpt") resolve here so old
CLIs, saved configs, and sibling repos keep working.
"""

from __future__ import annotations

CANONICAL_PROVIDERS = ("gemini-api", "openai-api", "openai-oauth", "none")

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

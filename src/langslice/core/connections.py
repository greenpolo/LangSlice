"""The provider names: the canonical access methods.

Plain strings, no provider imported, so the core can compare names (the GPT or
Gemini prompt wording, a spec's validation) without the providers layer.
:mod:`langslice.providers.connections` re-exports these and documents what each
name means; a new provider's name goes here, its transport in the providers
package.
"""

from __future__ import annotations

CANONICAL_PROVIDERS = ("gemini-api", "openai-api", "openai-oauth", "none")

CUSTOM_PROVIDER = "custom"

def canonical_provider(name: str) -> str:
    """A provider name, trimmed and lower-cased (canonical names only)."""
    return (name or "").strip().lower()

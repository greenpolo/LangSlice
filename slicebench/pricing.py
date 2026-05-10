"""Per-model token pricing for SliceBench cost estimation.

Returns a `TracePricing` instance from the harness, or None for free/local
models (Gemma open weights, llama.cpp local). Cost is computed from the
recorder's `usage_totals` via `estimate_cost_usd`.

Pricing values are dollars per **million tokens**. Public pricing for preview
models drifts; update this table when Google publishes official rates.
Last refreshed: 2026-05-07.
"""

from __future__ import annotations

from langslice_harness.harness.estimation.trace_collection import TracePricing

# Gemini 3 Flash Preview — placeholder pricing (Google hasn't published official
# rates as of 2026-05). Modeled after Gemini 2.5 Flash ($0.30 / $2.50). Update
# when the public rate card lands.
GEMINI_3_FLASH_PREVIEW_PRICING = TracePricing(
    input_per_million=0.30,
    output_per_million=2.50,
    cached_input_per_million=0.075,
)

# Gemini 3.1 Pro Preview — already defined upstream, mirrored here for clarity.
GEMINI_31_PRO_PREVIEW_PRICING = TracePricing(
    input_per_million=2.0,
    output_per_million=12.0,
    cached_input_per_million=0.20,
)

_PRICING_BY_MODEL: dict[str, TracePricing] = {
    "gemini-3-flash-preview": GEMINI_3_FLASH_PREVIEW_PRICING,
    "gemini-3.1-pro-preview": GEMINI_31_PRO_PREVIEW_PRICING,
    "gemini-3-pro-image-preview": GEMINI_31_PRO_PREVIEW_PRICING,
}


def pricing_for(model: str) -> TracePricing | None:
    """Return per-million-token pricing for a model, or None for free/local."""
    if not isinstance(model, str):
        return None
    if model in _PRICING_BY_MODEL:
        return _PRICING_BY_MODEL[model]
    # Open-weight Gemma served free via AI Studio + local llama.cpp / ollama / litellm-proxy
    # all return None — no cost is charged.
    return None

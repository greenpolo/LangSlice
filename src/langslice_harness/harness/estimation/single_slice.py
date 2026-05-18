"""Builder for the single-slice position-estimation LlmAgent."""

from __future__ import annotations

from typing import Any

from google.adk.agents import LlmAgent
from google.genai import types

from langslice_harness.atlas.space import Plane
from langslice_harness.harness.estimation.prompts import build_single_slice_prompt
from langslice_harness.harness.estimation.tools import (
    fetch_atlas,
    submit_estimate,
)
from langslice_harness.harness.estimation.validators import gate_submit_tool


def _is_native_gemini_string(model: str | object) -> bool:
    if not isinstance(model, str):
        return False
    lowered = model.strip().lower()
    return lowered.startswith("gemini-") or lowered.startswith("models/gemini-")


def _wants_thinking_prefix(model: str | object) -> bool:
    """Return True for langslice-ft adapters trained with thinking_signature mode.

    The thinking-signature SFT corpus rewrites every system prompt to
    ``f"<|think|>{prompt}"`` (see ``models/langslice-gemma-4/training/sft/
    render.py``); the model learned to emit a concise ``<|channel>thought
    \\n…<channel|>`` block per turn ONLY when it sees that literal token
    early in its context. Without it, the model goes off-distribution into
    verbose freeform reasoning and overflows the 8192 context window before
    submitting (observed: 39% session-failure rate on slicebench tiny).
    The chat template's ``enable_thinking`` Jinja branch is unrelated —
    it's a separate template variable that the renderer never set.
    """
    if not isinstance(model, str):
        return False
    lowered = model.strip().lower()
    return "langslice-ft" in lowered


def build_single_slice_agent(
    *,
    atlas_name: str,
    plane: Plane,
    species: str,
    pos_lo: float,
    pos_hi: float,
    model: str | object = "gemini-3-flash-preview",
    temperature: float = 1.0,
    media_resolution: str = "MEDIA_RESOLUTION_MEDIUM",
    thinking_config: object | None = None,
) -> LlmAgent:
    """Construct the single-slice LlmAgent with atlas fetch + final submit tools."""
    config_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": 4000,
    }
    if thinking_config is not None:
        config_kwargs["thinking_config"] = thinking_config
    config_kwargs["media_resolution"] = media_resolution
    if _is_native_gemini_string(model):
        config_kwargs["http_options"] = types.HttpOptions(
            # ``attempts`` includes the original request. Match or exceed the
            # legacy /main loop's 4 total attempts for transient Gemini 429s.
            retry_options=types.HttpRetryOptions(initial_delay=1, attempts=5)
        )

    instruction = build_single_slice_prompt(
        atlas_name=atlas_name, plane=plane,
        pos_lo=pos_lo, pos_hi=pos_hi, species=species,
    )
    if _wants_thinking_prefix(model):
        instruction = f"<|think|>{instruction}"

    return LlmAgent(
        model=model,  # type: ignore[arg-type]
        name="single_slice_position_estimator",
        instruction=instruction,
        tools=[fetch_atlas, submit_estimate],
        generate_content_config=types.GenerateContentConfig(**config_kwargs),
        # gate_submit_tool intentionally NOT wired here for the langslice-ft
        # (Gemma 4) model. The training corpus (round_2_corpus_capped8.jsonl)
        # is ~52% direct-submit traces with zero fetch_atlas calls, so a small
        # model trained on it has learned to submit immediately when confident.
        # The broad+narrow sweep gate is a Gemini-shaped optimization that
        # kneecaps a model trained without it (rejected submits inflate context
        # until 8192 overflow). Re-enable only when serving Gemini, not Gemma-ft.
    )

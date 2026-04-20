"""Builder for the single-slice position-estimation LlmAgent."""

from __future__ import annotations

from typing import Any

from google.adk.agents import LlmAgent
from google.genai import types

from langslice.atlas.space import Plane
from langslice.harness.estimation.prompts import build_single_slice_prompt
from langslice.harness.estimation.tools import (
    fetch_atlas,
    side_by_side,
    submit_estimate,
    zoom,
)
from langslice.harness.estimation.validators import gate_submit_tool


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
    """Construct the single-slice LlmAgent with all four tools wired."""
    config_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": 4000,
    }
    if thinking_config is not None:
        config_kwargs["thinking_config"] = thinking_config
    config_kwargs["media_resolution"] = media_resolution

    return LlmAgent(
        model=model,  # type: ignore[arg-type]
        name="single_slice_position_estimator",
        instruction=build_single_slice_prompt(
            atlas_name=atlas_name, plane=plane,
            pos_lo=pos_lo, pos_hi=pos_hi, species=species,
        ),
        tools=[fetch_atlas, zoom, side_by_side, submit_estimate],
        generate_content_config=types.GenerateContentConfig(**config_kwargs),
        before_tool_callback=gate_submit_tool,
    )

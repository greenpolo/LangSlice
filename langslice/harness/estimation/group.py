"""Builder for the multi-slice group position-estimation LlmAgent."""

from __future__ import annotations

from typing import Any

from google.adk.agents import LlmAgent
from google.genai import types

from langslice.atlas.space import Plane
from langslice.harness.estimation.prompts import build_group_prompt
from langslice.harness.estimation.tools import (
    fetch_atlas,
    side_by_side,
    submit_group_estimate,
    zoom,
)
from langslice.harness.estimation.validators import gate_submit_tool


def build_group_agent(
    *,
    atlas_name: str,
    plane: Plane,
    species: str,
    pos_lo: float,
    pos_hi: float,
    n_slices: int,
    interval_mm: float,
    thickness_um: int,
    model: str | object = "gemini-3-flash-preview",
    temperature: float = 1.0,
    media_resolution: str = "MEDIA_RESOLUTION_MEDIUM",
    thinking_config: object | None = None,
) -> LlmAgent:
    """Construct the multi-slice group LlmAgent with all four tools wired."""
    config_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": 8000,
    }
    if thinking_config is not None:
        config_kwargs["thinking_config"] = thinking_config
    config_kwargs["media_resolution"] = media_resolution

    return LlmAgent(
        model=model,  # type: ignore[arg-type]
        name="group_position_estimator",
        instruction=build_group_prompt(
            atlas_name=atlas_name, plane=plane,
            pos_lo=pos_lo, pos_hi=pos_hi, species=species,
            n_slices=n_slices, interval_mm=interval_mm,
            thickness_um=thickness_um,
        ),
        tools=[fetch_atlas, zoom, side_by_side, submit_group_estimate],
        generate_content_config=types.GenerateContentConfig(**config_kwargs),
        before_tool_callback=gate_submit_tool,
    )

"""Builder for the registration review LlmAgent."""

from __future__ import annotations

from typing import Any

from google.adk.agents import LlmAgent
from google.genai import types

from langslice_harness.harness.registration.tools import (
    confirm_registration,
    make_generate_registration_candidate_tool,
)


def _is_native_gemini_string(model: str | object) -> bool:
    if not isinstance(model, str):
        return False
    lowered = model.strip().lower()
    return lowered.startswith("gemini-") or lowered.startswith("models/gemini-")


def build_registration_review_agent(
    *,
    atlas_name: str,
    position_mm: float,
    image_provider: str,
    image_model: str | None,
    model: str | object,
    openai_image_route: str = "images",
    max_candidates: int = 3,
    temperature: float = 0.2,
    media_resolution: str = "MEDIA_RESOLUTION_MEDIUM",
    thinking_config: object | None = None,
    on_progress: object | None = None,
    on_trace: object | None = None,
) -> LlmAgent:
    """Construct the ADK review agent for dense registration candidates."""
    config_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": 4000,
        "media_resolution": media_resolution,
    }
    if thinking_config is not None:
        config_kwargs["thinking_config"] = thinking_config
    if _is_native_gemini_string(model):
        config_kwargs["http_options"] = types.HttpOptions(
            retry_options=types.HttpRetryOptions(initial_delay=1, attempts=5)
        )

    instruction = (
        f"You are reviewing an image-gen registration for {atlas_name} "
        f"at AP {position_mm:.3f} mm.\n\n"
        f"Use `generate_registration_candidate` to create up to {max_candidates} preview "
        "bundles. Each candidate includes the model-generated segmentation, the "
        "Elastix-warped atlas, and the warped atlas borders overlaid on the histology "
        "slice. Inspect the images carefully, revise the prompt if needed, and call "
        "`confirm_registration` only when you have selected the best candidate.\n\n"
        f"Current image provider: {image_provider}. "
        f"Current image model: {image_model or 'default'}. "
        f"OpenAI image route: {openai_image_route}."
    )

    generate_registration_candidate = make_generate_registration_candidate_tool(
        on_progress=on_progress,
        on_trace=on_trace,
    )

    return LlmAgent(
        model=model,  # type: ignore[arg-type]
        name="registration_review_agent",
        instruction=instruction,
        tools=[generate_registration_candidate, confirm_registration],
        generate_content_config=types.GenerateContentConfig(**config_kwargs),
    )

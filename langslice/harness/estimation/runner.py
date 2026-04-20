"""Runner wrapper: drives nudge-on-no-tool-call, retry-with-fresh-session, and max-iteration cap."""

from __future__ import annotations

import io
import logging
from typing import Any

from google.adk.runners import InMemoryRunner
from google.genai import types
from PIL import Image

from langslice.atlas.core import (
    get_in_plane_long_edge,
    get_position_range_mm,
    load_atlas,
)
from langslice.atlas.space import Plane
from langslice.harness.estimation._types import PositionResult
from langslice.harness.estimation.session import (
    ARTIFACT_TARGET,
    build_initial_state,
)
from langslice.harness.estimation.single_slice import build_single_slice_agent
from langslice.image_prep import (
    adaptive_preprocess,
    normalize_image,
    prepare_image_for_vlm,
)

logger = logging.getLogger(__name__)

_APP_NAME = "langslice"
_USER_ID = "langslice-user"

_DEFAULT_MAX_ITERATIONS_SINGLE = 20
_DEFAULT_MAX_RETRIES = 2

_NUDGE_BROAD = (
    "Please continue. Call `fetch_atlas` with widely spaced positions "
    "(e.g., [2, 4, 6, 8, 10]) to find the correct neighborhood."
)
_NUDGE_NARROW = (
    "Please narrow down. Call `fetch_atlas` with tightly spaced positions "
    "around your best candidate (e.g., [4.0, 4.2, 4.4, 4.6, 4.8])."
)
_NUDGE_VERIFY = (
    "Please continue. Verify your candidate by checking nearby positions "
    "with `fetch_atlas`, or call `submit_estimate` if confident."
)


def _pick_nudge(state: dict[str, Any]) -> str:
    if not state.get("saw_broad_sweep"):
        return _NUDGE_BROAD
    if not state.get("saw_narrow_sweep"):
        return _NUDGE_NARROW
    return _NUDGE_VERIFY


def _encode_target_part(
    image: Image.Image, atlas_long_edge: int, *, apply_clahe: bool = True,
) -> types.Part:
    """Normalize, downscale, optionally CLAHE, and encode as a JPEG ``types.Part``.

    Order matches ``eval_group.py`` and ``whole_brain.estimation_agents``:
    normalize → downscale-to-atlas-long-edge → CLAHE. CLAHE is on by default;
    the pre-ADK Flash baseline (0.14-0.25mm MAE on M01) was measured with it.
    """
    normalized = normalize_image(image)
    prepped = prepare_image_for_vlm(normalized, max_long_edge=atlas_long_edge).image
    if apply_clahe:
        prepped = adaptive_preprocess(prepped)
    buf = io.BytesIO()
    prepped.convert("RGB").save(buf, format="JPEG", quality=85)
    return types.Part.from_bytes(mime_type="image/jpeg", data=buf.getvalue())


async def run_single_slice_session(
    *,
    image: Image.Image,
    atlas_name: str,
    plane: Plane = "coronal",
    model: str | object = "gemini-3-flash-preview",
    species: str | None = None,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS_SINGLE,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    temperature: float = 1.0,
    thinking_level: str = "MEDIUM",
    apply_clahe: bool = True,
) -> PositionResult:
    """Drive a single-slice position-estimation session to completion.

    Seeds the target image as an artifact, runs the ADK single-slice agent,
    counts function-call events against ``max_iterations``, and retries with
    a fresh session up to ``max_retries`` times if the agent does not submit.
    Falls back to the atlas midpoint if all retries are exhausted.
    """
    atlas = load_atlas(atlas_name)
    pos_lo, pos_hi = get_position_range_mm(atlas, plane=plane)
    atlas_long_edge = get_in_plane_long_edge(atlas, plane=plane)

    # MEDIUM thinking is the validated sweet spot for Flash (0.14mm MAE on M01).
    # Only wire a thinking_config for string-named Gemini models; LiteLlm
    # wrappers drive their own thinking knobs.
    thinking_cfg: object | None = None
    if isinstance(model, str):
        from langslice import vlm_config
        thinking_cfg = vlm_config.build_thinking_config(model, thinking_level)

    species_val = species or str(atlas.metadata.get("species", "mouse"))
    agent = build_single_slice_agent(
        atlas_name=atlas_name,
        plane=plane,
        species=species_val,
        pos_lo=pos_lo,
        pos_hi=pos_hi,
        model=model,
        temperature=temperature,
        thinking_config=thinking_cfg,
    )

    runner = InMemoryRunner(agent=agent, app_name=_APP_NAME)
    # InMemoryRunner always wires in-memory services, but the base class types
    # them Optional; assert for the type checker.
    assert runner.artifact_service is not None
    assert runner.session_service is not None

    initial_state_template = build_initial_state(
        atlas_name=atlas_name,
        plane=plane,
        pos_lo=pos_lo,
        pos_hi=pos_hi,
        n_slices=1,
        interval_mm=0.0,
        thickness_um=50,
        max_iterations=max_iterations,
    )

    # Encode target image once as a Part for reuse across retries.
    target_part = _encode_target_part(image, atlas_long_edge, apply_clahe=apply_clahe)

    for attempt in range(max_retries):
        session_id = f"single_slice_attempt_{attempt}"
        await runner.session_service.create_session(
            app_name=_APP_NAME,
            user_id=_USER_ID,
            session_id=session_id,
            state=dict(initial_state_template),
        )

        # Seed the target image as an artifact so tools that reference
        # "target" can load it (e.g., zoom, side_by_side in later phases).
        await runner.artifact_service.save_artifact(
            app_name=_APP_NAME,
            user_id=_USER_ID,
            session_id=session_id,
            filename=ARTIFACT_TARGET,
            artifact=target_part,
        )

        new_message = types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=f"Target slice (artifact key: '{ARTIFACT_TARGET}'):"
                ),
                target_part,
                types.Part.from_text(text="Determine its position in the atlas."),
            ],
        )

        tool_call_count = 0
        capped = False
        async for event in runner.run_async(
            user_id=_USER_ID,
            session_id=session_id,
            new_message=new_message,
        ):
            fcs = event.get_function_calls() or []
            tool_call_count += len(fcs)
            if tool_call_count > max_iterations:
                logger.warning(
                    "Hit max_iterations=%d on attempt %d; forcing end.",
                    max_iterations,
                    attempt + 1,
                )
                capped = True
                break

            # Re-read session state after each event (mutated by tools).
            current = await runner.session_service.get_session(
                app_name=_APP_NAME,
                user_id=_USER_ID,
                session_id=session_id,
            )
            if current is not None and current.state.get("result") is not None:
                break

        final = await runner.session_service.get_session(
            app_name=_APP_NAME,
            user_id=_USER_ID,
            session_id=session_id,
        )
        if final is not None and final.state.get("result") is not None:
            result = final.state["result"]
            return PositionResult(
                position_mm=float(result["position_mm"]),
                reasoning=str(result["reasoning"]),
            )

        # Nudge text is captured for possible future reuse (e.g., continuing
        # the same session rather than restarting). The plan specifies a
        # fresh-session retry, so we just log the choice and loop.
        final_state = final.state if final is not None else initial_state_template
        nudge = _pick_nudge(final_state)
        logger.info(
            "Attempt %d did not submit (capped=%s); nudge=%r; retrying with fresh session.",
            attempt + 1,
            capped,
            nudge,
        )

    mid = (pos_lo + pos_hi) / 2.0
    logger.warning(
        "All %d retries exhausted; falling back to %.2f mm midpoint.",
        max_retries,
        mid,
    )
    return PositionResult(
        position_mm=mid,
        reasoning=(
            "Agent did not submit within iteration+retry budget; "
            "fell back to atlas midpoint."
        ),
    )

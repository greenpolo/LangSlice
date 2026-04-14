"""Single-slice AP estimation via OpenAI-compatible Chat Completions.

Port of ``langslice.estimation.google.ap_single_slice`` adapted for the
Chat Completions API.  Uses ``get_openai_client()`` / ``get_openai_model()``
from :mod:`langslice.openai_config` and sends images inline as base64
data URIs instead of the Gemini File API.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import Any, cast

from PIL import Image

from langslice.agent_trace import (
    image_part_from_pil,
    json_part,
    model_event,
    runtime_event,
)
from langslice.estimation._types import APResult
from langslice.estimation.debug import write_debug_artifacts
from langslice.estimation.openai.common import (
    _APLoopState,
    _build_image_content,
    _build_text_content,
    _emit_trace,
    _extract_text,
    _extract_usage,
    _format_usage,
    _get_position_range_lazy,
    _history_message_count,
    _image_to_base64,
    _image_to_bytes,
    _load_atlas_lazy,
    _to_float,
)
from langslice.estimation.openai.tool_definitions import (
    _build_nudge_text,
    _extract_function_calls,
    _process_ap_function_calls,
    _tool_declarations,
)
from langslice.image_prep import normalize_image, prepare_image_for_vlm
from langslice.openai_config import get_openai_client, get_openai_model
from langslice.retry import retry_with_backoff

logger = logging.getLogger(__name__)


def estimate_position(
    image: Image.Image,
    atlas_name: str,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
    debug_dir: str | None = None,
    max_iterations: int = 20,
    show_borders: bool = False,
    anatomy_hints: str = "",
    model_name: str | None = None,
    send_individually: bool = True,
    atlas_resolution: int = 1024,
) -> APResult:
    """Agentic AP estimation using tool-use with self-correction.

    The model receives tools to explore the atlas freely:
    - fetch_atlas: view coronal sections at specific AP positions
    - submit_estimate: declare the final answer

    Uses the OpenAI Chat Completions API with manually-managed conversation
    history.  Images are sent inline as base64 data URIs.  The model runs
    until it submits or hits *max_iterations*.

    Set ``LANGSLICE_VLM_DEBUG_DIR`` to save all artifacts for review.
    """
    from datetime import datetime

    def _progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        logger.info(msg)

    client = get_openai_client()
    atlas = _load_atlas_lazy(atlas_name)
    pos_lo, pos_hi = _get_position_range_lazy(atlas)

    atlas_obj_meta = cast(Any, atlas)
    species = atlas_obj_meta.metadata.get("species", "mouse")

    # --- Prepare target image ---
    target_normalized = normalize_image(image)
    target_prep = prepare_image_for_vlm(target_normalized)
    target_prepared = target_prep.image
    target_bytes = _image_to_bytes(target_prepared)
    target_h = target_prepared.height
    target_info: dict[str, Any] = {
        "original_width": target_prep.original_size[0],
        "original_height": target_prep.original_size[1],
        "width": target_prepared.width,
        "height": target_prepared.height,
        "mode": target_prepared.mode,
        "jpeg_bytes": len(target_bytes),
        "vlm_scale_factor": round(target_prep.scale_factor, 6),
    }
    if target_prep.downsampled:
        _progress(
            "Target image resized for VLM budget: "
            f"{target_prep.original_size[0]}x{target_prep.original_size[1]}px -> "
            f"{target_prepared.width}x{target_prepared.height}px "
            f"(scale={target_prep.scale_factor:.4f})"
        )
    _progress(
        "Target image prepared: "
        f"{target_prepared.width}x{target_prepared.height}px, "
        f"mode={target_prepared.mode}, jpeg_bytes={len(target_bytes)}"
    )

    # --- Debug artifact setup ---
    debug_root = debug_dir if debug_dir is not None else os.environ.get("LANGSLICE_VLM_DEBUG_DIR")
    run_dir: str | None = None
    if debug_root:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_atlas = atlas_name.replace("/", "_").replace("\\", "_")
        run_dir = os.path.join(debug_root, f"{timestamp}_{safe_atlas}")
        os.makedirs(run_dir, exist_ok=True)
        target_prepared.save(os.path.join(run_dir, "target.jpg"), quality=85)
        _progress(f"Debug artifacts -> {run_dir}")

    target_b64 = _image_to_base64(target_prepared)

    _emit_trace(
        on_trace,
        runtime_event(
            stage="ap",
            title="Prepared AP estimation inputs",
            summary=(
                f"Target image "
                f"{target_prepared.width}x{target_prepared.height}px "
                f"prepared for Chat Completions"
            ),
            parts=[
                image_part_from_pil(
                    target_prepared,
                    label="Target slice",
                    image_bytes=target_bytes,
                    path=os.path.join(run_dir, "target.jpg") if run_dir else None,
                    metadata={
                        "transport": "base64_inline",
                        "vlm_scale_factor": target_info["vlm_scale_factor"],
                    },
                )
            ],
            metadata=target_info,
        ),
    )

    # --- System prompt ---
    system_instruction = (
        "You are an expert neuroanatomist. You are given a histology brain slice image "
        "and must determine its Anterior-Posterior (AP) position within a reference atlas. "
        "The coordinate system is: 0.0 mm is the extreme anterior edge (e.g. olfactory bulb), "
        "while larger mm values move posterior toward the cerebellum and brainstem. "
        f"Atlas: {atlas_name} ({species}). "
        f"Valid AP range: {pos_lo:.2f}\u2013{pos_hi:.2f} mm. "
        "0.0 mm = anterior (olfactory bulb); higher mm = posterior.\n\n"
        "You have tools to fetch atlas reference images at any AP coordinate "
        "and submit your estimate.\n\n"
        f"{anatomy_hints}"
        "RECOMMENDED STRATEGY:\n"
        "1. Call `fetch_atlas` with broadly spaced positions (e.g., [2, 4, 6, 8, 10]) "
        "   to find the general region.\n"
        "2. Call `fetch_atlas` with tighter positions around your best match.\n"
        "3. Call `fetch_atlas` with very fine positions (e.g., 0.1-0.2mm apart) to pinpoint.\n"
        "4. Verify neighbors, then submit.\n\n"
        "You can request up to 8 positions per call, spaced however you like \u2014 "
        "cluster them densely around a candidate, or spread them widely to explore.\n\n"
        "IMPORTANT: If at ANY point the atlas images don't look similar to the "
        "target, DO NOT continue narrowing in the same area. Go back and try a "
        "completely different region. It is better to restart your search than to "
        "commit to a wrong neighborhood.\n\n"
        "Think carefully before each tool call, but always follow up with an action."
    )

    max_iterations = max(1, int(max_iterations))
    effective_model = model_name or get_openai_model()
    tools = _tool_declarations()

    target_info["input_transport"] = "base64_inline"
    _emit_trace(
        on_trace,
        runtime_event(
            stage="ap",
            title="Initial AP request queued",
            summary="Target slice encoded as base64 inline",
            parts=[
                image_part_from_pil(
                    target_prepared,
                    label="Target slice sent to model",
                    image_bytes=target_bytes,
                    path=os.path.join(run_dir, "target.jpg") if run_dir else None,
                    metadata={
                        "transport": "base64_inline",
                        "vlm_scale_factor": target_info["vlm_scale_factor"],
                    },
                ),
                json_part(
                    {
                        "transport": "base64_inline",
                        "max_iterations": max_iterations,
                    },
                    label="Request context",
                ),
            ],
            metadata={
                "transport": "base64_inline",
                "max_iterations": max_iterations,
            },
        ),
    )

    # --- Build initial user message with image ---
    initial_user_message: dict[str, Any] = {
        "role": "user",
        "content": [
            _build_text_content(
                "Here is the target brain slice. Determine its AP position in the atlas."
            ),
            _build_image_content(target_b64),
        ],
    }

    # --- Main Chat Completions loop (with one retry on failure) ---
    state = _APLoopState(max_iterations=max_iterations)

    for attempt in range(2):
        if attempt == 1:
            _progress("Retrying AP estimation (attempt 2/2, fresh history)...")
            state = _APLoopState(max_iterations=max_iterations)

        messages: list[Any] = [
            {"role": "system", "content": system_instruction},
            initial_user_message,
        ]

        _progress(
            f"Starting agentic estimation (max {max_iterations} tool calls"
            + (f", attempt {attempt + 1}/2" if attempt > 0 else "")
            + ")..."
        )

        for iteration in range(max_iterations):
            request_metrics = _history_message_count(messages)
            turn_metric: dict[str, object] = {
                "iteration": iteration + 1,
                "request": request_metrics,
                "mode": "chat_completions",
            }

            if on_progress:
                msg_count = sum(
                    request_metrics.get(r, 0)
                    for r in ("system", "user", "assistant", "tool")
                )
                img_count = request_metrics.get("image_parts", 0)
                _progress(
                    f"chat.completions turn {iteration + 1}: "
                    f"sending {msg_count} messages, "
                    f"{img_count} images"
                )

            started_at = time.perf_counter()
            response = retry_with_backoff(
                lambda _m=messages: client.chat.completions.create(
                    model=effective_model,
                    messages=_m,
                    tools=tools,
                    max_tokens=4000,
                ),
                request_label=f"AP turn {iteration + 1}",
                on_progress=_progress,
            )
            turn_metric["wall_time_s"] = round(time.perf_counter() - started_at, 3)
            usage = _extract_usage(response)
            turn_metric["usage_metadata"] = usage
            state.turn_metrics.append(turn_metric)

            _progress(
                f"Turn {iteration + 1} completed in {turn_metric['wall_time_s']}s; "
                f"{_format_usage(usage)}"
            )

            # Emit trace for model text output
            text_output = _extract_text(response)
            if text_output:
                _emit_trace(
                    on_trace,
                    model_event(
                        stage="ap",
                        title=f"Model turn {iteration + 1}",
                        summary="Model returned text before the next tool step",
                        parts=[
                            json_part(
                                [text_output],
                                label="Model text",
                                collapsible=True,
                            )
                        ],
                        metadata={"iteration": iteration + 1, **usage},
                    ),
                )

            # Extract function calls
            function_calls, text_preview = _extract_function_calls(response)

            if not function_calls:
                # Append assistant message to history
                messages.append(response.choices[0].message)

                # Detect thought leaks: high completion tokens with no
                # tool call means the model is writing verbose text
                # instead of reasoning internally.
                completion_tokens = usage.get("completion_tokens", 0)
                is_thought_leak = completion_tokens > 1000
                if is_thought_leak:
                    _progress(
                        f"Thought leak detected ({completion_tokens} "
                        f"completion tokens). Nudging."
                    )
                    nudge = (
                        "You wrote a long text response instead of calling a tool. "
                        "Do NOT repeat this. Use your internal reasoning, then call "
                        "a tool: `fetch_atlas` or `submit_estimate`."
                    )
                else:
                    if text_preview and on_progress:
                        _progress(f"Agent reasoning/text: {text_preview[:200]}...")
                    nudge = _build_nudge_text(state)
                messages.append({"role": "user", "content": nudge})
                continue

            # Append assistant message (with tool_calls) to history
            messages.append(response.choices[0].message)

            # Process tool calls — returns (messages_to_add, estimate_submitted)
            result_messages, estimate_submitted = _process_ap_function_calls(
                function_calls,
                iteration=iteration,
                atlas=atlas,
                pos_lo=pos_lo,
                pos_hi=pos_hi,
                target_h=target_h,
                run_dir=run_dir,
                state=state,
                target_image=target_prepared,
                show_borders=show_borders,
                send_individually=send_individually,
                atlas_resolution=atlas_resolution,
                on_progress=_progress,
                on_trace=on_trace,
            )
            messages.extend(result_messages)

            if estimate_submitted:
                break

        if state.estimate_result:
            break
        if attempt == 0:
            _progress("Warning: no estimate in attempt 1. Retrying with fresh history...")

    # --- Finalize result ---
    final_pos: float
    final_reasoning: str
    if state.estimate_result:
        final_pos = _to_float(state.estimate_result.get("position_mm"), (pos_lo + pos_hi) / 2)
        final_reasoning = str(state.estimate_result["reasoning"])
    else:
        final_pos = (pos_lo + pos_hi) / 2
        final_reasoning = "Agent did not submit an estimate within the iteration limit."
        _progress(f"Warning: Agent did not submit. Falling back to midpoint: {final_pos:.2f}mm")

    _progress(
        f"Final position estimated: {final_pos:.2f} mm "
        f"({state.images_fetched} atlas images fetched)"
    )
    _emit_trace(
        on_trace,
        runtime_event(
            stage="ap",
            title="AP estimation completed",
            summary=f"Final position {final_pos:.2f} mm",
            parts=[
                json_part(
                    {
                        "position_mm": final_pos,
                        "reasoning": final_reasoning,
                        "images_fetched": state.images_fetched,
                    },
                    label="AP result",
                )
            ],
            metadata={"images_fetched": state.images_fetched},
        ),
    )

    if run_dir:
        write_debug_artifacts(
            run_dir=run_dir,
            atlas_name=atlas_name,
            mode_used="chat_completions",
            target_info=target_info,
            feature_flags={},
            max_iterations=max_iterations,
            state=state,
            final_pos=final_pos,
            final_reasoning=final_reasoning,
            cache_name=None,
            interaction_trace=[],
            history=[],
            on_progress=_progress,
        )

    return APResult(
        position_mm=final_pos,
        reasoning=final_reasoning,
        debug_dir=run_dir,
    )

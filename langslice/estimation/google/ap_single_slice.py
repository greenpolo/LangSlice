"""VLM-based brain slice estimation using Gemini generate_content API."""

import io
import logging
import time
from collections.abc import Callable
from typing import Any, cast

from google.genai import types
from PIL import Image

import langslice.vlm_config as vlm_config
from langslice.agent_trace import (
    image_part_from_pil,
    json_part,
    model_event,
    runtime_event,
)
from langslice.estimation._types import APResult
from langslice.estimation.google.common import (
    _APLoopState,
    _emit_trace,
    _extract_text_and_thoughts,
    _extract_usage_metadata,
    _format_usage_metadata,
    _get_position_range_lazy,
    _history_metrics,
    _image_to_bytes,
    _load_atlas_lazy,
    _to_float,
    _wait_for_uploaded_file,
)
from langslice.image_prep import normalize_image, prepare_image_for_vlm
from langslice.retry import (
    format_elapsed_seconds as _format_elapsed_seconds,  # noqa: F401 — re-exported
)
from langslice.retry import (
    retry_with_backoff,
)
from langslice.retry import (
    run_with_progress_heartbeat as _run_with_progress_heartbeat,  # noqa: F401 — re-exported
)
from langslice.vlm_config import get_client

logger = logging.getLogger(__name__)

_RESAMPLE_LANCZOS = Image.Resampling.LANCZOS


# ---------------------------------------------------------------------------
# Tool handler functions — extracted to estimator_tools.py
# ---------------------------------------------------------------------------
# Debug artifact writing — extracted to estimator_debug.py
from langslice.estimation.debug import write_debug_artifacts  # noqa: E402
from langslice.estimation.google.tool_definitions import (  # noqa: E402
    _build_nudge_text,
    _extract_function_calls,
    _process_ap_function_calls,
    _tool_declarations,
)


def estimate_position(
    image: Image.Image,
    atlas_name: str,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
    debug_dir: str | None = None,
    max_iterations: int = 20,
    media_resolution: str = "ultra_high",
    show_borders: bool = False,
    anatomy_hints: str = "",
    model_name: str | None = None,
    send_individually: bool = True,
) -> APResult:
    """Agentic AP estimation using tool-use with self-correction.

    The model receives tools to explore the atlas freely:
    - fetch_atlas: view coronal sections at specific AP positions
    - submit_estimate: declare the final answer

    Uses the Gemini generate_content API with manually-managed conversation
    history. The model runs until it submits or hits max iterations.

    Set ``LANGSLICE_VLM_DEBUG_DIR`` to save all artifacts for review.
    """
    import os
    from datetime import datetime

    def _progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        logger.info(msg)

    client = get_client()
    atlas = _load_atlas_lazy(atlas_name)
    pos_lo, pos_hi = _get_position_range_lazy(atlas)

    atlas_obj_meta = cast(Any, atlas)
    species = atlas_obj_meta.metadata.get("species", "mouse")

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
            "Target image resized for Gemini budget: "
            f"{target_prep.original_size[0]}x{target_prep.original_size[1]}px -> "
            f"{target_prepared.width}x{target_prepared.height}px "
            f"(scale={target_prep.scale_factor:.4f})"
        )
    _progress(
        "Target image prepared for Gemini: "
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

    _emit_trace(
        on_trace,
        runtime_event(
            stage="ap",
            title="Prepared AP estimation inputs",
            summary=(
                f"Target image "
                f"{target_prepared.width}x{target_prepared.height}px "
                f"prepared for Gemini"
            ),
            parts=[
                image_part_from_pil(
                    target_prepared,
                    label="Target slice",
                    image_bytes=target_bytes,
                    path=os.path.join(run_dir, "target.jpg") if run_dir else None,
                    metadata={
                        "transport": "file_api",
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
        "You can request up to 8 positions per call, spaced however you like — "
        "cluster them densely around a candidate, or spread them widely to explore.\n\n"
        "IMPORTANT: If at ANY point the atlas images don't look similar to the "
        "target, DO NOT continue narrowing in the same area. Go back and try a "
        "completely different region. It is better to restart your search than to "
        "commit to a wrong neighborhood.\n\n"
        "Think carefully before each tool call, but always follow up with an action."
    )

    feature_flags = vlm_config.feature_flags()
    temperature = vlm_config.TEMPERATURE
    thinking_level = vlm_config.THINKING_LEVEL

    max_iterations = max(1, int(max_iterations))
    state = _APLoopState(max_iterations=max_iterations)
    uploaded_files: list[Any] = []

    try:
        # --- Upload target image via File API ---
        target_buf = io.BytesIO(target_bytes)
        target_file = client.files.upload(
            file=target_buf,
            config=types.UploadFileConfig(
                mime_type="image/jpeg",
                display_name="target_slice",
            ),
        )
        target_file_name = getattr(target_file, "name", None)
        if not isinstance(target_file_name, str) or not target_file_name:
            raise RuntimeError("Gemini File API upload for target slice returned no file name")
        uploaded_files.append(target_file)

        _wait_for_uploaded_file(
            client,
            file_name=target_file_name,
            timeout_s=30.0,
            on_progress=_progress,
        )
        target_uri = getattr(target_file, "uri", None)
        if not isinstance(target_uri, str) or not target_uri:
            raise RuntimeError("Gemini File API upload for target slice returned no URI")

        target_info["input_transport"] = "file_api"
        _emit_trace(
            on_trace,
            runtime_event(
                stage="ap",
                title="Initial AP request queued",
                summary="Target slice uploaded via File API",
                parts=[
                    image_part_from_pil(
                        target_prepared,
                        label="Target slice sent to Gemini",
                        image_bytes=target_bytes,
                        path=os.path.join(run_dir, "target.jpg") if run_dir else None,
                        metadata={
                            "transport": "file_api",
                            "vlm_scale_factor": target_info["vlm_scale_factor"],
                        },
                    ),
                    json_part(
                        {
                            "transport": "file_api",
                            "temperature": temperature,
                            "thinking_level": thinking_level,
                            "max_iterations": max_iterations,
                        },
                        label="Request context",
                    ),
                ],
                metadata={
                    "transport": "file_api",
                    "temperature": temperature,
                    "thinking_level": thinking_level,
                    "max_iterations": max_iterations,
                },
            ),
        )

        # --- Build initial content (reused across retry attempts) ---
        initial_content = types.Content(role="user", parts=[
            types.Part.from_text(
                text="Here is the target brain slice. Determine its AP position in the atlas."
            ),
            types.Part.from_uri(file_uri=target_uri, mime_type="image/jpeg"),
        ])

        # --- Tool declarations ---
        tools = _tool_declarations()

        # --- Config ---
        effective_model = model_name or vlm_config.MODEL_NAME
        thinking_cfg = vlm_config.build_thinking_config(
            effective_model, thinking_level
        )
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=4000,
            thinking_config=cast(Any, thinking_cfg),
            tools=cast(Any, tools),
        )

        # --- Main generate_content loop (with one retry on failure) ---
        for attempt in range(2):
            if attempt == 1:
                _progress("Retrying AP estimation (attempt 2/2, fresh history)...")
                state = _APLoopState(max_iterations=max_iterations)

            contents: list[types.Content] = [initial_content]

            _progress(
                f"Starting agentic estimation (max {max_iterations} tool calls"
                + (f", attempt {attempt + 1}/2" if attempt > 0 else "")
                + ")..."
            )

            for iteration in range(max_iterations):
                request_metrics = _history_metrics(contents)
                turn_metric: dict[str, object] = {
                    "iteration": iteration + 1,
                    "request": request_metrics,
                    "mode": "generate_content",
                }

                if on_progress:
                    part_count = request_metrics['part_count']
                    img_count = request_metrics['image_parts']
                    img_bytes = request_metrics['image_bytes']
                    _progress(
                        f"generate_content turn {iteration + 1}: "
                        f"sending {part_count} parts, "
                        f"{img_count} images ({img_bytes} bytes)"
                    )

                started_at = time.perf_counter()
                response = retry_with_backoff(
                    lambda _c=contents: client.models.generate_content(
                        model=effective_model,
                        contents=_c,
                        config=config,
                    ),
                    request_label=f"AP turn {iteration + 1}",
                    on_progress=_progress,
                )
                turn_metric["wall_time_s"] = round(time.perf_counter() - started_at, 3)
                usage_metadata = _extract_usage_metadata(response)
                turn_metric["usage_metadata"] = usage_metadata
                state.turn_metrics.append(turn_metric)

                _progress(
                    f"Turn {iteration + 1} completed in {turn_metric['wall_time_s']}s; "
                    f"{_format_usage_metadata(usage_metadata)}"
                )

                # Append model response to history
                model_content = response.candidates[0].content
                contents.append(model_content)

                # Emit trace for model text/thought outputs
                text_outputs, thought_outputs = _extract_text_and_thoughts(model_content)
                if text_outputs or thought_outputs:
                    trace_parts: list[dict[str, object]] = []
                    if thought_outputs:
                        trace_parts.append(
                            json_part(thought_outputs, label="Reasoning summary", collapsible=True)
                        )
                    if text_outputs:
                        trace_parts.append(
                            json_part(
                                text_outputs,
                                label="Model text",
                                collapsible=True,
                            )
                        )
                    _emit_trace(
                        on_trace,
                        model_event(
                            stage="ap",
                            title=f"Model turn {iteration + 1}",
                            summary="Model returned text before the next tool step",
                            parts=trace_parts,
                            metadata={"iteration": iteration + 1, **usage_metadata},
                        ),
                    )

                # Extract function calls
                function_calls, text_preview = _extract_function_calls(response)

                if not function_calls:
                    # Detect thought leaks: high candidate tokens with no
                    # tool call means the model is writing verbose text
                    # instead of reasoning internally.
                    candidate_tokens = int(
                        usage_metadata.get("candidates_token_count", 0)
                    )
                    is_thought_leak = candidate_tokens > 1000
                    if is_thought_leak:
                        _progress(
                            f"Thought leak detected ({candidate_tokens} "
                            f"candidate tokens). Nudging."
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
                    contents.append(types.Content(role="user", parts=[
                        types.Part.from_text(text=nudge),
                    ]))
                    continue

                # Process tool calls — returns list[Part]
                result_parts = _process_ap_function_calls(
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
                    on_progress=_progress,
                    on_trace=on_trace,
                )
                if state.estimate_result:
                    break

                # Split parts: function_response Parts go in role='tool',
                # text/image Parts go in role='user'
                tool_parts = [p for p in result_parts if getattr(p, 'function_response', None)]
                other_parts = [p for p in result_parts if not getattr(p, 'function_response', None)]
                if tool_parts:
                    contents.append(types.Content(role="tool", parts=tool_parts))
                if other_parts:
                    contents.append(types.Content(role="user", parts=other_parts))

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

        feature_flags["effective_ap_use_file_api"] = True

        if run_dir:
            write_debug_artifacts(
                run_dir=run_dir,
                atlas_name=atlas_name,
                mode_used="generate_content",
                target_info=target_info,
                feature_flags=feature_flags,
                max_iterations=max_iterations,
                state=state,
                final_pos=final_pos,
                final_reasoning=final_reasoning,
                cache_name=None,
                interaction_trace=[],
                history=contents,
                on_progress=_progress,
            )

        return APResult(
            position_mm=final_pos,
            reasoning=final_reasoning,
            debug_dir=run_dir,
        )
    finally:
        for f in uploaded_files:
            try:
                client.files.delete(name=f.name)
            except Exception as exc:
                logger.warning("Failed to delete Gemini file %s: %s", getattr(f, "name", "?"), exc)

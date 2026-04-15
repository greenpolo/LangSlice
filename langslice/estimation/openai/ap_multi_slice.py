"""Multi-slice group AP estimation via OpenAI-compatible Responses API.

Port of ``langslice.estimation.google.ap_multi_slice`` adapted for the
Responses API.  Uses ``get_openai_client()`` / ``get_openai_model()``
from :mod:`langslice.openai_config` and sends images inline as base64
data URIs instead of the Gemini File API.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any, cast

from PIL import Image

from langslice.agent_trace import (
    image_part_from_pil,
    json_part,
    model_event,
    runtime_event,
    tool_call_event,
    tool_result_event,
)
from langslice.atlas.core import get_coronal_long_edge
from langslice.estimation._tool_logic import _validate_submit_group_estimate
from langslice.estimation._types import APResult, MultiSliceResult
from langslice.estimation.openai.common import (
    _build_image_content,
    _build_text_content,
    _emit_trace,
    _extract_text,
    _extract_usage,
    _format_usage,
    _get_position_range_lazy,
    _GroupLoopState,
    _history_message_count,
    _image_to_base64,
    _image_to_bytes,
    _load_atlas_lazy,
)
from langslice.estimation.openai.tool_definitions import (
    _build_nudge_text,
    _extract_function_calls,
    _group_tool_declarations,
    _handle_fetch_atlas,
)
from langslice.image_prep import normalize_image, prepare_image_for_vlm
from langslice.openai_config import get_openai_client, get_openai_model
from langslice.retry import retry_with_backoff

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


def _process_group_function_calls(
    function_calls: list[dict[str, object]],
    *,
    iteration: int,
    atlas: object,
    pos_lo: float,
    pos_hi: float,
    run_dir: str | None,
    state: _GroupLoopState,
    show_borders: bool = False,
    send_individually: bool = False,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Process group tool calls and return Chat Completions messages.

    Returns ``(messages, estimate_submitted)`` where *messages* is a list of
    input items (``function_call_output`` results and optional ``role='user'``
    image messages) to append to the conversation input.

    Handles ``fetch_atlas`` (delegated to ``_handle_fetch_atlas``) and
    ``submit_group_estimate`` (with full validation: count, range,
    monotonicity, interval plausibility, and sweep gates).
    """
    result_messages: list[dict[str, Any]] = []
    estimate_submitted = False

    def _append_tool_response(
        *,
        call_id: str,
        response: dict[str, object],
    ) -> None:
        result_messages.append({
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(response),
        })

    for call in function_calls:
        name = str(call.get("name", ""))
        args_obj = call.get("args", {})
        args = args_obj if isinstance(args_obj, dict) else {}
        call_id = str(call.get("call_id", ""))

        _emit_trace(
            on_trace,
            tool_call_event(
                stage="ap_group",
                tool_name=name,
                args=args,
                iteration=iteration + 1,
            ),
        )

        if on_progress:
            on_progress(f"Tool call [{iteration + 1}]: {name}({args})")

        if name == "fetch_atlas":
            fetch_items, _fn = _handle_fetch_atlas(
                args=args,
                tool_call_id=call_id,
                pos_lo=pos_lo,
                pos_hi=pos_hi,
                atlas=atlas,
                state=state,
                iteration=iteration,
                run_dir=run_dir,
                show_borders=show_borders,
                send_individually=send_individually,
                target_image=None,
                stage="ap_group",
                on_progress=on_progress,
                on_trace=on_trace,
            )
            result_messages.extend(fetch_items)

        elif name == "submit_group_estimate":
            positions_list = args.get("positions_mm", [])
            if not isinstance(positions_list, list):
                positions_list = []
            est_reasoning = str(args.get("reasoning", ""))
            error_response, est_positions, log_reason = _validate_submit_group_estimate(
                state=state,
                positions_list=positions_list,
                pos_lo=pos_lo,
                pos_hi=pos_hi,
                iteration=iteration,
            )
            if error_response is not None and log_reason is not None:
                _append_tool_response(
                    call_id=call_id,
                    response=error_response,
                )
                state.reasoning_log.append({
                    "iteration": iteration + 1,
                    "tool": name,
                    "args": args,
                    "result": log_reason,
                })
                if log_reason.startswith("Rejected: wrong count"):
                    summary = "Submit rejected: wrong count"
                elif log_reason.startswith("Rejected: out of range"):
                    summary = "Submit rejected: positions out of atlas range"
                elif log_reason == "Rejected: positions not strictly increasing":
                    summary = "Submit rejected: positions not monotonic"
                elif log_reason.startswith("Rejected: bad intervals"):
                    summary = "Submit rejected: interval plausibility failed"
                elif log_reason == "Rejected: no broad sweep yet":
                    summary = "Submit rejected: broad sweep required"
                else:
                    summary = "Submit rejected: narrow sweep required"
                _emit_trace(
                    on_trace,
                    tool_result_event(
                        stage="ap_group",
                        tool_name=name,
                        summary=summary,
                        parts=[json_part(args, label="Rejected submit")],
                        metadata={"iteration": iteration + 1, "status": "rejected"},
                    ),
                )
                continue

            est_positions = cast(list[float], est_positions)
            state.estimate_result = {
                "positions_mm": est_positions,
                "reasoning": est_reasoning,
            }
            state.reasoning_log.append({
                "iteration": iteration + 1,
                "tool": name,
                "args": args,
                "result": f"Submitted {len(est_positions)} positions",
            })
            if on_progress:
                pos_str = ", ".join(f"{p:.2f}" for p in est_positions)
                on_progress(f"Group estimate submitted: [{pos_str}] mm")
            _emit_trace(
                on_trace,
                tool_result_event(
                    stage="ap_group",
                    tool_name=name,
                    summary=f"Submitted {len(est_positions)} positions",
                    parts=[
                        json_part(state.estimate_result, label="Group estimate")
                    ],
                    metadata={
                        "iteration": iteration + 1,
                        "status": "accepted",
                    },
                ),
            )
            estimate_submitted = True
            # Acknowledge every tool call with a response (Chat Completions
            # requires a tool result for every tool_call in the assistant msg)
            _append_tool_response(
                call_id=call_id,
                response={
                    "status": "ok",
                    "positions_mm": est_positions,
                    "message": "Group estimate accepted.",
                },
            )

        else:
            _append_tool_response(
                call_id=call_id,
                response={"status": "error", "error": f"Unknown tool: {name}"},
            )

    return result_messages, estimate_submitted


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def estimate_group(
    images: list[Image.Image],
    atlas_name: str,
    interval_um: int,
    thickness_um: int = 50,
    *,
    max_iterations: int = 25,
    model_name: str | None = None,
    show_borders: bool = False,
    send_individually: bool = False,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
    debug_dir: str | None = None,
) -> MultiSliceResult:
    """Estimate AP positions for a group of consecutive brain slices.

    The model sees all slices simultaneously with known section spacing,
    enabling geometric reasoning that single-slice estimation cannot.

    Uses the OpenAI Chat Completions API with manually-managed conversation
    history.  Images are sent inline as base64 data URIs instead of the
    Gemini File API.  The model runs until it submits or hits
    *max_iterations*.

    Parameters
    ----------
    images
        2-8 slice images in anterior-to-posterior order.
    atlas_name
        BrainGlobe atlas name.
    interval_um
        Section interval in microns (center-to-center).
    thickness_um
        Slice thickness in microns.
    max_iterations
        Maximum Chat Completions turns.
    model_name
        Override the default model.
    """
    n_slices = len(images)
    if not 2 <= n_slices <= 8:
        raise ValueError(f"Expected 2-8 slices, got {n_slices}")

    interval_mm = interval_um / 1000.0

    def _progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        logger.info(msg)

    client = get_openai_client()
    atlas = _load_atlas_lazy(atlas_name)
    pos_lo, pos_hi = _get_position_range_lazy(atlas)
    atlas_long_edge = get_coronal_long_edge(atlas)

    atlas_obj_meta = cast(Any, atlas)
    species = atlas_obj_meta.metadata.get("species", "mouse")

    # --- Prepare all slice images ---
    prepared_images: list[Image.Image] = []
    image_bytes_list: list[bytes] = []
    for i, img in enumerate(images):
        normalized = normalize_image(img)
        prep = prepare_image_for_vlm(normalized, max_long_edge=atlas_long_edge)
        prepared = prep.image
        img_bytes = _image_to_bytes(prepared)
        prepared_images.append(prepared)
        image_bytes_list.append(img_bytes)
        _progress(
            f"Slice {i + 1}: {prepared.width}x{prepared.height}px, "
            f"{len(img_bytes)} bytes"
        )

    # --- Debug setup ---
    debug_root = (
        debug_dir
        if debug_dir is not None
        else os.environ.get("LANGSLICE_VLM_DEBUG_DIR")
    )
    run_dir: str | None = None
    if debug_root:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_atlas = atlas_name.replace("/", "_").replace("\\", "_")
        run_dir = os.path.join(
            debug_root, f"{timestamp}_{safe_atlas}_group{n_slices}"
        )
        os.makedirs(run_dir, exist_ok=True)
        for i, img in enumerate(prepared_images):
            img.save(
                os.path.join(run_dir, f"slice_{i + 1:02d}.jpg"), quality=85
            )
        _progress(f"Debug artifacts -> {run_dir}")

    _emit_trace(
        on_trace,
        runtime_event(
            stage="ap_group",
            title="Prepared multi-slice estimation inputs",
            summary=f"{n_slices} slices, interval {interval_um}µm",
            parts=[
                image_part_from_pil(
                    img,
                    label=f"Slice {i + 1}",
                    image_bytes=image_bytes_list[i],
                    path=(
                        os.path.join(run_dir, f"slice_{i + 1:02d}.jpg")
                        if run_dir
                        else None
                    ),
                    metadata={"transport": "base64_inline"},
                )
                for i, img in enumerate(prepared_images)
            ],
            metadata={
                "n_slices": n_slices,
                "interval_um": interval_um,
                "thickness_um": thickness_um,
                "transport": "base64_inline",
            },
        ),
    )

    # --- System prompt ---
    system_instruction = (
        "You are an expert neuroanatomist. You are given "
        f"{n_slices} consecutive histology brain slice images from the same "
        f"brain, ordered anterior-to-posterior (Slice 1 is most anterior, "
        f"Slice {n_slices} is most posterior).\n\n"
        "Section parameters:\n"
        f"- Slice thickness: {thickness_um} µm\n"
        f"- Section interval: {interval_um} µm (center-to-center)\n"
        f"- Consecutive slices are approximately {interval_mm:.3f} mm apart\n\n"
        f"Atlas: {atlas_name} ({species}). "
        f"Valid AP range: {pos_lo:.2f}–{pos_hi:.2f} mm. "
        "0.0 mm = anterior (olfactory bulb); higher = posterior.\n\n"
        "Your task: determine the AP position of EACH slice in the reference "
        "atlas.\n\n"
        "STRATEGY:\n"
        "1. IN YOUR FIRST RESPONSE, do two things simultaneously:\n"
        "   a) Examine all slices and describe 2-3 prominent anatomical "
        "landmarks you observe in Slice 1 and Slice "
        f"{n_slices} (e.g., 'anterior commissure visible', 'hippocampus "
        "forming', 'corpus callosum is thick', 'ventricles are large').\n"
        "   b) Based on those landmarks, estimate the approximate AP range "
        "and call `fetch_atlas` with broadly spaced positions (e.g., "
        "[2, 4, 6, 8, 10]) to find the general area.\n"
        "2. VERIFY: Compare the atlas sections from your broad sweep against "
        "the landmarks you described. Do they match? If NOT, try a "
        "completely different AP range.\n"
        "3. Narrow down by comparing atlas slices with your input slices.\n"
        f"4. Use the known {interval_mm:.3f} mm interval as a constraint — "
        "once you confidently match ANY slice, you can derive approximate "
        "positions for the others.\n"
        "5. Fine-tune individual positions by comparing each slice to nearby "
        "atlas sections.\n"
        f"6. Submit all {n_slices} positions via `submit_group_estimate`.\n\n"
        "IMPORTANT: The interval is approximate — actual spacing may vary "
        "slightly due to tissue preparation. Use it as a guide, not an "
        "absolute constraint.\n\n"
        "If atlas images don't look similar to your slices, DO NOT continue "
        "narrowing in the same area. Go back and try a completely different "
        "region."
    )

    max_iterations = max(1, int(max_iterations))
    effective_model = model_name or get_openai_model()
    tools = _group_tool_declarations(n_slices)

    # --- Build initial user message with all slice images ---
    initial_content_parts: list[dict[str, Any]] = [
        _build_text_content(
            f"Here are {n_slices} consecutive brain slices, ordered "
            f"anterior to posterior. Section interval: {interval_um} "
            f"µm ({interval_mm:.3f} mm). Slice thickness: "
            f"{thickness_um} µm. Determine the AP position of "
            "each slice in the atlas."
        ),
    ]
    for i, img in enumerate(prepared_images):
        initial_content_parts.append(_build_text_content(f"Slice {i + 1}:"))
        initial_content_parts.append(
            _build_image_content(_image_to_base64(img))
        )

    initial_user_message: dict[str, Any] = {
        "role": "user",
        "content": initial_content_parts,
    }

    # --- Main Chat Completions loop (with one retry on failure) ---
    state = _GroupLoopState(
        n_slices=n_slices,
        max_iterations=max_iterations,
        interval_mm=interval_mm,
    )

    for attempt in range(2):
        if attempt == 1:
            _progress(
                "Retrying group estimation (attempt 2/2, fresh input)..."
            )
            state = _GroupLoopState(
                n_slices=n_slices,
                max_iterations=max_iterations,
                interval_mm=interval_mm,
            )

        input_list: list[Any] = [initial_user_message]

        _progress(
            f"Starting group estimation "
            f"({n_slices} slices, max {max_iterations} turns"
            + (f", attempt {attempt + 1}/2" if attempt > 0 else "")
            + ")..."
        )

        for iteration in range(max_iterations):
            request_metrics = _history_message_count(input_list)
            turn_metric: dict[str, object] = {
                "iteration": iteration + 1,
                "request": request_metrics,
                "mode": "responses_api",
            }

            if on_progress:
                msg_count = sum(
                    request_metrics.get(r, 0)
                    for r in ("system", "user", "assistant", "tool")
                )
                img_count = request_metrics.get("image_parts", 0)
                _progress(
                    f"responses.create turn {iteration + 1}: "
                    f"sending {msg_count} items, "
                    f"{img_count} images"
                )

            started_at = time.perf_counter()
            response = retry_with_backoff(
                lambda _inp=input_list: client.responses.create(
                    model=effective_model,
                    instructions=system_instruction,
                    input=cast(Any, _inp),
                    tools=cast(Any, tools),
                ),
                request_label=f"Group estimation turn {iteration + 1}",
                on_progress=_progress,
            )
            turn_metric["wall_time_s"] = round(
                time.perf_counter() - started_at, 3
            )
            usage = _extract_usage(response)
            turn_metric["usage_metadata"] = usage
            state.turn_metrics.append(turn_metric)

            _progress(
                f"Turn {iteration + 1} completed in "
                f"{turn_metric['wall_time_s']}s; "
                f"{_format_usage(usage)}"
            )

            # Emit trace for model text output
            text_output = _extract_text(response)
            if text_output:
                _emit_trace(
                    on_trace,
                    model_event(
                        stage="ap_group",
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
                # Append model output to input for next turn
                input_list += response.output

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
                        "You wrote a long text response instead of calling "
                        "a tool. Do NOT repeat this. Use your internal "
                        "reasoning, then call a tool: `fetch_atlas` or "
                        "`submit_group_estimate`."
                    )
                else:
                    if text_preview and on_progress:
                        _progress(
                            f"Agent reasoning/text: {text_preview[:200]}..."
                        )
                    nudge = _build_nudge_text(
                        state, submit_tool="submit_group_estimate"
                    )
                input_list.append({"role": "user", "content": nudge})
                continue

            # Append model output (including function_call items) to input
            input_list += response.output

            # Process tool calls — returns (items_to_add, estimate_submitted)
            result_items, estimate_submitted = _process_group_function_calls(
                function_calls,
                iteration=iteration,
                atlas=atlas,
                pos_lo=pos_lo,
                pos_hi=pos_hi,
                run_dir=run_dir,
                state=state,
                show_borders=show_borders,
                send_individually=send_individually,
                on_progress=_progress,
                on_trace=on_trace,
            )
            input_list.extend(result_items)

            if estimate_submitted:
                break

        if state.estimate_result:
            break
        if attempt == 0:
            _progress(
                "Warning: no estimate in attempt 1. "
                "Retrying with fresh history..."
            )

    # --- Finalize result ---
    if state.estimate_result:
        est_positions = cast(
            list[float], state.estimate_result["positions_mm"]
        )
        est_reasoning = str(state.estimate_result["reasoning"])
    else:
        # Fallback: center the group around atlas midpoint, clamped
        mid = (pos_lo + pos_hi) / 2
        span = (n_slices - 1) * interval_mm
        start = mid - span / 2
        est_positions = [
            max(pos_lo, min(pos_hi, start + i * interval_mm))
            for i in range(n_slices)
        ]
        est_reasoning = (
            "Model did not submit an estimate within the iteration limit."
        )
        _progress(
            "Warning: no estimate submitted after retry. "
            f"Falling back to centered midpoint: "
            f"{est_positions[0]:.2f}-{est_positions[-1]:.2f}mm"
        )

    results = [
        APResult(
            position_mm=pos,
            reasoning=est_reasoning,
            debug_dir=run_dir,
        )
        for pos in est_positions
    ]

    pos_str = ", ".join(f"{p:.3f}" for p in est_positions)
    _progress(
        f"Group estimation complete: [{pos_str}] mm "
        f"({state.images_fetched} atlas images fetched)"
    )

    _emit_trace(
        on_trace,
        runtime_event(
            stage="ap_group",
            title="Group estimation completed",
            summary=f"{n_slices} positions estimated",
            parts=[
                json_part(
                    {
                        "positions_mm": est_positions,
                        "reasoning": est_reasoning,
                        "images_fetched": state.images_fetched,
                    },
                    label="Group result",
                )
            ],
            metadata={"images_fetched": state.images_fetched},
        ),
    )

    return MultiSliceResult(
        positions=results,
        group_reasoning=est_reasoning,
        debug_dir=run_dir,
    )

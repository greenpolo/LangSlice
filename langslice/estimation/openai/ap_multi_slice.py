"""Multi-slice group AP estimation via OpenAI-compatible Chat Completions.

Port of ``langslice.estimation.google.ap_multi_slice`` adapted for the
Chat Completions API.  Uses ``get_openai_client()`` / ``get_openai_model()``
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
    atlas_resolution: int = 1024,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Process group tool calls and return Chat Completions messages.

    Returns ``(messages, estimate_submitted)`` where *messages* is a list of
    message dicts (``role='tool'`` results and optional ``role='user'`` image
    messages) to append to the conversation history.

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
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(response),
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
                atlas_resolution=atlas_resolution,
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

            near_iteration_limit = iteration >= state.max_iterations - 2

            # Validate count
            if len(positions_list) != state.n_slices:
                _append_tool_response(
                    call_id=call_id,
                    response={
                        "status": "error",
                        "error": (
                            f"Expected {state.n_slices} positions, "
                            f"got {len(positions_list)}."
                        ),
                    },
                )
                state.reasoning_log.append({
                    "iteration": iteration + 1,
                    "tool": name,
                    "args": args,
                    "result": (
                        f"Rejected: wrong count "
                        f"({len(positions_list)} vs {state.n_slices})"
                    ),
                })
                _emit_trace(
                    on_trace,
                    tool_result_event(
                        stage="ap_group",
                        tool_name=name,
                        summary="Submit rejected: wrong count",
                        parts=[json_part(args, label="Rejected submit")],
                        metadata={"iteration": iteration + 1, "status": "rejected"},
                    ),
                )
                continue

            # Clamp positions to valid atlas range
            clamped_positions = [
                max(pos_lo, min(pos_hi, float(p)))
                for p in positions_list
            ]
            out_of_range = [
                (i, float(p))
                for i, p in enumerate(positions_list)
                if float(p) < pos_lo or float(p) > pos_hi
            ]
            if out_of_range and not near_iteration_limit:
                detail = "; ".join(
                    f"Slice {i + 1}: {p:.2f}mm"
                    for i, p in out_of_range
                )
                _append_tool_response(
                    call_id=call_id,
                    response={
                        "status": "error",
                        "error": (
                            f"Some positions are outside the atlas range "
                            f"[{pos_lo:.2f}, {pos_hi:.2f}]mm: {detail}. "
                            f"Please correct and resubmit."
                        ),
                    },
                )
                state.reasoning_log.append({
                    "iteration": iteration + 1,
                    "tool": name,
                    "args": args,
                    "result": f"Rejected: out of range ({detail})",
                })
                _emit_trace(
                    on_trace,
                    tool_result_event(
                        stage="ap_group",
                        tool_name=name,
                        summary="Submit rejected: positions out of atlas range",
                        parts=[json_part(args, label="Rejected submit")],
                        metadata={"iteration": iteration + 1, "status": "rejected"},
                    ),
                )
                continue
            # Use clamped positions from here on
            positions_list = clamped_positions

            # Check monotonicity (slices are anterior-to-posterior)
            if not near_iteration_limit:
                is_monotonic = all(
                    positions_list[i] <= positions_list[i + 1]
                    for i in range(len(positions_list) - 1)
                )
                if not is_monotonic:
                    _append_tool_response(
                        call_id=call_id,
                        response={
                            "status": "error",
                            "error": (
                                "Positions must be strictly increasing "
                                "(anterior-to-posterior order). Please fix "
                                "the ordering and resubmit."
                            ),
                        },
                    )
                    state.reasoning_log.append({
                        "iteration": iteration + 1,
                        "tool": name,
                        "args": args,
                        "result": "Rejected: positions not strictly increasing",
                    })
                    _emit_trace(
                        on_trace,
                        tool_result_event(
                            stage="ap_group",
                            tool_name=name,
                            summary="Submit rejected: positions not monotonic",
                            parts=[json_part(args, label="Rejected submit")],
                            metadata={"iteration": iteration + 1, "status": "rejected"},
                        ),
                    )
                    continue

            # Check interval plausibility
            if not near_iteration_limit:
                intervals = [
                    positions_list[i + 1] - positions_list[i]
                    for i in range(len(positions_list) - 1)
                ]
                bad_intervals = [
                    (i, iv)
                    for i, iv in enumerate(intervals)
                    if abs(iv - state.interval_mm) > max(
                        0.5 * state.interval_mm, 0.25
                    )
                ]
                if bad_intervals:
                    detail = "; ".join(
                        f"Slice {i + 1}->{i + 2}: {iv:.3f}mm"
                        for i, iv in bad_intervals
                    )
                    _append_tool_response(
                        call_id=call_id,
                        response={
                            "status": "error",
                            "error": (
                                f"Some intervals deviate >50% from the expected "
                                f"{state.interval_mm:.3f}mm: {detail}. "
                                f"Please reconsider and resubmit."
                            ),
                        },
                    )
                    state.reasoning_log.append({
                        "iteration": iteration + 1,
                        "tool": name,
                        "args": args,
                        "result": f"Rejected: bad intervals ({detail})",
                    })
                    _emit_trace(
                        on_trace,
                        tool_result_event(
                            stage="ap_group",
                            tool_name=name,
                            summary="Submit rejected: interval plausibility failed",
                            parts=[json_part(args, label="Rejected submit")],
                            metadata={"iteration": iteration + 1, "status": "rejected"},
                        ),
                    )
                    continue

            if not state.saw_broad_sweep and not near_iteration_limit:
                _append_tool_response(
                    call_id=call_id,
                    response={
                        "status": "error",
                        "error": "Run a broad `fetch_atlas` sweep before submitting.",
                    },
                )
                state.reasoning_log.append({
                    "iteration": iteration + 1,
                    "tool": name,
                    "args": args,
                    "result": "Rejected: no broad sweep yet",
                })
                _emit_trace(
                    on_trace,
                    tool_result_event(
                        stage="ap_group",
                        tool_name=name,
                        summary="Submit rejected: broad sweep required",
                        parts=[json_part(args, label="Rejected submit")],
                        metadata={"iteration": iteration + 1, "status": "rejected"},
                    ),
                )
                continue

            if not state.saw_narrow_sweep and not near_iteration_limit:
                _append_tool_response(
                    call_id=call_id,
                    response={
                        "status": "error",
                        "error": "Run a narrow `fetch_atlas` sweep before submitting.",
                    },
                )
                state.reasoning_log.append({
                    "iteration": iteration + 1,
                    "tool": name,
                    "args": args,
                    "result": "Rejected: no narrow sweep yet",
                })
                _emit_trace(
                    on_trace,
                    tool_result_event(
                        stage="ap_group",
                        tool_name=name,
                        summary="Submit rejected: narrow sweep required",
                        parts=[json_part(args, label="Rejected submit")],
                        metadata={"iteration": iteration + 1, "status": "rejected"},
                    ),
                )
                continue

            est_positions = [float(p) for p in positions_list]
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
    atlas_resolution: int = 1024,
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

    atlas_obj_meta = cast(Any, atlas)
    species = atlas_obj_meta.metadata.get("species", "mouse")

    # --- Prepare all slice images ---
    prepared_images: list[Image.Image] = []
    image_bytes_list: list[bytes] = []
    for i, img in enumerate(images):
        normalized = normalize_image(img)
        prep = prepare_image_for_vlm(normalized)
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
                "Retrying group estimation (attempt 2/2, fresh history)..."
            )
            state = _GroupLoopState(
                n_slices=n_slices,
                max_iterations=max_iterations,
                interval_mm=interval_mm,
            )

        messages: list[Any] = [
            {"role": "system", "content": system_instruction},
            initial_user_message,
        ]

        _progress(
            f"Starting group estimation "
            f"({n_slices} slices, max {max_iterations} turns"
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
                    max_tokens=8000,
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
                messages.append({"role": "user", "content": nudge})
                continue

            # Append assistant message (with tool_calls) to history
            messages.append(response.choices[0].message)

            # Process tool calls — returns (messages_to_add, estimate_submitted)
            result_messages, estimate_submitted = _process_group_function_calls(
                function_calls,
                iteration=iteration,
                atlas=atlas,
                pos_lo=pos_lo,
                pos_hi=pos_hi,
                run_dir=run_dir,
                state=state,
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

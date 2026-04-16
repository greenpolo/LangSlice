"""Multi-slice AP estimation using Gemini generate_content API.

Given a group of 2-8 consecutive brain slice images with known section
interval, estimates AP positions for all slices simultaneously.  The model
sees all slices at once, which provides geometric constraints that
single-slice estimation cannot leverage.

Uses the same generate_content loop pattern as ``ap_tool_use.py`` but with
a group-aware system prompt and a ``submit_group_estimate`` tool.
"""

from __future__ import annotations

import io
import logging
import os
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any, cast

from google.genai import types
from PIL import Image

import langslice.vlm_config as vlm_config
from langslice.agent_trace import (
    image_part_from_pil,
    json_part,
    model_event,
    runtime_event,
    tool_call_event,
    tool_result_event,
)
from langslice.atlas.core import get_coronal_long_edge
from langslice.estimation._tool_logic import (
    _validate_submit_group_estimate,
    submit_group_estimate_schema,
)
from langslice.estimation._types import APResult, MultiSliceResult
from langslice.estimation.google.common import (
    _emit_trace,
    _extract_text_and_thoughts,
    _extract_usage_metadata,
    _format_usage_metadata,
    _get_position_range_lazy,
    _GroupLoopState,
    _history_metrics,
    _image_to_bytes,
    _load_atlas_lazy,
    _wait_for_uploaded_file,
)
from langslice.estimation.google.tool_definitions import (
    _build_nudge_text,
    _extract_function_calls,
    _handle_fetch_atlas,
)
from langslice.image_prep import normalize_image, prepare_image_for_vlm
from langslice.retry import retry_with_backoff
from langslice.vlm_config import get_client

logger = logging.getLogger(__name__)

# Multi-slice reasoning benefits from Pro's stronger visual reasoning —
# comparing N slices against atlas sections simultaneously is more demanding
# than single-slice estimation.
_DEFAULT_MODEL = "gemini-3.1-pro-preview"

_MEDIA_RES_MAP: dict[str, str] = {
    "low": "MEDIA_RESOLUTION_LOW",
    "medium": "MEDIA_RESOLUTION_MEDIUM",
    "high": "MEDIA_RESOLUTION_HIGH",
    "ultra_high": "MEDIA_RESOLUTION_ULTRA_HIGH",
}


# ---------------------------------------------------------------------------
# Tool declarations
# ---------------------------------------------------------------------------


def _group_tool_declarations(n_slices: int) -> list[types.Tool]:
    """Tool declarations for multi-slice estimation (generate_content format)."""
    group_schema = submit_group_estimate_schema(n_slices)
    return [types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="fetch_atlas",
            description="Fetch 1-8 atlas coronal sections as a labeled grid.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "positions_mm": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 1,
                        "maxItems": 8,
                        "description": "AP positions in mm (1-8).",
                    },
                },
                "required": ["positions_mm"],
            },
        ),
        types.FunctionDeclaration(
            name=cast(str, group_schema["name"]),
            description=cast(str, group_schema["description"]),
            parameters_json_schema=cast(
                dict[str, object], group_schema["parameters"]
            ),
        ),
    ])]


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


def _process_group_function_calls_gc(
    function_calls: list[dict[str, object]],
    *,
    iteration: int,
    atlas: object,
    pos_lo: float,
    pos_hi: float,
    run_dir: str | None,
    state: _GroupLoopState,
    show_borders: bool = False,
    send_individually: bool = True,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
) -> list[types.Part]:
    """Process tool calls and return generate_content Parts.

    Returns a flat list of Parts.  The caller splits them:
    - Parts with ``.function_response`` -> ``Content(role='tool')``
    - Other Parts (text labels, images) -> ``Content(role='user')``
    """
    result_parts: list[types.Part] = []

    def _append_response(
        *,
        call_id: object,  # noqa: ARG001 — kept for parity
        name: str,
        response: dict[str, object],
        is_error: bool = False,  # noqa: ARG001 — kept for parity
    ) -> None:
        result_parts.append(
            types.Part.from_function_response(
                name=name,
                response=response,
            )
        )

    for call in function_calls:
        name = str(call.get("name", ""))
        args_obj = call.get("args", {})
        args = args_obj if isinstance(args_obj, dict) else {}
        call_id = call.get("call_id")

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
            fetch_parts, _fn = _handle_fetch_atlas(
                args=args,
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
            result_parts.extend(fetch_parts)

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
                _append_response(
                    call_id=call_id,
                    name=name,
                    response=error_response,
                    is_error=True,
                )
                state.reasoning_log.append(
                    {
                        "iteration": iteration + 1,
                        "tool": name,
                        "args": args,
                        "result": log_reason,
                    }
                )
                continue

            est_positions = cast(list[float], est_positions)
            state.estimate_result = {
                "positions_mm": est_positions,
                "reasoning": est_reasoning,
            }
            state.reasoning_log.append(
                {
                    "iteration": iteration + 1,
                    "tool": name,
                    "args": args,
                    "result": f"Submitted {len(est_positions)} positions",
                }
            )
            if on_progress:
                pos_str = ", ".join(f"{p:.2f}" for p in est_positions)
                on_progress(
                    f"Group estimate submitted: [{pos_str}] mm"
                )
            _emit_trace(
                on_trace,
                tool_result_event(
                    stage="ap_group",
                    tool_name=name,
                    summary=f"Submitted {len(est_positions)} positions",
                    parts=[
                        json_part(
                            state.estimate_result, label="Group estimate"
                        )
                    ],
                    metadata={
                        "iteration": iteration + 1,
                        "status": "accepted",
                    },
                ),
            )

        else:
            _append_response(
                call_id=call_id,
                name=name,
                response={
                    "status": "error",
                    "error": f"Unknown tool: {name}",
                },
                is_error=True,
            )

    return result_parts


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
    media_resolution: str | None = None,
    model_name: str | None = None,
    show_borders: bool = False,
    send_individually: bool = True,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
    debug_dir: str | None = None,
) -> MultiSliceResult:
    """Estimate AP positions for a group of consecutive brain slices.

    The model sees all slices simultaneously with known section spacing,
    enabling geometric reasoning that single-slice estimation cannot.

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
        Maximum generate_content turns.
    model_name
        Override the default Gemini model.
    """
    n_slices = len(images)
    if not 2 <= n_slices <= 8:
        raise ValueError(f"Expected 2-8 slices, got {n_slices}")

    interval_mm = interval_um / 1000.0

    def _progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        logger.info(msg)

    client = get_client()
    atlas = _load_atlas_lazy(atlas_name)
    pos_lo, pos_hi = _get_position_range_lazy(atlas)
    atlas_long_edge = get_coronal_long_edge(atlas)

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
                )
                for i, img in enumerate(prepared_images)
            ],
            metadata={
                "n_slices": n_slices,
                "interval_um": interval_um,
                "thickness_um": thickness_um,
            },
        ),
    )

    # --- System prompt ---
    atlas_obj_meta = cast(Any, atlas)
    species = atlas_obj_meta.metadata.get("species", "mouse")
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

    # --- Loop setup ---
    temperature = vlm_config.TEMPERATURE
    effective_model = model_name or _DEFAULT_MODEL
    _is_pro = "pro" in effective_model.lower()
    thinking_level = vlm_config.get_thinking_level_or(
        "HIGH" if _is_pro else "MEDIUM"
    )
    if media_resolution is None:
        media_resolution = "high" if _is_pro else "low"
    max_iterations = max(1, int(max_iterations))

    state = _GroupLoopState(
        n_slices=n_slices,
        max_iterations=max_iterations,
        interval_mm=interval_mm,
    )
    uploaded_files: list[Any] = []

    try:
        # --- Upload all slices via File API ---
        file_uris: list[str] = []
        for i, img_bytes in enumerate(image_bytes_list):
            buf = io.BytesIO(img_bytes)
            uploaded = client.files.upload(
                file=buf,
                config=types.UploadFileConfig(
                    mime_type="image/jpeg",
                    display_name=f"slice_{i + 1}",
                ),
            )
            file_name = getattr(uploaded, "name", None)
            if not isinstance(file_name, str) or not file_name:
                raise RuntimeError(
                    f"File API upload failed for slice {i + 1}"
                )
            uploaded_files.append(uploaded)
            _wait_for_uploaded_file(
                client,
                file_name=file_name,
                timeout_s=30.0,
                on_progress=_progress,
            )
            uri = getattr(uploaded, "uri", None)
            if not isinstance(uri, str) or not uri:
                raise RuntimeError(
                    f"File API returned no URI for slice {i + 1}"
                )
            file_uris.append(uri)

        _progress(f"Uploaded {n_slices} slices to Gemini File API")

        # --- Build initial contents ---
        initial_parts: list[types.Part] = [
            types.Part.from_text(
                text=(
                    f"Here are {n_slices} consecutive brain slices, ordered "
                    f"anterior to posterior. Section interval: {interval_um} "
                    f"\u00b5m ({interval_mm:.3f} mm). Slice thickness: "
                    f"{thickness_um} \u00b5m. Determine the AP position of "
                    "each slice in the atlas."
                ),
            ),
        ]
        for i, uri in enumerate(file_uris):
            initial_parts.append(
                types.Part.from_text(text=f"Slice {i + 1}:")
            )
            initial_parts.append(
                types.Part.from_uri(file_uri=uri, mime_type="image/jpeg")
            )
        initial_content = types.Content(role="user", parts=initial_parts)

        # --- Tool declarations ---
        tools = _group_tool_declarations(n_slices)

        # --- Config ---
        thinking_cfg = vlm_config.build_thinking_config(
            effective_model, thinking_level
        )
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=8000,
            thinking_config=cast(Any, thinking_cfg),
            tools=cast(Any, tools),
            media_resolution=cast(
                Any, _MEDIA_RES_MAP.get(media_resolution, "MEDIA_RESOLUTION_HIGH")
            ),
        )

        # --- Main generate_content loop (with one retry on failure) ---
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

            contents: list[types.Content] = [initial_content]

            _progress(
                f"Starting group estimation "
                f"({n_slices} slices, max {max_iterations} turns"
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
                    part_count = request_metrics["part_count"]
                    img_count = request_metrics["image_parts"]
                    img_bytes = request_metrics["image_bytes"]
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
                    request_label=f"Group estimation turn {iteration + 1}",
                    on_progress=_progress,
                )
                elapsed = round(time.perf_counter() - started_at, 3)
                turn_metric["wall_time_s"] = elapsed
                usage_metadata = _extract_usage_metadata(response)
                turn_metric["usage_metadata"] = usage_metadata
                state.turn_metrics.append(turn_metric)

                _progress(
                    f"Turn {iteration + 1} in {elapsed}s; "
                    f"{_format_usage_metadata(usage_metadata)}"
                )

                # Append model response to history
                model_content = response.candidates[0].content
                contents.append(model_content)

                # Emit trace for text/thought outputs
                text_outputs, thought_outputs = (
                    _extract_text_and_thoughts(model_content)
                )
                if text_outputs or thought_outputs:
                    trace_parts: list[dict[str, object]] = []
                    if thought_outputs:
                        trace_parts.append(
                            json_part(
                                thought_outputs,
                                label="Reasoning summary",
                                collapsible=True,
                            )
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
                            stage="ap_group",
                            title=f"Model turn {iteration + 1}",
                            summary="Model reasoning before next tool step",
                            parts=trace_parts,
                            metadata={
                                "iteration": iteration + 1,
                                **usage_metadata,
                            },
                        ),
                    )

                # Extract function calls
                function_calls, text_preview = _extract_function_calls(
                    response
                )

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
                            f"candidate tokens, no tool call). Nudging."
                        )
                        nudge = (
                            "You wrote a long text response instead of "
                            "calling a tool. Do NOT repeat this. Use your "
                            "internal reasoning, then call a tool: "
                            "`fetch_atlas` or `submit_group_estimate`."
                        )
                    else:
                        if text_preview and on_progress:
                            _progress(
                                f"Model text: {text_preview[:200]}..."
                            )
                        nudge = _build_nudge_text(state, submit_tool="submit_group_estimate")
                    contents.append(types.Content(role="user", parts=[
                        types.Part.from_text(text=nudge),
                    ]))
                    continue

                # Process tool calls — returns list[Part]
                result_parts = _process_group_function_calls_gc(
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
                # Got a result — no need to retry
                break
            if attempt == 0:
                _progress(
                    "Warning: no estimate submitted in attempt 1. "
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

    finally:
        for f in uploaded_files:
            try:
                client.files.delete(name=f.name)
            except Exception as exc:
                logger.warning(
                    "Failed to delete file %s: %s",
                    getattr(f, "name", "?"),
                    exc,
                )

"""OpenAI Chat Completions tool definitions for AP estimation.

Port of ``langslice.estimation.google.tool_definitions`` adapted for the
Chat Completions function-calling format.  Provider-agnostic helpers
(grid building, position math, nudge text) are imported from the Google
module.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any, cast

from PIL import Image

from langslice.agent_trace import (
    json_part,
    tool_call_event,
    tool_result_event,
)
from langslice.estimation.google.tool_definitions import (
    _build_atlas_grid as _build_atlas_grid,
)
from langslice.estimation.google.tool_definitions import (
    _build_nudge_text as _build_nudge_text,
)
from langslice.estimation.google.tool_definitions import (
    _has_neighbor_bracket as _has_neighbor_bracket,
)
from langslice.estimation.google.tool_definitions import (
    _is_broad_multi_sweep as _is_broad_multi_sweep,
)
from langslice.estimation.google.tool_definitions import (
    _is_narrow_multi_sweep as _is_narrow_multi_sweep,
)
from langslice.estimation.google.tool_definitions import (
    _sorted_unique_positions as _sorted_unique_positions,
)
from langslice.estimation.openai.common import (
    _APLoopState,
    _build_image_content,
    _build_text_content,
    _bytes_to_base64,
    _emit_trace,
    _fetch_atlas_slice_bytes,
    _image_to_bytes,
)

# ---------------------------------------------------------------------------
# Tool declarations in Chat Completions format
# ---------------------------------------------------------------------------


def _tool_declarations() -> list[dict[str, Any]]:
    """Return tool declarations for single-slice AP estimation.

    Format follows the OpenAI Chat Completions ``tools`` parameter::

        [{"type": "function", "function": {"name": ..., "parameters": ...}}]
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "fetch_atlas",
                "description": (
                    "Fetch atlas coronal sections at specific AP positions. Returns "
                    "a single labeled grid image for direct visual comparison. You "
                    "choose exactly which positions to see (1 to 8). Use this to "
                    "compare multiple positions at once — you can space them however "
                    "you like (evenly, densely around a candidate, etc.)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "positions_mm": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 1,
                            "maxItems": 8,
                            "description": (
                                "List of AP positions in mm to fetch. Choose any "
                                "positions you want — you can cluster them densely "
                                "around a candidate or spread them widely."
                            ),
                        },
                    },
                    "required": ["positions_mm"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_estimate",
                "description": (
                    "Submit your final AP position estimate. Only call this when "
                    "you are confident in your answer."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "position_mm": {
                            "type": "number",
                            "description": (
                                "Final estimated AP position in mm from the anterior edge"
                            ),
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Detailed reasoning for the estimate",
                        },
                    },
                    "required": ["position_mm", "reasoning"],
                },
            },
        },
    ]


def _group_tool_declarations(n_slices: int) -> list[dict[str, Any]]:
    """Return tool declarations for multi-slice group AP estimation.

    Same two tools as the Gemini variant (``fetch_atlas`` +
    ``submit_group_estimate``) but in Chat Completions format.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "fetch_atlas",
                "description": "Fetch 1-8 atlas coronal sections as a labeled grid.",
                "parameters": {
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
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_group_estimate",
                "description": (
                    f"Submit final AP estimates for all {n_slices} slices."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "positions_mm": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": (
                                f"AP positions in mm for all {n_slices} slices, "
                                "in order."
                            ),
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief reasoning.",
                        },
                    },
                    "required": ["positions_mm", "reasoning"],
                },
            },
        },
    ]


# ---------------------------------------------------------------------------
# Function call extraction (Chat Completions format)
# ---------------------------------------------------------------------------


def _extract_function_calls(
    response: object,
) -> tuple[list[dict[str, object]], str | None]:
    """Extract function calls from a Chat Completions response.

    Reads ``response.choices[0].message.tool_calls`` and parses each
    tool call's ``function.arguments`` JSON string.

    Returns ``(function_calls, text_preview)`` where each function call is
    ``{"call_id": str, "name": str, "args": dict}``.  ``text_preview`` is
    ``response.choices[0].message.content`` if any text was returned.
    """
    choices = getattr(response, "choices", None) or []
    if not choices:
        return [], None

    message = getattr(choices[0], "message", None)
    if message is None:
        return [], None

    text_preview: str | None = None
    content = getattr(message, "content", None)
    if isinstance(content, str) and content:
        text_preview = content

    tool_calls_attr = getattr(message, "tool_calls", None) or []
    function_calls: list[dict[str, object]] = []
    for tc in tool_calls_attr:
        tc_id = getattr(tc, "id", None) or ""
        func = getattr(tc, "function", None)
        if func is None:
            continue
        name = getattr(func, "name", "") or ""
        arguments_str = getattr(func, "arguments", "{}") or "{}"
        try:
            args = json.loads(arguments_str)
        except (json.JSONDecodeError, TypeError):
            args = {}
        if not isinstance(args, dict):
            args = {}
        function_calls.append({
            "call_id": str(tc_id),
            "name": str(name),
            "args": args,
        })

    return function_calls, text_preview


# ---------------------------------------------------------------------------
# Shared fetch_atlas handler (Chat Completions message format)
# ---------------------------------------------------------------------------


def _handle_fetch_atlas(
    *,
    args: dict[str, object],
    tool_call_id: str,
    pos_lo: float,
    pos_hi: float,
    atlas: object,
    state: _APLoopState,
    iteration: int,
    run_dir: str | None,
    show_borders: bool,
    send_individually: bool,
    atlas_resolution: int,
    target_image: Image.Image | None,
    stage: str,
    on_progress: Callable[[str], None] | None,
    on_trace: Callable[[dict[str, object]], None] | None,
) -> tuple[list[dict[str, Any]], str]:
    """Handle a ``fetch_atlas`` tool call for Chat Completions.

    Returns ``(response_items, function_name)`` where *response_items* is
    a list of message dicts to append to the conversation history:

    - ``{"role": "tool", "tool_call_id": ..., "content": json_string}``
    - ``{"role": "user", "content": [text_parts, image_parts]}``

    The caller appends all items directly to the messages list.
    """
    atlas_obj = cast(Any, atlas)
    response_items: list[dict[str, Any]] = []
    fn_name = "fetch_atlas"

    positions_list = args.get("positions_mm", [])
    if not isinstance(positions_list, list):
        positions_list = []

    positions = [max(pos_lo, min(pos_hi, float(p))) for p in positions_list[:8]]

    if not positions:
        response_items.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps({"status": "error", "error": "No valid positions provided"}),
        })
        return response_items, fn_name

    state.fetched_positions.extend(positions)
    if _is_broad_multi_sweep(positions):
        state.saw_broad_sweep = True
    if _is_narrow_multi_sweep(positions):
        state.saw_narrow_sweep = True

    pos_label = ", ".join(f"{p:.2f}" for p in positions)
    state.images_fetched += len(positions)

    if send_individually:
        # Tool result acknowledging the fetch
        response_items.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps({
                "status": "ok",
                "positions_mm": positions,
                "description": (
                    f"{len(positions)} atlas sections at: {pos_label} mm."
                ),
            }),
        })
        # Send individual atlas images as a user message with mixed content
        image_content_parts: list[dict[str, Any]] = []
        for idx, pos in enumerate(positions):
            try:
                ref_bytes = _fetch_atlas_slice_bytes(
                    atlas_obj,
                    pos,
                    max_long_edge=atlas_resolution,
                    show_borders=show_borders,
                )
                image_content_parts.append(
                    _build_text_content(f"[{idx + 1}] {pos:.2f} mm:")
                )
                image_content_parts.append(
                    _build_image_content(_bytes_to_base64(ref_bytes))
                )
            except (ValueError, IndexError):
                pass
        if image_content_parts:
            response_items.append({
                "role": "user",
                "content": image_content_parts,
            })
        if run_dir:
            grid_img = _build_atlas_grid(
                atlas, positions, show_borders=show_borders,
            )
            grid_img.save(
                os.path.join(
                    run_dir,
                    f"tool_{iteration + 1:02d}_atlas_{len(positions)}x.jpg",
                ),
                quality=85,
            )
    else:
        # Grid mode — single composite image
        grid_img = _build_atlas_grid(
            atlas,
            positions,
            target_image=target_image,
            show_borders=show_borders,
        )
        grid_bytes = _image_to_bytes(grid_img)
        if run_dir:
            grid_img.save(
                os.path.join(
                    run_dir,
                    f"tool_{iteration + 1:02d}_atlas_{len(positions)}x.jpg",
                ),
                quality=85,
            )
        response_items.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps({
                "status": "ok",
                "positions_mm": positions,
                "description": (
                    f"Grid of {len(positions)} atlas sections at: {pos_label} mm. "
                    "Each cell is numbered and labeled with its AP position."
                ),
            }),
        })
        response_items.append({
            "role": "user",
            "content": [
                _build_image_content(_bytes_to_base64(grid_bytes)),
            ],
        })

    state.reasoning_log.append({
        "iteration": iteration + 1,
        "tool": fn_name,
        "args": args,
        "result": f"Atlas: [{pos_label}] mm",
    })
    _emit_trace(
        on_trace,
        tool_result_event(
            stage=stage,
            tool_name=fn_name,
            summary=f"Returned {len(positions)} atlas sections",
            metadata={
                "iteration": iteration + 1,
                "positions": positions,
                "send_individually": send_individually,
            },
        ),
    )

    return response_items, fn_name


# ---------------------------------------------------------------------------
# Main tool dispatch (Chat Completions message format)
# ---------------------------------------------------------------------------


def _process_ap_function_calls(
    function_calls: list[dict[str, object]],
    *,
    iteration: int,
    atlas: object,
    pos_lo: float,
    pos_hi: float,
    target_h: int,
    run_dir: str | None,
    state: _APLoopState,
    target_image: Image.Image | None = None,
    show_borders: bool = False,
    send_individually: bool = True,
    atlas_resolution: int = 1024,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Process function calls and return Chat Completions messages.

    Returns ``(messages, estimate_submitted)`` where *messages* is a list
    of message dicts (``role='tool'`` results and optional ``role='user'``
    image messages) to append to the conversation history.

    Each tool result includes a ``tool_call_id`` matching the original
    call's ID, as required by the Chat Completions API.
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
                stage="ap",
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
                target_image=target_image,
                stage="ap",
                on_progress=on_progress,
                on_trace=on_trace,
            )
            result_messages.extend(fetch_items)

        elif name == "submit_estimate":
            est_pos = float(args.get("position_mm", 0.0))  # type: ignore[arg-type]
            est_reasoning = str(args.get("reasoning", ""))
            has_neighbor_check = _has_neighbor_bracket(
                state.fetched_positions,
                est_pos,
                pos_lo=pos_lo,
                pos_hi=pos_hi,
            )
            near_iteration_limit = iteration >= state.max_iterations - 2

            if not state.saw_broad_sweep and not near_iteration_limit:
                _append_tool_response(
                    call_id=call_id,
                    response={
                        "status": "error",
                        "error": (
                            "Run a broad `fetch_multiple_atlas_slices`"
                            " sweep before submitting."
                        ),
                    },
                )
                state.reasoning_log.append({
                    "iteration": iteration + 1,
                    "tool": name,
                    "args": args,
                    "result": f"Rejected submit at {est_pos:.2f}mm: no broad sweep yet",
                })
                _emit_trace(
                    on_trace,
                    tool_result_event(
                        stage="ap",
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
                        "error": (
                            "Run a narrowed"
                            " `fetch_multiple_atlas_slices`"
                            " sweep around your best candidate"
                            " before submitting."
                        ),
                    },
                )
                state.reasoning_log.append({
                    "iteration": iteration + 1,
                    "tool": name,
                    "args": args,
                    "result": f"Rejected submit at {est_pos:.2f}mm: no narrow sweep yet",
                })
                _emit_trace(
                    on_trace,
                    tool_result_event(
                        stage="ap",
                        tool_name=name,
                        summary="Submit rejected: narrow sweep required",
                        parts=[json_part(args, label="Rejected submit")],
                        metadata={"iteration": iteration + 1, "status": "rejected"},
                    ),
                )
                continue

            if not has_neighbor_check and not near_iteration_limit:
                lower = max(pos_lo, est_pos - 0.2)
                upper = min(pos_hi, est_pos + 0.2)
                _append_tool_response(
                    call_id=call_id,
                    response={
                        "status": "error",
                        "error": (
                            "Before submitting, verify at least"
                            " one lower and one higher"
                            " neighboring AP position around"
                            f" {est_pos:.2f} mm (for example"
                            f" {lower:.2f} mm and"
                            f" {upper:.2f} mm)."
                        ),
                    },
                )
                state.reasoning_log.append({
                    "iteration": iteration + 1,
                    "tool": name,
                    "args": args,
                    "result": f"Rejected submit at {est_pos:.2f}mm: neighborhood not bracketed",
                })
                _emit_trace(
                    on_trace,
                    tool_result_event(
                        stage="ap",
                        tool_name=name,
                        summary="Submit rejected: neighboring AP checks required",
                        parts=[json_part(args, label="Rejected submit")],
                        metadata={"iteration": iteration + 1, "status": "rejected"},
                    ),
                )
                continue

            state.estimate_result = {
                "position_mm": est_pos,
                "reasoning": est_reasoning,
            }
            state.reasoning_log.append({
                "iteration": iteration + 1,
                "tool": name,
                "args": args,
                "result": f"Submitted {est_pos:.2f}mm",
            })
            if on_progress:
                on_progress(
                    f"Agent submitted estimate: {est_pos:.2f}mm"
                )
            _emit_trace(
                on_trace,
                tool_result_event(
                    stage="ap",
                    tool_name=name,
                    summary=f"Submitted estimate {est_pos:.2f} mm",
                    parts=[json_part(state.estimate_result, label="Submitted estimate")],
                    metadata={"iteration": iteration + 1, "status": "accepted"},
                ),
            )
            estimate_submitted = True
            # Provide an acknowledgement so every tool call has a response
            _append_tool_response(
                call_id=call_id,
                response={
                    "status": "ok",
                    "position_mm": est_pos,
                    "message": "Estimate accepted.",
                },
            )

        else:
            _append_tool_response(
                call_id=call_id,
                response={"status": "error", "error": f"Unknown tool: {name}"},
            )

    return result_messages, estimate_submitted

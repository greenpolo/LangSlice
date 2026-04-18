"""OpenAI Responses API tool definitions for AP estimation."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, cast

from PIL import Image

from langslice.agent_trace import (
    json_part,
    tool_call_event,
    tool_result_event,
)
from langslice.estimation._tool_logic import (
    FETCH_ATLAS_SCHEMA,
    SUBMIT_ESTIMATE_SCHEMA,
    _handle_fetch_atlas_core,
    _validate_submit_estimate,
    submit_group_estimate_schema,
)
from langslice.estimation._tool_logic import (
    _build_atlas_grid as _build_atlas_grid,
)
from langslice.estimation._tool_logic import (
    _build_nudge_text as _build_nudge_text,
)
from langslice.estimation.openai.common import (
    _APLoopState,
    _build_image_content,
    _build_text_content,
    _bytes_to_base64,
    _emit_trace,
)


def _tool_declarations() -> list[dict[str, Any]]:
    """Return tool declarations for single-slice AP estimation."""
    return [
        {
            "type": "function",
            "name": FETCH_ATLAS_SCHEMA["name"],
            "description": FETCH_ATLAS_SCHEMA["description"],
            "parameters": FETCH_ATLAS_SCHEMA["parameters"],
        },
        {
            "type": "function",
            "name": SUBMIT_ESTIMATE_SCHEMA["name"],
            "description": SUBMIT_ESTIMATE_SCHEMA["description"],
            "parameters": SUBMIT_ESTIMATE_SCHEMA["parameters"],
        },
    ]


def _group_tool_declarations(n_slices: int) -> list[dict[str, Any]]:
    """Return tool declarations for multi-slice group AP estimation."""
    group_schema = submit_group_estimate_schema(n_slices)
    return [
        {
            "type": "function",
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
        {
            "type": "function",
            "name": cast(str, group_schema["name"]),
            "description": cast(str, group_schema["description"]),
            "parameters": cast(dict[str, object], group_schema["parameters"]),
        },
    ]


def _extract_function_calls(
    response: object,
) -> tuple[list[dict[str, object]], str | None]:
    """Extract function calls from a Responses API response."""
    output_items = getattr(response, "output", None) or []

    text_preview: str | None = None
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text:
        text_preview = output_text

    function_calls: list[dict[str, object]] = []
    for item in output_items:
        if getattr(item, "type", None) != "function_call":
            continue
        call_id = getattr(item, "call_id", None) or ""
        name = getattr(item, "name", "") or ""
        arguments_str = getattr(item, "arguments", "{}") or "{}"
        try:
            args = json.loads(arguments_str)
        except (json.JSONDecodeError, TypeError):
            args = {}
        if not isinstance(args, dict):
            args = {}
        function_calls.append({
            "call_id": str(call_id),
            "name": str(name),
            "args": args,
        })

    return function_calls, text_preview


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
    target_image: Image.Image | None,
    stage: str,
    on_progress: Callable[[str], None] | None,
    on_trace: Callable[[dict[str, object]], None] | None,
    image_detail: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Handle a ``fetch_atlas`` tool call for the Responses API."""
    _ = on_progress
    fetch_result = _handle_fetch_atlas_core(
        args=args,
        pos_lo=pos_lo,
        pos_hi=pos_hi,
        atlas=atlas,
        state=state,
        iteration=iteration,
        run_dir=run_dir,
        show_borders=show_borders,
        send_individually=send_individually,
        target_image=target_image,
        stage=stage,
        on_trace=on_trace,
    )

    response_items: list[dict[str, Any]] = [
        {
            "type": "function_call_output",
            "call_id": tool_call_id,
            "output": json.dumps(fetch_result.response),
        }
    ]
    if fetch_result.labeled_images:
        image_content_parts: list[dict[str, Any]] = []
        for label, image_bytes in fetch_result.labeled_images:
            image_content_parts.append(_build_text_content(label))
            image_content_parts.append(
                _build_image_content(_bytes_to_base64(image_bytes), detail=image_detail)
            )
        response_items.append({
            "role": "user",
            "content": image_content_parts,
        })
    elif fetch_result.grid_image_bytes is not None:
        response_items.append({
            "role": "user",
            "content": [
                _build_image_content(
                    _bytes_to_base64(fetch_result.grid_image_bytes),
                    detail=image_detail,
                ),
            ],
        })
    return response_items, fetch_result.function_name


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
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
    image_detail: str | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Process function calls and return Responses API input items."""
    _ = target_h
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
                target_image=target_image,
                stage="ap",
                on_progress=on_progress,
                on_trace=on_trace,
                image_detail=image_detail,
            )
            result_messages.extend(fetch_items)

        elif name == "submit_estimate":
            est_pos = float(args.get("position_mm", 0.0))  # type: ignore[arg-type]
            est_reasoning = str(args.get("reasoning", ""))
            error_response, log_reason = _validate_submit_estimate(
                state=state,
                est_pos=est_pos,
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
                if log_reason.endswith("no broad sweep yet"):
                    summary = "Submit rejected: broad sweep required"
                elif log_reason.endswith("no narrow sweep yet"):
                    summary = "Submit rejected: narrow sweep required"
                else:
                    summary = "Submit rejected: neighboring AP checks required"
                _emit_trace(
                    on_trace,
                    tool_result_event(
                        stage="ap",
                        tool_name=name,
                        summary=summary,
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
                on_progress(f"Agent submitted estimate: {est_pos:.2f}mm")
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

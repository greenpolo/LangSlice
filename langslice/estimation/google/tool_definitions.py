"""Tool handler logic for the Gemini AP estimation loop."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from google.genai import types as genai_types
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
)
from langslice.estimation._tool_logic import (
    _build_atlas_grid as _build_atlas_grid,
)
from langslice.estimation._tool_logic import (
    _build_nudge_text as _build_nudge_text,
)
from langslice.estimation.google.common import (
    _APLoopState,
    _emit_trace,
)


def _tool_declarations() -> list[genai_types.Tool]:
    """Return tool declarations for single-slice AP estimation."""
    return [genai_types.Tool(function_declarations=[
        genai_types.FunctionDeclaration(
            name=cast(str, FETCH_ATLAS_SCHEMA["name"]),
            description=cast(str, FETCH_ATLAS_SCHEMA["description"]),
            parameters_json_schema=cast(
                dict[str, object], FETCH_ATLAS_SCHEMA["parameters"]
            ),
        ),
        genai_types.FunctionDeclaration(
            name=cast(str, SUBMIT_ESTIMATE_SCHEMA["name"]),
            description=cast(str, SUBMIT_ESTIMATE_SCHEMA["description"]),
            parameters_json_schema=cast(
                dict[str, object], SUBMIT_ESTIMATE_SCHEMA["parameters"]
            ),
        ),
    ])]


def _extract_function_calls(
    response: object,
) -> tuple[list[dict[str, object]], str | None]:
    """Extract function calls from a generate_content response."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return [], None
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []
    function_calls: list[dict[str, object]] = []
    text_preview: str | None = None
    for part in parts:
        fc = getattr(part, "function_call", None)
        if fc is not None:
            function_calls.append({
                "call_id": getattr(fc, "id", None),
                "name": getattr(fc, "name", ""),
                "args": dict(getattr(fc, "args", {}) or {}),
            })
        elif text_preview is None:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                text_preview = text
    return function_calls, text_preview


def _handle_fetch_atlas(
    *,
    args: dict[str, object],
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
) -> tuple[list[genai_types.Part], str]:
    """Handle a ``fetch_atlas`` tool call shared by single- and multi-slice."""
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

    result_parts = [
        genai_types.Part.from_function_response(
            name=fetch_result.function_name,
            response=fetch_result.response,
        )
    ]
    for label, image_bytes in fetch_result.labeled_images:
        result_parts.append(genai_types.Part.from_text(text=label))
        result_parts.append(
            genai_types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        )
    if fetch_result.grid_image_bytes is not None:
        result_parts.append(
            genai_types.Part.from_bytes(
                data=fetch_result.grid_image_bytes,
                mime_type="image/jpeg",
            )
        )
    return result_parts, fetch_result.function_name


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
) -> list[genai_types.Part]:
    """Process function calls and return generate_content Parts."""
    _ = target_h
    result_parts: list[genai_types.Part] = []

    def _append_response(
        *,
        call_id: object,  # noqa: ARG001 - kept for parity
        name: str,
        response: dict[str, object],
        is_error: bool = False,  # noqa: ARG001 - kept for parity
    ) -> None:
        result_parts.append(
            genai_types.Part.from_function_response(
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
                stage="ap",
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
                target_image=target_image,
                stage="ap",
                on_progress=on_progress,
                on_trace=on_trace,
            )
            result_parts.extend(fetch_parts)

        elif name == "submit_estimate":
            est_pos = float(args.get("position_mm", 0.0))
            est_reasoning = str(args.get("reasoning", ""))
            error_response, log_reason = _validate_submit_estimate(
                state=state,
                est_pos=est_pos,
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
            state.reasoning_log.append(
                {
                    "iteration": iteration + 1,
                    "tool": name,
                    "args": args,
                    "result": f"Submitted {est_pos:.2f}mm",
                }
            )
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

        else:
            _append_response(
                call_id=call_id,
                name=name,
                response={"status": "error", "error": f"Unknown tool: {name}"},
                is_error=True,
            )

    return result_parts

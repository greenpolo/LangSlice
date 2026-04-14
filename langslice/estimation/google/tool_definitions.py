"""Tool handler logic for the AP estimation loop.

Functions in this module implement the tool-use dispatch and helpers for
the agentic AP estimator.  They were extracted from ``estimator.py`` purely
for readability — no behavioral changes.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, Protocol, cast

from google.genai import types as genai_types
from PIL import Image

from langslice.agent_trace import (
    json_part,
    tool_call_event,
    tool_result_event,
)
from langslice.estimation.google.common import (
    _APLoopState,
    _emit_trace,
    _fetch_atlas_slice_bytes,
    _image_to_bytes,
)
from langslice.image_prep import normalize_image


class _NudgeState(Protocol):
    """Minimal interface for _build_nudge_text — satisfied by both
    _APLoopState and _GroupLoopState."""

    saw_broad_sweep: bool
    saw_narrow_sweep: bool


_RESAMPLE_LANCZOS = Image.Resampling.LANCZOS


# ---------------------------------------------------------------------------
# Tool declarations as FunctionDeclaration objects for generate_content API
# ---------------------------------------------------------------------------


def _tool_declarations() -> list[genai_types.Tool]:
    """Return tool declarations for single-slice AP estimation (generate_content format)."""
    return [genai_types.Tool(function_declarations=[
        genai_types.FunctionDeclaration(
            name="fetch_atlas",
            description=(
                "Fetch atlas coronal sections at specific AP positions. Returns "
                "a single labeled grid image for direct visual comparison. You "
                "choose exactly which positions to see (1 to 8). Use this to "
                "compare multiple positions at once — you can space them however "
                "you like (evenly, densely around a candidate, etc.)."
            ),
            parameters_json_schema={
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
        ),
        genai_types.FunctionDeclaration(
            name="submit_estimate",
            description=(
                "Submit your final AP position estimate. Only call this when "
                "you are confident in your answer."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "position_mm": {
                        "type": "number",
                        "description": "Final estimated AP position in mm from the anterior edge",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Detailed reasoning for the estimate",
                    },
                },
                "required": ["position_mm", "reasoning"],
            },
        ),
    ])]


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------


def _build_atlas_grid(
    atlas: object,
    positions: list[float],
    *,
    target_image: Image.Image | None = None,
    grid_width: int = 2048,
    show_borders: bool = False,
    max_positions: int = 8,
) -> Image.Image:
    """Build a comparison image: target slice (left) + 2x2 atlas grid (right).

    If *target_image* is provided, it is placed on the left at its natural
    aspect ratio, scaled to match the grid height.  The atlas grid occupies
    the right side.  If no target, returns just the 2x2 grid.
    """
    from PIL import ImageDraw, ImageFont

    atlas_obj = cast(Any, atlas)
    n = min(len(positions), max_positions)

    # Dynamic grid layout based on count
    if n <= 2:
        cols = n
    elif n <= 4:
        cols = 2
    elif n <= 6:
        cols = 3
    else:
        cols = 4

    # Fetch and normalize atlas slices
    slices: list[tuple[Image.Image, float]] = []
    for pos in positions[:max_positions]:
        try:
            if show_borders:
                from langslice.atlas.core import get_composite_slice
                ref_img = get_composite_slice(atlas_obj, pos)
            else:
                from langslice.atlas.core import get_reference_slice
                ref_img = get_reference_slice(atlas_obj, pos)
            slices.append((normalize_image(ref_img), pos))
        except (ValueError, IndexError):
            pass

    if not slices:
        return Image.new("RGB", (grid_width, grid_width // 2), (0, 0, 0))

    label_height = 100
    gap = 12

    try:
        font_large = ImageFont.truetype("arial.ttf", 70)
        font_small = ImageFont.truetype("arial.ttf", 50)
    except OSError:
        font_large = ImageFont.load_default()
        font_small = font_large

    # Calculate atlas grid dimensions
    cell_width = grid_width // cols
    sample_w, sample_h = slices[0][0].size
    aspect = sample_h / sample_w
    cell_img_height = int(cell_width * aspect)
    cell_height = cell_img_height + label_height
    rows = (len(slices) + cols - 1) // cols
    grid_height = rows * cell_height

    # Scale target to match grid height (preserve aspect ratio)
    target_section_width = 0
    target_resized = None
    if target_image is not None:
        tw, th = target_image.size
        target_img_height = grid_height - label_height
        target_scale = target_img_height / th
        target_section_width = int(tw * target_scale) + gap
        target_resized = target_image.resize(
            (int(tw * target_scale), target_img_height),
            Image.Resampling.LANCZOS,
        )

    total_width = target_section_width + grid_width
    canvas = Image.new("RGB", (total_width, grid_height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    # Draw target on left
    if target_resized is not None:
        canvas.paste(target_resized.convert("RGB"), (0, 0))
        label = "TARGET SLICE"
        bbox = draw.textbbox((0, 0), label, font=font_large)
        text_w = bbox[2] - bbox[0]
        text_x = (target_section_width - gap - text_w) // 2
        text_y = grid_height - label_height + 10
        draw.text((text_x, text_y), label, fill=(255, 200, 0), font=font_large)

    # Draw 2x2 atlas grid on right
    for idx, (img, pos) in enumerate(slices):
        row = idx // cols
        col = idx % cols
        x = target_section_width + col * cell_width
        y = row * cell_height

        upscaled = img.resize(
            (cell_width - gap, cell_img_height), Image.Resampling.LANCZOS
        )
        canvas.paste(upscaled.convert("RGB"), (x, y))

        label = f"[{idx + 1}] {pos:.2f} mm"
        bbox = draw.textbbox((0, 0), label, font=font_small)
        text_w = bbox[2] - bbox[0]
        text_x = x + (cell_width - gap - text_w) // 2
        text_y = y + cell_img_height + 8
        draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font_small)

    return canvas


def _sorted_unique_positions(
    positions: list[float],
    *,
    tolerance: float = 0.02,
) -> list[float]:
    unique_positions: list[float] = []
    for pos in sorted(positions):
        if not unique_positions or abs(pos - unique_positions[-1]) > tolerance:
            unique_positions.append(pos)
    return unique_positions


def _is_broad_multi_sweep(positions: list[float]) -> bool:
    # Broad sweep no longer enforced — any grid/multi fetch counts
    if len(positions) < 3:
        return False
    return True


def _is_narrow_multi_sweep(positions: list[float]) -> bool:
    if len(positions) < 3:
        return False
    # Require a tight sweep: total span ≤ 1.0mm (was 1.5mm)
    return (max(positions) - min(positions)) <= 1.0


def _has_neighbor_bracket(
    fetched_positions: list[float],
    center_mm: float,
    *,
    pos_lo: float,
    pos_hi: float,
    tolerance: float = 0.25,
    edge_margin: float = 0.25,
) -> bool:
    unique_positions = _sorted_unique_positions(fetched_positions)
    has_lower = any(center_mm - tolerance <= pos < center_mm for pos in unique_positions)
    has_upper = any(center_mm < pos <= center_mm + tolerance for pos in unique_positions)

    needs_lower = center_mm > pos_lo + edge_margin
    needs_upper = center_mm < pos_hi - edge_margin
    return (has_lower or not needs_lower) and (has_upper or not needs_upper)


# ---------------------------------------------------------------------------
# Function call extraction (generate_content API)
# ---------------------------------------------------------------------------


def _extract_function_calls(
    response: object,
) -> tuple[list[dict[str, object]], str | None]:
    """Extract function calls from a generate_content response.

    Returns (function_calls, text_preview) where each function call is
    ``{"call_id": str|None, "name": str, "args": dict}``.
    """
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


# ---------------------------------------------------------------------------
# Nudge text builder
# ---------------------------------------------------------------------------


def _build_nudge_text(state: _NudgeState, submit_tool: str = "submit_estimate") -> str:
    if not state.saw_broad_sweep:
        return (
            "Please continue. Call `fetch_atlas` with widely spaced positions "
            "(e.g., [2, 4, 6, 8, 10]) to find the correct neighborhood."
        )
    if not state.saw_narrow_sweep:
        return (
            "Please narrow down. Call `fetch_atlas` with tightly spaced positions "
            "around your best candidate (e.g., [4.0, 4.2, 4.4, 4.6, 4.8])."
        )
    return (
        "Please continue. Verify your candidate by checking nearby positions "
        f"with `fetch_atlas`, or call `{submit_tool}` if confident."
    )


# ---------------------------------------------------------------------------
# Shared fetch_atlas handler
# ---------------------------------------------------------------------------


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
    atlas_resolution: int,
    target_image: Image.Image | None,
    stage: str,
    on_progress: Callable[[str], None] | None,
    on_trace: Callable[[dict[str, object]], None] | None,
) -> tuple[list[genai_types.Part], str]:
    """Handle a ``fetch_atlas`` tool call shared by single- and multi-slice.

    Returns ``(result_parts, function_name)`` where *result_parts* is a flat
    list of ``Part`` objects (function_response + any image/text parts) and
    *function_name* is always ``"fetch_atlas"``.

    The caller is responsible for splitting the parts into ``role='tool'``
    (function_response) and ``role='user'`` (image/text) contents.
    """
    atlas_obj = cast(Any, atlas)
    result_parts: list[genai_types.Part] = []
    fn_name = "fetch_atlas"

    positions_list = args.get("positions_mm", [])
    if not isinstance(positions_list, list):
        positions_list = []

    positions = [max(pos_lo, min(pos_hi, float(p))) for p in positions_list[:8]]

    if not positions:
        result_parts.append(
            genai_types.Part.from_function_response(
                name=fn_name,
                response={"status": "error", "error": "No valid positions provided"},
            )
        )
        return result_parts, fn_name

    state.fetched_positions.extend(positions)
    if _is_broad_multi_sweep(positions):
        state.saw_broad_sweep = True
    if _is_narrow_multi_sweep(positions):
        state.saw_narrow_sweep = True

    pos_label = ", ".join(f"{p:.2f}" for p in positions)
    state.images_fetched += len(positions)

    if send_individually:
        result_parts.append(
            genai_types.Part.from_function_response(
                name=fn_name,
                response={
                    "status": "ok",
                    "positions_mm": positions,
                    "description": (
                        f"{len(positions)} atlas sections at: {pos_label} mm."
                    ),
                },
            )
        )
        for idx, pos in enumerate(positions):
            try:
                ref_bytes = _fetch_atlas_slice_bytes(
                    atlas_obj,
                    pos,
                    max_long_edge=atlas_resolution,
                    show_borders=show_borders,
                )
                result_parts.append(
                    genai_types.Part.from_text(text=f"[{idx + 1}] {pos:.2f} mm:")
                )
                result_parts.append(
                    genai_types.Part.from_bytes(data=ref_bytes, mime_type="image/jpeg")
                )
            except (ValueError, IndexError):
                pass
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
        result_parts.append(
            genai_types.Part.from_function_response(
                name=fn_name,
                response={
                    "status": "ok",
                    "positions_mm": positions,
                    "description": (
                        f"Grid of {len(positions)} atlas sections at: {pos_label} mm. "
                        "Each cell is numbered and labeled with its AP position."
                    ),
                },
            )
        )
        result_parts.append(
            genai_types.Part.from_bytes(data=grid_bytes, mime_type="image/jpeg")
        )

    state.reasoning_log.append(
        {
            "iteration": iteration + 1,
            "tool": fn_name,
            "args": args,
            "result": f"Atlas: [{pos_label}] mm",
        }
    )
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

    return result_parts, fn_name


# ---------------------------------------------------------------------------
# Main tool dispatch (generate_content API format)
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
) -> list[genai_types.Part]:
    """Process function calls and return generate_content Parts.

    Returns a flat list of Parts.  The caller splits them:
    - Parts with ``.function_response`` -> ``Content(role='tool')``
    - Other Parts (text labels, images) -> ``Content(role='user')``
    """
    result_parts: list[genai_types.Part] = []

    def _append_response(
        *,
        call_id: object,  # noqa: ARG001 — kept for parity
        name: str,
        response: dict[str, object],
        is_error: bool = False,  # noqa: ARG001 — kept for parity
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
                atlas_resolution=atlas_resolution,
                target_image=target_image,
                stage="ap",
                on_progress=on_progress,
                on_trace=on_trace,
            )
            result_parts.extend(fetch_parts)

        elif name == "submit_estimate":
            est_pos = float(args.get("position_mm", 0.0))
            est_reasoning = str(args.get("reasoning", ""))
            has_neighbor_check = _has_neighbor_bracket(
                state.fetched_positions,
                est_pos,
                pos_lo=pos_lo,
                pos_hi=pos_hi,
            )
            near_iteration_limit = iteration >= state.max_iterations - 2

            if not state.saw_broad_sweep and not near_iteration_limit:
                _append_response(
                    call_id=call_id,
                    name=name,
                    response={
                        "status": "error",
                        "error": (
                            "Run a broad `fetch_multiple_atlas_slices`"
                            " sweep before submitting."
                        ),
                    },
                    is_error=True,
                )
                state.reasoning_log.append(
                    {
                        "iteration": iteration + 1,
                        "tool": name,
                        "args": args,
                        "result": f"Rejected submit at {est_pos:.2f}mm: no broad sweep yet",
                    }
                )
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
                _append_response(
                    call_id=call_id,
                    name=name,
                    response={
                        "status": "error",
                        "error": (
                            "Run a narrowed"
                            " `fetch_multiple_atlas_slices`"
                            " sweep around your best candidate"
                            " before submitting."
                        ),
                    },
                    is_error=True,
                )
                state.reasoning_log.append(
                    {
                        "iteration": iteration + 1,
                        "tool": name,
                        "args": args,
                        "result": f"Rejected submit at {est_pos:.2f}mm: no narrow sweep yet",
                    }
                )
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
                _append_response(
                    call_id=call_id,
                    name=name,
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
                    is_error=True,
                )
                state.reasoning_log.append(
                    {
                        "iteration": iteration + 1,
                        "tool": name,
                        "args": args,
                        "result": f"Rejected submit at {est_pos:.2f}mm: neighborhood not bracketed",
                    }
                )
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
            state.reasoning_log.append(
                {
                    "iteration": iteration + 1,
                    "tool": name,
                    "args": args,
                    "result": f"Submitted {est_pos:.2f}mm",
                }
            )
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

        else:
            _append_response(
                call_id=call_id,
                name=name,
                response={"status": "error", "error": f"Unknown tool: {name}"},
                is_error=True,
            )

    return result_parts

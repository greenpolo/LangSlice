"""Provider-agnostic tool schemas and validation helpers for AP estimation."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from PIL import Image

from langslice_harness.agent_trace import tool_result_event
from langslice_harness.estimation._shared_common import (
    _APLoopState,
    _emit_trace,
    _fetch_atlas_slice_bytes,
    _GroupLoopState,
    _image_to_bytes,
)
from langslice_harness.image_prep import normalize_image


class _NudgeState(Protocol):
    """Minimal interface for _build_nudge_text."""

    saw_broad_sweep: bool
    saw_narrow_sweep: bool


FETCH_ATLAS_SCHEMA = {
    "name": "fetch_atlas",
    "description": (
        "Fetch atlas coronal sections at specific AP positions. Returns "
        "a single labeled grid image for direct visual comparison. You "
        "choose exactly which positions to see (1 to 8). Use this to "
        "compare multiple positions at once - you can space them however "
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
                    "positions you want - you can cluster them densely "
                    "around a candidate or spread them widely."
                ),
            },
        },
        "required": ["positions_mm"],
    },
}


SUBMIT_ESTIMATE_SCHEMA = {
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
                "description": "Final estimated AP position in mm from the anterior edge",
            },
            "reasoning": {
                "type": "string",
                "description": "Detailed reasoning for the estimate",
            },
        },
        "required": ["position_mm", "reasoning"],
    },
}


def submit_group_estimate_schema(n_slices: int) -> dict[str, object]:
    return {
        "name": "submit_group_estimate",
        "description": f"Submit final AP estimates for all {n_slices} slices.",
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
    }


def _build_atlas_grid(
    atlas: object,
    positions: list[float],
    *,
    target_image: Image.Image | None = None,
    grid_width: int = 2048,
    cell_width: int | None = None,
    show_borders: bool = False,
    max_positions: int = 8,
) -> Image.Image:
    """Build a comparison image: target slice (left) + 2x2 atlas grid (right)."""
    from PIL import ImageDraw, ImageFont

    atlas_obj = cast(Any, atlas)
    n = min(len(positions), max_positions)

    if n <= 2:
        cols = n
    elif n <= 4:
        cols = 2
    elif n <= 6:
        cols = 3
    else:
        cols = 4

    if cell_width is not None:
        grid_width = cell_width * cols

    slices: list[tuple[Image.Image, float]] = []
    for pos in positions[:max_positions]:
        try:
            if show_borders:
                from langslice_harness.atlas.core import get_composite_slice

                ref_img = get_composite_slice(atlas_obj, pos)
            else:
                from langslice_harness.atlas.core import get_reference_slice

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

    cell_width = grid_width // cols
    sample_w, sample_h = slices[0][0].size
    aspect = sample_h / sample_w
    cell_img_height = int(cell_width * aspect)
    cell_height = cell_img_height + label_height
    rows = (len(slices) + cols - 1) // cols
    grid_height = rows * cell_height

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

    if target_resized is not None:
        canvas.paste(target_resized.convert("RGB"), (0, 0))
        label = "TARGET SLICE"
        bbox = draw.textbbox((0, 0), label, font=font_large)
        text_w = bbox[2] - bbox[0]
        text_x = (target_section_width - gap - text_w) // 2
        text_y = grid_height - label_height + 10
        draw.text((text_x, text_y), label, fill=(255, 200, 0), font=font_large)

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
    if len(positions) < 3:
        return False
    return True


def _is_narrow_multi_sweep(positions: list[float]) -> bool:
    if len(positions) < 3:
        return False
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


def _validate_submit_estimate(
    *,
    state: _APLoopState,
    est_pos: float,
    pos_lo: float,
    pos_hi: float,
    iteration: int,
) -> tuple[dict[str, object] | None, str | None]:
    has_neighbor_check = _has_neighbor_bracket(
        state.fetched_positions,
        est_pos,
        pos_lo=pos_lo,
        pos_hi=pos_hi,
    )
    near_iteration_limit = iteration >= state.max_iterations - 2

    if not state.saw_broad_sweep and not near_iteration_limit:
        return (
            {
                "status": "error",
                "error": (
                    "Run a broad `fetch_multiple_atlas_slices`"
                    " sweep before submitting."
                ),
            },
            f"Rejected submit at {est_pos:.2f}mm: no broad sweep yet",
        )

    if not state.saw_narrow_sweep and not near_iteration_limit:
        return (
            {
                "status": "error",
                "error": (
                    "Run a narrowed"
                    " `fetch_multiple_atlas_slices`"
                    " sweep around your best candidate"
                    " before submitting."
                ),
            },
            f"Rejected submit at {est_pos:.2f}mm: no narrow sweep yet",
        )

    if not has_neighbor_check and not near_iteration_limit:
        lower = max(pos_lo, est_pos - 0.2)
        upper = min(pos_hi, est_pos + 0.2)
        return (
            {
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
            f"Rejected submit at {est_pos:.2f}mm: neighborhood not bracketed",
        )

    return None, None


def _validate_submit_group_estimate(
    *,
    state: _GroupLoopState,
    positions_list: list[Any],
    pos_lo: float,
    pos_hi: float,
    iteration: int,
) -> tuple[dict[str, object] | None, list[float] | None, str | None]:
    if len(positions_list) != state.n_slices:
        return (
            {
                "status": "error",
                "error": (
                    f"Expected {state.n_slices} positions, "
                    f"got {len(positions_list)}."
                ),
            },
            None,
            f"Rejected: wrong count ({len(positions_list)} vs {state.n_slices})",
        )

    near_iteration_limit = iteration >= state.max_iterations - 2

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
        return (
            {
                "status": "error",
                "error": (
                    f"Some positions are outside the atlas range "
                    f"[{pos_lo:.2f}, {pos_hi:.2f}]mm: {detail}. "
                    f"Please correct and resubmit."
                ),
            },
            None,
            f"Rejected: out of range ({detail})",
        )

    positions_list = clamped_positions

    if not near_iteration_limit:
        is_monotonic = all(
            positions_list[i] <= positions_list[i + 1]
            for i in range(len(positions_list) - 1)
        )
        if not is_monotonic:
            return (
                {
                    "status": "error",
                    "error": (
                        "Positions must be strictly increasing "
                        "(anterior-to-posterior order). Please fix "
                        "the ordering and resubmit."
                    ),
                },
                None,
                "Rejected: positions not strictly increasing",
            )

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
            return (
                {
                    "status": "error",
                    "error": (
                        f"Some intervals deviate >50% from the expected "
                        f"{state.interval_mm:.3f}mm: {detail}. "
                        f"Please reconsider and resubmit."
                    ),
                },
                None,
                f"Rejected: bad intervals ({detail})",
            )

    if not state.saw_broad_sweep and not near_iteration_limit:
        return (
            {
                "status": "error",
                "error": "Run a broad `fetch_atlas` sweep before submitting.",
            },
            None,
            "Rejected: no broad sweep yet",
        )

    if not state.saw_narrow_sweep and not near_iteration_limit:
        return (
            {
                "status": "error",
                "error": "Run a narrow `fetch_atlas` sweep before submitting.",
            },
            None,
            "Rejected: no narrow sweep yet",
        )

    return None, [float(p) for p in positions_list], None


@dataclass
class FetchAtlasResult:
    function_name: str
    response: dict[str, object]
    labeled_images: list[tuple[str, bytes]] = field(default_factory=list)
    grid_image_bytes: bytes | None = None


def _handle_fetch_atlas_core(
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
    on_trace: Callable[[dict[str, object]], None] | None,
) -> FetchAtlasResult:
    atlas_obj = cast(Any, atlas)
    fn_name = "fetch_atlas"

    positions_list = args.get("positions_mm", [])
    if not isinstance(positions_list, list):
        positions_list = []

    positions = [max(pos_lo, min(pos_hi, float(p))) for p in positions_list[:8]]

    if not positions:
        return FetchAtlasResult(
            function_name=fn_name,
            response={"status": "error", "error": "No valid positions provided"},
        )

    state.fetched_positions.extend(positions)
    if _is_broad_multi_sweep(positions):
        state.saw_broad_sweep = True
    if _is_narrow_multi_sweep(positions):
        state.saw_narrow_sweep = True

    pos_label = ", ".join(f"{p:.2f}" for p in positions)
    state.images_fetched += len(positions)

    if send_individually:
        labeled_images: list[tuple[str, bytes]] = []
        for idx, pos in enumerate(positions):
            try:
                ref_bytes = _fetch_atlas_slice_bytes(
                    atlas_obj,
                    pos,
                    show_borders=show_borders,
                )
                labeled_images.append((f"[{idx + 1}] {pos:.2f} mm:", ref_bytes))
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
        result = FetchAtlasResult(
            function_name=fn_name,
            response={
                "status": "ok",
                "positions_mm": positions,
                "description": (
                    f"{len(positions)} atlas sections at: {pos_label} mm."
                ),
            },
            labeled_images=labeled_images,
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
        result = FetchAtlasResult(
            function_name=fn_name,
            response={
                "status": "ok",
                "positions_mm": positions,
                "description": (
                    f"Grid of {len(positions)} atlas sections at: {pos_label} mm. "
                    "Each cell is numbered and labeled with its AP position."
                ),
            },
            grid_image_bytes=grid_bytes,
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

    return result

"""Tool handler logic for the AP estimation loop.

Functions in this module implement the tool-use dispatch and helpers for
the agentic AP estimator.  They were extracted from ``estimator.py`` purely
for readability — no behavioral changes.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np
from PIL import Image
from google.genai import types
from langslice.agent_trace import (
    image_part_from_pil,
    json_part,
    tool_call_event,
    tool_result_event,
)
from langslice.image_prep import normalize_image

if TYPE_CHECKING:
    from langslice.ai.estimator import _APLoopState

_RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------


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
    if len(positions) < 4:
        return False
    return (max(positions) - min(positions)) >= 3.0


def _is_narrow_multi_sweep(positions: list[float]) -> bool:
    if len(positions) < 3:
        return False
    return (max(positions) - min(positions)) <= 1.5


def _has_neighbor_bracket(
    fetched_positions: list[float],
    center_mm: float,
    *,
    pos_lo: float,
    pos_hi: float,
    tolerance: float = 0.35,
    edge_margin: float = 0.25,
) -> bool:
    unique_positions = _sorted_unique_positions(fetched_positions)
    has_lower = any(center_mm - tolerance <= pos < center_mm for pos in unique_positions)
    has_upper = any(center_mm < pos <= center_mm + tolerance for pos in unique_positions)

    needs_lower = center_mm > pos_lo + edge_margin
    needs_upper = center_mm < pos_hi - edge_margin
    return (has_lower or not needs_lower) and (has_upper or not needs_upper)


def _get_regions_at_position(atlas: object, position_mm: float) -> list[str]:
    """Return brain region names visible at a given AP position."""
    from langslice.atlas.core import position_mm_to_index

    try:
        idx = position_mm_to_index(cast(Any, atlas), position_mm)
    except ValueError:
        return []

    atlas_obj = cast(Any, atlas)
    annotation_slice = np.asarray(atlas_obj.annotation[idx, :, :])
    unique_ids = np.unique(annotation_slice)
    unique_ids = unique_ids[unique_ids > 0]

    structures = atlas_obj.structures
    names: list[str] = []
    for uid in unique_ids[:30]:  # Cap at 30 to avoid huge lists
        uid_int = int(uid)
        if uid_int in structures:
            entry = structures[uid_int]
            names.append(f"{entry['acronym']} ({entry['name']})")
    return names


# ---------------------------------------------------------------------------
# Function call extraction
# ---------------------------------------------------------------------------


def _extract_generate_function_calls(
    model_content: types.Content,
) -> tuple[list[dict[str, object]], str | None]:
    model_parts = getattr(model_content, "parts", None) or []
    text_preview: str | None = None
    function_calls: list[dict[str, object]] = []
    for part in model_parts:
        text = getattr(part, "text", None)
        if text_preview is None and isinstance(text, str) and text:
            text_preview = text
        function_call = getattr(part, "function_call", None)
        if function_call is None:
            continue
        args = dict(function_call.args) if getattr(function_call, "args", None) else {}
        function_calls.append(
            {
                "call_id": None,
                "name": getattr(function_call, "name", ""),
                "args": args,
            }
        )
    return function_calls, text_preview


def _extract_interaction_function_calls(
    interaction: object,
) -> tuple[list[dict[str, object]], str | None]:
    outputs = getattr(interaction, "outputs", None) or []
    text_preview: str | None = None
    function_calls: list[dict[str, object]] = []
    for output in outputs:
        output_type = getattr(output, "type", None)
        if output_type == "text" and text_preview is None:
            text = getattr(output, "text", None)
            if isinstance(text, str) and text:
                text_preview = text
        if output_type != "function_call":
            continue
        arguments = getattr(output, "arguments", None)
        args = dict(arguments) if isinstance(arguments, dict) else {}
        function_calls.append(
            {
                "call_id": getattr(output, "id", None),
                "name": getattr(output, "name", ""),
                "args": args,
            }
        )
    return function_calls, text_preview


# ---------------------------------------------------------------------------
# Nudge text builder
# ---------------------------------------------------------------------------


def _build_nudge_text(state: _APLoopState) -> str:
    if not state.saw_broad_sweep:
        return (
            "Please continue with a broad coarse sweep now. Call `fetch_multiple_atlas_slices` "
            "with 4-5 widely spaced AP positions to find the correct neighborhood before reasoning further."
        )
    if not state.saw_narrow_sweep:
        return (
            "Please continue with a narrowed sweep now. Call `fetch_multiple_atlas_slices` "
            "around your best current neighborhood with tighter spacing before considering submission."
        )
    return (
        "Please continue. Before submitting, verify your leading candidate by checking at least one "
        "lower and one higher neighboring AP position around it using `fetch_multiple_atlas_slices` or `fetch_atlas_slice`."
    )


# ---------------------------------------------------------------------------
# Main tool dispatch
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
    client: Any,
    use_file_api: bool,
    uploaded_file_names: list[str],
    state: _APLoopState,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
) -> tuple[list[types.Part], list[dict[str, object]]]:
    # Lazy imports to avoid circular dependency with estimator.py
    from langslice.ai.estimator import _build_image_payload, _emit_trace, _image_to_bytes

    atlas_obj = cast(Any, atlas)
    generate_parts: list[types.Part] = []
    interaction_inputs: list[dict[str, object]] = []

    def _append_response(
        *,
        call_id: object,
        name: str,
        response: dict[str, object],
        is_error: bool = False,
    ) -> None:
        generate_parts.append(types.Part.from_function_response(name=name, response=response))
        interaction_inputs.append(
            {
                "type": "function_result",
                "call_id": str(call_id)
                if isinstance(call_id, str) and call_id
                else f"{iteration + 1}:{name}",
                "name": name,
                "result": response,
                "is_error": is_error,
            }
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

        if name == "fetch_atlas_slice":
            pos = float(args.get("position_mm", (pos_lo + pos_hi) / 2))
            pos = max(pos_lo, min(pos_hi, pos))
            state.fetched_positions.append(pos)
            try:
                from langslice.atlas.core import get_reference_slice

                ref_img = get_reference_slice(atlas_obj, pos)
                ref_prepared = normalize_image(ref_img)
                scale = target_h / ref_prepared.height
                new_w = max(1, int(round(ref_prepared.width * scale)))
                new_h = max(1, int(round(ref_prepared.height * scale)))
                ref_scaled = ref_prepared.resize((new_w, new_h), _RESAMPLE_LANCZOS)
                ref_bytes = _image_to_bytes(ref_scaled)
                image_payload = _build_image_payload(
                    client,
                    image_bytes=ref_bytes,
                    display_name=f"ap_{iteration + 1:02d}_slice_{pos:.3f}mm",
                    use_file_api=use_file_api,
                    uploaded_file_names=uploaded_file_names,
                    on_progress=on_progress,
                )
                state.images_fetched += 1

                if run_dir:
                    ref_scaled.save(
                        os.path.join(run_dir, f"tool_{iteration + 1:02d}_slice_{pos:.2f}mm.jpg"),
                        quality=95,
                    )

                _append_response(
                    call_id=call_id,
                    name=name,
                    response={
                        "position_mm": pos,
                        "status": "ok",
                        "description": f"Atlas coronal section at {pos:.2f}mm from anterior edge",
                    },
                )
                generate_parts.append(image_payload.part)
                if image_payload.interaction_input is not None:
                    interaction_inputs.append(image_payload.interaction_input)
                state.reasoning_log.append(
                    {
                        "iteration": iteration + 1,
                        "tool": name,
                        "args": args,
                        "result": f"Image at {pos:.2f}mm via {image_payload.transport}",
                    }
                )
                image_path = (
                    os.path.join(run_dir, f"tool_{iteration + 1:02d}_slice_{pos:.2f}mm.jpg")
                    if run_dir
                    else None
                )
                _emit_trace(
                    on_trace,
                    tool_result_event(
                        stage="ap",
                        tool_name=name,
                        summary=f"Fetched atlas slice at {pos:.2f} mm",
                        parts=[
                            image_part_from_pil(
                                ref_scaled,
                                label=f"Atlas slice {pos:.2f} mm",
                                image_bytes=ref_bytes,
                                path=image_path,
                                metadata={"transport": image_payload.transport},
                            )
                        ],
                        metadata={
                            "iteration": iteration + 1,
                            "position_mm": round(pos, 3),
                            "transport": image_payload.transport,
                        },
                    ),
                )
            except ValueError as exc:
                _append_response(
                    call_id=call_id,
                    name=name,
                    response={"status": "error", "error": str(exc)},
                    is_error=True,
                )
                state.reasoning_log.append(
                    {
                        "iteration": iteration + 1,
                        "tool": name,
                        "args": args,
                        "result": f"Error: {exc}",
                    }
                )
                _emit_trace(
                    on_trace,
                    tool_result_event(
                        stage="ap",
                        tool_name=name,
                        summary=f"Error: {exc}",
                        parts=[json_part({"error": str(exc)}, label="Error")],
                        metadata={"iteration": iteration + 1, "status": "error"},
                    ),
                )

        elif name == "fetch_multiple_atlas_slices":
            positions_list = args.get("positions_mm", [])
            if not isinstance(positions_list, list):
                positions_list = []

            positions = [max(pos_lo, min(pos_hi, float(p))) for p in positions_list[:5]]
            state.fetched_positions.extend(positions)
            if positions and _is_broad_multi_sweep(positions):
                state.saw_broad_sweep = True
            if positions and _is_narrow_multi_sweep(positions):
                state.saw_narrow_sweep = True

            if not positions:
                _append_response(
                    call_id=call_id,
                    name=name,
                    response={"status": "error", "error": "No valid positions provided"},
                    is_error=True,
                )
                state.reasoning_log.append(
                    {
                        "iteration": iteration + 1,
                        "tool": name,
                        "args": args,
                        "result": "Error: empty input",
                    }
                )
                continue

            successes: list[str] = []
            image_parts: list[dict[str, object]] = []
            from langslice.atlas.core import get_reference_slice

            for pos in positions:
                try:
                    ref_img = get_reference_slice(atlas_obj, pos)
                    ref_prepared = normalize_image(ref_img)
                    scale = target_h / ref_prepared.height
                    new_w = max(1, int(round(ref_prepared.width * scale)))
                    new_h = max(1, int(round(ref_prepared.height * scale)))
                    ref_scaled = ref_prepared.resize((new_w, new_h), _RESAMPLE_LANCZOS)
                    ref_bytes = _image_to_bytes(ref_scaled)
                    image_payload = _build_image_payload(
                        client,
                        image_bytes=ref_bytes,
                        display_name=f"ap_{iteration + 1:02d}_multi_{pos:.3f}mm",
                        use_file_api=use_file_api,
                        uploaded_file_names=uploaded_file_names,
                        on_progress=on_progress,
                    )
                    state.images_fetched += 1

                    if run_dir:
                        ref_scaled.save(
                            os.path.join(
                                run_dir, f"tool_{iteration + 1:02d}_multi_{pos:.2f}mm.jpg"
                            ),
                            quality=95,
                        )

                    _append_response(
                        call_id=call_id,
                        name=name,
                        response={
                            "position_mm": pos,
                            "status": "ok",
                            "description": f"Atlas coronal section at {pos:.2f}mm",
                        },
                    )
                    generate_parts.append(image_payload.part)
                    if image_payload.interaction_input is not None:
                        interaction_inputs.append(image_payload.interaction_input)
                    successes.append(f"{pos:.2f}mm")
                    image_path = (
                        os.path.join(run_dir, f"tool_{iteration + 1:02d}_multi_{pos:.2f}mm.jpg")
                        if run_dir
                        else None
                    )
                    image_parts.append(
                        image_part_from_pil(
                            ref_scaled,
                            label=f"Atlas slice {pos:.2f} mm",
                            image_bytes=ref_bytes,
                            path=image_path,
                            metadata={"transport": image_payload.transport},
                        )
                    )
                except Exception as exc:
                    _append_response(
                        call_id=call_id,
                        name=name,
                        response={"position_mm": pos, "status": "error", "error": str(exc)},
                        is_error=True,
                    )

            state.reasoning_log.append(
                {
                    "iteration": iteration + 1,
                    "tool": name,
                    "args": args,
                    "result": f"Fetched {len(successes)} slices: {', '.join(successes)}",
                }
            )
            _emit_trace(
                on_trace,
                tool_result_event(
                    stage="ap",
                    tool_name=name,
                    summary=f"Fetched {len(successes)} atlas slices",
                    parts=image_parts
                    or [json_part({"positions_mm": positions}, label="Positions")],
                    metadata={"iteration": iteration + 1, "positions_mm": positions},
                ),
            )

        elif name == "get_atlas_info":
            from langslice.atlas.core import get_atlas_info as _get_atlas_info_core

            info = _get_atlas_info_core(atlas_obj)
            info["coordinate_note"] = "0.0mm is extreme Anterior; higher mm is more Posterior."
            _append_response(call_id=call_id, name=name, response=info)
            state.reasoning_log.append(
                {"iteration": iteration + 1, "tool": name, "args": {}, "result": str(info)}
            )
            _emit_trace(
                on_trace,
                tool_result_event(
                    stage="ap",
                    tool_name=name,
                    summary="Atlas metadata returned",
                    parts=[json_part(info, label="Atlas info")],
                    metadata={"iteration": iteration + 1},
                ),
            )
            if on_progress:
                on_progress(f"  -> Atlas range: [{pos_lo:.2f}, {pos_hi:.2f}] mm")

        elif name == "get_region_names":
            pos = float(args.get("position_mm", (pos_lo + pos_hi) / 2))
            regions = _get_regions_at_position(atlas, pos)
            _append_response(
                call_id=call_id,
                name=name,
                response={"position_mm": pos, "regions": regions},
            )
            state.reasoning_log.append(
                {
                    "iteration": iteration + 1,
                    "tool": name,
                    "args": args,
                    "result": f"{len(regions)} regions",
                }
            )
            _emit_trace(
                on_trace,
                tool_result_event(
                    stage="ap",
                    tool_name=name,
                    summary=f"Returned {len(regions)} visible regions at {pos:.2f} mm",
                    parts=[json_part({"position_mm": pos, "regions": regions}, label="Regions")],
                    metadata={"iteration": iteration + 1, "position_mm": round(pos, 3)},
                ),
            )
            if on_progress:
                on_progress(f"  -> {len(regions)} regions at {pos:.2f}mm")

        elif name == "submit_estimate":
            est_pos = float(args.get("position_mm", 0.0))
            est_confidence = str(args.get("confidence", "unknown"))
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
                        "error": "Run a broad `fetch_multiple_atlas_slices` sweep before submitting.",
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
                        "error": "Run a narrowed `fetch_multiple_atlas_slices` sweep around your best candidate before submitting.",
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
                            "Before submitting, verify at least one lower and one higher neighboring AP position "
                            f"around {est_pos:.2f} mm (for example {lower:.2f} mm and {upper:.2f} mm)."
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
                "confidence": est_confidence,
                "reasoning": est_reasoning,
            }
            state.reasoning_log.append(
                {
                    "iteration": iteration + 1,
                    "tool": name,
                    "args": args,
                    "result": f"Submitted {est_pos:.2f}mm ({est_confidence})",
                }
            )
            if on_progress:
                on_progress(
                    f"Agent submitted estimate: {est_pos:.2f}mm (confidence: {est_confidence})"
                )
            _emit_trace(
                on_trace,
                tool_result_event(
                    stage="ap",
                    tool_name=name,
                    summary=f"Submitted estimate {est_pos:.2f} mm ({est_confidence})",
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

    return generate_parts, interaction_inputs

"""h10_tool_code.py — fetch_atlas_group tool declaration and handler.

Sliding-window atlas alignment tool for multi-slice AP estimation.
The model specifies a single starting_ap_mm and receives N atlas sections
spaced at the declared interval — one section per input slice — for direct
1:1 comparison.

HOW TO INTEGRATE
----------------
1. Tool declaration  → add to the list returned by ``_group_tool_dicts()``
   in ``langslice/estimation/google/ap_multi_slice.py``, alongside the
   existing ``fetch_atlas`` / ``get_atlas_info`` / ``get_region_names`` /
   ``submit_group_estimate`` dicts.

2. Handler elif block → add inside the ``for call in function_calls:`` loop
   in ``_process_group_function_calls()``, between the ``fetch_atlas`` block
   and the ``get_atlas_info`` block (i.e. after the closing ``_emit_trace``
   of ``fetch_atlas`` and before ``elif name == "get_atlas_info":``).

The handler assumes the following names are already in scope (they are in the
real function):
    atlas, atlas_obj, pos_lo, pos_hi, run_dir, state, iteration,
    media_resolution, show_borders, send_individually, atlas_resolution,
    on_progress, on_trace,
    _append_response, _append_image,
    _is_broad_multi_sweep, _is_narrow_multi_sweep,
    _build_atlas_grid, _image_to_bytes, _emit_trace,
    tool_result_event, json_part,
    cast, Any, os, json
"""

from __future__ import annotations

from typing import Any


# ===========================================================================
# 1. TOOL DECLARATION
# Place inside _group_tool_dicts(), after the fetch_atlas dict and before
# get_atlas_info.
# ===========================================================================

def _fetch_atlas_group_tool_dict(
    n_slices: int, interval_mm: float
) -> dict[str, Any]:
    """Return the fetch_atlas_group tool declaration dict.

    In the real file, ``n_slices`` and ``interval_mm`` are already in scope
    inside ``_group_tool_dicts(n_slices)`` (interval_mm comes from the outer
    closure via ``state`` — but at declaration time it is passed in as a
    parameter to ``_group_tool_dicts``; see integration note below).

    NOTE: ``_group_tool_dicts`` currently only receives ``n_slices``.  To make
    the description dynamic you will need to also pass ``interval_mm``:

        def _group_tool_dicts(n_slices: int, interval_mm: float) -> ...:

    Alternatively the description can hard-code "the declared interval" and
    the exact value can be omitted from the description string — that is the
    safe option if you don't want to change the signature.
    """
    return {
        "type": "function",
        "name": "fetch_atlas_group",
        "description": (
            f"Fetch {n_slices} atlas sections starting at a given AP position, "
            f"spaced at the declared {interval_mm:.3f}mm interval. Returns "
            "one atlas section per slice for direct 1:1 comparison. Use this "
            "to check if the group matches the atlas starting at a candidate AP."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "starting_ap_mm": {
                    "type": "number",
                    "description": (
                        "AP position in mm for the first slice (most anterior)."
                    ),
                },
            },
            "required": ["starting_ap_mm"],
        },
    }


# ===========================================================================
# 2. HANDLER elif BLOCK
# The code below is written as a standalone function so it is easy to read
# and test, but the body should be pasted as an ``elif name == "fetch_atlas_group":``
# block inside the ``for call in function_calls:`` loop in
# ``_process_group_function_calls()``.
# ===========================================================================

def _handle_fetch_atlas_group(
    *,
    call: dict[str, object],
    iteration: int,
    atlas: object,
    pos_lo: float,
    pos_hi: float,
    run_dir: str | None,
    state: object,   # _GroupLoopState in the real file
    media_resolution: str,
    show_borders: bool,
    send_individually: bool,
    atlas_resolution: int,
    on_progress: Any,
    on_trace: Any,
    # helpers already in scope in the real function
    _append_response: Any,
    _append_image: Any,
) -> list[dict[str, object]]:
    """Standalone wrapper — in the real file just paste the body as an elif."""
    import base64
    import json
    import os
    from typing import cast

    from langslice.agent_trace import json_part, tool_result_event
    from langslice.estimation.google.ap_image_gen import _fetch_atlas_slice_bytes
    from langslice.estimation.google.ap_tool_use import _emit_trace, _image_to_bytes
    from langslice.estimation.google.tool_definitions import (
        _build_atlas_grid,
        _is_broad_multi_sweep,
        _is_narrow_multi_sweep,
    )

    # -- pull typed state fields (already typed in the real function) --------
    state_obj = cast(Any, state)
    atlas_obj = cast(Any, atlas)
    name = "fetch_atlas_group"
    args_obj = call.get("args", {})
    args: dict[str, object] = args_obj if isinstance(args_obj, dict) else {}
    call_id = call.get("call_id")

    interaction_inputs: list[dict[str, object]] = []

    # In the real file _append_response / _append_image are closures already
    # in scope; they are passed as parameters here only for the standalone
    # wrapper.

    # -- parse & validate argument -------------------------------------------
    raw_start = args.get("starting_ap_mm")
    if raw_start is None:
        _append_response(
            call_id=call_id,
            name=name,
            response={
                "status": "error",
                "error": "starting_ap_mm is required.",
            },
            is_error=True,
        )
        # In the real elif block you would ``continue`` here; in this wrapper
        # we return early so the caller can decide.
        return interaction_inputs

    starting_ap = float(raw_start)

    # -- compute per-slice positions (interval-spaced window) ----------------
    positions = [
        max(pos_lo, min(pos_hi, starting_ap + i * state_obj.interval_mm))
        for i in range(state_obj.n_slices)
    ]

    # -- update state --------------------------------------------------------
    state_obj.fetched_positions.extend(positions)

    # The full span of the window counts as a broad sweep when there are ≥ 3
    # slices (same threshold as _is_broad_multi_sweep).
    if _is_broad_multi_sweep(positions):
        state_obj.saw_broad_sweep = True

    # The window is narrow when its total span fits within 1.0mm.
    if _is_narrow_multi_sweep(positions):
        state_obj.saw_narrow_sweep = True

    state_obj.images_fetched += len(positions)
    pos_label = ", ".join(f"{p:.2f}" for p in positions)

    # -- send images ---------------------------------------------------------
    if send_individually:
        from langslice.estimation.google.ap_image_gen import _fetch_atlas_slice_bytes

        _append_response(
            call_id=call_id,
            name=name,
            response={
                "status": "ok",
                "starting_ap_mm": starting_ap,
                "positions_mm": positions,
                "description": (
                    f"{len(positions)} atlas sections starting at "
                    f"{starting_ap:.2f}mm, spaced at "
                    f"{state_obj.interval_mm:.3f}mm: {pos_label} mm."
                ),
            },
        )
        for i, pos in enumerate(positions):
            try:
                ref_bytes = _fetch_atlas_slice_bytes(
                    atlas_obj,
                    pos,
                    max_long_edge=atlas_resolution,
                    show_borders=show_borders,
                )
                interaction_inputs.append(
                    {
                        "type": "text",
                        "text": f"Atlas for Slice {i + 1} at {pos:.2f}mm:",
                    }
                )
                _append_image(ref_bytes)
            except (ValueError, IndexError):
                pass

        if run_dir:
            grid_img = _build_atlas_grid(
                atlas, positions, show_borders=show_borders
            )
            grid_img.save(
                os.path.join(
                    run_dir,
                    f"tool_{iteration + 1:02d}_atlas_group_{len(positions)}x.jpg",
                ),
                quality=85,
            )
    else:
        # Grid fallback (send_individually=False path, mirrors fetch_atlas)
        grid_img = _build_atlas_grid(
            atlas,
            positions,
            show_borders=show_borders,
        )
        grid_bytes = _image_to_bytes(grid_img)
        if run_dir:
            grid_img.save(
                os.path.join(
                    run_dir,
                    f"tool_{iteration + 1:02d}_atlas_group_{len(positions)}x.jpg",
                ),
                quality=85,
            )
        _append_response(
            call_id=call_id,
            name=name,
            response={
                "status": "ok",
                "starting_ap_mm": starting_ap,
                "positions_mm": positions,
                "description": (
                    f"Grid of {len(positions)} atlas sections starting at "
                    f"{starting_ap:.2f}mm (one per slice). Each cell is "
                    f"numbered and labeled. Positions: {pos_label} mm."
                ),
            },
        )
        _append_image(grid_bytes)

    # -- reasoning log -------------------------------------------------------
    state_obj.reasoning_log.append(
        {
            "iteration": iteration + 1,
            "tool": name,
            "args": args,
            "result": f"Atlas group window: [{pos_label}] mm",
        }
    )

    # -- trace ---------------------------------------------------------------
    _emit_trace(
        on_trace,
        tool_result_event(
            stage="ap_group",
            tool_name=name,
            summary=(
                f"Returned {len(positions)} atlas sections "
                f"(window from {starting_ap:.2f}mm)"
            ),
            metadata={
                "iteration": iteration + 1,
                "starting_ap_mm": starting_ap,
                "positions": positions,
                "send_individually": send_individually,
            },
        ),
    )

    if on_progress:
        on_progress(
            f"  -> Atlas group window [{pos_label}] mm "
            f"(start {starting_ap:.2f}mm)"
        )

    return interaction_inputs


# ===========================================================================
# 3. READY-TO-PASTE elif BLOCK
# Copy everything between the dashed lines into the for-loop body in
# _process_group_function_calls(), directly after the closing _emit_trace
# call of the ``fetch_atlas`` block and before ``elif name == "get_atlas_info"``.
# ===========================================================================

ELIF_BLOCK = """
        # --- INSERT AFTER fetch_atlas block, BEFORE get_atlas_info block ---
        elif name == "fetch_atlas_group":
            raw_start = args.get("starting_ap_mm")
            if raw_start is None:
                _append_response(
                    call_id=call_id,
                    name=name,
                    response={
                        "status": "error",
                        "error": "starting_ap_mm is required.",
                    },
                    is_error=True,
                )
                continue

            starting_ap = float(raw_start)

            # Compute per-slice positions: interval-spaced sliding window
            positions = [
                max(pos_lo, min(pos_hi, starting_ap + i * state.interval_mm))
                for i in range(state.n_slices)
            ]

            state.fetched_positions.extend(positions)
            if _is_broad_multi_sweep(positions):
                state.saw_broad_sweep = True
            if _is_narrow_multi_sweep(positions):
                state.saw_narrow_sweep = True

            state.images_fetched += len(positions)
            pos_label = ", ".join(f"{p:.2f}" for p in positions)

            if send_individually:
                from langslice.estimation.google.ap_image_gen import (
                    _fetch_atlas_slice_bytes,
                )

                _append_response(
                    call_id=call_id,
                    name=name,
                    response={
                        "status": "ok",
                        "starting_ap_mm": starting_ap,
                        "positions_mm": positions,
                        "description": (
                            f"{len(positions)} atlas sections starting at "
                            f"{starting_ap:.2f}mm, spaced at "
                            f"{state.interval_mm:.3f}mm: {pos_label} mm."
                        ),
                    },
                )
                for i, pos in enumerate(positions):
                    try:
                        ref_bytes = _fetch_atlas_slice_bytes(
                            cast(Any, atlas),
                            pos,
                            max_long_edge=atlas_resolution,
                            show_borders=show_borders,
                        )
                        interaction_inputs.append(
                            {
                                "type": "text",
                                "text": f"Atlas for Slice {i + 1} at {pos:.2f}mm:",
                            }
                        )
                        _append_image(ref_bytes)
                    except (ValueError, IndexError):
                        pass
                if run_dir:
                    grid_img = _build_atlas_grid(
                        atlas, positions, show_borders=show_borders,
                    )
                    grid_img.save(
                        os.path.join(
                            run_dir,
                            f"tool_{iteration + 1:02d}_atlas_group_{len(positions)}x.jpg",
                        ),
                        quality=85,
                    )
            else:
                grid_img = _build_atlas_grid(
                    atlas,
                    positions,
                    show_borders=show_borders,
                )
                grid_bytes = _image_to_bytes(grid_img)
                if run_dir:
                    grid_img.save(
                        os.path.join(
                            run_dir,
                            f"tool_{iteration + 1:02d}_atlas_group_{len(positions)}x.jpg",
                        ),
                        quality=85,
                    )
                _append_response(
                    call_id=call_id,
                    name=name,
                    response={
                        "status": "ok",
                        "starting_ap_mm": starting_ap,
                        "positions_mm": positions,
                        "description": (
                            f"Grid of {len(positions)} atlas sections starting at "
                            f"{starting_ap:.2f}mm (one per slice). Each cell is "
                            f"numbered and labeled. Positions: {pos_label} mm."
                        ),
                    },
                )
                _append_image(grid_bytes)

            state.reasoning_log.append(
                {
                    "iteration": iteration + 1,
                    "tool": name,
                    "args": args,
                    "result": f"Atlas group window: [{pos_label}] mm",
                }
            )
            _emit_trace(
                on_trace,
                tool_result_event(
                    stage="ap_group",
                    tool_name=name,
                    summary=(
                        f"Returned {len(positions)} atlas sections "
                        f"(window from {starting_ap:.2f}mm)"
                    ),
                    metadata={
                        "iteration": iteration + 1,
                        "starting_ap_mm": starting_ap,
                        "positions": positions,
                        "send_individually": send_individually,
                    },
                ),
            )
            if on_progress:
                on_progress(
                    f"  -> Atlas group window [{pos_label}] mm "
                    f"(start {starting_ap:.2f}mm)"
                )
        # --- END fetch_atlas_group block ---
"""

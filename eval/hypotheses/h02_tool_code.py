"""H2: compare_slice_to_atlas tool implementation.

This tool takes a slice index and an AP position and returns a side-by-side
composite image of that specific input slice next to the atlas section at
that position.

Integration points:
1. Add tool declaration to _group_tool_dicts()
2. Add handler to _process_group_function_calls()
3. Pass prepared_images into _process_group_function_calls() as new kwarg
"""

# ---- Tool declaration (add to _group_tool_dicts return list) ----
TOOL_DECLARATION = {
    "type": "function",
    "name": "compare_slice_to_atlas",
    "description": (
        "Show a side-by-side comparison of one of your input slices with "
        "an atlas section at a specific AP position. Returns a single "
        "composite image with the input slice on the left and the atlas "
        "section on the right, making visual comparison easy."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "slice_index": {
                "type": "integer",
                "description": (
                    "Which input slice to compare (1-based: 1 = first/most "
                    "anterior slice)."
                ),
                "minimum": 1,
            },
            "position_mm": {
                "type": "number",
                "description": "AP position in mm for the atlas section to compare against.",
            },
        },
        "required": ["slice_index", "position_mm"],
    },
}


# ---- Handler block (add as elif in _process_group_function_calls) ----
# Assumes `prepared_images: list[Image.Image]` is passed as a new kwarg
# to _process_group_function_calls.
HANDLER_CODE = '''
        elif name == "compare_slice_to_atlas":
            slice_idx_raw = args.get("slice_index", 1)
            slice_idx = int(slice_idx_raw) - 1  # 0-based
            pos = float(args.get("position_mm", (pos_lo + pos_hi) / 2))
            pos = max(pos_lo, min(pos_hi, pos))

            if slice_idx < 0 or slice_idx >= state.n_slices:
                _append_response(
                    call_id=call_id,
                    name=name,
                    response={
                        "status": "error",
                        "error": (
                            f"slice_index must be 1-{state.n_slices}, "
                            f"got {slice_idx + 1}."
                        ),
                    },
                    is_error=True,
                )
                continue

            if prepared_images is None:
                _append_response(
                    call_id=call_id,
                    name=name,
                    response={
                        "status": "error",
                        "error": "Side-by-side comparison not available.",
                    },
                    is_error=True,
                )
                continue

            from langslice.estimation.google.ap_image_gen import (
                _fetch_atlas_slice_bytes,
            )

            try:
                # Get atlas slice
                ref_bytes = _fetch_atlas_slice_bytes(
                    cast(Any, atlas),
                    pos,
                    max_long_edge=atlas_resolution,
                    show_borders=show_borders,
                )
                ref_img = Image.open(io.BytesIO(ref_bytes))

                # Get the input slice
                slice_img = prepared_images[slice_idx]

                # Build side-by-side composite
                # Normalize heights
                target_h = max(slice_img.height, ref_img.height)
                s_w = int(slice_img.width * target_h / slice_img.height)
                r_w = int(ref_img.width * target_h / ref_img.height)
                gap = 20

                from PIL import ImageDraw, ImageFont
                canvas = Image.new(
                    "RGB",
                    (s_w + gap + r_w, target_h + 80),
                    (0, 0, 0),
                )
                canvas.paste(
                    slice_img.resize((s_w, target_h), Image.Resampling.LANCZOS).convert("RGB"),
                    (0, 0),
                )
                canvas.paste(
                    ref_img.resize((r_w, target_h), Image.Resampling.LANCZOS).convert("RGB"),
                    (s_w + gap, 0),
                )

                draw = ImageDraw.Draw(canvas)
                try:
                    font = ImageFont.truetype("arial.ttf", 50)
                except OSError:
                    font = ImageFont.load_default()

                draw.text(
                    (10, target_h + 10),
                    f"Slice {slice_idx + 1}",
                    fill=(255, 200, 0),
                    font=font,
                )
                draw.text(
                    (s_w + gap + 10, target_h + 10),
                    f"Atlas {pos:.2f}mm",
                    fill=(200, 200, 255),
                    font=font,
                )

                composite_bytes = _image_to_bytes(canvas)

                state.fetched_positions.append(pos)
                state.images_fetched += 1

                _append_response(
                    call_id=call_id,
                    name=name,
                    response={
                        "status": "ok",
                        "description": (
                            f"Side-by-side: Slice {slice_idx + 1} (left) vs "
                            f"Atlas at {pos:.2f}mm (right)."
                        ),
                    },
                )
                _append_image(composite_bytes)

                if run_dir:
                    canvas.save(
                        os.path.join(
                            run_dir,
                            f"tool_{iteration + 1:02d}_compare_s{slice_idx + 1}_{pos:.2f}.jpg",
                        ),
                        quality=85,
                    )

            except (ValueError, IndexError) as exc:
                _append_response(
                    call_id=call_id,
                    name=name,
                    response={
                        "status": "error",
                        "error": f"Failed to build comparison: {exc}",
                    },
                    is_error=True,
                )

            state.reasoning_log.append(
                {
                    "iteration": iteration + 1,
                    "tool": name,
                    "args": args,
                    "result": f"Compared slice {slice_idx + 1} vs atlas {pos:.2f}mm",
                }
            )
            _emit_trace(
                on_trace,
                tool_result_event(
                    stage="ap_group",
                    tool_name=name,
                    summary=f"Side-by-side: Slice {slice_idx + 1} vs {pos:.2f}mm",
                    metadata={
                        "iteration": iteration + 1,
                        "slice_index": slice_idx + 1,
                        "position_mm": round(pos, 3),
                    },
                ),
            )
'''

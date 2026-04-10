"""Image-gen model AP estimation using multi-pass zoom.

Uses image-generation models (``gemini-3.1-flash-image-preview``, etc.) for
AP estimation via a 3-pass zoom workflow:

1. **Broad scan** -- evenly-spaced slices spanning the full AP range.
2. **Neighborhood zoom** -- centered on the pass-1 pick.
3. **Fine zoom** -- ~0.05 mm spacing around the pass-2 pick.

Each pass builds a composite grid image (target on the left, labeled atlas
slices on the right) and sends it via ``generate_content`` as a stateless
call -- no conversation history is accumulated, so each pass gets the full
token budget.  The model picks the best-matching atlas image by number, and
the search window narrows for the next pass.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from PIL import Image

import langslice.vlm_config as vlm_config
from langslice.agent_trace import (
    image_part_from_pil,
    json_part,
    model_event,
    runtime_event,
)
from langslice.estimation.google.ap_tool_use import (
    APResult,
    _emit_trace,
    _extract_usage_metadata,
    _format_usage_metadata,
    _get_position_range_lazy,
    _image_to_bytes,
    _load_atlas_lazy,
)
from langslice.estimation.google.tool_definitions import _build_atlas_grid
from langslice.image_prep import normalize_image, prepare_image_for_vlm
from langslice.retry import (
    format_elapsed_seconds as _format_elapsed_seconds,
)
from langslice.retry import (
    retry_with_backoff,
)
from langslice.vlm_config import get_client

logger = logging.getLogger(__name__)

_SLICES_PER_PASS = 13
_FINE_RESOLUTION_MM = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_positions(
    lo: float,
    hi: float,
    count: int,
    *,
    margin: float = 0.0,
) -> list[float]:
    """Return *count* evenly spaced AP positions between *lo* and *hi*.

    When *margin* > 0 the endpoints are inset by that amount so that
    blank/empty edge slices are avoided.
    """
    lo = lo + margin
    hi = hi - margin
    if lo >= hi:
        return [round((lo + hi) / 2, 3)]
    if count <= 1:
        return [round((lo + hi) / 2, 3)]
    step = (hi - lo) / (count - 1)
    return [round(lo + i * step, 3) for i in range(count)]


def _species_from_atlas_name(atlas_name: str) -> str:
    from langslice.atlas.core import species_from_atlas_name
    return species_from_atlas_name(atlas_name)


def _parse_model_choice(
    text: str,
    positions: list[float],
) -> tuple[int | None, float | None]:
    """Extract the model's AP estimate from its text response.

    Prefers a direct mm value over an image number so the model can
    interpolate between atlas positions.

    Returns ``(nearest 0-based index, position_mm)`` or ``(None, None)``.
    """
    lo = min(positions)
    hi = max(positions)

    # Primary: look for an explicit mm estimate within the atlas
    # range. Try "N mm" first, then bare floats as fallback.
    mm_candidates: list[float] = []
    for match in re.finditer(r"(\d+\.?\d*)\s*mm", text):
        value = float(match.group(1))
        if lo - 1.0 <= value <= hi + 1.0:
            mm_candidates.append(value)

    if not mm_candidates:
        # Model may reply with a bare number (no "mm" suffix).
        for match in re.finditer(r"(\d+\.\d+)", text):
            value = float(match.group(1))
            if lo - 1.0 <= value <= hi + 1.0:
                mm_candidates.append(value)

    if mm_candidates:
        # Take the last plausible mm value (model's final answer).
        chosen_mm = mm_candidates[-1]
        nearest_idx = min(
            range(len(positions)),
            key=lambda i: abs(positions[i] - chosen_mm),
        )
        return nearest_idx, round(chosen_mm, 3)

    # Fallback: image number reference.
    patterns = [
        r"(?:image|atlas\s+(?:image|slice|section))\s*\[?#?(\d+)\]?",
        r"\[(\d+)\]",
        r"#(\d+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            num = int(match.group(1))
            if 1 <= num <= len(positions):
                return num - 1, positions[num - 1]

    return None, None


def _retry_generate_content(
    client: Any,
    *,
    model: str,
    contents: object,
    config: object,
    request_label: str,
    on_progress: Callable[[str], None] | None = None,
) -> Any:
    """Call ``client.models.generate_content`` with exponential backoff."""
    return retry_with_backoff(
        lambda: client.models.generate_content(
            model=model, contents=contents, config=config,
        ),
        request_label=request_label,
        on_progress=on_progress,
    )


def _extract_text_and_thoughts(response: object) -> tuple[list[str], list[str]]:
    """Extract text and thought parts from a ``generate_content`` response."""
    text_outputs: list[str] = []
    thought_outputs: list[str] = []
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if not isinstance(text, str) or not text:
                continue
            if bool(getattr(part, "thought", False)):
                thought_outputs.append(text)
            else:
                text_outputs.append(text)
    return text_outputs, thought_outputs


def _build_gen_config(
    *,
    model_name: str,
    thinking_level: str | None,
) -> Any:
    """Build a typed ``GenerateContentConfig`` for text-only image-gen output."""
    types_mod = importlib.import_module("google.genai.types")

    kwargs: dict[str, Any] = {
        "response_modalities": ["TEXT"],
        "temperature": vlm_config.TEMPERATURE,
    }

    if thinking_level is not None and vlm_config.supports_image_model_thinking(model_name):
        kwargs["thinking_config"] = types_mod.ThinkingConfig(
            thinking_level=thinking_level,
        )

    return types_mod.GenerateContentConfig(**kwargs)


def _image_to_typed_part(img_bytes: bytes, mime_type: str = "image/jpeg") -> Any:
    """Convert raw image bytes to a typed ``Part``."""
    types_mod = importlib.import_module("google.genai.types")
    return types_mod.Part.from_bytes(mime_type=mime_type, data=img_bytes)


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------


def _fetch_atlas_slice_bytes(
    atlas: Any,
    position_mm: float,
    *,
    max_long_edge: int = 512,
    show_borders: bool = False,
) -> bytes:
    """Fetch a single atlas slice, normalize, scale, and return JPEG bytes."""
    if show_borders:
        from langslice.atlas.core import get_composite_slice

        img = get_composite_slice(atlas, position_mm)
    else:
        from langslice.atlas.core import get_reference_slice

        img = get_reference_slice(atlas, position_mm)
    img = normalize_image(img)
    w, h = img.size
    long_edge = max(w, h)
    if long_edge > max_long_edge:
        scale = max_long_edge / long_edge
        img = img.resize(
            (round(w * scale), round(h * scale)),
            Image.Resampling.LANCZOS,
        )
    return _image_to_bytes(img)


def estimate_position_image_gen(
    image: Image.Image,
    atlas_name: str,
    *,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
    debug_dir: str | None = None,
    model_name: str | None = None,
    show_borders: bool = False,
    anatomy_hints: str = "",
    slices_per_pass: int = _SLICES_PER_PASS,
    send_individually: bool = False,
    atlas_resolution: int = 512,
    center_mm: float | None = None,
    bounds: tuple[float, float] | None = None,
) -> APResult:
    """Multi-pass zoom AP estimation using an image-generation model.

    Three passes narrow from the full atlas range down to ~0.05 mm resolution.
    Each pass is a stateless ``generate_content`` call so that the full token
    budget is available for the grid image on every pass.

    When *center_mm* is provided the broad scan (pass 1) is skipped and the
    neighborhood zoom starts directly around that position.

    When *bounds* ``(lo, hi)`` is provided, the search range is hard-clamped
    so no pass ever extends outside the given range.
    """

    def _progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        logger.info(msg)

    effective_model = model_name or vlm_config.MODEL_NAME
    client = get_client()
    atlas = _load_atlas_lazy(atlas_name)
    pos_lo, pos_hi = _get_position_range_lazy(atlas)
    if bounds is not None:
        pos_lo = max(pos_lo, bounds[0])
        pos_hi = min(pos_hi, bounds[1])
    species = _species_from_atlas_name(atlas_name)

    # --- Prepare target image ---------------------------------------------------
    target_normalized = normalize_image(image)
    target_prep = prepare_image_for_vlm(target_normalized)
    target_prepared = target_prep.image
    target_bytes = _image_to_bytes(target_prepared)

    target_info: dict[str, Any] = {
        "original_width": target_prep.original_size[0],
        "original_height": target_prep.original_size[1],
        "width": target_prepared.width,
        "height": target_prepared.height,
        "vlm_scale_factor": round(target_prep.scale_factor, 6),
        "model": effective_model,
        "workflow": "image_gen_ap",
    }
    if target_prep.downsampled:
        orig_w = target_prep.original_size[0]
        orig_h = target_prep.original_size[1]
        new_w = target_prepared.width
        new_h = target_prepared.height
        _progress(
            f"Target resized: {orig_w}x{orig_h} -> "
            f"{new_w}x{new_h}px"
        )

    # --- Debug directory --------------------------------------------------------
    debug_root = debug_dir or os.environ.get("LANGSLICE_VLM_DEBUG_DIR")
    run_dir: str | None = None
    if debug_root:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_atlas = atlas_name.replace("/", "_").replace("\\", "_")
        run_dir = os.path.join(debug_root, f"{ts}_{safe_atlas}_imggen")
        os.makedirs(run_dir, exist_ok=True)
        target_prepared.save(os.path.join(run_dir, "target.jpg"), quality=85)
        _progress(f"Debug artifacts -> {run_dir}")

    _emit_trace(
        on_trace,
        runtime_event(
            stage="ap",
            title="Image-gen AP estimation started",
            summary=f"Model: {effective_model}, {slices_per_pass} slices/pass",
            parts=[
                image_part_from_pil(
                    target_prepared,
                    label="Target slice",
                    image_bytes=target_bytes,
                ),
            ],
            metadata=target_info,
        ),
    )

    # --- Generation config ------------------------------------------------------
    thinking_level = (
        vlm_config.THINKING_LEVEL
        if vlm_config.supports_image_model_thinking(effective_model)
        else None
    )
    gen_config = _build_gen_config(
        model_name=effective_model,
        thinking_level=thinking_level,
    )

    # --- 3-pass zoom loop -------------------------------------------------------
    types_mod = importlib.import_module("google.genai.types")
    content_cls = types_mod.Content

    range_lo, range_hi = pos_lo, pos_hi
    pass_results: list[dict[str, object]] = []
    final_pos = (pos_lo + pos_hi) / 2.0
    final_reasoning = "No estimate obtained."
    pass_labels = ["Broad scan", "Neighborhood zoom", "Fine zoom"]
    anatomy_preamble = f"{anatomy_hints}\n\n" if anatomy_hints else ""

    # Inset the broad scan so the extreme anterior/posterior blanks are skipped.
    # Later passes are already centered on a valid pick so no margin needed.
    _BROAD_MARGIN_MM = 0.5

    # When a center is provided, skip the broad scan and start at the
    # neighborhood zoom (pass index 1) centered on the given position.
    if center_mm is not None:
        if bounds is not None:
            # Explicit bounds: use them directly as the neighborhood range.
            range_lo, range_hi = pos_lo, pos_hi
            start_pass = 1
        else:
            # No explicit bounds: compute a neighborhood window from the
            # full-atlas broad-scan spacing.
            full_lo, full_hi = _get_position_range_lazy(atlas)
            broad_spacing = (full_hi - full_lo - 2 * _BROAD_MARGIN_MM) / max(
                slices_per_pass - 1, 1
            )
            half = broad_spacing * 1.5
            range_lo = max(pos_lo, center_mm - half)
            range_hi = min(pos_hi, center_mm + half)
            start_pass = 1
        final_pos = center_mm
        _progress(
            f"Skipping broad scan — centering on {center_mm:.2f} mm "
            f"({range_lo:.2f}-{range_hi:.2f} mm)"
        )
    else:
        start_pass = 0

    for pass_idx in range(start_pass, 3):
        margin = _BROAD_MARGIN_MM if pass_idx == 0 else 0.0
        positions = _compute_positions(
            range_lo, range_hi, slices_per_pass, margin=margin,
        )
        spacing = (
            round(positions[1] - positions[0], 4) if len(positions) > 1 else 0.0
        )

        _progress(
            f"Pass {pass_idx + 1}/3 ({pass_labels[pass_idx]}): "
            f"{len(positions)} slices, {range_lo:.2f}-{range_hi:.2f} mm "
            f"(spacing {spacing:.3f} mm)"
        )

        _emit_trace(
            on_trace,
            runtime_event(
                stage="ap",
                title=f"Pass {pass_idx + 1}: {pass_labels[pass_idx]}",
                summary=f"{len(positions)} slices, spacing {spacing:.3f} mm",
                metadata={
                    "pass": pass_idx + 1,
                    "positions": positions,
                    "spacing_mm": spacing,
                    "send_individually": send_individually,
                },
            ),
        )

        # Build prompt
        pos_list = ", ".join(
            f"[{i + 1}] {p:.2f} mm" for i, p in enumerate(positions)
        )
        prompt = (
            f"{anatomy_preamble if pass_idx == 0 else ''}"
            f"This is a {species} histological brain section and "
            f"{len(positions)} reference atlas coronal sections (numbered):\n"
            f"{pos_list}\n\n"
            "What is the AP position (in mm) of the histology slice? "
            "It may fall between two atlas images. "
            "Reply with your best estimate as a number in mm."
        )

        if send_individually:
            # Send atlas slices first, then target slice last.
            # Placing the target image closest to the prompt leverages
            # VLM recency bias — the model compares the freshly-seen
            # target against atlas references it has already processed,
            # improving visual matching precision.
            request_parts: list[Any] = [
                types_mod.Part.from_text(
                    text="Reference atlas coronal sections:"
                ),
            ]
            for i, pos in enumerate(positions):
                try:
                    atlas_bytes = _fetch_atlas_slice_bytes(
                        atlas, pos,
                        max_long_edge=atlas_resolution,
                        show_borders=show_borders,
                    )
                    request_parts.append(
                        types_mod.Part.from_text(text=f"[{i + 1}] {pos:.2f} mm:")
                    )
                    request_parts.append(
                        _image_to_typed_part(atlas_bytes),
                    )
                except (ValueError, IndexError):
                    pass
            request_parts.append(
                types_mod.Part.from_text(text="Target histology slice:")
            )
            request_parts.append(_image_to_typed_part(target_bytes))
            request_parts.append(types_mod.Part.from_text(text=prompt))
            contents = [content_cls(role="user", parts=request_parts)]
        else:
            # Composite grid mode
            grid_img = _build_atlas_grid(
                atlas,
                positions,
                target_image=target_prepared,
                show_borders=show_borders,
                max_positions=slices_per_pass,
            )
            grid_bytes = _image_to_bytes(grid_img)
            if run_dir:
                grid_img.save(
                    os.path.join(run_dir, f"pass{pass_idx + 1}_grid.jpg"),
                    quality=85,
                )
            contents = [
                content_cls(
                    role="user",
                    parts=[
                        _image_to_typed_part(grid_bytes),
                        types_mod.Part.from_text(text=prompt),
                    ],
                ),
            ]

        started_at = time.perf_counter()
        response = _retry_generate_content(
            client,
            model=effective_model,
            contents=contents,
            config=gen_config,
            request_label=f"Image-gen AP pass {pass_idx + 1}",
            on_progress=_progress,
        )
        elapsed = round(time.perf_counter() - started_at, 3)

        usage = _extract_usage_metadata(response)
        _progress(
            f"Pass {pass_idx + 1}: {_format_elapsed_seconds(elapsed)}; "
            f"{_format_usage_metadata(usage)}"
        )

        # Extract text
        text_outputs, thought_outputs = _extract_text_and_thoughts(response)
        response_text = "\n".join(text_outputs) if text_outputs else ""

        if thought_outputs or text_outputs:
            trace_parts: list[dict[str, object]] = []
            if thought_outputs:
                trace_parts.append(
                    json_part(thought_outputs, label="Thinking", collapsible=True)
                )
            if text_outputs:
                trace_parts.append(json_part(text_outputs, label="Response"))
            _emit_trace(
                on_trace,
                model_event(
                    stage="ap",
                    title=f"Pass {pass_idx + 1} response",
                    summary=response_text[:200] if response_text else "(no text)",
                    parts=trace_parts,
                    metadata={"pass": pass_idx + 1, **usage},
                ),
            )

        _progress(f"Pass {pass_idx + 1} model output: {response_text[:300]}")

        # Parse choice
        chosen_idx, chosen_pos = _parse_model_choice(response_text, positions)

        pass_record: dict[str, object] = {
            "pass": pass_idx + 1,
            "label": pass_labels[pass_idx],
            "range": [range_lo, range_hi],
            "positions": positions,
            "spacing_mm": spacing,
            "response_text": response_text,
            "usage": dict(usage),
            "wall_time_s": elapsed,
        }

        if chosen_pos is not None:
            assert chosen_idx is not None
            final_pos = chosen_pos
            final_reasoning = response_text
            _progress(
                f"Pass {pass_idx + 1}: model chose "
                f"[{chosen_idx + 1}] -> {chosen_pos:.2f} mm"
            )
            pass_record["chosen_index"] = chosen_idx
            pass_record["chosen_position_mm"] = chosen_pos

            # Narrow range for next pass
            if pass_idx == 0:
                # Neighborhood: ±1.5x the pass-1 spacing around the pick
                half = spacing * 1.5
                range_lo = max(pos_lo, chosen_pos - half)
                range_hi = min(pos_hi, chosen_pos + half)
            elif pass_idx == 1:
                # Fine: ~0.05 mm spacing
                fine_span = _FINE_RESOLUTION_MM * (slices_per_pass - 1)
                range_lo = max(pos_lo, chosen_pos - fine_span / 2)
                range_hi = min(pos_hi, chosen_pos + fine_span / 2)
        else:
            _progress(
                f"Pass {pass_idx + 1}: could not parse model choice from response"
            )
            pass_record["error"] = "Failed to parse model choice"
            pass_results.append(pass_record)
            break

        pass_results.append(pass_record)

    # --- Finalize ---------------------------------------------------------------
    _progress(f"Image-gen AP estimate: {final_pos:.2f} mm")
    _emit_trace(
        on_trace,
        runtime_event(
            stage="ap",
            title="Image-gen AP estimation completed",
            summary=f"Final position {final_pos:.2f} mm",
            parts=[
                json_part(
                    {"position_mm": final_pos, "passes": pass_results},
                    label="AP result",
                ),
            ],
            metadata={"position_mm": final_pos, "num_passes": len(pass_results)},
        ),
    )

    if run_dir:
        with open(os.path.join(run_dir, "passes.json"), "w") as f:
            json.dump(pass_results, f, indent=2, default=str)

    return APResult(
        position_mm=final_pos,
        reasoning=final_reasoning,
        debug_dir=run_dir,
    )



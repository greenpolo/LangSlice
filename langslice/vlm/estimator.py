"""VLM-based brain slice estimation using Gemini."""

import base64
import io
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, cast

import numpy as np
from PIL import Image
from google.genai.types import Tool, ToolCodeExecution

from langslice.vlm.config import (
    CODE_EXECUTION_ENABLED,
    MODEL_NAME,
    THINKING_LEVEL,
    get_client,
)

logger = logging.getLogger(__name__)

# Retryable exception types (google-genai SDK error hierarchy)
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_INITIAL_BACKOFF_S = 1.0
_RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def _retry_generate(
    client: Any,
    *,
    model: str,
    contents: object,
    config: object,
) -> Any:
    """Wrapper around client.models.generate_content with exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except Exception as exc:
            last_exc = exc
            # Check if the error has a retryable HTTP status code
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            if isinstance(status, int) and status in _RETRYABLE_STATUS_CODES:
                if attempt < _MAX_RETRIES:
                    delay = _INITIAL_BACKOFF_S * (2 ** attempt)
                    logger.warning(
                        "Gemini API error (status %s), retrying in %.1fs (attempt %d/%d)",
                        status, delay, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(delay)
                    continue
            # Also retry on generic connection / timeout errors
            exc_name = type(exc).__name__.lower()
            if any(kw in exc_name for kw in ("timeout", "connection", "transport")):
                if attempt < _MAX_RETRIES:
                    delay = _INITIAL_BACKOFF_S * (2 ** attempt)
                    logger.warning(
                        "Transient error (%s), retrying in %.1fs (attempt %d/%d)",
                        type(exc).__name__, delay, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(delay)
                    continue
            # Non-retryable error — raise immediately
            raise
    # Exhausted retries
    assert last_exc is not None
    raise last_exc

@dataclass
class APResult:
    position_mm: float
    reasoning: str


@dataclass
class AffineResult:
    rotation: float
    translateX: float
    translateY: float
    reasoning: str



def _image_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    """Convert PIL Image to base64 string."""
    buf = io.BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")
    if fmt.upper() == "JPEG":
        img.save(buf, format=fmt, quality=95, subsampling=0)
    else:
        img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")



def _normalize_image(image: Image.Image) -> Image.Image:
    """Normalize an arbitrary PIL image to 8-bit RGB without modifying source data.

    Handles:
    - 16-bit / 32-bit grayscale (``I;16``, ``I``, ``F``): min-max scaled to 0-255
    - RGBA / palette / other modes: converted to RGB directly
    - Standard 8-bit RGB: returned as-is

    The source ``Image`` object is never mutated.
    """
    mode = image.mode

    # Already 8-bit RGB — nothing to do.
    if mode == "RGB":
        return image

    # High bit-depth modes need min-max normalization, not clipping.
    if mode in ("I", "I;16", "I;16B", "I;32", "F"):
        arr = np.asarray(image, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        else:
            arr = np.zeros_like(arr)
        gray8 = arr.astype(np.uint8)
        return Image.fromarray(gray8).convert("RGB")

    # All other modes (L, LA, RGBA, P, PA, CMYK, …) — let PIL handle it.
    return image.convert("RGB")



def _part_text(text: str) -> dict[str, object]:
    return {"text": text}


def _part_image_base64(image_b64: str) -> dict[str, object]:
    inline_data: object = {
        "mime_type": "image/jpeg",
        "data": image_b64,
    }
    return {"inline_data": inline_data}


def _contents_from_parts(parts: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{"role": "user", "parts": parts}]


def _generation_config(schema: dict[str, object]) -> dict[str, object]:
    config: dict[str, object] = {
        "thinking_config": {"thinking_level": THINKING_LEVEL},
        "response_mime_type": "application/json",
        "response_json_schema": schema,
    }
    if CODE_EXECUTION_ENABLED:
        config["tools"] = [Tool(code_execution=ToolCodeExecution())]
    return config


def _first_model_content(response: object) -> object:
    candidates = getattr(response, "candidates", None)
    if not isinstance(candidates, list) or not candidates:
        prompt_feedback = getattr(response, "prompt_feedback", None)
        raise RuntimeError(
            f"Gemini returned no candidates. prompt_feedback={prompt_feedback}"
        )

    first = candidates[0]
    finish_reason = getattr(first, "finish_reason", None)
    content = getattr(first, "content", None)
    if content is None:
        raise RuntimeError(
            f"Gemini candidate has no content. finish_reason={finish_reason}"
        )
    return content


def _extract_result(response: object) -> dict[str, object]:
    executable_code = getattr(response, "executable_code", None)
    if executable_code:
        logger.info(f"VLM executed code:\n{executable_code}")
    
    code_execution_result = getattr(response, "code_execution_result", None)
    if code_execution_result:
        logger.info(f"VLM code execution outcome:\n{code_execution_result}")
        
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, dict):
        parsed_dict = cast(dict[object, object], parsed)
        normalized: dict[str, object] = {}
        for k, v in parsed_dict.items():
            normalized[str(k)] = v
        return normalized

    model_dump = getattr(parsed, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            dumped_dict = cast(dict[object, object], dumped)
            normalized: dict[str, object] = {}
            for k, v in dumped_dict.items():
                normalized[str(k)] = v
            return normalized

    text = getattr(response, "text", None)
    if isinstance(text, str) and text:
        try:
            decoded = json.loads(text)
            if isinstance(decoded, dict):
                decoded_dict = cast(dict[object, object], decoded)
                normalized: dict[str, object] = {}
                for k, v in decoded_dict.items():
                    normalized[str(k)] = v
                return normalized
        except json.JSONDecodeError:
            pass
        logger.warning("Gemini response did not expose parsed JSON; returning empty result.")
    return {}


def _to_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _to_str(value: object, default: str = "N/A") -> str:
    if isinstance(value, str):
        return value
    return default


def _image_to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    """Convert PIL Image to raw bytes."""
    buf = io.BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")
    if fmt.upper() == "JPEG":
        img.save(buf, format=fmt, quality=95, subsampling=0)
    else:
        img.save(buf, format=fmt)
    return buf.getvalue()


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


def estimate_position(
    image: Image.Image,
    atlas_name: str,
    on_progress: Callable[[str], None] | None = None,
) -> APResult:
    """Agentic AP estimation using tool-use with self-correction.

    The model receives tools to explore the atlas freely:
    - fetch_atlas_slice: view any coronal section
    - get_atlas_info: get coordinate range and metadata
    - get_region_names: see what brain regions exist at a position
    - submit_estimate: declare the final answer

    Uses manual function calling so images can be injected alongside
    tool responses. The model runs until it submits or hits max iterations.

    Set ``LANGSLICE_VLM_DEBUG_DIR`` to save all artifacts for review.
    """
    import os
    from datetime import datetime
    from google.genai import types
    from langslice.atlas.core import (
        get_position_range_mm,
        get_reference_slice,
        load_atlas,
        get_atlas_info as _get_atlas_info_core,
    )

    def _progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        logger.info(msg)

    client = get_client()
    atlas = load_atlas(atlas_name)
    pos_lo, pos_hi = get_position_range_mm(atlas)

    target_prepared = _normalize_image(image)
    target_bytes = _image_to_bytes(target_prepared)
    target_h = target_prepared.height

    # --- Debug artifact setup ---
    debug_dir: str | None = os.environ.get("LANGSLICE_VLM_DEBUG_DIR")
    run_dir: str | None = None
    if debug_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_atlas = atlas_name.replace("/", "_").replace("\\", "_")
        run_dir = os.path.join(debug_dir, f"{timestamp}_{safe_atlas}")
        os.makedirs(run_dir, exist_ok=True)
        target_prepared.save(os.path.join(run_dir, "target.jpg"), quality=95)
        _progress(f"Debug artifacts → {run_dir}")

    # --- Tool declarations ---
    tool_declarations = types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="fetch_atlas_slice",
            description=(
                "Fetch a coronal brain atlas reference image at a specific "
                "anterior-posterior position. The image will be shown to you. "
                "Use this to visually compare against the target slice."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "position_mm": {
                        "type": "number",
                        "description": "AP position in mm from the anterior edge of the atlas",
                    },
                },
                "required": ["position_mm"],
            },
        ),
        types.FunctionDeclaration(
            name="get_atlas_info",
            description=(
                "Get atlas metadata including the valid AP coordinate range, "
                "resolution, species, and number of slices."
            ),
            parameters_json_schema={"type": "object", "properties": {}},
        ),
        types.FunctionDeclaration(
            name="get_region_names",
            description=(
                "Get the names and acronyms of brain regions visible at a "
                "specific AP position. Useful for confirming anatomical identity."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "position_mm": {
                        "type": "number",
                        "description": "AP position in mm from the anterior edge",
                    },
                },
                "required": ["position_mm"],
            },
        ),
        types.FunctionDeclaration(
            name="fetch_multiple_atlas_slices",
            description=(
                "Fetch up to 5 coronal brain atlas reference images at multiple "
                "anterior-posterior positions at once. The images will be shown to you "
                "in order. Use this to perform a rapid coarse sweep (e.g., check every 2mm) "
                "to quickly narrow down the general neighborhood."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "positions_mm": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "List of up to 5 AP positions in mm to fetch",
                    },
                },
                "required": ["positions_mm"],
            },
        ),
        types.FunctionDeclaration(
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
                    "confidence": {
                        "type": "string",
                        "description": "Confidence level: low, medium, or high",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Detailed reasoning for the estimate",
                    },
                },
                "required": ["position_mm", "confidence", "reasoning"],
            },
        ),
    ])

    # --- System prompt ---
    system_instruction = (
        "You are an expert neuroanatomist. You are given a histology brain slice image "
        "and must determine its Anterior-Posterior (AP) position within a reference atlas. "
        "The coordinate system is: 0.0mm is the extreme Anterior edge (e.g. olfactory bulb), "
        "while larger mm values move Posterior towards the cerebellum and brainstem. "
        "You have tools to fetch atlas reference images at any AP coordinate, query which "
        "brain regions exist at a given position, and get atlas metadata. \n\n"
        "RECOMMENDED STRATEGY:\n"
        "1. Coarse Sweep: Call `fetch_multiple_atlas_slices` with 4-5 widely spaced coordinates "
        "   (e.g., 2.0, 4.0, 6.0, 8.0) to instantly find the correct neighborhood.\n"
        "2. Finer Search: Identify the closest match, then call `fetch_multiple_atlas_slices` "
        "   again around that match with tighter spacing (e.g., ±0.5mm).\n"
        "3. Verification: Once narrowed down, check specific structural landmarks, or use "
        "   `get_region_names` to confirm anatomical identity.\n"
        "4. Submit: Call `submit_estimate` ONLY when you are highly confident.\n\n"
        "Don't guess blindly — use your tools to narrow down the answer methodically."
    )

    # --- Initial user message with target image ---
    initial_parts = [
        types.Part(text="Here is the target brain slice. Determine its AP position in the atlas."),
        types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=target_bytes)),
    ]
    history: list[types.Content] = [
        types.Content(role="user", parts=initial_parts),
    ]

    thinking_level = getattr(types.ThinkingLevel, THINKING_LEVEL, None)
    config = types.GenerateContentConfig(
        tools=[tool_declarations],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
        system_instruction=system_instruction,
    )

    max_iterations = 20
    estimate_result: dict[str, object] | None = None
    reasoning_log: list[dict[str, object]] = []
    images_fetched = 0

    _progress(f"Starting agentic estimation (max {max_iterations} tool calls)...")

    for iteration in range(max_iterations):
        response = _retry_generate(
            client,
            model=MODEL_NAME,
            contents=history,
            config=config,
        )

        # Append model response to history
        model_content: types.Content = cast(types.Content, _first_model_content(response))
        history.append(model_content)

        # Collect function calls from the response
        model_parts = getattr(model_content, "parts", None) or []
        function_calls = [p for p in model_parts if p.function_call]

        if not function_calls:
            # Model responded with thought/text but no tool call
            text_parts = [p.text for p in model_parts if p.text]
            if text_parts:
                _progress(f"Agent reasoning/text: {text_parts[0][:200]}...")
            else:
                _progress("Agent produced thought block but no tool calls.")
            
            # Prevent premature exit; prompt the model to actually call a tool
            nudge = types.Part(text="Please continue. You must call a tool to explore further, or call `submit_estimate` if you have finalized your answer.")
            history.append(types.Content(role="user", parts=[nudge]))
            continue


        # Process each function call
        tool_response_parts: list[types.Part] = []

        for fc_part in function_calls:
            fc = fc_part.function_call
            name = fc.name
            args = dict(fc.args) if fc.args else {}

            _progress(f"Tool call [{iteration + 1}]: {name}({args})")

            if name == "fetch_atlas_slice":
                pos = float(args.get("position_mm", (pos_lo + pos_hi) / 2))
                pos = max(pos_lo, min(pos_hi, pos))  # Clamp

                try:
                    ref_img = get_reference_slice(atlas, pos)
                    ref_prepared = _normalize_image(ref_img)
                    # Scale to match target height
                    scale = target_h / ref_prepared.height
                    new_w = max(1, int(round(ref_prepared.width * scale)))
                    new_h = max(1, int(round(ref_prepared.height * scale)))
                    ref_scaled = ref_prepared.resize((new_w, new_h), _RESAMPLE_LANCZOS)
                    ref_bytes = _image_to_bytes(ref_scaled)
                    images_fetched += 1

                    # Save debug image
                    if run_dir:
                        ref_scaled.save(
                            os.path.join(run_dir, f"tool_{iteration + 1:02d}_slice_{pos:.2f}mm.jpg"),
                            quality=95,
                        )

                    tool_response_parts.append(
                        types.Part.from_function_response(
                            name=name,
                            response={"position_mm": pos, "status": "ok", "description": f"Atlas coronal section at {pos:.2f}mm from anterior edge"},
                        )
                    )
                    # Inject the image so the model can see it
                    tool_response_parts.append(
                        types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=ref_bytes))
                    )

                    reasoning_log.append({"iteration": iteration + 1, "tool": name, "args": args, "result": f"Image at {pos:.2f}mm"})

                except ValueError as exc:
                    tool_response_parts.append(
                        types.Part.from_function_response(
                            name=name,
                            response={"status": "error", "error": str(exc)},
                        )
                    )
                    reasoning_log.append({"iteration": iteration + 1, "tool": name, "args": args, "result": f"Error: {exc}"})

            elif name == "fetch_multiple_atlas_slices":
                positions_list = args.get("positions_mm", [])
                if not isinstance(positions_list, list):
                    positions_list = []
                
                # Cap to 5 to prevent context blowout
                positions = [max(pos_lo, min(pos_hi, float(p))) for p in positions_list[:5]]
                
                if not positions:
                    tool_response_parts.append(
                        types.Part.from_function_response(
                            name=name,
                            response={"status": "error", "error": "No valid positions provided"},
                        )
                    )
                    reasoning_log.append({"iteration": iteration + 1, "tool": name, "args": args, "result": "Error: empty input"})
                else:
                    successes = []
                    for pos in positions:
                        try:
                            ref_img = get_reference_slice(atlas, pos)
                            ref_prepared = _normalize_image(ref_img)
                            scale = target_h / ref_prepared.height
                            new_w = max(1, int(round(ref_prepared.width * scale)))
                            new_h = max(1, int(round(ref_prepared.height * scale)))
                            ref_scaled = ref_prepared.resize((new_w, new_h), _RESAMPLE_LANCZOS)
                            ref_bytes = _image_to_bytes(ref_scaled)
                            images_fetched += 1

                            if run_dir:
                                ref_scaled.save(
                                    os.path.join(run_dir, f"tool_{iteration + 1:02d}_multi_{pos:.2f}mm.jpg"),
                                    quality=95,
                                )

                            # We must weave text parts + image blobs for the model to associate them
                            tool_response_parts.append(
                                types.Part.from_function_response(
                                    name=name,
                                    response={"position_mm": pos, "status": "ok", "description": f"Atlas coronal section at {pos:.2f}mm"},
                                )
                            )
                            tool_response_parts.append(
                                types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=ref_bytes))
                            )
                            successes.append(f"{pos:.2f}mm")
                        except Exception as exc:
                            # Log error inline and continue
                            tool_response_parts.append(
                                types.Part.from_function_response(
                                    name=name,
                                    response={"position_mm": pos, "status": "error", "error": str(exc)},
                                )
                            )
                    
                    reasoning_log.append({"iteration": iteration + 1, "tool": name, "args": args, "result": f"Fetched {len(successes)} slices: {', '.join(successes)}"})

            elif name == "get_atlas_info":
                info = _get_atlas_info_core(atlas)
                info["coordinate_note"] = "0.0mm is extreme Anterior; higher mm is more Posterior."
                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=name,
                        response=info,
                    )
                )
                reasoning_log.append({"iteration": iteration + 1, "tool": name, "args": {}, "result": str(info)})
                _progress(f"  → Atlas range: [{pos_lo:.2f}, {pos_hi:.2f}] mm")

            elif name == "get_region_names":
                pos = float(args.get("position_mm", (pos_lo + pos_hi) / 2))
                regions = _get_regions_at_position(atlas, pos)
                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=name,
                        response={"position_mm": pos, "regions": regions},
                    )
                )
                reasoning_log.append({"iteration": iteration + 1, "tool": name, "args": args, "result": f"{len(regions)} regions"})
                _progress(f"  → {len(regions)} regions at {pos:.2f}mm")

            elif name == "submit_estimate":
                est_pos = float(args.get("position_mm", 0.0))
                est_confidence = str(args.get("confidence", "unknown"))
                est_reasoning = str(args.get("reasoning", ""))
                estimate_result = {
                    "position_mm": est_pos,
                    "confidence": est_confidence,
                    "reasoning": est_reasoning,
                }
                reasoning_log.append({"iteration": iteration + 1, "tool": name, "args": args, "result": f"Submitted {est_pos:.2f}mm ({est_confidence})"})
                _progress(f"Agent submitted estimate: {est_pos:.2f}mm (confidence: {est_confidence})")
                break

            else:
                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=name,
                        response={"status": "error", "error": f"Unknown tool: {name}"},
                    )
                )

        if estimate_result:
            break

        # Append tool responses to history
        if tool_response_parts:
            history.append(types.Content(role="tool", parts=tool_response_parts))

    # --- Build final result ---
    final_pos: float
    final_reasoning: str

    if estimate_result:
        final_pos = _to_float(estimate_result.get("position_mm"), (pos_lo + pos_hi) / 2)
        final_reasoning = str(estimate_result["reasoning"])
    else:
        # Fallback: model never submitted
        final_pos = (pos_lo + pos_hi) / 2
        final_reasoning = "Agent did not submit an estimate within the iteration limit."
        _progress(f"Warning: Agent did not submit. Falling back to midpoint: {final_pos:.2f}mm")

    _progress(f"Final position estimated: {final_pos:.2f} mm ({images_fetched} atlas images fetched)")

    # --- Write reasoning.txt debug artifact ---
    if run_dir:
        reasoning_path = os.path.join(run_dir, "reasoning.txt")
        with open(reasoning_path, "w", encoding="utf-8") as f:
            f.write(f"AP Estimation — {atlas_name} — {datetime.now().isoformat()}\n")
            f.write(f"Model: {MODEL_NAME}\n")
            f.write("=" * 60 + "\n\n")
            for entry in reasoning_log:
                f.write(f"[{entry['iteration']}] {entry['tool']}({entry.get('args', {})})\n")
                f.write(f"    → {entry['result']}\n\n")
            f.write("=" * 60 + "\n")
            f.write(f"FINAL ESTIMATE: {final_pos:.2f} mm\n")
            if estimate_result:
                f.write(f"CONFIDENCE: {estimate_result.get('confidence', 'N/A')}\n")
                f.write(f"REASONING: {estimate_result.get('reasoning', 'N/A')}\n")
            f.write("=" * 60 + "\n")

            # Write full conversation history
            f.write("\n\nFULL CONVERSATION HISTORY\n")
            f.write("=" * 60 + "\n\n")
            for content in history:
                role = content.role if hasattr(content, "role") else "?"
                f.write(f"--- {role} ---\n")
                content_parts = content.parts or []
                for part in content_parts:
                    if part.text:
                        f.write(f"  TEXT: {part.text[:500]}\n")
                    if part.function_call:
                        f.write(f"  CALL: {part.function_call.name}({dict(part.function_call.args) if part.function_call.args else {}})\n")
                    if part.function_response:
                        f.write(f"  RESPONSE: {part.function_response.name} → {part.function_response.response}\n")
                    if part.inline_data:
                        blob_data = part.inline_data.data
                        data_len = len(blob_data) if isinstance(blob_data, (bytes, bytearray)) else 0
                        f.write(f"  IMAGE: {part.inline_data.mime_type} ({data_len} bytes)\n")
                f.write("\n")

        _progress(f"Reasoning log saved → {reasoning_path}")

    return APResult(
        position_mm=final_pos,
        reasoning=final_reasoning,
    )


def estimate_ap(
    image: Image.Image,
    atlas_name: str,
    on_progress: Callable[[str], None] | None = None,
) -> APResult:
    return estimate_position(
        image=image,
        atlas_name=atlas_name,
        on_progress=on_progress,
    )


def _build_matched_composite(
    target: Image.Image,
    atlas_name: str,
    position_mm: float,
    progress: Callable[[str], None],
) -> Image.Image | None:
    """Build a side-by-side composite with the atlas scaled to match the target's pixel dimensions.

    The atlas is resized so its height matches the target's height, preserving
    the atlas aspect ratio.  This gives the VLM visually comparable images
    regardless of the original pixel sizes or physical resolution.
    """
    from langslice.atlas.core import get_composite_slice, load_atlas

    try:
        atlas = load_atlas(atlas_name)
        atlas_composite = get_composite_slice(atlas, position_mm, opacity=0.4)
    except Exception as exc:
        progress(f"Warning: could not load atlas composite for affine prompt: {exc}")
        return None

    # Scale atlas height to match target height, preserving aspect ratio.
    target_rgb = _normalize_image(target)
    target_h = target_rgb.height
    scale_factor = target_h / atlas_composite.height
    new_w = max(1, int(round(atlas_composite.width * scale_factor)))
    new_h = max(1, int(round(atlas_composite.height * scale_factor)))
    atlas_scaled = atlas_composite.resize((new_w, new_h), _RESAMPLE_LANCZOS)

    progress(
        f"Atlas scaled for VLM: {atlas_composite.width}x{atlas_composite.height} -> "
        f"{new_w}x{new_h} (matched to target height {target_h}px)"
    )

    # Build side-by-side composite: [target | atlas_scaled]
    atlas_rgb = atlas_scaled.convert("RGB")
    out_h = max(target_rgb.height, atlas_rgb.height)
    gap = 20  # pixel gap between images
    out_w = target_rgb.width + gap + atlas_rgb.width
    composite = Image.new("RGB", (out_w, out_h), (0, 0, 0))

    target_y = (out_h - target_rgb.height) // 2
    atlas_y = (out_h - atlas_rgb.height) // 2
    composite.paste(target_rgb, (0, target_y))
    composite.paste(atlas_rgb, (target_rgb.width + gap, atlas_y))

    return composite


def estimate_affine(
    image: Image.Image,
    on_progress: Callable[[str], None] | None = None,
    atlas_name: str | None = None,
    position_mm: float | None = None,
    pixel_size_um: float | None = None,
) -> AffineResult:
    """
    Estimate 2D affine transformation to center and align a brain slice.

    When *atlas_name* and *position_mm* are provided the function constructs
    a side-by-side composite (slice + atlas at matched pixel dimensions) and
    sends it to the VLM, giving the model accurate visual context.

    *pixel_size_um* is accepted but ignored — atlas scaling is now purely
    visual (pixel-dimension matched), not physical.

    Returns rotation (degrees), translateX and translateY (% of image size).
    """

    def _progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        logger.info(msg)

    client = get_client()

    target_prepared = _normalize_image(image)
    target_b64 = _image_to_base64(target_prepared)

    # Build pixel-matched composite when atlas context is available.
    composite_b64: str | None = None
    has_atlas_context = atlas_name is not None and position_mm is not None
    if has_atlas_context:
        assert atlas_name is not None and position_mm is not None
        composite = _build_matched_composite(
            target_prepared, atlas_name, position_mm, _progress
        )
        if composite is not None:
            composite_b64 = _image_to_base64(composite)

    # Assemble prompt parts
    parts: list[dict[str, object]]
    if composite_b64 is not None:
        _progress("Estimating affine transformation (with atlas context)...")
        prompt_text = (
            "Using both images, estimate the 2D affine transformation "
            "(rotation in degrees, translateX and translateY as percentages of image size) "
            "needed to align the target brain slice to match the atlas orientation. "
            "The atlas reference shows the expected upright coronal orientation. "
            "Compare the tissue outline and internal structures to determine rotation and offset. "
        )
        if CODE_EXECUTION_ENABLED:
            prompt_text += (
                "You may actively inspect the images using Python code. For example, "
                "you can write code to measure the angle of the midline or detect the center "
                "of mass to verify your rotation and translation estimates."
            )
        parts = [
            _part_text("Image 1 is the target brain slice you need to register."),
            _part_image_base64(target_b64),
            _part_text(
                "Image 2 is a side-by-side composite showing the target brain slice (left) "
                "and the atlas reference at the same visual scale (right). "
                "The atlas has been resized to match the target image height for comparison."
            ),
            _part_image_base64(composite_b64),
            _part_text(prompt_text),
        ]
    else:
        _progress("Estimating affine transformation...")
        prompt_text = (
            "Analyze this brain slice image. Estimate the 2D affine transformation "
            "(rotation in degrees, translateX and translateY as percentages of image size) "
            "needed to center and align it upright. Assume the image might be slightly tilted "
            "or off-center. "
        )
        if CODE_EXECUTION_ENABLED:
            prompt_text += (
                "You may use Python code to analyze the image properties, such as finding the "
                "center of mass or detecting the main axis of the tissue, to inform your estimates."
            )
        parts = [
            _part_image_base64(target_b64),
            _part_text(prompt_text),
        ]

    response = _retry_generate(
        client,
        model=MODEL_NAME,
        contents=_contents_from_parts(parts),
        config=_generation_config(
            {
                "type": "object",
                "properties": {
                    "rotation": {
                        "type": "number",
                        "description": "Rotation in degrees to make the slice upright",
                    },
                    "translateX": {
                        "type": "number",
                        "description": "X translation as percentage (-50 to 50)",
                    },
                    "translateY": {
                        "type": "number",
                        "description": "Y translation as percentage (-50 to 50)",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for the transformation",
                    },
                },
                "required": ["rotation", "translateX", "translateY", "reasoning"],
            }
        ),
    )

    result = _extract_result(response)
    rotation = _to_float(result.get("rotation"), 0.0)
    translate_x = _to_float(result.get("translateX"), 0.0)
    translate_y = _to_float(result.get("translateY"), 0.0)
    reasoning = _to_str(result.get("reasoning"), "N/A")

    _progress(f"Affine estimated: rot={rotation} deg, tx={translate_x}%, ty={translate_y}%")

    return AffineResult(
        rotation=rotation,
        translateX=translate_x,
        translateY=translate_y,
        reasoning=reasoning,
    )

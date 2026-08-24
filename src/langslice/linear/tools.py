"""Position-estimation tools, wired as plain Python functions for ADK auto-wrapping.

On success, `fetch_atlas` returns a normal dictionary function response whose
``TOOL_MEDIA_PARTS_KEY`` entry holds a flat list of image ``types.Part``s. ADK
2.7+ moves those into the function-response Event's media parts (persisted in
session history, so earlier atlas sweeps stay visible on later turns) and drops
the key from the JSON the model reads. Everything the model needs in words --
positions, artifact keys, image order -- therefore lives in ordinary JSON
fields, not in text Parts, which ADK would leave behind as JSON noise.

Error paths return plain dicts (``{"status": "error", "error": ...}``) with no
media, so they reach the model as ordinary function responses.

Fetched atlas images are also saved as artifacts via
``tool_context.save_artifact`` for debugging and request replay.
"""

from __future__ import annotations

import io
from typing import Any

from google.genai import types
from PIL import Image

from langslice.adk import TOOL_MEDIA_PARTS_KEY
from langslice.atlas.core import (
    get_reference_slice,
    load_atlas,
)
from langslice.linear.session import (
    ARTIFACT_ATLAS_PREFIX,
    atlas_key,
)

# ---- Pure helpers -------------------------------------------------------


def _parse_atlas_key(source: str) -> float:
    if not source.startswith(ARTIFACT_ATLAS_PREFIX):
        raise ValueError(f"Not an atlas source: {source!r}")
    tail = source[len(ARTIFACT_ATLAS_PREFIX):]
    try:
        return float(tail)
    except ValueError as exc:
        raise ValueError(f"Bad atlas position in {source!r}") from exc


def _is_broad_sweep(positions: list[float]) -> bool:
    return len(positions) >= 3


def _is_narrow_sweep(positions: list[float]) -> bool:
    if len(positions) < 3:
        return False
    return (max(positions) - min(positions)) <= 1.0


def _clamp_and_dedupe_positions(
    positions: list[float], *, pos_lo: float, pos_hi: float, dedupe_tol: float = 0.02
) -> list[float]:
    # Defensive flatten: Gemini occasionally emits positions_mm as [[1.5, 2.5]]
    # (nested) instead of [1.5, 2.5]. Walk one level of nesting and coerce.
    flat: list[float] = []
    for p in positions:
        if isinstance(p, (list, tuple)):
            for q in p:
                try:
                    flat.append(float(q))
                except (TypeError, ValueError):
                    continue
        else:
            try:
                flat.append(float(p))
            except (TypeError, ValueError):
                continue
    clamped = [max(pos_lo, min(pos_hi, p)) for p in flat]
    out: list[float] = []
    for p in clamped:
        if any(abs(p - q) <= dedupe_tol for q in out):
            continue
        out.append(p)
    return out


def _image_to_jpeg_bytes(img: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _image_to_part(img: Image.Image) -> types.Part:
    return types.Part.from_bytes(
        mime_type="image/jpeg", data=_image_to_jpeg_bytes(img)
    )


async def fetch_atlas(
    positions_mm: list[float], tool_context: Any
) -> dict[str, Any]:
    """Fetch 1-8 atlas sections along the session's slicing plane.

    Positions outside the valid range are clamped. Duplicate positions within
    0.02 mm of an already-requested one are coalesced. Each returned slice is
    saved as an artifact keyed 'atlas:<mm:.2f>'.

    Returns:
        On success, status/positions/atlas keys plus the atlas images, attached
        in the same order as ``positions_mm``. The images stay visible to the
        model on every later turn.
        On failure, ``{"status": "error", "error": "BAD_ARGS" | "EMPTY_RESULT"}``.
    """
    state = tool_context.state
    if not positions_mm:
        return {"status": "error", "error": "BAD_ARGS"}

    pos_lo = float(state["pos_lo"])
    pos_hi = float(state["pos_hi"])
    plane = state["plane"]
    atlas_name = state["atlas"]

    capped = list(positions_mm)[:8]
    positions = _clamp_and_dedupe_positions(capped, pos_lo=pos_lo, pos_hi=pos_hi)
    if not positions:
        return {"status": "error", "error": "EMPTY_RESULT"}

    atlas = load_atlas(atlas_name)
    image_parts: list[types.Part] = []
    descriptions: list[str] = []
    for pos in positions:
        img = get_reference_slice(atlas, pos, plane=plane)
        part = _image_to_part(img)
        image_parts.append(part)
        await tool_context.save_artifact(filename=atlas_key(pos), artifact=part)
        descriptions.append(f"{pos:.2f} mm")

    state.setdefault("fetched_positions", []).extend(positions)
    state["images_fetched"] = int(state.get("images_fetched", 0)) + len(positions)
    if _is_broad_sweep(positions):
        state["saw_broad_sweep"] = True
    if _is_narrow_sweep(positions):
        state["saw_narrow_sweep"] = True

    plural = "s" if len(positions) != 1 else ""
    return {
        "status": "ok",
        "positions_mm": [round(float(pos), 2) for pos in positions],
        "atlas_keys": [atlas_key(pos) for pos in positions],
        # The attached images are unlabelled, so the ordering note is the
        # model's only way to tie an image to its position.
        "description": (
            f"Fetched {len(positions)} atlas section{plural}: "
            + ", ".join(descriptions)
            + ". The attached atlas images appear in that same order."
        ),
        TOOL_MEDIA_PARTS_KEY: image_parts,
    }


def submit_estimate(
    position_mm: float, reasoning: str = "", tool_context: Any = None
) -> dict[str, Any]:
    """Submit the final position estimate for the target slice.

    Only call this when you have completed broad + narrow atlas sweeps and
    verified at least one neighbor on each side of your candidate position.
    """
    tool_context.state["result"] = {"position_mm": float(position_mm), "reasoning": str(reasoning)}
    tool_context.actions.escalate = True
    return {"status": "ok", "position_mm": float(position_mm)}

"""I/O helpers for the BBox training data pipeline.

Responsibilities:
- Manifest schema (draft + final SFT).
- Pack a draft example into a Gemini Batch API request line.
- mm-strip and length-gate filters on Gemini responses.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

# ---------------------------------------------------------------------------
# Filters

_MM_PATTERNS = [
    re.compile(r"\b\d+(\.\d+)?\s*(mm|μm|um|microns?)\b", re.IGNORECASE),
    re.compile(r"\b(section|slice)\s*\d+\b", re.IGNORECASE),
    re.compile(r"\bbregma\b", re.IGNORECASE),
]


def mm_strip_filter(text: str) -> str | None:
    """Return text if no mm/coordinate/section-index leakage; else None."""
    for pat in _MM_PATTERNS:
        if pat.search(text):
            return None
    return text


def length_gate(text: str, min_sentences: int = 2, max_sentences: int = 6) -> bool:
    """Approximate sentence count via terminal punctuation."""
    n = sum(1 for ch in text if ch in ".!?")
    return min_sentences <= n <= max_sentences


# ---------------------------------------------------------------------------
# Prompt building

_SYSTEM_PROMPT = (
    "You are an expert neuroanatomist describing brain section morphology to "
    "a student. You describe how anatomical regions transform in shape, size, "
    "and boundary characteristics as the brain is sectioned along its "
    "anatomical axis. You use precise but accessible language. You never "
    "reference millimeter values, atlas coordinates, position numbers, or "
    "section indices."
)

_AXIS_BY_ORIENTATION = {
    "coronal": "anterior to posterior",
    "sagittal": "medial to lateral",
    "horizontal": "dorsal to ventral",
}

_USER_TEMPLATE_INSTRUCTIONS = (
    "Describe how this region transforms across the sections — its shape, "
    "size, boundary characteristics, and any sub-divisions visible. The "
    "bounding boxes are provided so you can locate the region in each "
    "section; you do not need to reference them in your description.\n\n"
    "Do NOT mention millimeter values, section indices, atlas coordinates, "
    "or position numbers. 2-4 sentences."
)


def _format_bboxes(example: dict) -> str:
    lines = []
    if example["is_hemisphere"]:
        lines.append("Per-section bounding boxes in section pixel coords [x1, y1, x2, y2]:")
        for idx, bbox in enumerate(example["bboxes"], start=1):
            lines.append(f"  Section {idx}: {bbox}")
    else:
        lines.append(
            "Per-section bounding boxes (left, right hemisphere) in section "
            "pixel coords [x1, y1, x2, y2]:"
        )
        for idx, pair in enumerate(example["bboxes"], start=1):
            lines.append(
                f"  Section {idx}: left={pair['left']}, right={pair['right']}"
            )
    return "\n".join(lines)


def build_user_text(example: dict) -> str:
    axis = _AXIS_BY_ORIENTATION[example["orientation"]]
    bboxes = _format_bboxes(example)
    return (
        f"Atlas: {example['atlas']}\n"
        f"Plane: {example['orientation']}\n"
        f"Region: {example['region']}\n"
        f"Section ordering: {axis}\n\n"
        f"{bboxes}\n\n"
        f"{_USER_TEMPLATE_INSTRUCTIONS}"
    )


def pack_batch_request(example: dict, images_b64: list[str]) -> dict:
    """Convert one draft example + N base64 PNGs into one Batch API request line."""
    image_parts = [
        {"inline_data": {"mime_type": "image/png", "data": b64}}
        for b64 in images_b64
    ]
    text_part = {"text": build_user_text(example)}
    return {
        "key": example["id"],
        "request": {
            "system_instruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
            "contents": [
                {"role": "user", "parts": image_parts + [text_part]}
            ],
            "generation_config": {
                "response_modalities": ["TEXT"],
                "temperature": 1.0,
                "thinking_config": {"thinking_level": "LOW"},
            },
        },
    }


# ---------------------------------------------------------------------------
# Manifest IO

def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------
# Response extraction

def extract_caption_from_response(response: dict) -> str | None:
    """Walk parts, drop thought parts, join text.

    Thought parts (`thought: true`) should only appear when
    `include_thoughts=True` is set, but this defensive filter prevents any
    accidental thought-part exposure from entering the SFT assistant content.
    """
    candidates = response.get("candidates", [])
    if not candidates:
        return None
    parts = candidates[0].get("content", {}).get("parts", [])
    answer_parts = [p.get("text", "") for p in parts if not p.get("thought")]
    text = "".join(answer_parts).strip()
    return text or None

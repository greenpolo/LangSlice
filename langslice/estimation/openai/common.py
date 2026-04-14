"""Shared helpers for OpenAI-compatible AP estimation workflows.

Mirrors ``langslice.estimation.google.common`` but adapted for the
Responses API message format.  Provider-agnostic helpers are
re-exported from the Google module so downstream code can import
everything from ``langslice.estimation.openai.common``.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from PIL import Image

from langslice.estimation.google.common import _APLoopState as _APLoopState
from langslice.estimation.google.common import _emit_trace as _emit_trace
from langslice.estimation.google.common import (
    _fetch_atlas_slice_bytes as _fetch_atlas_slice_bytes,
)
from langslice.estimation.google.common import (
    _get_position_range_lazy as _get_position_range_lazy,
)
from langslice.estimation.google.common import _GroupLoopState as _GroupLoopState
from langslice.estimation.google.common import _image_to_bytes as _image_to_bytes
from langslice.estimation.google.common import _load_atlas_lazy as _load_atlas_lazy
from langslice.estimation.google.common import _to_float as _to_float

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image encoding helpers
# ---------------------------------------------------------------------------


def _image_to_base64(image: Image.Image) -> str:
    """Convert a PIL Image to a base64 data URI (``data:image/jpeg;base64,...``).

    Uses :func:`_image_to_bytes` for the JPEG conversion step.
    """
    raw = _image_to_bytes(image)
    return _bytes_to_base64(raw)


def _bytes_to_base64(data: bytes, mime_type: str = "image/jpeg") -> str:
    """Convert raw bytes to a ``data:{mime_type};base64,...`` URI string."""
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


# ---------------------------------------------------------------------------
# Chat Completions content-part builders
# ---------------------------------------------------------------------------


def _build_image_content(b64_uri: str) -> dict[str, Any]:
    """Wrap a base64 data URI as a Responses API image content part.

    Returns::

        {"type": "input_image", "image_url": "<b64_uri>"}
    """
    return {"type": "input_image", "image_url": b64_uri}


def _build_text_content(text: str) -> dict[str, str]:
    """Wrap text as a Responses API text content part.

    Returns::

        {"type": "input_text", "text": "<text>"}
    """
    return {"type": "input_text", "text": text}


# ---------------------------------------------------------------------------
# Response extraction
# ---------------------------------------------------------------------------


def _extract_text(response: Any) -> str | None:
    """Extract the assistant's text from a Responses API response.

    Reads ``response.output_text``.  Returns ``None`` when no text content
    is available.
    """
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text:
        return text
    return None


def _extract_usage(response: Any) -> dict[str, int]:
    """Extract token usage from a Responses API response.

    Returns a dict with ``prompt_tokens``, ``completion_tokens``, and
    ``total_tokens``.  Missing fields default to ``0``.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    result: dict[str, int] = {}
    for field_name in fields:
        value = getattr(usage, field_name, None)
        result[field_name] = int(value) if isinstance(value, (int, float)) else 0
    return result


def _format_usage(usage: dict[str, int]) -> str:
    """Format a usage dict as a human-readable string.

    Example output::

        prompt=120, completion=45, total=165
    """
    parts: list[str] = []
    labels = (
        ("prompt_tokens", "prompt"),
        ("completion_tokens", "completion"),
        ("total_tokens", "total"),
    )
    for key, label in labels:
        value = usage.get(key)
        if value is not None:
            parts.append(f"{label}={value}")
    return ", ".join(parts) if parts else "usage unavailable"


# ---------------------------------------------------------------------------
# Message-history diagnostics
# ---------------------------------------------------------------------------


def _history_message_count(messages: list[dict[str, Any]]) -> dict[str, int]:
    """Count messages by role and total image content parts.

    Returns a dict with counts for ``system``, ``user``, ``assistant``,
    ``tool``, and ``image_parts``.
    """
    counts: dict[str, int] = {
        "system": 0,
        "user": 0,
        "assistant": 0,
        "tool": 0,
        "image_parts": 0,
    }

    for msg in messages:
        role = msg.get("role", "")
        if role in counts:
            counts[role] += 1

        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "input_image":
                    counts["image_parts"] += 1

    return counts

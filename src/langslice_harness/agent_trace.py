"""Helpers for structured agent trace events."""

from __future__ import annotations

import io
import json
from datetime import datetime

from PIL import Image


def _json_default(value: object) -> str:
    return str(value)


def format_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=_json_default)


def image_part_from_pil(
    image: Image.Image,
    *,
    label: str,
    image_format: str = "JPEG",
    mime_type: str | None = None,
    image_bytes: bytes | None = None,
    path: str | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    fmt = image_format.upper()
    if image_bytes is None:
        buf = io.BytesIO()
        prepared = image.convert("RGB") if image.mode != "RGB" and fmt == "JPEG" else image
        if fmt == "JPEG":
            prepared.save(buf, format=fmt, quality=95, subsampling=0)
        else:
            prepared.save(buf, format=fmt)
        image_bytes = buf.getvalue()
    if mime_type is None:
        mime_type = "image/png" if fmt == "PNG" else "image/jpeg"
    return {
        "kind": "image",
        "label": label,
        "mime_type": mime_type,
        "data": image_bytes,
        "width": int(image.width),
        "height": int(image.height),
        "path": path,
        "metadata": metadata or {},
    }


def text_part(
    text: str,
    *,
    label: str | None = None,
    monospace: bool = False,
    collapsible: bool = False,
) -> dict[str, object]:
    return {
        "kind": "text",
        "label": label,
        "text": text,
        "monospace": monospace,
        "collapsible": collapsible,
    }


def json_part(value: object, *, label: str, collapsible: bool = True) -> dict[str, object]:
    return text_part(
        format_json(value),
        label=label,
        monospace=True,
        collapsible=collapsible,
    )


def make_event(
    *,
    event_type: str,
    stage: str,
    role: str,
    title: str,
    summary: str | None = None,
    parts: list[dict[str, object]] | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "event_type": event_type,
        "stage": stage,
        "role": role,
        "title": title,
        "summary": summary,
        "parts": list(parts or []),
        "metadata": dict(metadata or {}),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def stage_event(stage: str, title: str, summary: str | None = None) -> dict[str, object]:
    return make_event(
        event_type="stage",
        stage=stage,
        role="system",
        title=title,
        summary=summary,
    )


def status_event(stage: str, message: str) -> dict[str, object]:
    return make_event(
        event_type="status",
        stage=stage,
        role="runtime",
        title=message,
    )


def tool_call_event(
    *,
    stage: str,
    tool_name: str,
    args: dict[str, object],
    iteration: int | None = None,
) -> dict[str, object]:
    title = f"Tool Call: {tool_name}"
    metadata: dict[str, object] = {"tool_name": tool_name, "args": args}
    if iteration is not None:
        metadata["iteration"] = iteration
    return make_event(
        event_type="tool_call",
        stage=stage,
        role="model",
        title=title,
        summary=None if not args else ", ".join(f"{key}={value}" for key, value in args.items()),
        parts=[json_part(args, label="Arguments")],
        metadata=metadata,
    )


def tool_result_event(
    *,
    stage: str,
    tool_name: str,
    summary: str | None = None,
    parts: list[dict[str, object]] | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    event_metadata = dict(metadata or {})
    event_metadata.setdefault("tool_name", tool_name)
    return make_event(
        event_type="tool_result",
        stage=stage,
        role="tool",
        title=f"Tool Result: {tool_name}",
        summary=summary,
        parts=parts,
        metadata=event_metadata,
    )


def model_event(
    *,
    stage: str,
    title: str,
    summary: str | None = None,
    parts: list[dict[str, object]] | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return make_event(
        event_type="model",
        stage=stage,
        role="model",
        title=title,
        summary=summary,
        parts=parts,
        metadata=metadata,
    )


def runtime_event(
    *,
    stage: str,
    title: str,
    summary: str | None = None,
    parts: list[dict[str, object]] | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return make_event(
        event_type="runtime",
        stage=stage,
        role="runtime",
        title=title,
        summary=summary,
        parts=parts,
        metadata=metadata,
    )

"""Sanitized, in-memory events for hosts observing a linear agent session.

Only public content fields cross this boundary. Provider signatures, encrypted
reasoning and model configuration are never serialized or forwarded.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langslice.adk import TOOL_MEDIA_PARTS_KEY
from langslice.linear.trace import _model_name

logger = logging.getLogger(__name__)
LiveCallback = Callable[[dict[str, Any]], None]


def _plain(value: Any) -> Any:
    """Copy JSON content without falling back to object reprs or media bytes."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {
            key: _plain(item) for key, item in value.items()
            if isinstance(key, str)
            and key not in {TOOL_MEDIA_PARTS_KEY, "thought_signature", "encrypted_content"}
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return "<non-text content>"


def _images(parts: Any, *, label: str) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    for part in parts or []:
        text = getattr(part, "text", None)
        if isinstance(text, str) and text:
            label = text
        inline = getattr(part, "inline_data", None)
        mime = getattr(inline, "mime_type", None)
        data = getattr(inline, "data", None)
        if isinstance(mime, str) and mime.startswith("image/") and isinstance(data, bytes):
            images.append({"data": bytes(data), "mime_type": mime, "label": label})
    return images


class LiveEvents:
    """Translate ADK content into detached UI events, isolating host failures."""

    def __init__(self, callback: LiveCallback) -> None:
        self.callback = callback
        self._streamed = {"text": "", "reasoning": ""}

    def emit(self, kind: str, **fields: Any) -> None:
        try:
            self.callback({"kind": kind, **fields})
        except Exception:
            logger.warning("Live session observer failed", exc_info=True)

    def start(self, agent: Any, seed: Any) -> None:
        instruction = getattr(agent, "instruction", "")
        self.emit("session", model=_model_name(getattr(agent, "model", "")),
                  text=instruction if isinstance(instruction, str) else "")
        parts = getattr(seed, "parts", None) or []
        self.emit("seed", text="\n".join(
            part.text for part in parts
            if isinstance(getattr(part, "text", None), str)
            and not getattr(part, "thought", False)
        ), images=_images(parts, label="Agent input"))

    def event(self, event: Any) -> None:
        partial = bool(getattr(event, "partial", False))
        parts = getattr(getattr(event, "content", None), "parts", None) or []
        texts: dict[str, list[str]] = {"text": [], "reasoning": []}
        for part in parts:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                texts["reasoning" if getattr(part, "thought", False) else "text"].append(text)
        for kind, chunks in texts.items():
            text = "".join(chunks)
            if partial:
                self._streamed[kind] += text
                if text:
                    self.emit(kind, text=text)
            elif text:
                previous = self._streamed[kind]
                # Providers aggregate final text after yielding token deltas.
                # A differing final version is a replacement, not more tokens.
                if text.startswith(previous):
                    remaining = text[len(previous):]
                    if remaining:
                        self.emit(kind, text=remaining)
                else:
                    self.emit(kind, text=text, revised=True)
        if partial:
            return
        self._streamed = {"text": "", "reasoning": ""}
        for part in parts:
            call = getattr(part, "function_call", None)
            if call is not None:
                self.emit("tool_call", name=getattr(call, "name", ""),
                          id=getattr(call, "id", None), args=_plain(getattr(call, "args", {})))
            response = getattr(part, "function_response", None)
            if response is not None:
                name = getattr(response, "name", "")
                self.emit("tool_result", name=name, id=getattr(response, "id", None),
                          response=_plain(getattr(response, "response", None)),
                          images=_images(getattr(response, "parts", None), label=name))

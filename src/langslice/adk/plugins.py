"""ADK plugins used by the position-estimation harness."""

from __future__ import annotations

import asyncio
import io
import json
import time
from pathlib import Path
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types
from PIL import Image

#: Working-set bounds for tool-returned images, in images. Every call resends
#: the whole history, so the images in context are the run's cost: run 4 on
#: M04 (2026-09-07) carried 170 images / 93k input tokens on its last call
#: and 1.3M over the run, at ~480 tokens per 512-px image. Upstream prompt
#: caching only pays when the prefix is byte-stable, so the set is trimmed in
#: batches (from above HIGH down to LOW) rather than one call at a time.
DEFAULT_HIGH_WATER_IMAGES = 48
DEFAULT_LOW_WATER_IMAGES = 16

#: Media-bearing tool calls kept whatever the bounds say. Two, so the model
#: can always compare its newest sweep against the one before it — a single
#: oversized sweep must not evict every other atlas image in context.
MIN_KEEP_TOOL_CALLS = 2

_DROPPED_TOOL = "dropped from context to keep the working set small. Call again to see them."
_DROPPED_USER = "(image dropped from context; `view_slices` shows it again)"


class WorkingSetImages:
    """Bound the images in context to a working set, trimming in batches.

    One instance per session (it remembers how much of the history is cut):
    the cut only ever moves forward, so between trims the request prefix is
    byte-identical and the upstream prompt cache can hit. Walking oldest to
    newest, tool images before the cut are dropped and their JSON says so;
    the newest :data:`MIN_KEEP_TOOL_CALLS` calls always keep their pixels.
    Once anything has been cut, user-message images older than the cut (the
    seed strip) go too — by then the stack has been reviewed through the
    tools and any section can be shown again on request.
    """

    def __init__(
        self,
        *,
        high: int = DEFAULT_HIGH_WATER_IMAGES,
        low: int = DEFAULT_LOW_WATER_IMAGES,
    ) -> None:
        if low > high:
            raise ValueError(f"low water {low} above high water {high}")
        self.high = high
        self.low = low
        self.cut = 0  # media-bearing tool calls dropped, oldest first
        self.trims = 0

    def __call__(self, contents: list[types.Content]) -> list[types.Content]:
        sites = [
            (ci, pi, len(part.function_response.parts))
            for ci, content in enumerate(contents)
            for pi, part in enumerate(content.parts or [])
            if part.function_response is not None and part.function_response.parts
        ]
        kept = sum(n for _, _, n in sites[self.cut :])
        if kept > self.high:
            cut = self.cut
            while kept > self.low and len(sites) - cut > MIN_KEEP_TOOL_CALLS:
                kept -= sites[cut][2]
                cut += 1
            if cut != self.cut:
                self.cut = cut
                self.trims += 1
        if self.cut == 0:
            return contents
        first_kept_content = sites[self.cut][0] if self.cut < len(sites) else len(contents)
        out = list(contents)
        for ci, pi, _ in sites[: self.cut]:
            parts = list(out[ci].parts or [])
            stale = parts[pi]
            assert stale.function_response is not None
            response = stale.function_response.response
            # The result must not go on saying "attached" about pixels that
            # are gone: the model reads that as a delivery failure.
            honest = (
                {**response, "images": _DROPPED_TOOL} if isinstance(response, dict) else response
            )
            parts[pi] = stale.model_copy(
                update={
                    "function_response": stale.function_response.model_copy(
                        update={"parts": None, "response": honest}
                    )
                }
            )
            out[ci] = out[ci].model_copy(update={"parts": parts})
        for ci in range(first_kept_content):
            content = out[ci]
            if content.role != "user" or not any(
                part.inline_data is not None for part in content.parts or []
            ):
                continue
            parts = [
                types.Part.from_text(text=_DROPPED_USER) if part.inline_data is not None else part
                for part in content.parts or []
            ]
            out[ci] = content.model_copy(update={"parts": parts})
        return out


class ModelCallPacingPlugin(BasePlugin):
    """Sleep before each ADK model request for eval quota pacing."""

    def __init__(
        self,
        delay_s: float,
        *,
        name: str = "langslice_model_call_pacing",
    ) -> None:
        super().__init__(name)
        self.delay_s = max(0.0, float(delay_s))

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        del callback_context, llm_request
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)
        return None


def part_summary(part: types.Part) -> dict[str, Any]:
    """Describe a part without copying its payload (media stays out of traces).

    Also used by the linear session tracer, which passes
    ``FunctionResponsePart``s here — same duck-typed attributes.
    """
    inline = getattr(part, "inline_data", None)
    if inline is not None:
        data = getattr(inline, "data", None)
        summary: dict[str, Any] = {
            "kind": "inline_data",
            "mime_type": getattr(inline, "mime_type", None),
            "byte_count": len(data) if data else 0,
            "media_resolution": str(getattr(part, "media_resolution", None) or ""),
        }
        if data:
            try:
                with Image.open(io.BytesIO(data)) as img:
                    summary["dimensions"] = [img.width, img.height]
            except Exception:
                summary["dimensions"] = None
        return summary

    file_data = getattr(part, "file_data", None)
    if file_data is not None:
        return {
            "kind": "file_data",
            "mime_type": getattr(file_data, "mime_type", None),
            "has_file_uri": bool(getattr(file_data, "file_uri", None)),
            "media_resolution": str(getattr(part, "media_resolution", None) or ""),
        }

    function_call = getattr(part, "function_call", None)
    if function_call is not None:
        return {
            "kind": "function_call",
            "name": getattr(function_call, "name", None),
            "arg_keys": sorted((getattr(function_call, "args", None) or {}).keys()),
        }

    function_response = getattr(part, "function_response", None)
    if function_response is not None:
        response = getattr(function_response, "response", None) or {}
        # ADK 2.7+ carries tool-returned media here, not as sibling content
        # parts, so count it or the capture hides the images entirely.
        media_parts = getattr(function_response, "parts", None) or []
        return {
            "kind": "function_response",
            "name": getattr(function_response, "name", None),
            "id": getattr(function_response, "id", None),
            "response_keys": sorted(response.keys()) if isinstance(response, dict) else [],
            "media_part_count": len(media_parts),
        }

    text = getattr(part, "text", None)
    if text is not None:
        return {"kind": "text", "char_count": len(text)}

    return {"kind": "other"}


class RequestCapturePlugin(BasePlugin):
    """Write redacted ADK model-request summaries for transport debugging."""

    def __init__(
        self,
        capture_dir: str | Path,
        *,
        run_label: str = "adk_request",
        name: str = "langslice_request_capture",
    ):
        super().__init__(name)
        self.capture_dir = Path(capture_dir)
        self.run_label = run_label
        self._counter = 0
        # Per-instance suffix so concurrent sessions don't clobber each other's
        # capture files (run_label alone is shared across all single-slice runs).
        import uuid as _uuid
        self._instance = _uuid.uuid4().hex[:8]

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        del callback_context
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self._counter += 1
        payload = {
            "model": str(getattr(llm_request, "model", "")),
            "turn_index": self._counter,
            "captured_at_unix": time.time(),
            "contents": [
                {
                    "role": getattr(content, "role", None),
                    "parts": [part_summary(part) for part in (content.parts or [])],
                }
                for content in (llm_request.contents or [])
            ],
        }
        path = self.capture_dir / f"{self.run_label}_{self._instance}_{self._counter:03d}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return None

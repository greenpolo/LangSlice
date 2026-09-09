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

#: Media-bearing tool calls whose images stay in context: the newest one.
#: The cost model, measured on the quota headers (runs 5 and 6, 2026-09-09):
#: a token that sits unchanged in the prefix is ~0.13x, a token after any
#: edit point is full price once, and an image is new (full price) the call
#: it arrives whatever happens later. Keeping tool images therefore costs
#: 0.13x the WHOLE accumulated image history on every call, which is
#: quadratic (run 6 climbed from 2.4k to 8.8k paid a call by call 15), while
#: dropping the previous call's images costs only the re-read of the small
#: text tail after them, because nothing image-heavy sits after the newest
#: result. So: the newest call keeps its pixels, everything older is text,
#: and the cut never moves back. The seed strip is different: it heads the
#: prefix and is never edited, so it rides at 0.13x for the whole run
#: (~1.2k a call at 36 sections) and the model always has one picture of
#: every section (Nash, 2026-09-09).
DEFAULT_KEEP_IMAGE_CALLS = 1

_DROPPED_TOOL = (
    "dropped from context: only the newest call's images are kept. Call again to see them."
)


class WorkingSetImages:
    """Keep only the newest tool call's images in context.

    One instance per session: the cut only moves forward, so the prefix
    before it is byte-identical from one request to the next and the
    upstream prompt cache pays for everything but the tail. Tool images
    before the cut are dropped and their JSON says so. User-message images
    (the seed strip) are never touched.
    """

    def __init__(self, *, keep_calls: int = DEFAULT_KEEP_IMAGE_CALLS) -> None:
        if keep_calls < 1:
            raise ValueError("keep_calls must be at least 1")
        self.keep_calls = keep_calls
        self.cut = 0  # media-bearing tool calls dropped, oldest first

    def __call__(self, contents: list[types.Content]) -> list[types.Content]:
        sites = [
            (ci, pi)
            for ci, content in enumerate(contents)
            for pi, part in enumerate(content.parts or [])
            if part.function_response is not None and part.function_response.parts
        ]
        if not sites:
            return contents
        self.cut = max(self.cut, len(sites) - self.keep_calls)
        if self.cut == 0:
            return contents
        out = list(contents)
        for ci, pi in sites[: self.cut]:
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

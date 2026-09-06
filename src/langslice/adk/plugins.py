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

# Tool images kept in context, newest first. 24 was sized for 8-image atlas
# sweeps; once every write returned its picture (16 per set_positions batch)
# it stripped every call but the newest two, and GPT-6 Astra reported, twice,
# that early tool results "said attached but showed no image" (2026-09-06).
# 128 covers a whole positioning run (a 134-image request was probed fine);
# what is dropped now says so in the tool result instead of claiming
# attachment.
DEFAULT_KEEP_LAST_TOOL_IMAGES = 128

#: Media-bearing tool calls kept whatever the budget says. Two, so the model
#: can always compare its newest sweep against the one before it — a single
#: oversized sweep must not evict every other atlas image in context.
MIN_KEEP_TOOL_CALLS = 2


def trim_stale_tool_images(
    contents: list[types.Content], *, keep_last: int = DEFAULT_KEEP_LAST_TOOL_IMAGES
) -> list[types.Content]:
    """Drop tool-returned images from all but the most recent tool calls.

    Without this, every atlas sweep is replayed to the model on every later
    turn (~2.75x prompt-token redundancy measured over real runs). The JSON
    tool results still name every fetched position, and the model can
    re-fetch any position it wants to see again, so only the newest calls
    keep their pixels: walking newest-to-oldest, the newest
    :data:`MIN_KEEP_TOOL_CALLS` calls are kept whatever their size, further
    media is kept while the running total stays within *keep_last*, and every
    call older than the first drop is dropped too. User-message images (the
    histology slice) are never touched.
    """
    media_sites = [
        (ci, pi, len(part.function_response.parts))
        for ci, content in enumerate(contents)
        for pi, part in enumerate(content.parts or [])
        if part.function_response is not None and part.function_response.parts
    ]
    kept = 0
    dropping = False
    drop: list[tuple[int, int]] = []
    for rank, (ci, pi, n) in enumerate(reversed(media_sites)):
        if rank >= MIN_KEEP_TOOL_CALLS and (dropping or kept + n > keep_last):
            dropping = True
            drop.append((ci, pi))
        else:
            kept += n
    if not drop:
        return contents
    out = list(contents)
    note = (
        f"dropped from context: older than the newest {keep_last} tool "
        "images. Call again to see them."
    )
    for ci, pi in drop:
        parts = list(out[ci].parts or [])
        stale = parts[pi]
        assert stale.function_response is not None
        response = stale.function_response.response
        # The result must not go on saying "attached" about pixels that are
        # gone: the model reads that as a delivery failure.
        honest = {**response, "images": note} if isinstance(response, dict) else response
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

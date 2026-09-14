"""ADK plugins used by the position-estimation harness."""

from __future__ import annotations

import asyncio
import io
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types
from PIL import Image

from langslice.adk import TOOL_MEDIA_DELIVERY_ID_KEY

#: Images in context before the working set is cut, and what it is cut to.
#: The upstream cache prices a token that sits unchanged in the prefix at
#: ~0.13x (measured against the quota headers, run 5, 2026-09-09), and an
#: image at the atlas's resolution is ~260 tokens, so images that STAY are
#: cheap and images that are removed cost a cache break at the removal point
#: on every later call. So: keep everything, cut rarely and in one batch, and
#: never touch the seed strip at the head of the prefix. 256 images at ~260
#: tokens is ~67k raw, ~8.7k paid per call at the 0.13x cache rate; a normal
#: run (seed 80, ~50 compares, ~40 write pictures) stays under it.
DEFAULT_MAX_IMAGES = 256
DEFAULT_KEEP_IMAGES = 128
#: Tools that open the transform stage. The first media-bearing call to one
#: of them is a stage boundary: every older tool image (the compares and
#: write pictures of positioning) is cut in one batch. Positions are written
#: and reviewed by then, so those pictures only cost their cached carry —
#: ~45k tokens a call on Astra's run 17 (2026-09-10), ~9% of a window over
#: the transform stage — against one re-read of the positioning text.
STAGE_BOUNDARY_TOOLS = frozenset(
    {"fit_affine", "adjust_transform", "adjust_transforms",
     "view_landmarks", "edit_landmarks", "warp_landmarks"}
)

#: Astra's run-19 debrief read the old wording ("dropped from context") as
#: "never delivered" and doubted comparisons it had actually made.
_DROPPED_TOOL = (
    "were shown when this call returned and have since been dropped from "
    "context to bound the request; call again to see them."
)


class WorkingSetImages:
    """Keep tool images in context until there are too many, then cut once.

    One instance per session: the cut only moves forward, oldest media-bearing
    tool calls first, so the prefix before it is byte-identical from one
    request to the next and the upstream prompt cache pays for everything but
    the newest tail. Tool images before the cut are dropped and their JSON
    says so. The seed strip (user-message images) is never touched: it sits
    at the head of the prefix, cached, and is the one picture of every
    section the model always has.
    """

    def __init__(
        self, *, max_images: int = DEFAULT_MAX_IMAGES, keep_images: int = DEFAULT_KEEP_IMAGES
    ) -> None:
        if not 0 < keep_images <= max_images:
            raise ValueError("need 0 < keep_images <= max_images")
        self.max_images = max_images
        self.keep_images = keep_images
        self.cut = 0  # media-bearing tool calls dropped, oldest first

    def __call__(self, contents: list[types.Content]) -> list[types.Content]:
        sites = [
            (ci, pi, len(part.function_response.parts))
            for ci, content in enumerate(contents)
            for pi, part in enumerate(content.parts or [])
            if part.function_response is not None and part.function_response.parts
        ]
        if not sites:
            return contents
        boundary = next(
            (
                k
                for k, (ci, pi, _) in enumerate(sites)
                if (contents[ci].parts or [])[pi].function_response.name  # type: ignore[union-attr]
                in STAGE_BOUNDARY_TOOLS
            ),
            None,
        )
        if boundary is not None:
            self.cut = max(self.cut, boundary)
        live = sum(n for _, _, n in sites[self.cut :])
        if live > self.max_images:
            cut = self.cut
            while cut < len(sites) and live > self.keep_images:
                live -= sites[cut][2]
                cut += 1
            self.cut = cut
        if self.cut == 0:
            return contents
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


class ToolMediaDeliveryPlugin(BasePlugin):
    """Report media-bearing function responses present in a model request.

    Register this after the context filter. The callback therefore observes
    the request after working-set pruning and can distinguish an image that
    was successfully rendered from one the model is actually about to see.
    """

    def __init__(
        self,
        delivered: Callable[[set[str]], None],
        *,
        name: str = "langslice_tool_media_delivery",
    ) -> None:
        super().__init__(name)
        self.delivered = delivered

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        del callback_context
        delivery_ids = {
            str(response[TOOL_MEDIA_DELIVERY_ID_KEY])
            for content in (llm_request.contents or [])
            for part in (content.parts or [])
            if (function_response := part.function_response) is not None
            and function_response.parts
            and isinstance((response := function_response.response), dict)
            and response.get(TOOL_MEDIA_DELIVERY_ID_KEY)
        }
        if delivery_ids:
            self.delivered(delivery_ids)
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

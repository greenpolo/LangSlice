"""ADK plugins of the session: pacing, retired names, strict arguments, media, capture."""

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

from langslice.doors.tools import TOOL_MEDIA_DELIVERY_ID_KEY

#: Images in context before the working set is cut, and what it is cut to.
#: Every image is kept; the oldest are cut in ONE batch only at a backstop far
#: above any run (a third of the 1,500 images the OpenAI lane accepts per
#: request), as coding agents do. An agent chooses later steps from earlier
#: pictures (its stain, its channels), and a cut breaks the prompt cache at
#: the cut point, so cutting rarely is also the cheap choice.
DEFAULT_MAX_IMAGES = 500
DEFAULT_KEEP_IMAGES = 250

#: Says the pictures were delivered: "dropped from context" alone reads as
#: "never shown".
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


class RetiredToolsPlugin(BasePlugin):
    """Answer a call of a retired tool with the tool to use instead.

    ADK (2.11) resolves a call's tool before its callbacks run and, for a
    name it does not know, hands the plugins' ``before_tool_callback`` a
    placeholder tool of that name; the "tool not found" error comes only
    when no callback answers. This answers a name in
    :data:`langslice.ops.registry.RETIRED` with
    :func:`~langslice.ops.registry.retired_payload` (``RETIRED_TOOL``,
    nothing done), every time it is called; any other name is left to ADK.
    """

    def __init__(self, *, name: str = "langslice_retired_tools") -> None:
        super().__init__(name)

    async def before_tool_callback(
        self, *, tool: Any, tool_args: dict[str, Any], tool_context: Any,
    ) -> dict[str, Any] | None:
        del tool_args, tool_context
        if getattr(tool, "func", None) is not None:  # a real tool of the run
            return None
        from langslice.ops.registry import retired_payload

        return retired_payload(str(getattr(tool, "name", "") or ""))


class StrictArgumentsPlugin(BasePlugin):
    """Refuse a tool call that carries arguments the tool does not take.

    ADK's ``FunctionTool`` keeps only the arguments its function names and
    drops the rest without a word, so a misplaced argument (a picture option
    outside ``view``) used to run the tool with its default. This answers such
    a call instead, before it runs, with
    :func:`langslice.doors.tools.arguments.argument_refusal`: the stray keys named
    and the accepted ones listed. Tools without a Python function are left
    alone.
    """

    def __init__(self, *, name: str = "langslice_strict_arguments") -> None:
        super().__init__(name)

    async def before_tool_callback(
        self, *, tool: Any, tool_args: dict[str, Any], tool_context: Any,
    ) -> dict[str, Any] | None:
        del tool_context
        func = getattr(tool, "func", None)
        if func is None:
            return None
        from langslice.doors.tools.arguments import argument_refusal

        try:
            return argument_refusal(func, dict(tool_args or {}))
        except (TypeError, ValueError):  # an unintrospectable tool keeps ADK's behaviour
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

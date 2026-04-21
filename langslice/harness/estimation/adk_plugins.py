"""ADK plugins used by the position-estimation harness."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from PIL import Image

_MULTIMODAL_TOOL_HISTORY_KEY = "temp:LANGSLICE_MULTIMODAL_TOOL_HISTORY"


def _as_part_list(result: Any) -> list[types.Part] | None:
    if isinstance(result, types.Part):
        return [result]
    if (
        isinstance(result, list)
        and result
        and all(isinstance(part, types.Part) for part in result)
    ):
        return list(result)
    return None


def _coerce_part_history(history: Any) -> list[types.Part]:
    """Normalize session-state history back to concrete ``types.Part`` objects.

    ADK session state may serialize pydantic ``Part`` instances into plain
    dictionaries between callbacks. Model requests must receive real Part
    objects, not dicts, because downstream SDK code accesses attributes such as
    ``part.inline_data``.
    """
    if not isinstance(history, list):
        return []
    parts: list[types.Part] = []
    for item in history:
        if isinstance(item, types.Part):
            parts.append(item)
        elif isinstance(item, dict):
            parts.append(types.Part.model_validate(item))
    return parts


class PersistentMultimodalToolResultsPlugin(BasePlugin):
    """Replay all multimodal tool results on every later model turn.

    ADK's stock multimodal-results plugin appends returned ``types.Part`` values
    to only the next LLM request. LangSlice's legacy non-ADK loop resent the
    full multimodal history every turn, so AP estimation expects earlier atlas
    sweeps to remain visible while the model narrows and verifies candidates.
    """

    def __init__(self, name: str = "langslice_persistent_multimodal_tool_results"):
        super().__init__(name)

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: Any,
    ) -> Any | None:
        del tool, tool_args
        parts = _as_part_list(result)
        if parts is None:
            return result

        history = _coerce_part_history(
            tool_context.state.get(_MULTIMODAL_TOOL_HISTORY_KEY, [])
        )
        history.extend(parts)
        tool_context.state[_MULTIMODAL_TOOL_HISTORY_KEY] = history
        return None

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        history = _coerce_part_history(
            callback_context.state.get(_MULTIMODAL_TOOL_HISTORY_KEY, [])
        )
        if history and llm_request.contents:
            parts = llm_request.contents[-1].parts
            if parts is None:
                llm_request.contents[-1].parts = history
            else:
                parts.extend(history)
        return None


def _part_summary(part: types.Part) -> dict[str, Any]:
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
        return {
            "kind": "function_response",
            "name": getattr(function_response, "name", None),
            "response_keys": sorted(response.keys()) if isinstance(response, dict) else [],
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
                    "parts": [_part_summary(part) for part in (content.parts or [])],
                }
                for content in (llm_request.contents or [])
            ],
        }
        path = self.capture_dir / f"{self.run_label}_{self._counter:03d}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return None

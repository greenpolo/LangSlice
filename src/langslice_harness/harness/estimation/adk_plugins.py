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
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from PIL import Image

MULTIMODAL_PARTS_RESULT_KEY = "_langslice_multimodal_parts"
_MULTIMODAL_TOOL_HISTORY_KEY = "temp:LANGSLICE_MULTIMODAL_TOOL_HISTORY"
_MULTIMODAL_TOOL_PARTS_BY_CALL_KEY = "temp:LANGSLICE_MULTIMODAL_TOOL_PARTS_BY_CALL"


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


def _coerce_part_map(history: Any) -> dict[str, list[types.Part]]:
    """Normalize a call-id keyed part map restored from ADK session state."""
    if not isinstance(history, dict):
        return {}
    out: dict[str, list[types.Part]] = {}
    for key, value in history.items():
        parts = _coerce_part_history(value)
        if parts:
            out[str(key)] = parts
    return out


class PersistentMultimodalToolResultsPlugin(BasePlugin):
    """Inject multimodal tool artifacts into later model requests.

    LangSlice tools return ordinary dictionary function responses, plus a
    private ``MULTIMODAL_PARTS_RESULT_KEY`` entry containing the image/text parts
    the model should see. This plugin removes the private entry before ADK
    builds the function_response, stores the parts keyed by function-call id,
    and appends them immediately after that function_response on model requests.
    Persistent mode mirrors the legacy non-ADK loop by replaying earlier atlas
    sweeps on every later turn.
    """

    def __init__(
        self,
        *,
        persistent: bool = True,
        name: str = "langslice_persistent_multimodal_tool_results",
    ):
        super().__init__(name)
        self.persistent = persistent

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: Any,
    ) -> Any | None:
        del tool_args
        if isinstance(result, dict) and MULTIMODAL_PARTS_RESULT_KEY in result:
            parts = _as_part_list(result.get(MULTIMODAL_PARTS_RESULT_KEY))
            public_result = dict(result)
            public_result.pop(MULTIMODAL_PARTS_RESULT_KEY, None)
            if parts is not None:
                parts_by_call = _coerce_part_map(
                    tool_context.state.get(_MULTIMODAL_TOOL_PARTS_BY_CALL_KEY, {})
                )
                call_id = str(
                    getattr(tool_context, "function_call_id", None)
                    or f"{tool.name}:{len(parts_by_call) + 1}"
                )
                parts_by_call[call_id] = parts
                tool_context.state[_MULTIMODAL_TOOL_PARTS_BY_CALL_KEY] = parts_by_call
            return public_result

        # Backward-compatible path for older tools/tests that return raw Parts.
        parts = _as_part_list(result)
        if parts is not None:
            history = _coerce_part_history(
                tool_context.state.get(_MULTIMODAL_TOOL_HISTORY_KEY, [])
            )
            history.extend(parts)
            tool_context.state[_MULTIMODAL_TOOL_HISTORY_KEY] = history
            return {"status": "ok"}

        return result

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        parts_by_call = _coerce_part_map(
            callback_context.state.get(_MULTIMODAL_TOOL_PARTS_BY_CALL_KEY, {})
        )
        consumed: set[str] = set()
        if parts_by_call and llm_request.contents:
            for content in llm_request.contents:
                if not content.parts:
                    continue
                modified_parts: list[types.Part] = []
                for part in content.parts:
                    modified_parts.append(part)
                    function_response = getattr(part, "function_response", None)
                    if function_response is None:
                        continue
                    call_id = str(getattr(function_response, "id", "") or "")
                    if call_id and call_id in parts_by_call:
                        modified_parts.extend(parts_by_call[call_id])
                        consumed.add(call_id)
                content.parts = modified_parts
            unmatched = [call_id for call_id in parts_by_call if call_id not in consumed]
            if unmatched:
                tail = llm_request.contents[-1]
                if tail.parts is None:
                    tail.parts = []
                for call_id in unmatched:
                    tail.parts.extend(parts_by_call[call_id])
                consumed.update(unmatched)
            if not self.persistent and consumed:
                for call_id in consumed:
                    parts_by_call.pop(call_id, None)
                callback_context.state[_MULTIMODAL_TOOL_PARTS_BY_CALL_KEY] = parts_by_call

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
            "id": getattr(function_response, "id", None),
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
                    "parts": [_part_summary(part) for part in (content.parts or [])],
                }
                for content in (llm_request.contents or [])
            ],
        }
        path = self.capture_dir / f"{self.run_label}_{self._instance}_{self._counter:03d}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return None

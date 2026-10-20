"""The bodies of plugin tools: what the tool door runs for a call.

A plugin tool is declared from its own stub (:func:`langslice.doors.declarations.declaration`
reads it from the variant) and wrapped by the toolbox exactly as a native tool
is: opening-read gate for writes, strict argument check, stale-deformation
sweep, picture saving, background notices, the job's write lock. This module
supplies the part between: it makes the call's :class:`ToolContext`, checks
the tool's per-call limits, runs the handler and turns its
:class:`~langslice.core.plugins.ToolResult` into the reply, with the pictures
under the media key like every native tool's.
"""

from __future__ import annotations

import inspect
import logging
import uuid
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from langslice.core import plugins as core
from langslice.core.plugins import ToolError, ToolResult, ToolSpec
from langslice.doors.declarations import Variant, declaration
from langslice.doors.plugins.context import Caller, ToolContext
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.ops.refusal import Refused

if TYPE_CHECKING:
    from langslice.doors.tools.door import Door

logger = logging.getLogger(__name__)


def _plain(value: Any) -> Any:
    """*value* as JSON values (numpy and tuples made plain); anything else as text."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    return str(value)


def _picture(item: Any) -> Image.Image | str:
    if isinstance(item, (Image.Image, str)):
        return item
    return Image.fromarray(np.asarray(item))


def _over_limit(tool: ToolSpec, arguments: Mapping[str, Any]) -> dict[str, Any] | None:
    cap = tool.limits.get("sections")
    if cap is not None:
        asked = arguments.get("sections")
        count = len(asked) if isinstance(asked, (list, tuple)) else 0
        if count > cap:
            return {"status": "error", "error": "TOO_MANY_SECTIONS", "max_sections": cap,
                    "requested": count}
    return None


def reply_of(tool: ToolSpec, result: Any) -> dict[str, Any]:
    """The reply a handler's *result* makes: its ``reply`` (``status`` ``ok``
    unless it says otherwise), and its pictures under the media key, at most
    ``limits["pictures"]`` of them (the rest counted in ``pictures_not_shown``)."""
    if not isinstance(result, ToolResult):
        return {"status": "error", "error": "PLUGIN_FAILED", "tool": tool.name,
                "message": f"{tool.name}'s handler returned {type(result).__name__}, "
                "not a ToolResult"}
    reply = _plain(dict(result.reply))
    if not isinstance(reply, dict):
        reply = {"result": reply}
    out: dict[str, Any] = {"status": "ok", **reply}
    pictures = [_picture(item) for item in result.pictures]
    cap = tool.limits.get("pictures")
    if cap is not None and sum(isinstance(item, Image.Image) for item in pictures) > cap:
        kept: list[Image.Image | str] = []
        shown = 0
        for item in pictures:
            if isinstance(item, Image.Image):
                if shown >= cap:
                    continue
                shown += 1
            kept.append(item)
        out["pictures_not_shown"] = len(pictures) - len(kept)
        pictures = kept
    if pictures:
        out[TOOL_MEDIA_PARTS_KEY] = pictures
    return out


def body(door: Door, tool: ToolSpec, variant: Variant) -> Callable[..., Any]:
    """The body of *tool* for *door*: takes the arguments its declaration in
    *variant* has, by name."""

    def run(**arguments: Any) -> Any:
        tool_context = arguments.pop("tool_context", None)  # ADK's: the call's id
        refused = _over_limit(tool, arguments)
        if refused is not None:
            return refused
        call_id = getattr(tool_context, "function_call_id", None) or uuid.uuid4().hex[:8]
        context = ToolContext(door, Caller.of_tool(tool), str(call_id))
        try:
            if tool.kind == "write" and not tool.long:
                with context.writing():  # everything the call writes is one undo step
                    result = tool.handler(context, **arguments)
            else:  # a long tool takes the lock only to apply; its writes are still one step
                result = tool.handler(context, **arguments)
        except ToolError as exc:
            return exc.payload()
        except Refused as refusal:
            return refusal.payload()
        except Exception as exc:  # noqa: BLE001 - a plugin's failure is a reply, not a crash
            logger.exception("plugin tool %s failed", tool.name)
            return {"status": "error", "error": "PLUGIN_FAILED", "tool": tool.name,
                    "message": f"{type(exc).__name__}: {exc}"}
        return reply_of(tool, result)

    declared = declaration(tool.name, variant)
    run.__signature__ = declared.signature.replace(  # type: ignore[attr-defined]
        parameters=[*declared.signature.parameters.values(), inspect.Parameter(
            "tool_context", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None)],
        return_annotation=inspect.Signature.empty)
    run.__name__ = tool.name
    return run


def bodies(door: Door, variant: Variant) -> dict[str, Callable[..., Any]]:
    """The bodies of every plugin tool the door's job has, by name."""
    return {name: body(door, tool, variant)
            for name, (_plugin, tool) in core.tool_specs(door.spec).items()}

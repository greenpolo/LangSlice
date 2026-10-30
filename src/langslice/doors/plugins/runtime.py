"""The bodies of plugin tools: what the tool door runs for a call.

A plugin tool is declared from its own stub
(:func:`langslice.tools.descriptions.render.declaration` reads it from the
run's facts), and each call runs through the tool pipeline's steps exactly as a
native tool's does (:data:`langslice.tools.pipeline.TOOL_STEPS`). This module
supplies the body: it makes the call's :class:`ToolContext`, checks the
tool's per-call limits, runs the handler and turns its
:class:`~langslice.core.plugins.ToolResult` into the reply, with the
pictures under the media key like every native tool's.
"""

from __future__ import annotations

import inspect
import logging
import uuid
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from langslice.core.jsonable import to_jsonable
from langslice.core.plugins import ToolResult, ToolSpec
from langslice.core.refusal import RefusalCode, Refused
from langslice.doors.plugins.context import Caller, ToolContext
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.job import plugins as core
from langslice.tools.descriptions.description import DescriptionFacts
from langslice.tools.descriptions.render import declaration

if TYPE_CHECKING:
    from langslice.tools.run import Run

logger = logging.getLogger(__name__)


def _picture(item: Any) -> Image.Image | str:
    if isinstance(item, (Image.Image, str)):
        return item
    return Image.fromarray(np.asarray(item))


def _check_limit(tool: ToolSpec, arguments: Mapping[str, Any]) -> None:
    """Refuse a call naming more sections than the tool's limits allow.

    The refusal is ``TOO_MANY_SECTIONS``, with the cap and the count.
    """
    cap = tool.limits.get("sections")
    if cap is not None:
        named = arguments.get("sections")
        count = len(named) if isinstance(named, (list, tuple)) else 0
        if count > cap:
            raise Refused(
                RefusalCode.TOO_MANY_SECTIONS,
                max_sections=cap,
                requested=count,
                message=f"{tool.name} takes at most {cap} sections per call "
                f"(its plugin's limit); {count} were named. Nothing was done: "
                "split them over calls.")


def reply_of(tool: ToolSpec, result: Any) -> dict[str, Any]:
    """The reply a handler's *result* makes: its ``reply`` (``status`` ``ok``
    unless it says otherwise), and its pictures under the media key, at most
    ``limits["pictures"]`` of them (the rest counted in ``pictures_not_shown``);
    ``PLUGIN_FAILED`` when it is no :class:`ToolResult`."""
    if not isinstance(result, ToolResult):
        kind = type(result).__name__  # pyright: ignore[reportAny]  # a plugin's own reply
        raise Refused(RefusalCode.PLUGIN_FAILED, tool=tool.name,
                      message=f"{tool.name}'s handler returned {kind}, not a ToolResult")
    reply = to_jsonable(dict(result.reply))
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


def body(run: Run, tool: ToolSpec,
         facts: DescriptionFacts) -> Callable[..., Any]:
    """The body of *tool* for *run*: takes the arguments its declaration in
    the run of *facts* has, by name."""

    def answer(**arguments: Any) -> Any:
        tool_context = arguments.pop("tool_context", None)  # ADK's: the call's id
        _check_limit(tool, arguments)
        call_id = getattr(tool_context, "function_call_id", None) or uuid.uuid4().hex[:8]
        context = ToolContext(run, Caller.of_tool(tool), str(call_id))
        try:
            if tool.kind == "write" and not tool.long:
                with context.writing():  # everything the call writes is one undo step
                    result = tool.handler(context, **arguments)
            else:  # a long tool takes the lock only to apply; its writes are still one step
                result = tool.handler(context, **arguments)
        except Refused:  # a ToolError too: the tool door answers it
            raise
        except Exception as exc:  # noqa: BLE001 - a plugin's failure is a reply, not a crash
            logger.exception("plugin tool %s failed", tool.name)
            raise Refused(RefusalCode.PLUGIN_FAILED, tool=tool.name,
                          message=f"{type(exc).__name__}: {exc}") from exc
        return reply_of(tool, result)

    declared = declaration(tool.name, facts)
    answer.__signature__ = declared.signature.replace(  # type: ignore[attr-defined]
        parameters=[*declared.signature.parameters.values(), inspect.Parameter(
            "tool_context", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None)],
        return_annotation=inspect.Signature.empty)
    answer.__name__ = tool.name
    return answer


def bodies(run: Run,
           facts: DescriptionFacts) -> dict[str, Callable[..., Any]]:
    """The bodies of every plugin tool the run's job has, by name."""
    return {name: body(run, tool, facts)
            for name, (_plugin, tool) in core.tool_specs(run.spec).items()}

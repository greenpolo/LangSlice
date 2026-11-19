"""The bodies of plugin tools, and what a job records of its plugins.

A plugin tool is declared from its own stub
(:func:`langslice.tools.descriptions.render.declaration` reads it from the
run's facts), and each call runs through the tool pipeline's steps exactly as a
native tool's does (:data:`langslice.tools.pipeline.TOOL_STEPS`). This module
supplies the body: it makes the call's :class:`ToolContext`, checks the
tool's per-call limits, runs the handler and turns its
:class:`~langslice.core.plugins.ToolResult` into the reply, with the
pictures under the media key like every native tool's.

A job also records the plugins it ran with. Each plugin tool's description is
an experimental variable, so it is kept exactly as the model sees it: per
plugin the module, factory, versions, the SHA-256 of the source files that
declared its tools and the plugin API version
(:attr:`langslice.job.plugins.LoadedPlugin.provenance`), and per tool its
kind, group, task, writes, limits and its rendered declaration (the name,
description and argument schema in this job's run). The record is
``job.json``'s ``plugins`` and the run trace's ``plugins`` line
(:func:`plugin_records`); a change of what a job runs with is also an event in
``logs/events.jsonl`` (:func:`record_plugins`).
"""

from __future__ import annotations

import inspect
import logging
import uuid
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from PIL import Image

from langslice.core.jsonable import to_jsonable
from langslice.core.pictures.media_keys import TOOL_MEDIA_PARTS_KEY
from langslice.core.plugins import ToolResult, ToolSpec
from langslice.core.refusal import RefusalCode, Refused
from langslice.job import plugins as core
from langslice.job.job import Job
from langslice.job.settings.tool_table import ENGINES
from langslice.plugins.caller import Caller
from langslice.plugins.context import ToolContext
from langslice.tools.descriptions.description import DescriptionFacts
from langslice.tools.descriptions.render import (
    arguments_schema,
    context_parameter,
    declaration,
    signed,
)

if TYPE_CHECKING:
    from langslice.tools.run import Run

logger = logging.getLogger(__name__)

PLUGINS_KEY = "plugins"


def _picture(item: Any) -> Image.Image | str:
    if isinstance(item, (Image.Image, str)):
        return item
    return Image.fromarray(np.asarray(item))


def _check_limit(tool: ToolSpec, arguments: Mapping[str, Any]) -> None:
    """Refuse a call naming more sections than the tool's limits allow.

    The refusal is ``BAD_ARGUMENT``, with the cap and the count.
    """
    cap = tool.limits.get("sections")
    if cap is not None:
        named = arguments.get("sections")
        count = len(named) if isinstance(named, (list, tuple)) else 0
        if count > cap:
            raise Refused(
                RefusalCode.BAD_ARGUMENT,
                max_sections=cap,
                requested=count,
                message=f"{tool.name} takes at most {cap} sections per call "
                f"(its plugin's limit); {count} were named. Nothing was done: "
                "split them over calls.",
            )


def reply_of(tool: ToolSpec, result: Any) -> dict[str, Any]:
    """The reply a handler's *result* makes: its ``reply`` (``status`` ``ok``
    unless it says otherwise), and its pictures under the media key, at most
    ``limits["pictures"]`` of them (the rest counted in ``pictures_not_shown``);
    ``INTERNAL`` when it is no :class:`ToolResult`."""
    if not isinstance(result, ToolResult):
        kind = type(cast("object", result)).__name__
        raise Refused(
            RefusalCode.INTERNAL,
            tool=tool.name,
            message=f"{tool.name}'s handler returned {kind}, not a ToolResult",
        )
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


def body(run: Run, tool: ToolSpec, facts: DescriptionFacts) -> Callable[..., Any]:
    """The body of *tool* for *run*: takes the arguments its declaration in
    the run of *facts* has, by name."""

    def answer(**arguments: Any) -> Any:
        tool_context = arguments.pop("tool_context", None)
        _check_limit(tool, arguments)
        call_id = getattr(tool_context, "function_call_id", None) or uuid.uuid4().hex[:8]
        context = ToolContext(run, Caller.of_tool(tool), str(call_id))
        try:
            if tool.kind == "write" and not tool.long:
                with context.writes.writing():
                    result = tool.handler(context, **arguments)
            else:
                result = tool.handler(context, **arguments)
        except Refused:
            raise
        except Exception as exc:
            logger.exception("plugin tool %s failed", tool.name)
            raise Refused(
                RefusalCode.INTERNAL, tool=tool.name, message=f"{type(exc).__name__}: {exc}"
            ) from exc
        return reply_of(tool, result)

    declared = declaration(tool.name, facts)
    answer.__name__ = tool.name
    return signed(
        answer,
        declared.signature.replace(
            parameters=[
                *declared.signature.parameters.values(),
                context_parameter(declared.signature),
            ],
            return_annotation=inspect.Signature.empty,
        ),
    )


def bodies(run: Run, facts: DescriptionFacts) -> dict[str, Callable[..., Any]]:
    """The bodies of every plugin tool the run's job has, by name."""
    return {
        name: body(run, tool, facts) for name, (_plugin, tool) in core.tool_specs(run.spec).items()
    }


def plugin_records(facts: DescriptionFacts) -> list[dict[str, Any]]:
    """The provenance record of every plugin the run of *facts* holds."""
    out: list[dict[str, Any]] = []
    for plugin in facts.plugins:
        if plugin.ref in ENGINES:
            continue
        tools = []
        for tool in plugin.tools:
            declared = declaration(tool.name, facts)
            tools.append(
                {
                    "name": tool.name,
                    "kind": tool.kind,
                    "group": tool.group,
                    "task": tool.task,
                    "writes": list(tool.writes),
                    "long": tool.long,
                    "limits": dict(tool.limits),
                    "version": tool.version,
                    **({"atlases": list(tool.atlases)} if tool.atlases else {}),
                    **({"planes": list(tool.planes)} if tool.planes else {}),
                    **({"meta": dict(tool.meta)} if tool.meta else {}),
                    "declaration": {
                        "name": tool.name,
                        "description": declared.doc,
                        "arguments": arguments_schema(tool.name, facts),
                    },
                }
            )
        out.append({**plugin.provenance, "tools": tools})
    return out


def record_plugins(job: Job, held: list[dict[str, Any]]) -> None:
    """Write *held* to the job's ``job.json`` (and say so in the event log)
    when it differs from what the file holds; a job that writes nothing
    (a dry run) writes nothing. Never raises: provenance must not stop a run."""
    if not held or not job.persist:
        return
    try:
        from langslice.job.folder.layout import read_job_file, write_job_file

        if (read_job_file(job.layout) or {}).get(PLUGINS_KEY) == held:
            return
        write_job_file(job.layout, **{PLUGINS_KEY: held})
        job.layout.log_event(
            "plugins",
            plugins=[
                {
                    "ref": item["ref"],
                    "sources": item.get("sources"),
                    "tools": [tool["name"] for tool in item["tools"]],
                }
                for item in held
            ],
        )
    except Exception:
        logger.warning("Could not record the plugins' provenance", exc_info=True)

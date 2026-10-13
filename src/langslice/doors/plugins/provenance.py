"""What a job records of the plugins it ran with.

Each plugin tool's description is an experimental variable, so it is kept
exactly as the model sees it: per plugin the module, factory, versions, the
SHA-256 of the source files that declared its tools and the plugin API
version (:attr:`langslice.core.plugins.LoadedPlugin.provenance`), and per
tool its kind, group, task, writes, limits and its rendered declaration (the
name, description and argument schema in this job's variant). The record is
``job.json``'s ``plugins`` and the run trace's ``plugins`` line; a change of
what a job runs with is also an event in ``logs/events.jsonl``.
"""

from __future__ import annotations

import logging
from typing import Any

from langslice.doors.declarations import Variant, arguments_schema, declaration
from langslice.job.job import Job

logger = logging.getLogger(__name__)

#: ``job.json``'s field.
KEY = "plugins"


def records(variant: Variant) -> list[dict[str, Any]]:
    """The provenance record of every plugin *variant* holds."""
    out: list[dict[str, Any]] = []
    for plugin in variant.plugins:
        tools = []
        for tool in plugin.tools:
            declared = declaration(tool.name, variant)
            tools.append({
                "name": tool.name, "kind": tool.kind, "group": tool.group, "task": tool.task,
                "writes": list(tool.writes), "long": tool.long, "limits": dict(tool.limits),
                "version": tool.version,
                **({"meta": dict(tool.meta)} if tool.meta else {}),
                "declaration": {"name": tool.name, "description": declared.doc,
                                "arguments": arguments_schema(tool.name, variant)},
            })
        out.append({**plugin.provenance, "tools": tools})
    return out


def record(job: Job, held: list[dict[str, Any]]) -> None:
    """Write *held* to the job's ``job.json`` (and say so in the event log)
    when it differs from what the file holds; a job that writes nothing
    (a dry run) writes nothing. Never raises: provenance must not stop a run."""
    if not held or not job.persist:
        return
    try:
        from langslice.job.layout import read_job_file, write_job_file

        if (read_job_file(job.layout) or {}).get(KEY) == held:
            return
        write_job_file(job.layout, **{KEY: held})
        job.layout.log_event("plugins", plugins=[
            {"ref": item["ref"], "sources": item.get("sources"),
             "tools": [tool["name"] for tool in item["tools"]]} for item in held])
    except Exception:  # noqa: BLE001
        logger.warning("Could not record the plugins' provenance", exc_info=True)

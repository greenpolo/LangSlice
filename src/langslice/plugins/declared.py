"""The :class:`~langslice.core.plugins.ToolSpec` of an in-tree engine's tool.

An engine's tool is declared once, as its row of the tool table
(:data:`langslice.job.settings.tool_table.TOOLS`): kind, group, task, what it
writes, whether it is long, its limits, what it needs and the atlases and
planes it is offered on. The engine's ``plugin.py`` supplies what only it can
(the stub, the handler, the description, its version and provenance) and
:func:`engine_tool` makes the spec from the two, so the two cannot drift.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from langslice.core.plugins import PluginError, ToolResult, ToolSpec
from langslice.job.settings.tool_table import TOOLS


def engine_tool(
    name: str,
    stub: Callable[..., Any],
    handler: Callable[..., ToolResult],
    *,
    description: str | Callable[[Any], str],
    version: str,
    meta: Mapping[str, Any],
) -> ToolSpec:
    """Build the spec of the engine tool *name* from its row of the tool table.

    Parameters
    ----------
    name
        The tool's name; it must be a row of the table that names an engine.
    stub
        The function whose signature is the tool's arguments.
    handler
        The function a call runs.
    description
        The tool's description, a string or a function of the run's tools.
    version
        The tool's version.
    meta
        The tool's provenance facts.

    Returns
    -------
    ToolSpec
        The spec, its declared fields those of the row. ``task`` is the one
        task the row serves, None for a tool that serves several.

    Raises
    ------
    PluginError
        When the table has no engine tool of that name.
    """
    row = TOOLS.get(name)
    if row is None or not row.engine:
        raise PluginError(f"{name!r} is no engine tool of the tool table")
    return ToolSpec(
        name,
        stub,
        handler,
        kind=row.kind,
        group=row.group,
        task=row.tasks[0] if len(row.tasks) == 1 else None,
        writes=row.writes,
        long=row.long,
        limits=dict(row.limits),
        needs=row.needs,
        version=version,
        meta=meta,
        atlases=row.atlases,
        planes=row.planes,
        description=description,
    )

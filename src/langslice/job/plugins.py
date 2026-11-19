"""Tool plugins, the job's part: the declaration checks and the registry.

A tool plugin adds tools to a job's toolbox at run time
(``docs/developers/plugins.md``); the data types its tools are built from
are the core's. This module checks a declaration against the job's form
(:func:`check_tool`: the native tools' names, the tasks, the section
properties, the planes), turns a plugin tool into a row of the tool table
(:func:`tool_def`) and keeps the plugins loaded in this process
(:func:`loaded`). Running a plugin's code is a door's, which registers the
loader with :func:`set_loader`; nothing here imports a module by name.
"""

from __future__ import annotations

import fnmatch
import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from langslice.core.plugins import (
    NAME_PATTERN,
    PLUGIN_API_VERSION,
    SCRIPT_CALLER,
    PluginError,
    ToolSpec,
)
from langslice.core.space import PLANES
from langslice.core.tool_kinds import GROUPS
from langslice.job.settings import frozen as freeze
from langslice.job.settings.tool_table import (
    ENGINES,
    TOOLS,
    ToolDef,
    engines_on,
    positions_on,
)

TASKS: tuple[str, ...] = tuple(freeze.TASK_PROPERTIES)
ANGLES = "angles"
SIGNATURE_FAILURES = (
    AttributeError,
    NameError,
    SyntaxError,
    TypeError,
    ValueError,
)


def check_tool(tool: object, *, taken: Sequence[str] = (), engine: str = "") -> str | None:
    """One line saying why *tool* is no valid :class:`ToolSpec`, or None.
    *taken* are the names of the other plugin tools of the same job;
    *engine* is the in-tree engine being loaded, whose tools are the rows
    of the tool table that name it."""
    if not isinstance(tool, ToolSpec):
        return f"the factory returned {type(tool).__name__}, not a ToolSpec"
    name = tool.name
    where = f"tool {name!r}"
    if not isinstance(name, str) or not NAME_PATTERN.match(name):
        return f"tool name {name!r} is not an identifier (letters, digits, underscores)"
    if engine:
        row = TOOLS.get(name)
        if row is None or row.engine != engine:
            return f"{where} is no tool the tool table lists for {engine}"
    elif name in TOOLS:
        return f"{where} clashes with the native tool of that name"
    if name == SCRIPT_CALLER:
        return f"{where} is the name a script's own writes are recorded under"
    if name in taken:
        return f"{where} clashes with another plugin tool of that name"
    if tool.api_version != PLUGIN_API_VERSION:
        return (
            f"{where} was written for plugin API {tool.api_version}; "
            f"this LangSlice has {PLUGIN_API_VERSION}"
        )
    if tool.kind not in ("read", "write"):
        return f"{where}: kind is read or write; got {tool.kind!r}"
    if tool.group not in GROUPS:
        return f"{where}: group is one of {', '.join(GROUPS)}; got {tool.group!r}"
    if tool.task is not None and tool.task not in (*TASKS, ANGLES):
        return (
            f"{where}: unknown task {tool.task!r}; the tasks are {', '.join(TASKS)} (or {ANGLES!r})"
        )
    if not callable(tool.handler):
        return f"{where}: the handler is not callable"
    if not callable(tool.stub):
        return f"{where}: the stub is not callable"
    if not tool.description and not (getattr(tool.stub, "__doc__", None) or "").strip():
        return f"{where}: the stub has no docstring and the tool no description"
    try:
        inspect.signature(tool.stub, eval_str=True)
    except SIGNATURE_FAILURES as exc:
        return f"{where}: the stub's signature cannot be read ({exc})"
    unknown = [item for item in tool.writes if item not in freeze.PROPERTIES]
    if unknown:
        return (
            f"{where}: writes {unknown} are no section properties; the properties are "
            f"{', '.join(freeze.PROPERTIES)}"
        )
    if tool.kind == "read" and tool.writes:
        return f"{where}: a read tool writes nothing; declare kind='write' for writes"
    for key, value in dict(tool.limits).items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            return f"{where}: limit {key!r} is a positive whole number"
    for group in tool.needs:
        if not group or not all(isinstance(item, str) and item for item in group):
            return f"{where}: needs is a tuple of non-empty tuples of names"
    if isinstance(tool.atlases, str) or not all(
        isinstance(item, str) and item for item in tool.atlases
    ):
        return f"{where}: atlases is a tuple of non-empty name patterns"
    if isinstance(tool.planes, str) or any(item not in PLANES for item in tool.planes):
        return f"{where}: planes are among {', '.join(PLANES)}; got {tuple(tool.planes)!r}"
    return None


def offered_on(tool: ToolSpec, atlas: str, plane: str) -> bool:
    """Whether *tool* is offered on a job with atlas name *atlas* and section
    *plane* (``ToolSpec.atlases`` and ``planes``)."""
    return (
        not tool.atlases or any(fnmatch.fnmatchcase(atlas, pattern) for pattern in tool.atlases)
    ) and (not tool.planes or plane in tool.planes)


def not_offered(spec: Any) -> list[str]:
    """The names of the tools *spec*'s ``plugins`` add that its atlas or
    plane does not fit; they count as switched off, so no text names them."""
    tools: list[ToolSpec] = []
    for plugin in added(cast("object", spec)):
        tools.extend(plugin.tools)
    return [tool.name for tool in tools if not offered_on(tool, str(spec.atlas), str(spec.plane))]


def tool_def(tool: ToolSpec) -> ToolDef:
    """The core's definition of a plugin tool (what the per-spec views list)."""
    row = TOOLS.get(tool.name)
    if row is not None:
        return row
    task = tool.task

    def when(spec: Any, _task: str | None = task) -> bool:
        if not offered_on(tool, str(spec.atlas), str(spec.plane)):
            return False
        if _task == ANGLES:
            return positions_on(spec)
        return _task is None or bool(spec.has(_task))

    return ToolDef(
        tool.name,
        tool.kind,
        tool.group,
        when=when,
        long=tool.long,
        needs=tuple(tuple(group) for group in tool.needs),
        writes=tuple(tool.writes),
    )


@dataclass(frozen=True, eq=False)
class LoadedPlugin:
    """A plugin as loaded: its reference (``module:factory``), the options the
    job gave it, its tools and its provenance record (source digests, versions)."""

    ref: str
    tools: tuple[ToolSpec, ...]
    provenance: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] | None = None

    @property
    def key(self) -> str:
        return key_of(self.ref, self.config)


def key_of(ref: str, config: Mapping[str, Any] | None = None) -> str:
    """The registry key of plugin *ref* with *config*: the reference itself
    without options, else the reference and a digest of the options."""
    if not config:
        return ref
    import hashlib
    import json

    text = json.dumps(config, sort_keys=True, default=str)
    return f"{ref}#{hashlib.sha256(text.encode()).hexdigest()[:8]}"


_LOADED: dict[str, LoadedPlugin] = {}


class _Loader:
    """The one registered plugin loader (:func:`set_loader`), or None."""

    load: Callable[[str, dict[str, Any] | None], LoadedPlugin] | None = None


_LOADER = _Loader()


def set_loader(
    loader: Callable[[str, dict[str, Any] | None], LoadedPlugin] | None,
) -> None:
    """Register the function that loads a plugin by reference and options. The
    doors do (``langslice.doors.plugins``); it returns the plugin after
    :func:`register`-ing it, or raises :class:`PluginError`."""
    _LOADER.load = loader


def register(plugin: LoadedPlugin) -> LoadedPlugin:
    """Make *plugin* the one loaded under its key."""
    _LOADED[plugin.key] = plugin
    return plugin


def forget(ref: str) -> None:
    """Drop every plugin loaded under *ref* (a reload starts from the source).
    Test support: it lets a test simulate a new process; no code calls it."""
    for key in [key for key in _LOADED if key == ref or key.startswith(ref + "#")]:
        del _LOADED[key]


def loaded(refs: Sequence[str], configs: Mapping[str, Any] | None = None) -> list[LoadedPlugin]:
    """The plugins named by *refs* (with the options *configs* gives each),
    loading the ones not loaded yet (:class:`PluginError` when one cannot be,
    or when two tools share a name)."""
    out: list[LoadedPlugin] = []
    names: dict[str, str] = {}
    for ref in refs:
        config = (configs or {}).get(ref)
        plugin = _LOADED.get(key_of(ref, config))
        if plugin is None:
            load = _LOADER.load
            if load is None:
                raise PluginError(
                    f"the tool plugin {ref!r} cannot be loaded here: plugins are loaded by "
                    "langslice.doors.plugins (import langslice.plugin_api first)"
                )
            plugin = load(ref, dict(config) if config is not None else None)
        for tool in plugin.tools:
            if tool.name in names:
                raise PluginError(
                    f"tool {tool.name!r} is added by both {names[tool.name]} and {ref}"
                )
            names[tool.name] = ref
        out.append(plugin)
    return out


def added(spec: object) -> list[LoadedPlugin]:
    """Load the plugins a job's ``plugins`` name.

    Parameters
    ----------
    spec
        The job's settings.

    Returns
    -------
    list of LoadedPlugin
        The plugins, in the order named; none for a job that names none.
    """
    refs = list(getattr(spec, "plugins", None) or ())
    return loaded(refs, getattr(spec, "plugin_config", None)) if refs else []


def engines(spec: object) -> list[LoadedPlugin]:
    """Load the in-tree engines a run has a tool of.

    Parameters
    ----------
    spec
        The job's settings; None: every engine.

    Returns
    -------
    list of LoadedPlugin
        The engines, in table order, each loaded with the options the
        job's ``plugin_config`` gives it.
    """
    refs = list(ENGINES) if spec is None else engines_on(spec)
    return loaded(refs, getattr(spec, "plugin_config", None)) if refs else []


def of_spec(spec: Any) -> list[LoadedPlugin]:
    """The in-tree engines a run of *spec* has a tool of, then its plugins."""
    settings = cast("object", spec)
    return [*engines(settings), *added(settings)]


def tool_defs(spec: Any) -> dict[str, ToolDef]:
    """The :class:`ToolDef` of every tool the ``plugins`` of *spec* add."""
    return {
        tool.name: tool_def(tool) for plugin in added(cast("object", spec)) for tool in plugin.tools
    }


def tool_specs(spec: Any) -> dict[str, tuple[LoadedPlugin, ToolSpec]]:
    """Every engine and plugin tool of *spec*, with what it came from."""
    sources = engines(spec) if spec is None else of_spec(spec)
    return {tool.name: (plugin, tool) for plugin in sources for tool in plugin.tools}

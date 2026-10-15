"""Tool plugins, the core's part: the data types and the registry of loaded plugins.

A tool plugin adds tools to a job's toolbox at run time (``docs/plugins.md``).
This module holds what the lower layers must know about one: the declaration
of a tool (:class:`ToolSpec`), what its handler returns (:class:`ToolResult`),
the layer a tool leaves on a section (:class:`Layer`), the one error a tool
raises (:class:`ToolError`), the checks a tool declaration must pass
(:func:`check_tool`) and the registry the per-spec views read
(:func:`tool_defs`, :func:`loaded`). Running a plugin's code is the doors'
(``langslice.doors.plugins`` registers the loader with :func:`set_loader`);
nothing here imports a module by name.

The public import path of the types is ``langslice.plugin_api``.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from langslice.core import frozen as freeze
from langslice.core.toolset import GROUPS, TOOLS, Group, Kind, ToolDef

#: The plugin API's version. A plugin that names another is refused.
PLUGIN_API_VERSION = 1

#: The tasks a plugin tool can belong to (the tool is on only while it is).
TASKS: tuple[str, ...] = tuple(freeze.TASK_PROPERTIES)

#: A tool name: a Python identifier, so a script can call it as a method.
NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

#: The kinds of layer.
LAYER_KINDS: tuple[str, ...] = ("points", "lines", "labels", "scalar", "picture")
#: The frames each kind may have (the first is the default).
KIND_FRAMES: dict[str, tuple[str, ...]] = {
    "points": ("section_px", "atlas_um"),
    "lines": ("section_px", "atlas_um"),
    "labels": ("section_px", "oriented"),
    "scalar": ("section_px", "oriented"),
    "picture": ("none",),
}

#: Most characters of a layer name.
MAX_LAYER_NAME = 48


class ToolError(Exception):
    """Raised by a plugin handler to refuse a call: nothing is written for
    it. ``code`` is the machine-readable reason, ``details`` plain facts; the
    reply is ``{"status": "error", "error": code, "message": ..., **details}``."""

    def __init__(self, code: str, message: str = "", **details: Any) -> None:
        super().__init__(message or code)
        self.code = str(code)
        self.message = str(message or code)
        self.details = details

    def payload(self) -> dict[str, Any]:
        return {"status": "error", "error": self.code, "message": self.message, **self.details}


class PluginError(ValueError):
    """A plugin that cannot be loaded or a tool declaration that fails the
    checks: the message is one line saying why."""


@dataclass(frozen=True, eq=False)
class ToolSpec:
    """One tool a plugin adds (``docs/plugins.md``).

    ``stub`` is a function whose signature is the tool's arguments and whose
    docstring is its description, written like the stubs of
    ``langslice.doors.declarations`` (needs marks allowed); ``handler`` is
    called as ``handler(ctx, **arguments)`` with a
    :class:`langslice.doors.plugins.ToolContext` and returns a
    :class:`ToolResult`. A ``write`` tool writes section properties only
    through the context, and only those named in ``writes``.
    """

    name: str
    stub: Callable[..., Any]
    handler: Callable[..., ToolResult]
    kind: Kind
    group: Group
    #: ``position`` | ``transform`` | ``nonlinear`` | None: the tool exists only
    #: while that task is on.
    task: str | None = None
    #: The section properties the tool may write (``langslice.core.frozen.PROPERTIES``).
    writes: tuple[str, ...] = ()
    #: Computes outside the job's write lock and applies under it.
    long: bool = False
    #: The most one call takes, by what it counts (``sections``: the length of
    #: the ``sections`` argument).
    limits: Mapping[str, int] = field(default_factory=dict)
    #: What the tool needs while it is on (groups of names; as ``ToolDef.needs``).
    needs: tuple[tuple[str, ...], ...] = ()
    #: The plugin's own version, recorded in the job's provenance.
    version: str = ""
    #: The plugin API version the tool was written for.
    api_version: int = PLUGIN_API_VERSION
    #: Anything the plugin wants kept with the tool in the job's provenance
    #: (JSON values), e.g. the text of a tool description it was given.
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """What a handler returns: the reply (JSON values) and its pictures."""

    reply: dict[str, Any] = field(default_factory=dict)
    #: Pictures (PIL images, captions burned in) sent with the reply and saved
    #: in the job folder like every picture a tool shows.
    pictures: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class Layer:
    """Something a tool leaves on a section that the agent and the user can
    see: points, lines, a label or scalar raster, or a finished picture.

    Build one with :meth:`points`, :meth:`lines`, :meth:`labels`,
    :meth:`scalar` or :meth:`picture`. ``frame`` says what the coordinates
    are: ``section_px`` (pixels of the section's image file, ``[x, y]``,
    pixel centres at integers, ``(0, 0)`` the centre of the first pixel; a
    raster covers the whole file whatever its size; a raster computed on a
    ``section_image`` array is in frame ``oriented``, which :meth:`ToolContext.add_layer`
    turns back to the file's orientation) or ``atlas_um``
    (``[x, y, z]`` in BrainGlobe atlas micrometres, the atlas's axis order).
    """

    kind: str
    frame: str
    name: str = ""
    #: points: ``(N, 2)`` (``section_px``) or ``(N, 3)`` (``atlas_um``).
    xy: np.ndarray | None = None
    #: points: one text label per point; lines: one per line.
    tags: tuple[str, ...] = ()
    #: points: one number per point.
    values: tuple[float, ...] = ()
    #: lines: ``(coordinates, closed)`` each; coordinates ``(M, 2|3)``.
    paths: tuple[tuple[np.ndarray, bool], ...] = ()
    #: labels: a 2-D integer raster; scalar: a 2-D float raster.
    raster: np.ndarray | None = None
    #: scalar: the level drawn as an outline on a picture.
    level: float = 0.5
    #: picture: a PIL image.
    image: Any = None

    @classmethod
    def points(cls, points: Any, *, labels: Sequence[str] = (),
                     values: Sequence[float] = (), frame: str = "section_px",
                     name: str = "") -> Layer:
        array = np.asarray(points, dtype=np.float64)
        layer = cls("points", frame, name=name, xy=array,
                    tags=tuple(str(item) for item in labels),
                    values=tuple(float(item) for item in values))
        layer.check()
        return layer

    @classmethod
    def lines(cls, lines: Sequence[Any], *, closed: bool | Sequence[bool] = False,
                    labels: Sequence[str] = (), frame: str = "section_px",
                    name: str = "") -> Layer:
        flags = list(closed) if isinstance(closed, Sequence) and not isinstance(
            closed, (str, bytes)) else [bool(closed)] * len(lines)
        made = tuple((np.asarray(line, dtype=np.float64), bool(flag))
                     for line, flag in zip(lines, flags, strict=True))
        layer = cls("lines", frame, name=name, paths=made,
                    tags=tuple(str(item) for item in labels))
        layer.check()
        return layer

    @classmethod
    def labels(cls, raster: Any, *, frame: str, name: str = "") -> Layer:
        layer = cls("labels", frame, name=name, raster=np.asarray(raster))
        layer.check()
        return layer

    @classmethod
    def scalar(cls, raster: Any, *, frame: str, level: float = 0.5, name: str = "") -> Layer:
        layer = cls("scalar", frame, name=name,
                    raster=np.asarray(raster, dtype=np.float32), level=float(level))
        layer.check()
        return layer

    @classmethod
    def picture(cls, image: Any, *, name: str = "") -> Layer:
        layer = cls("picture", "none", name=name, image=image)
        layer.check()
        return layer

    def check(self) -> None:
        """``ToolError`` ``BAD_LAYER`` naming what is wrong."""
        if self.kind not in LAYER_KINDS:
            raise ToolError("BAD_LAYER", f"a layer's kind is one of {', '.join(LAYER_KINDS)}; "
                            f"got {self.kind!r}")
        if self.frame not in KIND_FRAMES[self.kind]:
            raise ToolError("BAD_LAYER", f"a {self.kind} layer's frame is "
                            f"{' or '.join(KIND_FRAMES[self.kind])}; got {self.frame!r}")
        if len(self.name) > MAX_LAYER_NAME:
            raise ToolError("BAD_LAYER", f"a layer's name is at most {MAX_LAYER_NAME} characters")
        width = 2 if self.frame == "section_px" else 3
        if self.kind == "points":
            if self.xy is None or self.xy.ndim != 2 or self.xy.shape[1] != width:
                raise ToolError("BAD_LAYER", f"points in {self.frame} are an (N, {width}) array")
            if not np.isfinite(self.xy).all():
                raise ToolError("BAD_LAYER", "points must be finite")
            if self.tags and len(self.tags) != len(self.xy):
                raise ToolError("BAD_LAYER", "points take one label each, or none")
            if self.values and len(self.values) != len(self.xy):
                raise ToolError("BAD_LAYER", "points take one value each, or none")
        elif self.kind == "lines":
            for coordinates, _closed in self.paths:
                if coordinates.ndim != 2 or coordinates.shape[1] != width \
                        or len(coordinates) < 2 or not np.isfinite(coordinates).all():
                    raise ToolError("BAD_LAYER", f"each line in {self.frame} is a finite "
                                    f"(M, {width}) array with at least two points")
            if self.tags and len(self.tags) != len(self.paths):
                raise ToolError("BAD_LAYER", "lines take one label each, or none")
        elif self.kind in ("labels", "scalar"):
            raster = self.raster
            if raster is None or raster.ndim != 2 or raster.size == 0:
                raise ToolError("BAD_LAYER", f"a {self.kind} layer is a 2-D raster")
            if self.kind == "labels" and not np.issubdtype(raster.dtype, np.integer):
                raise ToolError("BAD_LAYER", "a labels raster holds integers")
            if self.kind == "labels" and raster.size and int(raster.min()) < 0:
                raise ToolError("BAD_LAYER", "label ids are not negative")
        elif self.image is None:
            raise ToolError("BAD_LAYER", "a picture layer holds an image")


# --- the declaration checks -----------------------------------------------------------


def check_tool(tool: Any, *, taken: Sequence[str] = ()) -> str | None:
    """One line saying why *tool* is no valid :class:`ToolSpec`, or None.
    *taken* are the names of the other plugin tools of the same job."""
    if not isinstance(tool, ToolSpec):
        return f"the factory returned {type(tool).__name__}, not a ToolSpec"
    name = tool.name
    where = f"tool {name!r}"
    if not isinstance(name, str) or not NAME_PATTERN.match(name):
        return f"tool name {name!r} is not an identifier (letters, digits, underscores)"
    if name in TOOLS:
        return f"{where} clashes with the native tool of that name"
    if name in taken:
        return f"{where} clashes with another plugin tool of that name"
    if tool.api_version != PLUGIN_API_VERSION:
        return (f"{where} was written for plugin API {tool.api_version}; "
                f"this LangSlice has {PLUGIN_API_VERSION}")
    if tool.kind not in ("read", "write"):
        return f"{where}: kind is read or write; got {tool.kind!r}"
    if tool.group not in GROUPS:
        return f"{where}: group is one of {', '.join(GROUPS)}; got {tool.group!r}"
    if tool.task is not None and tool.task not in TASKS:
        return f"{where}: unknown task {tool.task!r}; the tasks are {', '.join(TASKS)}"
    if not callable(tool.handler):
        return f"{where}: the handler is not callable"
    if not callable(tool.stub):
        return f"{where}: the stub is not callable"
    if not (getattr(tool.stub, "__doc__", None) or "").strip():
        return f"{where}: the stub has no docstring (it is the tool's description)"
    try:
        inspect.signature(tool.stub, eval_str=True)
    except Exception as exc:  # noqa: BLE001 - an unevaluable annotation
        return f"{where}: the stub's signature cannot be read ({exc})"
    unknown = [item for item in tool.writes if item not in freeze.PROPERTIES]
    if unknown:
        return (f"{where}: writes {unknown} are no section properties; the properties are "
                f"{', '.join(freeze.PROPERTIES)}")
    if tool.kind == "read" and tool.writes:
        return f"{where}: a read tool writes nothing; declare kind='write' for writes"
    for key, value in dict(tool.limits).items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            return f"{where}: limit {key!r} is a positive whole number"
    for group in tool.needs:
        if not group or not all(isinstance(item, str) and item for item in group):
            return f"{where}: needs is a tuple of non-empty tuples of names"
    return None


def tool_def(tool: ToolSpec) -> ToolDef:
    """The core's definition of a plugin tool (what the per-spec views list)."""
    task = tool.task

    def when(spec: Any, _task: str | None = task) -> bool:
        return _task is None or bool(spec.has(_task))

    return ToolDef(tool.name, tool.kind, tool.group, when=when, long=tool.long,
                   needs=tuple(tuple(group) for group in tool.needs),
                   writes=tuple(tool.writes))


# --- the registry of loaded plugins ----------------------------------------------------


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
_LOADER: Callable[[str, dict[str, Any] | None], LoadedPlugin] | None = None


def set_loader(loader: Callable[[str, dict[str, Any] | None], LoadedPlugin] | None) -> None:
    """Register the function that loads a plugin by reference and options. The
    doors do (``langslice.doors.plugins``); it returns the plugin after
    :func:`register`-ing it, or raises :class:`PluginError`."""
    global _LOADER
    _LOADER = loader


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
            if _LOADER is None:
                raise PluginError(
                    f"the tool plugin {ref!r} cannot be loaded here: plugins are loaded by "
                    "langslice.doors.plugins (import langslice.plugin_api first)")
            plugin = _LOADER(ref, dict(config) if config is not None else None)
        for tool in plugin.tools:
            if tool.name in names:
                raise PluginError(f"tool {tool.name!r} is added by both {names[tool.name]} "
                                  f"and {ref}")
            names[tool.name] = ref
        out.append(plugin)
    return out


def of_spec(spec: Any) -> list[LoadedPlugin]:
    """The plugins a ``JobSpec`` names (none for a spec with none)."""
    refs = list(getattr(spec, "plugins", None) or ())
    return loaded(refs, getattr(spec, "plugin_config", None)) if refs else []


def tool_defs(spec: Any) -> dict[str, ToolDef]:
    """The :class:`ToolDef` of every tool the plugins of *spec* add."""
    return {tool.name: tool_def(tool) for plugin in of_spec(spec) for tool in plugin.tools}


def tool_specs(spec: Any) -> dict[str, tuple[LoadedPlugin, ToolSpec]]:
    """Every tool the plugins of *spec* add, with the plugin it came from."""
    return {tool.name: (plugin, tool) for plugin in of_spec(spec) for tool in plugin.tools}

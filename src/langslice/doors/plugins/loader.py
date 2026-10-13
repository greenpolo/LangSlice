"""Loading a tool plugin: import it, call its factory, check what it returns.

A plugin is named ``module:factory`` (:attr:`langslice.core.spec.JobSpec.plugins`).
:func:`load` imports the module, calls the factory and checks every tool it
returns (:func:`langslice.core.plugins.check_tool`, plus a name no retired tool
had); the result is a :class:`~langslice.core.plugins.LoadedPlugin` that
carries its provenance: the module and factory, the plugin's and the package's
versions, the SHA-256 of the source files of everything it declared, and the
plugin API version.

A plugin may also be handed to the library as objects (:func:`as_refs`): a
factory function, a list of :class:`~langslice.core.plugins.ToolSpec` or one.
A factory defined at module level is named by its own ``module:qualname`` and
so reloads in another process; a list of tools is registered under a name
made from their digest and reloads only in this process.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from langslice.core import plugins as core
from langslice.core.plugins import (
    PLUGIN_API_VERSION,
    LoadedPlugin,
    PluginError,
    ToolSpec,
    check_tool,
)
from langslice.ops.registry import RETIRED

logger = logging.getLogger(__name__)

#: Prefix of the reference an in-process list of tools is registered under.
INLINE_PREFIX = "inline:"


def _split(ref: str) -> tuple[str, str]:
    module, colon, factory = ref.partition(":")
    if not colon or not module.strip() or not factory.strip():
        raise PluginError(f"plugin {ref!r} is not module:factory")
    return module.strip(), factory.strip()


def _digest(files: Sequence[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in files:
        try:
            out[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            out[str(path)] = ""
    return out


def _source_files(objects: Sequence[Any]) -> list[Path]:
    seen: dict[str, Path] = {}
    for item in objects:
        try:
            raw = inspect.getsourcefile(item)
        except (TypeError, OSError):
            raw = None
        if raw:
            path = Path(raw).resolve()
            seen.setdefault(str(path), path)
    return sorted(seen.values())


def _package_version(module_name: str) -> str:
    head = module_name.split(".")[0]
    try:
        return importlib.metadata.version(head)
    except importlib.metadata.PackageNotFoundError:
        return ""


def build(ref: str, factory: Callable[..., Any], *, module_name: str,
          config: dict[str, Any] | None = None) -> LoadedPlugin:
    """Call *factory* (with *config* when the job has one for this plugin),
    check its tools and make the :class:`LoadedPlugin` (not yet registered)."""
    try:
        made = factory(config) if config is not None else factory()
    except Exception as exc:  # noqa: BLE001 - a plugin's own failure, said in one line
        raise PluginError(f"plugin {ref}: the factory failed: {type(exc).__name__}: {exc}"
                          ) from exc
    if isinstance(made, ToolSpec):
        made = [made]
    if not isinstance(made, (list, tuple)):
        raise PluginError(f"plugin {ref}: the factory returns a list of ToolSpec; got "
                          f"{type(made).__name__}")
    taken: list[str] = []
    for tool in made:
        reason = check_tool(tool, taken=taken)
        if reason is None and tool.name in RETIRED:
            reason = f"tool {tool.name!r} is the name of a retired tool"
        if reason is not None:
            raise PluginError(f"plugin {ref}: {reason}")
        taken.append(tool.name)
    module = sys.modules.get(module_name)
    declared = getattr(module, "PLUGIN_API_VERSION", PLUGIN_API_VERSION)
    if declared != PLUGIN_API_VERSION:
        raise PluginError(f"plugin {ref} was written for plugin API {declared}; this "
                          f"LangSlice has {PLUGIN_API_VERSION}")
    sources = _source_files([factory, *(tool.handler for tool in made),
                             *(tool.stub for tool in made)])
    provenance = {
        "ref": ref,
        "module": module_name,
        "factory": getattr(factory, "__qualname__", getattr(factory, "__name__", "")),
        "version": next((tool.version for tool in made if tool.version), ""),
        "package_version": _package_version(module_name),
        "api_version": PLUGIN_API_VERSION,
        "sources": _digest(sources),
        "tools": [tool.name for tool in made],
    }
    if config is not None:
        provenance["config"] = config
    return LoadedPlugin(ref=ref, tools=tuple(made), provenance=provenance, config=config)


def load(ref: str, config: dict[str, Any] | None = None) -> LoadedPlugin:
    """Load the plugin *ref* (``module:factory``) and register it. A plugin
    already loaded under the same reference is replaced. *config* is what the
    job's ``plugin_config`` holds for the reference, passed to the factory.
    :class:`~langslice.core.plugins.PluginError` (a ``ValueError``), one line,
    when it cannot be loaded or declares an invalid tool."""
    if ref.startswith(INLINE_PREFIX):
        raise PluginError(f"plugin {ref} was passed to the library as objects in another "
                          "process; pass a factory function or module:factory to reuse it")
    module_name, factory_name = _split(ref)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - import errors of any kind
        raise PluginError(f"plugin {ref}: cannot import {module_name!r}: "
                          f"{type(exc).__name__}: {exc}") from exc
    factory: Any = module
    for part in factory_name.split("."):
        factory = getattr(factory, part, None)
        if factory is None:
            raise PluginError(f"plugin {ref}: {module_name} has no {factory_name!r}")
    if not callable(factory):
        raise PluginError(f"plugin {ref}: {factory_name!r} is not callable")
    return core.register(build(ref, factory, module_name=module_name, config=config))


def as_refs(plugins: Any) -> list[str]:
    """The references of what the library was handed: strings stay, a factory
    function is registered and named ``module:qualname``, ToolSpecs are
    registered under ``inline:<digest>``. The plugins are loaded (and so
    checked) here."""
    items = [plugins] if isinstance(plugins, (str, ToolSpec)) or callable(plugins) \
        else list(plugins or ())
    refs: list[str] = []
    loose: list[ToolSpec] = []
    for item in items:
        if isinstance(item, ToolSpec):
            loose.append(item)
        elif isinstance(item, str):
            refs.append(item)
        elif callable(item):
            module_name = getattr(item, "__module__", "")
            name = getattr(item, "__qualname__", "")
            if not module_name or not name or "<" in name:
                raise PluginError("a plugin factory passed as an object is a function "
                                  "defined at module level (a lambda or local function has "
                                  "no name to reload it by)")
            ref = f"{module_name}:{name}"
            core.register(build(ref, item, module_name=module_name))
            refs.append(ref)
        else:
            raise PluginError(f"a plugin is module:factory, a factory function or ToolSpecs; "
                              f"got {type(item).__name__}")
    if loose:
        digest = hashlib.sha256(",".join(sorted(tool.name for tool in loose)).encode()
                                ).hexdigest()[:10]
        ref = f"{INLINE_PREFIX}{digest}"
        core.register(build(ref, lambda: list(loose), module_name=INLINE_PREFIX))
        refs.append(ref)
    return list(dict.fromkeys(refs))


def install() -> None:
    """Register :func:`load` as the loader the core calls for a plugin not
    loaded yet (done once, when this package is imported)."""
    core.set_loader(load)

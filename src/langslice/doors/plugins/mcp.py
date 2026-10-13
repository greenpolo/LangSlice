"""The generic MCP adapter plugin: LangSlice as a client of a locally configured MCP server.

``langslice.doors.plugins.mcp:tools`` is a tool plugin like any other
(``JobSpec.plugins``) whose options (``JobSpec.plugin_config``) name a server
the user runs on this machine and the tools of it the job may use. It
connects to the server (a command LangSlice starts, or a URL), lists its
tools, and offers each allow-listed one as a plugin tool named
``<server>__<tool>`` (letters, digits and underscores; other characters of the
remote name become ``_``): the declaration is the server's description and
input schema, and the call is the server's tool.

The options (one server, or ``{"servers": [...]}``)::

    {"server": "napari",                      # prefix of the tool names
     "command": ["python", "-m", "my_server"], # a local program, or
     "url": "http://127.0.0.1:8000/mcp",      # a streamable-HTTP endpoint
     "env": {}, "cwd": null, "timeout_s": 120,
     "tools": {                                # the allow-list: nothing else is offered
       "add_points": {
         "kind": "write",                      # read | write (default: write when
                                               # results writes, else read)
         "group": "Common", "task": null, "long": false, "limits": {},
         "inputs": {"image_path": {"send": "section_image", "section": "section"}},
         "fixed": {"viewer": "main"},
         "results": [{"to": "layer", "section": "section", "from": "points",
                      "kind": "points", "frame": "section_px", "name": "napari_points"}]
       }}}

What the remote tool can reach is only what this mapping gives it. ``inputs``
fills arguments LangSlice supplies instead of the model: ``section_image``
(a PNG of the named section written under the plugin's folder in the job, the
argument being its path, or the image inline as base64 with ``"inline":
true``) and ``placement`` (the section's position, angles, orientation and
transform as an object). ``results`` maps the server's answer (its structured
content, or its text read as JSON) into the job through the plugin context,
and nothing else is written: ``layer`` (points or lines left as a layer),
``transform`` (six numbers as the section's transform), ``position`` (a
position in millimetres). Text, structured data and images the server returns
are in the reply. The server's descriptions are text from outside the
project: they are shown to the model as given (``[[`` made ``[ [`` so they
cannot act as a rendering mark) and recorded verbatim in the job's
provenance; no description, schema or answer changes the job's settings, the
tools offered, or a file outside the plugin's folder.
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import concurrent.futures
import hashlib
import inspect
import io
import json
import logging
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from langslice.core.plugins import (
    NAME_PATTERN,
    Layer,
    PluginError,
    ToolError,
    ToolResult,
    ToolSpec,
)

logger = logging.getLogger(__name__)

#: Seconds to start a server and list its tools.
CONNECT_TIMEOUT_S = 60.0
#: Seconds one call may take unless the options say.
DEFAULT_TIMEOUT_S = 120.0
#: Characters of a server's text answer kept in a reply.
MAX_TEXT = 8000
#: Most points / vertices one mapped layer takes.
MAX_LAYER_POINTS = 200_000

_UNSAFE = re.compile(r"[^A-Za-z0-9_]")


# --- the connection --------------------------------------------------------------------


class Connection:
    """One MCP server session kept open on its own thread and event loop
    (the client's transport must be opened and closed by one task)."""

    def __init__(self, options: Mapping[str, Any]) -> None:
        self.options = dict(options)
        self.tools: list[Any] = []
        self._loop = asyncio.new_event_loop()
        self._queue: asyncio.Queue[Any] | None = None
        self._ready: concurrent.futures.Future[None] = concurrent.futures.Future()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True,
                                        name=f"mcp-{options.get('server', 'server')}")
        self._thread.start()
        self._done = asyncio.run_coroutine_threadsafe(self._main(), self._loop)
        try:
            self._ready.result(timeout=CONNECT_TIMEOUT_S)
        except BaseException:
            self.close()
            raise

    def _transport(self) -> Any:
        options = self.options
        if options.get("url"):
            from mcp.client.streamable_http import streamablehttp_client

            return streamablehttp_client(str(options["url"]))
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client

        command = [str(part) for part in options["command"]]
        return stdio_client(StdioServerParameters(
            command=command[0], args=command[1:],
            env={str(k): str(v) for k, v in (options.get("env") or {}).items()} or None,
            cwd=options.get("cwd")))

    async def _main(self) -> None:
        from mcp import ClientSession

        try:
            async with self._transport() as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    cursor: str | None = None
                    while True:
                        page = await session.list_tools(cursor)
                        self.tools += list(page.tools)
                        cursor = page.nextCursor
                        if not cursor:
                            break
                    self._queue = asyncio.Queue()
                    self._ready.set_result(None)
                    while True:
                        item = await self._queue.get()
                        if item is None:
                            break
                        name, arguments, timeout, future = item
                        try:
                            future.set_result(await asyncio.wait_for(
                                session.call_tool(name, arguments), timeout))
                        except BaseException as exc:  # noqa: BLE001 - handed to the caller
                            future.set_exception(exc)
        except BaseException as exc:  # noqa: BLE001
            if not self._ready.done():
                self._ready.set_exception(exc)
            else:
                logger.warning("MCP server %s ended: %s", self.options.get("server"), exc)
        finally:
            self._queue = None

    def call(self, name: str, arguments: Mapping[str, Any], timeout: float) -> Any:
        """The server's answer to tool *name* (``CallToolResult``)."""
        queue = self._queue
        if queue is None:
            raise ToolError("MCP_UNAVAILABLE", f"the {self.options.get('server')} MCP server "
                            "is not running")
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        self._loop.call_soon_threadsafe(queue.put_nowait, (name, dict(arguments), timeout,
                                                           future))
        try:
            return future.result(timeout=timeout + 5)
        except (concurrent.futures.TimeoutError, TimeoutError):
            raise ToolError("MCP_TIMEOUT", f"{name} did not answer within {timeout:g} s") \
                from None

    def close(self) -> None:
        queue = self._queue
        if queue is not None:
            self._loop.call_soon_threadsafe(queue.put_nowait, None)
        try:
            self._done.result(timeout=10)
        except BaseException:  # noqa: BLE001 - closing never raises
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


_CONNECTIONS: dict[str, Connection] = {}


def connection(options: Mapping[str, Any]) -> Connection:
    """The open connection for a server's options, made on first use and shared
    by every job of this process that names the same server."""
    key = hashlib.sha256(json.dumps(
        {k: options.get(k) for k in ("command", "url", "env", "cwd")}, sort_keys=True,
        default=str).encode()).hexdigest()
    held = _CONNECTIONS.get(key)
    if held is None or held._queue is None:  # noqa: SLF001 - a dead one is replaced
        held = _CONNECTIONS[key] = Connection(options)
    return held


def close_all() -> None:
    """Close every open server connection (at exit)."""
    for held in list(_CONNECTIONS.values()):
        held.close()
    _CONNECTIONS.clear()


atexit.register(close_all)


# --- the options ----------------------------------------------------------------------


def _servers(config: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(config, Mapping):
        raise PluginError("the MCP adapter needs options: a server (command or url) and the "
                          "tools to allow (plugin_config)")
    servers = config.get("servers")
    items = [dict(item) for item in servers] if isinstance(servers, Sequence) \
        and not isinstance(servers, (str, bytes)) else [dict(config)]
    for item in items:
        name = item.get("server")
        if not isinstance(name, str) or not NAME_PATTERN.match(name):
            raise PluginError("each MCP server needs a 'server' name that is an identifier")
        if bool(item.get("command")) == bool(item.get("url")):
            raise PluginError(f"MCP server {name}: give a command or a url, one of them")
        if item.get("command") and (not isinstance(item["command"], list)
                                    or not all(isinstance(p, str) for p in item["command"])):
            raise PluginError(f"MCP server {name}: command is a list of strings")
        allowed = item.get("tools")
        if not isinstance(allowed, Mapping) or not allowed:
            raise PluginError(f"MCP server {name}: 'tools' is the allow-list (a non-empty "
                              "mapping of tool name to its options); nothing is offered "
                              "without it")
    return items


def _annotation(schema: Mapping[str, Any]) -> Any:
    enum = schema.get("enum")
    if isinstance(enum, list) and enum and all(isinstance(v, (str, int)) for v in enum):
        return Literal[tuple(enum)]  # type: ignore[valid-type]
    kind = schema.get("type")
    if kind == "array":
        inner = schema.get("items")
        return list[_annotation(inner)] if isinstance(inner, Mapping) else list[str]  # type: ignore[misc]
    scalar: dict[str, Any] = {"string": str, "integer": int, "number": float, "boolean": bool,
                              "object": dict[str, Any]}
    return scalar.get(kind if isinstance(kind, str) else "", str)


def _doc(description: str, properties: Mapping[str, Any], shown: Sequence[str],
         fallback: str) -> str:
    """The stub's docstring: the server's description, then its arguments'
    descriptions (all verbatim but for ``[[``)."""
    text = (description or "").strip() or fallback
    lines = [text.replace("[[", "[ [")]
    entries = []
    for name in shown:
        said = str((properties.get(name) or {}).get("description") or "").strip()
        if said:
            entries.append(f"    {name}: " + said.replace("[[", "[ [").replace("\n", "\n        "))
    if entries:
        lines += ["", "Args:", *entries]
    return "\n".join(lines)


def section_arguments(options: Mapping[str, Any]) -> list[str]:
    """The names of the call arguments that name a section for the mapping
    (``inputs`` and ``results`` rules), in order of first use."""
    names = [str(rule.get("section") or "section")
             for rule in [*dict(options.get("inputs") or {}).values(),
                          *(options.get("results") or [])]]
    return list(dict.fromkeys(names))


def make_stub(name: str, remote: Any, hidden: Sequence[str], fixed: Sequence[str],
              server: str, sections: Sequence[str] = ()) -> Callable[..., Any]:
    """A stub (signature and description) from the server's tool *remote*,
    without the arguments LangSlice fills itself (*hidden*, *fixed*) and with
    the arguments the mapping needs that the server has none of (*sections*:
    the filename of the section a mapped input or result is about)."""
    schema = remote.inputSchema or {}
    properties = dict(schema.get("properties") or {})
    required = set(schema.get("required") or [])
    skipped = set(hidden) | set(fixed)
    shown = [prop for prop in properties if prop not in skipped]
    for prop in shown:
        if not NAME_PATTERN.match(prop):
            raise PluginError(f"{name}: the server's argument {prop!r} is not an identifier")
    for extra in sections:
        if extra not in properties:
            properties[extra] = {"type": "string",
                                 "description": "Filename of the section the call is about."}
            required.add(extra)
            shown.insert(0, extra)
    parameters: list[inspect.Parameter] = []
    annotations: dict[str, Any] = {}
    for prop in sorted(shown, key=lambda p: (p not in required, shown.index(p))):
        spec = properties[prop]
        annotation = _annotation(spec)
        if prop in required:
            parameter = inspect.Parameter(prop, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                          annotation=annotation)
        else:
            default = spec.get("default")
            parameter = inspect.Parameter(
                prop, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default if default is not None else _empty(annotation),
                annotation=annotation)
        parameters.append(parameter)
        annotations[prop] = annotation
    annotations["return"] = dict[str, Any]

    def stub(**_arguments: Any) -> dict[str, Any]:
        ...

    stub.__name__ = stub.__qualname__ = name
    stub.__doc__ = _doc(str(remote.description or ""), properties, shown,
                        f"{remote.name} (a tool of the {server} MCP server)")
    stub.__signature__ = inspect.Signature(parameters, return_annotation=dict[str, Any])  # type: ignore[attr-defined]
    stub.__annotations__ = annotations
    return stub


def _empty(annotation: Any) -> Any:
    origin = getattr(annotation, "__origin__", None)
    if origin is list or annotation is list:
        return []
    if origin is dict or annotation is dict:
        return {}
    if annotation is str:
        return ""
    if annotation is bool:
        return False
    if annotation in (int, float):
        return 0
    return ""


# --- the call ---------------------------------------------------------------------------


def _pick(data: Any, path: str) -> Any:
    """The value at the dotted *path* of nested mappings and lists (``a.b.0``)."""
    for part in path.split("."):
        if isinstance(data, Mapping):
            data = data.get(part)
        elif isinstance(data, list) and part.lstrip("-").isdigit():
            data = data[int(part)] if -len(data) <= int(part) < len(data) else None
        else:
            return None
        if data is None:
            return None
    return data


def _answer(result: Any) -> tuple[str, Any, list[Image.Image]]:
    """``(text, data, images)`` of a ``CallToolResult``: its text blocks joined,
    its structured content (else the text read as JSON, else None), its
    image blocks."""
    texts: list[str] = []
    images: list[Image.Image] = []
    for block in result.content or []:
        kind = getattr(block, "type", "")
        if kind == "text":
            texts.append(str(block.text))
        elif kind == "image":
            try:
                images.append(Image.open(io.BytesIO(base64.b64decode(block.data))).copy())
            except Exception:  # noqa: BLE001 - an undecodable image is left out
                logger.warning("an MCP image block could not be decoded", exc_info=True)
    text = "\n".join(texts)
    data = result.structuredContent
    if data is None and text:
        try:
            data = json.loads(text)
        except ValueError:
            data = None
    return text, data, images


def _fill(ctx: Any, rules: Mapping[str, Mapping[str, Any]], arguments: Mapping[str, Any],
          ) -> dict[str, Any]:
    """The remote arguments LangSlice supplies from the job (*rules*)."""
    out: dict[str, Any] = {}
    for argument, rule in rules.items():
        send = rule.get("send")
        section = arguments.get(str(rule.get("section") or "section"))
        if section in (None, ""):
            raise ToolError("BAD_ARGS", f"{rule.get('section') or 'section'} names the section "
                            f"whose {send} is sent")
        if send == "section_image":
            image = ctx.section_image(section, level=rule.get("level", "high"))
            if rule.get("inline"):
                buffer = io.BytesIO()
                Image.fromarray(image).save(buffer, format="PNG")
                out[argument] = base64.b64encode(buffer.getvalue()).decode("ascii")
            else:
                folder = ctx.work_dir() / "inputs"
                folder.mkdir(parents=True, exist_ok=True)
                view = ctx.section(section)
                path = folder / (re.sub(r"[^A-Za-z0-9._-]", "_", Path(view.id).stem) + ".png")
                Image.fromarray(image).save(path)
                out[argument] = str(path)
        elif send == "placement":
            view = ctx.section(section)
            out[argument] = {
                "id": view.id, "position_mm": view.position_mm,
                "pitch_deg": view.angles[0], "yaw_deg": view.angles[1], "flip": view.flip,
                "rotation_deg": view.rotation_deg,
                "affine": None if view.affine is None else list(view.affine.matrix or ()),
                "file_size": list(view.file_size), "pixel_size_um": view.pixel_size_um}
        else:
            raise ToolError("BAD_CONFIG", f"unknown input rule {send!r}")
    return out


def _layer(rule: Mapping[str, Any], value: Any, default_name: str) -> Layer:
    kind = str(rule.get("kind", "points"))
    frame = str(rule.get("frame", "section_px"))
    name = str(rule.get("name") or default_name)
    try:
        if kind == "points":
            points = np.asarray(value, dtype=np.float64)
            if points.size > MAX_LAYER_POINTS * 3:
                raise ToolError("BAD_LAYER", "too many points")
            return Layer.points(points, frame=frame, name=name)
        if kind == "lines":
            if sum(len(line) for line in value) > MAX_LAYER_POINTS:
                raise ToolError("BAD_LAYER", "too many vertices")
            return Layer.lines(value, closed=bool(rule.get("closed", False)), frame=frame,
                               name=name)
    except (TypeError, ValueError) as exc:
        raise ToolError("BAD_LAYER", f"the server's answer is no {kind} layer: {exc}") from exc
    raise ToolError("BAD_CONFIG", f"a mapped layer is points or lines; got {kind!r}")


def _apply(ctx: Any, rules: Sequence[Mapping[str, Any]], arguments: Mapping[str, Any],
           data: Any, tool_name: str) -> list[dict[str, Any]]:
    """Map the server's answer into the job through the context, rule by rule."""
    applied: list[dict[str, Any]] = []
    for rule in rules:
        target = rule.get("to")
        value = _pick(data, str(rule.get("from") or "")) if rule.get("from") else data
        section = arguments.get(str(rule.get("section") or "section"))
        if value is None or section in (None, ""):
            applied.append({"to": target, "status": "skipped",
                            "reason": "the answer has no " + str(rule.get("from"))})
            continue
        if target == "layer":
            ref = ctx.add_layer(section, _layer(rule, value, tool_name))
            applied.append({"to": "layer", "status": "ok", "section": ref.section,
                            "name": ref.name})
        elif target == "transform":
            done = ctx.set_transforms([{"id": section, "params": value}])
            applied.append({"to": "transform", **done.reply()})
        elif target == "position":
            done = ctx.set_positions([{"id": section, "position_mm": float(value)}])
            applied.append({"to": "position", **done.reply()})
        else:
            raise ToolError("BAD_CONFIG", f"unknown result rule {target!r}")
    return applied


def handler_for(server: Mapping[str, Any], remote_name: str, options: Mapping[str, Any],
                remote_properties: Sequence[str]) -> Callable[..., ToolResult]:
    inputs = dict(options.get("inputs") or {})
    fixed = dict(options.get("fixed") or {})
    rules = [dict(rule) for rule in options.get("results") or []]
    timeout = float(server.get("timeout_s") or DEFAULT_TIMEOUT_S)

    def handler(ctx: Any, **arguments: Any) -> ToolResult:
        remote_arguments = {key: value for key, value in arguments.items()
                            if key in remote_properties and key not in inputs
                            and value is not None}
        remote_arguments.update(_fill(ctx, inputs, arguments))
        remote_arguments.update(fixed)
        answer = connection(server).call(remote_name, remote_arguments, timeout)
        text, data, images = _answer(answer)
        if getattr(answer, "isError", False):
            raise ToolError("MCP_TOOL_ERROR", (text or "the server reported an error")[:MAX_TEXT],
                            server=server["server"], tool=remote_name)
        reply: dict[str, Any] = {"server": server["server"], "tool": remote_name}
        if text:
            reply["text"] = text[:MAX_TEXT] + ("..." if len(text) > MAX_TEXT else "")
        if data is not None and len(json.dumps(data, default=str)) <= MAX_TEXT:
            reply["data"] = data
        if rules:
            reply["applied"] = _apply(ctx, rules, arguments, data, remote_name)
        return ToolResult(reply, list(images))

    return handler


def tools(config: Mapping[str, Any] | None = None) -> list[ToolSpec]:
    """The allow-listed tools of the configured MCP server(s), as plugin tools
    (see the module text). Connects to each server now."""
    out: list[ToolSpec] = []
    for server in _servers(config):
        try:
            held = connection(server)
        except Exception as exc:  # noqa: BLE001 - a server that does not start
            raise PluginError(f"MCP server {server['server']} could not be started: "
                              f"{type(exc).__name__}: {exc}") from exc
        by_name = {tool.name: tool for tool in held.tools}
        for remote_name, options in dict(server["tools"]).items():
            options = dict(options or {})
            remote = by_name.get(remote_name)
            if remote is None:
                raise PluginError(f"MCP server {server['server']} has no tool {remote_name!r} "
                                  f"(it lists: {', '.join(sorted(by_name)) or 'none'})")
            name = f"{server['server']}__{_UNSAFE.sub('_', remote_name)}"
            results = [dict(rule) for rule in options.get("results") or []]
            writes = tuple(dict.fromkeys(
                {"transform": "affine", "position": "position"}[rule["to"]]
                for rule in results if rule.get("to") in ("transform", "position")))
            kind = str(options.get("kind") or ("write" if results else "read"))
            if kind == "read" and results:
                raise PluginError(f"{name}: a read tool maps no results into the job")
            hidden = list(dict(options.get("inputs") or {}))
            fixed = list(dict(options.get("fixed") or {}))
            stub = make_stub(name, remote, hidden, fixed, server["server"],
                             section_arguments(options))
            out.append(ToolSpec(
                name=name, stub=stub,
                handler=handler_for(server, remote_name, options,
                                    list((remote.inputSchema or {}).get("properties") or {})),
                kind=kind,  # type: ignore[arg-type]
                group=options.get("group", "Common"), task=options.get("task"),
                writes=writes, long=bool(options.get("long", False)),
                limits=dict(options.get("limits") or {}),
                needs=tuple(tuple(group) for group in options.get("needs") or ()),
                version=str(options.get("version") or ""),
                meta={"server": server["server"], "remote_tool": remote_name,
                      "remote_description": str(remote.description or ""),
                      "remote_input_schema": dict(remote.inputSchema or {}),
                      "mapping": {key: options.get(key) for key in ("inputs", "fixed", "results")
                                  if options.get(key)}}))
    return out

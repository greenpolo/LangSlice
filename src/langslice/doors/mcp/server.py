"""LangSlice tools for MCP hosts, with job folders, saved ABBA jobs and paged pictures.

The host owns its conversation; LangSlice owns validated settings, tool gates,
checkpoints and optional loopback-only ABBA updates. No model is called here.
"""

from __future__ import annotations

import base64
import inspect
import json
import logging
import os
import sys
import threading
from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from functools import partial
from io import TextIOWrapper
from pathlib import Path
from typing import Any

import anyio
from anyio import to_thread
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.stdio import stdio_server
from mcp.types import ContentBlock, ImageContent, TextContent, ToolAnnotations
from PIL import Image

from langslice.agent.engine import EngineContext, build_context
from langslice.core.jpeg import encode_jpeg
from langslice.core.opening import CLAUDE_IMAGE_LIMIT, CLAUDE_MAX_VIEW_EDGE, opening_items
from langslice.core.spec import JobSpec
from langslice.core.state import StackState
from langslice.doors.api.abba_worker import (
    PreparedLinear,
    checkpoint_callback,
    prepare_linear,
    public_event,
)
from langslice.doors.api.saved_jobs import load_job
from langslice.doors.card import write_card
from langslice.doors.jobs import (
    NoJob,
    close_job,
    find,
    image_model_off,
    provider_connected,
    read_spec,
)
from langslice.doors.mcp.host_channel import HostChannel
from langslice.doors.statement import job_statement, opening_for_mcp, read_notes
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.doors.tools.reply import (
    REPLY_BYTES,
    Item,
    fit_reply,
    paged,
    shrunk_note,
    strip_bytes,
)
from langslice.doors.tools.toolbox import ToolBox, build_tools
from langslice.doors.trace import TRACE_DIR_ENV, HostTrace
from langslice.job.job import Job
from langslice.ops.registry import VERBS, retired_payload

logger = logging.getLogger(__name__)

SERVER_NAME = "langslice"

#: Tools that only look (the registry's read verbs; the door's own
#: ``show_stack`` is marked read-only where it is declared). Hosts may use the
#: hint to skip a confirmation.
READ_ONLY_TOOLS = frozenset(name for name, verb in VERBS.items() if verb.kind == "read")

INSTRUCTIONS = (
    "LangSlice places histology sections in a brain atlas. Call `start_job` "
    "first: it opens the stack and answers with the job description. Read every "
    "show_stack page before writing. Work only through the LangSlice tools and "
    "finish with `submit`."
)


class EventRelay:
    """The session's tool events (``tool_start`` / ``tool_end``, the
    ``show_stack`` pages as ``seed``), forwarded to ``target`` when a host
    listens (the ABBA channel of a saved ABBA job), dropped otherwise. The
    toolbox emits them (:func:`langslice.doors.tools.toolbox.build_tools`'s
    ``on_event``); they carry saved view paths, never image bytes."""

    def __init__(self) -> None:
        self.target: Callable[[dict[str, Any]], None] | None = None

    def __call__(self, event: dict[str, Any]) -> None:
        target = self.target
        if target is not None:
            target(event)


@dataclass
class Session:
    """One open job as this door holds it.

    ``job`` is the core :class:`~langslice.job.job.Job` (state, spec,
    undo/redo, checkpoint, submit gates), the one ``langslice.agent.engine.run``
    opens; ``box`` is the tools over it, whose gates and delivery bookkeeping
    this door shares with the ADK agent's. The rest is the door's own: the
    saved-job record, the host channel and its update callback, the opening
    pages and the trace.
    """

    job: Job
    ctx: EngineContext
    box: ToolBox
    trace: HostTrace | None
    job_id: str = ""
    job_dir: Path | None = None
    notes: str = ""
    #: Sends the host (ABBA) its live update after every checkpoint.
    host_update: Callable[[StackState], None] | None = None
    prepared: PreparedLinear | None = None
    channel: HostChannel | None = None
    pages: list[list[ContentBlock]] = field(default_factory=list)
    lock: Any = field(default_factory=threading.RLock, repr=False)
    #: The job's nonlinear task names an image model this door cannot reach
    #: (:func:`langslice.doors.jobs.image_model_off`): its tools and statement
    #: are a run without one.
    image_model_off: bool = False
    #: Tool events, forwarded to the host channel of a saved ABBA job.
    events: EventRelay = field(default_factory=EventRelay)

    @property
    def spec(self) -> JobSpec:
        return self.job.spec

    @property
    def state(self) -> StackState:
        return self.job.state

    def close(self) -> None:
        """Let the job go: its host channel closed, its running image-model
        calls settled and recorded, its pictures written
        (:func:`langslice.doors.jobs.close_job`, as every door ends a job).
        When another job replaces it and when the server stops."""
        if self.channel is not None:
            self.channel.close()
        try:
            close_job(self.job)
        except Exception:
            logger.warning("Could not finish the job in %s", self.job.folder, exc_info=True)


# --- content conversion ----------------------------------------------------


def image_block(image: Image.Image | bytes) -> ImageContent:
    """One core picture (or its JPEG bytes) as an MCP image block (the doors'
    JPEG encoding)."""
    data = image if isinstance(image, bytes) else encode_jpeg(image)
    return ImageContent(type="image", data=base64.b64encode(data).decode("ascii"),
                        mimeType="image/jpeg")


def blocks_of(items: list[Item]) -> list[ContentBlock]:
    """Texts and JPEG bytes (:data:`langslice.doors.tools.reply.Item`) as MCP blocks."""
    return [TextContent(type="text", text=item) if isinstance(item, str) else image_block(item)
            for item in items]


def result_blocks(result: Any) -> list[ContentBlock]:
    """A toolbox result as MCP content: its JSON, then its media in order.

    The tools return plain pictures (PIL images, captions burned in) and
    lines of text under ``TOOL_MEDIA_PARTS_KEY``; each picture becomes an
    image block in the doors' JPEG encoding (:func:`image_block`), each
    non-empty text a text block. ``images_attached`` counts both. The whole
    reply stays within the host's reply budget
    (:func:`langslice.doors.tools.reply.fit_reply`): past it every picture is
    shrunk together and a last text says so and how to get full-size ones.
    """
    media: list[Any] = []
    if isinstance(result, dict):
        body = dict(result)
        listed = body.pop(TOOL_MEDIA_PARTS_KEY, None)
        if isinstance(listed, list):
            media = [item for item in listed if isinstance(item, (str, Image.Image))]
            body["images_attached"] = len(media)
        elif listed is not None:
            # Already text (e.g. a note that the pictures were not produced).
            body[TOOL_MEDIA_PARTS_KEY] = listed
    else:
        body = result
    items: list[Item] = [json.dumps(body, default=str)]
    items += [item if isinstance(item, str) else encode_jpeg(item)
              for item in media if not isinstance(item, str) or item]
    fitted, shrunk = fit_reply(items)
    if shrunk is not None:
        fitted.append(shrunk_note(sum(isinstance(item, bytes) for item in fitted), shrunk))
    return blocks_of(fitted)


# --- trace -----------------------------------------------------------------


def describe_blocks(blocks: list[ContentBlock]) -> list[dict[str, Any]]:
    described: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, ImageContent):
            described.append({
                "image": block.mimeType,
                "bytes": len(base64.b64decode(block.data)),
            })
        elif isinstance(block, TextContent):
            described.append({"text": block.text})
    return described


# --- the job ---------------------------------------------------------------


def open_job(
    spec: JobSpec,
    atlas_loader: Callable[[str], Any] | None = None,
    folder: Path | None = None,
) -> Session:
    """Open *spec*'s folder the way the engine does, without a model: its
    job folder next to the images (``<images>/langslice``), or *folder*, a
    saved job's (the same place, as its index names it). The nonlinear
    task's image-model tool is offered only when that model is connected
    (:func:`langslice.doors.jobs.image_model_off`)."""
    off = image_model_off(spec, provider_connected(spec))
    ctx = build_context(spec, atlas_loader=atlas_loader, job_folder=folder)
    job = Job.open(spec, ctx, folder=ctx.job_folder, results_path=ctx.results_path)
    write_card(job.layout)
    trace_dir = os.environ.get(TRACE_DIR_ENV)
    trace = HostTrace(trace_dir, ctx.image_folder) if trace_dir else None
    # The host is Claude: its pictures are capped at Claude's largest image.
    events = EventRelay()
    box = build_tools(job.state, ctx, spec, job=job, max_view_edge=CLAUDE_MAX_VIEW_EDGE,
                      image_model_connected=not off, on_event=events, door="mcp")
    return Session(job, ctx, box, trace, image_model_off=off, events=events)


#: The serialized bytes of one ``show_stack`` page, JSON and base64 included
#: (every MCP reply's budget, :data:`langslice.doors.tools.reply.REPLY_BYTES`).
PAGE_BYTES = REPLY_BYTES


def opening_pages(session: Session) -> list[list[ContentBlock]]:
    """The opening strips (:mod:`langslice.core.opening`) at Claude's image
    size, each strip composed within a page's byte budget (fewer sections per
    strip, never a shrunk strip: :func:`langslice.doors.tools.reply.strip_bytes`),
    paged under :data:`PAGE_BYTES` (:func:`langslice.doors.tools.reply.paged`);
    a strip and its text stay together."""
    items = opening_items(session.state, session.ctx, limit=CLAUDE_IMAGE_LIMIT,
                          max_bytes=strip_bytes(PAGE_BYTES))
    encoded: list[Item] = [item if isinstance(item, str) else encode_jpeg(item)
                           for item in items]
    return [blocks_of(page) for page in paged(encoded, PAGE_BYTES)]


def save_page(session: Session, page: int, blocks: list[ContentBlock]) -> list[str]:
    """Save the page's pictures in the job folder, as the bytes the host got;
    return their paths (written in the background)."""
    from langslice.job.views import PICTURE_FILE, captured

    pictures = [base64.b64decode(block.data) for block in blocks
                if isinstance(block, ImageContent)]
    try:
        with captured() as saved:
            session.job.views.save(tool="show_stack", arguments={"page": page},
                                   pictures=[(data, None) for data in pictures])
        return [str(item.folder / PICTURE_FILE) for item in saved]
    except Exception:  # saving must never break a page
        logger.warning("Could not queue the pictures of show_stack page %s", page,
                       exc_info=True)
        return []


def briefing(session: Session) -> list[ContentBlock]:
    """Text only; opening pictures are available through show_stack. The
    statement asks the host to read every page before writing, so from here
    every write is refused until it has (the toolbox's opening-read gate,
    :meth:`langslice.doors.tools.toolbox.ToolBox.require_opening`)."""
    if not session.pages:
        session.pages = opening_pages(session)
    session.box.require_opening(len(session.pages))
    return [TextContent(type="text", text=job_statement(
        session.spec, session.state, session.ctx, door="mcp", tool_names=session.box.names,
        opening=opening_for_mcp(len(session.pages)), notes=session.notes,
        max_resolution=session.box.max_view_edge, image_model_off=session.image_model_off,
    ))]


def open_saved_job(job_id: str, atlas_loader: Callable[[str], Any] | None) -> Session:
    """A saved ABBA job (:mod:`langslice.doors.api.saved_jobs`), by id."""
    folder, record = load_job(job_id)
    prepared = prepare_linear(record["params"])
    session = open_job(prepared.spec, atlas_loader, folder)
    session.job_id, session.job_dir, session.prepared = job_id, folder, prepared
    session.notes = read_notes(session.job.layout)
    trace_dir = prepared.trace_dir
    if trace_dir:
        session.trace = HostTrace(trace_dir, session.ctx.image_folder)
    session.channel = HostChannel(job_id, record.get("host_channel"))
    channel = session.channel
    session.events.target = lambda event: channel.event(
        {"kind": "agent_event", "event": public_event(event)})
    checkpoints = checkpoint_callback(prepared, session.channel.event)
    checkpoints.attach(session.job, session.ctx)
    session.host_update = checkpoints
    session.host_update(session.state)
    return session


def open_folder(
    path: str, spec_for: Callable[[str], JobSpec],
    atlas_loader: Callable[[str], Any] | None, *, fresh: bool = False,
) -> Session:
    """The job of the folder the host names: an image folder or its job
    folder. A job made there (``langslice-job FOLDER init``, any door)
    opens as it stands, with its saved settings and notes, resuming its
    checkpoint; a folder without one, or any folder when *fresh*
    (``langslice mcp --fresh``), gets its job from this server's job flags
    (*spec_for*)."""
    try:
        folder = None if fresh else find(path)
    except NoJob:
        folder = None
    if folder is None:
        return open_job(spec_for(path), atlas_loader)
    session = open_job(read_spec(folder), atlas_loader, folder)
    session.notes = read_notes(session.job.layout)
    return session


def finish(session: Session) -> None:
    if session.host_update is not None:
        session.host_update(session.state)
    session.job.emit_results(session.ctx.progress)
    if session.job_dir is not None and session.prepared is not None:
        result = {"state": session.state.to_dict(), "output_dir": str(session.job_dir),
                  "final_updates": session.prepared.final_updates,
                  "abba": dict(session.prepared.abba)}
        if session.prepared.final_angles is not None:
            result["host_angles"] = session.prepared.final_angles
        target = session.job.layout.host_result_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if session.channel is not None:
            session.channel.send({"type": "result", "result": result})
            session.channel.close()


def host_tool(session: Session, tool: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap one toolbox tool for MCP: same name, docstring and arguments.

    ``tool_context`` is ADK's and is dropped from the schema; the tools accept
    ``None``. Tools run in a worker thread (the toolbox serializes them) so a
    long fit does not stall the protocol.
    """
    signature = inspect.signature(tool, eval_str=True)
    parameters = [p for name, p in signature.parameters.items() if name != "tool_context"]

    async def run(**kwargs: Any) -> list[ContentBlock]:
        try:
            def invoke() -> Any:
                if session.host_update is None:
                    result = tool(**kwargs)
                else:
                    with session.job.observe(session.host_update):
                        result = tool(**kwargs)
                if tool.__name__ == "submit" and session.state.submitted:
                    finish(session)
                return result

            def serialized() -> Any:
                with session.lock:
                    return invoke()

            result = await to_thread.run_sync(serialized)
        except Exception as exc:
            logger.exception("Tool %s failed", tool.__name__)
            result = {"status": "error", "error": type(exc).__name__, "message": str(exc)}
        blocks = result_blocks(result)
        if session.trace is not None:
            session.trace.write("tool_result", name=tool.__name__, args=kwargs,
                            content=describe_blocks(blocks))
        return blocks

    run.__name__ = tool.__name__
    run.__doc__ = tool.__doc__
    run.__signature__ = signature.replace(  # type: ignore[attr-defined]
        parameters=parameters, return_annotation=list[ContentBlock]
    )
    return run


def strict_arguments(server: FastMCP, name: str, tool: Callable[..., Any],
                     trace: Callable[[], HostTrace | None] = lambda: None) -> None:
    """Refuse unknown or misplaced arguments on the registered tool *name*.

    FastMCP validates arguments against a model built from the signature and
    drops keys it does not know, so a stray argument would run the tool with
    its default. The registered tool's argument check is replaced by one that
    first applies :func:`langslice.doors.tools.arguments.argument_refusal` (the
    same rule the ADK plugin and the toolbox apply) to the arguments as sent,
    after FastMCP's JSON pre-parse of string-encoded objects, then
    :func:`~langslice.doors.tools.arguments.normalize_arguments`, so what the ADK
    agent may send (a corrected index as a number, a null picture option)
    passes FastMCP's schema check here too. A refusal is traced like any
    tool result (*trace*: the session's trace, when it keeps one).
    """
    from langslice.doors.tools.arguments import argument_refusal, normalize_arguments

    registered = server._tool_manager.get_tool(name)  # noqa: SLF001 — FastMCP has no hook
    if registered is None:
        return
    metadata = registered.fn_metadata
    base = type(metadata)

    class Strict(base):  # type: ignore[valid-type, misc]
        async def call_fn_with_arg_validation(
            self, fn: Callable[..., Any], fn_is_async: bool,
            arguments_to_validate: dict[str, Any],
            arguments_to_pass_directly: dict[str, Any] | None,
        ) -> Any:
            arguments = self.pre_parse_json(arguments_to_validate)
            refusal = argument_refusal(tool, arguments)
            if refusal is not None:
                blocks = result_blocks(refusal)
                held = trace()
                if held is not None:
                    held.write("tool_result", name=name, args=arguments,
                               content=describe_blocks(blocks))
                return blocks
            return await super().call_fn_with_arg_validation(
                fn, fn_is_async, normalize_arguments(tool, arguments),
                arguments_to_pass_directly)

    registered.fn_metadata = Strict.model_construct(
        **{field: getattr(metadata, field) for field in base.model_fields})


# --- the server ------------------------------------------------------------


class LangSliceServer(FastMCP):
    """FastMCP, answering a retired tool's name (``ops.registry.RETIRED``)
    with ``RETIRED_TOOL`` and the tool to use instead, every time it is
    called, where FastMCP would raise "Unknown tool" (traced like any tool
    result in the open job's trace, *traced*: the server's sessions)."""

    traced: dict[str, Any] = {}

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if self._tool_manager.get_tool(name) is None:  # noqa: SLF001 — FastMCP has no hook
            retired = retired_payload(name)
            if retired is not None:
                blocks = result_blocks(retired)
                session = self.traced.get("job")
                if session is not None and session.trace is not None:
                    session.trace.write("tool_result", name=name, args=arguments,
                                        content=describe_blocks(blocks))
                return blocks
        return await super().call_tool(name, arguments)


def build_server(
    spec_for: Callable[[str], JobSpec],
    folder: str | None = None,
    *,
    job_id: str | None = None,
    atlas_loader: Callable[[str], Any] | None = None,
    sessions: dict[str, Session] | None = None,
    fresh: bool = False,
) -> LangSliceServer:
    """The server. *spec_for* turns a folder into this server's job spec,
    for a folder without a job, or for every folder when *fresh*
    (:func:`open_folder`).

    With *folder*, that job opens now and its tools are listed from the
    start; otherwise ``start_job`` names the folder and the tools appear then.
    *job_id* opens that saved job now, for hosts that list tools only once, at
    startup (Claude Code in print mode ignores a later tool-list change).
    *atlas_loader* is for tests and offline hosts, as in the engine.
    *sessions* holds the open job under ``"job"`` (the caller closes it when
    the server stops: :func:`serve`).
    """
    server = LangSliceServer(SERVER_NAME, instructions=INSTRUCTIONS)
    current: dict[str, Session] = {} if sessions is None else sessions
    server.traced = current

    def install(session: Session) -> None:
        """List the session's tools: the verbs the registry gives its spec
        (``session.box.tools``, declared by :mod:`langslice.doors.declarations`),
        then the door's own ``show_stack``."""
        old = current.get("job")
        if old is not None:
            old.close()
            server.remove_tool("show_stack")
            for name in old.box.names:
                server.remove_tool(name)
        for tool in session.box.tools:
            server.add_tool(
                host_tool(session, tool),
                annotations=ToolAnnotations(readOnlyHint=tool.__name__ in READ_ONLY_TOOLS),
                structured_output=False,
            )
            strict_arguments(server, tool.__name__, tool, lambda: session.trace)
        async def show_stack(page: int) -> list[ContentBlock]:
            """Read an opening-picture page (1-based); read every page before writes."""
            if not session.pages:
                session.pages = await to_thread.run_sync(opening_pages, session)
            if not 1 <= page <= len(session.pages):
                raise ValueError(f"page must be between 1 and {len(session.pages)}")
            blocks = session.pages[page - 1]
            session.box.opening_read(page)
            views = save_page(session, page, blocks)
            # The opening pages are this door's seed: the host's viewer
            # follows the whole stack and its log shows the saved pictures.
            try:
                session.events({"kind": "seed", "page": page, "views": views,
                                "text": f"Opening pictures, page {page}"})
            except Exception:
                logger.warning("Could not forward the show_stack event", exc_info=True)
            if session.trace is not None:
                session.trace.write("show_stack", page=page, content=describe_blocks(blocks))
            return blocks

        server.add_tool(show_stack, structured_output=False,
                        annotations=ToolAnnotations(readOnlyHint=True))
        current["job"] = session

    async def start_job(
        image_folder: str = "", job_id: str = "", ctx: Context | None = None,
    ) -> list[ContentBlock]:
        """Open the job the user named: an image_folder (or its job folder),
        or a saved ABBA job's job_id.

        Returns the job facts and status table without images. Read every
        show_stack page before writing. Omit both arguments to repeat the
        current briefing. Never supply both job_id and image_folder.
        """
        if job_id and image_folder:
            raise ValueError("Supply job_id or image_folder, not both")
        session = current.get("job")
        if job_id and (session is None or session.job_id != job_id):
            session = await to_thread.run_sync(open_saved_job, job_id, atlas_loader)
            install(session)
            if ctx is not None:
                await ctx.session.send_tool_list_changed()
        wanted = os.path.abspath(os.path.expanduser(image_folder)) if image_folder else None
        if wanted is not None and (session is None or wanted not in (
                session.ctx.image_folder, str(session.job.folder))):
            session = await to_thread.run_sync(
                partial(open_folder, wanted, spec_for, atlas_loader, fresh=fresh))
            install(session)
            if ctx is not None:
                await ctx.session.send_tool_list_changed()
        if session is None:
            return [TextContent(type="text", text=json.dumps({
                "status": "error",
                "error": "NO_FOLDER",
                "message": "Name a job: start_job(image_folder=...), or a saved ABBA "
                           "job's start_job(job_id=...).",
            }))]
        blocks = await to_thread.run_sync(briefing, session)
        if session.trace is not None:
            session.trace.write("briefing", content=describe_blocks(blocks))
        return blocks

    server.add_tool(start_job, structured_output=False)
    if job_id:
        install(open_saved_job(job_id, atlas_loader))
    elif folder is not None:
        install(open_folder(os.path.abspath(os.path.expanduser(folder)), spec_for,
                            atlas_loader, fresh=fresh))
    return server


def serve(
    spec_for: Callable[[str], JobSpec], folder: str | None = None, job_id: str | None = None,
    *, fresh: bool = False,
) -> None:
    """Run the server over stdio until the host disconnects.

    Scientific libraries print through Python and native stdout alike, so
    the protocol gets a private duplicate of stdout and everything else that
    writes there lands on stderr — the host's log.
    """
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    sys.stdout.flush()
    wire = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    with redirect_stdout(sys.stderr):
        sessions: dict[str, Session] = {}
        server = build_server(spec_for, folder, job_id=job_id, sessions=sessions, fresh=fresh)
        try:
            anyio.run(_run_stdio, server, wire)
        finally:
            # The host went away: running image-model calls land, pictures
            # are written (Session.close).
            for session in sessions.values():
                session.close()


async def _run_stdio(server: FastMCP, wire: Any) -> None:
    out = anyio.wrap_file(TextIOWrapper(wire, encoding="utf-8", write_through=True))
    async with stdio_server(stdout=out) as (read_stream, write_stream):
        # FastMCP.run_stdio_async always writes to sys.stdout; the low-level
        # server takes the protocol stream explicitly.
        lowlevel = server._mcp_server  # pyright: ignore[reportPrivateUsage]
        await lowlevel.run(read_stream, write_stream, lowlevel.create_initialization_options())

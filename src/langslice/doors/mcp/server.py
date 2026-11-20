"""LangSlice tools for MCP hosts, with job folders and paged pictures.

The host owns its conversation; LangSlice owns validated settings, tool gates
and checkpoints. No model is called here.
"""

from __future__ import annotations

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
from io import FileIO, TextIOWrapper
from pathlib import Path
from typing import Any, BinaryIO

import anyio
from anyio import to_thread
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.tools import Tool
from mcp.server.lowlevel.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.session import ServerSession
from mcp.server.stdio import stdio_server
from mcp.types import ContentBlock, TextContent, ToolAnnotations

from langslice.core.pictures.opening import (
    CLAUDE_MAX_VIEW_EDGE,
)
from langslice.core.refusal import RefusalCode, Refused
from langslice.core.stack.stack import Stack
from langslice.doors.call_log import TRACE_DIR_ENV, HostTrace
from langslice.doors.jobs import (
    JobContext,
    NoJob,
    close_job,
    context,
    find,
    image_model_off,
    provider_connected,
    read_spec,
)
from langslice.doors.mcp.pages import (
    describe_blocks,
    opening_pages,
    result_blocks,
    save_page,
)
from langslice.doors.reference_card import write_card
from langslice.doors.stdio import protocol_input
from langslice.doors.tools.toolbox import build_tools
from langslice.job.folder.layout import locate_job_folder
from langslice.job.job import Job
from langslice.job.settings.spec import JobSpec
from langslice.job.settings.tool_table import refusal
from langslice.tools.catalog import VERBS
from langslice.tools.descriptions.door_text import (
    opening_for_mcp,
    read_notes,
    review_lines,
)
from langslice.tools.descriptions.render import signed
from langslice.tools.descriptions.statement import (
    StatementFacts,
    job_statement,
)
from langslice.tools.pipeline import ToolBox

logger = logging.getLogger(__name__)

SERVER_NAME = "langslice"

READ_ONLY_TOOLS = frozenset(name for name, verb in VERBS.items() if verb.kind == "read")

INSTRUCTIONS = (
    "LangSlice places histology sections in a brain atlas. The job comes in the message "
    "that launched you; call `start_job` with its folder first: it opens the job and gives "
    "you its tools. Read every show_stack page before writing. Work only through the "
    "LangSlice tools and finish with `submit`."
)


class LangSliceServer(FastMCP):
    """FastMCP, with the two members of its internals this door needs.

    FastMCP has no public way to read a registered tool or to reach the
    protocol server under it (to run it over the door's own streams); a
    subclass reaches both through its own protected members, here and nowhere
    else.
    """

    def registered_tool(self, name: str) -> Tool | None:
        """Return the tool registered under *name*, None when there is none."""
        return self._tool_manager.get_tool(name)

    def protocol_server(self) -> Server:
        """Return the protocol server this FastMCP serves."""
        return self._mcp_server


@dataclass
class Session:
    """One open job as this door holds it.

    ``job`` is the core :class:`~langslice.job.job.Job` (state, spec,
    undo/redo, checkpoint, submit gates), the one
    ``langslice.doors.jobs.open_folder`` opens; ``box`` is the tools over it,
    whose gates and delivery bookkeeping this door shares with the ADK agent's.
    The rest is the door's own: the user's notes, the opening pages and the
    trace.
    """

    job: Job
    ctx: JobContext
    box: ToolBox
    trace: HostTrace | None
    notes: str = ""
    pages: list[list[ContentBlock]] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    image_model_off: bool = False

    @property
    def spec(self) -> JobSpec:
        return self.job.spec

    @property
    def state(self) -> Stack:
        return self.job.state

    def close(self) -> None:
        """Let the job go: its running image-model calls settled and
        recorded, its pictures written (:func:`langslice.doors.jobs.close_job`,
        as every door ends a job). When another job replaces it and when the
        server stops."""
        try:
            close_job(self.job)
        except Exception:
            logger.warning("Could not finish the job in %s", self.job.folder, exc_info=True)


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
    job_folder = folder or locate_job_folder(spec.image_folder, spec.job_dir, emit=logger.info)
    ctx = context(spec, job_folder, atlas_loader=atlas_loader)
    off = image_model_off(spec, provider_connected(spec))
    reason = refusal(spec, image_model=not off)
    if reason is not None:
        raise ValueError(f"{reason} (the job's image model is not connected here)")
    job = Job.open(spec, ctx, folder=job_folder, results_path=ctx.results_path)
    spec = ctx.spec = job.spec
    write_card(job.layout)
    trace_dir = os.environ.get(TRACE_DIR_ENV)
    trace = HostTrace(trace_dir, ctx.image_folder) if trace_dir else None
    box = build_tools(
        job.state,
        ctx,
        spec,
        job=job,
        max_view_edge=CLAUDE_MAX_VIEW_EDGE,
        image_model_connected=not off,
        door="mcp",
    )
    box.answers_crashes = True
    return Session(job, ctx, box, trace, image_model_off=off)


def mcp_statement(session: Session, *, scripting: str = "") -> str:
    """The job statement as the MCP door words it: the toolbox's tool names, its
    picture limit, the ``show_stack`` pages (drawn now when not yet), the
    user's notes and the image-model line. ``langslice job FOLDER launch --role
    register`` (:mod:`langslice.doors.commands.job.launch`) embeds it in the
    message that launches an MCP registration agent; *scripting* is the
    paragraph on scripts that ``--role register-scripting`` adds after the
    opening's (``door_text.SCRIPTING``, filled)."""
    if not session.pages:
        session.pages = opening_pages(session)
    facts = StatementFacts.from_run(
        session.spec,
        session.state,
        session.ctx,
        tool_names=session.box.names,
        max_resolution=session.box.max_view_edge,
    )
    opening = opening_for_mcp(len(session.pages))
    if scripting:
        opening = [*opening, "", scripting]
    review = review_lines(session.job.layout, session.state, session.spec.switched_off)
    return job_statement(
        facts,
        door="mcp",
        opening=opening,
        notes=session.notes,
        image_model_off=session.image_model_off,
        review=review,
    )


def arm_opening(session: Session) -> int:
    """Require the opening pages to be read before any write (the toolbox's
    opening-read gate, :meth:`langslice.tools.pipeline.ToolBox.require_opening`);
    the number of pages."""
    if not session.pages:
        session.pages = opening_pages(session)
    session.box.require_opening(len(session.pages))
    return len(session.pages)


def opened_reply(session: Session) -> list[ContentBlock]:
    """``start_job``'s reply for a job a launching message described: the job
    opened, its sections, and the pages of its opening; arms the opening-read gate."""
    pages = arm_opening(session)
    shown = "page 1" if pages == 1 else f"pages 1 to {pages}"
    return [
        TextContent(
            type="text",
            text=(
                f"Job opened: {session.job.folder}. {len(session.state.in_order())} sections. "
                f"The opening is show_stack {shown}: read every page before any write."
            ),
        )
    ]


def open_folder(
    path: str,
    spec_for: Callable[[str], JobSpec],
    atlas_loader: Callable[[str], Any] | None,
) -> Session:
    """The job of the folder the host names: an image folder or its job
    folder. A job made there (``langslice job FOLDER init``, any door)
    opens as it stands, with its saved settings and notes, resuming its
    checkpoint; a folder without one gets its job from this server's job
    flags (*spec_for*)."""
    try:
        folder = find(path)
    except NoJob:
        folder = None
    if folder is None:
        return open_job(spec_for(path), atlas_loader)
    session = open_job(read_spec(folder), atlas_loader, folder)
    session.notes = read_notes(session.job.layout)
    return session


def finish(session: Session) -> None:
    session.job.emit_results(session.ctx.progress)


def host_tool(session: Session, tool: Callable[..., object]) -> Callable[..., Any]:
    """Wrap one toolbox tool for MCP: same name, docstring and arguments.

    ``tool_context`` is ADK's and is dropped from the schema; the tools accept
    ``None``. Tools run in a worker thread (the toolbox serializes them) so a
    long fit does not stall the protocol.
    """
    signature = inspect.signature(tool, eval_str=True)
    parameters = [p for name, p in signature.parameters.items() if name != "tool_context"]

    async def run(**kwargs: Any) -> list[ContentBlock]:
        def invoke() -> object:
            result: object = tool(**kwargs)
            if tool.__name__ == "submit" and session.state.submitted:
                finish(session)
            return result

        def serialized() -> object:
            with session.lock:
                return invoke()

        result = await to_thread.run_sync(serialized)
        blocks = result_blocks(result, job=session.job)
        if session.trace is not None:
            session.trace.write(
                "tool_result", name=tool.__name__, args=kwargs, content=describe_blocks(blocks)
            )
        return blocks

    run.__name__ = tool.__name__
    run.__doc__ = tool.__doc__
    return signed(
        run, signature.replace(parameters=parameters, return_annotation=list[ContentBlock])
    )


def strict_arguments(
    server: LangSliceServer,
    name: str,
    tool: Callable[..., Any],
    trace: Callable[[], HostTrace | None] = lambda: None,
) -> None:
    """Refuse unknown or misplaced arguments on the registered tool *name*.

    FastMCP validates arguments against a model built from the signature and
    drops keys it does not know, so a stray argument would run the tool with
    its default. The registered tool's argument check is replaced by one that
    first applies :func:`langslice.tools.arguments.argument_refusal` (the same
    rule the ADK plugin and the toolbox apply) to the arguments as sent, after
    FastMCP's JSON pre-parse of string-encoded objects, then
    :func:`~langslice.tools.arguments.normalize_arguments`, so what the ADK
    agent may send (a number where a filename belongs, a null picture option)
    passes FastMCP's schema check here too. A refusal is traced like any tool
    result (*trace*: the session's trace, when it keeps one).
    """
    from langslice.tools.arguments import argument_refusal, normalize_arguments

    registered = server.registered_tool(name)
    if registered is None:
        return
    metadata = registered.fn_metadata
    base = type(metadata)

    class Strict(base):
        async def call_fn_with_arg_validation(
            self,
            fn: Callable[..., Any],
            fn_is_async: bool,
            arguments_to_validate: dict[str, Any],
            arguments_to_pass_directly: dict[str, Any] | None,
        ) -> Any:
            arguments = self.pre_parse_json(arguments_to_validate)
            refusal = argument_refusal(tool, arguments)
            if refusal is not None:
                blocks = result_blocks(refusal.payload())
                held = trace()
                if held is not None:
                    held.write(
                        "tool_result", name=name, args=arguments, content=describe_blocks(blocks)
                    )
                return blocks
            return await super().call_fn_with_arg_validation(
                fn, fn_is_async, normalize_arguments(tool, arguments), arguments_to_pass_directly
            )

    registered.fn_metadata = Strict.model_construct(
        **{field: getattr(metadata, field) for field in base.model_fields}
    )


def build_server(
    spec_for: Callable[[str], JobSpec],
    folder: str | None = None,
    *,
    atlas_loader: Callable[[str], Any] | None = None,
    sessions: dict[str, Session] | None = None,
) -> LangSliceServer:
    """The server. *spec_for* turns a folder into this server's job spec,
    for a folder without a job (:func:`open_folder`).

    With *folder*, that job opens now and its tools are listed from the
    start (for hosts that list tools only once, at startup: Claude Code in
    print mode ignores a later tool-list change); otherwise ``start_job``
    names the folder and the tools appear then. *atlas_loader* is for tests
    and offline hosts, as in the engine.
    *sessions* holds the open job under ``"job"`` (the caller closes it when
    the server stops: :func:`serve`).
    """
    server = LangSliceServer(SERVER_NAME, instructions=INSTRUCTIONS)
    current: dict[str, Session] = {} if sessions is None else sessions

    def install(session: Session) -> None:
        """List the session's tools: the verbs the registry gives its spec
        (``session.box.tools``, declared by
        :mod:`langslice.tools.descriptions.render`),
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
            save_page(session, page, blocks)
            if session.trace is not None:
                session.trace.write("show_stack", page=page, content=describe_blocks(blocks))
            return blocks

        server.add_tool(
            show_stack, structured_output=False, annotations=ToolAnnotations(readOnlyHint=True)
        )
        current["job"] = session

    async def start_job(
        image_folder: str = "",
        ctx: Context[ServerSession, object, object] | None = None,
    ) -> list[ContentBlock]:
        """Open the job named in the message that launched you: its image_folder (or
        its job folder). It answers with the job's section count and the pages of
        its opening, and gives you the job's tools. Read every show_stack page
        before writing. Omit the argument to repeat that reply.
        """
        session = current.get("job")
        wanted = os.path.abspath(os.path.expanduser(image_folder)) if image_folder else None
        if wanted is not None and (
            session is None or wanted not in (session.ctx.image_folder, str(session.job.folder))
        ):
            session = await to_thread.run_sync(partial(open_folder, wanted, spec_for, atlas_loader))
            install(session)
            if ctx is not None:
                await ctx.session.send_tool_list_changed()
        if session is None:
            refusal = Refused(RefusalCode.BAD_ARGUMENT, "Name a job: start_job(image_folder=...).")
            reply = json.dumps(refusal.payload())
            return [TextContent(type="text", text=reply)]
        blocks = await to_thread.run_sync(opened_reply, session)
        if session.trace is not None:
            session.trace.write("briefing", content=describe_blocks(blocks))
        return blocks

    server.add_tool(start_job, structured_output=False)
    if folder is not None:
        install(open_folder(os.path.abspath(os.path.expanduser(folder)), spec_for, atlas_loader))
    return server


def serve(spec_for: Callable[[str], JobSpec], folder: str | None = None) -> None:
    """Run the server over stdio until the host disconnects.

    Scientific libraries print through Python and native stdout alike, so
    the protocol gets a private duplicate of stdout and everything else that
    writes there lands on stderr — the host's log. It reads a private
    duplicate of stdin too (:func:`langslice.doors.stdio.protocol_input`):
    on Windows a module loaded while the protocol waits on the process's
    own stdin never finishes loading.
    """
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    sys.stdout.flush()
    wire = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    with redirect_stdout(sys.stderr), protocol_input() as host_input:
        sessions: dict[str, Session] = {}
        server = build_server(spec_for, folder, sessions=sessions)
        try:
            anyio.run(_run_stdio, server, wire, host_input)
        finally:
            for session in sessions.values():
                session.close()


def initialization_options(server: LangSliceServer) -> InitializationOptions:
    """What the server tells the host when it connects: its capabilities with
    ``tools.listChanged``, so a host takes the tool list again when
    ``start_job`` installs a job's tools (``send_tool_list_changed``); a host
    ignores the notice from a server that does not advertise it."""
    lowlevel = server.protocol_server()
    return lowlevel.create_initialization_options(NotificationOptions(tools_changed=True))


async def _run_stdio(server: LangSliceServer, wire: FileIO, host_input: BinaryIO) -> None:
    text = TextIOWrapper(wire, encoding="utf-8", newline="\n", write_through=True)
    out = anyio.wrap_file(text)
    source = anyio.wrap_file(TextIOWrapper(host_input, encoding="utf-8", errors="replace"))
    streams = stdio_server(stdin=source, stdout=out)
    async with streams as (read_stream, write_stream):
        lowlevel = server.protocol_server()
        await lowlevel.run(read_stream, write_stream, initialization_options(server))

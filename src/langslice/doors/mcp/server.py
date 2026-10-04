"""LangSlice tools for Claude hosts, with saved ABBA jobs and paged pictures.

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
import uuid
from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO, TextIOWrapper
from pathlib import Path
from typing import Any

import anyio
from anyio import to_thread
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.stdio import stdio_server
from mcp.types import ContentBlock, ImageContent, TextContent, ToolAnnotations
from PIL import Image

from langslice.agent.engine import EngineContext, build_context
from langslice.agent.trace import TRACE_DIR_ENV
from langslice.core.opening import CLAUDE_IMAGE_LIMIT, CLAUDE_MAX_VIEW_EDGE, opening_items
from langslice.core.spec import JobSpec
from langslice.core.state import StackState
from langslice.doors.card import write_card
from langslice.doors.mcp.host_channel import HostChannel
from langslice.doors.mcp.prompt import job_statement
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.doors.tools.media import encode_jpeg
from langslice.doors.tools.toolbox import ToolBox, build_tools
from langslice.hosts.api.abba_worker import PreparedLinear, checkpoint_callback, prepare_linear
from langslice.hosts.api.claude_jobs import load_job
from langslice.job.job import Job
from langslice.ops.registry import VERBS

logger = logging.getLogger(__name__)

SERVER_NAME = "langslice"

#: Tools that only look (the registry's read verbs, plus the door's own
#: ``show_stack``). Hosts may use the hint to skip a confirmation.
READ_ONLY_TOOLS = frozenset(name for name, verb in VERBS.items() if verb.kind == "read")

INSTRUCTIONS = (
    "LangSlice places histology sections in a brain atlas. Call `start_job` "
    "first: it opens the stack and answers with the job description. Read every "
    "show_stack page before writing. Work only through the LangSlice tools and "
    "finish with `submit`."
)


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
    trace: McpTrace | None
    job_id: str = ""
    job_dir: Path | None = None
    notes: str = ""
    #: Sends the host (ABBA) its live update after every checkpoint.
    host_update: Callable[[StackState], None] | None = None
    prepared: PreparedLinear | None = None
    channel: HostChannel | None = None
    pages: list[list[ContentBlock]] = field(default_factory=list)
    lock: Any = field(default_factory=threading.RLock, repr=False)

    @property
    def spec(self) -> JobSpec:
        return self.job.spec

    @property
    def state(self) -> StackState:
        return self.job.state


# --- content conversion ----------------------------------------------------


def image_block(image: Image.Image) -> ImageContent:
    """One core picture as an MCP image block (the doors' JPEG encoding)."""
    return ImageContent(
        type="image", data=base64.b64encode(encode_jpeg(image)).decode("ascii"),
        mimeType="image/jpeg",
    )


def result_blocks(result: Any) -> list[ContentBlock]:
    """A toolbox result as MCP content: its JSON, then its media in order.

    The tools return plain pictures (PIL images, captions burned in) and
    lines of text under ``TOOL_MEDIA_PARTS_KEY``; each picture becomes an
    image block in the doors' JPEG encoding (:func:`image_block`), each
    non-empty text a text block. ``images_attached`` counts both.
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
    blocks: list[ContentBlock] = [
        TextContent(type="text", text=json.dumps(body, default=str))
    ]
    for item in media:
        if isinstance(item, str):
            if item:
                blocks.append(TextContent(type="text", text=item))
        else:
            blocks.append(image_block(item))
    return blocks


# --- trace -----------------------------------------------------------------


class McpTrace:
    """JSONL record of what the host was shown and what it called.

    The host's own words between calls never reach this server, so unlike
    :class:`langslice.agent.trace.SessionTrace` there is no model record.
    Images are descriptors, never bytes.
    """

    def __init__(self, trace_dir: str | Path, folder: str) -> None:
        name = Path(folder).name or "stack"
        self.path = Path(trace_dir) / f"mcp_{name}_{uuid.uuid4().hex[:8]}.jsonl"

    def write(self, kind: str, **fields: Any) -> None:
        record = {"kind": kind, **fields, "at": datetime.now().isoformat(timespec="seconds")}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        except Exception:
            logger.warning("Could not write trace record to %s", self.path, exc_info=True)


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
    saved job's (the same place, as its index names it)."""
    if spec.has("nonlinear"):
        raise ValueError("Image generation is unavailable through the Claude connector")
    ctx = build_context(spec, atlas_loader=atlas_loader, job_folder=folder)
    job = Job.open(spec, ctx, folder=ctx.job_folder, results_path=ctx.results_path)
    write_card(job.layout)
    trace_dir = os.environ.get(TRACE_DIR_ENV)
    trace = McpTrace(trace_dir, ctx.image_folder) if trace_dir else None
    # The host is Claude: its pictures are capped at Claude's largest image.
    box = build_tools(job.state, ctx, spec, job=job, max_view_edge=CLAUDE_MAX_VIEW_EDGE)
    return Session(job, ctx, box, trace)


# Budget includes JSON/text overhead, not only encoded image bytes.
PAGE_BYTES = 680_000


def page_size(blocks: list[ContentBlock]) -> int:
    return len(json.dumps([block.model_dump(exclude_none=True) for block in blocks]).encode())


def _fit_page(blocks: list[ContentBlock]) -> list[ContentBlock]:
    """Shrink oversized pictures together; never merge labelled images."""
    while page_size(blocks) > PAGE_BYTES:
        changed = False
        for block in blocks:
            if isinstance(block, ImageContent):
                with Image.open(BytesIO(base64.b64decode(block.data))) as image:
                    if max(image.size) <= 32:
                        continue
                    image = image.convert("RGB")
                    image.thumbnail((max(1, int(image.width * .8)), max(1, int(image.height * .8))))
                    stream = BytesIO()
                    image.save(stream, format="JPEG", quality=80)
                    block.data = base64.b64encode(stream.getvalue()).decode("ascii")
                    block.mimeType = "image/jpeg"
                    changed = True
        if not changed:
            raise ValueError("Opening picture labels exceed the page budget")
    return blocks


def opening_pages(session: Session) -> list[list[ContentBlock]]:
    """The opening strips (:mod:`langslice.core.opening`) at Claude's image
    size, paged under :data:`PAGE_BYTES`; a strip and its text stay together."""
    pages: list[list[ContentBlock]] = []
    page: list[ContentBlock] = []
    pending: list[ContentBlock] = []
    for item in opening_items(session.state, session.ctx, limit=CLAUDE_IMAGE_LIMIT):
        if isinstance(item, str):
            pending.append(TextContent(type="text", text=item))
            continue
        pending.append(image_block(item))
        group = _fit_page(pending)
        if page and page_size(page + group) > PAGE_BYTES:
            pages.append(page)
            page = []
        page.extend(group)
        pending = []
    if page or pending:
        pages.append(page + pending)
    return pages


def save_page(session: Session, page: int, blocks: list[ContentBlock]) -> None:
    """Save the page's pictures in the job folder, as the bytes the host got."""
    pictures = [base64.b64decode(block.data) for block in blocks
                if isinstance(block, ImageContent)]
    try:
        session.job.views.save(tool="show_stack", arguments={"page": page},
                               pictures=[(data, None) for data in pictures])
    except Exception:  # saving must never break a page
        logger.warning("Could not queue the pictures of show_stack page %s", page,
                       exc_info=True)


def briefing(session: Session) -> list[ContentBlock]:
    """Text only; opening pictures are available through show_stack."""
    if not session.pages:
        session.pages = opening_pages(session)
    return [TextContent(type="text", text=job_statement(
        session.spec, session.state, session.ctx, len(session.pages), session.notes,
        session.box.names, max_resolution=session.box.max_view_edge,
    ))]


def open_saved_job(job_id: str, atlas_loader: Callable[[str], Any] | None) -> Session:
    folder, record = load_job(job_id)
    if record["kind"] == "folder":
        return open_folder_job(job_id, folder, record, atlas_loader)
    prepared = prepare_linear(record["params"])
    if prepared.spec.has("nonlinear"):
        raise ValueError("Image generation is unavailable in Claude mode")
    session = open_job(prepared.spec, atlas_loader, folder)
    session.job_id, session.job_dir, session.prepared = job_id, folder, prepared
    session.notes = record.get("notes", "")
    trace_dir = prepared.trace_dir
    if trace_dir:
        session.trace = McpTrace(trace_dir, session.ctx.image_folder)
    session.channel = HostChannel(job_id, record.get("host_channel"))
    session.host_update = checkpoint_callback(prepared, session.channel.event)
    session.host_update(session.state)
    return session


def open_folder_job(
    job_id: str, folder: Path, record: dict[str, Any],
    atlas_loader: Callable[[str], Any] | None,
) -> Session:
    """A plain-folder job: its checkpoint and results live in its job folder.

    Reopening it (a restarted server, a new chat) resumes from that checkpoint.
    """
    spec = JobSpec.from_dict(record["spec"])
    spec.resume = True
    spec.out = None
    session = open_job(spec, atlas_loader, folder)
    session.job_id, session.job_dir = job_id, folder
    session.notes = record.get("notes", "")
    if record.get("trace_dir"):
        session.trace = McpTrace(record["trace_dir"], session.ctx.image_folder)
    return session


def finish(session: Session) -> None:
    if session.host_update is not None:
        session.host_update(session.state)
    session.job.emit_results(session.ctx.progress)
    if session.job_dir is not None and session.prepared is not None:
        result = {"state": session.state.to_dict(), "output_dir": str(session.job_dir),
                  "final_updates": session.prepared.final_updates}
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
        # Pictures from earlier calls have reached the host by now: the call
        # that follows them is the host's next move. The placement gates
        # (compare-before-write, review-after-write) read this record.
        session.box.begin_model_call()
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


def strict_arguments(server: FastMCP, name: str, tool: Callable[..., Any]) -> None:
    """Refuse unknown or misplaced arguments on the registered tool *name*.

    FastMCP validates arguments against a model built from the signature and
    drops keys it does not know, so a stray argument would run the tool with
    its default. The registered tool's argument check is replaced by one that
    first applies :func:`langslice.doors.tools.arguments.argument_refusal` (the
    same rule the ADK plugin and the toolbox apply) to the arguments as sent,
    after FastMCP's JSON pre-parse of string-encoded objects, then
    :func:`~langslice.doors.tools.arguments.normalize_arguments`, so what the ADK
    agent may send (a corrected index as a number, a null picture option)
    passes FastMCP's schema check here too.
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
                return result_blocks(refusal)
            return await super().call_fn_with_arg_validation(
                fn, fn_is_async, normalize_arguments(tool, arguments),
                arguments_to_pass_directly)

    registered.fn_metadata = Strict.model_construct(
        **{field: getattr(metadata, field) for field in base.model_fields})


# --- the server ------------------------------------------------------------


def build_server(
    spec_for: Callable[[str], JobSpec],
    folder: str | None = None,
    *,
    job_id: str | None = None,
    atlas_loader: Callable[[str], Any] | None = None,
) -> FastMCP:
    """The server. *spec_for* turns a folder into this server's job spec.

    With *folder*, that job opens now and its tools are listed from the
    start; otherwise ``start_job`` names the folder and the tools appear then.
    *job_id* opens that saved job now, for hosts that list tools only once, at
    startup (Claude Code in print mode ignores a later tool-list change).
    *atlas_loader* is for tests and offline hosts, as in the engine.
    """
    server = FastMCP(SERVER_NAME, instructions=INSTRUCTIONS)
    current: dict[str, Session] = {}

    def install(session: Session) -> None:
        """List the session's tools: the verbs the registry gives its spec
        (``session.box.tools``, declared by :mod:`langslice.doors.declarations`),
        then the door's own ``show_stack``."""
        old = current.get("job")
        if old is not None:
            if old.channel is not None:
                old.channel.close()
            server.remove_tool("show_stack")
            for name in old.box.names:
                server.remove_tool(name)
        for tool in session.box.tools:
            server.add_tool(
                host_tool(session, tool),
                annotations=ToolAnnotations(readOnlyHint=tool.__name__ in READ_ONLY_TOOLS),
                structured_output=False,
            )
            strict_arguments(server, tool.__name__, tool)
        async def show_stack(page: int) -> list[ContentBlock]:
            """Read an opening-picture page (1-based); read every page before writes."""
            if not session.pages:
                session.pages = await to_thread.run_sync(opening_pages, session)
            if not 1 <= page <= len(session.pages):
                raise ValueError(f"page must be between 1 and {len(session.pages)}")
            blocks = session.pages[page - 1]
            save_page(session, page, blocks)
            if session.trace is not None:
                session.trace.write("show_stack", page=page, content=describe_blocks(blocks))
            return blocks

        server.add_tool(show_stack, structured_output=False,
                        annotations=ToolAnnotations(readOnlyHint=True))
        current["job"] = session

    async def start_job(
        image_folder: str = "", job_id: str = "", ctx: Context | None = None,
    ) -> list[ContentBlock]:
        """Open a saved LangSlice job by job_id, or a development image_folder.

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
        if wanted is not None and (session is None or wanted != session.ctx.image_folder):
            session = await to_thread.run_sync(open_job, spec_for(wanted), atlas_loader)
            install(session)
            if ctx is not None:
                await ctx.session.send_tool_list_changed()
        if session is None:
            return [TextContent(type="text", text=json.dumps({
                "status": "error",
                "error": "NO_FOLDER",
                "message": "Name a saved job: start_job(job_id=...), or an image_folder.",
            }))]
        blocks = await to_thread.run_sync(briefing, session)
        if session.trace is not None:
            session.trace.write("briefing", content=describe_blocks(blocks))
        return blocks

    server.add_tool(start_job, structured_output=False)
    if job_id:
        install(open_saved_job(job_id, atlas_loader))
    elif folder is not None:
        install(open_job(spec_for(os.path.abspath(os.path.expanduser(folder))), atlas_loader))
    return server


def serve(
    spec_for: Callable[[str], JobSpec], folder: str | None = None, job_id: str | None = None,
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
        server = build_server(spec_for, folder, job_id=job_id)
        anyio.run(_run_stdio, server, wire)


async def _run_stdio(server: FastMCP, wire: Any) -> None:
    out = anyio.wrap_file(TextIOWrapper(wire, encoding="utf-8", write_through=True))
    async with stdio_server(stdout=out) as (read_stream, write_stream):
        # FastMCP.run_stdio_async always writes to sys.stdout; the low-level
        # server takes the protocol stream explicitly.
        lowlevel = server._mcp_server  # pyright: ignore[reportPrivateUsage]
        await lowlevel.run(read_stream, write_stream, lowlevel.create_initialization_options())

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
from google.genai import types
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.stdio import stdio_server
from mcp.types import ContentBlock, ImageContent, TextContent, ToolAnnotations

from langslice.adk import TOOL_MEDIA_PARTS_KEY
from langslice.api.abba_worker import PreparedLinear, checkpoint_callback, prepare_linear
from langslice.api.claude_jobs import load_job
from langslice.linear.atlas_fetch import atlas_strip_parts
from langslice.linear.checkpoint import load_checkpoint, observe_checkpoints, save_checkpoint
from langslice.linear.engine import (
    EngineContext,
    apply_host_inputs,
    build_context,
    emit_results,
    ingest,
)
from langslice.linear.render import stack_image_parts
from langslice.linear.spec import JobSpec
from langslice.linear.state import StackState
from langslice.linear.toolbox import ToolBox, build_tools
from langslice.linear.trace import TRACE_DIR_ENV
from langslice.mcp_server.host_channel import HostChannel
from langslice.mcp_server.prompt import job_statement

logger = logging.getLogger(__name__)

SERVER_NAME = "langslice"

#: Tools that only look. Hosts may use the hint to skip a confirmation.
READ_ONLY_TOOLS = frozenset(
    {"status", "view_slices", "view_atlas", "view_placement", "view_stack"}
)

INSTRUCTIONS = (
    "LangSlice places histology sections in a brain atlas. Call `start_job` "
    "first: it opens the stack and answers with the job description. Read every "
    "show_stack page before writing. Work only through the LangSlice tools and "
    "finish with `submit`."
)


@dataclass
class Job:
    """One open folder: the same objects ``langslice.linear.engine.run`` holds."""

    spec: JobSpec
    ctx: EngineContext
    state: StackState
    box: ToolBox
    trace: McpTrace | None
    job_id: str = ""
    job_dir: Path | None = None
    notes: str = ""
    checkpoint: Callable[[Any], None] | None = None
    prepared: PreparedLinear | None = None
    channel: HostChannel | None = None
    pages: list[list[ContentBlock]] = field(default_factory=list)
    lock: Any = field(default_factory=threading.RLock, repr=False)


# --- content conversion ----------------------------------------------------


def part_blocks(part: types.Part) -> list[ContentBlock]:
    """One genai Part as MCP content: inline images and text, nothing else."""
    blob = part.inline_data
    if blob is not None and blob.data:
        return [
            ImageContent(
                type="image",
                data=base64.b64encode(blob.data).decode("ascii"),
                mimeType=blob.mime_type or "image/jpeg",
            )
        ]
    if part.text:
        return [TextContent(type="text", text=part.text)]
    return []


def result_blocks(result: Any) -> list[ContentBlock]:
    """A toolbox result as MCP content: its JSON, then its pictures in order."""
    images: list[types.Part] = []
    if isinstance(result, dict):
        body = dict(result)
        media = body.pop(TOOL_MEDIA_PARTS_KEY, None)
        if isinstance(media, list):
            images = [part for part in media if isinstance(part, types.Part)]
            body["images_attached"] = len(images)
        elif media is not None:
            # Already text (e.g. a note that the pictures were not produced).
            body[TOOL_MEDIA_PARTS_KEY] = media
    else:
        body = result
    blocks: list[ContentBlock] = [
        TextContent(type="text", text=json.dumps(body, default=str))
    ]
    for part in images:
        blocks.extend(part_blocks(part))
    return blocks


# --- trace -----------------------------------------------------------------


class McpTrace:
    """JSONL record of what the host was shown and what it called.

    The host's own words between calls never reach this server, so unlike
    :class:`langslice.linear.trace.SessionTrace` there is no model record.
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
    checkpoint_path: str | None = None,
) -> Job:
    """Open *spec*'s folder the way the engine does, without a model.

    *checkpoint_path* moves the checkpoint (and the resume point) out of the
    image folder, into a saved job's own directory.
    """
    if spec.has("nonlinear"):
        raise ValueError("Image generation is unavailable through the Claude connector")
    ctx = build_context(spec, atlas_loader=atlas_loader)
    if checkpoint_path is not None:
        ctx.checkpoint_path = checkpoint_path
    state = load_checkpoint(ctx.checkpoint_path) if spec.resume else None
    if state is not None:
        ctx.progress(f"[ingest] resuming from {ctx.checkpoint_path}")
        state.spec = spec.to_dict()
        state.submitted = False
    else:
        state = ingest(spec, ctx)
        apply_host_inputs(state, spec)
    save_checkpoint(state, ctx.checkpoint_path)
    trace_dir = os.environ.get(TRACE_DIR_ENV)
    trace = McpTrace(trace_dir, ctx.image_folder) if trace_dir else None
    return Job(spec, ctx, state, build_tools(state, ctx, spec), trace)


# Budget includes JSON/text overhead, not only encoded image bytes.
PAGE_BYTES = 680_000


def page_size(blocks: list[ContentBlock]) -> int:
    return len(json.dumps([block.model_dump(exclude_none=True) for block in blocks]).encode())


def _fit_page(blocks: list[ContentBlock]) -> list[ContentBlock]:
    """Shrink oversized pictures together; never merge labelled images."""
    from PIL import Image

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


def opening_pages(job: Job) -> list[list[ContentBlock]]:
    sections = stack_image_parts(job.state, job.ctx)
    pages: list[list[ContentBlock]] = []
    page: list[ContentBlock] = []
    pending: list[ContentBlock] = []
    for part in sections:
        pending.extend(part_blocks(part))
        if part.inline_data is None:
            continue
        group = _fit_page(pending)
        if page and page_size(page + group) > PAGE_BYTES:
            pages.append(page)
            page = []
        page.extend(group)
        pending = []
    if page or pending:
        pages.append(page + pending)
    atlas: list[ContentBlock] = [TextContent(type="text", text="Atlas reference strip")]
    for part in atlas_strip_parts(job.ctx, job.state):
        atlas.extend(part_blocks(part))
    pages.append(_fit_page(atlas))
    return pages


def briefing(job: Job) -> list[ContentBlock]:
    """Text only; opening pictures are available through show_stack."""
    if not job.pages:
        job.pages = opening_pages(job)
    return [TextContent(type="text", text=job_statement(
        job.spec, job.state, job.ctx, len(job.pages), job.notes, job.box.names,
    ))]


def open_saved_job(job_id: str, atlas_loader: Callable[[str], Any] | None) -> Job:
    folder, record = load_job(job_id)
    if record["kind"] == "folder":
        return open_folder_job(job_id, folder, record, atlas_loader)
    prepared = prepare_linear(record["params"])
    if prepared.spec.has("nonlinear"):
        raise ValueError("Image generation is unavailable in Claude mode")
    prepared.spec.out = str(folder / "linear_results.json")
    job = open_job(prepared.spec, atlas_loader)
    job.job_id, job.job_dir, job.prepared = job_id, folder, prepared
    job.notes = record.get("notes", "")
    job.ctx.checkpoint_path = str(folder / "linear_state.json")
    save_checkpoint(job.state, job.ctx.checkpoint_path)
    trace_dir = prepared.trace_dir
    if trace_dir:
        job.trace = McpTrace(trace_dir, job.ctx.image_folder)
    job.channel = HostChannel(job_id, record.get("host_channel"))
    job.checkpoint = checkpoint_callback(prepared, job.channel.event)
    job.checkpoint(job.state)
    return job


def open_folder_job(
    job_id: str, folder: Path, record: dict[str, Any],
    atlas_loader: Callable[[str], Any] | None,
) -> Job:
    """A plain-folder job: its checkpoint and results live in the job directory.

    Reopening it (a restarted server, a new chat) resumes from that checkpoint.
    """
    spec = JobSpec.from_dict(record["spec"])
    spec.resume = True
    spec.out = str(folder / "linear_results.json")
    job = open_job(spec, atlas_loader, checkpoint_path=str(folder / "linear_state.json"))
    job.job_id, job.job_dir = job_id, folder
    job.notes = record.get("notes", "")
    if record.get("trace_dir"):
        job.trace = McpTrace(record["trace_dir"], job.ctx.image_folder)
    return job


def finish(job: Job) -> None:
    if job.checkpoint is not None:
        job.checkpoint(job.state)
    emit_results(job.state, job.ctx)
    if job.job_dir is not None and job.prepared is not None:
        result = {"state": job.state.to_dict(), "output_dir": str(job.job_dir),
                  "final_updates": job.prepared.final_updates}
        (job.job_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if job.channel is not None:
            job.channel.send({"type": "result", "result": result})
            job.channel.close()


def host_tool(job: Job, tool: Callable[..., Any]) -> Callable[..., Any]:
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
        job.box.begin_model_call()
        try:
            def invoke() -> Any:
                if job.checkpoint is None:
                    result = tool(**kwargs)
                else:
                    with observe_checkpoints(job.checkpoint):
                        result = tool(**kwargs)
                if tool.__name__ == "submit" and job.state.submitted:
                    finish(job)
                return result

            def serialized() -> Any:
                with job.lock:
                    return invoke()

            result = await to_thread.run_sync(serialized)
        except Exception as exc:
            logger.exception("Tool %s failed", tool.__name__)
            result = {"status": "error", "error": type(exc).__name__, "message": str(exc)}
        blocks = result_blocks(result)
        if job.trace is not None:
            job.trace.write("tool_result", name=tool.__name__, args=kwargs,
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
    first applies :func:`langslice.linear.arguments.argument_refusal` (the
    same rule the ADK plugin and the toolbox apply) to the arguments as sent,
    after FastMCP's JSON pre-parse of string-encoded objects.
    """
    from langslice.linear.arguments import argument_refusal

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
            refusal = argument_refusal(tool, self.pre_parse_json(arguments_to_validate))
            if refusal is not None:
                return result_blocks(refusal)
            return await super().call_fn_with_arg_validation(
                fn, fn_is_async, arguments_to_validate, arguments_to_pass_directly)

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
    current: dict[str, Job] = {}

    def install(job: Job) -> None:
        old = current.get("job")
        if old is not None:
            if old.channel is not None:
                old.channel.close()
            server.remove_tool("show_stack")
            for name in old.box.names:
                server.remove_tool(name)
        for tool in job.box.tools:
            server.add_tool(
                host_tool(job, tool),
                annotations=ToolAnnotations(readOnlyHint=tool.__name__ in READ_ONLY_TOOLS),
                structured_output=False,
            )
            strict_arguments(server, tool.__name__, tool)
        async def show_stack(page: int) -> list[ContentBlock]:
            """Read an opening-picture page (1-based); read every page before writes."""
            if not job.pages:
                job.pages = await to_thread.run_sync(opening_pages, job)
            if not 1 <= page <= len(job.pages):
                raise ValueError(f"page must be between 1 and {len(job.pages)}")
            blocks = job.pages[page - 1]
            if job.trace is not None:
                job.trace.write("show_stack", page=page, content=describe_blocks(blocks))
            return blocks

        server.add_tool(show_stack, structured_output=False,
                        annotations=ToolAnnotations(readOnlyHint=True))
        current["job"] = job

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
        job = current.get("job")
        if job_id and (job is None or job.job_id != job_id):
            job = await to_thread.run_sync(open_saved_job, job_id, atlas_loader)
            install(job)
            if ctx is not None:
                await ctx.session.send_tool_list_changed()
        wanted = os.path.abspath(os.path.expanduser(image_folder)) if image_folder else None
        if wanted is not None and (job is None or wanted != job.ctx.image_folder):
            job = await to_thread.run_sync(open_job, spec_for(wanted), atlas_loader)
            install(job)
            if ctx is not None:
                await ctx.session.send_tool_list_changed()
        if job is None:
            return [TextContent(type="text", text=json.dumps({
                "status": "error",
                "error": "NO_FOLDER",
                "message": "Name a saved job: start_job(job_id=...), or an image_folder.",
            }))]
        blocks = await to_thread.run_sync(briefing, job)
        if job.trace is not None:
            job.trace.write("briefing", content=describe_blocks(blocks))
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

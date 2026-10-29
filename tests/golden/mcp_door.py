"""Run the golden scenario's MCP door: a client session with ``langslice mcp``.

The server is the installed ``langslice mcp`` command (the console script
beside this interpreter) over stdio, started with the job flags and no
folder, and the client is the ``mcp`` library's, as a host runs them: it
initialises, lists the tools, opens the job with ``start_job``, lists the
job's tools, reads every ``show_stack`` page, then reads, writes, undoes and
makes calls the tools refuse. Each result is kept block by block: a text
block whole, an image block as its decoded pixels' digest.
``tests/test_golden_outputs.py`` runs it and compares the job folder.

Run it by hand as ``python tests/golden/mcp_door.py IMAGES CALLS_JSON``.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, ImageContent, TextContent
from PIL import Image
from stack import (
    ATLAS,
    BACK,
    CUTTING_ANGLES,
    FRONT,
    HAND_TRANSFORMS,
    MIDDLE,
    MISSING,
    PIXEL_SIZE_UM,
    TASKS,
    ZOOM_BOX,
    picture_digest,
    placements,
)

#: The MCP server as installed: the ``langslice`` console script of this
#: environment.
COMMAND = Path(sys.executable).with_name("langslice")
#: ``langslice mcp`` with the job flags of a folder that holds no job yet.
SERVER_ARGUMENTS = [
    "mcp",
    "--atlas",
    ATLAS,
    "--tasks",
    ",".join(TASKS),
    "--image-provider",
    "none",
    "--pixel-size-um",
    str(PIXEL_SIZE_UM),
]


def block_entry(block: object) -> dict[str, object]:
    """Return what is pinned of one content block.

    Parameters
    ----------
    block
        A content block of a tool result.

    Returns
    -------
    dict
        A text block's ``type`` and ``text``; an image block's ``type``,
        ``mimeType`` and its decoded picture's size and pixel digest; any
        other block as the library dumps it.
    """
    if isinstance(block, TextContent):
        return {"type": block.type, "text": block.text}
    if isinstance(block, ImageContent):
        data = base64.b64decode(block.data)
        with Image.open(io.BytesIO(data)) as image:
            digest = picture_digest(image)
        return {"type": block.type, "mimeType": block.mimeType, **digest}
    return {"type": "other", "dump": repr(block)}


def result_entry(
    name: str, arguments: dict[str, object], result: CallToolResult
) -> dict[str, object]:
    """Return one tool call as it is pinned.

    Parameters
    ----------
    name
        The tool called.
    arguments
        The arguments it was called with.
    result
        What the server answered.

    Returns
    -------
    dict
        ``verb``, ``arguments``, ``is_error`` and ``content``, one entry per
        block.
    """
    return {
        "verb": name,
        "arguments": arguments,
        "is_error": result.isError,
        "content": [block_entry(block) for block in result.content],
    }


async def tool_list(session: ClientSession) -> dict[str, object]:
    """Return the tools the server lists, as a call record.

    Parameters
    ----------
    session
        The initialised client session.

    Returns
    -------
    dict
        ``verb`` ``list_tools`` and per tool its name, description, input
        schema and annotations.
    """
    listed = await session.list_tools()
    tools = [
        tool.model_dump(mode="json", exclude_none=True)
        for tool in listed.tools
    ]
    return {"verb": "list_tools", "tools": tools}


def job_calls() -> list[tuple[str, dict[str, object]]]:
    """Return the tool calls made after the opening pages are read.

    Returns
    -------
    list of tuple
        ``(tool, arguments)`` per call, in order.
    """
    transforms = {"sections": HAND_TRANSFORMS}
    return [
        ("status", {}),
        (
            "position_sections",
            {"sections": placements(), "cutting_angles": CUTTING_ANGLES},
        ),
        ("look", {"mode": "overlay", "sections": [FRONT]}),
        ("zoom", {"box": ZOOM_BOX}),
        ("interactive_transform", transforms),
        ("elastix_affine", {"sections": [MIDDLE]}),
        ("mark_damage", {"section": BACK, "regions": ["CA1:left"]}),
        ("grep_atlas", {"query": "CA1"}),
        ("read_file", {"path": "views.jsonl", "limit": 2}),
        ("note", {"text": "Golden scenario, MCP."}),
        ("undo", {}),
        ("look", {"mode": "overlay", "sections": [MISSING]}),
        ("status", {"bogus": 1}),
        ("start_job", {}),
    ]


async def converse(
    images: Path, errors: io.TextIOBase
) -> list[dict[str, object]]:
    """Hold the client session and record every answer.

    Parameters
    ----------
    images
        The image folder the job is made for.
    errors
        Where the server's stderr goes.

    Returns
    -------
    list of dict
        The session's records: the server's greeting, the tool lists and
        every call.
    """
    # the whole environment: the library's default passes only a few names
    server = StdioServerParameters(
        command=str(COMMAND),
        args=SERVER_ARGUMENTS,
        env=dict(os.environ),
        cwd=str(images.parent),
    )
    records: list[dict[str, object]] = []
    async with stdio_client(server, errlog=errors) as (read, write):
        async with ClientSession(read, write) as session:
            greeting = await session.initialize()
            records.append(
                {
                    "verb": "initialize",
                    "greeting": greeting.model_dump(
                        mode="json", exclude_none=True
                    ),
                }
            )
            records.append(await tool_list(session))
            opened = {"image_folder": str(images)}
            result = await session.call_tool("start_job", opened)
            records.append(result_entry("start_job", opened, result))
            records.append(await tool_list(session))

            # every opening page, then one past the last for its refusal
            page = 1
            while True:
                arguments = {"page": page}
                result = await session.call_tool("show_stack", arguments)
                records.append(result_entry("show_stack", arguments, result))
                if result.isError:
                    break
                page += 1

            for name, arguments in job_calls():
                result = await session.call_tool(name, arguments)
                records.append(result_entry(name, arguments, result))
    return records


def main() -> None:
    """Run the session on the first folder; write its records to the second."""
    images, calls_path = Path(sys.argv[1]), Path(sys.argv[2])
    with (images.parent / "mcp-server.log").open("w") as errors:
        records = anyio.run(converse, images, errors)
    calls_path.write_text(json.dumps(records, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()

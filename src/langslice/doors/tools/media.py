"""Pictures as message images: the one place core pictures become ``types.Part``.

The core (:mod:`langslice.core`,
:mod:`langslice.core.atlas_fetch`, :mod:`langslice.core.opening`) draws
plain PIL images, captions burned in, and the opening as a sequence of texts
and strips. The tools (:mod:`langslice.doors.tools.toolbox`) return those plain
pictures and texts under ``TOOL_MEDIA_PARTS_KEY``. The doors turn them into
what their host reads: the ADK agent through this module (:func:`packaged`
wraps each tool; JPEG ``types.Part``), the MCP server through
:func:`encode_jpeg` straight into MCP image blocks. ``google.genai`` is
ADK's message format, so nothing in the core or the tools imports it.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from google.genai import types
from PIL import Image

from langslice.core.jpeg import JPEG_QUALITY as JPEG_QUALITY
from langslice.core.jpeg import encode_jpeg as encode_jpeg
from langslice.core.opening import opening_items
from langslice.core.state import StackState
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.doors.tools.view_options import image_limit as image_limit
from langslice.doors.tools.view_options import view_edge_limit as view_edge_limit

if TYPE_CHECKING:  # the driver's context imports the toolbox
    from langslice.agent.engine import EngineContext



def image_to_part(img: Image.Image, *, quality: int = JPEG_QUALITY) -> types.Part:
    """One PIL image as a JPEG ``types.Part``."""
    return types.Part.from_bytes(mime_type="image/jpeg", data=encode_jpeg(img, quality=quality))


def items_to_parts(items: Sequence[Any]) -> list[types.Part]:
    """Texts and pictures, in order, as message parts (a part passes through)."""
    return [
        item if isinstance(item, types.Part)
        else types.Part.from_text(text=item) if isinstance(item, str)
        else image_to_part(item)
        for item in items
    ]


# --- tool results ----------------------------------------------------------------


def package_result(result: Any) -> Any:
    """A tool's result as ADK takes it: its media list (plain pictures and
    texts under ``TOOL_MEDIA_PARTS_KEY``) as message parts, in order.

    ADK moves image parts into the function response and drops the key from
    the JSON the model reads (:mod:`langslice.doors.tools`). Everything else is
    returned as it is; a result without a media list is returned unchanged.
    """
    if not isinstance(result, dict):
        return result
    media = result.get(TOOL_MEDIA_PARTS_KEY)
    if not isinstance(media, list):
        return result
    return {**result, TOOL_MEDIA_PARTS_KEY: items_to_parts(media)}


def packaged(tool: Callable[..., Any]) -> Callable[..., Any]:
    """*tool* for the ADK agent: same name, docstring and signature, its
    pictures packaged by :func:`package_result`."""

    @functools.wraps(tool)
    def run(*args: Any, **kwargs: Any) -> Any:
        return package_result(tool(*args, **kwargs))

    return run


def packaged_tools(tools: Sequence[Callable[..., Any]]) -> list[Callable[..., Any]]:
    """Every tool of a toolbox, :func:`packaged`."""
    return [packaged(tool) for tool in tools]


# --- the opening ---------------------------------------------------------------


# --- reply size: a host that caps one reply's bytes ----------------------------------

#: The serialized bytes one MCP reply may hold, its JSON and texts included
#: and every picture base64 (Claude Desktop refuses a reply past ~1 MB).
REPLY_BYTES = 680_000
#: Room a strip's page keeps for its texts and the JSON around them.
TEXT_ROOM = 8_000

#: A reply's content in order: a text, or a picture's JPEG bytes.
Item = str | bytes


def strip_bytes(reply_bytes: int = REPLY_BYTES) -> int:
    """The largest JPEG one opening strip may encode to on a page of
    *reply_bytes* (base64 grows bytes by 4/3), the composition budget of
    :func:`langslice.core.opening.opening_items` (``max_bytes``)."""
    return (reply_bytes - TEXT_ROOM) * 3 // 4


def reply_bytes(items: Sequence[Item]) -> int:
    """*items* serialized as MCP content blocks are (texts, base64 JPEGs)."""
    import base64
    import json

    blocks = [{"type": "text", "text": item} if isinstance(item, str)
              else {"type": "image", "data": base64.b64encode(item).decode("ascii"),
                    "mimeType": "image/jpeg"} for item in items]
    return len(json.dumps(blocks).encode())


def fit_reply(items: Sequence[Item], budget: int = REPLY_BYTES,
              ) -> tuple[list[Item], tuple[int, int] | None]:
    """*items* within *budget* bytes: as they are when they fit, else every
    picture shrunk together (each to 0.8 of its size a round, re-encoded in
    the doors' JPEG) until they do; nothing is dropped. Returns the items
    and, when shrunk, ``(largest long edge before, after)``. ``ValueError``
    when the texts alone pass the budget."""
    import io

    out = list(items)
    if reply_bytes(out) <= budget:
        return out, None

    def edge(item: bytes) -> int:
        with Image.open(io.BytesIO(item)) as image:
            return max(image.size)

    before = max((edge(item) for item in out if isinstance(item, bytes)), default=0)
    while reply_bytes(out) > budget:
        changed = False
        for index, item in enumerate(out):
            if isinstance(item, str):
                continue
            with Image.open(io.BytesIO(item)) as image:
                if max(image.size) <= 32:
                    continue
                picture = image.convert("RGB")
            picture.thumbnail((max(1, int(picture.width * .8)),
                               max(1, int(picture.height * .8))))
            out[index] = encode_jpeg(picture)
            changed = True
        if not changed:
            raise ValueError("The reply's texts alone exceed the host's reply budget")
    after = max((edge(item) for item in out if isinstance(item, bytes)), default=0)
    return out, (before, after)


def shrunk_note(count: int, edges: tuple[int, int], budget: int = REPLY_BYTES) -> str:
    """What a reply whose pictures :func:`fit_reply` shrank says about it."""
    return (f"The {count} picture{'s' if count != 1 else ''} above were shrunk together to fit "
            f"the host's reply limit ({budget // 1000} KB): the largest went from "
            f"{edges[0]} to {edges[1]} px on its long edge. For full-size pictures, ask for "
            "fewer sections per call or a smaller view.resolution.")


def paged(items: Sequence[Item], budget: int = REPLY_BYTES) -> list[list[Item]]:
    """*items* (texts, each picture after the texts that introduce it) on
    pages of at most *budget* bytes. A picture and its texts stay on one
    page; a group past the budget alone is shrunk (:func:`fit_reply`);
    texts after the last picture end the last page."""
    pages: list[list[Item]] = []
    page: list[Item] = []
    pending: list[Item] = []
    for item in items:
        pending.append(item)
        if isinstance(item, str):
            continue
        group, _shrunk = fit_reply(pending, budget)
        if page and reply_bytes(page + group) > budget:
            pages.append(page)
            page = []
        page.extend(group)
        pending = []
    if page or pending:
        pages.append(page + pending)
    return pages


def strip_edge(ctx: EngineContext) -> int:
    """Long edge of each opening strip for this run's model lane."""
    return image_limit(ctx)[0]


def opening_parts(
    state: StackState, ctx: EngineContext, *, limit: tuple[int, int] | None = None,
) -> list[types.Part]:
    """The opening strips (:func:`langslice.core.opening.opening_items`) as
    message parts; *limit* None is this run's model lane (:func:`image_limit`)."""
    return items_to_parts(opening_items(state, ctx, limit=limit or image_limit(ctx)))

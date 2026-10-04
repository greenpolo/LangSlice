"""Pictures as message images: the one place core pictures become ``types.Part``.

The core (:mod:`langslice.core`, :mod:`langslice.linear.render`,
:mod:`langslice.linear.atlas_fetch`, :mod:`langslice.linear.opening`) draws
plain PIL images, captions burned in, and the opening as a sequence of texts
and strips. The tools (:mod:`langslice.linear.toolbox`) return those plain
pictures and texts under ``TOOL_MEDIA_PARTS_KEY``. The doors turn them into
what their host reads: the ADK agent through this module (:func:`packaged`
wraps each tool; JPEG ``types.Part``), the MCP server through
:func:`encode_jpeg` straight into MCP image blocks. ``google.genai`` is
ADK's message format, so nothing in the core or the tools imports it.
"""

from __future__ import annotations

import functools
import io
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from google.genai import types
from PIL import Image

from langslice.adk import TOOL_MEDIA_PARTS_KEY
from langslice.linear.opening import opening_items
from langslice.linear.state import StackState
from langslice.linear.view_options import image_limit as image_limit
from langslice.linear.view_options import view_edge_limit as view_edge_limit

if TYPE_CHECKING:  # the driver's context imports the toolbox
    from langslice.linear.engine import EngineContext

#: JPEG quality of every picture a tool or the opening sends.
JPEG_QUALITY = 85


def encode_jpeg(img: Image.Image, *, quality: int = JPEG_QUALITY) -> bytes:
    """One PIL image as JPEG bytes (RGB), the encoding every door sends."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


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
    the JSON the model reads (:mod:`langslice.adk`). Everything else is
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


def strip_edge(ctx: EngineContext) -> int:
    """Long edge of each opening strip for this run's model lane."""
    return image_limit(ctx)[0]


def opening_parts(
    state: StackState, ctx: EngineContext, *, limit: tuple[int, int] | None = None,
) -> list[types.Part]:
    """The opening strips (:func:`langslice.linear.opening.opening_items`) as
    message parts; *limit* None is this run's model lane (:func:`image_limit`)."""
    return items_to_parts(opening_items(state, ctx, limit=limit or image_limit(ctx)))

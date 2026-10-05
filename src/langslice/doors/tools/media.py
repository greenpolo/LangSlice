"""Pictures as message images: the one place core pictures become ``types.Part``.

The core (:mod:`langslice.core`,
:mod:`langslice.core.atlas_fetch`, :mod:`langslice.core.opening`) draws
plain PIL images, captions burned in, and the opening as a sequence of texts
and strips. The tools (:mod:`langslice.doors.tools.toolbox`) return those plain
pictures and texts under ``TOOL_MEDIA_PARTS_KEY``. The doors turn them into
what their host reads: the ADK agent through this module (:func:`packaged`
wraps each tool; JPEG ``types.Part``), the MCP server through
:func:`langslice.core.jpeg.encode_jpeg` straight into MCP image blocks
(within :mod:`langslice.doors.tools.reply`'s byte budget). ``google.genai``
is ADK's message format, so nothing else in the core or the doors imports it.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from google.genai import types
from PIL import Image

from langslice.core.jpeg import JPEG_QUALITY, encode_jpeg
from langslice.core.opening import opening_items
from langslice.core.state import StackState
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.doors.tools.view_options import image_limit

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


def opening_parts(
    state: StackState, ctx: EngineContext, *, limit: tuple[int, int] | None = None,
) -> list[types.Part]:
    """The opening strips (:func:`langslice.core.opening.opening_items`) as
    message parts; *limit* None is this run's model lane (:func:`image_limit`)."""
    return items_to_parts(opening_items(state, ctx, limit=limit or image_limit(ctx)))

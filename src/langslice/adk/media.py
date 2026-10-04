"""Pictures as message images: the one place core pictures become ``types.Part``.

The core (:mod:`langslice.linear.render`, :mod:`langslice.linear.atlas_fetch`,
:mod:`langslice.linear.opening`) draws plain PIL images, captions burned in,
and the opening as a sequence of texts and strips. The doors turn them into
what their host reads: the ADK toolbox and seed message through this module
(JPEG ``types.Part``), the MCP server through :func:`encode_jpeg` straight
into MCP image blocks. ``google.genai`` is ADK's message format, so nothing
in the core imports it.

The encoded section and atlas pictures the comparison tools send are cached
on the driver's context (``EngineContext.reference_parts``), so a picture
asked for again is not re-encoded.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import TYPE_CHECKING

from google.genai import types
from PIL import Image

from langslice.linear.appearance import Look
from langslice.linear.atlas_fetch import atlas_picture
from langslice.linear.opening import DEFAULT_IMAGE_LIMIT, IMAGE_LIMITS, opening_items
from langslice.linear.render import (
    opening_edge,
    picture_edge,
    reference_slice_picture,
    render_cache_key,
)
from langslice.linear.state import SliceState, StackState
from langslice.providers.registry import canonical_provider

if TYPE_CHECKING:  # the driver's context imports the toolbox, which imports this
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


def items_to_parts(items: Sequence[str | Image.Image]) -> list[types.Part]:
    """Texts and pictures, in order, as message parts."""
    return [
        types.Part.from_text(text=item) if isinstance(item, str) else image_to_part(item)
        for item in items
    ]


def reference_slice_part(
    ctx: EngineContext, record: SliceState, *, long_edge: int | None = None,
    look: Look = None,
) -> types.Part:
    """:func:`langslice.linear.render.reference_slice_picture`, encoded and
    cached per display state.

    The cached caption retains the index/flags at first display; current
    state is carried separately in tool text. *long_edge* None is the run's
    opening size; another size is its own entry.
    """
    long_edge = long_edge or opening_edge(ctx)
    key = ("section", *render_cache_key(ctx, record, long_edge=long_edge, frame=True, look=look))
    if key not in ctx.reference_parts:
        ctx.reference_parts[key] = image_to_part(
            reference_slice_picture(ctx, record, long_edge=long_edge, look=look)
        )
    return ctx.reference_parts[key].model_copy(deep=True)


def atlas_part(
    ctx: EngineContext, state: StackState, position_mm: float, *,
    long_edge: int | None = None, prepared: Image.Image | None = None,
) -> types.Part:
    """:func:`langslice.linear.atlas_fetch.atlas_picture`, encoded and cached
    by position, plane, angles and size."""
    long_edge = long_edge or picture_edge(ctx)
    key = ("atlas", state.plane, float(position_mm), state.pitch_deg, state.yaw_deg,
           int(long_edge))
    if key in ctx.reference_parts:
        return ctx.reference_parts[key].model_copy(deep=True)
    part = image_to_part(
        atlas_picture(ctx, state, position_mm, long_edge=long_edge, prepared=prepared)
    )
    ctx.reference_parts[key] = part
    return part.model_copy(deep=True)


# --- the opening ---------------------------------------------------------------


def image_limit(ctx: EngineContext) -> tuple[int, int]:
    """``(long edge, patch budget)`` of one image for this run's model lane
    (:data:`langslice.linear.opening.IMAGE_LIMITS`)."""
    model = str(getattr(ctx, "model", "") or "")
    provider = canonical_provider(model.split("/", 1)[0]) if "/" in model else ""
    return IMAGE_LIMITS.get(provider, DEFAULT_IMAGE_LIMIT)


def strip_edge(ctx: EngineContext) -> int:
    """Long edge of each opening strip for this run's model lane."""
    return image_limit(ctx)[0]


def opening_parts(
    state: StackState, ctx: EngineContext, *, limit: tuple[int, int] | None = None,
) -> list[types.Part]:
    """The opening strips (:func:`langslice.linear.opening.opening_items`) as
    message parts; *limit* None is this run's model lane (:func:`image_limit`)."""
    return items_to_parts(opening_items(state, ctx, limit=limit or image_limit(ctx)))

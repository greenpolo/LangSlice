"""Opening pages and reply blocks within the MCP reply budget."""

from __future__ import annotations

import base64
import json
import logging
from io import BytesIO
from typing import TYPE_CHECKING, Any

from mcp.types import ContentBlock, ImageContent, TextContent
from PIL import Image

from langslice.core.geometry.pixel_frames import resized_recipe
from langslice.core.pictures.drawn import PictureNote
from langslice.core.pictures.jpeg import encode_jpeg
from langslice.core.pictures.opening import (
    CLAUDE_IMAGE_LIMIT,
    opening_items,
)
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.job.job import Job
from langslice.tools.reply import (
    REPLY_BYTES,
    Item,
    fit_reply,
    paged,
    shrunk_note,
    strip_bytes,
)

if TYPE_CHECKING:
    from langslice.doors.mcp.server import Session

logger = logging.getLogger(__name__)


# --- content conversion ----------------------------------------------------


def image_block(image: Image.Image | bytes) -> ImageContent:
    """One core picture (or its JPEG bytes) as an MCP image block (the doors'
    JPEG encoding)."""
    data = image if isinstance(image, bytes) else encode_jpeg(image)
    return ImageContent(
        type="image",
        data=base64.b64encode(data).decode("ascii"),
        mimeType="image/jpeg",
    )


def blocks_of(items: list[Item]) -> list[ContentBlock]:
    """Texts and JPEG bytes (``reply.Item``) as MCP blocks."""
    return [
        TextContent(type="text", text=item)
        if isinstance(item, str)
        else image_block(item)
        for item in items
    ]


def result_blocks(
    result: Any, *, job: Job | None = None
) -> list[ContentBlock]:
    """A toolbox result as MCP content: its JSON, then its media in order.

    The tools return plain pictures (PIL images, captions burned in) and lines
    of text under ``TOOL_MEDIA_PARTS_KEY``; each picture becomes an image block
    in the doors' JPEG encoding (:func:`image_block`), each non-empty text a
    text block. ``images_attached`` counts both. The whole reply stays within
    the host's reply budget (:func:`langslice.tools.reply.fit_reply`): past it
    every picture is shrunk together and a last text says so and how to get
    full-size ones.
    """
    media: list[Any] = []
    if isinstance(result, dict):
        body = dict(result)
        listed = body.pop(TOOL_MEDIA_PARTS_KEY, None)
        if isinstance(listed, list):
            media = [
                item for item in listed if isinstance(item, (str, Image.Image))
            ]
            body["images_attached"] = len(media)
        elif listed is not None:
            # Already text (e.g. a note that the pictures were not produced).
            body[TOOL_MEDIA_PARTS_KEY] = listed
    else:
        body = result
    items: list[Item] = [json.dumps(body, default=str)]
    items += [
        item if isinstance(item, str) else encode_jpeg(item)
        for item in media
        if not isinstance(item, str) or item
    ]
    fitted, shrunk = fit_reply(items)
    if shrunk is not None:
        if job is not None and isinstance(body, dict):
            # Each delivered size gets its own immutable picture number. The
            # old saved picture remains valid for callers who already saw it.
            refs = body.get("pictures") or []
            indexes = [
                i for i, item in enumerate(fitted) if isinstance(item, bytes)
            ]
            replacements: dict[int, int] = {}
            for ref, index in zip(refs, indexes, strict=False):
                old = job.views.lookup(ref["id"])
                if old is None:
                    raise ValueError(
                        f"Returned picture {ref['id']} is not saved"
                    )
                original, delivered = items[index], fitted[index]
                assert isinstance(original, bytes) and isinstance(
                    delivered, bytes
                )
                with (
                    Image.open(BytesIO(original)) as source,
                    Image.open(BytesIO(delivered)) as target,
                ):
                    recipe = resized_recipe(
                        old.recipe, source.size, target.size
                    )
                held = PictureNote(
                    sections=old.sections,
                    mode=old.mode,
                    recipe=recipe,
                    caption=old.caption,
                    extra={"transport_source_picture": old.seq},
                )
                names = job.views.save(
                    tool=old.tool,
                    pictures=[(delivered, held)],
                    arguments={"transport_source_picture": old.seq},
                )
                replacements[old.seq] = int(names[0].split("_", 1)[0])

            def picture_entries(
                entries: list[dict[str, Any]],
            ) -> list[dict[str, Any]]:
                return [
                    {**entry, "id": replacements.get(entry["id"], entry["id"])}
                    for entry in entries
                ]

            if refs:
                body["pictures"] = picture_entries(refs)
            if body.get("results"):
                body["results"] = [
                    {**one, "pictures": picture_entries(one["pictures"])}
                    if one.get("pictures")
                    else one
                    for one in body["results"]
                ]
            fitted[0] = json.dumps(body, default=str)
        fitted.append(
            shrunk_note(
                sum(isinstance(item, bytes) for item in fitted), shrunk
            )
        )
    return blocks_of(fitted)


# --- trace -----------------------------------------------------------------


def describe_blocks(blocks: list[ContentBlock]) -> list[dict[str, Any]]:
    described: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, ImageContent):
            described.append(
                {
                    "image": block.mimeType,
                    "bytes": len(base64.b64decode(block.data)),
                }
            )
        elif isinstance(block, TextContent):
            described.append({"text": block.text})
    return described


#: The serialized bytes of one ``show_stack`` page, JSON and base64 included
#: (every MCP reply's budget, :data:`langslice.tools.reply.REPLY_BYTES`).
PAGE_BYTES = REPLY_BYTES


def opening_pages(session: Session) -> list[list[ContentBlock]]:
    """The opening strips (:mod:`langslice.core.pictures.opening`) at Claude's
    image
    size, each strip composed within a page's byte budget (fewer sections per
    strip, never a shrunk strip: :func:`langslice.tools.reply.strip_bytes`),
    paged under :data:`PAGE_BYTES` (:func:`langslice.tools.reply.paged`);
    a strip and its text stay together."""
    items = opening_items(
        session.state,
        session.ctx,
        limit=CLAUDE_IMAGE_LIMIT,
        max_bytes=strip_bytes(PAGE_BYTES),
    )
    encoded: list[Item] = [
        item if isinstance(item, str) else encode_jpeg(item) for item in items
    ]
    return [blocks_of(page) for page in paged(encoded, PAGE_BYTES)]


def save_page(
    session: Session, page: int, blocks: list[ContentBlock]
) -> list[str]:
    """Save the page's pictures in the job folder, as the bytes the host got;
    return their paths (written in the background)."""
    from langslice.job.pictures import PICTURE_FILE, captured

    pictures = [
        base64.b64decode(block.data)
        for block in blocks
        if isinstance(block, ImageContent)
    ]
    try:
        with captured() as saved:
            session.job.views.save(
                tool="show_stack",
                arguments={"page": page},
                pictures=[(data, None) for data in pictures],
            )
        return [str(item.folder / PICTURE_FILE) for item in saved]
    except Exception:  # saving must never break a page
        logger.warning(
            "Could not queue the pictures of show_stack page %s",
            page,
            exc_info=True,
        )
        return []

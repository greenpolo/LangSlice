"""The reply byte budget of a host that caps one reply, and pictures fitted to it.

The MCP door holds every reply within :data:`REPLY_BYTES` (Claude Desktop
refuses a reply past about 1 MB): :func:`fit_reply` shrinks a reply's
pictures together, :func:`paged` spreads the opening over pages, and
:func:`strip_bytes` is the opening strips' composition budget. No model
framework is imported here.
"""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image

from langslice.core.jpeg import encode_jpeg

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
            "fewer sections per call or a smaller resolution.")


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

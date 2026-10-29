"""The library's reply shape."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from langslice.core.jsonable import Json, to_jsonable


class Reply(dict):  # type: ignore[type-arg]
    """A verb's reply: a plain dict of JSON values (``json.dumps(reply)``
    works), its pictures' files under ``artifacts``; the pictures
    themselves, as PIL images in the order a model receives them, on
    :attr:`images`, which is not a key."""

    images: list[Any]

    def __init__(self, body: Mapping[str, Any] | None = None,
                 images: Sequence[Any] = ()) -> None:
        super().__init__(body or {})
        self.images = list(images)


def library_reply(verb: str, reply: Any, saved: Sequence[Any]) -> Reply:
    """A tool's reply for a script: the pictures out of it (their files
    under ``artifacts``, :func:`langslice.job.pictures.artifacts`; the images
    on :attr:`Reply.images`; their lines of text under ``media_texts``), the
    status rows uniform
    (:func:`langslice.core.stack.status.with_uniform_rows`),
    everything else JSON values
    (:func:`~langslice.core.jsonable.to_jsonable`); a picture that could
    not be saved is a line under ``warnings``."""
    from langslice.core.stack.status import with_uniform_rows
    from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
    from langslice.job.pictures import artifacts

    body = dict(reply) if isinstance(reply, Mapping) else {"result": reply}
    media = body.pop(TOOL_MEDIA_PARTS_KEY, None)
    media = media if isinstance(media, list) else []
    texts = [item for item in media if isinstance(item, str)]
    if texts:
        body["media_texts"] = texts
    # a mapping in, a mapping out: the rows made uniform, the rest kept
    uniform = cast("Mapping[str, Json]", with_uniform_rows(body))
    out: dict[str, object] = {**to_jsonable(uniform)}
    out["artifacts"], unsaved = artifacts(saved, verb)
    if unsaved:
        out["warnings"] = unsaved
    return Reply(out, [item for item in media if not isinstance(item, str)])

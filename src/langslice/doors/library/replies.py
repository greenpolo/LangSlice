"""The library's reply shape."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


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


def plain(value: Any) -> Any:
    """*value* as JSON values: mappings with text keys, lists, numbers,
    text, booleans and null; a numpy value as its Python value, anything
    else as its text (as the agent CLI prints it)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=str) if isinstance(value, (set, frozenset)) else value
        return [plain(item) for item in items]
    import numpy as np

    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    return str(value)


def library_reply(verb: str, reply: Any, saved: Sequence[Any]) -> Reply:
    """A tool's reply for a script: the pictures out of it (their files
    under ``artifacts``, :func:`langslice.job.pictures.artifacts`; the images
    on :attr:`Reply.images`; their lines of text under ``media_texts``), the
    status rows uniform
    (:func:`langslice.core.stack.status.with_uniform_rows`),
    everything else JSON values (:func:`plain`); a picture that could not be
    saved is a line under ``warnings``."""
    from langslice.core.stack.status import with_uniform_rows
    from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
    from langslice.job.pictures import artifacts

    body = dict(reply) if isinstance(reply, Mapping) else {"result": reply}
    media = body.pop(TOOL_MEDIA_PARTS_KEY, None)
    media = media if isinstance(media, list) else []
    texts = [item for item in media if isinstance(item, str)]
    if texts:
        body["media_texts"] = texts
    out = plain(with_uniform_rows(body))
    out["artifacts"], unsaved = artifacts(saved, verb)
    if unsaved:
        out["warnings"] = unsaved
    return Reply(out, [item for item in media if not isinstance(item, str)])

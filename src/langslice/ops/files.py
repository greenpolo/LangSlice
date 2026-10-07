"""Read-only access to the job folder for an agent that has no shell of its own.

Three functions modelled on a coding agent's file tools: :func:`list_files`,
:func:`search_files` and :func:`read_file`. Every path is relative to the job
folder and kept inside it (:mod:`langslice.job.browse`); a path that leaves it
is refused (``BAD_PATH``). Each answers a small record whose ``text`` is the
model-facing reply, capped (``browse.LIST_ENTRIES``, ``SEARCH_MATCHES``,
``READ_LINES``, ``TEXT_BYTES``) with a line saying what was left out.

What is worth opening: ``views.jsonl`` (one JSON line per picture the model
was shown: number, tool, sections, caption, step), ``state.json`` (the whole
state, its run ``notes`` included), ``job.json`` (the settings and the user's
notes), ``registration.json``, ``history/``, ``exports/`` and ``logs/``.
A picture file is not returned as pixels: it is answered with its index record
and the way to see it again (``zoom`` or ``look`` by its number).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from langslice.job import browse
from langslice.job.browse import PathRefused
from langslice.ops.refusal import Refused

if TYPE_CHECKING:
    from langslice.job.job import Job
    from langslice.job.views import PictureRecord


@dataclass(frozen=True)
class Listing:
    """The items of :func:`list_files`; ``text`` is the reply."""

    path: str
    entries: tuple[browse.Entry, ...]
    hidden: int
    text: str


@dataclass(frozen=True)
class Found:
    """The matches of :func:`search_files`; ``text`` is the reply."""

    matches: tuple[str, ...]
    hidden: int
    files_searched: int
    text: str


@dataclass(frozen=True)
class FileText:
    """What :func:`read_file` answers; ``text`` is the reply.

    ``kind`` is ``text``, ``picture`` or ``binary``; ``picture`` is the number
    of the indexed picture a picture file belongs to (None when it has none).
    """

    path: str
    kind: str
    text: str
    first_line: int = 0
    last_line: int = 0
    total_lines: int = 0
    picture: int | None = None


def _folder(job: Job) -> Path:
    return job.layout.folder


def _resolved(job: Job, path: str) -> Path:
    try:
        return browse.resolve(_folder(job), path)
    except PathRefused as error:
        raise Refused("BAD_PATH", message=str(error)) from error


def _size(size: int) -> str:
    if size < 10_000:
        return f"{size} B"
    if size < 10_000_000:
        return f"{size / 1000:.0f} kB"
    return f"{size / 1_000_000:.1f} MB"


def list_files(job: Job, path: str = ".", pattern: str = "") -> Listing:
    """The files and folders at *path* (folders first), each with its kind and size.

    With a glob *pattern* (``*.json``, ``views/*``), every file below *path*
    whose name or path matches. At most ``browse.LIST_ENTRIES`` are shown; the
    reply ends with ``N more`` for the rest.
    """
    start = _resolved(job, path)
    entries, hidden = browse.list_entries(_folder(job), start, str(pattern or ""))
    shown = browse.relative(_folder(job), start)
    rows = [f"{e.path}/" if e.kind == "dir" else f"{e.path}  {e.kind}, {_size(e.size)}"
            for e in entries]
    if not rows:
        rows = ["(nothing)"]
    if hidden:
        rows.append(f"... {hidden} more (narrow the path or the pattern)")
    head = f"{shown}:" if start.is_dir() else f"{shown} (a file):"
    return Listing(shown, tuple(entries), hidden, "\n".join([head, *rows]))


def search_files(job: Job, query: str, path: str = ".", glob: str = "") -> Found:
    """Lines matching *query* (a regular expression, or plain text when it is not
    one; upper and lower case alike) in the text files under *path*.

    Each match is ``file:line: text``. At most ``browse.SEARCH_MATCHES`` are
    shown; the reply says how many more there are. Pictures and binary files are
    not searched. *glob* keeps files whose name or path matches it.
    """
    if not str(query or ""):
        raise Refused("BAD_ARGS", message="query is empty.")
    start = _resolved(job, path)
    lines, hidden, files = browse.search(_folder(job), start, str(query), str(glob or ""))
    rows = list(lines) or [f"No match in {files} file(s)."]
    if hidden:
        rows.append(f"... {hidden} more match(es) not shown (narrow the path or the glob, "
                    "or make the query more specific)")
    return Found(tuple(lines), hidden, files, "\n".join(rows))


def _picture_record(job: Job, real: Path) -> PictureRecord | None:
    """The indexed picture whose folder holds *real* (or is *real*)."""
    root = browse.root_of(_folder(job))
    try:
        parent = real.parent.relative_to(root).as_posix()
        own = real.relative_to(root).as_posix()
    except ValueError:
        return None
    for record in job.views.records():
        folder = record.path.rstrip("/")
        if folder and folder in (parent, own):
            return record
    return None


def _record_text(record: PictureRecord) -> list[str]:
    lines = [f"Picture {record.seq}: {record.name or record.tool}"]
    lines.append(f"tool: {record.tool}" + (f", mode: {record.mode}" if record.mode else "")
                 + f", call {record.call}")
    if record.sections:
        lines.append("sections: " + ", ".join(record.sections))
    if record.caption:
        lines.append(f"caption: {record.caption}")
    if record.step is not None:
        lines.append(f"step: {record.step}")
    if record.recipe:
        lines.append("recipe: " + json.dumps(record.recipe, separators=(",", ":"), default=str))
    if record.layers:
        lines.append("it has label and border layers (labels.tif, borders.png) in its folder")
    if record.path:
        lines.append(f"folder: {record.path}")
    return lines


def _dimensions(real: Path) -> str:
    try:
        with Image.open(real) as image:
            return f"{image.width} x {image.height} pixels"
    except Exception:  # noqa: BLE001 - an unreadable image is described by size alone
        return "an image"


def read_file(job: Job, path: str, offset: int = 0, limit: int = browse.READ_LINES) -> FileText:
    """A text file with line numbers, *limit* lines from line *offset* + 1.

    The reply says which lines it shows, the file's line count and, when lines
    remain, the offset to continue from. A picture file is not returned as
    pixels: its reply is its picture-index record (number, tool, sections,
    caption, step) and says to see it again with ``zoom`` or ``look`` by that
    number. A binary file is described by its size.
    """
    real = _resolved(job, path)
    shown = browse.relative(_folder(job), real)
    if real.is_dir():
        raise Refused("BAD_PATH", message=f"'{shown}' is a folder; list it with list_files.")
    kind = browse.kind_of(real)
    if kind == "picture":
        record = _picture_record(job, real)
        facts = f"{_dimensions(real)}, {_size(real.stat().st_size)}"
        if record is None:
            return FileText(shown, kind, f"{shown} is a picture file ({facts}) with no entry in "
                            "the picture index. Pixels are not returned by this tool.")
        body = [f"{shown} is a picture file ({facts}); pixels are not returned here.",
                *_record_text(record),
                f"To see it again, call zoom or look with picture number {record.seq}."]
        return FileText(shown, kind, "\n".join(body), picture=record.seq)
    if kind == "binary":
        return FileText(shown, kind, f"{shown} is a binary file ({_size(real.stat().st_size)}); "
                        "its contents are not shown.")
    try:
        numbered, total, more = browse.read_lines(real, offset, limit)
    except OSError as error:
        raise Refused("BAD_PATH", message=f"'{shown}' cannot be read: {error}") from error
    start = max(0, int(offset))
    if not numbered:
        text = (f"{shown} has {total} line(s); nothing at offset {start}." if total
                else f"{shown} is empty.")
        return FileText(shown, kind, text, total_lines=total)
    first = start + 1
    last = start + len(numbered)
    rows = [f"{shown}: lines {first}-{last} of {total}", *numbered]
    if more:
        rows.append(f"... {total - last} more line(s); continue with offset={last}")
    return FileText(shown, kind, "\n".join(rows), first, last, total)


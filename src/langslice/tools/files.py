"""Read-only access to the job folder, for an agent with no shell of its own.

Three tools modelled on a coding agent's file tools: :func:`list_files`,
:func:`search_files` and :func:`read_file`, each over its operation on the
job (:func:`list_folder`, :func:`search_folder`, :func:`read_job_file`).
Every picture shown so far is written and indexed before each runs. Every
path is relative to the job folder and kept inside it
(:mod:`langslice.job.browse`); a path that leaves it is refused
(``BAD_PATH``), and one that does not exist is ``NOT_FOUND``, whose message
names the nearest folder above it that exists and what it holds. Each
operation answers a small record whose ``text`` is the model-facing reply, capped
(``browse.LIST_ENTRIES``, ``SEARCH_MATCHES``, ``READ_LINES``, ``TEXT_BYTES``)
with a line saying what was left out.

What is worth opening: ``views.jsonl`` (one JSON line per picture the model
was shown: number, tool, sections, caption, step), ``state.json`` (the whole
state, its run ``notes`` included), ``job.json`` (the settings and the user's
notes), ``registration.json``, ``history/``, ``exports/`` and ``logs/``.
A picture file is not returned as pixels: it is answered with its index record
and the way to see it again (``zoom`` by its number; where the job has no
``zoom``, the ``look`` call that draws a ``look`` picture's view again).
``trace/``, the agent transcripts, is neither listed, searched nor read
(:data:`langslice.job.browse.HIDDEN`).
"""

from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from langslice.core.pictures.look import RENDERER as LOOK_RENDERER
from langslice.core.pictures.modes import LOOK_MODES
from langslice.core.refusal import RefusalCode, Refused
from langslice.job import browse
from langslice.job.settings.needs import render

if TYPE_CHECKING:
    from langslice.job.job import Job
    from langslice.job.pictures import PictureRecord
    from langslice.tools.run import Run


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


def list_files(run: Run, path: str, glob: str) -> dict[str, object]:
    """List a folder of the job; return the ``list_files`` reply.

    Parameters
    ----------
    run
        The run the tool is called in.
    path
        A path in the job folder; empty is the folder itself.
    glob
        A pattern every file below *path* is matched against; empty lists
        *path* itself.

    Returns
    -------
    dict
        ``status`` and the listing's ``text``.
    """
    run.job.views.flush()  # every picture shown so far is on disk and indexed
    done = list_folder(run.job, str(path or "."), str(glob or ""))
    return {"status": "ok", "text": done.text}


def search_files(
    run: Run, query: str, path: str, glob: str
) -> dict[str, object]:
    """Search the job's text files; return the ``search_files`` reply.

    Parameters
    ----------
    run
        The run the tool is called in.
    query
        A regular expression, or plain text when it is not one.
    path
        The folder or file searched; empty is the job folder.
    glob
        A pattern the files searched must match; empty keeps every file.

    Returns
    -------
    dict
        ``status`` and the matches' ``text``.
    """
    run.job.views.flush()
    done = search_folder(
        run.job, str(query or ""), str(path or "."), str(glob or "")
    )
    return {"status": "ok", "text": done.text}


def read_file(
    run: Run, path: str, offset: int, limit: int
) -> dict[str, object]:
    """Read a file of the job; return the ``read_file`` reply.

    Parameters
    ----------
    run
        The run the tool is called in.
    path
        The file, in the job folder.
    offset
        The lines skipped before the first line shown.
    limit
        The most lines shown; 0 is the default.

    Returns
    -------
    dict
        ``status``, the ``text`` and, for a picture file, its ``picture``
        number.

    Raises
    ------
    Refused
        ``BAD_ARGS`` when *offset* or *limit* is not a whole number.
    """
    run.job.views.flush()
    try:
        done = read_job_file(
            run.job, str(path or ""), int(offset or 0), int(limit or 400)
        )
    except Refused:
        raise
    except (TypeError, ValueError):
        raise Refused(
            RefusalCode.BAD_ARGS, "offset and limit are whole numbers."
        ) from None
    reply: dict[str, object] = {"status": "ok", "text": done.text}
    if done.picture is not None:
        reply["picture"] = done.picture
    return reply


def _folder(job: Job) -> Path:
    return job.layout.folder


def _size(size: int) -> str:
    if size < 10_000:
        return f"{size} B"
    if size < 10_000_000:
        return f"{size / 1000:.0f} kB"
    return f"{size / 1_000_000:.1f} MB"


def list_folder(job: Job, path: str = ".", glob: str = "") -> Listing:
    """The files and folders at *path* (folders first), each with its kind and size.

    With a *glob* (``*.json``, ``views/*``; ``*`` is every file), every file
    below *path* whose name or path matches. At most ``browse.LIST_ENTRIES``
    are shown; the reply ends with ``N more`` for the rest.
    """
    start = browse.resolve(_folder(job), path)
    entries, hidden = browse.list_entries(_folder(job), start, str(glob or ""))
    shown = browse.relative(_folder(job), start)
    rows = [f"{e.path}/" if e.kind == "dir" else f"{e.path}  {e.kind}, {_size(e.size)}"
            for e in entries]
    if not rows:
        rows = ["(nothing)"]
    if hidden:
        rows.append(f"... {hidden} more (narrow the path or the glob)")
    head = f"{shown}:" if start.is_dir() else f"{shown} (a file):"
    return Listing(shown, tuple(entries), hidden, "\n".join([head, *rows]))


def search_folder(job: Job, query: str, path: str = ".", glob: str = "") -> Found:
    """Lines matching *query* (a regular expression, or plain text when it is not
    one; upper and lower case alike) in the text files under *path*.

    Each match is ``file:line: text``. At most ``browse.SEARCH_MATCHES`` are
    shown; the reply says how many more there are. Pictures and binary files are
    not searched. *glob* keeps files whose name or path matches it.
    """
    if not str(query or ""):
        raise Refused(RefusalCode.BAD_ARGS, message="query is empty.")
    start = browse.resolve(_folder(job), path)
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


def _look_arguments(record: PictureRecord) -> dict[str, Any] | None:
    """The ``look`` arguments that draw *record*'s view again (from its recipe),
    or None when ``look`` did not draw it."""
    recipe = record.recipe or {}
    args = recipe.get("args") or {}
    if recipe.get("renderer") != LOOK_RENDERER or args.get("mode") not in LOOK_MODES:
        return None
    out: dict[str, Any] = {"mode": args["mode"]}
    for key in ("sections", "positions_mm", "images", "masks"):
        if args.get(key):
            out[key] = list(args[key])
    if args.get("atlas_layers") is not None:
        out["atlas_layers"] = list(args["atlas_layers"])
    if args.get("warp") not in (None, "applied"):
        out["warp"] = args["warp"]
    if args.get("registrations"):
        out["registrations"] = {str(name): int(number)
                                for name, number, _held in args["registrations"]}
    return out


def _see_again(record: PictureRecord, off: Collection[str]) -> str:
    """The reply's line on how to see *record* again, written with its needs
    and rendered for the job's disabled list *off*: ``zoom`` by its number,
    else the ``look`` call that draws its view again, else nothing."""
    redraw = _look_arguments(record)
    zoomed = bool(((record.recipe or {}).get("args") or {}).get("zoom"))
    with_look = "" if redraw is None else (
        f"[[?look:{redraw['mode']} To draw its view again, call look with {{call}} "
        f"(it draws the stack as it is now{', and the whole picture, unzoomed' if zoomed else ''}"
        ").]]")
    text = render(f"[[?zoom To see it again, call zoom with picture {record.seq} and a box "
                  "over the part to see (a box as large as the picture shows all of it)."
                  f"||{with_look}]]", off)
    if redraw is not None:  # the arguments go in after the marks are read
        text = text.replace("{call}", ", ".join(f"{key}={value!r}"
                                                for key, value in redraw.items()))
    return text


def _dimensions(real: Path) -> str:
    try:
        with Image.open(real) as image:
            return f"{image.width} x {image.height} pixels"
    except Exception:  # noqa: BLE001 - an unreadable image is described by size alone
        return "an image"


def read_job_file(
    job: Job, path: str, offset: int = 0, limit: int = browse.READ_LINES
) -> FileText:
    """A text file with line numbers, *limit* lines from line *offset* + 1.

    The reply says which lines it shows, the file's line count and, when lines
    remain, the offset to continue from. A picture file is not returned as
    pixels: its reply is its picture-index record (number, tool, sections,
    caption, step) and says to see it again with ``zoom`` by that
    number, or, where the job has no ``zoom``, the ``look`` call that draws a
    ``look`` picture's view again (on the stack as it is now). A binary file is
    described by its size.
    """
    real = browse.resolve(_folder(job), path)
    shown = browse.relative(_folder(job), real)
    if real.is_dir():
        raise Refused(RefusalCode.BAD_PATH, message=f"'{shown}' is a folder"
                      + ("." if job.spec.off("list_files") else "; list it with list_files."))
    kind = browse.kind_of(real)
    if kind == "picture":
        record = _picture_record(job, real)
        facts = f"{_dimensions(real)}, {_size(real.stat().st_size)}"
        if record is None:
            return FileText(shown, kind, f"{shown} is a picture file ({facts}) with no entry in "
                            "the picture index. Pixels are not returned by this tool.")
        body = [f"{shown} is a picture file ({facts}); pixels are not returned here.",
                *_record_text(record)]
        again = _see_again(record, job.spec.switched_off)
        if again:
            body.append(again)
        return FileText(shown, kind, "\n".join(body), picture=record.seq)
    if kind == "binary":
        return FileText(shown, kind, f"{shown} is a binary file ({_size(real.stat().st_size)}); "
                        "its contents are not shown.")
    try:
        numbered, total, more = browse.read_lines(real, offset, limit)
    except OSError as error:
        raise Refused(RefusalCode.BAD_PATH,
                      message=f"'{shown}' cannot be read: {error}") from error
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

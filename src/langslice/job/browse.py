"""Read-only browsing of the job folder: safe paths, listings, searches, text.

A native agent without a shell of its own looks at the job's files through
these functions (:mod:`langslice.ops.files` wraps them). Everything is kept
inside the job folder:

* a path is taken relative to the folder (an absolute path is accepted only
  when it already lies inside), and is resolved before it is used, so ``..``
  and a symlink pointing out of the folder are both refused
  (:class:`PathRefused`);
* a walk never follows a symlink out of the folder, and skips its target;
* a binary file is described (its kind and size), never dumped.

The caps keep one answer a small part of a tool reply (the reply budget,
:data:`langslice.doors.tools.reply.REPLY_BYTES`, is far larger).
"""

from __future__ import annotations

import fnmatch
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

#: Entries one listing shows; the rest are counted.
LIST_ENTRIES = 200
#: Matches one search shows; the rest are counted.
SEARCH_MATCHES = 60
#: Lines one read returns when no limit is given, and the most it ever returns.
READ_LINES = 400
READ_LINES_MAX = 2000
#: Bytes of text one read or one search reply holds.
TEXT_BYTES = 60_000
#: A line longer than this is cut (said on the line).
LINE_CHARS = 400
#: A file larger than this is not searched.
SEARCH_FILE_BYTES = 8_000_000

PICTURE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"})


class PathRefused(ValueError):
    """A path outside the job folder, or one that cannot be read; the message is plain."""


@dataclass(frozen=True)
class Entry:
    """One listed item: its path relative to the job folder."""

    path: str
    kind: str  # "dir", "text", "picture", "binary"
    size: int


def root_of(folder: Path) -> Path:
    return Path(os.path.realpath(folder))


def resolve(folder: Path, path: str = ".") -> Path:
    """*path* resolved inside *folder*; :class:`PathRefused` when it leaves it
    (``..``, an absolute path elsewhere, a symlink pointing out) or does not exist."""
    root = root_of(folder)
    text = str(path or ".")
    if "\x00" in text:
        raise PathRefused("The path holds a null character.")
    given = Path(text)
    candidate = given if given.is_absolute() else root / given
    resolved = Path(os.path.realpath(candidate))
    if resolved != root and root not in resolved.parents:
        raise PathRefused(f"'{text}' is outside the job folder; only files inside it can be read.")
    if not resolved.exists():
        raise PathRefused(f"'{text}' does not exist in the job folder.")
    return resolved


def relative(folder: Path, path: Path) -> str:
    """*path* (a real path inside the folder) as shown to the model: relative, '/' separated."""
    return path.relative_to(root_of(folder)).as_posix() or "."


def is_text(path: Path) -> bool:
    """Whether *path* reads as text: no NUL byte and valid UTF-8 in its first 8 KB."""
    try:
        with path.open("rb") as handle:
            head = handle.read(8192)
    except OSError:
        return False
    if b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as error:
        # A multi-byte character cut by the 8 KB window is not binary.
        if error.start < len(head) - 4:
            return False
    return True


def kind_of(path: Path) -> str:
    if path.is_dir():
        return "dir"
    if path.suffix.lower() in PICTURE_SUFFIXES:
        return "picture"
    return "text" if is_text(path) else "binary"


def _inside(root: Path, path: Path) -> Path | None:
    """The real path of *path* when it stays inside *root*, else None."""
    real = Path(os.path.realpath(path))
    return real if real == root or root in real.parents else None


def walk(folder: Path, start: Path) -> Iterator[Path]:
    """Every file under *start* (sorted, depth first) as a path under the folder,
    skipping symlinks that leave the job folder and never descending a symlinked
    directory."""
    root = root_of(folder)
    for current, dirs, files in os.walk(start, followlinks=False):
        dirs.sort()
        base = Path(current)
        for name in sorted(files):
            path = base / name
            if _inside(root, path) is not None:
                yield path


def _entry(root: Path, path: Path) -> Entry:
    real = Path(os.path.realpath(path))
    kind = kind_of(real)
    size = 0
    if kind != "dir":
        try:
            size = real.stat().st_size
        except OSError:
            pass
    return Entry(Path(os.path.abspath(path)).relative_to(root).as_posix() or ".", kind, size)


def list_entries(folder: Path, start: Path, pattern: str = "",
                 cap: int = LIST_ENTRIES) -> tuple[list[Entry], int]:
    """``(shown, hidden)``: the items under *start* and how many the cap left out.

    Without *pattern*, the items directly in *start* (folders first). With one,
    every file below *start* whose name or relative path matches the glob.
    """
    root = root_of(folder)
    found: list[Entry] = []
    if start.is_file():
        found.append(_entry(root, start))
    elif not pattern:
        for child in sorted(start.iterdir(), key=lambda c: (not c.is_dir(), c.name)):
            if _inside(root, child) is not None:
                found.append(_entry(root, child))
    else:
        for path in walk(folder, start):
            shown = Path(os.path.abspath(path)).relative_to(root).as_posix()
            if fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(shown, pattern):
                found.append(_entry(root, path))
    return found[:cap], max(0, len(found) - cap)


def compile_query(query: str) -> re.Pattern[str]:
    """*query* as a case-insensitive regex; plain text when it is not a valid one."""
    try:
        return re.compile(query, re.IGNORECASE)
    except re.error:
        return re.compile(re.escape(query), re.IGNORECASE)


def cut(line: str) -> str:
    line = line.rstrip("\r\n")
    if len(line) > LINE_CHARS:
        return f"{line[:LINE_CHARS]} ...[line cut, {len(line)} characters]"
    return line


def search(folder: Path, start: Path, query: str, glob: str = "",
           cap: int = SEARCH_MATCHES) -> tuple[list[str], int, int]:
    """``(lines, hidden, files)``: matches as ``path:line: text`` (at most *cap* and
    :data:`TEXT_BYTES`), the matches not shown, and the files searched.

    Binary files, pictures and files over :data:`SEARCH_FILE_BYTES` are skipped.
    """
    pattern = compile_query(query)
    root = root_of(folder)
    candidates = [start] if start.is_file() else walk(folder, start)
    lines: list[str] = []
    hidden = 0
    files = 0
    used = 0
    for path in candidates:
        real = Path(os.path.realpath(path))
        shown = relative(folder, real)
        if glob and not (fnmatch.fnmatch(path.name, glob) or fnmatch.fnmatch(shown, glob)):
            continue
        if real.suffix.lower() in PICTURE_SUFFIXES or not real.is_file():
            continue
        if root != real and root not in real.parents:
            continue
        try:
            if real.stat().st_size > SEARCH_FILE_BYTES or not is_text(real):
                continue
            files += 1
            with real.open("r", encoding="utf-8", errors="replace") as handle:
                for number, text in enumerate(handle, 1):
                    if not pattern.search(text):
                        continue
                    entry = f"{shown}:{number}: {cut(text)}"
                    if len(lines) < cap and used + len(entry) <= TEXT_BYTES:
                        lines.append(entry)
                        used += len(entry) + 1
                    else:
                        hidden += 1
        except OSError:
            continue
    return lines, hidden, files


def read_lines(path: Path, offset: int, limit: int) -> tuple[list[str], int, bool]:
    """``(numbered, total, more)``: the lines after the first *offset* of *path*,
    each as ``"    12\\ttext"``.

    At most *limit* lines and :data:`TEXT_BYTES` of text; *total* is the file's line
    count and *more* whether lines after the last one shown remain.
    """
    limit = max(1, min(int(limit), READ_LINES_MAX))
    offset = max(0, int(offset))
    shown: list[str] = []
    used = 0
    total = 0
    stopped = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for number, text in enumerate(handle, 1):
            total = number
            if number <= offset or stopped:
                continue
            entry = f"{number:>6}\t{cut(text)}"
            if len(shown) >= limit or used + len(entry) > TEXT_BYTES:
                stopped = True
                continue
            shown.append(entry)
            used += len(entry) + 1
    return shown, total, stopped

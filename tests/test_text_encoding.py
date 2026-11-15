"""Every text file is read and written as UTF-8, named in the call.

Without an encoding Python uses the locale's: UTF-8 on Linux and macOS,
cp1252 on most Windows machines, where a "µ" in a reply or a record fails to
write or reads back garbled. So every built-in ``open`` in text mode and
every ``read_text`` / ``write_text`` names its encoding.

And every text file LangSlice writes (``src/``, ``tools/``) names its line
ends: without ``newline`` Windows writes CRLF, so a job's files would differ
byte for byte from the same job written on Linux.

Third-party code vendored as published (``src/langslice_vendor/``) is not
read: LangSlice calls none of its file readers or writers.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FOLDERS = ("src", "tests", "tools")
#: Third-party code kept as published; LangSlice calls none of its file I/O.
VENDORED = REPO / "src" / "langslice_vendor"


def _sources(folder: str) -> list[Path]:
    return [path for path in sorted((REPO / folder).rglob("*.py"))
            if not path.is_relative_to(VENDORED)]


def _text_mode(node: ast.Call) -> bool:
    mode = node.args[1] if len(node.args) > 1 else next(
        (k.value for k in node.keywords if k.arg == "mode"), None)
    if mode is None:
        return True
    return not (isinstance(mode, ast.Constant) and "b" in str(mode.value))


def _missing(node: ast.Call) -> bool:
    if any(k.arg in ("encoding", None) for k in node.keywords):
        return False
    if any(isinstance(a, ast.Starred) for a in node.args):
        return False
    if isinstance(node.func, ast.Name) and node.func.id == "open":
        return len(node.args) < 4 and _text_mode(node)
    if isinstance(node.func, ast.Attribute) and node.func.attr == "read_text":
        return not node.args
    if isinstance(node.func, ast.Attribute) and node.func.attr == "write_text":
        return len(node.args) < 2
    return False


def test_text_files_name_their_encoding():
    found = []
    for folder in FOLDERS:
        for path in _sources(folder):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            found += [f"{path.relative_to(REPO).as_posix()}:{node.lineno}"
                      for node in ast.walk(tree)
                      if isinstance(node, ast.Call) and _missing(node)]
    assert not found, "no encoding named (add encoding=\"utf-8\"):\n" + "\n".join(found)


def _writes_text(node: ast.Call) -> bool:
    """A text write that names no ``newline``: ``write_text``, or ``open``,
    ``os.fdopen`` or ``Path.open`` with a writing text mode."""
    if any(k.arg in ("newline", None) for k in node.keywords):
        return False
    if any(isinstance(a, ast.Starred) for a in node.args):
        return False
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "write_text":
        return True
    if isinstance(func, ast.Name) and func.id == "open":
        at = 1
    elif isinstance(func, ast.Attribute) and func.attr == "fdopen":
        at = 1
    elif isinstance(func, ast.Attribute) and func.attr == "open" and not (
            isinstance(func.value, ast.Name) and func.value.id in ("Image", "zipfile")):
        at = 0
    else:
        return False
    mode = node.args[at] if len(node.args) > at else next(
        (k.value for k in node.keywords if k.arg == "mode"), None)
    if not (isinstance(mode, ast.Constant) and isinstance(mode.value, str)):
        return False
    return "b" not in mode.value and any(c in mode.value for c in "wax+")


def test_text_files_name_their_line_ends():
    found = []
    for folder in ("src", "tools"):
        for path in _sources(folder):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            found += [f"{path.relative_to(REPO).as_posix()}:{node.lineno}"
                      for node in ast.walk(tree)
                      if isinstance(node, ast.Call) and _writes_text(node)]
    assert not found, "no line ends named (add newline=\"\\n\"):\n" + "\n".join(found)

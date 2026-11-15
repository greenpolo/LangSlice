"""Every text file is read and written as UTF-8, named in the call.

Without an encoding Python uses the locale's: UTF-8 on Linux and macOS,
cp1252 on most Windows machines, where a "µ" in a reply or a record fails to
write or reads back garbled. So every built-in ``open`` in text mode and
every ``read_text`` / ``write_text`` names its encoding.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FOLDERS = ("src", "tests", "tools")


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
        for path in sorted((REPO / folder).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            found += [f"{path.relative_to(REPO).as_posix()}:{node.lineno}"
                      for node in ast.walk(tree)
                      if isinstance(node, ast.Call) and _missing(node)]
    assert not found, "no encoding named (add encoding=\"utf-8\"):\n" + "\n".join(found)

"""Every program LangSlice starts is told what its stdin is.

A child inherits its parent's stdin unless told otherwise. Under the MCP
server and the Fiji connector's service that stdin is the host's pipe, which
a thread reads; on Windows, Git for Windows (and other MSYS programs) block at
start-up querying a pipe with a read pending, so `git rev-parse` in the
registration's provenance hung the server. Each call passes
``stdin=subprocess.DEVNULL``, or ``stdin=None`` where the caller's stdin is
meant (an interactive pass-through), so the choice is made in the code.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STARTERS = {"run", "Popen", "call", "check_call", "check_output"}


def _starts_a_program(node: ast.Call) -> bool:
    func = node.func
    return (isinstance(func, ast.Attribute) and func.attr in STARTERS
            and isinstance(func.value, ast.Name) and func.value.id == "subprocess")


def test_every_subprocess_names_its_stdin():
    found = []
    for path in sorted((REPO / "src").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found += [f"{path.relative_to(REPO).as_posix()}:{node.lineno}"
                  for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and _starts_a_program(node)
                  and not any(k.arg in ("stdin", "input", None) for k in node.keywords)]
    assert not found, "no stdin named (add stdin=subprocess.DEVNULL):\n" + "\n".join(found)

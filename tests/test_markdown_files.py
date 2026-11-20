"""Every Markdown file in the repository is of a kind the guide allows.

A Markdown file is written only where one of ``ALLOWED`` says so (the
root ``CLAUDE.md``, "Markdown files"); a new kind is the owner's decision,
made by adding its pattern here.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

ALLOWED = (
    r"README\.md",
    r"CONTRIBUTING\.md",
    r"STYLE\.md",
    r"(.+/)?(CLAUDE|AGENTS)\.md",
    r"docs/.+\.md",
    r"(connectors|packaging|src/langslice/engines)/([^/]+/)?README\.md",
    r"(connectors/[^/]+|src/langslice)/skills/[^/]+/SKILL\.md",
    r"connectors/[^/]+/agents/[^/]+\.md",
)


def test_every_agents_file_equals_the_claude_file_beside_it() -> None:
    differing = [
        str(agents.relative_to(ROOT))
        for agents in ROOT.rglob("AGENTS.md")
        if ".venv" not in agents.parts
        and "node_modules" not in agents.parts
        and (agents.parent / "CLAUDE.md").is_file()
        and agents.read_bytes() != (agents.parent / "CLAUDE.md").read_bytes()
    ]
    assert not differing, f"AGENTS.md differs from CLAUDE.md beside it: {differing}"


def test_every_markdown_file_is_of_an_allowed_kind() -> None:
    try:
        listed = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.md"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split("\n")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    allowed = re.compile("|".join(f"(?:{pattern})" for pattern in ALLOWED))
    stray = [path for path in listed if path and not allowed.fullmatch(path)]
    assert not stray, (
        f"Markdown files of no allowed kind: {stray}. Delete them, or have the "
        "owner add their kind to ALLOWED in tests/test_markdown_files.py."
    )

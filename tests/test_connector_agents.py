"""The connectors' registration agents carry LangSlice's one system prompt,
followed only by a line about their host's sandbox where it needs one, and
their setup skills find `langslice` and put it on PATH in the same words."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from langslice.tools.descriptions.statement import SYSTEM_PROMPT

CONNECTORS = Path(__file__).resolve().parents[1] / "connectors"


def _words(text: str) -> str:
    return " ".join(text.split())


def _markdown_body(path: Path) -> str:
    match = re.match(r"---\n.*?\n---\n(.*)", path.read_text(encoding="utf-8"), re.S)
    assert match, path
    return match.group(1)


MARKDOWN = [CONNECTORS / "claude-code" / "agents" / "register.md",
            CONNECTORS / "claude-code" / "agents" / "register-scripting.md",
            CONNECTORS / "antigravity" / "agents" / "register.md",
            CONNECTORS / "antigravity" / "agents" / "register-scripting.md"]
TOML = [CONNECTORS / "codex" / "agents" / "register.toml",
        CONNECTORS / "codex" / "agents" / "register_scripting.toml"]


@pytest.mark.parametrize("path", MARKDOWN, ids=lambda p: f"{p.parent.parent.name}/{p.name}")
def test_markdown_agent_body_is_the_system_prompt(path: Path) -> None:
    body = _words(_markdown_body(path))
    assert body.startswith(_words(SYSTEM_PROMPT))
    rest = body[len(_words(SYSTEM_PROMPT)):].strip()
    assert not rest or ("sandbox" in rest and "-scripting" in path.name), rest


@pytest.mark.parametrize("path", TOML, ids=lambda p: p.name)
def test_codex_agent_instructions_are_the_system_prompt(path: Path) -> None:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    assert _words(data["developer_instructions"]) == _words(SYSTEM_PROMPT)


SETUP = [CONNECTORS / host / "skills" / "setup" / "SKILL.md"
         for host in ("claude-code", "codex", "antigravity")]


def _section(path: Path, heading: str) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## )", text, re.S | re.M)
    assert match, (path, heading)
    return match.group(1)


@pytest.mark.parametrize("heading", ["1. Find `langslice`", "2. Put LangSlice's folder on PATH"])
def test_setup_skills_share_find_and_path_steps(heading: str) -> None:
    def text(path: Path) -> str:
        return re.sub(r"Keep this absolute path for [^.]*\.", "", _section(path, heading))

    first, *others = SETUP
    for path in others:
        assert text(path) == text(first), path

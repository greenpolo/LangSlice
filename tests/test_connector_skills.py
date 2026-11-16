"""The registration skills have one source, ``src/langslice/skills/``, and
every connector carries a byte-identical copy of each.

An intended change to a skill is made in the source; then
``LANGSLICE_UPDATE_SKILLS=1 python -m pytest tests/test_connector_skills.py``
copies it into the connectors and fails once ("copied: review git diff"), so
a copy can never pass CI unreviewed.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "langslice" / "skills"
HOSTS = ("claude-code", "codex", "antigravity")
UPDATE_ENV = "LANGSLICE_UPDATE_SKILLS"


def _skills() -> list[str]:
    return sorted(p.name for p in SOURCE.iterdir() if (p / "SKILL.md").is_file())


def _files(folder: Path) -> dict[str, bytes]:
    return {p.relative_to(folder).as_posix(): p.read_bytes()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def test_every_skill_has_a_name_and_a_description() -> None:
    for name in _skills():
        head = (SOURCE / name / "SKILL.md").read_text(encoding="utf-8").split("---")[1]
        assert f"name: {name}\n" in head, name
        assert "\ndescription: " in head, name


def test_every_example_a_skill_names_exists() -> None:
    for name in _skills():
        text = (SOURCE / name / "SKILL.md").read_text(encoding="utf-8")
        for word in text.split("`"):
            if word.startswith("examples/"):
                assert (SOURCE / name / word).is_file(), (name, word)


@pytest.mark.parametrize("host", HOSTS)
def test_connector_skills_are_copies_of_the_source(host: str) -> None:
    stale: list[str] = []
    for name in _skills():
        source, copy = SOURCE / name, ROOT / "connectors" / host / "skills" / name
        if _files(source) != (_files(copy) if copy.is_dir() else {}):
            stale.append(name)
            if os.environ.get(UPDATE_ENV):
                shutil.rmtree(copy, ignore_errors=True)
                shutil.copytree(source, copy)
    if stale and os.environ.get(UPDATE_ENV):
        pytest.fail(f"copied {', '.join(stale)} into {host}: review git diff")
    assert not stale, (f"connectors/{host}/skills differ from src/langslice/skills for "
                       f"{', '.join(stale)}; run with {UPDATE_ENV}=1 to copy them")

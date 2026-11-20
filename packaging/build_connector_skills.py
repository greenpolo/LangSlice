"""Copy the canonical skills into each connector's ``skills`` folder.

``src/langslice/skills/`` is the one source. Claude Code, Codex and
Antigravity install their plugin straight from a folder of the Git
repository, so each connector's copy of the shared skills stays committed;
this script writes it. ``--check`` writes nothing and exits 1 when a
committed copy differs from what would be written.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "langslice" / "skills"
CONNECTORS = ("claude-code", "antigravity", "codex")


def wanted() -> dict[Path, bytes]:
    files = {}
    for connector in CONNECTORS:
        for path in sorted(SOURCE.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                target = ROOT / "connectors" / connector / "skills" / path.relative_to(SOURCE)
                files[target] = path.read_bytes()
    return files


def present() -> set[Path]:
    found = set()
    for connector in CONNECTORS:
        for skill in SOURCE.iterdir():
            folder = ROOT / "connectors" / connector / "skills" / skill.name
            found.update(path for path in folder.rglob("*") if path.is_file())
    return found


def build(check: bool = False) -> list[Path]:
    files = wanted()
    drift = [
        path for path, data in files.items() if not path.is_file() or path.read_bytes() != data
    ]
    drift += sorted(present() - files.keys())
    if not check:
        for path in drift:
            if path in files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(files[path])
            else:
                path.unlink()
    return drift


def main(argv: list[str]) -> int:
    check = "--check" in argv
    drift = build(check)
    for path in drift:
        verb = "differs" if check else "wrote"
        sys.stdout.write(verb + ": " + path.relative_to(ROOT).as_posix() + chr(10))
    return 1 if check and drift else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

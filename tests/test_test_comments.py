"""The tests carry no comments, as the code does not (``STYLE.md`` 7.1).

``tools/stylecheck`` checks ``src`` and ``tools``; this checks ``tests``.
A test's name and docstring say what it shows. A directive a tool reads
(``# noqa: CODE``, ``# type: ignore[code]``, ``# pyright: ignore[code]``)
stands alone, without words after it.
"""

from __future__ import annotations

import re
import tokenize
from pathlib import Path

TESTS = Path(__file__).resolve().parent

DIRECTIVE = re.compile(
    r"#\s*(?:noqa(?::\s*[A-Z]+[0-9]+(?:\s*,\s*[A-Z]+[0-9]+)*)?"
    r"|type:\s*ignore(?:\[[^\]]*\])?"
    r"|pyright:\s*ignore(?:\[[^\]]*\])?"
    r"|pyright:\s*(?:basic|strict)"
    r"|pragma:\s*no cover"
    r"|fmt:\s*\w+)"
)


def _comments(path: Path) -> list[tuple[int, str]]:
    with tokenize.open(path) as source:
        return [
            (token.start[0], token.string)
            for token in tokenize.generate_tokens(source.readline)
            if token.type == tokenize.COMMENT
        ]


def _is_allowed(line: int, text: str) -> bool:
    if line == 1 and text.startswith("#!"):
        return True
    return not DIRECTIVE.sub("", text).strip()


def test_no_test_file_carries_a_comment() -> None:
    found = [
        f"{path.relative_to(TESTS.parent)}:{line}: {text}"
        for path in sorted(TESTS.rglob("*.py"))
        for line, text in _comments(path)
        if not _is_allowed(line, text)
    ]
    assert not found, (
        "Delete these comments; say it in the test's name or docstring "
        "(STYLE.md 7.1):\n" + "\n".join(found)
    )

"""LangSlice's own Python carries no comment and no waiver.

A name says what a line is and a docstring says why. A ``noqa``, a
``type: ignore``, a ``pyright: ignore``, a ``pragma`` and a ``fmt`` switch
are comments too, so one rule covers them all. Code copied from other labs
(``src/langslice_vendor``) stays as published.
"""

from __future__ import annotations

import tokenize
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHECKED = (REPO / "src" / "langslice", REPO / "tests")


def comments_of(path: Path) -> list[str]:
    with tokenize.open(path) as source:
        return [
            f"{path.relative_to(REPO).as_posix()}:{token.start[0]}: {token.string}"
            for token in tokenize.generate_tokens(source.readline)
            if token.type == tokenize.COMMENT
            and not (token.start[0] == 1 and token.string.startswith("#!"))
        ]


def test_no_file_carries_a_comment_or_a_waiver() -> None:
    carried = [
        comment
        for folder in CHECKED
        for path in sorted(folder.rglob("*.py"))
        for comment in comments_of(path)
    ]
    assert not carried, (
        "Delete these; change the code until no waiver is needed and say the "
        "rest in a name or a docstring:\n" + "\n".join(carried)
    )

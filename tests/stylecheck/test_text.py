"""Tests of the model-facing text checks."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from tools.stylecheck.core import Config
from tools.stylecheck.testing import flagged_lines

CONFIG = Config.load(Path(__file__).parents[2] / "pyproject.toml")


def test_matching_indentation_is_flagged() -> None:
    source = """
    def edit(doc, line):
        indent = " " * 8
        a = line.startswith(" " * 12)
        b = doc.replace("    ", "")
        c = (indent + "Args") in doc
        d = doc.index("\\n\\n" + indent + "The base prompt")
        e = doc.replace("\\n", "\\n" + " " * 4)
        f = " ".join(line.split())
    """
    assert flagged_lines(source, "text-surgery") == [4, 5, 6, 7]


def test_reworking_a_docstring_is_flagged_outside_the_renderer() -> None:
    source = """
    import textwrap

    def edit(tool, run):
        run.__doc__ = tool.__doc__
        a = tool.__doc__.replace("x", "y")
        b = (tool.__doc__ or "").split("\\n")
        c = textwrap.dedent(tool.__doc__)
        d = tool.__doc__[:10]
        setattr(run, "__doc__", "text")
        e = tool.__doc__
    """
    assert flagged_lines(source, "text-surgery") == [5, 6, 7, 8, 9, 10]

    config = dataclasses.replace(
        CONFIG, docs_renderer_module="langslice.doors.declarations"
    )
    relpath = "src/langslice/doors/declarations.py"
    lines = flagged_lines(
        source, "text-surgery", relpath=relpath, config=config
    )
    assert lines == []


def _assembler(n_literals: int, n_branches: int) -> str:
    lines = ["def statement(a):", "    parts = []"]
    for index in range(n_branches):
        lines.append(f"    if a == {index}:")
        lines.append(
            "        parts.append('a clause long enough to be prose')"
        )
    for _ in range(n_literals - n_branches):
        lines.append("    parts.append(f'another clause of prose {a}')")
    lines.append("    raise ValueError('a message long enough to be prose')")
    lines.append("    return ' '.join(parts)")
    return "\n".join(lines) + "\n"


def test_text_assembled_in_branches_is_flagged() -> None:
    assert flagged_lines(_assembler(12, 6), "text-assembly") == [1]


def test_fewer_literals_or_branches_are_allowed() -> None:
    assert flagged_lines(_assembler(11, 6), "text-assembly") == []
    assert flagged_lines(_assembler(14, 5), "text-assembly") == []

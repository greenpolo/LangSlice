"""Tests of the paragraph length check."""

from __future__ import annotations

from tools.stylecheck.core import Severity
from tools.stylecheck.testing import check_source


def _steps(count: int, *, indent: str = "    ") -> str:
    """Write ``count`` assignment lines at an indent."""
    return "".join(f"{indent}x{index} = {index}\n" for index in range(count))


def test_paragraph_length_warns_about_a_long_run() -> None:
    source = f"def run(a):\n{_steps(16)}    return a\n"
    violations = check_source(source, "paragraph-length")
    assert len(violations) == 1
    assert violations[0].severity is Severity.WARNING
    assert "17 lines" in violations[0].message


def test_paragraph_length_counts_comment_lines() -> None:
    source = f"def run(a):\n    # the step\n{_steps(15)}"
    assert len(check_source(source, "paragraph-length")) == 1


def test_paragraph_length_accepts_paragraphs() -> None:
    source = f"def run(a):\n{_steps(10)}\n{_steps(10)}"
    assert check_source(source, "paragraph-length") == []


def test_paragraph_length_leaves_out_the_docstring() -> None:
    docstring = '    """Run it.\n' + "    More.\n" * 10 + '    """\n'
    source = f"def run(a):\n{docstring}{_steps(10)}"
    assert check_source(source, "paragraph-length") == []


def test_paragraph_length_leaves_a_single_long_statement_alone() -> None:
    arguments = "".join(f"        x{index}={index},\n" for index in range(20))
    source = f"def run(a):\n    return call(\n{arguments}    )\n"
    assert check_source(source, "paragraph-length") == []


def test_paragraph_length_reads_a_nested_function_with_its_outer() -> None:
    source = (
        f"def outer(a):\n{_steps(8)}    def inner(b):\n"
        f"{_steps(8, indent='        ')}    return inner\n"
    )
    violations = check_source(source, "paragraph-length")
    assert [violation.symbol for violation in violations] == ["outer"]

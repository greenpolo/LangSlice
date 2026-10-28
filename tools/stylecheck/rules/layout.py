"""Check that a function is written in paragraphs separated by blank lines.

A paragraph is a run of non-blank lines inside a function's body; comment
lines belong to the paragraph they sit in, a blank line ends it, and the
docstring is not part of it. A run made of a single statement is left
alone: a blank line cannot split it, and the expression checks cover a
statement that is too long.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator

from tools.stylecheck.core import (
    Config,
    Module,
    Project,
    Severity,
    Violation,
    function_body,
    module_check,
)
from tools.stylecheck.rules.enclosing import (
    body_nodes,
    enclosing_function,
    parent_map,
)
from tools.stylecheck.rules.scopes import functions

Function = ast.FunctionDef | ast.AsyncFunctionDef


@module_check("paragraph-length")
def check_paragraph_length(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Warn about a run of lines in a function with no blank line (1.3).

    A run longer than ``max_paragraph_lines`` that holds the start of more
    than one statement is two steps that need a blank line between them,
    or a step that needs its own function. Comment lines count. A nested
    function's lines are read as part of the function around it.

    Parameters
    ----------
    module
        The parsed file to check.
    project
        The package index; not used by this check.
    config
        The settings; ``max_paragraph_lines`` is the limit.

    Yields
    ------
    Violation
        A warning per run over the limit, at its first line.
    """
    del project
    limit = config.max_paragraph_lines
    parents = parent_map(module.tree)
    for function in functions(module.tree):
        if enclosing_function(function, parents) is not None:
            continue
        for first, length in long_paragraphs(module, function, limit=limit):
            yield module.violation(
                "paragraph-length",
                first,
                f"A paragraph runs {length} lines without a blank line; "
                f"the limit is {limit}.",
                severity=Severity.WARNING,
            )


def long_paragraphs(
    module: Module, function: Function, *, limit: int
) -> Iterator[tuple[int, int]]:
    """Yield each run of a function's lines that is over the limit.

    Parameters
    ----------
    module
        The parsed file the function is in.
    function
        A function not nested in another.
    limit
        The longest allowed run, in lines.

    Yields
    ------
    tuple of int
        The run's first 1-based line and its length, for each run longer
        than ``limit`` that starts more than one statement.
    """
    first_line = _body_start(module, function)
    if first_line is None:
        return
    last_line = function.end_lineno or function.lineno
    starts = _statement_starts(function)
    run: list[int] = []
    for number in range(first_line, last_line + 2):
        is_blank = number > last_line or not module.lines[number - 1].strip()
        if not is_blank:
            run.append(number)
            continue
        if len(run) > limit and len(starts.intersection(run)) > 1:
            yield run[0], len(run)
        run = []


def _body_start(module: Module, function: Function) -> int | None:
    """Return the first line after the docstring, or None for no body.

    With no docstring, comment lines just above the first statement
    belong to the body.
    """
    body = function_body(function)
    if not body:
        return None
    if body is not function.body:
        docstring = function.body[0]
        return (docstring.end_lineno or docstring.lineno) + 1
    start = body[0].lineno
    while start - 1 > function.lineno and _is_comment(module, start - 1):
        start -= 1
    return start


def _is_comment(module: Module, line: int) -> bool:
    """Tell whether a 1-based line of a module holds only a comment."""
    return module.lines[line - 1].strip().startswith("#")


def _statement_starts(function: Function) -> set[int]:
    """Return the line each statement in a function's body starts on."""
    return {
        node.lineno
        for node in body_nodes(function)
        if isinstance(node, ast.stmt)
    }

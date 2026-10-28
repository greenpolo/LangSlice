"""Checks on model-facing text: no text surgery, no clause-by-clause text.

*Text surgery* is editing text by its layout: matching a run of spaces
(``" " * 8``, or a literal of spaces) with ``startswith``, ``in`` or
``replace``, or assigning or reworking a ``__doc__``. *Text assembly* is
a function that builds prose clause by clause inside branches; it is
recognised by counting its long string literals and its branches. The
message of a ``raise`` and of a logging call does not count as prose.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator

from tools.stylecheck.core import (
    Config,
    Module,
    Project,
    Violation,
    function_body,
    module_check,
)
from tools.stylecheck.rules.scopes import (
    SCOPE_NODES,
    call_name,
    dotted_name,
    functions,
    module_matches,
    own_nodes,
    string_value,
)

#: The finding for text matched by its indentation.
INDENT_MATCH = (
    "Text is matched by its indentation; build it from structured parts "
    "instead of editing it (9.2)."
)

#: The finding for a reworked docstring.
DOCSTRING_EDIT = (
    "A __doc__ is assigned or reworked; only the renderer of tool "
    "descriptions edits docstrings (9.2)."
)

#: String methods that find or edit text by a pattern.
MATCHING_METHODS = frozenset(
    {
        "startswith",
        "endswith",
        "replace",
        "removeprefix",
        "removesuffix",
        "find",
        "index",
        "split",
        "partition",
        "count",
        "lstrip",
        "strip",
    }
)

#: Modules whose functions rework text.
TEXT_MODULES = ("re.", "textwrap.")

#: The fewest long literals that make a function's text assembled.
MIN_ASSEMBLY_LITERALS = 12

#: The fewest branches that make a function's text assembled.
MIN_ASSEMBLY_BRANCHES = 6

#: The length from which a literal is prose, not a key or a short label.
MIN_PROSE_LENGTH = 20

#: Logging methods whose message is not model-facing text.
LOG_METHODS = frozenset(
    {"debug", "info", "warning", "error", "exception", "critical", "log"}
)


@module_check("text-surgery")
def check_text_surgery(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag text edited by matching indentation, and reworked docstrings.

    Guide rule 9.2. Flags a run of spaces (``" " * N``, a literal of two
    or more spaces, or a name bound to either in the same function) used
    as the pattern of a matching string method or tested with ``in``. Outside
    ``docs_renderer_module`` it also flags an assignment to ``__doc__``,
    and a ``__doc__`` that is sliced, concatenated, given to a ``re`` or
    ``textwrap`` function, or has a string method called on it.

    Parameters
    ----------
    module
        The parsed file whose expressions are read.
    project
        The package index, unused by this check.
    config
        The settings that name the docs renderer module.

    Yields
    ------
    Violation
        One finding per edit.
    """
    for scope in [module.tree, *functions(module.tree)]:
        indents = _indent_names(scope)
        for node in own_nodes(scope):
            if _matches_indent(node, indents):
                yield module.violation("text-surgery", node, INDENT_MATCH)

    if module_matches(module.dotted, (config.docs_renderer_module,)):
        return
    for node in ast.walk(module.tree):
        if _edits_docstring(node):
            yield module.violation("text-surgery", node, DOCSTRING_EDIT)


def _is_spaces(node: ast.AST, indents: set[str]) -> bool:
    """Tell whether an expression is a run of spaces."""
    text = string_value(node)
    if text is not None:
        return len(text) >= 2 and not text.strip(" ")
    if isinstance(node, ast.Name):
        return node.id in indents
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return (
            string_value(node.left) == " " or string_value(node.right) == " "
        )
    return False


def _indent_names(scope: ast.AST) -> set[str]:
    """Return the names a scope binds to a run of spaces."""
    names: set[str] = set()
    for node in own_nodes(scope):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and _is_spaces(node.value, set()):
            names.add(target.id)
    return names


def _matches_indent(node: ast.AST, indents: set[str]) -> bool:
    """Tell whether a node matches text against a run of spaces."""
    if isinstance(node, ast.Call):
        func = node.func
        if not isinstance(func, ast.Attribute):
            return False
        if func.attr not in MATCHING_METHODS:
            return False
        return bool(node.args) and _has_spaces(node.args[0], indents)
    if isinstance(node, ast.Compare):
        tests = (ast.In, ast.NotIn)
        if not any(isinstance(operator, tests) for operator in node.ops):
            return False
        return _has_spaces(node.left, indents)
    return False


def _has_spaces(node: ast.AST, indents: set[str]) -> bool:
    """Tell whether an expression is or concatenates a run of spaces."""
    if _is_spaces(node, indents):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _has_spaces(node.left, indents) or _has_spaces(
            node.right, indents
        )
    return False


def _is_docstring_attribute(node: ast.AST) -> bool:
    """Tell whether a node is a ``__doc__`` attribute."""
    return isinstance(node, ast.Attribute) and node.attr == "__doc__"


def _edits_docstring(node: ast.AST) -> bool:
    """Tell whether a node assigns, slices or reworks a ``__doc__``."""
    if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        return any(_is_docstring_attribute(target) for target in targets)
    if isinstance(node, ast.Subscript):
        return _is_docstring_attribute(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _is_docstring_attribute(node.left) or _is_docstring_attribute(
            node.right
        )
    if isinstance(node, ast.Call):
        return _call_edits_docstring(node)
    return False


def _call_edits_docstring(node: ast.Call) -> bool:
    """Tell whether a call reworks a ``__doc__`` or sets one."""
    func = node.func
    if isinstance(func, ast.Attribute) and _docstring_receiver(func.value):
        return True
    name = call_name(node)
    if name == "setattr" and len(node.args) >= 2:
        return string_value(node.args[1]) == "__doc__"
    if name.startswith(TEXT_MODULES):
        return any(_docstring_receiver(arg) for arg in node.args)
    return False


def _docstring_receiver(node: ast.AST) -> bool:
    """Tell whether an expression is a ``__doc__``, possibly ``or ""``."""
    if isinstance(node, ast.BoolOp):
        return any(_is_docstring_attribute(value) for value in node.values)
    return _is_docstring_attribute(node)


@module_check("text-assembly")
def check_text_assembly(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a function that assembles prose inside many branches.

    Guide rule 9.1. A function with at least 12 string or f-string
    literals of 20 characters or more and at least 6 branches (``if``,
    ``elif``, conditional expressions and ``case`` clauses) is flagged.
    The docstring, nested definitions, and the messages of ``raise``
    statements and logging calls are not counted.

    Parameters
    ----------
    module
        The parsed file whose functions are read.
    project
        The package index, unused by this check.
    config
        The settings, unused by this check.

    Yields
    ------
    Violation
        One finding per function that assembles text.
    """
    for function in functions(module.tree):
        n_literals, n_branches = _text_counts(function)
        if (
            n_literals >= MIN_ASSEMBLY_LITERALS
            and n_branches >= MIN_ASSEMBLY_BRANCHES
        ):
            yield module.violation(
                "text-assembly",
                function,
                f"{function.name!r} builds text from {n_literals} literals "
                f"across {n_branches} branches; write each variant whole as "
                "a template and choose among them (9.1).",
            )


def _text_counts(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[int, int]:
    """Count a function's prose literals and its branches."""
    n_literals = 0
    n_branches = 0
    pending: list[ast.AST] = list(function_body(function))
    while pending:
        node = pending.pop()
        if isinstance(node, SCOPE_NODES) or _is_message(node):
            continue
        if isinstance(node, (ast.If, ast.IfExp, ast.match_case)):
            n_branches += 1
        if _prose_length(node) >= MIN_PROSE_LENGTH:
            n_literals += 1
            continue
        pending.extend(ast.iter_child_nodes(node))
    return n_literals, n_branches


def _is_message(node: ast.AST) -> bool:
    """Tell whether a node is a raise or a logging call."""
    if isinstance(node, ast.Raise):
        return True
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr not in LOG_METHODS:
        return False
    receiver = dotted_name(func.value).lower()
    return "log" in receiver


def _prose_length(node: ast.AST) -> int:
    """Return a literal's length in characters, or 0 for other nodes."""
    text = string_value(node)
    if text is not None:
        return len(text)
    if not isinstance(node, ast.JoinedStr):
        return 0
    length = 0
    for part in node.values:
        part_text = string_value(part)
        length += len(part_text) if part_text is not None else 1
    return length

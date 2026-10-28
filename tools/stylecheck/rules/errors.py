"""Checks on errors: broad handlers, printing, hand-built error replies.

A *broad handler* is an ``except`` clause that catches ``Exception`` or
``BaseException``, alone or in a tuple, or a bare ``except``. An *error
code* is an upper-case string of two or more words joined by
underscores, such as ``"BAD_ARGS"``; where only a code can stand, such
as the first argument of ``Refused``, one upper-case word of three
letters or more (``"FROZEN"``) is one too.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator

from tools.stylecheck.core import (
    Config,
    Module,
    Project,
    Violation,
    module_check,
)
from tools.stylecheck.rules.annotations import last_name
from tools.stylecheck.rules.scopes import (
    FUNCTION_NODES,
    SCOPE_NODES,
    call_name,
    comment_on,
    module_matches,
    own_nodes,
    parameters,
    string_value,
)

#: The exception classes that make a handler broad.
BROAD_EXCEPTIONS = frozenset({"Exception", "BaseException"})

#: The finding for a broad handler that drops the error.
INERT_HANDLER = (
    "A broad except only passes, returns None or logs at debug; raise, "
    "return a real value or report the skip (8.3)."
)

#: The finding for a broad handler without a stated reason.
UNEXPLAINED_HANDLER = (
    "A broad except needs a comment on its line naming the failure it "
    "expects (8.3)."
)

#: A lint waiver inside a comment, which names a rule and not a failure.
WAIVER_PATTERN = re.compile(r"(?:noqa|type:\s*ignore)(?::?\s*[\w\[\],\s]*)?")

#: The fewest words a comment needs to name the failure it expects.
MIN_REASON_WORDS = 2

#: Calls that write to the terminal instead of the log.
PRINT_CALLS = frozenset(
    {
        "print",
        "sys.stdout.write",
        "sys.stderr.write",
        "rich.print",
        "click.echo",
        "typer.echo",
        "pprint.pprint",
    }
)

#: Modules whose ``print``-like functions write to the terminal.
PRINT_MODULES = frozenset({"rich", "pprint", "click", "typer"})

#: What an error code looks like anywhere: two or more upper-case words.
CODE_PATTERN = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+")

#: What an error code looks like in a code place: one upper-case word of
#: three letters or more is enough there (``"FROZEN"``).
WORD_PATTERN = re.compile(r"[A-Z][A-Z0-9_]{2,}")

#: The logging levels, upper-case words that are not codes.
LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

#: The item that marks a dict as an error reply.
ERROR_ITEM = ("status", "error")

#: Upper-case names that Python itself defines, which are not codes.
PYTHON_NAMES = frozenset({"TYPE_CHECKING", "KW_ONLY"})

#: Record keys and call keywords under which a value is an error code.
CODE_KEYS = frozenset({"code", "error"})

#: Name endings of the constructors that take an error code.
REFUSAL_SUFFIXES = ("Refused", "Refusal", "Error", "Exception")

#: The scopes whose names are read apart: the module and each definition.
SCOPES = (ast.Module, *SCOPE_NODES)

#: The strings a scope binds to names, by name, with the bound node.
Bound = dict[str, tuple[ast.expr, str]]


@module_check("broad-except")
def check_broad_except(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a broad handler that loses the error or does not say why.

    Guide rule 8.3. A broad handler (see the module docstring) whose body
    only passes, continues, breaks, returns None or logs at debug level
    is flagged. Otherwise it needs a comment on its ``except`` line that
    names the failure it expects, in at least two words besides any lint
    waiver. A handler that raises needs neither.

    Parameters
    ----------
    module
        The parsed file whose handlers are read.
    project
        The package index, unused by this check.
    config
        The settings, unused by this check.

    Yields
    ------
    Violation
        One finding per broad handler that is not allowed.
    """
    for node in ast.walk(module.tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        problem = _handler_problem(module, node)
        if problem:
            yield module.violation("broad-except", node, problem)


def _handler_problem(module: Module, handler: ast.ExceptHandler) -> str:
    """Say what is wrong with an exception handler, or return ``""``."""
    if not _is_broad(handler) or _raises(handler):
        return ""
    if all(_is_inert(statement) for statement in handler.body):
        return INERT_HANDLER
    if not _names_failure(module, handler):
        return UNEXPLAINED_HANDLER
    return ""


def _is_broad(handler: ast.ExceptHandler) -> bool:
    """Tell whether a handler catches everything or ``Exception``."""
    caught = handler.type
    if caught is None:
        return True
    classes = caught.elts if isinstance(caught, ast.Tuple) else [caught]
    return any(last_name(item) in BROAD_EXCEPTIONS for item in classes)


def _raises(handler: ast.ExceptHandler) -> bool:
    """Tell whether a handler's own body raises."""
    return any(isinstance(node, ast.Raise) for node in own_nodes(handler))


def _is_inert(statement: ast.stmt) -> bool:
    """Tell whether a statement drops the error without a trace."""
    if isinstance(statement, (ast.Pass, ast.Continue, ast.Break)):
        return True
    if isinstance(statement, ast.Return):
        value = statement.value
        return value is None or (
            isinstance(value, ast.Constant) and value.value is None
        )
    if not isinstance(statement, ast.Expr):
        return False
    if isinstance(statement.value, ast.Constant):
        return True
    return call_name(statement.value).endswith(".debug")


def _names_failure(module: Module, handler: ast.ExceptHandler) -> bool:
    """Tell whether the ``except`` line's comment names a failure."""
    last_line = handler.body[0].lineno - 1
    comment = comment_on(
        module, handler.lineno, max(last_line, handler.lineno)
    )
    reason = WAIVER_PATTERN.sub(" ", comment.replace("#", " "))
    words = re.findall(r"[A-Za-z]{2,}", reason)
    return len(words) >= MIN_REASON_WORDS


@module_check("print-call")
def check_print_call(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag writing to the terminal outside the command modules.

    Guide rule 8.5. Flags a call of one of ``PRINT_CALLS``, the builtin
    ``print`` and the terminal writers of ``sys``, ``rich``, ``click``,
    ``typer`` and ``pprint``, also when imported under another name,
    outside ``cli_modules``.

    Parameters
    ----------
    module
        The parsed file whose calls are read.
    project
        The package index, unused by this check.
    config
        The settings that list the command modules.

    Yields
    ------
    Violation
        One finding per terminal write.
    """
    if module_matches(module.dotted, config.cli_modules):
        return
    imported = _imported_printers(module.tree)
    for node in ast.walk(module.tree):
        name = call_name(node)
        name = imported.get(name, name)
        if name in PRINT_CALLS:
            yield module.violation(
                "print-call",
                node,
                f"{name}() writes to the terminal; report through logging, "
                "or move the call to a command module (8.5).",
            )


def _imported_printers(tree: ast.Module) -> dict[str, str]:
    """Map local names bound by ``from rich import print`` and the like."""
    printers: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module not in PRINT_MODULES:
            continue
        for alias in node.names:
            printers[alias.asname or alias.name] = (
                f"{node.module}.{alias.name}"
            )
    return printers


@module_check("error-dict")
def check_error_dict(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag an error reply built by hand as a dict.

    Guide rule 8.1. Flags a dict literal or ``dict(...)`` call holding a
    ``"status": "error"`` item, outside ``codes_module``, which owns the
    refusal type and its conversion to a reply.

    Parameters
    ----------
    module
        The parsed file whose dicts are read.
    project
        The package index, unused by this check.
    config
        The settings that name the codes module.

    Yields
    ------
    Violation
        One finding per hand-built error dict.
    """
    if module_matches(module.dotted, (config.codes_module,)):
        return
    for node in ast.walk(module.tree):
        if _is_error_dict(node):
            yield module.violation(
                "error-dict",
                node,
                "An error reply is built by hand; raise Refused with a "
                "code and let the door convert it (8.1).",
            )


def _is_error_dict(node: ast.AST) -> bool:
    """Tell whether a node builds a dict with ``status`` set to error."""
    if isinstance(node, ast.Dict):
        items = [
            (string_value(key), string_value(value))
            for key, value in zip(node.keys, node.values, strict=True)
        ]
        return ERROR_ITEM in items
    if isinstance(node, ast.Call) and call_name(node) == "dict":
        items = [
            (keyword.arg, string_value(keyword.value))
            for keyword in node.keywords
        ]
        return ERROR_ITEM in items
    return False


@module_check("error-code-literal")
def check_error_code_literal(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag an error code written as a bare string.

    Guide rule 8.1. Outside ``codes_module``, a string is flagged where
    it is used as a code. The *code places* are the first argument of a
    refusal or error constructor (a name ending in ``Refused``,
    ``Refusal``, ``Error`` or ``Exception``, or starting with
    ``refuse``), a ``code=`` or ``error=`` argument, a value stored under
    a ``"code"`` or ``"error"`` key, and a comparison with a code (an
    operand named ``code``, ending in ``_code``, or read as ``.code`` or
    ``["code"]``). A string in a code place is flagged when it looks
    like a code (see the module docstring), one word included
    (``"FROZEN"``); in any other comparison only a code of two or more
    words is. The string may be built (``"BAD_" + kind``,
    ``f"NO_{what}"``), chosen (``code or "FROZEN"``, ``"A" if x else
    "B"``) or first bound to a name in the same function or at module
    level; a name is reported at the string it was bound to.

    Parameters
    ----------
    module
        The parsed file whose literals are read.
    project
        The package index, unused by this check.
    config
        The settings that name the codes module.

    Yields
    ------
    Violation
        One finding per code string.
    """
    if module_matches(module.dotted, (config.codes_module,)):
        return
    reported: set[int] = set()
    for literal, text in _module_codes(module.tree):
        if id(literal) in reported:
            continue
        reported.add(id(literal))
        yield module.violation(
            "error-code-literal",
            literal,
            f"Error code {text!r} is a bare string; use the member of the "
            "codes enum (8.1).",
        )


def _module_codes(tree: ast.Module) -> Iterator[tuple[ast.expr, str]]:
    """Yield every code string of a module, scope by scope, with its text.

    A string bound to a name and used as a code twice is yielded twice.
    """
    module_bound = _bound_texts(tree, {})
    for scope in ast.walk(tree):
        if not isinstance(scope, SCOPES):
            continue
        bound = _bound_texts(scope, module_bound)
        for node in own_nodes(scope):
            yield from _code_literals(node, bound)


def _code_literals(node: ast.AST, bound: Bound) -> list[tuple[ast.expr, str]]:
    """Return the code strings a node uses as codes, with their text."""
    strong, weak = _code_places(node)
    places = [
        *((item, WORD_PATTERN) for item in strong),
        *((item, CODE_PATTERN) for item in weak),
    ]
    codes: list[tuple[ast.expr, str]] = []
    for item, pattern in places:
        for literal, text in _texts(item, bound):
            if _is_code(text, pattern):
                codes.append((literal, text))
    return codes


def _code_places(node: ast.AST) -> tuple[list[ast.expr], list[ast.expr]]:
    """Return a node's code places, and its other compared operands."""
    if isinstance(node, ast.Call):
        return _call_codes(node), []
    if isinstance(node, ast.Compare):
        return _compared(node)
    if isinstance(node, ast.Dict):
        values = [
            value
            for key, value in zip(node.keys, node.values, strict=True)
            if string_value(key) in CODE_KEYS
        ]
        return values, []
    if isinstance(node, ast.Assign):
        return _stored_codes(node), []
    return [], []


def _is_code(text: str, pattern: re.Pattern[str]) -> bool:
    """Tell whether a string's text looks like an error code."""
    if text in PYTHON_NAMES or text in LOG_LEVELS:
        return False
    return pattern.fullmatch(text) is not None


def _texts(node: ast.expr, bound: Bound) -> list[tuple[ast.expr, str]]:
    """Return the strings an expression may evaluate to, with their nodes.

    A name bound to a string in the scope stands for that string; a
    conditional expression or an ``or``/``and`` gives each alternative.
    """
    if isinstance(node, ast.Name):
        return [bound[node.id]] if node.id in bound else []
    if isinstance(node, ast.IfExp):
        return [*_texts(node.body, bound), *_texts(node.orelse, bound)]
    if isinstance(node, ast.BoolOp):
        texts: list[tuple[ast.expr, str]] = []
        for value in node.values:
            texts.extend(_texts(value, bound))
        return texts
    text = built_text(node)
    return [] if text is None else [(node, text)]


def built_text(node: ast.expr) -> str | None:
    """Return the text a string literal, sum or f-string spells.

    Parameters
    ----------
    node
        The expression that may build a string.

    Returns
    -------
    str or None
        The text, with each part that is not a literal written as
        ``X``; None when the expression holds no string literal.
    """
    text = string_value(node)
    if text is not None:
        return text
    if isinstance(node, ast.JoinedStr):
        return _formatted_text(node)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _sum_text(node)
    return None


def _formatted_text(node: ast.JoinedStr) -> str | None:
    """Return an f-string's text, its fields written ``X``."""
    if not any(string_value(value) for value in node.values):
        return None
    return "".join(string_value(value) or "X" for value in node.values)


def _sum_text(node: ast.BinOp) -> str | None:
    """Return the text of ``a + b``, a part that is no string written ``X``."""
    left = built_text(node.left)
    right = built_text(node.right)
    if left is None and right is None:
        return None
    return f"{left or 'X'}{right or 'X'}"


def _bound_texts(scope: ast.AST, outer: Bound) -> Bound:
    """Map each name a scope binds once to a string, to that string.

    A name bound more than once, or to anything but a string, is left
    out, since the code cannot tell which value a later use reads. The
    module's names (``outer``) are seen too, unless the scope binds the
    name itself or takes it as a parameter.
    """
    taken: set[str] = set()
    if isinstance(scope, (*FUNCTION_NODES, ast.Lambda)):
        taken = {parameter.arg for parameter in parameters(scope)}
    counts, bound = _string_bindings(scope)

    visible = {
        name: item
        for name, item in outer.items()
        if name not in counts and name not in taken
    }
    for name, item in bound.items():
        if counts[name] == 1 and name not in taken:
            visible[name] = item
    return visible


def _string_bindings(scope: ast.AST) -> tuple[dict[str, int], Bound]:
    """Count how often a scope binds each name, and find its strings."""
    counts: dict[str, int] = {}
    bound: Bound = {}
    for node in own_nodes(scope):
        targets, value = _assignment(node)
        text = None if value is None else built_text(value)
        for target in targets:
            counts[target] = counts.get(target, 0) + 1
            if value is not None and text is not None:
                bound[target] = (value, text)
    return counts, bound


def _assignment(node: ast.AST) -> tuple[list[str], ast.expr | None]:
    """Return the plain names a statement binds and the value it binds."""
    if isinstance(node, ast.Assign):
        names = [
            target.id
            for target in node.targets
            if isinstance(target, ast.Name)
        ]
        return names, node.value
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id], node.value
    if isinstance(node, (ast.AugAssign, ast.NamedExpr)):
        target = node.target
        if isinstance(target, ast.Name):
            return [target.id], None
    return [], None


def _call_codes(node: ast.Call) -> list[ast.expr]:
    """Return a call's arguments that sit in a code's place."""
    candidates = [
        keyword.value for keyword in node.keywords if keyword.arg in CODE_KEYS
    ]
    callee = call_name(node).rpartition(".")[2]
    is_refusal = callee.endswith(REFUSAL_SUFFIXES) or callee.lstrip(
        "_"
    ).startswith("refuse")
    if is_refusal and node.args:
        candidates.append(node.args[0])
    return candidates


def _compared(node: ast.Compare) -> tuple[list[ast.expr], list[ast.expr]]:
    """Split a comparison's operands by whether a code is compared.

    Returns the operands, tuples unpacked, as code places when one
    operand is a code, and as other operands otherwise.
    """
    operands: list[ast.expr] = []
    for operand in [node.left, *node.comparators]:
        if isinstance(operand, (ast.Tuple, ast.List, ast.Set)):
            operands.extend(operand.elts)
        else:
            operands.append(operand)
    if any(_is_code_subject(operand) for operand in operands):
        return operands, []
    return [], operands


def _is_code_subject(node: ast.expr) -> bool:
    """Tell whether an operand reads a code, by its name or its key."""
    if isinstance(node, ast.Subscript):
        return string_value(node.slice) in CODE_KEYS
    if isinstance(node, ast.Call) and call_name(node).endswith(".get"):
        return bool(node.args) and string_value(node.args[0]) in CODE_KEYS
    name = last_name(node)
    return name == "code" or name.endswith("_code")


def _stored_codes(node: ast.Assign) -> list[ast.expr]:
    """Return the value of ``x["code"] = ...``."""
    for target in node.targets:
        if isinstance(target, ast.Subscript):
            if string_value(target.slice) in CODE_KEYS:
                return [node.value]
    return []

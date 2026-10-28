"""Walk the parts of a syntax tree that the checks share.

A *scope* here is a function, class or lambda. The checks often need the
statements that belong to one function and not to the functions nested in
it, the parameters a function declares, the dotted name an expression
spells, and the comments written on a span of lines. This module holds
nothing that registers a check.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator

from tools.stylecheck.core import Module

#: The nodes that open a function; a check that reads one function stops
#: at each of these inside it.
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)

#: The nodes that open a new scope of names.
SCOPE_NODES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
)

#: The names a method gives the object it is bound to.
BOUND_NAMES = frozenset({"self", "cls"})


def dotted_name(node: ast.AST) -> str:
    """Spell a name or an attribute chain as a dotted string.

    Parameters
    ----------
    node
        Any expression node from a parsed module.

    Returns
    -------
    str
        ``"sys.stdout.write"`` for that attribute chain, the bare name for
        a name, and ``""`` for any other expression.
    """
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def call_name(node: ast.AST) -> str:
    """Spell the callee of a call as a dotted string.

    Parameters
    ----------
    node
        Any node; only a call has a callee.

    Returns
    -------
    str
        The callee's dotted name, or ``""`` when the node is not a call
        or its callee is not a plain name or attribute chain.
    """
    if not isinstance(node, ast.Call):
        return ""
    return dotted_name(node.func)


def string_value(node: ast.AST | None) -> str | None:
    """Return the text of a string constant.

    Parameters
    ----------
    node
        Any expression node, or nothing.

    Returns
    -------
    str or None
        The constant's text, or None when the node is not a string.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def functions(
    tree: ast.AST,
) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Yield every function and method under a node, at any depth.

    Parameters
    ----------
    tree
        A module or any node that may contain definitions.

    Yields
    ------
    ast.FunctionDef or ast.AsyncFunctionDef
        Each definition, in the order ``ast.walk`` meets it.
    """
    for node in ast.walk(tree):
        if isinstance(node, FUNCTION_NODES):
            yield node


def own_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """Yield the nodes of a scope's body, not those of scopes inside it.

    Parameters
    ----------
    scope
        A function, class or module node.

    Yields
    ------
    ast.AST
        Every node under the body; a nested function, class or lambda is
        yielded itself but not entered.
    """
    pending = list(ast.iter_child_nodes(scope))
    while pending:
        node = pending.pop()
        yield node
        if not isinstance(node, SCOPE_NODES):
            pending.extend(ast.iter_child_nodes(node))


def parameters(
    function: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
    *,
    include_bound: bool = True,
) -> list[ast.arg]:
    """Return every parameter a function declares, in order.

    Parameters
    ----------
    function
        The function or lambda whose signature is read.
    include_bound
        False leaves out a first positional parameter named ``self`` or
        ``cls``, the object a method is bound to.

    Returns
    -------
    list of ast.arg
        Positional-only, positional, ``*args``, keyword-only and
        ``**kwargs`` parameters.
    """
    arguments = function.args
    declared = [*arguments.posonlyargs, *arguments.args]
    if not include_bound and declared and declared[0].arg in BOUND_NAMES:
        declared = declared[1:]
    if arguments.vararg is not None:
        declared.append(arguments.vararg)
    declared.extend(arguments.kwonlyargs)
    if arguments.kwarg is not None:
        declared.append(arguments.kwarg)
    return declared


def methods(tree: ast.AST) -> Iterator[tuple[ast.ClassDef, ast.AST]]:
    """Yield each class with each function written directly in its body.

    Parameters
    ----------
    tree
        A module or any node that may contain classes.

    Yields
    ------
    tuple of ast.ClassDef and ast.AST
        The class and one of its methods.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if isinstance(item, FUNCTION_NODES):
                yield node, item


def comment_on(module: Module, first_line: int, last_line: int) -> str:
    """Join the comments written on a span of lines.

    Parameters
    ----------
    module
        The parsed file whose comments are read.
    first_line
        The first 1-based line of the span.
    last_line
        The last 1-based line of the span, included.

    Returns
    -------
    str
        The comments' text, with their ``#`` signs, joined by spaces;
        ``""`` when the span holds none.
    """
    texts = [
        text
        for line, text in module.comments
        if first_line <= line <= last_line
    ]
    return " ".join(texts)


def module_matches(dotted: str, names: tuple[str, ...]) -> bool:
    """Tell whether a module is one of the named modules or inside one.

    Parameters
    ----------
    dotted
        The module's import name.
    names
        Module or package names from the settings; an empty name matches
        nothing.

    Returns
    -------
    bool
        True when ``dotted`` equals a name or lies in a package so named.
    """
    for name in names:
        if name and (dotted == name or dotted.startswith(f"{name}.")):
            return True
    return False

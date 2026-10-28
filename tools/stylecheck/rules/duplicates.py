"""Find function bodies written twice across the package.

Two bodies are the same when their syntax trees match after the docstrings
are dropped and each function's own local names (its parameters and the
names it assigns) are renamed in order of first use. Names the function
reads from outside, such as the functions it calls and the modules it
uses, are kept, and so are attribute names and constants: two bodies that
call different helpers in the same pattern are not copies, while a copy
with its variables renamed still is.
"""

from __future__ import annotations

import ast
import copy
from collections.abc import Iterator
from dataclasses import dataclass

from tools.stylecheck.core import (
    Config,
    Project,
    Violation,
    function_body,
    project_check,
)
from tools.stylecheck.rules.enclosing import body_nodes
from tools.stylecheck.rules.scopes import FUNCTION_NODES, parameters

Function = ast.FunctionDef | ast.AsyncFunctionDef

#: The fewest statements, counted at every depth, a body needs before a
#: repeat of it is worth a shared function.
MIN_STATEMENTS = 4


@dataclass(frozen=True)
class Copy:
    """Say where one function body is written.

    Parameters
    ----------
    path
        The file, relative to the repository root.
    symbol
        The function's qualified name in that file.
    line
        The line of the ``def``.
    """

    path: str
    symbol: str
    line: int


@project_check("duplicate-body")
def check_duplicate_body(
    project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a function body that repeats one written elsewhere (10.4).

    Bodies under four statements are ignored. Every copy after the first,
    in file and line order, is reported, naming the first.

    Parameters
    ----------
    project
        The package index, with every parsed module.
    config
        The settings; not used by this check.

    Yields
    ------
    Violation
        One per repeated body.
    """
    del config
    first_copies: dict[str, Copy] = {}
    modules = sorted(
        project.modules.values(), key=lambda module: module.relpath
    )

    for module in modules:
        for definition in module.definitions:
            node = definition.node
            if not isinstance(node, FUNCTION_NODES):
                continue
            key = body_key(node)
            if not key:
                continue

            here = Copy(module.relpath, definition.qualname, node.lineno)
            first = first_copies.setdefault(key, here)
            if first is here:
                continue

            yield Violation(
                "duplicate-body",
                here.path,
                here.symbol,
                here.line,
                f"The body repeats {first.path}:{first.symbol}; keep one "
                "copy and import it.",
            )


def body_key(function: Function) -> str:
    """Return a function body's normalised form, or ``""`` for a short one.

    Parameters
    ----------
    function
        The function whose body is compared.

    Returns
    -------
    str
        The dump of the body with docstrings dropped and local names
        renamed; ``""`` when the body has fewer than four statements.
    """
    body = function_body(function)
    n_statements = sum(
        1 for node in body_nodes(function) if isinstance(node, ast.stmt)
    )
    if n_statements < MIN_STATEMENTS:
        return ""
    local_names = _local_names(function)
    statements = [copy.deepcopy(statement) for statement in body]
    # wrapped in a module so the typed generic_visit visits each statement
    wrapper = ast.Module(body=statements, type_ignores=[])
    renamed = _LocalRenamer(local_names).generic_visit(wrapper)
    return ast.dump(renamed)


def _local_names(function: Function) -> set[str]:
    """Return the names a function binds: parameters and assignments."""
    names = {argument.arg for argument in parameters(function)}
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (*FUNCTION_NODES, ast.ClassDef)):
            names.add(node.name)
    return names


class _LocalRenamer(ast.NodeTransformer):
    """Rename local names to placeholders, in order of first use.

    Docstrings of nested functions are dropped on the way.
    """

    def __init__(self, local_names: set[str]) -> None:
        self.local_names = local_names
        self.renamed: dict[str, str] = {}

    def placeholder(self, name: str) -> str:
        """Return the placeholder for a local name, or the name itself.

        Parameters
        ----------
        name
            An identifier read or bound in the body.

        Returns
        -------
        str
            ``"_1"``, ``"_2"``, ... for local names in order of first
            use; any other name unchanged.
        """
        if name not in self.local_names:
            return name
        if name not in self.renamed:
            self.renamed[name] = f"_{len(self.renamed) + 1}"
        return self.renamed[name]

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Rename a local name read or written.

        Parameters
        ----------
        node
            A name in the body.

        Returns
        -------
        ast.Name
            The same node, renamed when it is local.
        """
        node.id = self.placeholder(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        """Rename a parameter of a nested function or lambda.

        Parameters
        ----------
        node
            A parameter in the body.

        Returns
        -------
        ast.arg
            The same node, renamed, with its annotation kept.
        """
        node.arg = self.placeholder(node.arg)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Rename a nested function and drop its docstring.

        Parameters
        ----------
        node
            A function nested in the body.

        Returns
        -------
        ast.FunctionDef
            The same node, renamed, without its docstring.
        """
        node.name = self.placeholder(node.name)
        node.body = function_body(node) or [ast.Pass()]
        self.generic_visit(node)
        return node

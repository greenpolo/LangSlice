"""Find the docstrings a model reads, so the docstring rules leave them alone.

A tool's description is the docstring of its stub: each function the
module that renders tool descriptions (the setting
``docs_renderer_module``) lists in its table of stubs, the module-level
``STUBS`` assignment, and every function a plugin passes as the stub of a
``ToolSpec(name, stub, body, ...)``. The renderer's other functions are
ordinary code and are checked like any other. That text is pinned by the text
snapshot and changed only on purpose, so the rules about docstring form and
line length do not ask an agent to rewrite it. Everything else about a
stub, such as its complexity or its names, is still checked.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from tools.stylecheck.core import Config, Module, Project, Violation

#: The checks and tool codes that judge a docstring's form and wording.
DOCSTRING_RULES = frozenset(
    {
        "summary-first-word",
        "parameter-description",
        "docstring-references",
        "docstring-layer",
        "history-comment",
        "ruff:E501",
    }
)

#: The rule prefixes that judge docstrings as a family.
DOCSTRING_PREFIXES = ("numpydoc:", "ruff:D")

#: The renderer module's table of tool stubs; a function named in its
#: value is a stub, and no other function of the module is.
STUB_TABLE = "STUBS"


@dataclass(frozen=True)
class Stub:
    """A function whose docstring is a tool description."""

    path: str
    symbol: str
    first_line: int
    last_line: int


def find_stubs(project: Project, config: Config) -> list[Stub]:
    """Return every tool stub in the package.

    Parameters
    ----------
    project
        The parsed package.
    config
        The settings; ``docs_renderer_module`` names the declarations.

    Returns
    -------
    list of Stub
        Each stub with the lines its docstring spans.
    """
    stubs: list[Stub] = []
    for module in project.modules.values():
        if module.dotted == config.docs_renderer_module:
            names = _table_stub_names(module)
        else:
            names = _toolspec_stub_names(module)
        stubs.extend(_stubs_named(module, names))
    return stubs


def drop_stub_docstring_findings(
    violations: list[Violation], stubs: list[Stub]
) -> list[Violation]:
    """Remove the docstring findings that fall on a tool stub.

    Parameters
    ----------
    violations
        Every finding of a run.
    stubs
        The tool stubs, from :func:`find_stubs`.

    Returns
    -------
    list of Violation
        The findings, without those that judge a stub's docstring.
    """
    kept: list[Violation] = []
    for violation in violations:
        if not _is_docstring_rule(violation.rule):
            kept.append(violation)
            continue
        if not any(_covers(stub, violation) for stub in stubs):
            kept.append(violation)
    return kept


def _is_docstring_rule(rule: str) -> bool:
    """Return whether a rule judges a docstring's form or wording."""
    return rule in DOCSTRING_RULES or rule.startswith(DOCSTRING_PREFIXES)


def _covers(stub: Stub, violation: Violation) -> bool:
    """Return whether a finding is about a stub's docstring."""
    if violation.path != stub.path:
        return False
    if stub.first_line <= violation.line <= stub.last_line:
        return True
    is_line_rule = violation.rule in {"ruff:E501", "history-comment"}
    return violation.symbol == stub.symbol and not is_line_rule


def _table_stub_names(module: Module) -> set[str]:
    """Return the module's functions named in its table of stubs."""
    functions = {
        node.name
        for node in module.tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    names: set[str] = set()
    for value in _stub_table_values(module.tree):
        for node in ast.walk(value):
            if isinstance(node, ast.Name) and node.id in functions:
                names.add(node.id)
    return names


def _stub_table_values(tree: ast.Module) -> list[ast.expr]:
    """Return the values assigned to the table of stubs at module level."""
    values: list[ast.expr] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if any(_is_name(target, STUB_TABLE) for target in targets):
            values.append(value)
    return values


def _is_name(node: ast.expr, name: str) -> bool:
    """Return whether an assignment target is a plain name."""
    return isinstance(node, ast.Name) and node.id == name


def _toolspec_stub_names(module: Module) -> set[str]:
    """Return the names passed as the stub of a ``ToolSpec`` call."""
    names: set[str] = set()
    for node in ast.walk(module.tree):
        if not isinstance(node, ast.Call):
            continue
        if not _is_toolspec(node.func) or len(node.args) < 2:
            continue
        stub = node.args[1]
        if isinstance(stub, ast.Name):
            names.add(stub.id)
    return names


def _is_toolspec(func: ast.expr) -> bool:
    """Return whether a call's target is ``ToolSpec``, by name."""
    if isinstance(func, ast.Name):
        return func.id == "ToolSpec"
    if isinstance(func, ast.Attribute):
        return func.attr == "ToolSpec"
    return False


def _stubs_named(module: Module, names: set[str]) -> list[Stub]:
    """Return the module's top-level functions with the given names."""
    stubs: list[Stub] = []
    for node in module.tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in names or ast.get_docstring(node) is None:
            continue

        docstring = node.body[0]
        stubs.append(
            Stub(
                path=module.relpath,
                symbol=node.name,
                first_line=docstring.lineno,
                last_line=docstring.end_lineno or docstring.lineno,
            )
        )
    return stubs

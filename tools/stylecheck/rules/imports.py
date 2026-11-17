"""Check that imports sit at the top of a module, with two named exceptions.

An import inside a function is allowed when the module is a slow
optional dependency whose top-level name is in ``heavy_modules``, or when
every package module the statement imports, directly or through others,
imports the module the function is in when it loads, so that the import
at the top would close a cycle. The import graph read
here is the one Python follows at load time: the module-level imports,
not those inside functions or under ``if TYPE_CHECKING:``.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator

from tools.stylecheck.core import (
    Config,
    Module,
    Project,
    Violation,
    imported_modules,
    module_check,
)
from tools.stylecheck.rules.scopes import (
    dotted_name,
    functions,
    own_nodes,
)

#: The finding for a local import that is neither slow nor in a cycle.
MOVE_TO_TOP = (
    "Move the import to the top of the module: an import inside a "
    "function is for a slow dependency (stylecheck.heavy_modules) or a "
    "module that imports this one when it loads (2.3)."
)

#: The guard whose body only a type checker reads, not Python at load.
TYPE_CHECKING = "TYPE_CHECKING"

Statement = ast.Import | ast.ImportFrom


@module_check("local-import")
def check_local_import(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag an import inside a function that has no checked reason.

    Guide rule 2.3. See the module docstring for the two reasons that
    allow one. The cycle is read from the package's import graph: the
    imported module must import this one when it loads.

    Parameters
    ----------
    module
        The parsed file whose functions are read.
    project
        The package index, whose modules give the import graph.
    config
        The settings that list the slow optional dependencies.

    Yields
    ------
    Violation
        One finding per local import without a valid reason.
    """
    for function in functions(module.tree):
        for node in own_nodes(function):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                problem = _local_import_problem(module, project, config, node)
                if problem:
                    yield module.violation("local-import", node, problem)


def _local_import_problem(
    module: Module, project: Project, config: Config, node: Statement
) -> str:
    """Say why a local import is not allowed, or return ``""``.

    A slow dependency may be imported where it is used. Any other module
    may be only when importing it at the top would close a cycle: every
    package module the statement loads imports this one at load time.
    """
    names = imported_modules(module, node)
    if all(name.split(".")[0] in config.heavy_modules for name in names):
        return ""
    targets = _imported_targets(project, module, node)
    in_cycle = bool(targets) and all(
        module.dotted in _reachable(project, target) for target in targets
    )
    return "" if in_cycle else MOVE_TO_TOP


def _imported_targets(
    project: Project, module: Module, node: Statement
) -> list[str]:
    """Return the package modules an import statement loads.

    A ``from`` import of a submodule (``from langslice.ops import
    views``) loads that submodule; of any other name, its source module.
    """
    sources = imported_modules(module, node)
    if isinstance(node, ast.Import):
        candidates = sources
    else:
        candidates = [
            _submodule_or_source(project, sources[0], alias.name)
            for alias in node.names
        ]
    targets = {_module_for(project, name) for name in candidates}
    return sorted(target for target in targets if target)


def _submodule_or_source(project: Project, source: str, name: str) -> str:
    """Return ``source.name`` when it is a module, else ``source``."""
    submodule = f"{source}.{name}"
    if submodule in project.modules:
        return submodule
    return source


def _reachable(project: Project, start: str) -> set[str]:
    """Return every package module ``start`` imports when it loads."""
    seen: set[str] = set()
    pending = [start]
    while pending:
        loaded = project.modules.get(pending.pop())
        if loaded is None:
            continue
        for name in _load_time_imports(loaded):
            target = _module_for(project, name)
            if target and target not in seen:
                seen.add(target)
                pending.append(target)
    return seen


def _load_time_imports(module: Module) -> Iterator[str]:
    """Yield the names a module imports when it loads.

    A ``from`` import yields its source and each name it imports, which
    may be a module.
    """
    for node in _load_time_statements(module.tree.body):
        sources = imported_modules(module, node)
        yield from sources
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                yield f"{sources[0]}.{alias.name}"


def _load_time_statements(body: list[ast.stmt]) -> Iterator[Statement]:
    """Yield the imports a block runs, outside functions and type checks."""
    for statement in body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            yield statement
        for block in _load_time_blocks(statement):
            yield from _load_time_statements(block)


def _load_time_blocks(statement: ast.stmt) -> list[list[ast.stmt]]:
    """Return the blocks of a statement that run when the module loads."""
    if isinstance(statement, ast.If):
        if dotted_name(statement.test).endswith(TYPE_CHECKING):
            return [statement.orelse]
        return [statement.body, statement.orelse]
    if isinstance(statement, ast.Try):
        handlers = [handler.body for handler in statement.handlers]
        return [
            statement.body,
            *handlers,
            statement.orelse,
            statement.finalbody,
        ]
    if isinstance(statement, ast.With):
        return [statement.body]
    return []


def _module_for(project: Project, name: str) -> str:
    """Return the package module an imported name lives in, or ``""``."""
    if name in project.modules:
        return name
    parent = name.rpartition(".")[0]
    if parent in project.modules:
        return parent
    return ""

"""Check the size and naming of modules and classes, and empty ``__init__``.

A module that has not found its topic tends to be named for having none
(``utils``, ``helpers``, ``misc``, ``common``) or for being the rest of
another module (``_part2``, ``_more``, ``_extra``, a trailing digit). A
package's ``__init__.py`` holds no logic, so code cannot hide there when a
module reaches its size limit.
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
from tools.stylecheck.rules.complexity import statement_blocks
from tools.stylecheck.rules.scopes import FUNCTION_NODES

#: Words that name no topic, banned as any ``_``-separated part of a
#: module or package name.
TOPICLESS_WORDS = frozenset(
    {"utils", "util", "helpers", "helper", "misc", "common"}
)

#: Endings that mark a module as the rest of another one.
OVERFLOW_ENDING = re.compile(r"(?:_|^)(?:part\d*|more|extra)$|(?<!\d)\d$")

#: The module-level names an ``__init__.py`` may assign.
INIT_NAMES = frozenset({"__all__", "__version__"})


@module_check("module-size")
def check_module_size(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a module longer than the limit (2.2).

    Every physical line counts. A module that reaches the limit is split
    by topic, never by halves.

    Parameters
    ----------
    module
        The parsed file to check.
    project
        The package index; not used by this check.
    config
        The settings; ``max_module_lines`` is the limit.

    Yields
    ------
    Violation
        One for a module over the limit, giving its length.
    """
    del project
    limit = config.max_module_lines
    length = len(module.lines)
    if length > limit:
        yield module.violation(
            "module-size",
            1,
            f"The module is {length} lines; the limit is {limit}.",
        )


@module_check("module-name")
def check_module_name(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a module or package named for having no topic (2.1, 2.2).

    Flagged: a name with ``utils``, ``util``, ``helpers``, ``helper``,
    ``misc`` or ``common`` as one of its ``_``-separated words, and a name
    ending in ``part``, ``partN``, ``more``, ``extra`` or a digit. A
    package is judged by its ``__init__.py``.

    Parameters
    ----------
    module
        The parsed file to check.
    project
        The package index; not used by this check.
    config
        The settings; not used by this check.

    Yields
    ------
    Violation
        One for a module whose name says no topic.
    """
    del project, config
    name = module.dotted.rsplit(".", 1)[-1]
    words = set(name.strip("_").split("_"))
    if words & TOPICLESS_WORDS or OVERFLOW_ENDING.search(name.strip("_")):
        yield module.violation(
            "module-name",
            1,
            f"The name {name!r} says no topic; name the module by the one "
            "thing it holds.",
        )


@module_check("init-logic")
def check_init_logic(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a statement in ``__init__.py`` that is not an import (2.4).

    Allowed: the docstring, imports and re-exports, ``__all__`` and
    ``__version__``, and a ``try`` or ``if`` block holding only those (an
    optional import, a version fallback, ``if TYPE_CHECKING``).

    Parameters
    ----------
    module
        The parsed file to check; other files are skipped.
    project
        The package index; not used by this check.
    config
        The settings; not used by this check.

    Yields
    ------
    Violation
        One per statement that does work.
    """
    del project, config
    if not module.is_package_init:
        return
    for index, statement in enumerate(module.tree.body):
        if index == 0 and _is_docstring(statement):
            continue
        if not _is_init_statement(statement):
            yield module.violation(
                "init-logic",
                statement,
                "An __init__.py holds only imports, __all__ and "
                "__version__; move this into a module.",
            )


def _is_docstring(statement: ast.stmt) -> bool:
    """Tell whether a statement is a bare string."""
    return isinstance(statement, ast.Expr) and (
        isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _is_init_statement(statement: ast.stmt) -> bool:
    """Tell whether a statement may appear in an ``__init__.py``."""
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        return True
    if isinstance(statement, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        return _assigns_init_name(statement)

    if not isinstance(statement, (ast.Try, ast.If)):
        return False
    for block in statement_blocks(statement):
        if not all(_is_init_statement(nested) for nested in block):
            return False
    return True


def _assigns_init_name(
    statement: ast.Assign | ast.AugAssign | ast.AnnAssign,
) -> bool:
    """Tell whether an assignment sets only ``__all__`` or ``__version__``."""
    if isinstance(statement, ast.Assign):
        targets = statement.targets
    else:
        targets = [statement.target]
    return all(
        isinstance(target, ast.Name) and target.id in INIT_NAMES
        for target in targets
    )


@module_check("class-size")
def check_class_size(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a class with more methods or lines than the limits (3.8).

    Methods are the functions written directly in the class body,
    properties included; lines run from the first decorator to the last
    line of the class.

    Parameters
    ----------
    module
        The parsed file to check.
    project
        The package index; not used by this check.
    config
        The settings; ``max_class_methods`` and ``max_class_lines`` are
        the limits.

    Yields
    ------
    Violation
        One per class over either limit, giving both measures.
    """
    del project
    method_limit = config.max_class_methods
    line_limit = config.max_class_lines
    for node in ast.walk(module.tree):
        if not isinstance(node, ast.ClassDef):
            continue
        n_methods = sum(
            1 for item in node.body if isinstance(item, FUNCTION_NODES)
        )
        first_line = node.lineno
        if node.decorator_list:
            first_line = node.decorator_list[0].lineno
        n_lines = (node.end_lineno or node.lineno) - first_line + 1

        if n_methods > method_limit or n_lines > line_limit:
            yield module.violation(
                "class-size",
                node,
                f"The class has {n_methods} methods and {n_lines} lines; "
                f"the limits are {method_limit} and {line_limit}.",
            )

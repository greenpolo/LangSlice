"""Check decorators: how many a definition carries, and where wrappers live.

Behaviour shared by many functions is written as an explicit list of steps
applied in a loop, not as wrappers composed by nesting calls. The project's
own decorators live in one module (``decorators_module``); a *decorator*
here is a function that returns a function nested in it which wraps
another: one marked with ``functools.wraps``, one that calls a parameter of
the function around it, or one that is itself such a decorator.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator

from tools.stylecheck.core import (
    Config,
    Module,
    Project,
    Violation,
    module_check,
    project_check,
)
from tools.stylecheck.rules.enclosing import (
    decorator_names,
    enclosing_function,
    parent_map,
)
from tools.stylecheck.rules.scopes import (
    FUNCTION_NODES,
    functions,
    own_nodes,
    parameters,
)

Function = ast.FunctionDef | ast.AsyncFunctionDef

#: How many wrapping calls around a function make a composed wrapper.
MIN_COMPOSITION_DEPTH = 3

#: The decorators that mark a function as a wrapper of another.
WRAPS_DECORATORS = frozenset({"wraps", "functools.wraps"})


@module_check("decorator-count")
def check_decorator_count(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a function or class with more decorators than the limit (10.3).

    Parameters
    ----------
    module
        The parsed file to check.
    project
        The package index; not used by this check.
    config
        The settings; ``max_decorators`` is the limit.

    Yields
    ------
    Violation
        One per definition over the limit.
    """
    del project
    limit = config.max_decorators
    for node in ast.walk(module.tree):
        if not isinstance(node, (*FUNCTION_NODES, ast.ClassDef)):
            continue
        count = len(node.decorator_list)
        if count > limit:
            yield module.violation(
                "decorator-count",
                node,
                f"The definition carries {count} decorators; the limit is "
                f"{limit}.",
            )


@project_check("decorator-composition")
def check_decorator_composition(
    project: Project, config: Config
) -> Iterator[Violation]:
    """Flag wrappers composed by nesting three or more calls (10.3).

    Two forms are flagged. A chain of calls, each the first positional
    argument of the one around it, in which three callees are the
    package's own decorators: ``_a(_b(_c(tool, lock), job), job)``. And
    ``_a(_b(_c(f)))``, one argument each, around a function or class of
    the package or a lambda. Write the steps as a module-level list
    applied in one loop instead.

    Parameters
    ----------
    project
        The package index; its modules tell which functions are
        decorators and which names are imported from where.
    config
        The settings; not used by this check.

    Yields
    ------
    Violation
        One per outermost composed call.
    """
    del config
    decorators = {
        module.dotted: _decorator_names(module)
        for module in project.modules.values()
    }
    for module in project.modules.values():
        yield from _compositions(module, project, decorators)


def _compositions(
    module: Module, project: Project, decorators: dict[str, set[str]]
) -> Iterator[Violation]:
    """Yield a violation for each composed wrapper call in one module."""
    known = _package_function_names(module, project)
    wrappers = _imported_decorators(module, decorators)
    parents = parent_map(module.tree)

    for node in ast.walk(module.tree):
        if not isinstance(node, ast.Call):
            continue
        if _is_first_argument(node, parents):
            continue

        depth = max(wrapper_count(node, wrappers), wrapper_depth(node, known))
        if depth >= MIN_COMPOSITION_DEPTH:
            yield module.violation(
                "decorator-composition",
                node,
                f"Wrappers are composed by {depth} nested calls; apply "
                "the steps from a module-level list in one loop.",
            )


def _is_first_argument(
    node: ast.Call, parents: dict[ast.AST, ast.AST]
) -> bool:
    """Tell whether a call is the first positional argument of a call."""
    parent = parents.get(node)
    return (
        isinstance(parent, ast.Call)
        and bool(parent.args)
        and parent.args[0] is node
    )


def wrapper_count(call: ast.Call, wrappers: set[str]) -> int:
    """Count the decorators along a chain of first positional arguments.

    Parameters
    ----------
    call
        The outermost call.
    wrappers
        The names that refer to the package's decorators.

    Returns
    -------
    int
        How many calls in the chain call one of ``wrappers``.
    """
    count = 0
    node: ast.expr = call
    while isinstance(node, ast.Call):
        callee = node.func
        if isinstance(callee, ast.Name) and callee.id in wrappers:
            count += 1
        if not node.args:
            break
        node = node.args[0]
    return count


def wrapper_depth(call: ast.Call, known: set[str]) -> int:
    """Count the one-argument calls nested around a function.

    Parameters
    ----------
    call
        The outermost call.
    known
        The names that refer to functions or classes of the package.

    Returns
    -------
    int
        3 for ``a(b(c(f)))`` when ``f`` is in ``known`` or is a lambda;
        0 when the chain of single positional arguments does not end at
        a function.
    """
    if len(call.args) != 1:
        return 0
    argument = call.args[0]
    if isinstance(argument, ast.Lambda):
        return 1
    if isinstance(argument, ast.Name) and argument.id in known:
        return 1
    if not isinstance(argument, ast.Call):
        return 0
    inner = wrapper_depth(argument, known)
    if inner == 0:
        return 0
    return inner + 1


def _decorator_names(module: Module) -> set[str]:
    """Return the names of the decorators a module defines, at any depth."""
    return {
        function.name
        for function in functions(module.tree)
        if is_decorator(function)
    }


def _imported_decorators(
    module: Module, decorators: dict[str, set[str]]
) -> set[str]:
    """Return the names that refer to package decorators in a module."""
    names = set(decorators.get(module.dotted, set()))
    for node in ast.walk(module.tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        defined = decorators.get(node.module, set())
        for alias in node.names:
            if alias.name in defined:
                names.add(alias.asname or alias.name)
    return names


def _package_function_names(module: Module, project: Project) -> set[str]:
    """Return the names in a module that refer to package functions.

    These are the module's own module-level functions and classes, and
    the public ones it imports from other package modules.
    """
    names = set(project.public_names.get(module.dotted, set()))
    for node in module.tree.body:
        if isinstance(node, (*FUNCTION_NODES, ast.ClassDef)):
            names.add(node.name)
    for node in ast.walk(module.tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        public = project.public_names.get(node.module, set())
        for alias in node.names:
            if alias.name in public:
                names.add(alias.asname or alias.name)
    return names


@module_check("decorator-location")
def check_decorator_location(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a decorator defined outside the decorators module (10.3).

    A decorator is a function that returns a function nested in it which
    wraps another. The check is off while ``decorators_module`` is unset.

    Parameters
    ----------
    module
        The parsed file to check.
    project
        The package index; not used by this check.
    config
        The settings; ``decorators_module`` names the one module where
        decorators live.

    Yields
    ------
    Violation
        One per outermost decorator found elsewhere.
    """
    del project
    if not config.decorators_module:
        return
    if module.dotted == config.decorators_module:
        return

    parents = parent_map(module.tree)
    for function in functions(module.tree):
        if not is_decorator(function):
            continue
        outer = enclosing_function(function, parents)
        if isinstance(outer, FUNCTION_NODES) and is_decorator(outer):
            continue

        yield module.violation(
            "decorator-location",
            function,
            "The function is a decorator; move it to "
            f"{config.decorators_module}.",
        )


def is_decorator(function: Function) -> bool:
    """Tell whether a function returns a nested wrapper of another function.

    Parameters
    ----------
    function
        The function to judge.

    Returns
    -------
    bool
        True when it returns a function nested directly in it that is
        marked with ``functools.wraps``, calls one of ``function``'s
        parameters, or is itself a decorator.
    """
    nested = {
        node.name: node
        for node in own_nodes(function)
        if isinstance(node, FUNCTION_NODES)
    }
    outer_names = {argument.arg for argument in parameters(function)}
    for node in own_nodes(function):
        if not isinstance(node, ast.Return):
            continue
        if not isinstance(node.value, ast.Name):
            continue
        inner = nested.get(node.value.id)
        if inner is not None and _wraps(inner, outer_names):
            return True
    return False


def _wraps(inner: Function, outer_names: set[str]) -> bool:
    """Tell whether a nested function wraps a parameter of its outer one."""
    if WRAPS_DECORATORS.intersection(decorator_names(inner)):
        return True
    for node in ast.walk(inner):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        if isinstance(callee, ast.Name) and callee.id in outer_names:
            return True
    return is_decorator(inner)

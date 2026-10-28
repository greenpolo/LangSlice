"""Checks against metaprogramming and mutable module state.

These flag code whose behaviour a reader cannot learn from the
definitions in front of them: names rebound from inside functions,
attributes forwarded at run time, classes built at run time, and reads of
another object's private attributes by string.
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
)
from tools.stylecheck.rules.annotations import last_name
from tools.stylecheck.rules.scopes import (
    BOUND_NAMES,
    call_name,
    dotted_name,
    module_matches,
    string_value,
)

#: The finding for a call of ``globals()``.
GLOBALS_CALL = (
    "globals() reads or rebinds module names at run time; use an explicit "
    "table (2.7)."
)

#: The finding for ``setattr`` on a module.
MODULE_SETATTR = (
    "setattr on a module changes it at run time; define the name in the "
    "module itself (10.2)."
)

#: Calls that build a class at run time.
CLASS_FACTORIES = frozenset(
    {"types.new_class", "dataclasses.make_dataclass", "make_dataclass"}
)

#: The one metaclass allowed: the one ``abc.ABC`` uses.
ALLOWED_METACLASS = "ABCMeta"

#: Methods that forward attribute access at run time.
FORWARDING_METHODS = frozenset({"__getattr__", "__getattribute__"})

#: Calls that reach an attribute by a name given as a string.
ATTRIBUTE_ACCESSORS = frozenset({"getattr", "hasattr", "setattr", "delattr"})


@module_check("global-statement")
def check_global_statement(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a ``global`` statement.

    Guide rule 2.7. A module name is never rebound from inside a
    function; state with a lifetime belongs to an object.

    Parameters
    ----------
    module
        The parsed file whose statements are read.
    project
        The package index, unused by this check.
    config
        The settings, unused by this check.

    Yields
    ------
    Violation
        One finding per ``global`` statement.
    """
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Global):
            names = ", ".join(node.names)
            yield module.violation(
                "global-statement",
                node,
                f"A global statement rebinds {names}; keep the state in an "
                "object with a documented lifetime (2.7).",
            )


@module_check("globals-or-module-setattr")
def check_globals_or_module_setattr(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag ``globals()`` and ``setattr`` on a module.

    Guide rules 2.7 and 10.2. A module object is a name bound by an
    ``import`` statement (or ``from package import module`` for a module
    of the package), or ``sys.modules[...]``.

    Parameters
    ----------
    module
        The parsed file whose calls are read.
    project
        The package index, used to tell modules from other imported names.
    config
        The settings, unused by this check.

    Yields
    ------
    Violation
        One finding per such call.
    """
    module_names = _module_names(module, project)
    for node in ast.walk(module.tree):
        name = call_name(node)
        if name == "globals":
            message = GLOBALS_CALL
        elif name == "setattr" and _targets_module(node, module_names):
            message = MODULE_SETATTR
        else:
            continue
        yield module.violation("globals-or-module-setattr", node, message)


def _module_names(module: Module, project: Project) -> set[str]:
    """Return the local names a module binds to imported modules."""
    names: set[str] = set()
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            names.update(_imported_package_modules(node, project))
    return names


def _imported_package_modules(
    node: ast.ImportFrom, project: Project
) -> list[str]:
    """Return the names ``from package import module`` binds to modules."""
    return [
        alias.asname or alias.name
        for alias in node.names
        if f"{node.module}.{alias.name}" in project.modules
    ]


def _targets_module(node: ast.AST, module_names: set[str]) -> bool:
    """Tell whether a ``setattr`` call's first argument is a module."""
    if not isinstance(node, ast.Call) or not node.args:
        return False
    target = node.args[0]
    if isinstance(target, ast.Name):
        return target.id in module_names
    if isinstance(target, ast.Subscript):
        return dotted_name(target.value) == "sys.modules"
    return False


@module_check("getattr-forwarding")
def check_getattr_forwarding(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag ``__getattr__`` and ``__getattribute__`` definitions.

    Guide rule 10.2. Flags them in any class of the package, and a
    module-level ``__getattr__`` outside ``__init__.py`` (the rule on
    ``__init__.py`` content covers that file).

    Parameters
    ----------
    module
        The parsed file whose definitions are read.
    project
        The package index, unused by this check.
    config
        The settings, unused by this check.

    Yields
    ------
    Violation
        One finding per forwarding definition.
    """
    for definition in module.definitions:
        if definition.node.name not in FORWARDING_METHODS:
            continue
        in_class = "." in definition.qualname
        if not in_class and module.is_package_init:
            continue
        yield module.violation(
            "getattr-forwarding",
            definition.node,
            f"{definition.qualname} forwards attribute access at run time; "
            "define the attributes, or call the real object (10.2).",
        )


@module_check("runtime-type")
def check_runtime_type(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a class built at run time.

    Guide rule 10.2. Flags ``type(name, bases, namespace)`` with three
    arguments, ``types.new_class`` and ``dataclasses.make_dataclass``.

    Parameters
    ----------
    module
        The parsed file whose calls are read.
    project
        The package index, unused by this check.
    config
        The settings, unused by this check.

    Yields
    ------
    Violation
        One finding per class built at run time.
    """
    for node in ast.walk(module.tree):
        name = call_name(node)
        builds_type = name == "type" and _argument_count(node) == 3
        if not builds_type and name not in CLASS_FACTORIES:
            continue
        yield module.violation(
            "runtime-type",
            node,
            f"{name}() builds a class at run time; write the class out "
            "(10.2).",
        )


def _argument_count(node: ast.AST) -> int:
    """Return how many positional arguments a call passes."""
    if isinstance(node, ast.Call):
        return len(node.args)
    return 0


@module_check("metaclass")
def check_metaclass(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a metaclass other than ``ABCMeta``.

    Guide rule 10.2. ``abc.ABC`` and ``metaclass=ABCMeta`` are allowed;
    every other ``metaclass=`` keyword is flagged.

    Parameters
    ----------
    module
        The parsed file whose classes are read.
    project
        The package index, unused by this check.
    config
        The settings, unused by this check.

    Yields
    ------
    Violation
        One finding per class with another metaclass.
    """
    for node in ast.walk(module.tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for keyword in node.keywords:
            if keyword.arg != "metaclass":
                continue
            if last_name(keyword.value) == ALLOWED_METACLASS:
                continue
            yield module.violation(
                "metaclass",
                node,
                f"Class {node.name!r} uses a metaclass; write the behaviour "
                "as plain methods or a decorator (10.2).",
            )


@module_check("private-getattr")
def check_private_getattr(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag reaching another object's private attribute by string.

    Guide rule 2.6. Flags ``getattr``, ``hasattr``, ``setattr`` and
    ``delattr`` with a literal name that starts with one underscore (not
    a dunder name) on an object other than ``self`` or ``cls``, outside
    ``infra_package``. Attribute syntax (``x._name``) is ruff's SLF001.

    Parameters
    ----------
    module
        The parsed file whose calls are read.
    project
        The package index, unused by this check.
    config
        The settings that name the infrastructure package.

    Yields
    ------
    Violation
        One finding per private access by string.
    """
    if module_matches(module.dotted, (config.infra_package,)):
        return
    for node in ast.walk(module.tree):
        attribute = _private_access(node)
        if attribute:
            yield module.violation(
                "private-getattr",
                node,
                f"Private attribute {attribute!r} is reached by string on "
                "another object; use its public interface, or move the "
                "access into the infrastructure package (2.6).",
            )


def _private_access(node: ast.AST) -> str:
    """Return the private name a string accessor reaches, or ``""``."""
    if not isinstance(node, ast.Call):
        return ""
    if call_name(node) not in ATTRIBUTE_ACCESSORS:
        return ""
    args = node.args
    if len(args) < 2 or dotted_name(args[0]) in BOUND_NAMES:
        return ""
    attribute = string_value(args[1]) or ""
    if not attribute.startswith("_") or attribute.startswith("__"):
        return ""
    return attribute

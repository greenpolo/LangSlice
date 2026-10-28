"""Checks on names: banned words, function name forms, one name per type.

A name's *parts* are the words it is made of when split on underscores,
lower-cased, with leading and trailing underscores dropped: ``self._held``
has the one part ``held`` and ``threshold`` has the one part
``threshold``.
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
from tools.stylecheck.rules.annotations import (
    declared_path,
    last_name,
    parse_string,
)
from tools.stylecheck.rules.scopes import BOUND_NAMES, functions, parameters

#: How many re-exports deep a class name is followed to its definition.
MAX_REEXPORTS = 8

#: Function name prefixes that promise a ``bool`` result.
PREDICATE_PREFIXES = ("is_", "has_", "can_")

#: Return annotations a predicate may carry.
PREDICATE_RETURNS = frozenset({"bool", "TypeGuard", "TypeIs"})


@module_check("banned-name")
def check_banned_name(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a variable, parameter or attribute named with a banned word.

    Guide rule 4.3. A name is flagged when one of its parts is a stem in
    ``banned_name_stems``, so ``held``, ``held_value`` and ``self._held``
    are caught while ``threshold`` is not. Each name is reported once per
    enclosing definition.

    Parameters
    ----------
    module
        The parsed file whose names are read.
    project
        The package index, unused by this check.
    config
        The settings that list the banned stems.

    Yields
    ------
    Violation
        One finding per banned name per enclosing definition.
    """
    stems = frozenset(config.banned_name_stems)
    reported: set[tuple[str, str]] = set()
    for node in ast.walk(module.tree):
        name = _bound_name(node)
        stem = _banned_part(name, stems)
        if not stem:
            continue

        place = (module.symbol_at(getattr(node, "lineno", 1)), name)
        if place not in reported:
            reported.add(place)
            message = (
                f"Name {name!r} contains the banned word {stem!r}; name "
                "what the value is instead (4.3)."
            )
            yield module.violation("banned-name", node, message)


def _bound_name(node: ast.AST) -> str:
    """Return the name a node binds, or ``""`` when it binds none."""
    if isinstance(node, ast.arg):
        return node.arg
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
        return node.attr
    if isinstance(node, ast.ExceptHandler):
        return node.name or ""
    if isinstance(node, (ast.MatchAs, ast.MatchStar)):
        return node.name or ""
    return ""


def name_parts(name: str) -> list[str]:
    """Split an identifier into its lower-cased words.

    Parameters
    ----------
    name
        An identifier such as ``_held_value``.

    Returns
    -------
    list of str
        The words between underscores, without empty ones.
    """
    return [part for part in name.lower().split("_") if part]


def _banned_part(name: str, stems: frozenset[str]) -> str:
    """Return the first banned stem among a name's parts, or ``""``."""
    for part in name_parts(name):
        if part in stems:
            return part
    return ""


@module_check("function-name")
def check_function_name(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag function names in a banned form.

    Guide rule 4.2. Flags a function whose name ends in one of
    ``banned_function_suffixes``, and a function named ``is_``, ``has_``
    or ``can_`` whose return annotation is present and is not ``bool``
    (or a ``TypeGuard``).

    Parameters
    ----------
    module
        The parsed file whose functions are read.
    project
        The package index, unused by this check.
    config
        The settings that list the banned suffixes.

    Yields
    ------
    Violation
        One finding per misnamed function.
    """
    for function in functions(module.tree):
        name = function.name
        suffix = _banned_suffix(name, config.banned_function_suffixes)
        if suffix:
            message = (
                f"Function {name!r} ends in {suffix!r}; name it for what "
                "it does or yields (4.2)."
            )
            yield module.violation("function-name", function, message)

        if _is_misdeclared_predicate(function):
            message = (
                f"Function {name!r} is named as a predicate but does not "
                "return bool; rename it or return bool (4.2)."
            )
            yield module.violation("function-name", function, message)


def _banned_suffix(name: str, suffixes: tuple[str, ...]) -> str:
    """Return the banned suffix a function name ends in, or ``""``."""
    for suffix in suffixes:
        if name.endswith(suffix) and name != suffix:
            return suffix
    return ""


def _is_misdeclared_predicate(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """Tell whether a predicate-named function declares a non-bool return."""
    if not function.name.lstrip("_").startswith(PREDICATE_PREFIXES):
        return False
    if function.returns is None:
        return False
    returns = parse_string(function.returns)
    if isinstance(returns, ast.Subscript):
        returns = returns.value
    return last_name(returns) not in PREDICATE_RETURNS


@module_check("type-name")
def check_type_name(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a parameter of a listed type that is not named for the type.

    Guide rule 4.4. ``type_names`` maps a class to the one parameter
    name it takes. A parameter annotated with that class, directly or
    through ``Optional`` or ``| None``, must be named that name or end in
    ``_<name>`` (a qualifier, as in ``source_image``). The annotation is
    resolved through the module's imports, and through the re-exports of
    the modules it names, to the package class it means. A key of the
    table is that class's dotted name (``langslice.job.job.Job``), or its
    bare name (``Job``) when the package defines one class of that name;
    a bare key that several package classes share matches none of them,
    so the table names each by its dotted name.

    Parameters
    ----------
    module
        The parsed file whose signatures are read.
    project
        The package index, which knows the package's classes.
    config
        The settings holding the table of types and names.

    Yields
    ------
    Violation
        One finding per misnamed parameter.
    """
    if not config.type_names:
        return

    resolver = ClassResolver(project)
    for function in functions(module.tree):
        for parameter in parameters(function):
            if parameter.annotation is None or parameter.arg in BOUND_NAMES:
                continue
            written = declared_path(parameter.annotation)
            target = resolver.resolve(module, written)
            expected = _expected_name(target, project, config.type_names)
            if expected and not _has_type_name(parameter.arg, expected):
                yield module.violation(
                    "type-name",
                    parameter,
                    f"Parameter {parameter.arg!r} is a {written}; name it "
                    f"{expected!r} or end it in '_{expected}' (4.4).",
                )


def _expected_name(
    target: str, project: Project, type_names: dict[str, str]
) -> str:
    """Return the name a resolved class requires, or ``""``."""
    if not target:
        return ""
    if target in type_names:
        return type_names[target]
    bare = target.rpartition(".")[2]
    if len(project.classes_named(bare)) != 1:
        return ""
    return type_names.get(bare, "")


def _has_type_name(name: str, expected: str) -> bool:
    """Tell whether a parameter name is the type's name or qualifies it."""
    bare = name.lstrip("_")
    return bare == expected or bare.endswith(f"_{expected}")


class ClassResolver:
    """Resolve a class name written in a module to the package class.

    Parameters
    ----------
    project
        The package index, whose modules and classes are read.
    """

    def __init__(self, project: Project) -> None:
        self.project = project
        self.bindings: dict[str, dict[str, str]] = {}

    def resolve(self, module: Module, written: str) -> str:
        """Return the dotted name of the package class a name means.

        Parameters
        ----------
        module
            The parsed file the name is written in.
        written
            The name as written, bare or dotted (``deformation.Job``).

        Returns
        -------
        str
            The class's dotted name, such as ``langslice.job.job.Job``;
            ``""`` when the name is not a package class.
        """
        head, _, rest = written.partition(".")
        bound = self.names_bound(module).get(head)
        if not head or bound is None:
            return ""
        dotted = f"{bound}.{rest}" if rest else bound
        for _ in range(MAX_REEXPORTS):
            if dotted in self.project.classes:
                return dotted
            dotted = self._reexported(dotted)
            if not dotted:
                return ""
        return ""

    def names_bound(self, module: Module) -> dict[str, str]:
        """Map the names a module binds by import or class to dotted names.

        Parameters
        ----------
        module
            The parsed file whose imports and classes are read.

        Returns
        -------
        dict of str to str
            Each imported name, at any depth of the file, to what it
            imports, and each module-level class to its dotted name.
        """
        if module.dotted not in self.bindings:
            self.bindings[module.dotted] = _names_bound(module)
        return self.bindings[module.dotted]

    def _reexported(self, dotted: str) -> str:
        """Return what a module's name is imported from, or ``""``."""
        module_name, _, name = dotted.rpartition(".")
        module = self.project.modules.get(module_name)
        if module is None:
            return ""
        target = self.names_bound(module).get(name, "")
        return "" if target == dotted else target


def _names_bound(module: Module) -> dict[str, str]:
    """Map a module's imported names and classes to their dotted names."""
    bound: dict[str, str] = {}
    for node in ast.walk(module.tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update(_import_bindings(module, node))
    for node in module.tree.body:
        if isinstance(node, ast.ClassDef):
            bound[node.name] = f"{module.dotted}.{node.name}"
    return bound


def _import_bindings(
    module: Module, node: ast.Import | ast.ImportFrom
) -> dict[str, str]:
    """Map the names one import statement binds to what they import."""
    if isinstance(node, ast.ImportFrom):
        source = imported_modules(module, node)[0]
        return {
            alias.asname or alias.name: f"{source}.{alias.name}"
            for alias in node.names
        }
    bound: dict[str, str] = {}
    for alias in node.names:
        if alias.asname:
            bound[alias.asname] = alias.name
        else:
            top = alias.name.partition(".")[0]
            bound[top] = top
    return bound

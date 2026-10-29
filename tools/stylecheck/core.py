"""The parsed module, the package index and the finding every check shares.

A check is a function registered with :func:`module_check` (it sees one
module at a time) or :func:`project_check` (it sees the whole package at
once). Each yields :class:`Violation` records. A violation names its place by
the qualified name of the enclosing function or class, not by line number, so
the baseline survives edits elsewhere in the file.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import io
import tokenize
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, TypeGuard, TypeVar

from tools.stylecheck.jsondata import as_list, as_table, as_text, load_toml


class Severity(enum.StrEnum):
    """How a violation counts: an error fails the check, a warning is shown."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Violation:
    """One finding of one check.

    Parameters
    ----------
    rule
        The check's name, such as ``"cognitive-complexity"``.
    path
        The file, relative to the repository root, with forward slashes.
    symbol
        The qualified name of the enclosing definition (``"Job.save"``), or
        ``"<module>"`` for code at module level.
    line
        The line the finding points at, for the report only.
    message
        One sentence saying what is wrong and what the limit is.
    severity
        Whether the finding fails the check.
    """

    rule: str
    path: str
    symbol: str
    line: int
    message: str
    severity: Severity = Severity.ERROR

    @property
    def key(self) -> tuple[str, str, str]:
        """Return the baseline key: rule, path and symbol, without the line."""
        return (self.rule, self.path, self.symbol)


@dataclass(frozen=True)
class Definition:
    """A function or class in a module, with the lines it spans."""

    qualname: str
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    first_line: int
    last_line: int


@dataclass
class Module:
    """One parsed source file.

    Parameters
    ----------
    path
        The file on disk.
    relpath
        The file relative to the repository root, with forward slashes.
    dotted
        The import name (``"langslice.core.space"``); for a package's
        ``__init__.py`` it is the package's name.
    source
        The file's text.
    tree
        The parsed syntax tree.
    """

    path: Path
    relpath: str
    dotted: str
    source: str
    tree: ast.Module
    lines: list[str] = field(init=False)
    definitions: list[Definition] = field(init=False)
    comments: list[tuple[int, str]] = field(init=False)

    def __post_init__(self) -> None:
        self.lines = self.source.splitlines()
        self.definitions = list(_definitions(self.tree, prefix=""))
        self.comments = list(_comments(self.source))

    @property
    def is_package_init(self) -> bool:
        """Return whether the file is a package's ``__init__.py``."""
        return self.path.name == "__init__.py"

    def symbol_at(self, line: int) -> str:
        """Return the qualified name of the innermost definition at a line.

        Parameters
        ----------
        line
            A 1-based line number in the file.

        Returns
        -------
        str
            The definition's qualified name, or ``"<module>"`` when the line
            is outside every function and class.
        """
        innermost = "<module>"
        for definition in self.definitions:
            if definition.first_line <= line <= definition.last_line:
                innermost = definition.qualname
        return innermost

    def violation(
        self,
        rule: str,
        node: ast.AST | int,
        message: str,
        *,
        severity: Severity = Severity.ERROR,
    ) -> Violation:
        """Make a violation located at a node or a line of this module.

        Parameters
        ----------
        rule
            The check's name.
        node
            The syntax node the finding is about, or a 1-based line number.
        message
            One sentence saying what is wrong and what the limit is.
        severity
            Whether the finding fails the check.

        Returns
        -------
        Violation
            The finding, named by the enclosing definition.
        """
        if isinstance(node, int):
            line = node
        else:
            line = getattr(node, "lineno", 1)
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            symbol = self.symbol_at(node.lineno)
        else:
            symbol = self.symbol_at(line)
        return Violation(rule, self.relpath, symbol, line, message, severity)


def _definitions(tree: ast.AST, prefix: str) -> Iterator[Definition]:
    """Yield every function and class under a node, outermost first."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            qualname = f"{prefix}{node.name}"
            first_line = node.lineno
            if node.decorator_list:
                first_line = node.decorator_list[0].lineno
            last_line = node.end_lineno or node.lineno
            yield Definition(qualname, node, first_line, last_line)
            yield from _definitions(node, prefix=f"{qualname}.")
        else:
            yield from _definitions(node, prefix=prefix)


def _comments(source: str) -> Iterator[tuple[int, str]]:
    """Yield the line and text of every ``#`` comment in a source file."""
    readline = io.StringIO(source).readline
    try:
        for token in tokenize.generate_tokens(readline):
            if token.type == tokenize.COMMENT:
                yield token.start[0], token.string
    except (tokenize.TokenError, SyntaxError):
        return


@dataclass
class ClassInfo:
    """What the index knows about a class defined in the package."""

    dotted: str
    module: str
    attributes: set[str]
    is_dataclass: bool


@dataclass
class Project:
    """Every module in the checked trees, and what is known across them.

    Parameters
    ----------
    root
        The repository root.
    modules
        Every parsed module, keyed by its dotted name.
    """

    root: Path
    modules: dict[str, Module]
    classes: dict[str, ClassInfo] = field(init=False)
    imports: dict[str, set[str]] = field(init=False)
    public_names: dict[str, set[str]] = field(init=False)

    def __post_init__(self) -> None:
        self.classes = {}
        self.imports = {}
        self.public_names = {}
        for module in self.modules.values():
            self.imports[module.dotted] = set(_imported_modules(module))
            self.public_names[module.dotted] = set(_public_names(module.tree))
            for info in _classes(module):
                self.classes[info.dotted] = info

    def classes_named(self, name: str) -> list[ClassInfo]:
        """Return the package's classes that have one bare name.

        Parameters
        ----------
        name
            A class name without its module, such as ``"Job"``.

        Returns
        -------
        list of ClassInfo
            Every class of that name, in any module; empty when none is.
        """
        return [
            info
            for info in self.classes.values()
            if info.dotted.rsplit(".", 1)[-1] == name
        ]


def _imported_modules(module: Module) -> Iterator[str]:
    """Yield the dotted names a module imports, at any depth in the file.

    A ``from`` import yields its source module and each name it imports
    from it, since the name may itself be a module.
    """
    for node in ast.walk(module.tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        sources = imported_modules(module, node)
        yield from sources
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                yield f"{sources[0]}.{alias.name}"


def imported_modules(
    module: Module, node: ast.Import | ast.ImportFrom
) -> list[str]:
    """Return the absolute names of the modules an import statement reads.

    Parameters
    ----------
    module
        The parsed file the statement is in, against which a relative
        import is resolved.
    node
        An ``import`` or ``from ... import`` statement.

    Returns
    -------
    list of str
        One name per module of an ``import``; the one source module of a
        ``from`` import.
    """
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if not node.level:
        return [node.module or ""]
    package = module.dotted.split(".")
    if not module.is_package_init:
        package = package[:-1]
    base = package[: len(package) - node.level + 1]
    if node.module:
        base.append(node.module)
    return [".".join(base)]


def _public_names(tree: ast.Module) -> Iterator[str]:
    """Yield the module-level functions and classes not starting with ``_``."""
    for node in tree.body:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            if not node.name.startswith("_"):
                yield node.name


def _classes(module: Module) -> Iterator[ClassInfo]:
    """Yield each module-level class with the attributes it declares."""
    for node in module.tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        is_dataclass = any(
            "dataclass" in ast.unparse(decorator)
            for decorator in node.decorator_list
        )
        yield ClassInfo(
            dotted=f"{module.dotted}.{node.name}",
            module=module.dotted,
            attributes=set(_declared_attributes(node)),
            is_dataclass=is_dataclass,
        )


def _declared_attributes(node: ast.ClassDef) -> Iterator[str]:
    """Yield a class's annotated fields, methods and ``self`` attributes."""
    for item in ast.walk(node):
        if isinstance(item, ast.AnnAssign):
            if isinstance(item.target, ast.Name):
                yield item.target.id
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield item.name
        elif _is_self_attribute_store(item):
            yield item.attr


def _is_self_attribute_store(node: ast.AST) -> TypeGuard[ast.Attribute]:
    """Tell whether a node assigns an attribute of ``self``."""
    if not isinstance(node, ast.Attribute):
        return False
    is_store = isinstance(node.ctx, ast.Store)
    return (
        is_store
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


@dataclass(frozen=True)
class Config:
    """The checker's settings, read from ``[tool.stylecheck]``.

    Every field has the default the style guide states; ``pyproject.toml``
    may override any of them. ``layers`` is read from the import-linter
    ``layers`` contract, highest layer first, so the docstring layer check
    and the import rule share one source.
    """

    paths: tuple[str, ...] = ("src/langslice", "tools/stylecheck")
    baseline: str = "tools/stylecheck/baseline.json"
    max_cognitive_complexity: int = 15
    max_nesting: int = 4
    max_body_lines: int = 100
    max_parameters: int = 6
    max_module_lines: int = 800
    max_class_methods: int = 30
    max_class_lines: int = 600
    max_paragraph_lines: int = 15
    max_returned_values: int = 3
    max_decorators: int = 2
    max_docstring_references: int = 4
    min_parameter_words: int = 3
    banned_name_stems: tuple[str, ...] = (
        "held",
        "shown",
        "drawn",
        "placed",
        "asked",
        "wanted",
        "picked",
        "found",
        "thing",
        "stuff",
    )
    banned_function_suffixes: tuple[str, ...] = ("_words", "_of")
    banned_summary_words: tuple[str, ...] = (
        "The",
        "A",
        "An",
        "Whether",
        "What",
        "One",
        "Every",
        "Where",
    )
    type_names: dict[str, str] = field(default_factory=dict)
    heavy_modules: tuple[str, ...] = (
        "torch",
        "SimpleITK",
        "fireants",
        "itk",
        "cv2",
    )
    json_boundary_modules: tuple[str, ...] = ()
    cli_modules: tuple[str, ...] = ()
    decorators_module: str = ""
    checkers_module: str = "langslice.core.checks"
    codes_module: str = ""
    infra_package: str = ""
    docs_renderer_module: str = ""
    layers: tuple[tuple[str, ...], ...] = ()

    @classmethod
    def load(cls, pyproject: Path) -> Config:
        """Read the settings from a ``pyproject.toml``.

        Parameters
        ----------
        pyproject
            The project file; a missing ``[tool.stylecheck]`` table means
            every default applies.

        Returns
        -------
        Config
            The settings, with lists turned into tuples.
        """
        document = load_toml(pyproject.read_text(encoding="utf-8"))
        tool = as_table(document.get("tool")) or {}
        table = as_table(tool.get("stylecheck")) or {}
        settings = {
            name: _tuple_if_list(value) for name, value in table.items()
        }

        importlinter = as_table(tool.get("importlinter")) or {}
        contracts = as_list(importlinter.get("contracts")) or []
        settings["layers"] = _layers(contracts)
        return dataclasses.replace(cls(), **settings)


def _tuple_if_list(value: object) -> object:
    """Return a TOML array as a tuple, and any other value unchanged."""
    items = as_list(value)
    return value if items is None else tuple(items)


def _layers(contracts: list[object]) -> tuple[tuple[str, ...], ...]:
    """Return the packages of import-linter's layers contract, highest first.

    Each layer is a tuple of the packages it holds side by side
    (``"doors : agent"``), named under the contract's first container.
    """
    for contract in contracts:
        table = as_table(contract) or {}
        if as_text(table.get("type")) != "layers":
            continue
        containers = as_list(table.get("containers")) or [""]
        container = str(containers[0])
        layers = as_list(table.get("layers")) or []
        return tuple(_layer_packages(container, layer) for layer in layers)
    return ()


def _layer_packages(container: str, layer: object) -> tuple[str, ...]:
    """Split one layer of the contract into its packages' dotted names."""
    names = str(layer).split(":")
    return tuple(f"{container}.{name.strip()}" for name in names)


ModuleCheck = Callable[[Module, Project, Config], Iterator[Violation]]
ProjectCheck = Callable[[Project, Config], Iterator[Violation]]

#: A check of either kind, as :class:`Registration` records it.
Check = TypeVar("Check", ModuleCheck, ProjectCheck)

#: Every registered check, by name, for the life of the process. A check
#: registers itself when the module that defines it is imported;
#: importing :mod:`tools.stylecheck.rules` imports them all.
MODULE_CHECKS: dict[str, ModuleCheck] = {}
PROJECT_CHECKS: dict[str, ProjectCheck] = {}

#: The checks that report at most one error per file, at its module level:
#: a state of the whole file, which every piece of a split file shares.
PER_FILE_CHECKS: set[str] = set()


@dataclass(frozen=True)
class Registration(Generic[Check]):
    """Record the check it decorates in a table, under the check's name.

    Parameters
    ----------
    table
        The table the check goes into, ``MODULE_CHECKS`` or
        ``PROJECT_CHECKS``.
    name
        The check's name, used in reports and the baseline; it is also the
        ``rule`` of every violation the check yields.
    per_file
        Whether the check reports at most one error per file, at its
        module level.
    """

    table: dict[str, Check]
    name: str
    per_file: bool = False

    def __call__(self, check: Check) -> Check:
        """Record a check and return it unchanged.

        Parameters
        ----------
        check
            The function that yields the check's violations.

        Returns
        -------
        Check
            The same function.

        Raises
        ------
        ValueError
            When a check of either kind already has the name.
        """
        if self.name in MODULE_CHECKS or self.name in PROJECT_CHECKS:
            raise ValueError(f"Two checks are named {self.name!r}.")
        self.table[self.name] = check
        if self.per_file:
            PER_FILE_CHECKS.add(self.name)
        return check


def module_check(
    name: str, *, per_file: bool = False
) -> Registration[ModuleCheck]:
    """Register a check that looks at one module at a time.

    Parameters
    ----------
    name
        The check's name, used in reports and the baseline.
    per_file
        Declare that the check reports at most one error per file, at its
        module level, by construction.

    Returns
    -------
    Registration
        The decorator that records the check in ``MODULE_CHECKS``.
    """
    return Registration(MODULE_CHECKS, name, per_file)


def project_check(name: str) -> Registration[ProjectCheck]:
    """Register a check that looks at the whole package at once.

    Parameters
    ----------
    name
        The check's name, used in reports and the baseline.

    Returns
    -------
    Registration
        The decorator that records the check in ``PROJECT_CHECKS``.
    """
    return Registration(PROJECT_CHECKS, name)


def function_body(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.stmt]:
    """Return a function's statements without its docstring.

    Parameters
    ----------
    node
        The function whose body is read.

    Returns
    -------
    list of ast.stmt
        The body's statements, the first left out when it is a string.
    """
    body = node.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def load_project(root: Path, config: Config) -> Project:
    """Parse every Python file under the configured paths.

    Parameters
    ----------
    root
        The repository root.
    config
        The settings; ``config.paths`` lists the trees to read.

    Returns
    -------
    Project
        Every module that parsed. A file with a syntax error is skipped;
        ruff reports it.
    """
    modules: dict[str, Module] = {}
    for tree_path in config.paths:
        base = root / tree_path
        for path in sorted(base.rglob("*.py")):
            relpath = path.relative_to(root).as_posix()
            source = path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source, filename=relpath)
            except SyntaxError:
                continue
            dotted = module_name_from_path(relpath)
            modules[dotted] = Module(path, relpath, dotted, source, tree)
    return Project(root, modules)


def module_name_from_path(relpath: str) -> str:
    """Return a file's import name.

    Parameters
    ----------
    relpath
        The file relative to the repository root, with forward slashes.

    Returns
    -------
    str
        The dotted name, without a leading ``src`` folder; a package's
        ``__init__.py`` gives the package's name.
    """
    parts = list(Path(relpath).with_suffix("").parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)

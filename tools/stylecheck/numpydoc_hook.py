"""Docstring validation by numpydoc, reported as the checker's own.

STYLE.md 6.1 and 6.2 are numpydoc's: a public function, method or class
has a numpydoc docstring whose ``Parameters`` list every parameter and no
other, a ``Returns`` when it returns a value, and a one-line summary that
starts with a capital, ends with a full stop and is not a third-person
verb. The checks run are the ``checks`` of ``[tool.numpydoc_validation]``
in ``pyproject.toml``, through numpydoc's own AST validator, on the trees
the checker reads. Each finding is a violation named ``numpydoc:<CODE>``.

The tool stubs' docstrings are the text a model reads, not API
documentation; the runner drops the findings on them
(:mod:`tools.stylecheck.model_text`). Three departures from numpydoc's
hook keep the other findings precise: a function defined inside another
function is not public and is skipped; a property has no ``Returns`` (it
is documented as a noun); and a dataclass whose docstring has
``Parameters`` is compared with its fields, which the hook would otherwise
take for unknown parameters. ``# numpydoc ignore``
comments are not honoured: the baseline is the one list of exceptions.
"""

# numpydoc ships no type information, and annotates a validator's
# parameters as a one-item tuple; this module types what it uses.
# pyright: reportMissingTypeStubs=false, reportIncompatibleMethodOverride=false

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpydoc.validate
from numpydoc.hooks.validate_docstrings import AstValidator
from numpydoc.validate import ERROR_MSGS, get_validation_checks

from tools.stylecheck.core import Config, Module, Project, Violation
from tools.stylecheck.jsondata import as_list, as_table, as_text, load_toml

#: Decorators that make a function a property, which has no Returns.
PROPERTY_DECORATORS = ("property", "cached_property", "setter", "getter")

#: numpydoc's ``validate``, with the report type its documentation gives.
VALIDATE = cast(Callable[..., dict[str, object]], numpydoc.validate.validate)

#: Annotations that mark a class attribute as no constructor field.
NOT_FIELDS = ("ClassVar", "KW_ONLY")

#: Base classes whose annotated fields are the constructor's parameters.
FIELD_BASES = ("NamedTuple", "BaseModel")


@dataclass(frozen=True)
class NumpydocSettings:
    """Hold the numpydoc checks to run and the names to leave out.

    Parameters
    ----------
    checks
        The numpydoc codes to report, such as ``GL08`` and ``PR01``.
    exclude
        Regular expressions; an object whose dotted name (file stem, then
        qualified name) matches one is not validated.
    """

    checks: frozenset[str]
    exclude: tuple[str, ...]

    @classmethod
    def from_pyproject(cls, pyproject: Path) -> NumpydocSettings:
        """Read the settings from ``[tool.numpydoc_validation]``.

        Parameters
        ----------
        pyproject
            The project file that holds the table.

        Returns
        -------
        NumpydocSettings
            The checks and exclusions the table gives; no checks when the
            table is missing.
        """
        checks, exclude = _numpydoc_options(pyproject)
        expanded: set[str] = get_validation_checks(set(checks))
        return cls(frozenset(expanded), tuple(exclude))


def numpydoc_violations(project: Project, config: Config) -> list[Violation]:
    """Validate every public docstring of the package.

    Parameters
    ----------
    project
        The parsed package; its root holds ``pyproject.toml``, whose
        ``[tool.numpydoc_validation]`` table lists the checks.
    config
        The checker's settings, unused: numpydoc's come from its table.

    Returns
    -------
    list of Violation
        One error per numpydoc finding, named ``numpydoc:<CODE>``.
    """
    pyproject = project.root / "pyproject.toml"
    settings = NumpydocSettings.from_pyproject(pyproject)
    if not settings.checks:
        return []
    violations: list[Violation] = []
    for module in project.modules.values():
        violations.extend(module_violations(module, settings))
    return violations


def module_violations(
    module: Module, settings: NumpydocSettings
) -> list[Violation]:
    """Validate the docstrings of one module.

    Parameters
    ----------
    module
        The parsed file.
    settings
        The checks to report and the names to leave out.

    Returns
    -------
    list of Violation
        One error per finding, at the definition the docstring belongs to.
    """
    stem = module.path.stem
    targets = [_Target(module.tree, "<module>", 1, [])]
    targets.extend(_public_targets(module.tree, stem, settings.exclude))
    violations: list[Violation] = []
    for target in targets:
        violations.extend(_target_violations(module, target, settings))
    return violations


def _target_violations(
    module: Module, target: _Target, settings: NumpydocSettings
) -> list[Violation]:
    """Return numpydoc's findings on one definition, as violations."""
    name = module.path.stem
    if target.qualname != "<module>":
        name = f"{name}.{target.qualname}"

    report = VALIDATE(
        name,
        _Validator,
        ast_node=target.node,
        filename=str(module.path),
        ancestry=target.ancestry,
    )
    # cast: numpydoc reports each error as a (code, message) pair
    errors = cast(list[tuple[str, str]], report["errors"])

    return [
        Violation(
            f"numpydoc:{code}",
            module.relpath,
            target.qualname,
            target.line,
            f"{description.rstrip('.')}.",
        )
        for code, description in errors
        if code in settings.checks
    ]


def numpydoc_explanation(code: str, root: Path) -> str:
    """Return numpydoc's message for one of its codes, or the codes run.

    Parameters
    ----------
    code
        A numpydoc code such as ``PR01``; empty for the whole set.
    root
        The repository root, whose ``pyproject.toml`` lists the codes.

    Returns
    -------
    str
        The code's message template, the codes the checker runs when
        ``code`` is empty, or a sentence saying the code is unknown.
    """
    if not code:
        settings = NumpydocSettings.from_pyproject(root / "pyproject.toml")
        return (
            "numpydoc validation of public objects with "
            f"{', '.join(sorted(settings.checks))}; each finding is "
            "numpydoc:<CODE>."
        )
    messages: dict[str, str] = ERROR_MSGS
    if code not in messages:
        return f"numpydoc has no check {code!r}."
    return f"numpydoc {code}: {messages[code]}"


@dataclass(frozen=True)
class _Target:
    """A node to validate, its qualified name and its enclosing nodes."""

    node: ast.AST
    qualname: str
    line: int
    ancestry: list[ast.AST]


def _public_targets(
    tree: ast.Module, stem: str, exclude: tuple[str, ...]
) -> Iterator[_Target]:
    """Yield the public classes, functions and methods of a module."""
    pending: list[tuple[ast.stmt, str, list[ast.AST]]] = [
        (node, "", [tree]) for node in tree.body
    ]

    while pending:
        node, prefix, ancestry = pending.pop(0)
        if not isinstance(
            node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        qualname = f"{prefix}{node.name}"
        if _is_private(f"{stem}.{qualname}", exclude) or _is_overload(node):
            continue
        yield _Target(node, qualname, node.lineno, ancestry)
        if isinstance(node, ast.ClassDef):
            inner: list[ast.AST] = [*ancestry, node]
            pending.extend(
                (child, f"{qualname}.", inner) for child in node.body
            )


def _is_private(name: str, exclude: tuple[str, ...]) -> bool:
    """Return whether a dotted name matches one of the exclusions."""
    return any(re.search(pattern, name) for pattern in exclude)


def _is_overload(
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """Return whether a definition is a ``typing.overload`` stub."""
    return any(
        _decorator_name(item) == "overload" for item in node.decorator_list
    )


def _decorator_name(decorator: ast.expr) -> str:
    """Return the last dotted part of a decorator, without its call."""
    if isinstance(decorator, ast.Call):
        decorator = decorator.func
    if isinstance(decorator, ast.Attribute):
        return decorator.attr
    if isinstance(decorator, ast.Name):
        return decorator.id
    return ""


class _Validator(AstValidator):
    """numpydoc's AST validator, with the three departures of this module."""

    @property
    def source_file_name(self) -> str:
        """Return no file, so ``# numpydoc ignore`` comments are not read."""
        return ""

    @property
    def method_source(self) -> str:
        """Return the function's source, rebuilt from its syntax tree."""
        return ast.unparse(self.node)

    @property
    def method_returns_something(self) -> bool:
        """Return whether the function returns a value; a property does not."""
        node = self.node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names = {_decorator_name(item) for item in node.decorator_list}
            if names.intersection(PROPERTY_DECORATORS):
                return False
        returns: bool = super().method_returns_something
        return returns

    @property
    def signature_parameters(self) -> tuple[str, ...]:
        """Return the parameters, taking a documented record's fields."""
        parameters: tuple[str, ...] = super().signature_parameters
        node = self.node
        if not isinstance(node, ast.ClassDef) or parameters:
            return parameters
        documented: list[object] = self.doc["Parameters"]
        if documented and _is_record(node):
            return tuple(_fields(node))
        return parameters


def _is_record(node: ast.ClassDef) -> bool:
    """Return whether a class takes its annotated fields as parameters."""
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if item.name == "__init__":
                return False
    decorators = {_decorator_name(item) for item in node.decorator_list}
    bases = {_decorator_name(item) for item in node.bases}
    return "dataclass" in decorators or bool(bases.intersection(FIELD_BASES))


def _fields(node: ast.ClassDef) -> Iterator[str]:
    """Yield a record's constructor fields, in the order they appear."""
    for item in node.body:
        if not isinstance(item, ast.AnnAssign):
            continue
        if not isinstance(item.target, ast.Name):
            continue
        annotation = ast.unparse(item.annotation)
        if any(name in annotation for name in NOT_FIELDS):
            continue
        if _is_init_false(item.value):
            continue
        yield item.target.id


def _is_init_false(value: ast.expr | None) -> bool:
    """Return whether a field's default is ``field(init=False)``."""
    if not isinstance(value, ast.Call):
        return False
    keywords = {keyword.arg: keyword.value for keyword in value.keywords}
    init = keywords.get("init")
    return isinstance(init, ast.Constant) and init.value is False


def _numpydoc_options(pyproject: Path) -> tuple[list[str], list[str]]:
    """Return the ``checks`` and ``exclude`` of numpydoc's table."""
    document = load_toml(pyproject.read_text(encoding="utf-8"))
    tool = as_table(document.get("tool"))
    if tool is None:
        return [], []
    table = as_table(tool.get("numpydoc_validation"))
    if table is None:
        return [], []
    return _strings(table.get("checks")), _strings(table.get("exclude"))


def _strings(value: object) -> list[str]:
    """Return a TOML value as a list of strings, accepting one string."""
    text = as_text(value)
    if text is not None:
        return [text]
    items = as_list(value)
    if items is None:
        return []
    return [str(item) for item in items]

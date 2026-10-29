"""Check that a docstring in a lower layer does not name a higher one.

The layers are import-linter's, highest first, as the settings hold
them. A docstring names a higher layer when it writes a dotted path into
it (``langslice.doors.library`` anywhere in the text, or ``doors.cli``
and ``doors/cli.py`` in double backticks or a role), or when it writes,
in double backticks or a role, a public name defined only in higher
layers. Only *distinctive* names count: a CamelCase name of four letters
or more, or a lower-case name with an underscore in it. A plain word such
as ``save`` names too many things to be told apart. The tools' names
(each ``ToolDef("name", ...)`` of ``TOOLSET_MODULE``, the module that
lists the tools) are the vocabulary every layer shares, not references
upward, and do not count by bare name; a dotted path to the higher
module that defines one still does.
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
from tools.stylecheck.rules.docstrings import (
    Docstring,
    backticked_names,
    docstrings,
)
from tools.stylecheck.rules.scopes import (
    FUNCTION_NODES,
    call_name,
    module_matches,
    parameters,
    string_value,
)

#: The name of the layered package.
PACKAGE_ROOT = "langslice"

#: A dotted path into the package written anywhere in a docstring.
PACKAGE_PATH_PATTERN = re.compile(r"\blangslice\.(\w+)")

#: A class name worth matching: capitalised, with a lower-case letter.
CLASS_PATTERN = re.compile(r"[A-Z][A-Za-z0-9]*[a-z][A-Za-z0-9]*")

#: A function name worth matching: lower-case words joined by ``_``.
FUNCTION_PATTERN = re.compile(r"_?[a-z][a-z0-9]*(?:_[a-z0-9]+)+")

#: The shortest class name worth matching.
MIN_CLASS_LETTERS = 4

#: The module that lists the tools, one ``ToolDef`` call per tool.
TOOLSET_MODULE = "langslice.job.settings.tool_table"

#: The call that declares one tool, its name the first argument.
TOOL_DECLARATION = "ToolDef"

#: The characters that separate parts of a dotted path or a file path.
PATH_SEPARATORS = re.compile(r"[./]")


def layer_rank(dotted: str, layers: tuple[tuple[str, ...], ...]) -> int:
    """Return the position of a module's layer, highest layer first.

    Parameters
    ----------
    dotted
        The module's import name.
    layers
        The layered packages from the settings, highest first.

    Returns
    -------
    int
        The index of the layer that holds the module, or -1 when no
        layer does.
    """
    for rank, packages in enumerate(layers):
        if module_matches(dotted, packages):
            return rank
    return -1


@module_check("docstring-layer")
def check_docstring_layer(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a docstring that names a layer above its own module's layer.

    Guide rule 6.4. See the module docstring for what counts as naming a
    higher layer. Each name is reported once per docstring.

    Parameters
    ----------
    module
        The parsed file whose docstrings are read.
    project
        The package index, which lists each module's public names.
    config
        The settings that hold the layers.

    Yields
    ------
    Violation
        One finding per higher-layer name per docstring.
    """
    rank = layer_rank(module.dotted, config.layers)
    if rank <= 0:
        return
    higher = _higher_packages(config.layers, rank)
    names = _names_only_above(project, config.layers, rank)
    for docstring in docstrings(module.tree):
        for token, package in _upward_names(docstring, higher, names):
            yield module.violation(
                "docstring-layer",
                docstring.line,
                f"The docstring names {token!r} from the higher layer "
                f"{package.removeprefix('langslice.')!r}; a lower layer "
                "does not mention the layers above it (6.4).",
            )


def _higher_packages(
    layers: tuple[tuple[str, ...], ...], rank: int
) -> dict[str, str]:
    """Map each package above a layer, by short and full name, to itself."""
    packages: dict[str, str] = {}
    for group in layers[:rank]:
        for package in group:
            packages[package] = package
            packages[package.rpartition(".")[2]] = package
    return packages


def _names_only_above(
    project: Project, layers: tuple[tuple[str, ...], ...], rank: int
) -> dict[str, str]:
    """Map each distinctive public name defined only above a layer.

    The value is the package of the higher layer that defines it. The
    tools' names are left out.
    """
    above: dict[str, str] = {}
    elsewhere = tool_names(project)
    for dotted, names in project.public_names.items():
        module_rank = layer_rank(dotted, layers)
        if 0 <= module_rank < rank:
            package = _layer_package(dotted, layers[module_rank])
            above.update((name, package) for name in names)
        else:
            elsewhere.update(names)
    return {
        name: package
        for name, package in above.items()
        if name not in elsewhere and _is_distinctive(name)
    }


def tool_names(project: Project) -> set[str]:
    """Return the names of the tools the toolset module declares.

    Parameters
    ----------
    project
        The package index; its module ``TOOLSET_MODULE`` is read.

    Returns
    -------
    set of str
        The first argument of each ``ToolDef(...)`` call there that is a
        string; empty when the package has no such module.
    """
    module = project.modules.get(TOOLSET_MODULE)
    if module is None:
        return set()
    names: set[str] = set()
    for node in ast.walk(module.tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if call_name(node).rpartition(".")[2] != TOOL_DECLARATION:
            continue
        name = string_value(node.args[0])
        if name is not None:
            names.add(name)
    return names


def _layer_package(dotted: str, packages: tuple[str, ...]) -> str:
    """Return which package of a layer holds a module."""
    for package in packages:
        if module_matches(dotted, (package,)):
            return package
    return ""


def _is_distinctive(name: str) -> bool:
    """Tell whether a name is specific enough to match in prose."""
    if FUNCTION_PATTERN.fullmatch(name):
        return True
    if len(name) < MIN_CLASS_LETTERS:
        return False
    return CLASS_PATTERN.fullmatch(name) is not None


def _upward_names(
    docstring: Docstring, higher: dict[str, str], names: dict[str, str]
) -> list[tuple[str, str]]:
    """Return each higher-layer name a docstring writes, with its package."""
    upward: dict[str, str] = {}
    for match in PACKAGE_PATH_PATTERN.finditer(docstring.text):
        package = higher.get(f"langslice.{match.group(1)}")
        if package:
            upward.setdefault(match.group(0), package)
    own = _parameter_names(docstring)
    for token in backticked_names(docstring.text):
        package = _token_package(token, higher, names, own)
        if package:
            upward.setdefault(token, package)
    return list(upward.items())


def _parameter_names(docstring: Docstring) -> set[str]:
    """Return the parameter names of the function a docstring documents."""
    if not isinstance(docstring.owner, FUNCTION_NODES):
        return set()
    return {parameter.arg for parameter in parameters(docstring.owner)}


def _token_package(
    token: str,
    higher: dict[str, str],
    names: dict[str, str],
    own: set[str],
) -> str:
    """Return the higher package a backticked token names, or ``""``."""
    parts = PATH_SEPARATORS.split(token.lstrip("~.").removesuffix("()"))
    if parts[0] in own:
        return ""
    if len(parts) > 1 and parts[0] == PACKAGE_ROOT:
        return higher.get(f"{PACKAGE_ROOT}.{parts[1]}", "")
    if len(parts) > 1 and parts[0] in higher:
        return higher[parts[0]]
    for part in (parts[0], parts[-1]):
        if part in names:
            return names[part]
    return ""

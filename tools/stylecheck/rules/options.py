"""Check that an option value compared as a bare string lives in one module.

An *option site* is a place where a string literal is tested against a
value: ``mode == "mi"``, ``mode != "mi"``, ``mode in ("mi", "ncc")``,
``name.startswith("mi")`` or ``case "mi":``. Only literals that look like
an option value count: lower-case words joined by ``_`` or ``-``, at
least two characters long, and not one of the words used to parse a
boolean from text (``"true"``, ``"yes"``, ``"on"`` and their opposites).
Upper-case codes are left to the error-code-literal check; paths, file
suffixes, sentences, dunder names and single letters are not options.
The names of Python's builtins (``"dict"``, ``"setattr"``) are compared
by code that reads code, and are not options either. A literal on the
left of ``in`` (``"key" in payload``) tests a key, not an option, and is
not a site. Nor is a comparison with what the platform or an array
reports about itself (``os.name``, ``sys.platform``, a ``dtype``).
"""

from __future__ import annotations

import ast
import builtins
import re
from collections.abc import Iterator

from tools.stylecheck.core import Config, Project, Violation, project_check
from tools.stylecheck.rules.scopes import call_name, dotted_name, string_value

#: What an option value looks like: lower-case words joined by ``_``/``-``.
OPTION_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:[_-][a-z0-9]+)*")

#: Words that parse a boolean from text; they are not the package's options.
BOOLEAN_WORDS = frozenset({"true", "false", "yes", "no", "on", "off"})

#: Python's builtin names, which code that reads code compares against.
BUILTIN_NAMES = frozenset(dir(builtins))

#: Values the platform reports, compared with its own names, not options.
PLATFORM_SUBJECTS = frozenset({"os.name", "sys.platform", "platform.system"})

#: Methods that test a string's ends against a literal.
END_TESTS = frozenset({"startswith", "endswith"})

#: How many module names a message lists before it says how many more.
LISTED_MODULES = 3

Site = tuple[str, ast.AST]


@project_check("shared-option-literal")
def check_shared_option_literal(
    project: Project, config: Config
) -> Iterator[Violation]:
    """Flag an option value compared as a bare string in several modules.

    Guide rule 5.2. Collects every option site (see the module docstring
    for what counts) in every module; a literal compared in two or more
    modules is reported at every site, in every one of those modules. A
    module that adds a site of a shared literal therefore always gets a
    finding of its own, and no other module's findings depend on the
    order of paths.

    Parameters
    ----------
    project
        The package index whose modules are read.
    config
        The settings, unused by this check.

    Yields
    ------
    Violation
        One finding per site of a literal shared between modules.
    """
    del config
    sites: dict[str, dict[str, list[ast.AST]]] = {}
    for module in project.modules.values():
        for literal, node in _option_sites(module.tree):
            per_module = sites.setdefault(literal, {})
            per_module.setdefault(module.dotted, []).append(node)

    for literal, per_module in sites.items():
        if len(per_module) < 2:
            continue
        names = sorted(per_module)
        message = _message(literal, names)
        for dotted in names:
            module = project.modules[dotted]
            for node in per_module[dotted]:
                yield module.violation("shared-option-literal", node, message)


def _message(literal: str, names: list[str]) -> str:
    """Write the one-sentence finding for a shared literal."""
    short_names = [name.removeprefix("langslice.") for name in names]
    listed = ", ".join(short_names[:LISTED_MODULES])
    if len(short_names) > LISTED_MODULES:
        listed = f"{listed} and {len(short_names) - LISTED_MODULES} more"
    return (
        f"Option value {literal!r} is compared as a bare string in "
        f"{len(names)} modules ({listed}); make it a StrEnum member (5.2)."
    )


def _option_sites(tree: ast.Module) -> Iterator[Site]:
    """Yield each option literal a module compares, with its node."""
    for node in ast.walk(tree):
        for literal in _compared_literals(node):
            if _is_option(literal):
                yield literal, node


def _is_option(literal: str) -> bool:
    """Tell whether a literal looks like an option value."""
    if len(literal) < 2 or literal in BOOLEAN_WORDS:
        return False
    if literal in BUILTIN_NAMES:
        return False
    return OPTION_PATTERN.fullmatch(literal) is not None


def _compared_literals(node: ast.AST) -> list[str]:
    """Return the string literals a node tests a value against."""
    if isinstance(node, ast.Compare):
        return _compare_literals(node)
    if isinstance(node, ast.MatchValue):
        text = string_value(node.value)
        return [] if text is None else [text]
    if isinstance(node, ast.Call):
        return _end_test_literals(node)
    return []


def _compare_literals(node: ast.Compare) -> list[str]:
    """Return the literals of ``==``, ``!=`` and ``in`` comparisons."""
    operands = [node.left, *node.comparators]
    if any(_is_platform_subject(operand) for operand in operands):
        return []
    literals: list[str] = []
    left = node.left
    for operator, right in zip(node.ops, node.comparators, strict=True):
        if isinstance(operator, (ast.Eq, ast.NotEq)):
            literals.extend(_equality_literals(left, right))
        elif isinstance(operator, (ast.In, ast.NotIn)):
            if string_value(left) is None:
                literals.extend(_collection_literals(right))
        left = right
    return literals


def _is_platform_subject(node: ast.expr) -> bool:
    """Tell whether an operand is what the platform or an array reports."""
    name = dotted_name(node) or call_name(node)
    if name in PLATFORM_SUBJECTS:
        return True
    return "dtype" in name.split(".")


def _equality_literals(left: ast.expr, right: ast.expr) -> list[str]:
    """Return the literal side of an equality against a non-constant."""
    for literal, other in ((left, right), (right, left)):
        text = string_value(literal)
        if text is not None and not isinstance(other, ast.Constant):
            return [text]
    return []


def _collection_literals(node: ast.expr) -> list[str]:
    """Return the string items of a literal tuple, list or set."""
    if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return []
    texts = [string_value(item) for item in node.elts]
    return [text for text in texts if text is not None]


def _end_test_literals(node: ast.Call) -> list[str]:
    """Return the literals passed to ``startswith`` or ``endswith``."""
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr not in END_TESTS:
        return []
    if not node.args or isinstance(func.value, ast.Constant):
        return []
    text = string_value(node.args[0])
    if text is not None:
        return [text]
    return _collection_literals(node.args[0])

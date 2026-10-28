"""Check that a line holds one idea, in conditionals and comprehensions.

A conditional expression (``a if test else b``) is allowed only as the
whole value of an assignment or a ``return``, and never inside another. A
comprehension has one ``for``, at most one ``if``, no comprehension inside
it and no ``:=``. The body of a lambda is its return value, so a lambda
whose whole body is a conditional expression is allowed.

Two older spellings of a conditional are conditionals in disguise and
are flagged wherever they stand: a pair indexed by a test,
``("no", "yes")[flag]`` (a two-item tuple or list literal indexed by
anything but a constant), and ``flag and "yes" or "no"`` used as a
value (an ``or`` whose first operand is an ``and``, outside the test of
an ``if``, ``while``, ``assert``, conditional expression or
comprehension, with an operand that is a value rather than a truth: a
string, number, f-string, display, attribute, subscript, arithmetic or
a call other than a test such as ``isinstance``).
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
from tools.stylecheck.rules.enclosing import parent_map
from tools.stylecheck.rules.scopes import call_name

#: The comprehension nodes.
COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

#: Statements whose whole value may be a conditional expression.
VALUE_STATEMENTS = (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Return)

#: The finding for a conditional expression inside a larger expression.
MISPLACED_CONDITIONAL = (
    "A conditional expression is used inside a larger expression; assign "
    "it to a name first."
)

#: The finding for a pair indexed by a test.
INDEXED_PAIR = (
    "A pair indexed by a test is a conditional in disguise; write "
    "'a if test else b' as the whole value of an assignment (1.2)."
)

#: The finding for ``test and a or b`` used as a value.
AND_OR_CONDITIONAL = (
    "'test and a or b' is a conditional in disguise, and wrong when a is "
    "falsy; write 'a if test else b' as the whole value of an "
    "assignment (1.2)."
)

#: Expressions whose value is a truth, not a value chosen by a test.
TRUTH_NODES = (ast.Compare, ast.BoolOp, ast.Name, ast.NamedExpr)

#: Calls that answer a test with a truth.
TRUTH_CALLS = frozenset(
    {"bool", "isinstance", "issubclass", "hasattr", "callable", "any", "all"}
)

#: Method-name prefixes of calls that answer a test with a truth.
TRUTH_PREFIXES = ("is_", "has_", "can_", "exists", "startswith", "endswith")


@module_check("nested-conditional")
def check_nested_conditional(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a conditional expression inside another (1.2).

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
        One per outermost conditional expression that holds another.
    """
    del project, config
    for node in outermost_holders(module.tree, ast.IfExp):
        yield module.violation(
            "nested-conditional",
            node,
            "A conditional expression holds another; write the branches "
            "as an if/elif statement.",
        )


def outermost_holders(
    tree: ast.AST, kind: type[ast.AST] | tuple[type[ast.AST], ...]
) -> Iterator[ast.AST]:
    """Yield each outermost node of a kind that holds another of that kind.

    Parameters
    ----------
    tree
        A parsed module.
    kind
        The node type, or a tuple of node types, to look for.

    Yields
    ------
    ast.AST
        Each node of ``kind`` that no node of ``kind`` contains and that
        contains one.
    """
    parents = parent_map(tree)
    for node in ast.walk(tree):
        if not isinstance(node, kind) or _has_ancestor(node, kind, parents):
            continue
        inner = [
            child
            for child in ast.walk(node)
            if isinstance(child, kind) and child is not node
        ]
        if inner:
            yield node


@module_check("conditional-placement")
def check_conditional_placement(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a conditional expression that is not a whole assigned value (1.2).

    Allowed only as the whole right-hand side of an assignment (plain,
    augmented or annotated), the whole value of a ``return``, or the whole
    body of a lambda. Inside a call's arguments, a comprehension, an
    f-string or an operator it is flagged. A conditional expression
    inside another is left to the nested-conditional check. The two
    disguised conditionals of the module docstring are flagged wherever
    they stand.

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
        One per misplaced conditional expression.
    """
    del project, config
    parents = parent_map(module.tree)
    for node in ast.walk(module.tree):
        message = _placement_problem(node, parents)
        if message:
            yield module.violation("conditional-placement", node, message)


def _placement_problem(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    """Say how a node misplaces or disguises a conditional, or ``""``."""
    if _is_indexed_pair(node):
        return INDEXED_PAIR
    if _is_and_or_value(node, parents):
        return AND_OR_CONDITIONAL
    if not isinstance(node, ast.IfExp):
        return ""
    if _has_ancestor(node, ast.IfExp, parents):
        return ""
    if _is_whole_value(node, parents.get(node)):
        return ""
    return MISPLACED_CONDITIONAL


def _is_indexed_pair(node: ast.AST) -> bool:
    """Tell whether a node indexes a two-item literal by a non-constant."""
    if not isinstance(node, ast.Subscript):
        return False
    pair = node.value
    if not isinstance(pair, (ast.Tuple, ast.List)) or len(pair.elts) != 2:
        return False
    return not isinstance(node.slice, (ast.Constant, ast.Slice))


def _is_and_or_value(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """Tell whether a node is ``test and a or b`` standing for a value."""
    if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.Or):
        return False
    first = node.values[0]
    if not isinstance(first, ast.BoolOp) or not isinstance(first.op, ast.And):
        return False
    chosen = [first.values[-1], *node.values[1:]]
    if not any(_is_value(operand) for operand in chosen):
        return False
    return not _is_in_test(node, parents)


def _is_value(node: ast.expr) -> bool:
    """Tell whether an operand is a value rather than a truth."""
    if isinstance(node, ast.Constant):
        return not isinstance(node.value, bool) and node.value is not None
    if isinstance(node, ast.UnaryOp):
        return not isinstance(node.op, ast.Not)
    if isinstance(node, ast.Call):
        name = call_name(node).rpartition(".")[2]
        return name not in TRUTH_CALLS and not name.startswith(TRUTH_PREFIXES)
    return not isinstance(node, TRUTH_NODES)


def _is_in_test(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """Tell whether an expression is, or is part of, a test of truth."""
    child = node
    parent = parents.get(node)
    while isinstance(parent, (ast.BoolOp, ast.UnaryOp)):
        child, parent = parent, parents.get(parent)
    if isinstance(parent, (ast.If, ast.While, ast.IfExp, ast.Assert)):
        return parent.test is child
    if isinstance(parent, ast.comprehension):
        return child in parent.ifs
    return isinstance(parent, ast.Call) and call_name(parent) == "bool"


def _is_whole_value(node: ast.IfExp, parent: ast.AST | None) -> bool:
    """Tell whether an expression is the whole value of its statement."""
    if isinstance(parent, VALUE_STATEMENTS):
        return parent.value is node
    return isinstance(parent, ast.Lambda) and parent.body is node


def _has_ancestor(
    node: ast.AST,
    kind: type[ast.AST] | tuple[type[ast.AST], ...],
    parents: dict[ast.AST, ast.AST],
) -> bool:
    """Tell whether a node of the given kind contains this one."""
    current = parents.get(node)
    while current is not None:
        if isinstance(current, kind):
            return True
        if isinstance(current, ast.stmt):
            return False
        current = parents.get(current)
    return False


@module_check("nested-comprehension")
def check_nested_comprehension(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a comprehension inside another comprehension (1.2).

    A generator passed to ``any``, ``all`` or ``"".join`` inside a
    comprehension counts too.

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
        One per outermost comprehension that holds another.
    """
    del project, config
    for node in outermost_holders(module.tree, COMPREHENSIONS):
        yield module.violation(
            "nested-comprehension",
            node,
            "A comprehension holds another; build the inner one first or "
            "write a loop.",
        )


@module_check("comprehension-clauses")
def check_comprehension_clauses(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a comprehension with two ``for`` or two ``if`` clauses (1.2).

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
        One per comprehension with more than one ``for`` or ``if``.
    """
    del project, config
    for node in ast.walk(module.tree):
        if not isinstance(node, COMPREHENSIONS):
            continue
        n_loops = len(node.generators)
        n_conditions = sum(len(loop.ifs) for loop in node.generators)
        if n_loops > 1 or n_conditions > 1:
            yield module.violation(
                "comprehension-clauses",
                node,
                f"A comprehension has {n_loops} 'for' and {n_conditions} "
                "'if' clauses; the limit is one of each.",
            )


@module_check("walrus-in-comprehension")
def check_walrus_in_comprehension(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag an assignment expression (``:=``) inside a comprehension (1.2).

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
        One per ``:=`` inside a comprehension.
    """
    del project, config
    parents = parent_map(module.tree)
    for node in ast.walk(module.tree):
        if isinstance(node, ast.NamedExpr) and _has_ancestor(
            node, COMPREHENSIONS, parents
        ):
            yield module.violation(
                "walrus-in-comprehension",
                node,
                "An assignment expression ':=' is used inside a "
                "comprehension; compute the value in a loop instead.",
            )

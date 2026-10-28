"""Measure how hard a function is to hold in mind: complexity and nesting.

Cognitive complexity follows G. Ann Campbell, *Cognitive Complexity: a new
way of measuring understandability* (SonarSource, 2017). Each of these adds
one: ``if``, ``elif``, ``else``, a conditional expression, a loop, an
``except`` clause, a ``match``, a sequence of like boolean operators, and a
function that calls itself (once per function, however many calls).
``if``, a conditional expression, a loop, ``except`` and ``match`` add one
more for each level of nesting they sit at. The bodies of those
structures, and of nested functions and lambdas, sit one level deeper.
``try``, ``finally``, ``with`` and the ``else`` of a loop or ``try`` add
nothing and do not nest. A nested function adds its score to the
function around it and is not scored alone.

Python's comprehensions are not in the paper; each ``for`` clause is
counted as a loop and each ``if`` clause adds one, and the expression they
produce sits one level deeper.

The scores were compared with complexipy 8.3 on every LangSlice function
(2,342): 1,906 agree. Every difference seen comes from a place where
complexipy departs from the paper. It does not look inside some
expressions: keyword arguments, the operands of ``not``, of arithmetic and
of boolean operators (other than nested boolean operators), subscripts,
attribute access on a call, f-strings, ``yield`` and ``await``, so a
conditional expression or a comprehension there goes uncounted. It nests
the body of a nested ``def`` only when the ``def`` is not the first
statement of the function around it. It does not count a method that
calls itself through ``self``, and it does count a method that calls a
module function of the same name, which is not recursion.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterator

from tools.stylecheck.core import (
    Config,
    Module,
    Project,
    Violation,
    function_body,
    module_check,
)
from tools.stylecheck.rules.enclosing import (
    enclosing_function,
    owner_class,
    parent_map,
)
from tools.stylecheck.rules.scopes import (
    BOUND_NAMES,
    FUNCTION_NODES,
    functions,
    own_nodes,
)

Function = ast.FunctionDef | ast.AsyncFunctionDef

#: The comprehension nodes, which are scored as loops.
COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

#: The statements that open a block, for the nesting depth.
BLOCK_STATEMENTS = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.TryStar,
    ast.Match,
)


@module_check("cognitive-complexity")
def check_cognitive_complexity(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a function whose cognitive complexity is over the limit (3.1).

    Every function not nested in another is scored with the Cognitive
    Complexity definition (SonarSource, 2017); a nested function's score
    is part of the one around it.

    Parameters
    ----------
    module
        The parsed file to check.
    project
        The package index; not used by this check.
    config
        The settings; ``max_cognitive_complexity`` is the limit.

    Yields
    ------
    Violation
        One per function over the limit, giving its score.
    """
    del project
    parents = parent_map(module.tree)
    limit = config.max_cognitive_complexity
    for function in functions(module.tree):
        if enclosing_function(function, parents) is not None:
            continue
        score = cognitive_complexity(
            function, is_method=owner_class(function, parents) is not None
        )
        if score > limit:
            yield module.violation(
                "cognitive-complexity",
                function,
                f"Cognitive complexity is {score}; the limit is {limit}.",
            )


def cognitive_complexity(function: Function, *, is_method: bool) -> int:
    """Score a function by the Cognitive Complexity definition.

    Parameters
    ----------
    function
        The function to score, with every function nested in it.
    is_method
        Whether the function is a method, so that a call through
        ``self`` or ``cls`` to its own name counts as recursion.

    Returns
    -------
    int
        The score; 0 for a function with no branch at all.
    """
    score = _block_score(function.body, 0)
    return score + _recursion_score(function, is_method=is_method)


def _block_score(statements: list[ast.stmt], nesting: int) -> int:
    """Score a list of statements at one nesting level."""
    return sum(_score(statement, nesting) for statement in statements)


def _score(node: ast.AST, nesting: int) -> int:
    """Score any node, dispatching on its type."""
    scorer = _SCORERS.get(type(node), _children_score)
    return scorer(node, nesting)


def _children_score(node: ast.AST, nesting: int) -> int:
    """Score a node that adds nothing itself, by its children."""
    return sum(_score(child, nesting) for child in ast.iter_child_nodes(node))


def _if_score(node: ast.AST, nesting: int) -> int:
    """Score an ``if`` with its ``elif`` and ``else`` branches."""
    assert isinstance(node, ast.If)
    total = 1 + nesting + _score(node.test, nesting)
    total += _block_score(node.body, nesting + 1)
    branch = node
    while branch.orelse:
        if _is_elif(branch):
            branch = branch.orelse[0]
            assert isinstance(branch, ast.If)
            total += 1 + _score(branch.test, nesting)
            total += _block_score(branch.body, nesting + 1)
        else:
            total += 1 + _block_score(branch.orelse, nesting + 1)
            break
    return total


def _is_elif(node: ast.If) -> bool:
    """Tell whether an ``if``'s else branch is written as ``elif``.

    An ``elif`` starts in the same column as its ``if``; an ``if``
    written inside an ``else:`` block is indented further.
    """
    if len(node.orelse) != 1:
        return False
    branch = node.orelse[0]
    return isinstance(branch, ast.If) and branch.col_offset == node.col_offset


def _loop_score(node: ast.AST, nesting: int) -> int:
    """Score a ``for`` or ``while`` loop; its ``else`` adds nothing."""
    assert isinstance(node, (ast.For, ast.AsyncFor, ast.While))
    total = 1 + nesting
    if isinstance(node, ast.While):
        total += _score(node.test, nesting)
    else:
        total += _score(node.iter, nesting)
    total += _block_score(node.body, nesting + 1)
    return total + _block_score(node.orelse, nesting + 1)


def _try_score(node: ast.AST, nesting: int) -> int:
    """Score a ``try``: each ``except`` counts, the rest does not nest."""
    assert isinstance(node, (ast.Try, ast.TryStar))
    total = _block_score(node.body, nesting)
    for handler in node.handlers:
        total += 1 + nesting
        total += _block_score(handler.body, nesting + 1)
    total += _block_score(node.orelse, nesting)
    return total + _block_score(node.finalbody, nesting)


def _match_score(node: ast.AST, nesting: int) -> int:
    """Score a ``match`` as the paper scores a ``switch``."""
    assert isinstance(node, ast.Match)
    total = 1 + nesting + _score(node.subject, nesting)
    for case in node.cases:
        if case.guard is not None:
            total += _score(case.guard, nesting + 1)
        total += _block_score(case.body, nesting + 1)
    return total


def _conditional_score(node: ast.AST, nesting: int) -> int:
    """Score a conditional expression; its parts sit one level deeper."""
    assert isinstance(node, ast.IfExp)
    return 1 + nesting + _children_score(node, nesting + 1)


def _boolean_score(node: ast.AST, nesting: int) -> int:
    """Score a sequence of like boolean operators as one."""
    assert isinstance(node, ast.BoolOp)
    return 1 + _operands_score(node, nesting)


def _operands_score(node: ast.BoolOp, nesting: int) -> int:
    """Score the operands of a boolean sequence.

    An operand that continues the sequence with the same operator, as in
    ``a and (b and c)``, adds nothing itself.
    """
    total = 0
    for value in node.values:
        same = isinstance(value, ast.BoolOp) and type(value.op) is type(
            node.op
        )
        if same:
            assert isinstance(value, ast.BoolOp)
            total += _operands_score(value, nesting)
        else:
            total += _score(value, nesting)
    return total


def _comprehension_score(node: ast.AST, nesting: int) -> int:
    """Score a comprehension: each ``for`` as a loop, each ``if`` as one."""
    assert isinstance(node, COMPREHENSIONS)
    total = 0
    for index, generator in enumerate(node.generators):
        total += 1 + nesting
        iter_nesting = nesting if index == 0 else nesting + 1
        total += _score(generator.iter, iter_nesting)
        for condition in generator.ifs:
            total += 1 + _score(condition, nesting + 1)
    if isinstance(node, ast.DictComp):
        total += _score(node.key, nesting + 1)
        return total + _score(node.value, nesting + 1)
    return total + _score(node.elt, nesting + 1)


def _function_score(node: ast.AST, nesting: int) -> int:
    """Score a nested function: no increment, its body one level deeper."""
    assert isinstance(node, FUNCTION_NODES)
    score = _block_score(node.body, nesting + 1)
    return score + _recursion_score(node, is_method=False)


def _lambda_score(node: ast.AST, nesting: int) -> int:
    """Score a lambda: no increment, its body one level deeper."""
    assert isinstance(node, ast.Lambda)
    return _score(node.body, nesting + 1)


def _class_score(node: ast.AST, nesting: int) -> int:
    """Score a class written inside a function by its body."""
    assert isinstance(node, ast.ClassDef)
    return _block_score(node.body, nesting)


def _recursion_score(function: Function, *, is_method: bool) -> int:
    """Return 1 when a function calls itself in its own body, else 0.

    A method calls itself through ``self.name`` or ``cls.name``; calls
    made inside functions nested in it belong to those functions.
    """
    for node in own_nodes(function):
        if isinstance(node, ast.Call) and _names_function(
            node.func, function.name, is_method=is_method
        ):
            return 1
    return 0


def _names_function(callee: ast.expr, name: str, *, is_method: bool) -> bool:
    """Tell whether a callee is the named function, or the named method."""
    if isinstance(callee, ast.Name):
        return not is_method and callee.id == name
    if not is_method or not isinstance(callee, ast.Attribute):
        return False
    receiver = callee.value
    return (
        isinstance(receiver, ast.Name)
        and receiver.id in BOUND_NAMES
        and callee.attr == name
    )


_SCORERS: dict[type[ast.AST], Callable[[ast.AST, int], int]] = {
    ast.If: _if_score,
    ast.For: _loop_score,
    ast.AsyncFor: _loop_score,
    ast.While: _loop_score,
    ast.Try: _try_score,
    ast.TryStar: _try_score,
    ast.Match: _match_score,
    ast.IfExp: _conditional_score,
    ast.BoolOp: _boolean_score,
    ast.ListComp: _comprehension_score,
    ast.SetComp: _comprehension_score,
    ast.DictComp: _comprehension_score,
    ast.GeneratorExp: _comprehension_score,
    ast.FunctionDef: _function_score,
    ast.AsyncFunctionDef: _function_score,
    ast.Lambda: _lambda_score,
    ast.ClassDef: _class_score,
}


@module_check("nesting-depth")
def check_nesting_depth(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a function whose blocks nest deeper than the limit (3.1).

    A block is an ``if``, ``for``, ``while``, ``with``, ``try`` or
    ``match``; ``elif``, ``else``, ``except`` and ``finally`` sit at the
    depth of the block they belong to. Each function is measured alone: a
    nested function starts again at zero (the nested-definition check
    covers it).

    Parameters
    ----------
    module
        The parsed file to check.
    project
        The package index; not used by this check.
    config
        The settings; ``max_nesting`` is the limit.

    Yields
    ------
    Violation
        One per function over the limit, giving its depth.
    """
    del project
    limit = config.max_nesting
    for function in functions(module.tree):
        depth = nesting_depth(function_body(function))
        if depth > limit:
            yield module.violation(
                "nesting-depth",
                function,
                f"Blocks nest {depth} deep; the limit is {limit}.",
            )


def nesting_depth(statements: list[ast.stmt]) -> int:
    """Return how deep blocks nest in a list of statements.

    Parameters
    ----------
    statements
        A function's body, or any block of statements.

    Returns
    -------
    int
        The largest number of blocks around any statement, counting the
        block itself; nested functions and classes are not entered.
    """
    deepest = 0
    for statement in statements:
        if not isinstance(statement, BLOCK_STATEMENTS):
            continue
        inner = max(
            (nesting_depth(block) for block in statement_blocks(statement)),
            default=0,
        )
        deepest = max(deepest, 1 + inner)
    return deepest


def statement_blocks(statement: ast.stmt) -> list[list[ast.stmt]]:
    """Return the statement lists a block statement holds at one depth.

    Parameters
    ----------
    statement
        An ``if``, loop, ``with``, ``try`` or ``match`` statement.

    Returns
    -------
    list of list of ast.stmt
        Each branch's statements: an ``if`` with each ``elif`` and its
        ``else``, a ``try`` with its handlers, ``else`` and ``finally``.
    """
    if isinstance(statement, ast.If):
        return _if_blocks(statement)
    if isinstance(statement, ast.Match):
        return [case.body for case in statement.cases]
    if isinstance(statement, (ast.Try, ast.TryStar)):
        handlers = [handler.body for handler in statement.handlers]
        return [
            statement.body,
            *handlers,
            statement.orelse,
            statement.finalbody,
        ]

    if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
        return [statement.body, statement.orelse]
    assert isinstance(statement, (ast.With, ast.AsyncWith))
    return [statement.body]


def _if_blocks(statement: ast.If) -> list[list[ast.stmt]]:
    """Return an ``if``'s branches, with each ``elif`` at the same depth."""
    blocks = [statement.body]
    branch = statement
    while branch.orelse:
        if not _is_elif(branch):
            blocks.append(branch.orelse)
            break
        branch = branch.orelse[0]
        assert isinstance(branch, ast.If)
        blocks.append(branch.body)
    return blocks

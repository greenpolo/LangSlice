"""Tests of the cognitive complexity and nesting depth checks."""

from __future__ import annotations

import ast
import textwrap

import pytest

from tools.stylecheck.rules.complexity import (
    cognitive_complexity,
    nesting_depth,
)
from tools.stylecheck.testing import check_source


def _score(source: str) -> int:
    """Score the first function of a snippet."""
    tree = ast.parse(textwrap.dedent(source))
    function = tree.body[0]
    is_method = False
    if isinstance(function, ast.ClassDef):
        function = function.body[0]
        is_method = True
    assert isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
    return cognitive_complexity(function, is_method=is_method)


# Each case's score agrees with complexipy 8.3, except those marked
# "paper", where complexipy departs from the SonarSource definition.
CASES = [
    ("def f(a):\n    return a\n", 0),
    ("def f(a):\n    if a:\n        return 1\n", 1),
    (
        "def f(a):\n    if a:\n        return 1\n    else:\n"
        "        return 2\n",
        2,
    ),
    (
        "def f(a, b):\n    if a:\n        return 1\n    elif b:\n"
        "        return 2\n    else:\n        return 3\n",
        3,
    ),
    (
        "def f(a, b):\n    if a:\n        return 1\n    else:\n"
        "        if b:\n            return 2\n",
        4,
    ),
    (
        "def f(a, b):\n    for x in a:\n        if b:\n            return 1\n",
        3,
    ),
    ("def f(a):\n    x = 1 if a else 2\n    return x\n", 1),
    ("def f(a, b, c):\n    return a and b and c\n", 1),
    ("def f(a, b, c, d):\n    return a and b or c and d\n", 3),
    ("def f(a, b, c):\n    return a and not (b and c)\n", 2),
    ("def f(a, b, c):\n    return a and (b and c)\n", 1),
    (
        "def f(a):\n    try:\n        return a()\n    except ValueError:\n"
        "        return 1\n    except KeyError:\n        return 2\n"
        "    finally:\n        pass\n",
        2,
    ),
    (
        "def f(a):\n    for x in a:\n        try:\n            return a()\n"
        "        except ValueError:\n            if x:\n"
        "                return 1\n",
        6,
    ),
    ("def f(a):\n    with a:\n        if a:\n            return 1\n", 1),
    (
        "def f(a):\n    while a:\n        break\n    else:\n"
        "        return 1\n",
        1,
    ),
    ("def f(a):\n    return [x for x in a if x]\n", 2),
    ("def f(a):\n    if a:\n        return [x for x in a if x]\n", 4),
    ("def f(a):\n    if a:\n        return [[y for y in x] for x in a]\n", 6),
    (
        "def f(a):\n    match a:\n        case 1:\n            return 1\n"
        "        case 2:\n            if a:\n                return 2\n",
        3,
    ),
    (
        "def f(n):\n    if n:\n        return f(n - 1) + f(n - 2)\n"
        "    return 0\n",
        2,
    ),
    ("def f(a):\n    g = lambda b: 1 if b else 2\n    return g(a)\n", 2),
    (
        "def f(a, b, c):\n    return a if b else (c if a else b)\n",
        3,
    ),
    (
        "def f(a, b):\n    for x in a:\n        if x:\n            pass\n"
        "        elif b:\n            if x:\n                pass\n"
        "        else:\n            pass\n",
        8,
    ),
    # paper: a nested function's body sits one level deeper
    (
        "def f(a):\n    def inner(b):\n        if b:\n            return 1\n"
        "    return inner(a)\n",
        2,
    ),
    # paper: complexipy skips a conditional in a keyword argument
    ("def f(a):\n    return g(k=1 if a else 2)\n", 1),
    # paper: complexipy skips the operand of `not`
    ("def f(a):\n    if not g([x for x in a]):\n        return 1\n", 2),
    # paper: a method calling itself through self is recursion
    (
        "class C:\n    def m(self, a):\n        if a:\n"
        "            return self.m(a)\n",
        2,
    ),
    # a method calling a module function of the same name is not
    ("class C:\n    def m(self, a):\n        return m(a)\n", 0),
]


@pytest.mark.parametrize(("source", "expected"), CASES)
def test_cognitive_complexity_scores(source: str, expected: int) -> None:
    assert _score(source) == expected


def test_cognitive_complexity_flags_over_the_limit() -> None:
    branches = "".join(
        f"        if x == {value}:\n            continue\n"
        for value in range(8)
    )
    source = f"def busy(a):\n    for x in a:\n{branches}"
    violations = check_source(source, "cognitive-complexity")
    assert [violation.symbol for violation in violations] == ["busy"]
    assert "is 17; the limit is 15" in violations[0].message


def test_cognitive_complexity_scores_a_nested_function_with_its_outer() -> (
    None
):
    inner_branches = "".join(
        f"            if b == {value}:\n                return {value}\n"
        for value in range(6)
    )
    source = (
        "def outer(a):\n    if a:\n        def inner(b):\n"
        f"{inner_branches}        return inner\n"
    )
    violations = check_source(source, "cognitive-complexity")
    assert [violation.symbol for violation in violations] == ["outer"]


def test_cognitive_complexity_passes_a_flat_function() -> None:
    source = """
    def tidy(a):
        if a:
            return 1
        return 2
    """
    assert check_source(source, "cognitive-complexity") == []


def test_nesting_depth_counts_blocks() -> None:
    tree = ast.parse(
        textwrap.dedent(
            """
            if a:
                for x in a:
                    with x:
                        pass
            elif b:
                pass
            """
        )
    )
    assert nesting_depth(tree.body) == 3


def test_nesting_depth_flags_five_blocks() -> None:
    source = """
    def deep(a):
        for x in a:
            if x:
                with x:
                    try:
                        while x:
                            pass
                    except ValueError:
                        pass
    """
    violations = check_source(source, "nesting-depth")
    assert len(violations) == 1
    assert "5 deep" in violations[0].message


def test_nesting_depth_keeps_elif_chains_flat() -> None:
    source = """
    def chain(a):
        for x in a:
            if x == 1:
                pass
            elif x == 2:
                pass
            elif x == 3:
                if x:
                    with x:
                        pass
    """
    assert check_source(source, "nesting-depth") == []


def test_nesting_depth_measures_a_nested_function_alone() -> None:
    # the game: move the inner loop into a nested def; the nested
    # definition check catches that, and depth starts again inside it
    source = """
    def outer(a):
        for x in a:
            if x:
                def inner(y):
                    for z in y:
                        if z:
                            pass
                inner(x)
    """
    assert check_source(source, "nesting-depth") == []
    assert check_source(source, "nested-definition") != []

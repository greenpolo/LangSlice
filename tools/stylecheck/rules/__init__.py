"""The checks. Importing this package registers every check in it.

Each module below defines checks and registers them when it is imported;
a new module of checks is added to this list. The helper modules
(``annotations``, ``enclosing``, ``scopes``) register nothing.
"""

from tools.stylecheck.rules import (
    arguments,
    comments,
    complexity,
    data,
    decorators,
    docstring_layers,
    docstrings,
    duplicates,
    errors,
    expressions,
    functions,
    imports,
    layout,
    metaprogramming,
    modules,
    names,
    options,
    text,
)

__all__ = [
    "arguments",
    "comments",
    "complexity",
    "data",
    "decorators",
    "docstring_layers",
    "docstrings",
    "duplicates",
    "errors",
    "expressions",
    "functions",
    "imports",
    "layout",
    "metaprogramming",
    "modules",
    "names",
    "options",
    "text",
]

"""Check how functions take their arguments and hand back results.

Options are keyword-only, ``**kwargs`` and ``*args`` pass through whole,
and a function does not write its results into a list or dict its caller
passed in. A signature that a base class or framework fixes is exempt
where the guide allows: a ``__dunder__`` method, or a method that
overrides a base (see :mod:`tools.stylecheck.rules.enclosing`).
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
from tools.stylecheck.rules.enclosing import (
    body_nodes,
    decorator_names,
    is_dunder,
    is_override,
    owner_class,
    parent_map,
)
from tools.stylecheck.rules.scopes import functions, parameters

Function = ast.FunctionDef | ast.AsyncFunctionDef

#: Methods that write into a list, dict or set in place.
MUTATING_METHODS = frozenset(
    {"append", "extend", "insert", "update", "setdefault", "add"}
)

#: Annotations that make a parameter a container the caller owns.
CONTAINER_TYPES = frozenset(
    {
        "list",
        "dict",
        "set",
        "List",
        "Dict",
        "Set",
        "MutableMapping",
        "MutableSequence",
        "MutableSet",
        "defaultdict",
        "OrderedDict",
        "Counter",
    }
)

#: Methods only a list or dict has, so an unannotated parameter that
#: calls one is taken to be a container.
CONTAINER_ONLY_METHODS = frozenset({"append", "extend", "setdefault"})

#: The decorators that make a function a wrapper with another's signature.
WRAPS_DECORATORS = frozenset({"wraps", "functools.wraps"})

#: The numpy name for a pre-allocated output array, the accepted form.
NUMPY_OUTPUT = "out"


@module_check("keyword-only-options")
def check_keyword_only_options(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a parameter with a default that is not keyword-only (3.5).

    Required inputs come first, positionally; every option, a parameter
    with a default, comes after ``*``. Exempt: ``__dunder__`` methods and
    methods that override a base, whose signature the base fixes.

    Parameters
    ----------
    module
        The parsed file to check.
    project
        The package index, used to tell an override.
    config
        The settings; not used by this check.

    Yields
    ------
    Violation
        One per function, naming its first positional option.
    """
    del config
    parents = parent_map(module.tree)
    for function in functions(module.tree):
        if not function.args.defaults:
            continue
        if _has_fixed_signature(function, parents, project):
            continue
        positional = [*function.args.posonlyargs, *function.args.args]
        first = positional[len(positional) - len(function.args.defaults)]
        yield module.violation(
            "keyword-only-options",
            function,
            f"The option {first.arg!r} has a default but can be passed by "
            "position; put the options after '*'.",
        )


def _has_fixed_signature(
    function: Function, parents: dict[ast.AST, ast.AST], project: Project
) -> bool:
    """Tell whether a function's signature is fixed by Python or a base."""
    if is_dunder(function.name):
        return True
    return is_override(function, owner_class(function, parents), project)


@module_check("kwargs-forwarding")
def check_kwargs_forwarding(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag ``**kwargs`` or ``*args`` that are not passed through whole (3.5).

    ``**kwargs`` must appear in the body exactly once, as ``**kwargs`` in
    one call. ``*args`` may only be iterated, unpacked into a call, or
    handed whole to a call, as a homogeneous sequence; indexing it or
    unpacking it into names makes it hidden positional parameters.
    Exempt: a wrapper marked with ``functools.wraps``, whose signature is
    the wrapped function's, and an unused ``*args`` or ``**kwargs`` that
    a fixed signature demands: in a ``__dunder__`` method, an override,
    or named with a leading underscore (``*_``) to say it is ignored.

    Parameters
    ----------
    module
        The parsed file to check.
    project
        The package index, used to tell an override.
    config
        The settings; not used by this check.

    Yields
    ------
    Violation
        One per misused ``**kwargs`` or ``*args``.
    """
    del config
    parents = parent_map(module.tree)
    for function in functions(module.tree):
        if WRAPS_DECORATORS.intersection(decorator_names(function)):
            continue
        fixed = _has_fixed_signature(function, parents, project)
        for message in _variadic_messages(function, parents, fixed=fixed):
            yield module.violation("kwargs-forwarding", function, message)


def _variadic_messages(
    function: Function, parents: dict[ast.AST, ast.AST], *, fixed: bool
) -> Iterator[str]:
    """Yield a message for each misused ``*args`` or ``**kwargs``."""
    kwarg = function.args.kwarg
    if kwarg is not None and not _is_ignored(function, kwarg, fixed=fixed):
        message = _kwargs_message(function, kwarg.arg, parents)
        if message:
            yield message
    vararg = function.args.vararg
    if vararg is not None and not _is_ignored(function, vararg, fixed=fixed):
        message = _args_message(function, vararg.arg, parents)
        if message:
            yield message


def _is_ignored(function: Function, argument: ast.arg, *, fixed: bool) -> bool:
    """Tell whether an unused variadic parameter is accepted on purpose."""
    if _uses(function, argument.arg):
        return False
    return fixed or argument.arg.startswith("_")


def _uses(function: Function, name: str) -> list[ast.Name]:
    """Return every read of a name in a function's body."""
    return [
        node
        for node in body_nodes(function)
        if isinstance(node, ast.Name) and node.id == name
    ]


def _kwargs_message(
    function: Function, name: str, parents: dict[ast.AST, ast.AST]
) -> str:
    """Say how ``**kwargs`` is misused, or return ``""``."""
    uses = _uses(function, name)
    if len(uses) == 1:
        parent = parents.get(uses[0])
        if isinstance(parent, ast.keyword) and parent.arg is None:
            return ""
    return (
        f"'**{name}' must be forwarded whole as '**{name}' in exactly one "
        "call and not read otherwise."
    )


def _args_message(
    function: Function, name: str, parents: dict[ast.AST, ast.AST]
) -> str:
    """Say how ``*args`` is misused, or return ``""``."""
    uses = _uses(function, name)
    if uses and all(_is_whole_sequence_use(use, parents) for use in uses):
        return ""
    return (
        f"'*{name}' must be used as one sequence, iterated or passed on "
        "whole, and never indexed or unpacked."
    )


def _is_whole_sequence_use(
    use: ast.Name, parents: dict[ast.AST, ast.AST]
) -> bool:
    """Tell whether a read of ``*args`` treats it as a whole sequence.

    Unpacked with ``*`` (into a call or a display), iterated, or passed
    as a positional argument.
    """
    parent = parents.get(use)
    if isinstance(parent, ast.Starred):
        return True
    if isinstance(parent, (ast.For, ast.AsyncFor, ast.comprehension)):
        return parent.iter is use
    if isinstance(parent, ast.Call):
        return use in parent.args
    return False


@module_check("output-parameter")
def check_output_parameter(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a function that writes into a container it was passed (3.5).

    A parameter annotated as a list, dict or set (or unannotated and
    used with ``append``, ``extend`` or ``setdefault``) that the function
    mutates (with ``append``, ``extend``, ``insert``, ``update``,
    ``setdefault``, ``add``, item assignment or ``del``) is a second
    return channel. A parameter the function rebinds first is its own; a
    numpy ``out`` parameter and ``self`` or ``cls`` are fine.

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
        One per mutated parameter, at its first mutation.
    """
    del project, config
    for function in functions(module.tree):
        rebound = _rebound_names(function)
        for argument in parameters(function, include_bound=False):
            if argument.arg == NUMPY_OUTPUT or argument.arg in rebound:
                continue

            site = _first_mutation(function, argument)
            if site is None:
                continue

            # Named by the function that owns the parameter, even when
            # the write happens in a function nested in it.
            yield Violation(
                "output-parameter",
                module.relpath,
                module.symbol_at(function.lineno),
                site.lineno,
                f"The function writes into its parameter "
                f"{argument.arg!r}; return the result instead.",
            )


def _rebound_names(function: Function) -> set[str]:
    """Return the names a function assigns anywhere in its body."""
    return {
        node.id
        for node in body_nodes(function)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }


def _first_mutation(
    function: Function, argument: ast.arg
) -> ast.stmt | ast.expr | None:
    """Return the first node that mutates a parameter, or None."""
    is_container = _is_container_annotation(argument.annotation)
    if argument.annotation is not None and not is_container:
        return None
    if is_container:
        methods = MUTATING_METHODS
    else:
        methods = CONTAINER_ONLY_METHODS
    for node in body_nodes(function):
        if not isinstance(node, (ast.stmt, ast.expr)):
            continue
        if _mutates(node, argument.arg, methods, is_container):
            return node
    return None


def _mutates(
    node: ast.AST, name: str, methods: frozenset[str], is_container: bool
) -> bool:
    """Tell whether a node mutates the named parameter in place."""
    if isinstance(node, ast.Call):
        callee = node.func
        return (
            isinstance(callee, ast.Attribute)
            and _is_name(callee.value, name)
            and callee.attr in methods
        )
    if not is_container:
        return False
    if isinstance(node, (ast.Assign, ast.Delete)):
        return any(_is_item_in(target, name) for target in node.targets)
    if isinstance(node, (ast.AugAssign, ast.AnnAssign)):
        return _is_item_in(node.target, name)
    return False


def _is_item_in(target: ast.expr, name: str) -> bool:
    """Tell whether an assignment target is an item of the named value."""
    return isinstance(target, ast.Subscript) and _is_name(target.value, name)


def _is_name(node: ast.expr, name: str) -> bool:
    """Tell whether an expression is the bare name given."""
    return isinstance(node, ast.Name) and node.id == name


def _is_container_annotation(annotation: ast.expr | None) -> bool:
    """Tell whether an annotation names a mutable list, dict or set.

    ``list[int]``, ``dict[str, X] | None`` and ``typing.List`` count; a
    string annotation is read the same way.
    """
    if annotation is None:
        return False
    if isinstance(annotation, ast.Constant) and isinstance(
        annotation.value, str
    ):
        try:
            annotation = ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            return False

    if isinstance(annotation, ast.BinOp):
        return _is_container_annotation(
            annotation.left
        ) or _is_container_annotation(annotation.right)
    if isinstance(annotation, ast.Subscript):
        annotation = annotation.value
    if isinstance(annotation, ast.Attribute):
        return annotation.attr in CONTAINER_TYPES
    return isinstance(annotation, ast.Name) and (
        annotation.id in CONTAINER_TYPES
    )

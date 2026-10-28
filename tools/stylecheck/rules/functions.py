"""Check each function's shape: its length, its parameters, what it returns.

Also here: the ban on nested definitions and on forwarders, the two ways a
function hides work from a reader. A *forwarder* is a function whose body
only hands its own parameters to another call, or only reads an attribute
of a parameter; callers should call the real thing. A public function whose
docstring calls itself a "public alias" is a deliberate alias at a door and
is not a forwarder.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator

from tools.stylecheck.core import (
    Config,
    Module,
    Project,
    Violation,
    function_body,
    module_check,
)
from tools.stylecheck.rules.annotations import (
    declared_class,
    last_name,
    parse_string,
    subscript_items,
)
from tools.stylecheck.rules.enclosing import (
    decorator_names,
    is_dunder,
    is_override,
    is_property,
    owner_class,
    parent_map,
)
from tools.stylecheck.rules.scopes import (
    BOUND_NAMES,
    FUNCTION_NODES,
    functions,
    module_matches,
    own_nodes,
    parameters,
    string_value,
)

Function = ast.FunctionDef | ast.AsyncFunctionDef

#: The phrase a docstring uses to mark a deliberate alias at a door.
ALIAS_MARKER = "public alias"

#: Decorators that leave a function's behaviour as written, so a decorated
#: forwarder is still a forwarder.
PLAIN_DECORATORS = frozenset({"staticmethod", "classmethod"})

#: The longest nested callback, in lines, that may be handed to a call.
MAX_CALLBACK_LINES = 10

#: Keywords whose lambda value is a small sort key or fallback.
LAMBDA_KEYWORDS = frozenset({"key", "default"})

#: The fewest string keys that make a returned dict literal a record.
MIN_RECORD_KEYS = 3

#: The calls that make a named tuple class from a list of fields.
NAMED_TUPLE_FACTORIES = frozenset({"namedtuple", "NamedTuple"})

NESTED_NODES = (*FUNCTION_NODES, ast.ClassDef, ast.Lambda)


@module_check("body-lines")
def check_body_lines(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a function whose body is longer than the limit (3.1).

    The body runs from the first statement after the docstring to the
    function's last line; the docstring is not counted, so shortening it
    gains nothing.

    Parameters
    ----------
    module
        The parsed file to check.
    project
        The package index; not used by this check.
    config
        The settings; ``max_body_lines`` is the limit.

    Yields
    ------
    Violation
        One per function over the limit, giving its length.
    """
    del project
    limit = config.max_body_lines
    for function in functions(module.tree):
        body = function_body(function)
        if not body:
            continue
        last_line = function.end_lineno or function.lineno
        length = last_line - body[0].lineno + 1
        if length > limit:
            yield module.violation(
                "body-lines",
                function,
                f"The body is {length} lines; the limit is {limit}.",
            )


@module_check("parameters")
def check_parameters(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a function with more parameters than the limit (3.1).

    Every positional and keyword parameter counts, ``*args`` and
    ``**kwargs`` one each; a leading ``self`` or ``cls`` does not.

    Parameters
    ----------
    module
        The parsed file to check.
    project
        The package index; not used by this check.
    config
        The settings; ``max_parameters`` is the limit.

    Yields
    ------
    Violation
        One per function over the limit, giving its count.
    """
    del project
    limit = config.max_parameters
    for function in functions(module.tree):
        count = len(parameters(function, include_bound=False))
        if count > limit:
            yield module.violation(
                "parameters",
                function,
                f"The function takes {count} parameters; the limit is "
                f"{limit}.",
            )


@module_check("returned-values")
def check_returned_values(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a return of many values packed in a sequence or a dict (3.6).

    Each of these, with more than ``max_returned_values`` items, should
    be a frozen dataclass: a returned tuple or list literal, a returned
    name bound in the same function to one, a returned call of a
    named tuple, and a return annotated ``tuple[...]`` or with a named
    tuple. So should a returned dict literal, or ``dict(...)`` call,
    with three or more string keys, except in the modules that build
    JSON (``json_boundary_modules``). A *named tuple* is a class
    derived from ``NamedTuple``, or made by ``namedtuple(...)`` or
    ``NamedTuple(...)``, in the module or a package module it imports;
    its items are its fields.

    Parameters
    ----------
    module
        The parsed file to check.
    project
        The package index; the named tuples of the modules the file
        imports count, besides its own.
    config
        The settings; ``max_returned_values`` is the tuple limit.

    Yields
    ------
    Violation
        One per offending ``return`` statement, or one per function
        whose annotation alone declares too many values.
    """
    limit = config.max_returned_values
    allow_dicts = module_matches(module.dotted, config.json_boundary_modules)
    widths = named_tuple_widths(module, project)
    for function in functions(module.tree):
        yield from _returned_values(
            module, function, widths, limit=limit, allow_dicts=allow_dicts
        )


def _returned_values(
    module: Module,
    function: Function,
    widths: dict[str, int],
    *,
    limit: int,
    allow_dicts: bool,
) -> Iterator[Violation]:
    """Yield the returned-values findings of one function."""
    flagged = False
    sequences = _bound_sequences(function)
    for node in own_nodes(function):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        message = _return_message(node.value, sequences, widths, limit)
        if not message and not allow_dicts:
            message = _dict_message(node.value)
        if message:
            flagged = True
            yield module.violation("returned-values", node, message)

    count = _annotated_count(function.returns, widths)
    if not flagged and count > limit:
        yield module.violation(
            "returned-values",
            function,
            f"The function is annotated to return {count} values; "
            f"return at most {limit}, or a frozen dataclass.",
        )


def _return_message(
    value: ast.expr,
    sequences: dict[str, int],
    widths: dict[str, int],
    limit: int,
) -> str:
    """Say how a return packs too many values, or return ``""``."""
    count = 0
    if isinstance(value, (ast.Tuple, ast.List)):
        count = _fixed_length(value)
    elif isinstance(value, ast.Name):
        count = sequences.get(value.id, 0)
    elif isinstance(value, ast.Call):
        count = widths.get(last_name(value.func), 0)
    if count <= limit:
        return ""
    return (
        f"The function returns {count} values; return at most {limit}, "
        "or a frozen dataclass."
    )


def _dict_message(value: ast.expr) -> str:
    """Say that a return is a dict of results, or return ``""``."""
    if _record_keys(value) < MIN_RECORD_KEYS:
        return ""
    return (
        "The function returns a dict of results; return a frozen "
        "dataclass with named fields instead."
    )


def _bound_sequences(function: Function) -> dict[str, int]:
    """Map each name a function binds once to a tuple or list literal.

    The value is the literal's length. A name bound more than once, or
    whose value has an attribute read (``values.append(...)``), is a
    growing collection and is left out.
    """
    counts: dict[str, int] = {}
    lengths: dict[str, int] = {}
    for node in own_nodes(function):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            counts[target.id] = counts.get(target.id, 0) + 1
            if isinstance(node.value, (ast.Tuple, ast.List)):
                lengths[target.id] = _fixed_length(node.value)

    receivers = _attribute_owners(function)
    return {
        name: length
        for name, length in lengths.items()
        if counts[name] == 1 and name not in receivers
    }


def _attribute_owners(function: Function) -> set[str]:
    """Return the names a function reads an attribute of (``x.append``)."""
    return {
        node.value.id
        for node in own_nodes(function)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
    }


def _fixed_length(value: ast.Tuple | ast.List) -> int:
    """Count a literal's items, or return 0 when one is unpacked (``*x``)."""
    if any(isinstance(item, ast.Starred) for item in value.elts):
        return 0
    return len(value.elts)


def _annotated_count(returns: ast.expr | None, widths: dict[str, int]) -> int:
    """Count the values a return annotation declares, or return 0.

    ``tuple[a, b, c, d]`` declares four; ``tuple[a, ...]`` declares one
    sequence; a named tuple declares its fields.
    """
    if returns is None:
        return 0
    returns = parse_string(returns)
    if isinstance(returns, ast.Subscript):
        if last_name(returns.value).lower() != "tuple":
            return 0
        items = subscript_items(returns)
        if any(_is_ellipsis(item) for item in items):
            return 1
        return len(items)
    return widths.get(declared_class(returns), 0)


def _is_ellipsis(node: ast.expr) -> bool:
    """Tell whether an expression is the literal ``...``."""
    return isinstance(node, ast.Constant) and node.value is Ellipsis


def named_tuple_widths(module: Module, project: Project) -> dict[str, int]:
    """Map each named tuple a module can name to its number of fields.

    Parameters
    ----------
    module
        The parsed file; its own named tuples count, and those of the
        package modules it imports.
    project
        The package index, which holds the imported modules.

    Returns
    -------
    dict of str to int
        The number of fields of each named tuple, by its bare name; a
        name defined twice keeps the larger count.
    """
    sources = [module.tree]
    for name in sorted(project.imports.get(module.dotted, set())):
        imported = project.modules.get(name)
        if imported is not None and imported is not module:
            sources.append(imported.tree)

    widths: dict[str, int] = {}
    for tree in sources:
        for name, width in _named_tuples(tree):
            widths[name] = max(width, widths.get(name, 0))
    return widths


def _named_tuples(tree: ast.Module) -> Iterator[tuple[str, int]]:
    """Yield each named tuple a module defines, with its field count."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and _is_named_tuple_class(node):
            fields = [
                item for item in node.body if isinstance(item, ast.AnnAssign)
            ]
            yield node.name, len(fields)
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            width = _functional_width(node.value)
            target = node.targets[0]
            if width and isinstance(target, ast.Name):
                yield target.id, width


def _is_named_tuple_class(node: ast.ClassDef) -> bool:
    """Tell whether a class derives from ``NamedTuple``."""
    return any(last_name(base) == "NamedTuple" for base in node.bases)


def _functional_width(call: ast.Call) -> int:
    """Count the fields of ``namedtuple(...)`` or ``NamedTuple(...)``."""
    if last_name(call.func) not in NAMED_TUPLE_FACTORIES:
        return 0
    if len(call.args) < 2:
        return 0
    fields = call.args[1]
    text = string_value(fields)
    if text is not None:
        return len(text.replace(",", " ").split())
    if isinstance(fields, (ast.List, ast.Tuple)):
        return len(fields.elts)
    return 0


def _record_keys(value: ast.expr) -> int:
    """Count the string keys of a dict literal or a ``dict(...)`` call."""
    if isinstance(value, ast.Dict):
        return sum(
            1
            for key in value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        )
    is_dict_call = (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "dict"
    )
    if not is_dict_call:
        return 0
    assert isinstance(value, ast.Call)
    return sum(1 for keyword in value.keywords if keyword.arg is not None)


@module_check("nested-definition")
def check_nested_definition(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a function, class or lambda defined inside a function (3.4).

    Allowed: a function of at most ten lines whose name is only ever
    handed to calls as an argument (a callback, a ``functools.partial``
    or a ``key=``); a lambda passed as ``key=`` or ``default=``; and any
    nested definition in the decorators module, where a decorator's
    wrapper lives. A nested function that is returned, called or stored
    is a closure factory and is flagged.

    Parameters
    ----------
    module
        The parsed file to check.
    project
        The package index; not used by this check.
    config
        The settings; ``decorators_module`` names the exempt module.

    Yields
    ------
    Violation
        One per nested definition.
    """
    del project
    if module.dotted == config.decorators_module:
        return

    parents = parent_map(module.tree)
    for node in ast.walk(module.tree):
        if not isinstance(node, NESTED_NODES):
            continue
        outer = _nearest_scope(node, parents)
        if not isinstance(outer, (*FUNCTION_NODES, ast.Lambda)):
            continue
        if _is_allowed_nested(node, outer, parents):
            continue

        yield module.violation(
            "nested-definition",
            node,
            f"A {_kind(node)} is defined inside a function; move it to "
            "module level and pass it what it needs.",
        )


def _nearest_scope(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> ast.AST | None:
    """Return the nearest function, class or lambda around a node."""
    current = parents.get(node)
    while current is not None and not isinstance(current, NESTED_NODES):
        current = parents.get(current)
    return current


def _kind(node: ast.AST) -> str:
    """Name the kind of a nested definition for a message."""
    if isinstance(node, ast.Lambda):
        return "lambda"
    if isinstance(node, ast.ClassDef):
        return "class"
    return "function"


def _is_allowed_nested(
    node: ast.AST, outer: ast.AST, parents: dict[ast.AST, ast.AST]
) -> bool:
    """Tell whether a nested definition is one of the allowed forms."""
    if isinstance(node, ast.Lambda):
        parent = parents.get(node)
        return isinstance(parent, ast.keyword) and (
            parent.arg in LAMBDA_KEYWORDS
        )
    if not isinstance(node, FUNCTION_NODES):
        return False
    length = (node.end_lineno or node.lineno) - node.lineno + 1
    if length > MAX_CALLBACK_LINES:
        return False
    return _is_only_handed_to_calls(node.name, outer, parents)


def _is_only_handed_to_calls(
    name: str, outer: ast.AST, parents: dict[ast.AST, ast.AST]
) -> bool:
    """Tell whether every use of a name is as an argument of a call."""
    uses = [
        node
        for node in ast.walk(outer)
        if isinstance(node, ast.Name)
        and node.id == name
        and isinstance(node.ctx, ast.Load)
    ]
    if not uses:
        return False

    for use in uses:
        parent = parents.get(use)
        if isinstance(parent, ast.keyword):
            parent = parents.get(parent)
        if not isinstance(parent, ast.Call) or parent.func is use:
            return False
    return True


@module_check("forwarder")
def check_forwarder(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a function that only forwards its parameters or an attribute (3.3).

    The body, docstring aside, is one statement that either calls one
    function with exactly the function's own parameters, unchanged, or
    returns an attribute chain of a parameter (``job.layout.folder``).
    Exempt: properties, ``__dunder__`` methods, methods that override a
    base, methods that reach through a private attribute of ``self``
    (``self._queue.join()``), decorated functions (the decorator adds
    behaviour), and public functions whose docstring calls them a
    "public alias".

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
        One per forwarder.
    """
    del config
    parents = parent_map(module.tree)
    for function in functions(module.tree):
        if _is_exempt_forwarder(function, parents, project):
            continue
        message = _forwarder_message(function)
        if message:
            yield module.violation("forwarder", function, message)


def _is_exempt_forwarder(
    function: Function, parents: dict[ast.AST, ast.AST], project: Project
) -> bool:
    """Tell whether a function may be a forwarder by design."""
    if is_dunder(function.name) or is_property(function):
        return True
    if any(name not in PLAIN_DECORATORS for name in decorator_names(function)):
        return True
    if is_override(function, owner_class(function, parents), project):
        return True
    docstring = ast.get_docstring(function) or ""
    is_public = not function.name.startswith("_")
    return is_public and ALIAS_MARKER in docstring.lower()


def _forwarder_message(function: Function) -> str:
    """Say how a function forwards, or return ``""`` when it does not."""
    statement = _only_statement(function)
    if statement is None:
        return ""
    value = statement.value
    if isinstance(value, ast.Await):
        value = value.value

    if isinstance(value, ast.Call):
        return _call_forwarder_message(value, function)
    if isinstance(statement, ast.Return) and _is_parameter_attribute(
        value, function
    ):
        return (
            "The function only returns an attribute of a parameter; read "
            "the attribute directly."
        )
    return ""


def _only_statement(function: Function) -> ast.Return | ast.Expr | None:
    """Return a body's single ``return`` or expression statement, if any.

    Statements that do nothing (``pass``, a bare constant) are not
    counted, so padding a forwarder with one does not hide it.
    """
    body = [
        statement
        for statement in function_body(function)
        if not _is_no_op(statement)
    ]
    if len(body) == 1 and isinstance(body[0], (ast.Return, ast.Expr)):
        return body[0]
    return None


def _is_no_op(statement: ast.stmt) -> bool:
    """Tell whether a statement is ``pass`` or a bare constant."""
    if isinstance(statement, ast.Pass):
        return True
    return isinstance(statement, ast.Expr) and isinstance(
        statement.value, ast.Constant
    )


def _call_forwarder_message(call: ast.Call, function: Function) -> str:
    """Say that a call only forwards the parameters, or return ``""``."""
    if _is_private_delegation(call.func):
        return ""
    names = [
        argument.arg for argument in parameters(function, include_bound=False)
    ]
    if not _passes_exactly(call, names):
        return ""
    return (
        "The function only passes its parameters to another call; call "
        "that directly."
    )


def _is_private_delegation(node: ast.expr | None) -> bool:
    """Tell whether a chain from ``self`` goes through a private attribute.

    A method that delegates to ``self._queue.join()`` is the class's own
    interface to private state; callers cannot reach it any other way.
    """
    private = False
    while isinstance(node, ast.Attribute):
        if node.attr.startswith("_") and not is_dunder(node.attr):
            private = True
        node = node.value
    return private and isinstance(node, ast.Name) and node.id in BOUND_NAMES


def _passes_exactly(call: ast.Call, names: list[str]) -> bool:
    """Tell whether a call passes exactly these names, each once, unchanged.

    The callee must be a plain name or attribute chain, so the call does
    no other work.
    """
    if not _is_attribute_chain(call.func):
        return False
    passed = [_passed_name(argument) for argument in call.args]
    passed.extend(_passed_name(keyword.value) for keyword in call.keywords)
    if "" in passed:
        return False
    return sorted(passed) == sorted(names)


def _passed_name(argument: ast.expr) -> str:
    """Return the name an argument passes unchanged, or ``""``."""
    if isinstance(argument, ast.Starred):
        argument = argument.value
    if isinstance(argument, ast.Name):
        return argument.id
    return ""


def _is_attribute_chain(node: ast.expr) -> bool:
    """Tell whether an expression is a name or a chain of attributes."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name)


def _is_parameter_attribute(node: ast.expr | None, function: Function) -> bool:
    """Tell whether an expression reads an attribute chain of a parameter.

    For a method, ``self.name`` alone is an ordinary accessor; a chain
    two deep through ``self`` (``self.job.folder``) is a forwarder.
    """
    if not isinstance(node, ast.Attribute) or _is_private_delegation(node):
        return False
    depth = 0
    while isinstance(node, ast.Attribute):
        depth += 1
        node = node.value
    if not isinstance(node, ast.Name):
        return False
    names = {argument.arg for argument in parameters(function)}
    if node.id not in names:
        return False
    return node.id not in BOUND_NAMES or depth >= 2

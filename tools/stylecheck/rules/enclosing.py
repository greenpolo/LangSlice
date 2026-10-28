"""Find what encloses a definition, and whether a method overrides a base.

Several checks exempt a method whose signature another class fixes. A
method *overrides* when a decorator says so (``typing.override``, JPype's
``JOverride``), when a base class in the package defines the same name, or
when its class derives from a class outside the package that defines the
name. The checker cannot import a base from outside the package, so what
such a base defines is the table ``EXTERNAL_METHODS`` below, by the base's
bare name, with the dispatch prefixes of ``EXTERNAL_PREFIXES``
(``visit_`` of ``ast.NodeVisitor``, ``do_`` of an HTTP handler). Any
other method of a class with an external base is the class's own and is
checked like any method; an override the tables miss is marked with
``@override``. The plain bases below fix no signature and do not count.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator

from tools.stylecheck.core import ClassInfo, Project
from tools.stylecheck.rules.scopes import FUNCTION_NODES, dotted_name

#: Bases that give a class no inherited methods worth overriding, so a
#: class deriving only from these is treated as having no base.
PLAIN_BASES = frozenset(
    {
        "object",
        "ABC",
        "Protocol",
        "Generic",
        "NamedTuple",
        "TypedDict",
        "Enum",
        "StrEnum",
        "IntEnum",
        "Flag",
        "IntFlag",
        "Exception",
    }
)

#: The HTTP request handler methods a server calls on a subclass.
_HTTP_HANDLER_METHODS = frozenset(
    {
        "handle",
        "handle_one_request",
        "setup",
        "finish",
        "log_message",
        "log_request",
        "log_error",
        "send_error",
        "end_headers",
        "version_string",
        "address_string",
    }
)

#: The language model methods Google ADK calls on a model class.
_ADK_MODEL_METHODS = frozenset(
    {"supported_models", "generate_content_async", "connect"}
)

#: The callbacks Google ADK calls on a plugin.
_ADK_PLUGIN_METHODS = frozenset(
    {
        "on_user_message_callback",
        "before_run_callback",
        "after_run_callback",
        "on_event_callback",
        "before_agent_callback",
        "after_agent_callback",
        "before_model_callback",
        "after_model_callback",
        "on_model_error_callback",
        "before_tool_callback",
        "after_tool_callback",
        "on_tool_error_callback",
        "on_agent_error_callback",
        "on_run_error_callback",
        "close",
    }
)

#: The methods numpydoc's validator reads, which its subclass replaces.
_NUMPYDOC_VALIDATOR_METHODS = frozenset(
    {
        "name",
        "type",
        "source_file_name",
        "source_file_def_line",
        "signature_parameters",
        "method_source",
        "method_returns_something",
    }
)

#: The methods each base from outside the package defines for a subclass
#: to replace, by the base's bare name: the standard library's classes
#: whose methods are overridden by design, and the framework classes the
#: package derives from. A name not listed is the subclass's own method.
EXTERNAL_METHODS: dict[str, frozenset[str]] = {
    "JSONEncoder": frozenset({"default", "encode", "iterencode"}),
    "JSONDecoder": frozenset({"decode", "raw_decode"}),
    "NodeVisitor": frozenset({"visit", "generic_visit"}),
    "NodeTransformer": frozenset({"visit", "generic_visit"}),
    "Thread": frozenset({"run"}),
    "Process": frozenset({"run"}),
    "Filter": frozenset({"filter"}),
    "Formatter": frozenset({"format", "formatTime", "formatException"}),
    "Handler": frozenset({"emit", "handle", "format", "flush", "close"}),
    "StreamHandler": frozenset({"emit", "flush", "close"}),
    "BaseHTTPRequestHandler": _HTTP_HANDLER_METHODS,
    "SimpleHTTPRequestHandler": _HTTP_HANDLER_METHODS,
    "BaseRequestHandler": frozenset({"handle", "setup", "finish"}),
    "StreamRequestHandler": frozenset({"handle", "setup", "finish"}),
    "ArgumentParser": frozenset(
        {"error", "exit", "format_help", "format_usage", "parse_known_args"}
    ),
    "Cmd": frozenset({"default", "emptyline", "precmd", "postcmd"}),
    "TestCase": frozenset({"setUp", "tearDown", "setUpClass"}),
    "Optimizer": frozenset({"step", "zero_grad"}),
    "WarpSGD": frozenset({"step", "zero_grad"}),
    "BasePlugin": _ADK_PLUGIN_METHODS,
    "BaseLlm": _ADK_MODEL_METHODS,
    "AnthropicLlm": _ADK_MODEL_METHODS
    | frozenset({"_build_anthropic_kwargs", "_anthropic_client"}),
    "Validator": _NUMPYDOC_VALIDATOR_METHODS,
    "AstValidator": _NUMPYDOC_VALIDATOR_METHODS,
}

#: The method-name prefixes a standard library base dispatches on.
EXTERNAL_PREFIXES: dict[str, tuple[str, ...]] = {
    "NodeVisitor": ("visit_",),
    "NodeTransformer": ("visit_",),
    "BaseHTTPRequestHandler": ("do_",),
    "SimpleHTTPRequestHandler": ("do_",),
    "Cmd": ("do_", "help_", "complete_"),
    "TestCase": ("test",),
}

#: Decorators that make a method a property, read as an attribute.
PROPERTY_DECORATORS = frozenset(
    {"property", "cached_property", "functools.cached_property"}
)

#: How many project base classes deep an override is looked for; a
#: deeper hierarchy is treated as an override.
MAX_BASE_DEPTH = 8

Function = ast.FunctionDef | ast.AsyncFunctionDef


def parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    """Map every node under a tree to the node that contains it.

    Parameters
    ----------
    tree
        A parsed module or any node.

    Returns
    -------
    dict
        Each node, except ``tree`` itself, to its parent.
    """
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def body_nodes(function: Function) -> Iterator[ast.AST]:
    """Yield every node in a function's body, at any depth.

    Parameters
    ----------
    function
        The function whose body is walked; its signature and decorators
        are left out.

    Yields
    ------
    ast.AST
        Each node of each statement, functions nested in it included.
    """
    for statement in function.body:
        yield from ast.walk(statement)


def enclosing_function(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> Function | ast.Lambda | None:
    """Return the nearest function or lambda around a node.

    Parameters
    ----------
    node
        Any node of the tree ``parents`` was built from.
    parents
        The tree's parent map, from :func:`parent_map`.

    Returns
    -------
    ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda or None
        The innermost function or lambda containing the node, looking
        through classes; None at module level.
    """
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (*FUNCTION_NODES, ast.Lambda)):
            return current
        current = parents.get(current)
    return None


def owner_class(
    function: Function, parents: dict[ast.AST, ast.AST]
) -> ast.ClassDef | None:
    """Return the class a function is written in, if it is a method.

    Parameters
    ----------
    function
        A function of the tree ``parents`` was built from.
    parents
        The tree's parent map.

    Returns
    -------
    ast.ClassDef or None
        The class whose body holds the function directly, or None.
    """
    parent = parents.get(function)
    if isinstance(parent, ast.ClassDef):
        return parent
    return None


def is_dunder(name: str) -> bool:
    """Tell whether a name is a ``__dunder__`` name.

    Parameters
    ----------
    name
        A function or attribute name.

    Returns
    -------
    bool
        True when the name starts and ends with two underscores.
    """
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def decorator_names(node: Function | ast.ClassDef) -> list[str]:
    """Return the dotted name of each decorator, without its arguments.

    Parameters
    ----------
    node
        A decorated function or class.

    Returns
    -------
    list of str
        ``"functools.cache"`` for ``@functools.cache()``; ``""`` for a
        decorator that is not a name or attribute chain.
    """
    names: list[str] = []
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Call):
            decorator = decorator.func
        names.append(dotted_name(decorator))
    return names


def is_property(function: Function) -> bool:
    """Tell whether a method is a property, its setter or its deleter.

    Parameters
    ----------
    function
        The method whose decorators are read.

    Returns
    -------
    bool
        True for ``@property``, ``@cached_property`` and ``@x.setter``.
    """
    for name in decorator_names(function):
        if name in PROPERTY_DECORATORS:
            return True
        if name.endswith((".setter", ".deleter", ".getter")):
            return True
    return False


def is_override(
    function: Function, owner: ast.ClassDef | None, project: Project
) -> bool:
    """Tell whether a method's signature may be fixed by a base class.

    Parameters
    ----------
    function
        The function to judge.
    owner
        The class it is written in, or None for a plain function.
    project
        The package index, which knows the package's own classes.

    Returns
    -------
    bool
        True when a decorator marks an override, or when a base of
        ``owner`` defines the same name: a package class that does, or a
        class outside the package whose entry in ``EXTERNAL_METHODS`` or
        ``EXTERNAL_PREFIXES`` covers it.
    """
    for name in decorator_names(function):
        if name.rsplit(".", 1)[-1].lower().endswith("override"):
            return True
    if owner is None:
        return False
    return _bases_define(owner, function.name, project, depth=0)


def _bases_define(
    owner: ast.ClassDef, name: str, project: Project, *, depth: int
) -> bool:
    """Tell whether a base of ``owner`` defines ``name``."""
    if depth > MAX_BASE_DEPTH:
        return True
    for base_name in _base_names(owner):
        infos = project.classes_named(base_name)
        if not infos and _external_defines(base_name, name):
            return True
        for info in infos:
            if _class_defines(info, name, project, depth=depth):
                return True
    return False


def _external_defines(base_name: str, name: str) -> bool:
    """Tell whether a base outside the package defines a method name."""
    if name in EXTERNAL_METHODS.get(base_name, frozenset()):
        return True
    return name.startswith(EXTERNAL_PREFIXES.get(base_name, ()))


def _class_defines(
    info: ClassInfo, name: str, project: Project, *, depth: int
) -> bool:
    """Tell whether a package class or one of its bases defines a name."""
    if name in info.attributes:
        return True
    class_name = info.dotted.rsplit(".", 1)[-1]
    base = _class_node(project, info.module, class_name)
    if base is None:
        return False
    return _bases_define(base, name, project, depth=depth + 1)


def _base_names(owner: ast.ClassDef) -> list[str]:
    """Return the bare names of a class's bases, plain bases left out."""
    names: list[str] = []
    for base in owner.bases:
        if isinstance(base, ast.Subscript):
            base = base.value
        name = dotted_name(base).rsplit(".", 1)[-1]
        if name not in PLAIN_BASES:
            names.append(name)
    return names


def _class_node(
    project: Project, module_name: str, class_name: str
) -> ast.ClassDef | None:
    """Return a module-level class of a package module by its name."""
    module = project.modules.get(module_name)
    if module is None:
        return None
    for node in module.tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None

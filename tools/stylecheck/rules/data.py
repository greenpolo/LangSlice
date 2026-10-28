"""Checks on data and types: ``Any``, loose records, untrusted reads.

These checks read annotations as written; they resolve type aliases
defined in the same module (see the ``annotations`` helpers) but do no
type inference. A *package class* is a class defined in a checked module,
with the attributes the project index records for it.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator

from tools.stylecheck.core import (
    Config,
    Module,
    Project,
    Severity,
    Violation,
    module_check,
)
from tools.stylecheck.rules.annotations import (
    Aliases,
    contains_any,
    contains_loose_record,
    declared_class,
    last_name,
    mentions_syntax_node,
    module_aliases,
)
from tools.stylecheck.rules.scopes import (
    BOUND_NAMES,
    call_name,
    dotted_name,
    functions,
    methods,
    module_matches,
    own_nodes,
    parameters,
    string_value,
)

#: Calls that read an attribute by a name given as a string.
ATTRIBUTE_READERS = frozenset({"getattr", "hasattr"})


def _signature_annotations(
    tree: ast.Module,
) -> Iterator[tuple[ast.AST, str, ast.expr]]:
    """Yield each annotation of a signature or a class field.

    Each item is the node to report at, a phrase naming what is
    annotated, and the annotation itself.
    """
    for function in functions(tree):
        yield from _function_annotations(function)

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if isinstance(item, ast.AnnAssign):
                field = dotted_name(item.target)
                yield item, f"Field {field!r}", item.annotation


def _function_annotations(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Iterator[tuple[ast.AST, str, ast.expr]]:
    """Yield a function's parameter and return annotations."""
    for parameter in parameters(function):
        if parameter.annotation is not None:
            subject = f"Parameter {parameter.arg!r}"
            yield parameter, subject, parameter.annotation

    if function.returns is not None:
        subject = f"The return of {function.name!r}"
        yield function, subject, function.returns


@module_check("any-annotation")
def check_any_annotation(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag ``Any`` in a parameter, return or class-field annotation.

    Guide rule 5.3. ``Any`` is found directly or through a type alias
    defined in the same module. ``Any`` as the value type of a mapping
    keyed by ``str`` is left to the loose-record check. ``object`` is
    allowed.

    Parameters
    ----------
    module
        The parsed file whose annotations are read.
    project
        The package index, unused by this check.
    config
        The settings, unused by this check.

    Yields
    ------
    Violation
        One finding per annotation that uses ``Any``.
    """
    aliases = module_aliases(module, project)
    for node, subject, annotation in _signature_annotations(module.tree):
        if contains_any(annotation, aliases):
            yield module.violation(
                "any-annotation",
                node,
                f"{subject} is annotated with Any; declare the type, or "
                "use object where it really is anything (5.3).",
            )


@module_check("loose-record")
def check_loose_record(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag records without declared fields.

    Guide rule 5.1. Outside ``json_boundary_modules`` this flags a
    signature, class field or module variable annotated with a mapping
    from ``str`` to ``Any`` or ``object`` (``dict[str, Any]``,
    ``Mapping[str, object]``, an in-module alias of either), and every
    ``TypedDict``, in class or functional form. Inside those modules it
    flags a ``TypedDict`` with more than half its keys not required.

    Parameters
    ----------
    module
        The parsed file whose annotations and classes are read.
    project
        The package index, unused by this check.
    config
        The settings that list the JSON boundary modules.

    Yields
    ------
    Violation
        One finding per loose annotation or ``TypedDict``.
    """
    if module_matches(module.dotted, config.json_boundary_modules):
        yield from _optional_typed_dicts(module)
        return

    aliases = module_aliases(module, project)
    for node, subject, annotation in _record_annotations(module.tree):
        if contains_loose_record(annotation, aliases):
            yield module.violation(
                "loose-record",
                node,
                f"{subject} is a mapping from str to anything; declare "
                "the record as a frozen dataclass (5.1).",
            )

    for node, name in _typed_dicts(module.tree):
        yield module.violation(
            "loose-record",
            node,
            f"TypedDict {name!r} is outside the JSON boundary modules; "
            "declare the record as a frozen dataclass (5.1).",
        )


def _record_annotations(
    tree: ast.Module,
) -> Iterator[tuple[ast.AST, str, ast.expr]]:
    """Yield signature, field and module-variable annotations."""
    yield from _signature_annotations(tree)
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            name = dotted_name(node.target)
            yield node, f"Module variable {name!r}", node.annotation


def _typed_dicts(tree: ast.Module) -> Iterator[tuple[ast.AST, str]]:
    """Yield each ``TypedDict`` a module defines, with its name."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and _is_typed_dict_class(node):
            yield node, node.name
        elif isinstance(node, ast.Assign) and _is_typed_dict_call(node.value):
            yield node, dotted_name(node.targets[0])


def _is_typed_dict_class(node: ast.ClassDef) -> bool:
    """Tell whether a class derives from ``TypedDict``."""
    return any(last_name(base) == "TypedDict" for base in node.bases)


def _is_typed_dict_call(node: ast.expr) -> bool:
    """Tell whether an expression is the functional ``TypedDict`` form."""
    return call_name(node).rpartition(".")[2] == "TypedDict"


def _optional_typed_dicts(module: Module) -> Iterator[Violation]:
    """Flag a boundary ``TypedDict`` with most of its keys optional."""
    for node, name in _typed_dicts(module.tree):
        n_optional, n_keys = _key_counts(node)
        if n_keys and 2 * n_optional > n_keys:
            yield module.violation(
                "loose-record",
                node,
                f"TypedDict {name!r} has {n_optional} of {n_keys} keys "
                "not required; at most half may be optional (5.1).",
            )


def _key_counts(node: ast.AST) -> tuple[int, int]:
    """Count a ``TypedDict``'s optional keys and all its keys."""
    keywords, values = _typed_dict_parts(node)
    total = True
    for keyword in keywords:
        if keyword.arg == "total" and isinstance(keyword.value, ast.Constant):
            total = bool(keyword.value.value)

    n_optional = sum(1 for value in values if _is_optional_key(value, total))
    return n_optional, len(values)


def _typed_dict_parts(
    node: ast.AST,
) -> tuple[list[ast.keyword], list[ast.expr]]:
    """Return a ``TypedDict``'s keywords and the annotations of its keys."""
    if isinstance(node, ast.ClassDef):
        values = [
            item.annotation
            for item in node.body
            if isinstance(item, ast.AnnAssign)
        ]
        return node.keywords, values
    if not isinstance(node, ast.Assign) or not isinstance(
        node.value, ast.Call
    ):
        return [], []
    call = node.value
    if len(call.args) < 2 or not isinstance(call.args[1], ast.Dict):
        return call.keywords, []
    return call.keywords, list(call.args[1].values)


def _is_optional_key(annotation: ast.expr, total: bool) -> bool:
    """Tell whether a ``TypedDict`` key may be left out."""
    if isinstance(annotation, ast.Subscript):
        wrapper = last_name(annotation.value)
        if wrapper == "NotRequired":
            return True
        if wrapper == "Required":
            return False
    return not total


@module_check("or-empty-get")
def check_or_empty_get(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag reads that dig through undeclared nested mappings.

    Guide rule 5.1. Flags ``(x or {}).get(...)`` and a ``.get(...)`` on
    the result of another ``.get(...)``, including through ``or {}``. A
    chain is reported once, at its outermost read.

    Parameters
    ----------
    module
        The parsed file whose calls are read.
    project
        The package index, unused by this check.
    config
        The settings, unused by this check.

    Yields
    ------
    Violation
        One finding per chain of reads.
    """
    inner: set[int] = set()
    for node in ast.walk(module.tree):
        receiver = _get_receiver(node)
        if receiver is None or id(node) in inner:
            continue
        source = _defaulted_source(receiver)
        if source is None:
            continue
        inner.add(id(source))
        yield module.violation(
            "or-empty-get",
            node,
            "A .get chain reads an undeclared nested record; declare the "
            "record as a dataclass and read its fields (5.1).",
        )


def _get_receiver(node: ast.AST) -> ast.expr | None:
    """Return what ``.get`` is called on, when the node is such a call."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "get":
        return func.value
    return None


def _defaulted_source(receiver: ast.expr) -> ast.expr | None:
    """Return the inner read when a receiver is ``x or {}`` or a ``.get``.

    The inner read is returned so the caller can skip reporting it again.
    """
    if _get_receiver(receiver) is not None:
        return receiver
    if not isinstance(receiver, ast.BoolOp):
        return None
    if not isinstance(receiver.op, ast.Or):
        return None
    if not _is_empty_mapping(receiver.values[-1]):
        return None
    return receiver.values[0]


def _is_empty_mapping(node: ast.expr) -> bool:
    """Tell whether an expression is ``{}`` or ``dict()``."""
    if isinstance(node, ast.Dict):
        return not node.keys
    if isinstance(node, ast.Call) and call_name(node) == "dict":
        return not node.args and not node.keywords
    return False


@module_check("literal-getattr")
def check_literal_getattr(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag reading a declared attribute by its name as a string.

    Guide rule 5.4. Flags ``getattr(x, "name")``, ``hasattr(x, "name")``,
    ``vars(x)["name"]`` and ``x.__dict__["name"]`` where ``x`` is a
    parameter annotated with a package class (or ``self`` in a method)
    whose declared attributes include ``name``. A private name on another
    object is left to the private-getattr check.

    Parameters
    ----------
    module
        The parsed file whose functions are read.
    project
        The package index, which knows each class's attributes.
    config
        The settings, unused by this check.

    Yields
    ------
    Violation
        One finding per read of a declared attribute by string.
    """
    for function, classes in _typed_parameters(module, project):
        for node in own_nodes(function):
            read = _string_attribute_read(node)
            if read is None:
                continue
            owner, attribute = read
            if owner not in BOUND_NAMES and attribute.startswith("_"):
                continue
            if _declares(project, classes.get(owner, ""), attribute):
                yield module.violation(
                    "literal-getattr",
                    node,
                    f"Attribute {attribute!r} is declared on {owner!r}'s "
                    "type; read it as an attribute (5.4).",
                )


def _typed_parameters(
    module: Module, project: Project
) -> Iterator[tuple[ast.AST, dict[str, str]]]:
    """Yield each function with its parameters' declared class names.

    ``self`` and ``cls`` map to the enclosing class, by its dotted name;
    other parameters map to the bare class their annotation declares.
    """
    owners: dict[int, str] = {}
    for owner, method in methods(module.tree):
        owners[id(method)] = f"{module.dotted}.{owner.name}"
    for function in functions(module.tree):
        classes: dict[str, str] = {}
        for parameter in parameters(function):
            if parameter.annotation is not None:
                classes[parameter.arg] = declared_class(parameter.annotation)
        if id(function) in owners:
            classes["self"] = owners[id(function)]
            classes["cls"] = owners[id(function)]
        yield function, classes


def _string_attribute_read(node: ast.AST) -> tuple[str, str] | None:
    """Return the object and attribute of a read by string name."""
    if isinstance(node, ast.Call) and call_name(node) in ATTRIBUTE_READERS:
        if len(node.args) >= 2 and isinstance(node.args[0], ast.Name):
            attribute = string_value(node.args[1])
            if attribute is not None:
                return node.args[0].id, attribute
        return None
    if not isinstance(node, ast.Subscript):
        return None
    attribute = string_value(node.slice)
    owner = _namespace_owner(node.value)
    if attribute is None or not owner:
        return None
    return owner, attribute


def _namespace_owner(node: ast.expr) -> str:
    """Return ``x`` for ``vars(x)`` or ``x.__dict__``, else ``""``."""
    if isinstance(node, ast.Call) and call_name(node) == "vars":
        if len(node.args) == 1 and isinstance(node.args[0], ast.Name):
            return node.args[0].id
        return ""
    if isinstance(node, ast.Attribute) and node.attr == "__dict__":
        if isinstance(node.value, ast.Name):
            return node.value.id
    return ""


def _declares(project: Project, class_name: str, attribute: str) -> bool:
    """Tell whether a package class of that name declares an attribute."""
    if not class_name:
        return False
    if class_name in project.classes:
        return attribute in project.classes[class_name].attributes
    for info in project.classes_named(class_name):
        if attribute in info.attributes:
            return True
    return False


@module_check("isinstance-outside-checkers")
def check_isinstance_outside_checkers(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Warn on ``isinstance`` applied to a parameter below the door.

    Guide rule 5.6, a warning. Inputs are validated where they come in:
    tool arguments and host inputs with the shared checkers in
    ``checkers_module``, and parsed files in the modules that read them
    (``json_boundary_modules``). Below that, code trusts its inputs. A
    test of a syntax node's kind is dispatch, not validation, and is not
    flagged: a parameter annotated with an ``ast`` class, or a test whose
    every class is an attribute of ``ast``.

    Parameters
    ----------
    module
        The parsed file whose functions are read.
    project
        The package index, unused by this check.
    config
        The settings that name the checkers module and the modules that
        read files.

    Yields
    ------
    Violation
        One warning per ``isinstance`` on a parameter.
    """
    doors = (config.checkers_module, *config.json_boundary_modules)
    if module_matches(module.dotted, doors):
        return

    aliases = module_aliases(module, project)
    for function in functions(module.tree):
        names = _untyped_by_syntax(function, aliases)
        for node in own_nodes(function):
            name = _isinstance_subject(node)
            if name in names and name not in BOUND_NAMES:
                yield module.violation(
                    "isinstance-outside-checkers",
                    node,
                    f"isinstance on parameter {name!r} validates below the "
                    "door; validate where the value comes in, with the "
                    "shared checkers (5.6).",
                    severity=Severity.WARNING,
                )


def _untyped_by_syntax(
    function: ast.FunctionDef | ast.AsyncFunctionDef, aliases: Aliases
) -> set[str]:
    """Return a function's parameters not annotated as syntax nodes."""
    names: set[str] = set()
    for parameter in parameters(function):
        annotation = parameter.annotation
        if annotation is None or not mentions_syntax_node(annotation, aliases):
            names.add(parameter.arg)
    return names


def _isinstance_subject(node: ast.AST) -> str:
    """Return the name an ``isinstance`` call tests, unless it is on ast."""
    if not isinstance(node, ast.Call) or call_name(node) != "isinstance":
        return ""
    if len(node.args) != 2 or not isinstance(node.args[0], ast.Name):
        return ""
    if _names_syntax_nodes(node.args[1]):
        return ""
    return node.args[0].id


def _names_syntax_nodes(node: ast.expr) -> bool:
    """Tell whether every class in an isinstance argument is from ast."""
    classes = node.elts if isinstance(node, ast.Tuple) else [node]
    return all(dotted_name(item).startswith("ast.") for item in classes)

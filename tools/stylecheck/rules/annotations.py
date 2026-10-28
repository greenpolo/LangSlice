"""Read type annotations: the class they declare, ``Any``, loose records.

A *type alias* here is a module-level assignment of a type expression to a
name, such as ``Record = dict[str, Any]``. The helpers resolve aliases
defined in the same module, and aliases imported by name from another
module of the package, so a check sees through ``Record`` to what it
names. A *loose record* is a mapping from ``str`` to ``Any`` or
``object``: a record whose fields nobody declared. This module registers
no check.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator

from tools.stylecheck.core import Module, Project, imported_modules
from tools.stylecheck.rules.scopes import dotted_name, string_value

#: Subscripted names that only wrap the type they declare.
WRAPPER_NAMES = frozenset(
    {"Annotated", "Final", "ClassVar", "Required", "NotRequired", "ReadOnly"}
)

#: Mapping types that become a loose record when keyed by ``str`` and
#: valued by ``Any`` or ``object``.
MAPPING_NAMES = frozenset(
    {"dict", "Dict", "Mapping", "MutableMapping", "defaultdict"}
)

#: The value types that leave a record's fields undeclared.
LOOSE_VALUE_NAMES = frozenset({"Any", "object"})

#: The modules whose type names a module may import under another name
#: (``from typing import Any as A``); the alias stands for the name.
TYPING_MODULES = frozenset(
    {"typing", "typing_extensions", "collections", "collections.abc"}
)

Aliases = dict[str, ast.expr]


def type_aliases(tree: ast.Module) -> Aliases:
    """Collect the module-level names bound to a type expression.

    Parameters
    ----------
    tree
        The parsed module whose top-level statements are read.

    Returns
    -------
    dict of str to ast.expr
        Each alias name with the expression it stands for. Any
        subscript, union, name or attribute counts; a name that is not a
        type is harmless, since only names used in annotations are looked
        up.
    """
    aliases: Aliases = {}
    type_nodes = (ast.Subscript, ast.BinOp, ast.Name, ast.Attribute)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target = node.target
            value = node.value
        else:
            continue
        if isinstance(target, ast.Name) and isinstance(value, type_nodes):
            aliases[target.id] = value
    return aliases


def module_aliases(module: Module, project: Project) -> Aliases:
    """Collect a module's type aliases, with those it imports by name.

    Parameters
    ----------
    module
        The parsed file whose aliases are wanted.
    project
        The package index, whose modules supply imported aliases.

    Returns
    -------
    dict of str to ast.expr
        The aliases the module defines; each top-level
        ``from x import Name`` where ``Name`` is an alias in package
        module ``x``, under the name it is bound to here; and each name
        of ``typing`` or ``collections.abc`` imported under another
        name (``from typing import Any as A``), as the name it stands
        for.
    """
    aliases: Aliases = {}
    for node in module.tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module in TYPING_MODULES and not node.level:
            aliases.update(_renamed_typing_names(node))
        else:
            aliases.update(_imported_aliases(module, project, node))
    aliases.update(type_aliases(module.tree))
    return aliases


def _imported_aliases(
    module: Module, project: Project, node: ast.ImportFrom
) -> Aliases:
    """Map the package type aliases a ``from`` import binds, by local name."""
    source = project.modules.get(imported_modules(module, node)[0])
    if source is None:
        return {}
    imported = type_aliases(source.tree)
    return {
        alias.asname or alias.name: imported[alias.name]
        for alias in node.names
        if alias.name in imported
    }


def _renamed_typing_names(node: ast.ImportFrom) -> Aliases:
    """Map each name a ``from typing import X as Y`` renames to ``X``."""
    return {
        alias.asname: ast.Name(id=alias.name, ctx=ast.Load())
        for alias in node.names
        if alias.asname and alias.asname != alias.name
    }


def mentions_syntax_node(node: ast.expr, aliases: Aliases) -> bool:
    """Tell whether an annotation names a class from the ``ast`` module.

    Parameters
    ----------
    node
        An annotation, possibly quoted.
    aliases
        The module's type aliases, followed where the annotation uses one.

    Returns
    -------
    bool
        True when a part of the annotation is spelled ``ast.<name>``.
    """
    for part in _expanded(parse_string(node), aliases, set()):
        if dotted_name(part).startswith("ast."):
            return True
    return False


def parse_string(node: ast.expr) -> ast.expr:
    """Return the expression a string annotation spells.

    Parameters
    ----------
    node
        An annotation, possibly a quoted forward reference.

    Returns
    -------
    ast.expr
        The parsed expression for a string that parses, else ``node``.
    """
    text = string_value(node)
    if text is None:
        return node
    try:
        return ast.parse(text, mode="eval").body
    except SyntaxError:
        return node


def last_name(node: ast.AST) -> str:
    """Return the last part of a dotted name.

    Parameters
    ----------
    node
        Any expression node.

    Returns
    -------
    str
        ``"Path"`` for ``pathlib.Path`` or ``Path``; ``""`` for any other
        expression.
    """
    return dotted_name(node).rpartition(".")[2]


def subscript_items(node: ast.Subscript) -> list[ast.expr]:
    """Return the expressions inside a subscript's brackets.

    Parameters
    ----------
    node
        A subscripted type such as ``dict[str, Any]``.

    Returns
    -------
    list of ast.expr
        The items of a tuple subscript, or the single item.
    """
    if isinstance(node.slice, ast.Tuple):
        return list(node.slice.elts)
    return [node.slice]


def union_members(node: ast.expr) -> list[ast.expr]:
    """Split a union annotation into its members.

    Parameters
    ----------
    node
        An annotation: ``X | None``, ``Optional[X]``, ``Union[X, Y]`` or
        a plain type.

    Returns
    -------
    list of ast.expr
        The member types; a plain type is its own single member.
    """
    node = parse_string(node)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return [*union_members(node.left), *union_members(node.right)]
    if isinstance(node, ast.Subscript):
        base = last_name(node.value)
        if base == "Optional":
            return union_members(node.slice)
        if base == "Union":
            members: list[ast.expr] = []
            for item in subscript_items(node):
                members.extend(union_members(item))
            return members
    return [node]


def declared_class(annotation: ast.expr) -> str:
    """Return the bare class name an annotation declares.

    Parameters
    ----------
    annotation
        A parameter's annotation; ``None`` members of a union and
        wrappers such as ``Annotated`` are looked through.

    Returns
    -------
    str
        ``"Job"`` for ``Job``, ``"Job" | None`` or ``Optional[jobs.Job]``;
        ``""`` when the annotation names no single class.
    """
    return declared_path(annotation).rpartition(".")[2]


def declared_path(annotation: ast.expr) -> str:
    """Return the class an annotation declares, dotted as it is written.

    Parameters
    ----------
    annotation
        A parameter's annotation; ``None`` members of a union and
        wrappers such as ``Annotated`` are looked through.

    Returns
    -------
    str
        ``"jobs.Job"`` for ``Optional[jobs.Job]``, ``"Job"`` for
        ``"Job" | None``; ``""`` when the annotation names no single
        class.
    """
    members = [
        member for member in union_members(annotation) if not _is_none(member)
    ]
    if len(members) != 1:
        return ""
    member = members[0]
    if isinstance(member, ast.Subscript):
        if last_name(member.value) not in WRAPPER_NAMES:
            return ""
        return declared_path(subscript_items(member)[0])
    return dotted_name(member)


def _is_none(node: ast.expr) -> bool:
    """Tell whether an annotation member is ``None``."""
    return isinstance(node, ast.Constant) and node.value is None


def is_any(node: ast.AST) -> bool:
    """Tell whether an expression is ``Any`` or ``typing.Any``.

    Parameters
    ----------
    node
        Any expression node.

    Returns
    -------
    bool
        True for the bare name or an attribute whose last part is it.
    """
    return last_name(node) == "Any"


def is_loose_mapping(node: ast.AST, aliases: Aliases) -> bool:
    """Tell whether a node is a mapping from ``str`` to anything.

    Parameters
    ----------
    node
        Any expression node, read without descending into it.
    aliases
        The module's type aliases, used to resolve the mapping, key and
        value types.

    Returns
    -------
    bool
        True for ``dict[str, Any]``, ``Mapping[str, object]``,
        ``MutableMapping[str, object | None]`` and the like, with any of
        the names possibly an alias of those. A union value is loose
        when one of its members is ``Any`` or ``object``, since the
        union is then no narrower.
    """
    if not isinstance(node, ast.Subscript):
        return False
    if resolved_name(node.value, aliases) not in MAPPING_NAMES:
        return False
    items = subscript_items(node)
    if len(items) != 2 or resolved_name(items[0], aliases) != "str":
        return False
    return _is_loose_value(items[1], aliases, set())


def resolved_name(node: ast.expr, aliases: Aliases) -> str:
    """Return the last part of a name, after following aliases of names.

    Parameters
    ----------
    node
        Any expression node, possibly a quoted name.
    aliases
        The module's type aliases.

    Returns
    -------
    str
        ``"Any"`` for ``A`` after ``from typing import Any as A``; the
        last dotted part of any other name; ``""`` for an expression
        that is not a name.
    """
    node = parse_string(node)
    seen: set[str] = set()
    while isinstance(node, ast.Name) and node.id in aliases:
        if node.id in seen:
            break
        seen.add(node.id)
        node = aliases[node.id]
    return last_name(node)


def _is_loose_value(node: ast.expr, aliases: Aliases, seen: set[str]) -> bool:
    """Tell whether a mapping's value type is ``Any``, ``object`` or wider."""
    node = parse_string(node)
    members = union_members(node)
    if members != [node]:
        return any(
            _is_loose_value(member, aliases, seen) for member in members
        )
    name = last_name(node)
    if name in LOOSE_VALUE_NAMES:
        return True
    if isinstance(node, ast.Name) and name in aliases and name not in seen:
        return _is_loose_value(aliases[name], aliases, seen | {name})
    return False


def contains_loose_record(node: ast.expr, aliases: Aliases) -> bool:
    """Tell whether an annotation holds a loose record anywhere in it.

    Parameters
    ----------
    node
        An annotation, possibly quoted.
    aliases
        The module's type aliases, followed where the annotation uses one.

    Returns
    -------
    bool
        True when the annotation, or an alias it uses, contains a mapping
        from ``str`` to ``Any`` or ``object``.
    """
    for part in _expanded(parse_string(node), aliases, set()):
        if is_loose_mapping(part, aliases):
            return True
    return False


def contains_any(node: ast.expr, aliases: Aliases) -> bool:
    """Tell whether an annotation uses ``Any`` outside a loose record.

    Parameters
    ----------
    node
        An annotation, possibly quoted.
    aliases
        The module's type aliases, followed where the annotation uses one.

    Returns
    -------
    bool
        True when ``Any`` appears in the annotation or an alias it uses,
        other than as the value type of a loose record, which the
        loose-record check reports instead.
    """
    for part in _expanded(parse_string(node), aliases, set()):
        if is_any(part):
            return True
    return False


def _expanded(
    node: ast.AST, aliases: Aliases, seen: set[str]
) -> Iterator[ast.AST]:
    """Yield an annotation's nodes, entering aliases, skipping record values.

    A loose mapping is yielded itself, then only its key type is entered,
    so ``Any`` in its value position is left to the loose-record check.
    """
    yield node
    if isinstance(node, ast.Name) and node.id in aliases:
        if node.id not in seen:
            yield from _expanded(aliases[node.id], aliases, seen | {node.id})
        return

    if isinstance(node, ast.Constant):
        parsed = parse_string(node)
        if parsed is not node:
            yield from _expanded(parsed, aliases, seen)
        return

    if isinstance(node, ast.Subscript) and is_loose_mapping(node, aliases):
        children: list[ast.AST] = [subscript_items(node)[0]]
    else:
        children = list(ast.iter_child_nodes(node))
    for child in children:
        yield from _expanded(child, aliases, seen)

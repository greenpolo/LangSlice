"""The settings of a job as the model declares them, walked once.

Every setting is a field of :class:`~langslice.job.settings.spec.JobSpec` or
of one of the models inside it, with its type, default and one-line
description. What a field cannot say, the commands' view of it, rides along
as a :class:`Cli` in the field's ``Annotated`` metadata. :func:`leaves`
walks a model, a task's section and the answers in ``inputs`` included, and
yields each field with its dotted path (``position.thickness_um``,
``inputs.positions``); the commands' flags, the flags' help, the window's
form and ``langslice job FOLDER schema init`` are all read from that walk.

Every spelling follows the path: the flag is the path with ``inputs.``
dropped and ``.`` and ``_`` written ``-`` (``--position-thickness-um``,
``--positions``), a switch that turns an on setting off is ``--no-`` and the
name (``--no-debrief``); the job's own key is ``spec.`` and the path.
"""

from __future__ import annotations

import types
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal, Union, cast, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

FlagGroup = Literal["job", "plugins", "session", "resume", "init", "launch", "wait"]

Kind = Literal["value", "csv", "repeat", "repeat_csv", "pairs", "json", "file"]

ANSWER_PREFIX = "inputs."


@dataclass(frozen=True, kw_only=True)
class Cli:
    """How the commands take one field.

    A field without one has no flag.

    Attributes
    ----------
    group
        Which commands add the flag: ``job`` every command that opens a stack,
        ``plugins`` and ``session`` ``linear run`` alone, ``resume`` the
        commands that continue a folder's job, ``init``, ``launch`` and ``wait``
        the flags of those ``langslice job`` commands.
    flag
        The flag when it is not the one the path gives.
    metavar
        The value's name in ``--help``.
    kind
        How the value is read: ``value`` (the field's own type), ``csv`` (one
        comma-separated text), ``repeat`` (repeatable, each as given),
        ``repeat_csv`` (repeatable, each a comma-separated list), ``pairs``
        (repeatable ``KEY=JSON``), ``json`` (inline JSON or a JSON file),
        ``file`` (the text of a UTF-8 file).
    deferred
        The command works the default out (it depends on the other flags),
        so the flag has none of its own.
    """

    group: FlagGroup = "job"
    flag: str | None = None
    metavar: str | None = None
    kind: Kind = "value"
    deferred: bool = False


@dataclass(frozen=True)
class Holds:
    """Marks a field that holds a dict whose keys another model describes."""

    model: type[BaseModel]


@dataclass(frozen=True)
class Leaf:
    """One setting: its path, its field and the commands' view of it."""

    path: str
    info: FieldInfo
    cli: Cli | None

    @property
    def default(self) -> Any:
        """The field's default."""
        return self.info.get_default(call_default_factory=True)

    @property
    def description(self) -> str:
        """The field's one-line description (the flag's help text)."""
        return " ".join((self.info.description or "").split())

    @property
    def is_switch(self) -> bool:
        """Whether the field is a yes/no setting."""
        return self._members() == (bool,)

    @property
    def flag(self) -> str:
        """The command-line flag the setting is given by."""
        if self.cli is not None and self.cli.flag is not None:
            return self.cli.flag
        name = "--" + self.path.removeprefix(ANSWER_PREFIX).replace(".", "-").replace("_", "-")
        if self.is_switch and self.default is True:
            return "--no-" + name.removeprefix("--")
        return name

    @property
    def dest(self) -> str:
        """The parsed arguments' name for the flag."""
        return self.path.removeprefix(ANSWER_PREFIX).replace(".", "_")

    @property
    def number(self) -> type | None:
        """``int`` or ``float`` for a numeric field, else None."""
        members = self._members()
        if float in members:
            return float
        if members == (int,):
            return int
        return None

    @property
    def choices(self) -> tuple[str, ...] | None:
        """The values of a field limited to a fixed list, else None."""
        for member in _union(self.info.annotation):
            if get_origin(member) is Literal:
                return tuple(str(item) for item in get_args(member))
        return None

    def _members(self) -> tuple[type, ...]:
        return tuple(
            member
            for member in _union(self.info.annotation)
            if isinstance(member, type) and member is not type(None)
        )


def _union(annotation: object) -> tuple[object, ...]:
    """The members of a type, or of a union of types."""
    if get_origin(annotation) in (Union, types.UnionType):
        return get_args(annotation)
    return (annotation,)


def leaves(model: type[BaseModel], prefix: str = "") -> Iterator[Leaf]:
    """Yield every setting of *model*, a task's section and the answers in
    ``inputs`` after the fields that hold them, in declaration order.

    Parameters
    ----------
    model
        The settings model (:class:`~langslice.job.settings.spec.JobSpec`).
    prefix
        The path of the model inside another, with its trailing dot.

    Yields
    ------
    Leaf
        One field with its dotted path and its commands' metadata.
    """
    for name, info in model.model_fields.items():
        holds = next((item for item in info.metadata if isinstance(item, Holds)), None)
        cli = next((item for item in info.metadata if isinstance(item, Cli)), None)
        annotation = cast("object", info.annotation)
        if holds is not None:
            yield from leaves(holds.model, f"{prefix}{name}.")
        elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
            yield from leaves(annotation, f"{prefix}{name}.")
        else:
            yield Leaf(f"{prefix}{name}", info, cli)

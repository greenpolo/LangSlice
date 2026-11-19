"""The base of a stack's stored shapes, which read and write their own JSON.

A shape (:class:`Shape`) is a Pydantic model: its fields are declared once,
in the order they are written, and the model reads a JSON object into
itself and writes itself back, so the state file, the undo history, the
registration listings and every picture's record of the stack hold one
form, and digests of a placement hash it.

A part of a section (:class:`Part`) is frozen. A key it does not declare is
kept (``extra``) and written back after the declared ones, so a file an
earlier LangSlice or a host wrote reads back the same. A value of the wrong
type is a ``ValueError`` naming its key (Pydantic's ``ValidationError``).
Numbers are checked strictly and come back as stored: an int stays an int,
a boolean is not a number, so a value read and written back is the same
JSON. :func:`omitted` declares a field a part leaves out of its JSON while
it is None.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Self

import numpy as np
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    StrictBool,
    StrictFloat,
    StrictInt,
    model_serializer,
)

from langslice.core.jsonable import JsonValue, to_jsonable


def _plain(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


Flag = Annotated[StrictBool, BeforeValidator(_plain)]
Whole = Annotated[StrictInt, BeforeValidator(_plain)]
Number = Annotated[StrictInt | StrictFloat, BeforeValidator(_plain)]
Point = tuple[Number, Number]
Object = Mapping[str, Any]
Written = dict[str, JsonValue]


def _is_none(value: object) -> bool:
    return value is None


def omitted() -> Any:
    """Declare a field that is left out of the JSON while it is None.

    Returns
    -------
    FieldInfo
        The field's default (None) and its rule for writing.
    """
    return Field(default=None, exclude_if=_is_none)


class Shape(BaseModel):
    """Hold a stored shape that reads and writes its JSON form."""

    model_config = ConfigDict(ser_json_inf_nan="constants")

    def to_json(self) -> dict[str, JsonValue]:
        """Return the shape in its JSON form: its fields in declared order.

        Returns
        -------
        dict
            JSON values only, under the keys the shape is stored by.
        """
        return self.model_dump(mode="json", by_alias=True, fallback=to_jsonable)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Self:
        """Return the shape a JSON object holds.

        Parameters
        ----------
        data
            A form :meth:`to_json` writes.

        Returns
        -------
        Shape

        Raises
        ------
        ValueError
            Naming any key that holds a value of the wrong type.
        """
        return cls.model_validate(data)


class Part(Shape):
    """Hold a frozen part of a section, with the keys it does not declare."""

    model_config = ConfigDict(frozen=True, extra="allow")

    def arranged(self, data: dict[str, Any]) -> dict[str, Any]:
        """Return the written fields in the order a variant of the part is stored in.

        Parameters
        ----------
        data
            The declared fields, then the other keys.

        Returns
        -------
        dict
            The same items; a part whose variants store keys in another
            order overrides this.
        """
        return data

    @model_serializer(mode="wrap")
    def written(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Write the fields, then the other keys, in :meth:`arranged` order."""
        return self.arranged(handler(self))

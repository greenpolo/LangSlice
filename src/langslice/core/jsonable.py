"""JSON values: their types, and the one function that makes a value one.

A reply, a picture's recipe, a saved fit's metadata and a plugin's facts
are JSON values by the time they leave the package: mappings with text
keys, lists, numbers, text, booleans and null. :func:`to_jsonable` turns
the values the code holds on the way (tuples, numpy numbers and arrays,
sets) into those.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias, Union, cast, overload

import numpy as np

#: A value as ``json.loads`` gives it back.
JsonValue: TypeAlias = Union[  # noqa: UP007 - a recursive alias needs the string form
    None, bool, int, float, str, list["JsonValue"], dict[str, "JsonValue"]
]
#: A JSON-shaped value as the code holds it (read-only containers).
Json: TypeAlias = Union[  # noqa: UP007 - a recursive alias needs the string form
    None, bool, int, float, str, Sequence["Json"], Mapping[str, "Json"]
]
#: A JSON object as the code holds it.
JsonObject: TypeAlias = Mapping[str, Json]


@overload
def to_jsonable(
    value: Mapping[str, Json], *, strict: bool = False
) -> dict[str, JsonValue]: ...
@overload
def to_jsonable(value: object, *, strict: bool = False) -> JsonValue: ...
def to_jsonable(value: object, *, strict: bool = False) -> JsonValue:
    """Return *value* as JSON values.

    Mappings become dicts with text keys, lists and tuples become lists,
    a numpy number its Python number and a numpy array nested lists.

    Parameters
    ----------
    value
        Anything; containers are converted all the way down.
    strict
        Raise for a value JSON cannot hold instead of writing its text;
        a set is such a value. Without it a set becomes a list sorted by
        each item's text, and anything else its text.

    Returns
    -------
    JsonValue
        A value ``json.dumps`` writes as it is.

    Raises
    ------
    TypeError
        With *strict*, for a value JSON cannot hold.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        number = cast("object", value.item())  # numpy types it loosely
        return to_jsonable(number, strict=strict)
    if isinstance(value, np.ndarray):
        nested = cast("object", value.tolist())  # numpy types it loosely
        return to_jsonable(nested, strict=strict)
    if isinstance(value, Mapping):
        # the keys and items of a mapping of any type, read as objects
        mapping = cast("Mapping[object, object]", value)
        return {
            str(key): to_jsonable(item, strict=strict)
            for key, item in mapping.items()
        }
    if isinstance(value, (list, tuple)):
        items = cast("Sequence[object]", value)  # a sequence of any type
        return [to_jsonable(item, strict=strict) for item in items]
    if strict:
        raise TypeError(
            f"Object of type {type(value).__name__} is not JSON serializable"
        )
    if isinstance(value, (set, frozenset)):
        members = cast("set[object]", value)  # a set of any type
        return [to_jsonable(item) for item in sorted(members, key=str)]
    return str(value)

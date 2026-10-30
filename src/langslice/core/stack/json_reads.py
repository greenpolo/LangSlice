"""Checked reads of JSON values, for the stack's JSON form.

Each reader takes a JSON object and a key and returns the value the object
holds under that key, checked: a missing value, or one of another type, is
a ``ValueError`` naming the key, the type wanted and the value found.
Numbers come back as stored (an int stays an int), so a value read and
written back is the same JSON.

==================  =====================================================
Reader              The value under the key
==================  =====================================================
``text_at``         text
``flag_at``         true or false
``integer_at``      a whole number (a boolean is not one)
``number_at``       a number (a boolean is not one)
``list_at``         a list (text is not one)
``numbers_at``      a list of numbers
``point_at``        two numbers, ``[x, y]``
``texts_at``        a list of texts
``object_at``       an object
``objects_at``      a list of objects
``text_map_at``     an object of texts
==================  =====================================================

``optional_at`` reads with any of them and gives None for an absent key;
``as_number`` and ``as_object`` check a value already read. ``other_keys``
and ``set_items`` serve the writing side: the keys a shape does not
declare, kept to be written back, and the items whose value is set.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TypeVar

from langslice.core.jsonable import Json, JsonObject, JsonValue, to_jsonable

_T = TypeVar("_T")


def _wrong_type(key: str, expected: str, value: Json) -> ValueError:
    return ValueError(f"{key} must be {expected}, not {value!r}")


def text_at(data: JsonObject, key: str) -> str:
    """Return the text an object holds under a key.

    Parameters
    ----------
    data, key
        The JSON object and the key read from it.

    Returns
    -------
    str
    """
    value = data.get(key)
    if not isinstance(value, str):
        raise _wrong_type(key, "text", value)
    return value


def optional_at(
    data: JsonObject, key: str, read: Callable[[JsonObject, str], _T]
) -> _T | None:
    """Return a value read with a reader, or None when the key is absent.

    Parameters
    ----------
    data, key
        The JSON object and the key read from it, which may be absent.
    read
        One of this module's readers, such as :func:`text_at`.

    Returns
    -------
    object or None
    """
    if key not in data:
        return None
    return read(data, key)


def flag_at(data: JsonObject, key: str) -> bool:
    """Return the boolean an object holds under a key.

    Parameters
    ----------
    data, key
        The JSON object and the key read from it.

    Returns
    -------
    bool
    """
    value = data.get(key)
    if not isinstance(value, bool):
        raise _wrong_type(key, "true or false", value)
    return value


def integer_at(data: JsonObject, key: str) -> int:
    """Return the whole number an object holds under a key.

    Parameters
    ----------
    data, key
        The JSON object and the key read from it.

    Returns
    -------
    int
    """
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _wrong_type(key, "a whole number", value)
    return value


def as_number(key: str, value: Json) -> float:
    """Return a value already read that must be a number, as stored.

    Parameters
    ----------
    key
        The key the value was read under, named in the error.
    value
        The value read, of any JSON type.

    Returns
    -------
    float
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _wrong_type(key, "a number", value)
    return value


def number_at(data: JsonObject, key: str) -> float:
    """Return the number an object holds under a key, as stored.

    Parameters
    ----------
    data, key
        The JSON object and the key read from it.

    Returns
    -------
    float
    """
    return as_number(key, data.get(key))


def list_at(data: JsonObject, key: str) -> Sequence[Json]:
    """Return the list an object holds under a key.

    Parameters
    ----------
    data, key
        The JSON object and the key read from it.

    Returns
    -------
    sequence
    """
    value = data.get(key)
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise _wrong_type(key, "a list", value)
    return value


def numbers_at(data: JsonObject, key: str) -> tuple[float, ...]:
    """Return the list of numbers an object holds under a key, as stored.

    Parameters
    ----------
    data, key
        The JSON object and the key read from it.

    Returns
    -------
    tuple of float
    """
    return tuple(as_number(key, item) for item in list_at(data, key))


def point_at(data: JsonObject, key: str) -> tuple[float, float]:
    """Return the ``[x, y]`` an object holds under a key, as stored.

    Parameters
    ----------
    data, key
        The JSON object and the key read from it.

    Returns
    -------
    tuple of float
    """
    values = numbers_at(data, key)
    if len(values) != 2:
        raise _wrong_type(key, "two numbers", data.get(key))
    return values[0], values[1]


def texts_at(data: JsonObject, key: str) -> tuple[str, ...]:
    """Return the list of texts an object holds under a key.

    Parameters
    ----------
    data, key
        The JSON object and the key read from it.

    Returns
    -------
    tuple of str
    """
    texts: list[str] = []
    for item in list_at(data, key):
        if not isinstance(item, str):
            raise _wrong_type(key, "a list of texts", data.get(key))
        texts.append(item)
    return tuple(texts)


def as_object(key: str, value: Json) -> JsonObject:
    """Return a value already read that must be a JSON object.

    Parameters
    ----------
    key
        The key the value was read under, named in the error.
    value
        The value read, of any JSON type.

    Returns
    -------
    JsonObject
    """
    if not isinstance(value, Mapping):
        raise _wrong_type(key, "an object", value)
    return value


def object_at(data: JsonObject, key: str) -> JsonObject:
    """Return the JSON object an object holds under a key.

    Parameters
    ----------
    data, key
        The JSON object and the key read from it.

    Returns
    -------
    JsonObject
    """
    return as_object(key, data.get(key))


def objects_at(data: JsonObject, key: str) -> tuple[JsonObject, ...]:
    """Return the list of JSON objects an object holds under a key.

    Parameters
    ----------
    data, key
        The JSON object and the key read from it.

    Returns
    -------
    tuple of JsonObject
    """
    return tuple(as_object(key, item) for item in list_at(data, key))


def text_map_at(data: JsonObject, key: str) -> dict[str, str]:
    """Return the object of texts an object holds under a key.

    Parameters
    ----------
    data, key
        The JSON object and the key read from it.

    Returns
    -------
    dict
        Each name and its text.
    """
    texts: dict[str, str] = {}
    for name, value in object_at(data, key).items():
        if not isinstance(value, str):
            raise _wrong_type(key, "an object of texts", value)
        texts[name] = value
    return texts


def other_keys(data: JsonObject, known: Sequence[str]) -> JsonObject:
    """Return the keys of an object a shape does not declare.

    Parameters
    ----------
    data
        The JSON object read.
    known
        The keys the shape declares.

    Returns
    -------
    JsonObject
        The other keys and their values, as they came.
    """
    return {key: value for key, value in data.items() if key not in known}


def set_items(items: Iterable[tuple[str, Json]]) -> dict[str, JsonValue]:
    """Return a JSON object of the items whose value is set.

    Parameters
    ----------
    items
        ``(key, value)`` pairs, in the order they are written.

    Returns
    -------
    dict
        Each key whose value is not None, with its value as JSON.
    """
    return {
        key: to_jsonable(value) for key, value in items if value is not None
    }

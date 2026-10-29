"""Checked reads of JSON values, for the stack's JSON form.

Each function reads one value out of a JSON object and checks its type;
a value of the wrong type is a ``ValueError`` naming its key. Numbers
are returned as stored (an int stays an int), so a value read and
written back is the same JSON.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TypeVar

from langslice.core.jsonable import Json, JsonObject, JsonValue, to_jsonable

_T = TypeVar("_T")


def wrong_type(key: str, expected: str, value: Json) -> ValueError:
    """Return the error for a key that holds a value of the wrong type.

    Parameters
    ----------
    key
        The key that was read.
    expected
        What the key should hold, in words ("text", "a number").
    value
        What it holds.

    Returns
    -------
    ValueError
        The error, naming the key, the type and the value.
    """
    return ValueError(f"{key} must be {expected}, not {value!r}")


def text_at(data: JsonObject, key: str) -> str:
    """Return the text an object holds under a key.

    Parameters
    ----------
    data
        The JSON object read.
    key
        The key whose value is text.

    Returns
    -------
    str
        The text.

    Raises
    ------
    ValueError
        When the value is missing or not text.
    """
    value = data.get(key)
    if not isinstance(value, str):
        raise wrong_type(key, "text", value)
    return value


def optional_at(
    data: JsonObject, key: str, read: Callable[[JsonObject, str], _T]
) -> _T | None:
    """Return a value read with a reader, or None when the key is absent.

    Parameters
    ----------
    data
        The JSON object read.
    key
        The key, which may be absent.
    read
        One of this module's readers, such as :func:`text_at`.

    Returns
    -------
    object or None
        What *read* returns, or None without the key.
    """
    if key not in data:
        return None
    return read(data, key)


def flag_at(data: JsonObject, key: str) -> bool:
    """Return the boolean an object holds under a key.

    Parameters
    ----------
    data
        The JSON object read.
    key
        The key whose value is true or false.

    Returns
    -------
    bool
        The value.

    Raises
    ------
    ValueError
        When the value is missing or not a boolean.
    """
    value = data.get(key)
    if not isinstance(value, bool):
        raise wrong_type(key, "true or false", value)
    return value


def integer_at(data: JsonObject, key: str) -> int:
    """Return the whole number an object holds under a key.

    Parameters
    ----------
    data
        The JSON object read.
    key
        The key whose value is a whole number.

    Returns
    -------
    int
        The number.

    Raises
    ------
    ValueError
        When the value is missing or not a whole number.
    """
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise wrong_type(key, "a whole number", value)
    return value


def as_number(key: str, value: Json) -> float:
    """Return a value that must be a number, as stored.

    Parameters
    ----------
    key
        The key the value was read under, for the error.
    value
        The value read, of any JSON type.

    Returns
    -------
    float
        The number; an int stays an int.

    Raises
    ------
    ValueError
        When the value is not a number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise wrong_type(key, "a number", value)
    return value


def number_at(data: JsonObject, key: str) -> float:
    """Return the number an object holds under a key, as stored.

    Parameters
    ----------
    data
        The JSON object read.
    key
        The key whose value is a number.

    Returns
    -------
    float
        The number; an int stays an int.
    """
    return as_number(key, data.get(key))


def list_at(data: JsonObject, key: str) -> Sequence[Json]:
    """Return the list an object holds under a key.

    Parameters
    ----------
    data
        The JSON object read.
    key
        The key whose value is a list.

    Returns
    -------
    sequence
        The list's items.

    Raises
    ------
    ValueError
        When the value is missing or not a list.
    """
    value = data.get(key)
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise wrong_type(key, "a list", value)
    return value


def numbers_at(data: JsonObject, key: str) -> tuple[float, ...]:
    """Return the list of numbers an object holds under a key.

    Parameters
    ----------
    data
        The JSON object read.
    key
        The key whose value is a list of numbers.

    Returns
    -------
    tuple of float
        The numbers, as stored.
    """
    return tuple(as_number(key, item) for item in list_at(data, key))


def point_at(data: JsonObject, key: str) -> tuple[float, float]:
    """Return the ``[x, y]`` an object holds under a key.

    Parameters
    ----------
    data
        The JSON object read.
    key
        The key whose value is two numbers.

    Returns
    -------
    tuple of float
        ``(x, y)``, as stored.

    Raises
    ------
    ValueError
        When the value is not two numbers.
    """
    values = numbers_at(data, key)
    if len(values) != 2:
        raise wrong_type(key, "two numbers", data.get(key))
    return values[0], values[1]


def texts_at(data: JsonObject, key: str) -> tuple[str, ...]:
    """Return the list of texts an object holds under a key.

    Parameters
    ----------
    data
        The JSON object read.
    key
        The key whose value is a list of texts.

    Returns
    -------
    tuple of str
        The texts.

    Raises
    ------
    ValueError
        When the value is not a list of texts.
    """
    texts: list[str] = []
    for item in list_at(data, key):
        if not isinstance(item, str):
            raise wrong_type(key, "a list of texts", data.get(key))
        texts.append(item)
    return tuple(texts)


def as_object(key: str, value: Json) -> JsonObject:
    """Return a value that must be a JSON object.

    Parameters
    ----------
    key
        The key the value was read under, for the error.
    value
        The value read, of any JSON type.

    Returns
    -------
    JsonObject
        The object.

    Raises
    ------
    ValueError
        When the value is not an object.
    """
    if not isinstance(value, Mapping):
        raise wrong_type(key, "an object", value)
    return value


def object_at(data: JsonObject, key: str) -> JsonObject:
    """Return the JSON object an object holds under a key.

    Parameters
    ----------
    data
        The JSON object read.
    key
        The key whose value is an object.

    Returns
    -------
    JsonObject
        The object.
    """
    return as_object(key, data.get(key))


def objects_at(data: JsonObject, key: str) -> tuple[JsonObject, ...]:
    """Return the list of JSON objects an object holds under a key.

    Parameters
    ----------
    data
        The JSON object read.
    key
        The key whose value is a list of objects.

    Returns
    -------
    tuple of JsonObject
        The objects.
    """
    return tuple(as_object(key, item) for item in list_at(data, key))


def text_map_at(data: JsonObject, key: str) -> dict[str, str]:
    """Return the object of texts an object holds under a key.

    Parameters
    ----------
    data
        The JSON object read.
    key
        The key whose value maps names to texts.

    Returns
    -------
    dict
        Each name and its text.

    Raises
    ------
    ValueError
        When a value is not text.
    """
    texts: dict[str, str] = {}
    for name, value in object_at(data, key).items():
        if not isinstance(value, str):
            raise wrong_type(key, "an object of texts", value)
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

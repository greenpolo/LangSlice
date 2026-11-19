"""Checked reads of JSON values from files that are not the stack's own.

Each reader takes a JSON object and a key and returns the value there,
checked: a missing value, or one of another type, is a ``ValueError``
naming the key. A deformation record's metadata, a picture's frame and a
job's saved settings are read with them; the stack's shapes read themselves
(:mod:`langslice.core.stack.part`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar

from langslice.core.jsonable import Json, JsonObject

_T = TypeVar("_T")


def _wrong_type(key: str, expected: str, value: Json) -> ValueError:
    return ValueError(f"{key} must be {expected}, not {value!r}")


def text_at(data: JsonObject, key: str) -> str:
    """Return the text under *key*."""
    value = data.get(key)
    if not isinstance(value, str):
        raise _wrong_type(key, "text", value)
    return value


def optional_at(data: JsonObject, key: str, read: Callable[[JsonObject, str], _T]) -> _T | None:
    """Return the value *read* makes of *key*, or None when the key is absent."""
    if key not in data:
        return None
    return read(data, key)


def flag_at(data: JsonObject, key: str) -> bool:
    """Return the boolean under *key*."""
    value = data.get(key)
    if not isinstance(value, bool):
        raise _wrong_type(key, "true or false", value)
    return value


def integer_at(data: JsonObject, key: str) -> int:
    """Return the whole number under *key* (a boolean is not one)."""
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _wrong_type(key, "a whole number", value)
    return value


def as_number(key: str, value: Json) -> float:
    """Return *value*, read under *key*, as a number as stored."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _wrong_type(key, "a number", value)
    return value


def number_at(data: JsonObject, key: str) -> float:
    """Return the number under *key*, as stored."""
    return as_number(key, data.get(key))


def list_at(data: JsonObject, key: str) -> Sequence[Json]:
    """Return the list under *key* (text is not one)."""
    value = data.get(key)
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise _wrong_type(key, "a list", value)
    return value


def numbers_at(data: JsonObject, key: str) -> tuple[float, ...]:
    """Return the list of numbers under *key*, as stored."""
    return tuple(as_number(key, item) for item in list_at(data, key))


def texts_at(data: JsonObject, key: str) -> tuple[str, ...]:
    """Return the list of texts under *key*."""
    items = list_at(data, key)
    if not all(isinstance(item, str) for item in items):
        raise _wrong_type(key, "a list of texts", data.get(key))
    return tuple(str(item) for item in items)


def as_object(key: str, value: Json) -> JsonObject:
    """Return *value*, read under *key*, as a JSON object."""
    if not isinstance(value, Mapping):
        raise _wrong_type(key, "an object", value)
    return value


def object_at(data: JsonObject, key: str) -> JsonObject:
    """Return the JSON object under *key*."""
    return as_object(key, data.get(key))


def text_map_at(data: JsonObject, key: str) -> dict[str, str]:
    """Return the object of texts under *key*."""
    texts: dict[str, str] = {}
    for name, value in object_at(data, key).items():
        if not isinstance(value, str):
            raise _wrong_type(key, "an object of texts", value)
        texts[name] = value
    return texts

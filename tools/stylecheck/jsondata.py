"""Read JSON and TOML text into values whose shape is checked before use.

``json.loads`` and ``tomllib.loads`` return values of no declared type.
These functions give them the type ``object`` and narrow them one level at
a time, so a reader of the baseline file, of ruff's report or of
``pyproject.toml`` checks every field it uses.
"""

from __future__ import annotations

import json
import tomllib
from typing import cast


def load_json(text: str) -> object:
    """Parse JSON text as a value of unknown shape.

    Parameters
    ----------
    text
        The JSON document to parse.

    Returns
    -------
    object
        The parsed value, to be narrowed with :func:`as_list` and
        :func:`as_table`.

    Raises
    ------
    json.JSONDecodeError
        When the text is not JSON.
    """
    # cast: the value is narrowed by its readers before any field is used
    return cast(object, json.loads(text))


def load_toml(text: str) -> dict[str, object]:
    """Parse TOML text as a table of values of unknown shape.

    Parameters
    ----------
    text
        The TOML document to parse, such as ``pyproject.toml``.

    Returns
    -------
    dict
        The document's top-level fields by name, each to be narrowed with
        :func:`as_list`, :func:`as_table` and :func:`as_text`.

    Raises
    ------
    tomllib.TOMLDecodeError
        When the text is not TOML.
    """
    # cast: a TOML document is a table whose values are narrowed by readers
    return cast(dict[str, object], tomllib.loads(text))


def as_list(value: object) -> list[object] | None:
    """Narrow a parsed value to a list.

    Parameters
    ----------
    value
        A value returned by :func:`load_json` or :func:`load_toml`, or an
        item inside one.

    Returns
    -------
    list of object or None
        The list, or None when the value is not one.
    """
    if not isinstance(value, list):
        return None
    # cast: a JSON list's items are themselves values of unknown shape
    return cast(list[object], value)


def as_table(value: object) -> dict[str, object] | None:
    """Narrow a parsed value to an object with text keys.

    Parameters
    ----------
    value
        A value returned by :func:`load_json` or :func:`load_toml`, or an
        item inside one.

    Returns
    -------
    dict or None
        The object's fields by name, or None when the value is not one.
    """
    if not isinstance(value, dict):
        return None
    # cast: a JSON object's keys are text and its values of unknown shape
    return cast(dict[str, object], value)


def as_text(value: object) -> str | None:
    """Narrow a parsed value to a string.

    Parameters
    ----------
    value
        A value returned by :func:`load_json` or :func:`load_toml`, or an
        item inside one.

    Returns
    -------
    str or None
        The string, or None when the value is not one.
    """
    if not isinstance(value, str):
        return None
    return value


def as_int(value: object) -> int | None:
    """Narrow a parsed value to a whole number.

    Parameters
    ----------
    value
        A value returned by :func:`load_json` or :func:`load_toml`, or an
        item inside one.

    Returns
    -------
    int or None
        The number, or None when the value is not a whole number (a
        boolean is not one).
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def canonical(value: object) -> object:
    """Return a value as JSON reads it back: lists for tuples, sorted keys.

    Parameters
    ----------
    value
        Text, numbers, booleans, None, and lists, tuples and dicts of them.

    Returns
    -------
    object
        The same data in the form a JSON round trip gives, so that a value
        read from TOML, from a dataclass or from a JSON file compares equal
        when it holds the same data.
    """
    return load_json(json.dumps(value, sort_keys=True, ensure_ascii=False))

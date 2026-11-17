"""Names taken from libraries that publish no types.

A library without types tells the type checker nothing about its
functions. The one function here fetches such a name as a plain object;
the module that uses it states the name's shape once, with ``cast``, and
everything after that is checked.
"""

from __future__ import annotations

import importlib


def untyped(module: str, name: str) -> object:
    """Return a module's attribute as an object of no stated type.

    Parameters
    ----------
    module
        The module's import name, such as ``scipy.ndimage``.
    name
        The attribute, such as ``map_coordinates``.

    Returns
    -------
    object
        The attribute; the caller casts it to the shape it has.
    """
    members: dict[str, object] = vars(importlib.import_module(module))
    return members[name]

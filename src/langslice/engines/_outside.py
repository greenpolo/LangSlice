"""Names an engine takes from libraries that publish no types.

The core's ``outside`` module, kept here because an engine imports
only the plugin API. The caller states the shape of the name it fetched
once, with ``cast``, and everything after that is checked.
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

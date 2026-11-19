"""Every LangSlice module imports.

A module that names something deleted fails here, whether or not another
test reaches it. A module whose own optional third-party library is not
installed is passed over, and so is a ``__main__`` (importing it runs it).
A vendored module (``langslice._vendor``) is passed over when anything it
imports is missing, its optional compiled engine included.
"""

from __future__ import annotations

import importlib
import pkgutil

import langslice

VENDORED = "langslice._vendor"


def test_every_module_imports() -> None:
    broken: list[str] = []
    for found in pkgutil.walk_packages(langslice.__path__, "langslice."):
        if found.name.endswith(".__main__"):
            continue
        try:
            importlib.import_module(found.name)
        except ImportError as error:
            vendored = found.name.startswith(VENDORED)
            if not vendored and (error.name or "").startswith("langslice"):
                broken.append(f"{found.name}: {error}")
    assert not broken, "\n".join(broken)

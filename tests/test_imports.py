"""Every LangSlice module imports.

A module that names something deleted fails here, whether or not another
test reaches it. A module whose own optional third-party library is not
installed is passed over, and so is a ``__main__`` (importing it runs it).
"""

from __future__ import annotations

import importlib
import pkgutil

import langslice


def test_every_module_imports() -> None:
    broken: list[str] = []
    for found in pkgutil.walk_packages(langslice.__path__, "langslice."):
        if found.name.endswith(".__main__"):
            continue
        try:
            importlib.import_module(found.name)
        except ImportError as error:
            if (error.name or "").startswith("langslice"):
                broken.append(f"{found.name}: {error}")
    assert not broken, "\n".join(broken)

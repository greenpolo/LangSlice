"""Tests of the checks against metaprogramming and module state."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from tools.stylecheck.core import Config
from tools.stylecheck.testing import flagged_lines

CONFIG = Config.load(Path(__file__).parents[2] / "pyproject.toml")


def test_global_statement_is_flagged() -> None:
    source = """
    _CACHE = None

    def load():
        global _CACHE
        _CACHE = 1
    """
    assert flagged_lines(source, "global-statement") == [5]


def test_globals_and_setattr_on_modules_are_flagged() -> None:
    source = """
    import sys
    import os.path as osp
    from langslice.core import space

    def patch(record):
        globals()["name"] = 1
        setattr(sys, "stdout", None)
        setattr(osp, "x", 1)
        setattr(space, "x", 1)
        setattr(sys.modules[__name__], "x", 1)
        setattr(record, "x", 1)
    """
    others = {"src/langslice/core/space.py": ""}
    lines = flagged_lines(source, "globals-or-module-setattr", others=others)
    assert lines == [7, 8, 9, 10, 11]


def test_getattr_forwarding_is_flagged() -> None:
    source = """
    class Proxy:
        def __getattr__(self, name):
            return getattr(self.inner, name)

    def __getattr__(name):
        return name
    """
    assert flagged_lines(source, "getattr-forwarding") == [3, 6]
    relpath = "src/langslice/core/__init__.py"
    assert flagged_lines(source, "getattr-forwarding", relpath=relpath) == [3]


def test_classes_built_at_run_time_are_flagged() -> None:
    source = """
    import dataclasses

    A = type("A", (), {})
    B = dataclasses.make_dataclass("B", ["x"])
    c = type(value)
    """
    assert flagged_lines(source, "runtime-type") == [4, 5]


def test_metaclasses_other_than_abcmeta_are_flagged() -> None:
    source = """
    import abc

    class A(metaclass=abc.ABCMeta):
        pass

    class B(abc.ABC):
        pass

    class C(metaclass=Registry):
        pass
    """
    assert flagged_lines(source, "metaclass") == [10]


def test_private_getattr_on_another_object_is_flagged() -> None:
    source = """
    def reach(model, self):
        a = getattr(model, "_additional_args", None)
        b = hasattr(threading, "_register_atexit")
        c = getattr(self, "_cache", None)
        d = getattr(model, "__name__")
        e = getattr(model, "name")
    """
    assert flagged_lines(source, "private-getattr") == [3, 4]
    config = dataclasses.replace(CONFIG, infra_package="langslice.infra")
    relpath = "src/langslice/infra/warm.py"
    lines = flagged_lines(
        source, "private-getattr", relpath=relpath, config=config
    )
    assert lines == []

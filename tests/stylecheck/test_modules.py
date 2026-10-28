"""Tests of the module and class checks: size, names, ``__init__``."""

from __future__ import annotations

import pytest

from tools.stylecheck.testing import check_source


def test_module_size_flags_a_long_module() -> None:
    source = "".join(f"x{index} = {index}\n" for index in range(801))
    violations = check_source(source, "module-size")
    assert len(violations) == 1
    assert "801 lines" in violations[0].message


def test_module_size_accepts_the_limit() -> None:
    source = "".join(f"x{index} = {index}\n" for index in range(800))
    assert check_source(source, "module-size") == []


@pytest.mark.parametrize(
    "name",
    [
        "utils",
        "helpers",
        "misc",
        "common",
        "image_utils",
        "_helpers",
        "canvas_part2",
        "canvas_more",
        "canvas_extra",
        "canvas2",
    ],
)
def test_module_name_flags_a_topicless_name(name: str) -> None:
    # the game: split an 800-line module into x_part2.py or x_more.py
    violations = check_source(
        "X = 1\n", "module-name", relpath=f"src/langslice/core/{name}.py"
    )
    assert len(violations) == 1


@pytest.mark.parametrize(
    "name", ["atlas", "landmarks", "masks", "ome_warp", "thin_plate"]
)
def test_module_name_accepts_a_topic(name: str) -> None:
    violations = check_source(
        "X = 1\n", "module-name", relpath=f"src/langslice/core/{name}.py"
    )
    assert violations == []


def test_module_name_judges_a_package_by_its_init() -> None:
    violations = check_source(
        '"""Helpers."""\n',
        "module-name",
        relpath="src/langslice/core/helpers/__init__.py",
    )
    assert len(violations) == 1


def test_init_logic_accepts_imports_and_version() -> None:
    source = '''
    """The package."""

    from __future__ import annotations

    from langslice.core.atlas import Atlas

    try:
        from langslice._version import __version__
    except ImportError:
        __version__ = "unknown"

    __all__ = ["Atlas"]
    '''
    violations = check_source(
        source, "init-logic", relpath="src/langslice/core/__init__.py"
    )
    assert violations == []


def test_init_logic_flags_code_moved_into_init() -> None:
    # the game: move the overflow of a long module into __init__.py
    source = '''
    """The package."""

    import logging

    LIMIT = 5

    try:
        import torch
    except ImportError:
        torch = None

    def helper():
        return LIMIT

    logging.getLogger("x").addFilter(None)
    '''
    violations = check_source(
        source, "init-logic", relpath="src/langslice/core/__init__.py"
    )
    assert len(violations) == 4


def test_init_logic_ignores_plain_modules() -> None:
    assert check_source("def f():\n    pass\n", "init-logic") == []


def test_class_size_flags_many_methods() -> None:
    methods = "".join(
        f"    def m{index}(self):\n        pass\n" for index in range(31)
    )
    violations = check_source(f"class Big:\n{methods}", "class-size")
    assert len(violations) == 1
    assert "31 methods" in violations[0].message


def test_class_size_flags_many_lines() -> None:
    fields = "".join(f"    x{index}: int = {index}\n" for index in range(600))
    violations = check_source(f"class Long:\n{fields}", "class-size")
    assert len(violations) == 1
    assert "601 lines" in violations[0].message

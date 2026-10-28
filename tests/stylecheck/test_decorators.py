"""Tests of the decorator checks: count, composition and location."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from tools.stylecheck.core import Config
from tools.stylecheck.testing import check_source

ROOT = Path(__file__).resolve().parents[2]

WRAPPERS = """
import functools


def serialized(tool, lock):
    @functools.wraps(tool)
    def run(*args, **kwargs):
        with lock:
            return tool(*args, **kwargs)
    return run


def saves_views(tool, job):
    def run(*args, **kwargs):
        result = tool(*args, **kwargs)
        job.save()
        return result
    return run


def announces(tool, job):
    @functools.wraps(tool)
    def run(*args, **kwargs):
        job.announce()
        return tool(*args, **kwargs)
    return run
"""


def _config(decorators_module: str) -> Config:
    """Load the settings with a decorators module set."""
    config = Config.load(ROOT / "pyproject.toml")
    return dataclasses.replace(config, decorators_module=decorators_module)


def test_decorator_count_flags_three() -> None:
    source = """
    import functools

    @functools.cache
    @staticmethod
    @other
    def busy():
        pass

    @dataclass(frozen=True)
    @other
    class Fine:
        pass
    """
    violations = check_source(source, "decorator-count")
    assert [violation.symbol for violation in violations] == ["busy"]


def test_decorator_composition_flags_wrappers_with_arguments() -> None:
    source = """
    from langslice.doors.wrappers import announces, saves_views, serialized

    def build_tools(bodies, job, lock):
        return [
            serialized(saves_views(announces(body, job), job), lock)
            for body in bodies
        ]
    """
    violations = check_source(
        source,
        "decorator-composition",
        relpath="src/langslice/doors/toolbox.py",
        others={"src/langslice/doors/wrappers.py": WRAPPERS},
    )
    assert len(violations) == 1
    assert "3 nested calls" in violations[0].message


def test_decorator_composition_flags_one_argument_wrappers() -> None:
    source = """
    def tool():
        pass

    WRAPPED = first(second(third(tool)))
    """
    assert len(check_source(source, "decorator-composition")) == 1


def test_decorator_composition_accepts_a_loop_and_data_calls() -> None:
    # the intended fix: one explicit list of steps applied in a loop
    source = """
    from langslice.doors.wrappers import announces, saves_views

    STEPS = (announces, saves_views)

    def build(body, job):
        for step in STEPS:
            body = step(body, job)
        return body

    def pixels(values):
        return np.asarray(np.stack(list(values)))

    def pooled(band, bands):
        return np.concatenate(list(pool.map(band, bands)))
    """
    violations = check_source(
        source,
        "decorator-composition",
        relpath="src/langslice/doors/toolbox.py",
        others={"src/langslice/doors/wrappers.py": WRAPPERS},
    )
    assert violations == []


def test_decorator_location_flags_a_decorator_elsewhere() -> None:
    violations = check_source(
        WRAPPERS,
        "decorator-location",
        relpath="src/langslice/doors/wrappers.py",
        config=_config("langslice.core.decorators"),
    )
    assert [violation.symbol for violation in violations] == [
        "serialized",
        "saves_views",
        "announces",
    ]


def test_decorator_location_accepts_the_decorators_module() -> None:
    violations = check_source(
        WRAPPERS,
        "decorator-location",
        relpath="src/langslice/core/decorators.py",
        config=_config("langslice.core.decorators"),
    )
    assert violations == []


def test_decorator_location_reports_a_factory_once() -> None:
    source = """
    import functools

    def with_config(name):
        def decorate(function):
            @functools.wraps(function)
            def run(*args, **kwargs):
                return function(*args, **kwargs)
            return run
        return decorate
    """
    violations = check_source(
        source,
        "decorator-location",
        config=_config("langslice.core.decorators"),
    )
    assert [violation.symbol for violation in violations] == ["with_config"]


def test_decorator_location_is_off_without_a_module() -> None:
    assert (
        check_source(WRAPPERS, "decorator-location", config=_config("")) == []
    )


def test_nested_definition_allows_wrappers_in_the_decorators_module() -> None:
    violations = check_source(
        WRAPPERS,
        "nested-definition",
        relpath="src/langslice/core/decorators.py",
        config=_config("langslice.core.decorators"),
    )
    assert violations == []

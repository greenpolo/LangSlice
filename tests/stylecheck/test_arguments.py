"""Tests of the argument checks: keyword-only options, variadics, outputs."""

from __future__ import annotations

from tools.stylecheck.testing import check_source


def test_keyword_only_options_flags_a_positional_default() -> None:
    source = """
    def fit_affine(section, plane, metric="mi"):
        pass
    """
    violations = check_source(source, "keyword-only-options")
    assert len(violations) == 1
    assert "'metric'" in violations[0].message


def test_keyword_only_options_accepts_options_after_star() -> None:
    # the game: make everything keyword-only, the data too; allowed,
    # as is a required positional first parameter
    source = """
    def fit_affine(section, plane, *, metric="mi"):
        pass

    def everything_keyword(*, section, plane, metric="mi"):
        pass
    """
    assert check_source(source, "keyword-only-options") == []


def test_keyword_only_options_exempts_fixed_signatures() -> None:
    source = """
    import json
    from typing import override

    class Encoder(json.JSONEncoder):
        def default(self, o, extra=None):
            return o

    class Point:
        def __init__(self, x, y=0.0):
            self.x = x

        @override
        def describe(self, verbose=False):
            return ""
    """
    assert check_source(source, "keyword-only-options") == []


def test_keyword_only_options_exempts_only_what_an_external_base_defines() -> (
    None
):
    """The gaming snippet of the review: any external base exempted all.

    A method of a class derived from a base outside the package is
    exempt only when that base defines the method's name.
    """
    source = """
    import ast
    import threading
    from types import SimpleNamespace

    class Options(SimpleNamespace):
        def fit(self, section, metric="mi"):
            pass

    class Worker(threading.Thread):
        def run(self, steps=1):
            pass

        def report(self, steps=1):
            pass

    class Renamer(ast.NodeTransformer):
        def visit_Name(self, node, depth=0):
            return node

        def rename(self, node, depth=0):
            return node
    """
    violations = check_source(source, "keyword-only-options")
    assert [violation.symbol for violation in violations] == [
        "Options.fit",
        "Worker.report",
        "Renamer.rename",
    ]


def test_keyword_only_options_reads_overrides_of_package_bases() -> None:
    source = """
    class Base:
        def render(self, picture, scale=1.0):
            pass

    class Child(Base):
        def render(self, picture, scale=1.0):
            pass

        def extra(self, picture, scale=1.0):
            pass
    """
    violations = check_source(source, "keyword-only-options")
    assert [violation.symbol for violation in violations] == [
        "Base.render",
        "Child.extra",
    ]


def test_kwargs_forwarding_accepts_one_whole_forward() -> None:
    source = """
    def plot(values, **kwargs):
        return axes.plot(values, **kwargs)

    def add(*layers):
        for layer in layers:
            stack.append(layer)
        return combine(*layers), len(layers)
    """
    assert check_source(source, "kwargs-forwarding") == []


def test_kwargs_forwarding_flags_kwargs_read_in_the_body() -> None:
    # the game: take options through **kwargs and read them
    source = """
    def fit(section, **kwargs):
        metric = kwargs.pop("metric", "mi")
        return run(section, metric, **kwargs)

    def twice(section, **kwargs):
        first(section, **kwargs)
        second(section, **kwargs)

    def record(kind, **fields):
        return {"kind": kind, **fields}
    """
    violations = check_source(source, "kwargs-forwarding")
    assert [violation.symbol for violation in violations] == [
        "fit",
        "twice",
        "record",
    ]


def test_kwargs_forwarding_flags_indexed_args() -> None:
    # the game: hidden positional parameters behind *args
    source = """
    def place(*args):
        section, plane = args
        return section, plane

    def first(*args):
        return args[0]
    """
    assert len(check_source(source, "kwargs-forwarding")) == 2


def test_kwargs_forwarding_exempts_wrappers_and_ignored_arguments() -> None:
    source = """
    import functools

    def logged(tool):
        @functools.wraps(tool)
        def run(*args, **kwargs):
            log(args, kwargs)
            return tool(*args, **kwargs)
        return run

    class Server:
        def stop(self, *_):
            self.stopping = True

        def __exit__(self, *exc_info):
            self.close()
    """
    assert check_source(source, "kwargs-forwarding") == []


def test_kwargs_forwarding_reads_only_the_methods_a_base_defines() -> None:
    """An unused ``*args`` is accepted only where the base fixes it."""
    source = """
    import logging

    class Quiet(logging.Filter):
        def filter(self, record, *args):
            return True

        def other(self, record, *args):
            return True
    """
    violations = check_source(source, "kwargs-forwarding")
    assert [violation.symbol for violation in violations] == ["Quiet.other"]


def test_kwargs_forwarding_flags_an_unused_kwargs() -> None:
    source = """
    def fit(section, **kwargs):
        return section
    """
    assert len(check_source(source, "kwargs-forwarding")) == 1


def test_output_parameter_flags_a_list_appended_to() -> None:
    source = """
    def collect(node, found: list[str]) -> None:
        found.append(node.name)

    def fill(table: dict[str, int] | None, key: str) -> None:
        table[key] = 1

    def unannotated(node, names):
        names.extend(node.names)
    """
    violations = check_source(source, "output-parameter")
    assert [violation.symbol for violation in violations] == [
        "collect",
        "fill",
        "unannotated",
    ]


def test_output_parameter_names_the_owner_of_a_closure_write() -> None:
    source = """
    def handler_for(result: dict[str, str]):
        def handle(event):
            result.update(event)
        return handle
    """
    violations = check_source(source, "output-parameter")
    assert [violation.symbol for violation in violations] == ["handler_for"]


def test_output_parameter_allows_the_accepted_forms() -> None:
    source = """
    import hashlib
    import numpy as np

    def digest(hasher: "hashlib._Hash", data: bytes) -> None:
        hasher.update(data)

    def scale(image: np.ndarray, out: np.ndarray) -> None:
        np.multiply(image, 2, out=out)
        out[0] = 1

    def copied(rows: list[str]) -> list[str]:
        rows = list(rows)
        rows.append("end")
        return rows

    class Store:
        def add(self, row: str) -> None:
            self.rows.append(row)
    """
    assert check_source(source, "output-parameter") == []

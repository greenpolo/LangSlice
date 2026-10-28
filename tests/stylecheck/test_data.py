"""Tests of the data and type checks."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from tools.stylecheck.core import Config, Severity
from tools.stylecheck.testing import check_source, flagged_lines

CONFIG = Config.load(Path(__file__).parents[2] / "pyproject.toml")
BOUNDARY = dataclasses.replace(
    CONFIG, json_boundary_modules=("langslice.job.formats",)
)


def test_any_in_signatures_and_fields_is_flagged() -> None:
    source = """
    from typing import Any
    import typing

    def fit(a: Any, b: "Any", *args: Any) -> typing.Any:
        x: Any = 1
        return x

    class Record:
        value: Any
        other: object
    """
    assert flagged_lines(source, "any-annotation") == [5, 5, 5, 5, 10]


def test_any_through_an_alias_is_flagged() -> None:
    source = """
    from typing import Any
    Loose = list[Any]
    Looser = Loose | None

    def fit(a: Looser, b: object) -> None:
        pass
    """
    assert flagged_lines(source, "any-annotation") == [6]


def test_any_through_an_imported_alias_is_flagged() -> None:
    source = """
    from langslice.core.types import Loose

    def fit(a: Loose) -> None:
        pass
    """
    others = {"src/langslice/core/types.py": "Loose = tuple[Any, ...]\n"}
    assert flagged_lines(source, "any-annotation", others=others) == [4]


def test_any_as_record_value_is_left_to_loose_record() -> None:
    source = """
    def fit(a: dict[str, Any]) -> list[dict[str, Any]]:
        pass
    """
    assert flagged_lines(source, "any-annotation") == []
    assert flagged_lines(source, "loose-record") == [2, 2]


def test_loose_record_forms_are_flagged() -> None:
    source = """
    from collections.abc import Mapping
    Record = Mapping[str, object]
    Payload = dict[str, Record]

    def fit(a: Record, b: "dict[str, object]", c: dict[str, int]) -> None:
        pass

    def read(a: dict[str, Payload]) -> None:
        pass

    table: dict[str, Any] = {}
    """
    assert flagged_lines(source, "loose-record") == [6, 6, 9, 12]


def test_any_under_another_name_is_flagged() -> None:
    """The gaming snippet of the review: ``Any`` imported as ``A``."""
    source = """
    from typing import Any as A
    import typing as t
    from typing import Mapping as M

    def fit(a: A, b: t.Any, c: list["A"], d: int) -> None:
        pass

    def read(a: M[str, A]) -> None:
        pass
    """
    assert flagged_lines(source, "any-annotation") == [6, 6, 6]
    assert flagged_lines(source, "loose-record") == [9]


def test_loose_record_with_a_union_value_is_flagged() -> None:
    """A union with ``Any`` or ``object`` in it is no narrower."""
    source = """
    from collections.abc import Mapping, MutableMapping
    from typing import Any, Optional, Union

    def fit(
        a: dict[str, object | None],
        b: Mapping[str, Any | None],
        c: MutableMapping[str, Optional[object]],
        d: dict[str, Union[int, Any]],
        e: dict[str, "object | None"],
        f: dict[str, int | None] | None,
        g: dict[str, Any] | None,
    ) -> None:
        pass
    """
    assert flagged_lines(source, "loose-record") == [6, 7, 8, 9, 10, 12]


def test_typed_dict_is_flagged_outside_the_boundary() -> None:
    source = """
    class Row(TypedDict):
        name: str

    Other = TypedDict("Other", {"name": str})
    """
    assert flagged_lines(source, "loose-record") == [2, 5]


def test_typed_dict_in_the_boundary_needs_most_keys_required() -> None:
    source = """
    class Fine(TypedDict):
        a: str
        b: NotRequired[str]

    class Loose(TypedDict, total=False):
        a: str
        b: str
        c: Required[str]

    Gamed = TypedDict("Gamed", {"a": NotRequired[str], "b": NotRequired[int]})

    def fit(a: dict[str, Any]) -> None:
        pass
    """
    relpath = "src/langslice/job/formats.py"
    lines = flagged_lines(
        source, "loose-record", relpath=relpath, config=BOUNDARY
    )
    assert lines == [6, 11]


def test_or_empty_get_chains_are_flagged_once() -> None:
    source = """
    a = (record.transform or {}).get("physical")
    b = record.get("a", {}).get("b").get("c")
    c = (record.get("a") or dict()).get("b")
    d = record.get("a") or {}
    e = (record or {"x": 1}).get("x")
    """
    assert flagged_lines(source, "or-empty-get") == [2, 3, 4]


DECLARED = {
    "src/langslice/job/job.py": """
    class Job:
        spec: str

        def __init__(self):
            self.folder = 1
            self._cache = None
    """
}


def test_literal_getattr_on_a_declared_attribute_is_flagged() -> None:
    source = """
    def fit(job: Job, other):
        a = getattr(job, "spec", None)
        b = hasattr(job, "folder")
        c = vars(job)["spec"]
        d = job.__dict__["folder"]
        e = getattr(job, "undeclared", None)
        f = getattr(other, "spec")
        g = getattr(job, "_cache")
    """
    lines = flagged_lines(source, "literal-getattr", others=DECLARED)
    assert lines == [3, 4, 5, 6]


def test_literal_getattr_on_self_is_flagged() -> None:
    source = """
    class Store:
        def __init__(self):
            self._cache = None

        def read(self):
            return getattr(self, "_cache", None)
    """
    assert flagged_lines(source, "literal-getattr") == [7]


def test_isinstance_on_a_parameter_warns() -> None:
    source = """
    import ast

    def fit(value, node: ast.AST, other):
        local = 1
        if isinstance(value, str) or isinstance(local, int):
            pass
        if isinstance(node, ast.Name) or isinstance(other, ast.Call):
            pass
    """
    hits = check_source(source, "isinstance-outside-checkers")
    assert [violation.line for violation in hits] == [6]
    assert hits[0].severity is Severity.WARNING


def test_isinstance_in_the_checkers_module_is_allowed() -> None:
    source = """
    def check_type(value):
        return isinstance(value, str)
    """
    relpath = "src/langslice/core/checks.py"
    assert (
        flagged_lines(source, "isinstance-outside-checkers", relpath=relpath)
        == []
    )


def test_shared_option_literal_is_flagged_in_every_module() -> None:
    first = """
    def fit(metric):
        return metric == "mutual_information"
    """

    others = {
        "src/langslice/ops/fit.py": """
        def run(metric, mode):
            if metric in ("mutual_information", "ncc"):
                pass
            if mode.startswith("mutual_information"):
                pass
            match metric:
                case "mutual_information":
                    pass
        """,
    }

    relpath = "src/langslice/core/a_first.py"
    rule = "shared-option-literal"
    assert flagged_lines(first, rule, relpath=relpath, others=others) == [3]

    second = check_source(
        others["src/langslice/ops/fit.py"],
        rule,
        relpath="src/langslice/ops/fit.py",
        others={relpath: first},
    )
    assert [violation.line for violation in second] == [3, 5, 8]
    assert "core.a_first" in second[0].message


def test_shared_option_literal_blames_the_module_that_adds_a_site() -> None:
    """The gaming case of the review: a new module sorting first.

    Adding a site in a module whose path sorts before the others must
    give that module its own finding, and the old modules' findings must
    stay the same, so an untouched file is never newly blamed.
    """
    old = {
        "src/langslice/core/landmark_views.py": """
        def pick(mode):
            return mode == "atlas"
        """,
        "src/langslice/ops/look.py": """
        def pick(mode):
            return mode == "atlas"
        """,
    }
    new_relpath = "src/langslice/agent/aaa_probe.py"
    new = """
    def probe(mode):
        return mode == "atlas"
    """
    rule = "shared-option-literal"
    assert flagged_lines(new, rule, relpath=new_relpath, others=old) == [3]

    for relpath, text in old.items():
        before = {
            path: other for path, other in old.items() if path != relpath
        }
        after = {**before, new_relpath: new}
        lines_before = flagged_lines(
            text, rule, relpath=relpath, others=before
        )
        lines_after = flagged_lines(text, rule, relpath=relpath, others=after)
        assert lines_before == lines_after == [3]


def test_shared_option_literal_ignores_what_is_not_an_option() -> None:
    first = """
    def fit(name, payload, value):
        a = name == "true"
        b = "status" in payload
        c = name == "BAD_ARGS"
        d = name == "a b"
        e = name.endswith(".json")
        f = os.name == "posix"
        g = value.dtype == "float32"
        h = name == "dict"
        i = name == "x"
    """

    relpath = "src/langslice/core/a_first.py"
    others = {"src/langslice/ops/fit.py": first}
    lines = flagged_lines(first, "shared-option-literal", relpath=relpath)
    assert lines == []

    lines = flagged_lines(
        first, "shared-option-literal", relpath=relpath, others=others
    )
    assert lines == []
    second = check_source(
        first,
        "shared-option-literal",
        relpath="src/langslice/ops/fit.py",
        others={relpath: first},
    )
    assert second == []

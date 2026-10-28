"""Tests of the name checks: banned words, function names, type names."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from tools.stylecheck.core import Config
from tools.stylecheck.testing import check_source

CONFIG = Config.load(Path(__file__).parents[2] / "pyproject.toml")


def _with_types(type_names: dict[str, str]) -> Config:
    return dataclasses.replace(CONFIG, type_names=type_names)


def test_banned_stem_is_flagged_as_a_word_part() -> None:
    source = """
    def fit(held, *, held_value=1):
        self._held = held_value
        for found in []:
            pass
    """
    hits = check_source(source, "banned-name")
    names = [violation.message.split("'")[1] for violation in hits]
    assert names == ["held", "held_value", "_held", "found"]


def test_banned_stem_inside_another_word_is_allowed() -> None:
    source = """
    def fit(threshold, refound_count, shownumber):
        placement = threshold
        return placement
    """
    assert check_source(source, "banned-name") == []


def test_banned_name_is_reported_once_per_definition() -> None:
    source = """
    def fit():
        drawn = 1
        drawn = 2
        drawn += 3
    """
    assert len(check_source(source, "banned-name")) == 1


def test_function_suffix_is_flagged() -> None:
    source = """
    def kind_of(x):
        pass

    def flag_words(flag):
        pass

    def kind(x):
        pass
    """
    hits = check_source(source, "function-name")
    assert [violation.line for violation in hits] == [2, 5]


def test_predicate_must_return_bool() -> None:
    source = """
    def is_ready(x) -> str:
        pass

    def has_layers(x) -> bool:
        pass

    def can_fit(x) -> TypeGuard[int]:
        pass

    def is_open(x):
        pass

    def _is_valid(x) -> "bool":
        pass
    """
    hits = check_source(source, "function-name")
    assert [violation.line for violation in hits] == [2]


#: A pretend package with two classes named ``Job`` and one ``Workspace``.
TYPES = {
    "src/langslice/job/job.py": "class Job:\n    pass\n",
    "src/langslice/job/__init__.py": "from langslice.job.job import Job\n",
    "src/langslice/core/workspace.py": "class Workspace:\n    pass\n",
    "src/langslice/core/deformation.py": "class Job:\n    pass\n",
}

#: The table naming the ambiguous ``Job`` by its dotted name.
TABLE = {"Workspace": "workspace", "langslice.job.job.Job": "job"}


def _misnamed(source: str, table: dict[str, str]) -> list[str]:
    hits = check_source(
        source, "type-name", others=TYPES, config=_with_types(table)
    )
    return [violation.message.split("'")[1] for violation in hits]


def test_type_name_requires_the_table_name_or_a_qualifier() -> None:
    source = """
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job

    def run(ws: Workspace, job: Job, source_job: Job, ctx: "Workspace"):
        pass
    """
    assert _misnamed(source, TABLE) == ["ws", "ctx"]


def test_type_name_sees_through_optional_and_dotted_names() -> None:
    source = """
    from typing import Optional
    from langslice import job as jobs
    from langslice.job import Job

    def run(a: Job | None, b: Optional[jobs.Job], c: list[Job], self=None):
        pass
    """
    assert _misnamed(source, TABLE) == ["a", "b"]


def test_type_name_resolves_a_class_through_the_imports() -> None:
    """The collision of the review: another module's class of one name.

    ``deformation.Job`` is not the job's ``Job``, so it takes no name
    from the table; a bare key shared by two classes matches neither.
    """
    source = """
    from langslice.core import deformation
    from langslice.job.job import Job

    class Local:
        pass

    def caption(fit: deformation.Job, current: Job, other: Local):
        pass
    """
    assert _misnamed(source, TABLE) == ["current"]
    assert _misnamed(source, {"Job": "job"}) == []
    assert _misnamed(source, {"Local": "local"}) == ["other"]


def test_type_name_ignores_a_class_outside_the_package() -> None:
    source = """
    from somewhere import Workspace

    def run(ws: Workspace, unknown: Job):
        pass
    """
    assert _misnamed(source, TABLE) == []


def test_type_name_does_nothing_without_a_table() -> None:
    source = """
    def run(ws: Workspace):
        pass
    """
    assert check_source(source, "type-name", config=_with_types({})) == []

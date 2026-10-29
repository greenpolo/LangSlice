"""The checker's runner, baseline, report and command line, on fixtures.

The command-line tests replace the registered checks with one fixture
check (an error per ``TODO`` line, a warning per ``NOTE`` line) and switch
the external tools off, so they test the framework and nothing else.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import re
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from tools.stylecheck import core, external, pyright_levels, run
from tools.stylecheck import rules as rules  # registers every check
from tools.stylecheck.__main__ import main
from tools.stylecheck.baseline import (
    Baseline,
    BaselineError,
    compare,
    count_errors,
    format_baseline,
    parse_baseline,
    prune,
)
from tools.stylecheck.core import Module, Project, Severity, Violation
from tools.stylecheck.history import HistoryError, ReferenceKind, reference
from tools.stylecheck.moves import Move, Moves, read_moves
from tools.stylecheck.pyright_baseline import (
    PyrightEntry,
    parse_pyright_baseline,
    pyright_growth,
)
from tools.stylecheck.scope import Scope, Tool
from tools.stylecheck.settings import (
    CONFIG_DIRECTIONS,
    Record,
    loosenings,
)

REPOSITORY = Path(__file__).resolve().parents[2]

#: A record of one check and no settings, for the file-format tests.
RECORD = Record(("rule-a",), {})

PYPROJECT = """\
[tool.stylecheck]
paths = ["pkg"]
baseline = "baseline.json"
"""


def _violation(
    *,
    rule: str = "rule-a",
    path: str = "pkg/a.py",
    symbol: str = "f",
    line: int = 1,
    severity: Severity = Severity.ERROR,
) -> Violation:
    return Violation(rule, path, symbol, line, "Message.", severity)


# --- scope ---------------------------------------------------------------


def test_scope_selects_rules_codes_and_tools() -> None:
    scope = Scope(rules=("rule-a", "ruff:E501", "numpydoc"))
    assert scope.selects_rule("rule-a")
    assert not scope.selects_rule("rule-b")
    assert scope.selects_rule("ruff:E501")
    assert not scope.selects_rule("ruff:SLF001")
    assert scope.selects_rule("numpydoc:GL08")
    assert scope.selects_tool("ruff")
    assert scope.selects_tool("numpydoc")
    assert not scope.selects_tool("format")
    assert Scope().selects_tool("format")


def test_scope_selects_files_and_folders_not_prefixes() -> None:
    scope = Scope(paths=("pkg/sub", "pkg/b.py"))
    assert scope.selects_path("pkg/sub/c.py")
    assert scope.selects_path("pkg/b.py")
    assert not scope.selects_path("pkg/subway.py")
    assert not scope.selects_path("pkg/a.py")
    assert Scope().is_whole
    assert not scope.is_whole


# --- counting and comparing ----------------------------------------------


def test_count_errors_leaves_out_warnings() -> None:
    violations = [
        _violation(),
        _violation(line=2),
        _violation(severity=Severity.WARNING),
    ]
    assert count_errors(violations) == {("rule-a", "pkg/a.py", "f"): 2}


def test_compare_fails_a_new_key_and_a_grown_key_only() -> None:
    baseline = {
        ("rule-a", "pkg/a.py", "f"): 2,
        ("rule-a", "pkg/a.py", "g"): 1,
    }
    violations = [
        _violation(symbol="f", line=1),
        _violation(symbol="f", line=2),
        _violation(symbol="g", line=5),
        _violation(symbol="g", line=6),
        _violation(rule="rule-b", symbol="h", line=9),
        _violation(rule="rule-b", symbol="h", severity=Severity.WARNING),
    ]

    comparison = compare(violations, baseline, Scope())
    assert [item.line for item in comparison.failing] == [5, 6, 9]
    assert [
        (item.key, item.allowed, item.current) for item in comparison.excess
    ] == [
        (("rule-a", "pkg/a.py", "g"), 1, 2),
        (("rule-b", "pkg/a.py", "h"), 0, 1),
    ]
    assert comparison.n_covered == 2
    assert comparison.shrinkable == ()


def test_compare_reports_entries_that_can_shrink_within_scope() -> None:
    baseline = {
        ("rule-a", "pkg/a.py", "f"): 3,
        ("rule-a", "pkg/gone.py", "g"): 1,
        ("rule-b", "pkg/a.py", "f"): 4,
    }
    violations = [_violation()]
    comparison = compare(violations, baseline, Scope(rules=("rule-a",)))
    shrinkable = [
        (item.key[1], item.allowed, item.current)
        for item in comparison.shrinkable
    ]
    assert shrinkable == [("pkg/a.py", 3, 1), ("pkg/gone.py", 1, 0)]
    assert not comparison.failing


def test_prune_lowers_and_removes_but_never_adds_or_raises() -> None:
    baseline = {
        ("rule-a", "pkg/a.py", "f"): 3,
        ("rule-a", "pkg/a.py", "g"): 1,
        ("rule-b", "pkg/b.py", "h"): 2,
    }
    current = {
        ("rule-a", "pkg/a.py", "f"): 1,
        ("rule-a", "pkg/a.py", "new"): 5,
        ("rule-b", "pkg/b.py", "h"): 9,
    }
    assert prune(baseline, current, Scope()) == {
        ("rule-a", "pkg/a.py", "f"): 1,
        ("rule-b", "pkg/b.py", "h"): 2,
    }


def test_prune_keeps_entries_outside_the_scope() -> None:
    baseline = {("rule-a", "pkg/a.py", "f"): 3, ("rule-b", "pkg/a.py", "f"): 2}
    pruned = prune(baseline, {}, Scope(rules=("rule-a",)))
    assert pruned == {("rule-b", "pkg/a.py", "f"): 2}


# --- the file ------------------------------------------------------------


def test_baseline_text_is_sorted_one_item_per_line_and_round_trips() -> None:
    counts = {
        ("rule-b", "pkg/a.py", "f"): 1,
        ("rule-a", "pkg/b.py", "<module>"): 2,
        ("rule-a", "pkg/a.py", "g"): 3,
    }
    record = Record(
        ("rule-b", "rule-a"),
        {"stylecheck.max_nesting": 4, "ruff.lint.select": ["E", "F"]},
    )

    text = format_baseline(Baseline(counts, record))
    lines = text.splitlines()
    assert lines[0] == "{" and lines[-1] == "}"
    assert lines[1:5] == ['"checks": [', '"rule-a",', '"rule-b"', "],"]
    assert '"stylecheck.max_nesting": 4' in lines
    entries = lines[lines.index('"entries": [') + 1 : -2]
    assert [json.loads(line.rstrip(","))["path"] for line in entries] == [
        "pkg/a.py",
        "pkg/b.py",
        "pkg/a.py",
    ]
    assert entries[0].startswith('{"rule": ')
    assert "line" not in text

    parsed = parse_baseline(text, "test")
    assert parsed.counts == counts
    assert set(parsed.record.checks) == set(record.checks)
    assert parsed.record.settings == record.settings
    assert format_baseline(parsed) == text
    empty = format_baseline(Baseline({}, Record((), {})))
    assert json.loads(empty) == {"checks": [], "settings": {}, "entries": []}


def _document(entries: str) -> str:
    return '{"checks": [], "settings": {}, "entries": [' + entries + "]}"


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        '{"entries": []}',
        '{"checks": {}, "settings": {}, "entries": []}',
        '{"checks": [1], "settings": {}, "entries": []}',
        '{"checks": [], "settings": [], "entries": []}',
        '{"checks": [], "settings": {}, "entries": {}}',
        _document("1"),
        _document('{"rule": "a", "path": "b", "symbol": "c"}'),
        _document('{"rule": "a", "path": "b", "symbol": "c", "count": 0}'),
        _document('{"rule": "a", "path": "b", "symbol": "c", "count": true}'),
        _document(
            '{"rule": "a", "path": "b", "symbol": "c", "count": 1, "line": 3}'
        ),
        _document(
            '{"rule": "a", "path": "b", "symbol": "c", "count": 1},'
            ' {"rule": "a", "path": "b", "symbol": "c", "count": 2}'
        ),
        "not json",
    ],
)
def test_a_malformed_baseline_is_refused(text: str) -> None:
    with pytest.raises(BaselineError):
        parse_baseline(text, "test")


# --- the record of checks and settings -------------------------------------


def test_every_config_field_has_a_direction() -> None:
    fields = {field.name for field in dataclasses.fields(core.Config)}
    assert set(CONFIG_DIRECTIONS) == fields


@pytest.mark.parametrize(
    ("name", "before", "after", "looser"),
    [
        ("stylecheck.max_nesting", 4, 3, False),
        ("stylecheck.max_nesting", 4, 5, True),
        ("stylecheck.min_parameter_words", 3, 2, True),
        ("stylecheck.banned_name_stems", ["held"], ["held", "x"], False),
        ("stylecheck.banned_name_stems", ["held", "x"], ["x"], True),
        ("stylecheck.cli_modules", ["a", "b"], ["a"], False),
        ("stylecheck.cli_modules", ["a"], ["a", "b"], True),
        (
            "stylecheck.type_names",
            {"Job": "job"},
            {"Job": "job", "A": "a"},
            False,
        ),
        ("stylecheck.type_names", {"Job": "job"}, {"Job": "work"}, True),
        ("stylecheck.decorators_module", "", "pkg.decorators", True),
        ("checker-ruff.select", ["E501"], ["E501", "N"], False),
        ("checker-ruff.ignore", ["N803"], ["N803", "N806"], True),
        ("ruff.lint.ignore", None, ["E501"], True),
        ("ruff.lint.per-file-ignores", {"a.py": ["D"]}, {}, False),
        (
            "ruff.lint.per-file-ignores",
            {"a.py": ["D"]},
            {"a.py": ["D", "E"]},
            True,
        ),
        ("ruff.lint.extend-per-file-ignores", None, {"b.py": ["D"]}, True),
        ("ruff.line-length", 79, 100, True),
        ("ruff.lint.isort.known-first-party", None, ["pkg"], True),
        ("numpydoc.checks", ["GL08"], [], True),
        ("numpydoc.exclude", ["x"], [], False),
        ("basedpyright.reportAny", "warning", "error", False),
        ("basedpyright.reportAny", "error", False, True),
        ("basedpyright.typeCheckingMode", "strict", "standard", True),
        ("basedpyright.include", ["src"], ["src", "tools"], False),
    ],
)
def test_a_setting_may_change_only_in_its_direction(
    name: str, before: object, after: object, looser: bool
) -> None:
    settings_before = {} if before is None else {name: before}
    changes = loosenings(
        Record((), settings_before), Record((), {name: after})
    )
    assert [change.name for change in changes] == ([name] if looser else [])


#: The fake rule sets of basedpyright's checking modes in the tests below.
LEVELS = {"strict": {"reportA": "error", "reportB": "none"}}


@pytest.mark.parametrize(
    ("before", "after", "looser"),
    [
        # a new key at least as strict as the mode's default for it
        ({}, {"basedpyright.reportB": "warning"}, False),
        ({}, {"basedpyright.reportA": "error"}, False),
        ({}, {"basedpyright.reportA": True}, False),
        # a new key below the mode's default
        ({}, {"basedpyright.reportA": "warning"}, True),
        # a rule the mode does not list: only the highest level is safe
        ({}, {"basedpyright.reportC": "error"}, False),
        ({}, {"basedpyright.reportC": "warning"}, True),
        # a key removed: its rule falls back to the mode's default
        ({"basedpyright.reportB": "warning"}, {}, True),
        ({"basedpyright.reportA": "warning"}, {}, False),
    ],
)
def test_a_basedpyright_key_is_set_against_its_modes_default(
    monkeypatch: pytest.MonkeyPatch,
    before: dict[str, object],
    after: dict[str, object],
    looser: bool,
) -> None:
    monkeypatch.setattr(pyright_levels, "mode_levels", lambda: LEVELS)
    mode = {"basedpyright.typeCheckingMode": "strict"}
    changes = loosenings(
        Record((), {**mode, **before}), Record((), {**mode, **after})
    )
    assert bool(changes) is looser


def test_the_modes_defaults_are_read_from_basedpyrights_bundle() -> None:
    bundle = """\
function getStrictDiagnosticRuleSet() {
    const diagSettings = {
        strictListInference: true,
        reportAny: 'none',
        reportUnusedImport: 'error', // a comment
    };
    return diagSettings;
}
const getAllDiagnosticRuleSet = () => ({
    reportAny: 'error',
});
function other() {
    reportAny: 'hint',
}
"""
    assert pyright_levels.parse_mode_levels(bundle) == {
        "strict": {"reportAny": "none", "reportUnusedImport": "error"},
        "all": {"reportAny": "error"},
    }
    # the installed basedpyright lists the strict mode the project uses
    if pyright_levels.mode_levels():
        assert "reportAny" in pyright_levels.mode_levels()["strict"]


def test_a_check_no_longer_run_is_a_loosening() -> None:
    changes = loosenings(Record(("a", "b"), {}), Record(("a",), {}))
    assert [(change.name, change.before) for change in changes] == [
        ("checks", '"b"')
    ]


def test_a_new_checker_field_is_not_a_loosening() -> None:
    after = Record((), {"stylecheck.max_new_thing": 3})
    assert loosenings(Record((), {}), after) == []


def test_a_record_covers_the_rules_it_ran() -> None:
    record = Record(
        ("rule-a",),
        {
            "checker-ruff.select": ["E501", "D2", "N"],
            "checker-ruff.ignore": ["N803"],
            "ruff.lint.extend-per-file-ignores": {
                "pkg/declarations.py": ["D", "E501"],
                "tests/**": ["N"],
            },
            "numpydoc.checks": ["GL08", "PR01"],
        },
    )
    assert record.covers(("rule-a", "pkg/a.py", "f"))
    assert not record.covers(("rule-b", "pkg/a.py", "f"))
    assert record.covers(("ruff:E501", "pkg/a.py", "f"))
    assert record.covers(("ruff:D205", "pkg/a.py", "f"))
    assert not record.covers(("ruff:D100", "pkg/a.py", "f"))
    assert not record.covers(("ruff:N803", "pkg/a.py", "f"))
    assert not record.covers(("ruff:E501", "pkg/declarations.py", "f"))
    assert not record.covers(("ruff:N802", "tests/sub/t.py", "f"))
    assert record.covers(("numpydoc:PR01", "pkg/a.py", "f"))
    assert not record.covers(("numpydoc:RT01", "pkg/a.py", "f"))


def test_a_record_gives_the_config_it_was_written_with() -> None:
    record = Record(
        (),
        {
            "stylecheck.max_nesting": 2,
            "stylecheck.cli_modules": ["pkg.cli"],
            "stylecheck.layers": [["pkg.top"], ["pkg.core"]],
        },
    )
    config = record.config(core.Config())
    assert config.max_nesting == 2
    assert config.cli_modules == ("pkg.cli",)
    assert config.layers == (("pkg.top",), ("pkg.core",))
    assert config.max_body_lines == core.Config().max_body_lines


# --- basedpyright's baseline -------------------------------------------------


def _pyright_text(entries: dict[str, list[tuple[str, int]]]) -> str:
    files = {
        path: [
            {
                "code": code,
                "range": {
                    "startColumn": column,
                    "endColumn": column + 3,
                    "lineCount": 1,
                },
            }
            for code, column in items
        ]
        for path, items in entries.items()
    }
    return json.dumps({"files": files}, indent=4)


def _pyright_entries(
    entries: dict[str, list[tuple[str, int]]],
) -> list[PyrightEntry]:
    return parse_pyright_baseline(_pyright_text(entries), "test")


def _never(_: object) -> bool:
    return False


def _no_sources(_: str) -> tuple[str, ...]:
    return ()


def test_pyright_baseline_is_read_per_entry() -> None:
    entries = _pyright_entries(
        {"./a.py": [("reportAny", 4), ("reportAny", 8)]}
    )
    assert entries == [
        PyrightEntry("./a.py", "reportAny", (4, 7, 1)),
        PyrightEntry("./a.py", "reportAny", (8, 11, 1)),
    ]
    with pytest.raises(Exception, match="files"):
        parse_pyright_baseline("[]", "test")


def test_pyright_growth_flags_a_new_file_and_a_grown_rule() -> None:
    before = _pyright_entries({"./a.py": [("reportAny", 4)]})
    after = _pyright_entries(
        {
            "./a.py": [("reportAny", 4), ("reportAny", 9)],
            "./b.py": [("reportAny", 1)],
        }
    )
    changes = pyright_growth(
        before,
        after,
        is_adopted=_never,
        move_sources=_no_sources,
    )
    assert [(c.path, c.code, c.allowed, c.current) for c in changes] == [
        ("./a.py", "reportAny", 1, 2),
        ("./b.py", "reportAny", 0, 1),
    ]
    shrunk = pyright_growth(
        after,
        before,
        is_adopted=_never,
        move_sources=_no_sources,
    )
    assert shrunk == []


def test_pyright_growth_follows_a_moved_file_only_when_declared() -> None:
    before = _pyright_entries({"./old.py": [("reportAny", 4)]})
    moved = _pyright_entries({"./new.py": [("reportAny", 4)]})
    copied = _pyright_entries(
        {"./new.py": [("reportAny", 4)], "./copy.py": [("reportAny", 4)]}
    )

    def from_old(path: str) -> tuple[str, ...]:
        return ("./old.py",) if path == "./new.py" else ()

    def growth(
        entries: list[PyrightEntry],
        sources: Callable[[str], tuple[str, ...]],
    ) -> int:
        changes = pyright_growth(
            before, entries, is_adopted=_never, move_sources=sources
        )
        return len(changes)

    assert growth(moved, from_old) == 0
    assert growth(copied, from_old) == 1
    # a gone file without a declared move takes its entries with it
    assert growth(moved, _no_sources) == 1


def test_pyright_growth_follows_a_declared_move_by_rule_once() -> None:
    # old.py stays; one entry left it for new.py at another span
    before = _pyright_entries(
        {"./old.py": [("reportAny", 4), ("reportAny", 20)]}
    )
    after = _pyright_entries(
        {"./old.py": [("reportAny", 4)], "./new.py": [("reportAny", 9)]}
    )
    twice = _pyright_entries(
        {
            "./old.py": [("reportAny", 4)],
            "./new.py": [("reportAny", 9), ("reportAny", 12)],
        }
    )
    other_rule = _pyright_entries(
        {"./old.py": [("reportAny", 4)], "./new.py": [("reportUnknown", 9)]}
    )

    def from_old(path: str) -> tuple[str, ...]:
        return ("./old.py",) if path == "./new.py" else ()

    def check(entries: list[PyrightEntry]) -> int:
        changes = pyright_growth(
            before,
            entries,
            is_adopted=_never,
            move_sources=from_old,
        )
        return len(changes)

    assert check(after) == 0
    assert check(twice) == 1
    assert check(other_rule) == 1
    unmapped = pyright_growth(
        before,
        after,
        is_adopted=_never,
        move_sources=_no_sources,
    )
    assert len(unmapped) == 1


# --- a file at a git commit ---------------------------------------------------


def _git(root: Path, *arguments: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def _init_repository(root: Path) -> None:
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "commit.gpgsign", "false")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Make a git repository with one commit and no baseline.

    Parameters
    ----------
    tmp_path
        pytest's empty folder for this test.

    Returns
    -------
    Path
        The repository's root.
    """
    _init_repository(tmp_path)
    (tmp_path / "README").write_text("start\n")
    _commit(tmp_path, "start")
    return tmp_path


def _commit(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message)
    return _git(root, "rev-parse", "HEAD")


def test_reference_is_the_file_at_the_commit(repo: Path) -> None:
    (repo / "file").write_text("one\n")
    first = _commit(repo, "add it")
    (repo / "file").write_text("two\n")
    found = reference(repo, first, "file")
    assert found.kind is ReferenceKind.AT_REF
    assert found.text == "one\n"


def test_reference_is_the_first_version_created_after_the_commit(
    repo: Path,
) -> None:
    before = _git(repo, "rev-parse", "HEAD")
    (repo / "file").write_text("one\n")
    _commit(repo, "create it")
    (repo / "file").write_text("two\n")
    _commit(repo, "change it in the same push")
    found = reference(repo, before, "file")
    assert found.kind is ReferenceKind.CREATED
    assert found.text == "one\n"
    assert "created in" in found.description


def test_reference_of_an_uncommitted_file_is_nothing(repo: Path) -> None:
    (repo / "file").write_text("one\n")
    found = reference(repo, "HEAD", "file")
    assert found.kind is ReferenceKind.NEW
    assert found.text is None
    assert "new since" in found.description


def test_reference_of_a_deleted_file_is_empty(repo: Path) -> None:
    (repo / "file").write_text("one\n")
    _commit(repo, "add it")
    (repo / "file").unlink()
    deleted = _commit(repo, "delete it")
    (repo / "file").write_text("again\n")
    _commit(repo, "recreate it")
    found = reference(repo, deleted, "file")
    assert found.kind is ReferenceKind.DELETED
    assert found.text is None


def test_reference_refuses_a_ref_that_names_no_commit(repo: Path) -> None:
    with pytest.raises(HistoryError):
        reference(repo, "no-such-branch", "file")


# --- the command line ----------------------------------------------------


def _todo_check(
    module: Module, project: Project, config: core.Config
) -> Iterator[Violation]:
    """Flag each TODO line as an error and each NOTE line as a warning."""
    for number, text in enumerate(module.lines, start=1):
        if "TODO" in text:
            yield module.violation("fixture-todo", number, "A TODO is left.")
        if "NOTE" in text:
            yield module.violation(
                "fixture-todo",
                number,
                "A NOTE is left.",
                severity=Severity.WARNING,
            )


def _long_check(
    module: Module, project: Project, config: core.Config
) -> Iterator[Violation]:
    """Flag a file longer than ``max_module_lines``."""
    if len(module.lines) > config.max_module_lines:
        yield module.violation("fixture-long", 1, "The file is long.")


@pytest.fixture
def project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Make a fixture project checked by the TODO and length checks alone.

    Parameters
    ----------
    tmp_path
        pytest's empty folder for this test.
    monkeypatch
        pytest's patcher, which swaps the registered checks and the
        external tools for this test only.

    Returns
    -------
    Path
        The project's root: ``pkg/a.py`` has two TODO errors and
        ``pkg/b.py`` one NOTE warning.
    """
    checks = {"fixture-todo": _todo_check, "fixture-long": _long_check}
    monkeypatch.setattr(core, "MODULE_CHECKS", checks)
    monkeypatch.setattr(core, "PROJECT_CHECKS", {})
    monkeypatch.setattr(run, "EXTERNAL_TOOLS", {})
    (tmp_path / "pyproject.toml").write_text(PYPROJECT)
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "a.py").write_text(
        "def f():\n    pass  # TODO\n\n\ndef g():\n    pass  # TODO\n"
    )
    (package / "b.py").write_text("x = 1  # NOTE\n")
    return tmp_path


def _main(root: Path, *arguments: str) -> int:
    return main(argv=["--root", str(root), *arguments])


def _read(root: Path) -> Baseline:
    return parse_baseline((root / "baseline.json").read_text(), "test")


def _baseline_counts(root: Path) -> dict[tuple[str, str, str], int]:
    return _read(root).counts


def _write_counts(root: Path, counts: dict[tuple[str, str, str], int]) -> None:
    baseline = Baseline(counts, _read(root).record)
    (root / "baseline.json").write_text(format_baseline(baseline))


def test_cli_fails_without_a_baseline_and_groups_by_file(
    project_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _main(project_root) == 1
    out = capsys.readouterr().out
    assert "pkg/a.py\n" in out
    assert "      2  fixture-todo  f: A TODO is left." in out
    assert "      6  fixture-todo  g: A TODO is left." in out
    assert "2 errors not covered by the baseline, in 1 files." in out
    assert "fixture-todo 1" in out  # the warning count
    assert "A NOTE is left." not in out


def test_cli_lists_warnings_when_asked(
    project_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _main(project_root, "--show-warnings")
    assert "A NOTE is left." in capsys.readouterr().out


def test_cli_write_baseline_then_pass(
    project_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _main(project_root, "--write-baseline") == 0
    assert _baseline_counts(project_root) == {
        ("fixture-todo", "pkg/a.py", "f"): 1,
        ("fixture-todo", "pkg/a.py", "g"): 1,
    }
    assert _main(project_root) == 0
    out = capsys.readouterr().out
    assert "2 errors are covered by the baseline." in out


def test_cli_write_baseline_refuses_to_overwrite(
    project_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _main(project_root, "--write-baseline") == 0
    assert _main(project_root, "--write-baseline") == 2
    assert "exists" in capsys.readouterr().err
    assert _main(project_root, "--write-baseline", "--replace-baseline") == 0


def test_cli_write_baseline_refuses_a_partial_run(
    project_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _main(project_root, "--write-baseline", "--paths", "pkg/a.py") == 2
    assert not (project_root / "baseline.json").exists()


def test_cli_replace_baseline_needs_write_baseline(project_root: Path) -> None:
    with pytest.raises(SystemExit):
        _main(project_root, "--replace-baseline")


def test_cli_a_new_error_in_a_baselined_definition_fails(
    project_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _main(project_root, "--write-baseline")
    path = project_root / "pkg" / "a.py"
    path.write_text(path.read_text() + "    x = 1  # TODO\n")
    assert _main(project_root) == 1
    out = capsys.readouterr().out
    assert "[2 here, the baseline allows 1]" in out


def test_cli_moving_an_error_within_a_definition_still_passes(
    project_root: Path,
) -> None:
    _main(project_root, "--write-baseline")
    path = project_root / "pkg" / "a.py"
    path.write_text("# a comment\n\n" + path.read_text())
    assert _main(project_root) == 0


def test_cli_prune_lowers_the_baseline(
    project_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _main(project_root, "--write-baseline")
    path = project_root / "pkg" / "a.py"
    path.write_text("def f():\n    pass\n\n\ndef g():\n    pass  # TODO\n")
    assert _main(project_root) == 0
    assert "1 baseline entries can shrink" in capsys.readouterr().out
    assert _main(project_root, "--prune") == 0
    assert _baseline_counts(project_root) == {
        ("fixture-todo", "pkg/a.py", "g"): 1
    }


def test_cli_prune_with_a_path_filter_keeps_other_files(
    project_root: Path,
) -> None:
    _main(project_root, "--write-baseline")
    (project_root / "pkg" / "a.py").write_text("x = 1\n")
    _main(project_root, "--prune", "--paths", "pkg/b.py")
    assert len(_baseline_counts(project_root)) == 2


def test_cli_prune_never_adds_an_entry(project_root: Path) -> None:
    _main(project_root, "--write-baseline")
    (project_root / "pkg" / "c.py").write_text("y = 2  # TODO\n")
    assert _main(project_root, "--prune") == 1
    assert ("fixture-todo", "pkg/c.py", "<module>") not in _baseline_counts(
        project_root
    )


def test_cli_rule_and_paths_filters(
    project_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _main(project_root, "--paths", "pkg/b.py") == 0
    assert _main(project_root, "--rule", "fixture-todo") == 1
    assert _main(project_root, "--rule", "no-such-rule") == 2
    assert "No check is named" in capsys.readouterr().err


def test_cli_explain_prints_the_docstring(
    project_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _main(project_root, "--explain", "fixture-todo") == 0
    assert "Flag each TODO line" in capsys.readouterr().out


def test_cli_against_a_commit(
    project_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _init_repository(project_root)
    start = _commit(project_root, "code without a baseline")
    _main(project_root, "--write-baseline")
    assert _main(project_root, "--against", start) == 0
    assert "new since" in capsys.readouterr().out
    created = _commit(project_root, "baseline")

    # a hand-edited baseline that lists a new error is refused
    counts = _baseline_counts(project_root)
    counts[("fixture-todo", "pkg/c.py", "<module>")] = 1
    _write_counts(project_root, counts)
    (project_root / "pkg" / "c.py").write_text("y = 2  # TODO\n")
    assert _main(project_root, "--against", created) == 1
    assert "The baseline grew" in capsys.readouterr().out


@pytest.fixture
def committed(project_root: Path) -> str:
    """Commit the fixture project with its baseline written.

    Parameters
    ----------
    project_root
        The fixture project, made a git repository here.

    Returns
    -------
    str
        The commit that holds the baseline.
    """
    _init_repository(project_root)
    assert _main(project_root, "--write-baseline") == 0
    return _commit(project_root, "baseline")


def _rewrite(root: Path) -> int:
    return _main(root, "--write-baseline", "--replace-baseline")


def test_cli_against_flags_a_raised_count_and_accepts_a_shrink(
    project_root: Path, committed: str, capsys: pytest.CaptureFixture[str]
) -> None:
    path = project_root / "pkg" / "a.py"
    path.write_text(path.read_text() + "    x = 1  # TODO\n")
    _rewrite(project_root)
    assert _main(project_root, "--against", committed) == 1
    out = capsys.readouterr().out
    assert "pkg/a.py  fixture-todo  g: 2 (the reference allows 1)" in out

    path.write_text("def f():\n    pass\n")
    _main(project_root, "--prune")
    assert _main(project_root, "--against", committed) == 0
    assert "The baseline only shrank" in capsys.readouterr().out


def test_cli_against_adopts_a_new_check_but_not_a_check_without_hits(
    project_root: Path,
    committed: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def note_check(
        module: Module, project: Project, config: core.Config
    ) -> Iterator[Violation]:
        for number, text in enumerate(module.lines, start=1):
            if "NOTE" in text:
                yield module.violation("fixture-note", number, "A NOTE.")

    checks = {**core.MODULE_CHECKS, "fixture-note": note_check}
    monkeypatch.setattr(core, "MODULE_CHECKS", checks)
    _rewrite(project_root)
    assert ("fixture-note", "pkg/b.py", "<module>") in _baseline_counts(
        project_root
    )
    assert _main(project_root, "--against", committed) == 0
    assert "1 errors entered with a new rule" in capsys.readouterr().out

    # fixture-long ran at the reference with no hits: its entries are new
    (project_root / "pkg" / "c.py").write_text("x = 1\n" * 900)
    _rewrite(project_root)
    assert _main(project_root, "--against", committed) == 1
    assert "pkg/c.py  fixture-long" in capsys.readouterr().out


def _set_pyproject(root: Path, extra: str) -> None:
    (root / "pyproject.toml").write_text(PYPROJECT + extra)


def test_cli_against_adopts_what_a_stricter_limit_reveals_only(
    project_root: Path, committed: str, capsys: pytest.CaptureFixture[str]
) -> None:
    (project_root / "pkg" / "c.py").write_text("x = 1\n" * 9)
    _commit(project_root, "a file of nine lines")
    reference_commit = _git(project_root, "rev-parse", "HEAD")

    _set_pyproject(project_root, "max_module_lines = 5\n")
    assert _main(project_root) == 1  # pkg/c.py and pkg/a.py are now long
    _rewrite(project_root)
    assert _main(project_root, "--against", reference_commit) == 0
    assert "2 errors entered" in capsys.readouterr().out

    # a long file added with the tightening is long under the old limit too
    (project_root / "pkg" / "d.py").write_text("x = 1\n" * 900)
    _rewrite(project_root)
    assert _main(project_root, "--against", reference_commit) == 1
    assert "pkg/d.py  fixture-long" in capsys.readouterr().out


def test_cli_against_refuses_a_looser_setting_unless_allowed(
    project_root: Path, committed: str, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_pyproject(project_root, "max_module_lines = 900\n")
    # the file still records the old limit
    assert _main(project_root, "--against", committed) == 1
    out = capsys.readouterr().out
    assert "records other checks or settings" in out
    assert "stylecheck.max_module_lines" in out

    _main(project_root, "--prune")
    assert _main(project_root, "--against", committed) == 1
    out = capsys.readouterr().out
    assert "stylecheck.max_module_lines: 800 -> 900" in out
    assert "the designer's decision" in out
    allowed = ("--against", committed, "--allow-loosening")
    assert _main(project_root, *allowed) == 0
    assert "Accepted with --allow-loosening." in capsys.readouterr().out


def test_cli_allow_loosening_needs_against(project_root: Path) -> None:
    with pytest.raises(SystemExit):
        _main(project_root, "--allow-loosening")


def test_cli_plain_run_notes_an_outdated_record(
    project_root: Path, committed: str, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_pyproject(project_root, "max_module_lines = 700\n")
    assert _main(project_root) == 0
    assert "--prune records the run's" in capsys.readouterr().out
    _main(project_root, "--prune")
    settings = _read(project_root).record.settings
    assert settings["stylecheck.max_module_lines"] == 700


def test_cli_against_follows_a_moved_file_only_through_the_map(
    project_root: Path, committed: str, capsys: pytest.CaptureFixture[str]
) -> None:
    package = project_root / "pkg"
    original = (package / "a.py").read_text()
    (package / "a.py").rename(package / "moved.py")
    _rewrite(project_root)
    assert _main(project_root, "--against", committed) == 1
    assert "pkg/moved.py  fixture-todo  f" in capsys.readouterr().out

    _write_moves(
        project_root,
        ("pkg/a.py", "f", "pkg/moved.py", "f"),
        ("pkg/a.py", "g", "pkg/moved.py", "g"),
    )
    assert _main(project_root, "--against", committed) == 0
    assert "2 errors moved with their files" in capsys.readouterr().out

    # a copy: the old file keeps its errors, so none of them moved
    (package / "a.py").write_text(original)
    _rewrite(project_root)
    assert _main(project_root, "--against", committed) == 1
    assert "pkg/moved.py  fixture-todo  f" in capsys.readouterr().out


def test_cli_against_lends_a_deleted_files_errors_to_no_other_file(
    project_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # a module-level error in a file that goes cannot pay for a new one
    (project_root / "pkg" / "c.py").write_text("y = 2  # TODO\n")
    _init_repository(project_root)
    _main(project_root, "--write-baseline")
    reference_commit = _commit(project_root, "c.py has an error")

    (project_root / "pkg" / "c.py").unlink()
    (project_root / "pkg" / "d.py").write_text("z = 3  # TODO\n")
    _rewrite(project_root)
    assert _main(project_root, "--against", reference_commit) == 1
    assert "pkg/d.py  fixture-todo  <module>" in capsys.readouterr().out


def _write_pyright(
    root: Path, entries: dict[str, list[tuple[str, int]]]
) -> None:
    folder = root / ".basedpyright"
    folder.mkdir(exist_ok=True)
    (folder / "baseline.json").write_text(_pyright_text(entries))


def test_cli_against_holds_the_pyright_baseline_to_only_shrinks(
    project_root: Path, committed: str, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_pyright(project_root, {"./pkg/a.py": [("reportAny", 4)]})
    # created after the reference and not committed: the creation
    assert _main(project_root, "--against", committed) == 0
    created = _commit(project_root, "pyright baseline")

    _write_pyright(
        project_root,
        {"./pkg/a.py": [("reportAny", 4)], "./pkg/b.py": [("reportAny", 1)]},
    )
    assert _main(project_root, "--against", created) == 1
    out = capsys.readouterr().out
    assert "basedpyright's baseline grew" in out
    assert "./pkg/b.py  reportAny: 1 (the reference allows 0)" in out

    # a rule whose report level changed since the reference is adopted;
    # a new key at the highest level tightens whatever the mode's default
    _set_pyproject(
        project_root, '\n[tool.basedpyright]\nreportAny = "error"\n'
    )
    _main(project_root, "--prune")
    assert _main(project_root, "--against", created) == 0
    # without a mode, a lower level may be below the unknown default
    _set_pyproject(
        project_root, '\n[tool.basedpyright]\nreportAny = "warning"\n'
    )
    _main(project_root, "--prune")
    assert _main(project_root, "--against", created) == 1
    assert _main(project_root, "--against", created, "--allow-loosening") == 0
    capsys.readouterr()


def test_cli_against_follows_a_moved_file_in_the_pyright_baseline(
    project_root: Path, committed: str, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_pyright(project_root, {"./pkg/a.py": [("reportAny", 4)]})
    created = _commit(project_root, "pyright baseline")
    package = project_root / "pkg"
    (package / "a.py").rename(package / "moved.py")
    _rewrite(project_root)
    _write_pyright(project_root, {"./pkg/moved.py": [("reportAny", 4)]})
    assert _main(project_root, "--against", created) == 1
    assert "basedpyright's baseline grew" in capsys.readouterr().out

    _write_moves(
        project_root,
        ("pkg/a.py", "f", "pkg/moved.py", "f"),
        ("pkg/a.py", "g", "pkg/moved.py", "g"),
    )
    assert _main(project_root, "--against", created) == 0
    assert "basedpyright's baseline only shrank" in capsys.readouterr().out


def _write_moves(root: Path, *moves: tuple[str, str, str, str]) -> None:
    folder = root / "tools" / "stylecheck"
    folder.mkdir(parents=True, exist_ok=True)
    fields = ("path", "symbol", "new_path", "new_symbol")
    document = {
        "moves": [dict(zip(fields, move, strict=True)) for move in moves]
    }
    (folder / "moves.json").write_text(json.dumps(document))


def test_cli_against_follows_a_declared_move_once(
    project_root: Path, committed: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # g leaves pkg/a.py, which stays, for pkg/c.py under a new name
    package = project_root / "pkg"
    (package / "a.py").write_text("def f():\n    pass  # TODO\n")
    (package / "c.py").write_text("def h():\n    pass  # TODO\n")
    _rewrite(project_root)
    assert _main(project_root, "--against", committed) == 1
    assert "pkg/c.py  fixture-todo  h" in capsys.readouterr().out

    _write_moves(project_root, ("pkg/a.py", "g", "pkg/c.py", "h"))
    assert _main(project_root, "--against", committed) == 0
    assert "1 errors moved with their files" in capsys.readouterr().out

    # at most the old count, and only once the old entry is gone
    (package / "c.py").write_text("def h():\n    pass  # TODO\n    # TODO\n")
    _rewrite(project_root)
    assert _main(project_root, "--against", committed) == 1
    (package / "c.py").write_text("def h():\n    pass  # TODO\n")
    (package / "a.py").write_text(
        "def f():\n    pass  # TODO\n\n\ndef g():\n    pass  # TODO\n"
    )
    _rewrite(project_root)
    assert _main(project_root, "--against", committed) == 1


def test_a_declared_move_renames_the_modules_the_settings_name() -> None:
    moves = Moves(
        (
            Move("src/pkg/a.py", "<module>", "src/pkg/c.py", "<module>"),
            Move("src/pkg/a.py", "<module>", "src/pkg/d.py", "<module>"),
            Move("src/pkg/a.py", "f", "src/pkg/d.py", "f"),
            Move(
                "src/pkg/agent/__init__.py",
                "<module>",
                "src/pkg/top/agent/__init__.py",
                "<module>",
            ),
        )
    )
    before: dict[str, object] = {
        "stylecheck.cli_modules": ["pkg.a", "pkg.b"],
        "stylecheck.docs_renderer_module": "pkg.a",
        "stylecheck.layers": [["pkg.top", "pkg.agent"], ["pkg.low"]],
    }
    after: dict[str, object] = {
        "stylecheck.cli_modules": ["pkg.b", "pkg.d"],
        "stylecheck.docs_renderer_module": "pkg.d",
        "stylecheck.layers": [["pkg.top"], ["pkg.low"]],
    }
    checks = ("rule-a",)
    reference, now = Record(checks, before), Record(checks, after)
    renamed = moves.translate(reference, now).settings
    assert renamed["stylecheck.cli_modules"] == ["pkg.b", "pkg.c", "pkg.d"]
    assert renamed["stylecheck.docs_renderer_module"] == "pkg.d"
    assert renamed["stylecheck.layers"] == [["pkg.top"], ["pkg.low"]]
    assert loosenings(moves.translate(reference, now), now) == []
    assert loosenings(reference, now)


def test_a_package_in_a_setting_stands_for_every_module_moved_out() -> None:
    rows = (
        Move("src/pkg/cli/__init__.py", "<module>", "src/pkg/x/p.py", "main"),
        Move("src/pkg/cli/run.py", "f", "src/pkg/y/run.py", "f"),
        Move("src/pkg/cli/run.py", "g", "src/pkg/cli/run.py", "h"),
    )
    before = Record((), {"stylecheck.cli_modules": ["pkg.cli", "pkg.z"]})
    after = Record((), {})
    gone = Moves(rows).translate(before, after).settings
    assert gone["stylecheck.cli_modules"] == [
        "pkg.cli.run",
        "pkg.x.p",
        "pkg.y.run",
        "pkg.z",
    ]
    # while a module inside the package is kept, the package stays named
    kept = Moves(rows, frozenset({"pkg.cli.run"}))
    assert kept.translate(before, after).settings[
        "stylecheck.cli_modules"
    ] == ["pkg.cli", "pkg.cli.run", "pkg.x.p", "pkg.y.run", "pkg.z"]


def test_a_module_split_into_parts_shares_its_module_level(
    project_root: Path, committed: str, capsys: pytest.CaptureFixture[str]
) -> None:
    package = project_root / "pkg"
    module_level = "x = 1  # TODO\n\n\n"
    (package / "a.py").write_text(
        module_level + "def f():\n    pass  # TODO\n\n\n"
        "def g():\n    pass  # TODO\n"
    )
    _rewrite(project_root)
    reference_commit = _commit(project_root, "a module-level error")

    # a.py splits into c.py and d.py; only c.py keeps the module-level line
    (package / "a.py").unlink()
    (package / "c.py").write_text(
        module_level + "def f():\n    pass  # TODO\n"
    )
    (package / "d.py").write_text("def g():\n    pass  # TODO\n")
    _rewrite(project_root)
    _write_moves(
        project_root,
        ("pkg/a.py", "<module>", "pkg/c.py", "<module>"),
        ("pkg/a.py", "<module>", "pkg/d.py", "<module>"),
        ("pkg/a.py", "f", "pkg/c.py", "f"),
        ("pkg/a.py", "g", "pkg/d.py", "g"),
    )
    assert _main(project_root, "--against", reference_commit) == 0
    capsys.readouterr()

    # both parts copy it: together they may have the old count, not each
    (package / "d.py").write_text(
        module_level + "def g():\n    pass  # TODO\n"
    )
    _rewrite(project_root)
    assert _main(project_root, "--against", reference_commit) == 1
    out = capsys.readouterr().out
    assert "fixture-todo  <module>: 1 (the reference allows 0)" in out


def _split_rows(*parts: str) -> list[tuple[str, str, str, str]]:
    return [("pkg/a.py", "<module>", part, "<module>") for part in parts]


def test_a_per_file_rule_is_inherited_by_each_piece_of_a_split(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tools = {Tool.FORMAT: external.format_violations}
    monkeypatch.setattr(run, "EXTERNAL_TOOLS", tools)
    package = project_root / "pkg"
    unformatted = "x=1\n"
    (package / "a.py").write_text(unformatted)
    _init_repository(project_root)
    assert _main(project_root, "--write-baseline") == 0
    assert ("format", "pkg/a.py", "<module>") in _baseline_counts(project_root)
    reference_commit = _commit(project_root, "an unformatted file")

    # a.py splits into three pieces, each the same unformatted code
    (package / "a.py").unlink()
    parts = ("pkg/c.py", "pkg/d.py", "pkg/e.py")
    for part in parts:
        (project_root / part).write_text(unformatted)
    _rewrite(project_root)
    _write_moves(project_root, *_split_rows(*parts))
    assert _main(project_root, "--against", reference_commit) == 0
    capsys.readouterr()

    # an unformatted file that no row leads to is new
    (package / "f.py").write_text(unformatted)
    _rewrite(project_root)
    assert _main(project_root, "--against", reference_commit) == 1
    assert "pkg/f.py  format  <module>: 1" in capsys.readouterr().out


def test_a_countable_rule_split_three_ways_shares_one_allowance(
    project_root: Path, committed: str, capsys: pytest.CaptureFixture[str]
) -> None:
    package = project_root / "pkg"
    module_level = "x = 1  # TODO\n"
    (package / "a.py").write_text(module_level)
    _rewrite(project_root)
    reference_commit = _commit(project_root, "a module-level error")

    (package / "a.py").unlink()
    parts = ("pkg/c.py", "pkg/d.py", "pkg/e.py")
    for part in parts:
        (project_root / part).write_text("y = 2\n")
    (package / "c.py").write_text(module_level)
    _rewrite(project_root)
    _write_moves(project_root, *_split_rows(*parts))
    assert _main(project_root, "--against", reference_commit) == 0
    capsys.readouterr()

    for part in parts:
        (project_root / part).write_text(module_level)
    _rewrite(project_root)
    assert _main(project_root, "--against", reference_commit) == 1
    out = capsys.readouterr().out
    assert out.count("fixture-todo  <module>: 1 (the reference allows 0)") == 2


def test_cli_against_follows_a_rename_inside_a_file(
    project_root: Path, committed: str, capsys: pytest.CaptureFixture[str]
) -> None:
    path = project_root / "pkg" / "a.py"
    path.write_text(path.read_text().replace("def g", "def h"))
    _rewrite(project_root)
    assert _main(project_root, "--against", committed) == 1
    assert "pkg/a.py  fixture-todo  h" in capsys.readouterr().out
    _write_moves(project_root, ("pkg/a.py", "g", "pkg/a.py", "h"))
    assert _main(project_root, "--against", committed) == 0


def test_cli_against_follows_no_row_that_does_not_hold(
    project_root: Path, committed: str, capsys: pytest.CaptureFixture[str]
) -> None:
    package = project_root / "pkg"
    (package / "a.py").write_text("def f():\n    pass  # TODO\n")
    (package / "c.py").write_text("def h():\n    pass  # TODO\n")
    _rewrite(project_root)

    # the old definition was not at the reference: the row is inert
    _write_moves(project_root, ("pkg/a.py", "nope", "pkg/c.py", "h"))
    assert _main(project_root, "--against", committed) == 1
    out = capsys.readouterr().out
    assert "pkg/c.py  fixture-todo  h" in out
    assert "not followed" not in out

    # the new definition is not in the code
    _write_moves(project_root, ("pkg/a.py", "g", "pkg/c.py", "nope"))
    assert _main(project_root, "--against", committed) == 1
    assert "  pkg/a.py g -> pkg/c.py nope" in capsys.readouterr().out


def test_cli_prune_against_drops_the_rows_inert_at_the_reference(
    project_root: Path, committed: str, capsys: pytest.CaptureFixture[str]
) -> None:
    package = project_root / "pkg"
    (package / "a.py").rename(package / "moved.py")
    _rewrite(project_root)
    live = [
        ("pkg/a.py", "f", "pkg/moved.py", "f"),
        ("pkg/a.py", "g", "pkg/moved.py", "g"),
    ]
    inert = [
        ("pkg/never.py", "f", "pkg/moved.py", "f"),
        ("pkg/b.py", "never", "pkg/moved.py", "f"),
    ]
    _write_moves(project_root, *live, *inert)
    # a row whose old definition is not at the reference is not reported
    assert _main(project_root, "--against", committed) == 0
    assert "not followed" not in capsys.readouterr().out

    assert _main(project_root, "--prune") == 0
    assert len(read_moves(project_root).rows) == 4  # no reference given
    assert _main(project_root, "--prune", "--against", committed) == 0
    assert "Dropped 2 move rows" in capsys.readouterr().out
    assert [row.old + row.new for row in read_moves(project_root).rows] == [
        row[:2] + row[2:] for row in live
    ]

    # once the move has landed, its rows are inert too
    _commit(project_root, "the move")
    assert _main(project_root, "--prune", "--against", "HEAD") == 0
    assert read_moves(project_root).rows == ()
    assert _main(project_root, "--against", "HEAD") == 0


def test_cli_against_fails_while_the_baseline_holds_a_fixed_error(
    project_root: Path, committed: str, capsys: pytest.CaptureFixture[str]
) -> None:
    (project_root / "pkg" / "a.py").write_text("def f():\n    pass\n")
    assert _main(project_root) == 0
    assert _main(project_root, "--against", committed) == 1
    out = capsys.readouterr().out
    assert "allows 2 errors the code no longer has, in 2 entries" in out
    assert "--prune" in out
    _main(project_root, "--prune")
    assert _main(project_root, "--against", committed) == 0


# --- every check carries its documentation ---------------------------------


def test_every_registered_check_is_kebab_case_and_names_a_guide_rule() -> None:
    checks = {**core.MODULE_CHECKS, **core.PROJECT_CHECKS}
    assert checks
    for name, check in checks.items():
        assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name), name
        docstring = inspect.getdoc(check) or ""
        assert re.search(r"\b\d{1,2}\.\d\b", docstring), name


def _guide_sections() -> dict[str, str]:
    """Return the text of each numbered rule of STYLE.md, by its number."""
    text = (REPOSITORY / "STYLE.md").read_text(encoding="utf-8")
    parts = re.split(r"^\*\*(\d{1,2}\.\d) ", text, flags=re.MULTILINE)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def test_every_check_is_named_in_the_guide_under_its_rule() -> None:
    sections = _guide_sections()
    checks = {**core.MODULE_CHECKS, **core.PROJECT_CHECKS}
    for name, check in checks.items():
        docstring = inspect.getdoc(check) or ""
        cited = re.search(r"\b(\d{1,2}\.\d)\b", docstring)
        assert cited, name
        assert f"`{name}`" in sections.get(cited.group(1), ""), (
            f"STYLE.md {cited.group(1)} does not name {name}"
        )

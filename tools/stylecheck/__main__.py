"""The style checker's command line: ``python -m tools.stylecheck``.

Run from the repository root, it checks every file under the configured
trees, compares the errors with the baseline and prints the errors the
baseline does not cover, grouped by file. It exits with 0 when there are
none, 1 when there are (or when ``--against`` finds the baseline grew), and
2 when it cannot run (a bad option, rule, baseline file or git reference).
``--against`` is described in :mod:`tools.stylecheck.against`.

Options::

    --rule NAME          only this check, ruff:<CODE>, numpydoc:<CODE>,
                         or a whole tool (ruff, numpydoc, format); repeatable
    --paths PATH ...     only these files or folders
    --explain NAME       print what a check flags, and exit
    --show-warnings      list each warning, not only their counts
    --prune              lower the baseline to the current counts
    --write-baseline     create the baseline from the current errors
    --replace-baseline   with --write-baseline, overwrite an existing one
    --against REF        also fail if a baseline grew, or a setting
                         loosened, since git REF
    --allow-loosening    with --against, accept looser settings
    --root PATH          the repository root (default: the current folder)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from tools.stylecheck.against import check_against
from tools.stylecheck.baseline import (
    Baseline,
    BaselineError,
    compare,
    count_errors,
    prune,
    read_baseline,
    write_baseline,
)
from tools.stylecheck.core import Config, Project, Violation, load_project
from tools.stylecheck.external import ExternalToolError
from tools.stylecheck.history import HistoryError
from tools.stylecheck.pyright_baseline import PyrightBaselineError
from tools.stylecheck.report import format_against, format_report
from tools.stylecheck.run import (
    CheckFailedError,
    UnknownRuleError,
    check_names,
    explain,
    run_checks,
    validate_rule,
)
from tools.stylecheck.scope import Scope
from tools.stylecheck.settings import Record, current_record, differences


class OptionError(Exception):
    """Signal a command-line option the checker cannot use."""


#: The errors that stop a run with exit status 2 and a one-line message.
USAGE_ERRORS = (
    OptionError,
    BaselineError,
    CheckFailedError,
    ExternalToolError,
    HistoryError,
    PyrightBaselineError,
    UnknownRuleError,
)


class Arguments(argparse.Namespace):
    """Hold the parsed command line, with the type of each option."""

    root: str = "."
    rule: list[str] | None = None
    paths: list[str] | None = None
    explain: str | None = None
    show_warnings: bool = False
    prune: bool = False
    write_baseline: bool = False
    replace_baseline: bool = False
    against: str | None = None
    allow_loosening: bool = False


def main(*, argv: list[str] | None = None) -> int:
    """Run the checker as the command line asks.

    Parameters
    ----------
    argv
        The arguments after the program name; None reads ``sys.argv``.

    Returns
    -------
    int
        The exit status: 0 when nothing new fails, 1 when something does,
        2 when the checker cannot run.
    """
    parser = _parser()
    arguments = parser.parse_args(argv, namespace=Arguments())
    if arguments.replace_baseline and not arguments.write_baseline:
        parser.error("--replace-baseline goes with --write-baseline.")
    if arguments.allow_loosening and arguments.against is None:
        parser.error("--allow-loosening goes with --against.")
    root = Path(arguments.root).resolve()
    try:
        return _run(arguments, root)
    except USAGE_ERRORS as error:
        print(f"stylecheck: {error}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    """Return the parser of the command line's options."""
    parser = argparse.ArgumentParser(
        prog="python -m tools.stylecheck",
        description="Check LangSlice's code against STYLE.md.",
    )
    parser.add_argument("--root", default=".", help="the repository root")

    # what to check
    parser.add_argument(
        "--rule",
        action="append",
        metavar="NAME",
        help="only this check, ruff:<CODE>, numpydoc:<CODE> or a tool",
    )
    parser.add_argument(
        "--paths", nargs="+", metavar="PATH", help="only these files"
    )

    # what to print
    parser.add_argument(
        "--explain", metavar="NAME", help="print what a check flags"
    )
    parser.add_argument(
        "--show-warnings", action="store_true", help="list each warning"
    )

    # the baseline
    parser.add_argument(
        "--prune", action="store_true", help="lower the baseline's counts"
    )
    parser.add_argument(
        "--write-baseline", action="store_true", help="create the baseline"
    )
    parser.add_argument(
        "--replace-baseline",
        action="store_true",
        help="overwrite an existing baseline with --write-baseline",
    )

    # the baseline's history
    parser.add_argument(
        "--against", metavar="REF", help="fail if the baseline grew since REF"
    )
    parser.add_argument(
        "--allow-loosening",
        action="store_true",
        help="with --against, accept settings looser than at REF",
    )
    return parser


def _run(arguments: Arguments, root: Path) -> int:
    """Carry out one command line and return its exit status."""
    if arguments.explain is not None:
        print(explain(arguments.explain, root))
        return 0
    for rule in arguments.rule or []:
        validate_rule(rule)
    scope = Scope(
        rules=tuple(arguments.rule or ()),
        paths=_relative_paths(arguments.paths or [], root),
    )

    started = time.perf_counter()
    config = Config.load(root / "pyproject.toml")
    project = load_project(root, config)
    record = current_record(root, config, check_names())
    if arguments.write_baseline:
        replace = arguments.replace_baseline
        return _write(project, config, scope, record, replace=replace)
    violations = run_checks(project, config, scope)

    baseline_path = root / config.baseline
    baseline = _baseline(baseline_path, record, pruning=arguments.prune)
    if arguments.prune:
        counts = prune(baseline.counts, count_errors(violations), scope)
        baseline = Baseline(counts, record)
        write_baseline(baseline_path, baseline)
        print(f"Pruned {config.baseline}.")

    comparison = compare(violations, baseline.counts, scope)
    report = format_report(
        comparison, violations, show_warnings=arguments.show_warnings
    )
    print(report, end="")
    status = 1 if comparison.failing else 0
    if arguments.against is not None:
        status = max(status, _against(arguments, project, config, baseline))
    elif baseline_path.exists() and differences(baseline.record, record):
        print(
            f"{config.baseline} records other checks or settings than this "
            "run uses; --prune records the run's."
        )

    elapsed = time.perf_counter() - started
    print(f"Checked {len(project.modules)} files in {elapsed:.1f} s.")
    return status


def _baseline(path: Path, record: Record, *, pruning: bool) -> Baseline:
    """Return the baseline file, or an empty one when it is missing."""
    if path.exists():
        return read_baseline(path)
    if pruning:
        raise BaselineError(f"There is no baseline to prune at {path}.")
    print(f"No baseline at {path}: every error counts.")
    return Baseline({}, record)


def _write(
    project: Project,
    config: Config,
    scope: Scope,
    record: Record,
    *,
    replace: bool,
) -> int:
    """Create the baseline from the current errors and report its size."""
    path = project.root / config.baseline
    if not scope.is_whole:
        raise OptionError(
            "--write-baseline covers every rule and file; drop --rule and "
            "--paths."
        )
    if path.exists() and not replace:
        raise BaselineError(
            f"{config.baseline} exists; it may only shrink (--prune). Pass "
            "--replace-baseline to overwrite it on purpose."
        )

    violations: list[Violation] = run_checks(project, config, scope)
    counts = count_errors(violations)
    write_baseline(path, Baseline(counts, record))
    print(
        f"Wrote {len(counts)} entries covering {sum(counts.values())} "
        f"errors to {config.baseline}."
    )
    return 0


def _against(
    arguments: Arguments, project: Project, config: Config, baseline: Baseline
) -> int:
    """Compare with ``--against``'s commit; return 1 when the commit fails."""
    record = current_record(project.root, config, check_names())
    result = check_against(
        arguments.against or "HEAD",
        project,
        config,
        baseline,
        record=record,
        allow_loosening=arguments.allow_loosening,
    )
    print(format_against(result), end="")
    return int(result.fails)


def _relative_paths(paths: list[str], root: Path) -> tuple[str, ...]:
    """Return the ``--paths`` relative to the root, slash-separated."""
    relative: list[str] = []
    for path in paths:
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root):
            raise OptionError(f"{path} is outside the repository {root}.")
        name = resolved.relative_to(root).as_posix()
        if name == ".":
            return ()
        relative.append(name)
    return tuple(relative)


if __name__ == "__main__":
    sys.exit(main())

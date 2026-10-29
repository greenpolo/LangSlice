"""The checker's report: what fails, grouped by file, then the totals.

Each failing error is one line under its file: the line number, the rule,
the enclosing definition and the message. When a definition has more errors
of a rule than the baseline allows, every one of them is listed, since the
checker cannot tell which is new, with both counts. The totals say how many
errors the baseline covers, which entries can shrink and how many warnings
each check gave.
"""

from __future__ import annotations

from collections import Counter
from itertools import groupby

from tools.stylecheck.against import AgainstResult
from tools.stylecheck.baseline import Comparison, CountChange
from tools.stylecheck.core import Severity, Violation
from tools.stylecheck.scope import Key

#: What a refused loosening says: who decides it and how it is made.
LOOSENING_REFUSAL = (
    "Loosening a check is the designer's decision, never a fix: the "
    "designer makes it in a commit of its own and checks that commit with "
    "--against REF --allow-loosening. CI never passes that flag, so its "
    "lint job fails on that one commit by design."
)


def format_report(
    comparison: Comparison,
    violations: list[Violation],
    *,
    show_warnings: bool = False,
) -> str:
    """Write the report of one run.

    Parameters
    ----------
    comparison
        The run's errors set against the baseline.
    violations
        Every finding of the run, for the warning counts.
    show_warnings
        List each warning under its file as well as counting them.

    Returns
    -------
    str
        The report, ending with a newline.
    """
    lines = _failing_lines(comparison)
    warnings = [
        violation
        for violation in violations
        if violation.severity is Severity.WARNING
    ]
    if show_warnings and warnings:
        lines.append("Warnings:")
        lines.extend(_file_lines(warnings, {}))
    lines.extend(_summary_lines(comparison, warnings))
    return "\n".join(lines) + "\n"


def format_against(result: AgainstResult) -> str:
    """Write the report of ``--against``.

    Parameters
    ----------
    result
        Both baselines and the settings set against the reference.

    Returns
    -------
    str
        A paragraph per comparison, ending with a newline.
    """
    lines = _growth_lines(result)
    lines.extend(_slack_lines(result))
    lines.extend(_move_lines(result))
    lines.extend(_pyright_lines(result))
    if result.unrecorded:
        lines.append(
            "The baseline records other checks or settings than this run "
            f"uses: {', '.join(result.unrecorded)}. Record the run's with "
            "--prune, or with --write-baseline --replace-baseline when a "
            "stricter setting adds errors."
        )
    lines.extend(_loosening_lines(result))
    return "\n".join(lines) + "\n"


def _growth_lines(result: AgainstResult) -> list[str]:
    """Return the report lines of the checker's baseline against its past."""
    growth = result.growth
    accepted: list[str] = []
    if growth.n_adopted:
        accepted.append(
            f"{growth.n_adopted} errors entered with a new rule or a "
            "stricter setting"
        )
    if growth.n_moved:
        accepted.append(f"{growth.n_moved} errors moved with their files")
    note = f" ({'; '.join(accepted)})" if accepted else ""

    if not growth.changes:
        return [
            f"The baseline only shrank, compared with {result.reference}"
            f"{note}."
        ]

    lines = [f"The baseline grew, compared with {result.reference}{note}:"]
    for change in growth.changes:
        rule, path, symbol = change.key
        lines.append(
            f"  {path}  {rule}  {symbol}: {change.current} "
            f"(the reference allows {change.allowed})"
        )
    lines.append(
        "The baseline may only shrink: fix the new errors instead of "
        "listing them."
    )
    return lines


def _slack_lines(result: AgainstResult) -> list[str]:
    """Return the report lines of baseline entries above the run's counts."""
    if not result.slack:
        return []
    n_lower = sum(change.allowed - change.current for change in result.slack)
    return [
        f"The baseline allows {n_lower} errors the code no longer has, in "
        f"{len(result.slack)} entries. Run with --prune and commit the "
        "smaller baseline with the fixes, so a fixed error cannot come "
        "back unseen."
    ]


def _move_lines(result: AgainstResult) -> list[str]:
    """Return the report lines of the move map's rows it does not follow."""
    if not result.unhonoured:
        return []
    lines = [
        "These rows of the move map are not followed: their new definition "
        "is not in the code:"
    ]
    for row in result.unhonoured:
        lines.append(
            f"  {row.path} {row.symbol} -> {row.new_path} {row.new_symbol}"
        )
    return lines


def _pyright_lines(result: AgainstResult) -> list[str]:
    """Return the report lines of basedpyright's baseline."""
    if not result.pyright_changes:
        return [
            "basedpyright's baseline only shrank, compared with "
            f"{result.pyright_reference}."
        ]

    lines = [
        "basedpyright's baseline grew, compared with "
        f"{result.pyright_reference}:"
    ]
    for change in result.pyright_changes:
        lines.append(
            f"  {change.path}  {change.code}: {change.current} "
            f"(the reference allows {change.allowed})"
        )
    lines.append(
        "Fix the new type errors; `basedpyright --writebaseline` may not "
        "list them."
    )
    return lines


def _loosening_lines(result: AgainstResult) -> list[str]:
    """Return the report lines of the settings looser than the reference's."""
    if not result.loosening:
        return []
    lines = [f"Looser than in {result.reference}:"]
    for change in result.loosening:
        before = change.before or "unset"
        after = change.after or "unset"
        lines.append(f"  {change.name}: {before} -> {after} ({change.rule})")
    if result.loosening_allowed:
        lines.append("Accepted with --allow-loosening.")
    else:
        lines.append(LOOSENING_REFUSAL)
    return lines


def _failing_lines(comparison: Comparison) -> list[str]:
    """Return the report lines of the failing errors, by file."""
    allowed = {change.key: change for change in comparison.excess}
    return _file_lines(list(comparison.failing), allowed)


def _file_lines(
    violations: list[Violation], excess: dict[Key, CountChange]
) -> list[str]:
    """Return one heading per file and one line per violation under it."""
    lines: list[str] = []
    ordered = sorted(violations, key=lambda item: (item.path, item.line))
    for path, group in groupby(ordered, key=lambda item: item.path):
        lines.append(path)
        lines.extend(_violation_line(item, excess) for item in group)
        lines.append("")
    return lines


def _violation_line(
    violation: Violation, excess: dict[Key, CountChange]
) -> str:
    """Return the report line of one violation."""
    line = (
        f"  {violation.line:>5}  {violation.rule}  {violation.symbol}: "
        f"{violation.message}"
    )
    change = excess.get(violation.key)
    if change is not None and change.allowed:
        line += (
            f" [{change.current} here, the baseline allows {change.allowed}]"
        )
    return line


def _summary_lines(
    comparison: Comparison, warnings: list[Violation]
) -> list[str]:
    """Return the closing totals of the report."""
    n_failing = len(comparison.failing)
    n_files = len({violation.path for violation in comparison.failing})
    lines = [
        f"{n_failing} errors not covered by the baseline, in {n_files} files.",
        f"{comparison.n_covered} errors are covered by the baseline.",
    ]

    if comparison.shrinkable:
        n_lower = sum(
            change.allowed - change.current for change in comparison.shrinkable
        )
        lines.append(
            f"{len(comparison.shrinkable)} baseline entries can shrink, by "
            f"{n_lower} errors; run with --prune to lower them."
        )

    if warnings:
        counts = Counter(warning.rule for warning in warnings)
        listed = ", ".join(
            f"{rule} {count}" for rule, count in sorted(counts.items())
        )
        lines.append(
            f"Warnings, not counted by the baseline: {listed} "
            "(--show-warnings lists them)."
        )
    return lines

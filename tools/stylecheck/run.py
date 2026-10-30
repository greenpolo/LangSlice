"""Run every check and the external tools over the package.

The checks registered in :mod:`tools.stylecheck.rules` see the parsed
package; the external tools (ruff's extra rules, the formatter, numpydoc)
run on the same trees. Every finding comes back as a violation, so the
baseline counts them all the same way.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path

from tools.stylecheck import core
from tools.stylecheck import rules as rules  # registers every check
from tools.stylecheck.core import Config, Project, Violation
from tools.stylecheck.external import (
    FORMAT_EXPLANATION,
    PER_FILE_RULES,
    format_violations,
    ruff_explanation,
    ruff_violations,
)
from tools.stylecheck.model_text import (
    drop_stub_docstring_findings,
    find_stubs,
)
from tools.stylecheck.numpydoc_hook import (
    numpydoc_explanation,
    numpydoc_violations,
)
from tools.stylecheck.scope import Scope, Tool

#: Each external tool by the name its rules start with.
EXTERNAL_TOOLS: dict[Tool, Callable[[Project, Config], list[Violation]]] = {
    Tool.RUFF: ruff_violations,
    Tool.FORMAT: format_violations,
    Tool.NUMPYDOC: numpydoc_violations,
}


class CheckFailedError(Exception):
    """Signal that a check raised instead of reporting violations."""


class UnknownRuleError(Exception):
    """Signal a rule name that is no check, external code or tool."""


def run_checks(
    project: Project, config: Config, scope: Scope
) -> list[Violation]:
    """Run the checks and tools a scope selects and return their findings.

    Parameters
    ----------
    project
        The parsed package.
    config
        The checker's settings.
    scope
        The rules and files to report; every check still sees the whole
        package, so checks across modules stay correct.

    Returns
    -------
    list of Violation
        The findings in the scope, errors and warnings, sorted by file and
        line.

    Raises
    ------
    CheckFailedError
        When a check raises an exception.
    """
    violations = _checked(project, config, scope)
    for tool, run in EXTERNAL_TOOLS.items():
        if scope.selects_tool(tool):
            violations.extend(run(project, config))

    # the descriptions a model reads are judged by the text snapshot
    stubs = find_stubs(project)
    violations = drop_stub_docstring_findings(violations, stubs)

    selected = [
        violation
        for violation in violations
        if scope.selects_rule(violation.rule)
        and scope.selects_path(violation.path)
    ]
    return sorted(selected, key=_report_order)


def internal_violations(project: Project, config: Config) -> list[Violation]:
    """Run the checker's own checks, without the external tools.

    Parameters
    ----------
    project
        The parsed package.
    config
        The checker's settings, which may differ from the file's (the
        settings a baseline was written with, for ``--against``).

    Returns
    -------
    list of Violation
        Every finding of the registered checks, errors and warnings,
        except those the model-facing descriptions are excused from.

    Raises
    ------
    CheckFailedError
        When a check raises an exception.
    """
    violations = _checked(project, config, Scope())
    stubs = find_stubs(project)
    return drop_stub_docstring_findings(violations, stubs)


def _checked(
    project: Project, config: Config, scope: Scope
) -> list[Violation]:
    """Run the registered checks a scope selects, in name order."""
    violations: list[Violation] = []
    for name, check in sorted(core.MODULE_CHECKS.items()):
        if scope.selects_rule(name):
            violations.extend(_module_check(name, check, project, config))
    for name, check in sorted(core.PROJECT_CHECKS.items()):
        if scope.selects_rule(name):
            violations.extend(_project_check(name, check, project, config))
    return violations


def _report_order(violation: Violation) -> tuple[str, int, str]:
    """Return the sort key of a violation: file, line, then rule."""
    return (violation.path, violation.line, violation.rule)


def _module_check(
    name: str, check: core.ModuleCheck, project: Project, config: Config
) -> list[Violation]:
    """Run one module check on every module, naming it if it fails."""
    violations: list[Violation] = []
    for module in project.modules.values():
        try:
            violations.extend(check(module, project, config))
        except Exception as error:  # a check's own bug; stop with its name
            raise CheckFailedError(
                f"The check {name} failed on {module.relpath}: {error!r}."
            ) from error
    return violations


def _project_check(
    name: str, check: core.ProjectCheck, project: Project, config: Config
) -> list[Violation]:
    """Run one project check, naming it if it fails."""
    try:
        return list(check(project, config))
    except Exception as error:  # a check's own bug; stop with its name
        raise CheckFailedError(
            f"The check {name} failed: {error!r}."
        ) from error


def per_file_rules() -> frozenset[str]:
    """Return the rules that report at most one error per file.

    Returns
    -------
    frozenset of str
        The checks registered with ``per_file`` and the external tools'
        rules of that kind, such as ``format``.
    """
    return frozenset(core.PER_FILE_CHECKS) | PER_FILE_RULES


def check_names() -> list[str]:
    """List the registered checks and the external tools, sorted.

    Returns
    -------
    list of str
        Every check's name, then ``format``, ``numpydoc`` and ``ruff``.
    """
    names = sorted([*core.MODULE_CHECKS, *core.PROJECT_CHECKS])
    return [*names, *sorted(EXTERNAL_TOOLS)]


def validate_rule(name: str) -> None:
    """Refuse a rule name that names nothing the checker runs.

    Parameters
    ----------
    name
        A check's name, an external tool's name, or a code such as
        ``ruff:E501``.

    Raises
    ------
    UnknownRuleError
        When the name is none of those.
    """
    tool, separator, code = name.partition(":")
    if name in check_names():
        return
    if separator and tool in (Tool.RUFF, Tool.NUMPYDOC) and code:
        return
    raise UnknownRuleError(
        f"No check is named {name!r}; the checks are "
        f"{', '.join(check_names())}, and ruff:<CODE> or numpydoc:<CODE>."
    )


def explain(name: str, root: Path) -> str:
    """Return the description of one check, tool or external code.

    Parameters
    ----------
    name
        What ``--explain`` was given.
    root
        The repository root, where ruff runs and ``pyproject.toml`` is.

    Returns
    -------
    str
        A check's docstring, ruff's or numpydoc's own text for a code, or
        the rule set an external tool runs.

    Raises
    ------
    UnknownRuleError
        When the name names nothing the checker runs.
    """
    validate_rule(name)
    checks = {**core.MODULE_CHECKS, **core.PROJECT_CHECKS}
    if name in checks:
        return inspect.getdoc(checks[name]) or f"{name} has no docstring."

    tool, _, code = name.partition(":")
    if tool == Tool.RUFF:
        return ruff_explanation(code, root)
    if tool == Tool.NUMPYDOC:
        return numpydoc_explanation(code, root)
    return FORMAT_EXPLANATION

"""Run one check on a snippet of source, for the checker's own tests."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from tools.stylecheck import rules as rules  # registers every check
from tools.stylecheck.core import (
    MODULE_CHECKS,
    PROJECT_CHECKS,
    Config,
    Module,
    Project,
    Violation,
    module_name_from_path,
)

#: The repository root, whose ``pyproject.toml`` gives the default settings.
REPOSITORY = Path(__file__).resolve().parents[2]

#: Where a snippet lives unless a test says otherwise: a core module, which
#: every layer rule and boundary setting treats as ordinary package code.
SNIPPET_PATH = "src/langslice/core/example.py"


def check_source(
    source: str,
    rule: str,
    *,
    relpath: str = SNIPPET_PATH,
    others: dict[str, str] | None = None,
    config: Config | None = None,
) -> list[Violation]:
    """Run one check on a snippet and return what it finds in that snippet.

    Parameters
    ----------
    source
        The module's text; it is dedented first.
    rule
        The name of the check to run.
    relpath
        Where the snippet pretends to live, relative to the repository root.
    others
        More modules of the pretend package, as relative path to text, for
        checks that look across modules.
    config
        The settings; the repository's own when omitted, layers included.

    Returns
    -------
    list of Violation
        The findings located in the snippet itself, in order of line.
    """
    if config is None:
        config = Config.load(REPOSITORY / "pyproject.toml")
    project = snippet_project({relpath: source, **(others or {})})
    violations = _run(rule, project, config)
    in_snippet = [
        violation for violation in violations if violation.path == relpath
    ]
    return sorted(in_snippet, key=lambda violation: violation.line)


def flagged_lines(
    source: str,
    rule: str,
    *,
    relpath: str = SNIPPET_PATH,
    others: dict[str, str] | None = None,
    config: Config | None = None,
) -> list[int]:
    """Run one check on a snippet and return the lines it flags there.

    Parameters
    ----------
    source
        The module's text; it is dedented first.
    rule
        The name of the check to run.
    relpath
        Where the snippet pretends to live, relative to the repository root.
    others
        More modules of the pretend package, as relative path to text.
    config
        The settings; the repository's own when omitted.

    Returns
    -------
    list of int
        The line of each finding in the snippet, in order; a line flagged
        twice is listed twice.
    """
    violations = check_source(
        source, rule, relpath=relpath, others=others, config=config
    )
    return [violation.line for violation in violations]


def _run(rule: str, project: Project, config: Config) -> list[Violation]:
    """Run one registered check over every module of a project."""
    if rule not in MODULE_CHECKS:
        return list(PROJECT_CHECKS[rule](project, config))
    check = MODULE_CHECKS[rule]
    violations: list[Violation] = []
    for module in project.modules.values():
        violations.extend(check(module, project, config))
    return violations


def snippet_project(files: dict[str, str]) -> Project:
    """Parse snippets as the modules of a pretend package.

    Parameters
    ----------
    files
        Each module's text by the path it pretends to have, relative to
        the repository root; each text is dedented first.

    Returns
    -------
    Project
        The package index of those modules, rooted at the repository.
    """
    modules: dict[str, Module] = {}
    for relpath, text in files.items():
        source = textwrap.dedent(text)
        tree = ast.parse(source, filename=relpath)
        dotted = module_name_from_path(relpath)
        modules[dotted] = Module(Path(relpath), relpath, dotted, source, tree)
    return Project(REPOSITORY, modules)

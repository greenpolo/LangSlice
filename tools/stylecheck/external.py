"""The ruff rules and the formatter check, reported as the checker's own.

The main lint run (``ruff check .``) keeps the rules every file already
passes. The rules below are the rest of STYLE.md 12: each finding becomes a
violation named ``ruff:<CODE>``, placed at its enclosing definition, so it
is counted against the baseline like any other check. The formatter's check
gives one violation, ``format``, per file it would change. Both run on the
configured trees with ruff's settings from ``pyproject.toml`` (the numpy
docstring convention, the standard aliases), with two differences from the
main run: lines may be as long as the formatter's width (``line-length``,
79 columns), not the main run's ``max-line-length``, and the table
``per-file-ignores``, which holds the main run's own ignores, is emptied;
``extend-per-file-ignores`` holds the ignores both runs keep.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from tools.stylecheck.core import Config, Project, Violation
from tools.stylecheck.jsondata import (
    as_int,
    as_list,
    as_table,
    load_json,
    load_toml,
)

#: The ruff rules of STYLE.md 12 that the main lint run leaves out: line
#: length and import order (they need the baseline at 79 columns), the
#: docstring layout rules numpydoc does not check, and the rules of 2.6,
#: 5.5, 7.4, 8.3 and 10.1. B006 and B904 are on in the main run already.
RUFF_SELECT = (
    "E501",
    "I001",
    "D2",
    "D3",
    "D4",
    "BLE",
    "S110",
    "SLF001",
    "ERA001",
    "ICN001",
    "PGH003",
    "PTH",
    "N",
)

#: Rules in those families that another tool owns or the guide allows:
#: numpydoc owns the summary (SS02, SS03, SS05, SS06 against D205, D400,
#: D401, D403, D415) and an empty docstring (GL08 against D419); 4.3 allows
#: a capital for a mathematical symbol (N803, N806); 8.1 names the one
#: exception ``Refused`` (N818).
RUFF_IGNORE = (
    "D205",
    "D400",
    "D401",
    "D403",
    "D415",
    "D419",
    "N803",
    "N806",
    "N818",
)

#: ruff's width when ``[tool.ruff]`` gives no ``line-length``.
DEFAULT_LINE_LENGTH = 88

#: What ``--explain format`` prints.
FORMAT_EXPLANATION = (
    "STYLE.md 1.1: every file is formatted by `ruff format` at 79 columns. "
    "The check reports one violation per file the formatter would change; "
    "run `ruff format <file>` to fix it."
)


class ExternalToolError(Exception):
    """Signal that ruff could not run or printed something unreadable."""


@dataclass(frozen=True)
class Diagnostic:
    """Hold one entry of ruff's JSON report.

    Parameters
    ----------
    code
        The rule's code, such as ``E501``; ``syntax`` for a parse error.
    path
        The file relative to the repository root, with forward slashes.
    line
        The line the finding starts on.
    message
        Ruff's sentence about the finding.
    """

    code: str
    path: str
    line: int
    message: str


def ruff_violations(project: Project, config: Config) -> list[Violation]:
    """Run ruff's extra rules and return their findings as violations.

    Parameters
    ----------
    project
        The parsed package; its modules name each finding's definition.
    config
        The settings; ``config.paths`` lists the trees ruff checks.

    Returns
    -------
    list of Violation
        One error per ruff finding, named ``ruff:<CODE>``.

    Raises
    ------
    ExternalToolError
        When ruff fails or prints something that is not its JSON report.
    """
    # the formatter's width, and the main run's own ignores left out
    width = _line_length(project.root)
    overrides = [
        "--config",
        f"lint.pycodestyle.max-line-length = {width}",
        "--config",
        "lint.per-file-ignores = {}",
    ]

    command = [
        "check",
        "--no-cache",
        "--exit-zero",
        *overrides,
        "--output-format",
        "json",
        "--select",
        ",".join(RUFF_SELECT),
        "--ignore",
        ",".join(RUFF_IGNORE),
        *config.paths,
    ]
    diagnostics = _run_ruff(project.root, command, ok_codes=(0,))

    modules = {module.relpath: module for module in project.modules.values()}
    violations: list[Violation] = []
    for diagnostic in diagnostics:
        symbol = "<module>"
        if diagnostic.path in modules:
            symbol = modules[diagnostic.path].symbol_at(diagnostic.line)
        violation = Violation(
            f"ruff:{diagnostic.code}",
            diagnostic.path,
            symbol,
            diagnostic.line,
            diagnostic.message,
        )
        violations.append(violation)
    return violations


def format_violations(project: Project, config: Config) -> list[Violation]:
    """Return one violation for each file the formatter would change.

    Parameters
    ----------
    project
        The parsed package; it gives the repository root.
    config
        The settings; ``config.paths`` lists the trees to check.

    Returns
    -------
    list of Violation
        One ``format`` error per unformatted file, at the module level.

    Raises
    ------
    ExternalToolError
        When the formatter fails or prints something unreadable.
    """
    command = [
        "format",
        "--check",
        "--no-cache",
        "--output-format",
        "json",
        *config.paths,
    ]
    diagnostics = _run_ruff(project.root, command, ok_codes=(0, 1))

    paths = sorted({diagnostic.path for diagnostic in diagnostics})
    message = (
        "The file is not formatted; run `ruff format` on it (79 columns)."
    )
    return [
        Violation("format", path, "<module>", 1, message) for path in paths
    ]


def ruff_explanation(code: str, root: Path) -> str:
    """Return ruff's own description of one rule, or the rules run.

    Parameters
    ----------
    code
        A ruff rule code such as ``E501``; empty for the whole rule set.
    root
        The repository root, where ruff runs.

    Returns
    -------
    str
        The text ``ruff rule <code>`` prints, or the selected and ignored
        rules when ``code`` is empty.

    Raises
    ------
    ExternalToolError
        When ruff knows no rule of that code.
    """
    if not code:
        return (
            f"ruff check with {', '.join(RUFF_SELECT)}, except "
            f"{', '.join(RUFF_IGNORE)}; each finding is ruff:<CODE>."
        )
    done = subprocess.run(
        [sys.executable, "-m", "ruff", "rule", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=root,
    )
    if done.returncode != 0:
        raise ExternalToolError(f"ruff knows no rule {code!r}.")
    return done.stdout.strip()


def _line_length(root: Path) -> int:
    """Return the formatter's width from ``[tool.ruff]``, or ruff's default."""
    document = load_toml((root / "pyproject.toml").read_text(encoding="utf-8"))
    tool = as_table(document.get("tool")) or {}
    ruff = as_table(tool.get("ruff")) or {}
    return as_int(ruff.get("line-length")) or DEFAULT_LINE_LENGTH


def _run_ruff(
    root: Path, arguments: list[str], *, ok_codes: tuple[int, ...]
) -> list[Diagnostic]:
    """Run ruff in ``root`` and return the entries of its JSON report."""
    done = subprocess.run(
        [sys.executable, "-m", "ruff", *arguments],
        capture_output=True,
        text=True,
        check=False,
        cwd=root,
    )
    if done.returncode not in ok_codes:
        raise ExternalToolError(
            f"ruff {arguments[0]} exited with {done.returncode}: "
            f"{done.stderr.strip()}"
        )
    if not done.stdout.strip():
        return []

    try:
        document = as_list(load_json(done.stdout))
    except json.JSONDecodeError as error:
        raise ExternalToolError(f"ruff printed no JSON: {error}.") from error
    if document is None:
        raise ExternalToolError("ruff's JSON report is not a list.")
    return [_diagnostic(item, root) for item in document]


def _diagnostic(item: object, root: Path) -> Diagnostic:
    """Read one entry of ruff's JSON report."""
    entry = as_table(item)
    if entry is None:
        raise ExternalToolError("ruff's JSON report has a non-object entry.")
    path = Path(str(entry.get("filename", "")))
    if path.is_absolute():
        path = path.resolve().relative_to(root.resolve())

    line = 1
    location = as_table(entry.get("location"))
    if location is not None:
        row = location.get("row")
        if isinstance(row, int):
            line = row

    code = str(entry.get("code") or "syntax")
    message = _sentence(str(entry.get("message", "")))
    return Diagnostic(code, path.as_posix(), line, message)


def _sentence(text: str) -> str:
    """Return a tool's message with a capital and a final full stop."""
    text = text.strip()
    if not text:
        return "Ruff gives no message."
    text = text[0].upper() + text[1:]
    if text.endswith((".", "?", "!")):
        return text
    return f"{text}."

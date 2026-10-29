"""The report level each basedpyright checking mode gives each rule.

A ``report`` key that ``[tool.basedpyright]`` does not set reports at its
mode's level, so a key added at a level is a tightening when the level is
at least that default, and a key removed is one when the default is at
least the level it had. basedpyright keeps its modes' rule sets in its
bundled language server (``dist/pyright.js``, one function per mode, one
``reportName: 'level'`` line per rule); this module reads them from there.
When basedpyright is not installed, or its bundle has another shape, no
default is known.
"""

from __future__ import annotations

import functools
import importlib.util
import re
from pathlib import Path

#: The start of a mode's rule set in the bundle, with the mode's name.
_RULE_SET_START = re.compile(
    r"^(?:function get(\w+)DiagnosticRuleSet\(\) \{"
    r"|const get(\w+)DiagnosticRuleSet = \(\) => \(\{)$"
)

#: One rule and its level inside a rule set.
_RULE_LEVEL = re.compile(r"^\s+(report\w+): '(\w+)',")


def default_level(mode: str | None, rule: str) -> str | None:
    """Return the level a checking mode gives a rule when no key sets it.

    Parameters
    ----------
    mode
        The ``typeCheckingMode``, such as ``strict``; None when unset.
    rule
        The rule, such as ``reportAny``.

    Returns
    -------
    str or None
        The level, such as ``error``; None when the mode is unset or the
        default is not known.
    """
    levels = mode_levels()
    if mode is None or mode not in levels:
        return None
    return levels[mode].get(rule)


@functools.cache
def mode_levels() -> dict[str, dict[str, str]]:
    """Return each checking mode's level of each rule, from basedpyright.

    Returns
    -------
    dict of str to dict of str to str
        For each mode, by its lowercase name, each rule's level; empty
        when the bundle is not found.
    """
    bundle = _bundle()
    if bundle is None:
        return {}
    return parse_mode_levels(bundle.read_text(encoding="utf-8"))


def parse_mode_levels(text: str) -> dict[str, dict[str, str]]:
    """Read the rule sets of basedpyright's bundled language server.

    Parameters
    ----------
    text
        The text of ``dist/pyright.js``.

    Returns
    -------
    dict of str to dict of str to str
        For each mode, by its lowercase name, each rule's level.
    """
    levels: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for line in text.splitlines():
        start = _RULE_SET_START.match(line)
        if start is not None:
            mode = (start.group(1) or start.group(2)).lower()
            current = levels.setdefault(mode, {})
        elif line.startswith("}"):
            current = None
        elif current is not None:
            rule = _RULE_LEVEL.match(line)
            if rule is not None:
                current[rule.group(1)] = rule.group(2)
    return levels


def _bundle() -> Path | None:
    """Return basedpyright's bundled language server, if it is installed."""
    spec = importlib.util.find_spec("basedpyright")
    if spec is None or spec.origin is None:
        return None
    bundle = Path(spec.origin).parent / "dist" / "pyright.js"
    if not bundle.exists():
        return None
    return bundle

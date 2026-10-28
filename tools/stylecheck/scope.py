"""Which rules and files one run of the checker covers.

A run covers every rule and every file unless the command line names some.
A rule is named by its check's name (``cognitive-complexity``), by an
external tool's code (``ruff:E501``) or by a whole external tool (``ruff``).
A path names a file or a folder, relative to the repository root.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

#: A baseline key: the rule, the file and the enclosing definition.
Key = tuple[str, str, str]


class Tool(enum.StrEnum):
    """Name an external tool; a tool's codes are named ``<tool>:<CODE>``."""

    RUFF = "ruff"
    NUMPYDOC = "numpydoc"
    FORMAT = "format"


@dataclass(frozen=True)
class Scope:
    """Select the rules and files a run covers.

    Parameters
    ----------
    rules
        Rule names, external tool codes or external tool names; empty means
        every rule.
    paths
        Files or folders relative to the repository root, with forward
        slashes; empty means every file.
    """

    rules: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()

    @property
    def is_whole(self) -> bool:
        """Return whether the run covers every rule and every file."""
        return not self.rules and not self.paths

    def selects_rule(self, rule: str) -> bool:
        """Return whether a violation's rule is in the run.

        Parameters
        ----------
        rule
            A check's name or an external code such as ``ruff:E501``.

        Returns
        -------
        bool
            True when no rules were named, when this rule was named, or
            when the external tool it belongs to was named.
        """
        if not self.rules:
            return True
        tool = rule.partition(":")[0]
        return rule in self.rules or tool in self.rules

    def selects_tool(self, tool: str) -> bool:
        """Return whether any rule of an external tool is in the run.

        Parameters
        ----------
        tool
            The external tool's name, such as ``ruff`` or ``numpydoc``.

        Returns
        -------
        bool
            True when no rules were named, when the tool was named, or when
            one of its codes was named.
        """
        if not self.rules:
            return True
        prefix = f"{tool}:"
        for rule in self.rules:
            if rule == tool or rule.startswith(prefix):
                return True
        return False

    def selects_path(self, path: str) -> bool:
        """Return whether a file is in the run.

        Parameters
        ----------
        path
            The file relative to the repository root, with forward slashes.

        Returns
        -------
        bool
            True when no paths were named, or the file is one of them or
            lies under one of them.
        """
        if not self.paths:
            return True
        for selected in self.paths:
            if path == selected or path.startswith(f"{selected}/"):
                return True
        return False

    def contains(self, key: Key) -> bool:
        """Return whether a baseline key falls inside the run.

        Parameters
        ----------
        key
            The rule, file and definition of a baseline entry.

        Returns
        -------
        bool
            True when both the key's rule and its file are in the run.
        """
        rule, path, _ = key
        return self.selects_rule(rule) and self.selects_path(path)

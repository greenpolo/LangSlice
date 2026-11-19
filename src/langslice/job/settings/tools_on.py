"""Which tools a job has on, for the text a model reads.

A sentence that names a tool is written only where the tool is on: the code
that words a text asks a :class:`ToolsOn` and builds the text from the pieces
that stand, as plain Python. A name is off when it is in the job's list of
switched-off tools (:func:`langslice.job.settings.tool_table.is_off`), so a
plugin's tool is asked the same way as a native one.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass

from langslice.job.settings.tool_table import is_off


@dataclass(frozen=True)
class ToolsOn:
    """The tools that are off for a job; every other tool is on."""

    off: frozenset[str] = frozenset()

    def has(self, name: str) -> bool:
        """Whether the tool *name* is on."""
        return not is_off(self.off, name)

    def has_all(self, names: Collection[str]) -> bool:
        """Whether every tool of *names* is on."""
        return all(self.has(name) for name in names)

    def has_any(self, names: Collection[str]) -> bool:
        """Whether at least one tool of *names* is on."""
        return any(self.has(name) for name in names)

    def say(self, name: str, text: str, otherwise: str = "") -> str:
        """*text* when the tool *name* is on, else *otherwise*."""
        return text if self.has(name) else otherwise

    def say_all(self, names: Collection[str], text: str, otherwise: str = "") -> str:
        """*text* when every tool of *names* is on, else *otherwise*."""
        return text if self.has_all(names) else otherwise

    def say_any(self, names: Collection[str], text: str, otherwise: str = "") -> str:
        """*text* when at least one tool of *names* is on, else *otherwise*."""
        return text if self.has_any(names) else otherwise

    def listed(self, items: Iterable[tuple[str | None, str]], last: str = "and") -> str:
        """The items whose tool is on, as "a, b and c" (*last* ``"or"``: "a, b
        or c"); an item whose tool is None or empty always stands.

        Parameters
        ----------
        items
            Pairs of the tool an item needs and the item's text.
        last
            The word before the last item.

        Returns
        -------
        str
            The standing items, empty when none stands.
        """
        shown = [text for name, text in items if not name or self.has(name)]
        if len(shown) < 2:
            return "".join(shown)
        return f"{', '.join(shown[:-1])} {last} {shown[-1]}"

"""Matching the file names of an imported registration to the sections."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Sequence

#: QuickNII / DeepSlice section numbers: ``_s`` then digits (DeepSlice's
#: ``number_sections``, non-legacy).
SECTION_NUMBER = re.compile(r"_s(\d+)")


class AmbiguousMatch(ValueError):
    """The file's entries cannot be matched to the job's sections one to one."""


# --- matching entries to section files -----------------------------------------------


def base_name(filename: str) -> str:
    """The last path component of *filename* (``/`` and ``\\`` both separate)."""
    return re.split(r"[\\/]", filename)[-1]


def section_number(filename: str) -> int | None:
    """The one ``_sNNN`` section number in *filename*'s base name, else None."""
    found = SECTION_NUMBER.findall(os.path.splitext(base_name(filename))[0])
    return int(found[0]) if len(found) == 1 else None


def match_sections(names: Sequence[str], section_ids: Iterable[str],
                   ) -> dict[int, tuple[str, str]]:
    """Entry index -> ``(section id, rule)`` (see the module text for the
    rules). :class:`AmbiguousMatch` when an entry matches two sections under
    its rule, or two entries match one section."""
    ids = list(section_ids)

    def stem(name: str) -> str:
        return os.path.splitext(base_name(name))[0].casefold()

    rules = (
        ("exact", lambda name, sid: base_name(name) == sid),
        ("stem", lambda name, sid: stem(name) == stem(sid)),
        ("section number", lambda name, sid: section_number(name) is not None
         and section_number(name) == section_number(sid)),
    )
    matched: dict[int, tuple[str, str]] = {}
    for index, name in enumerate(names):
        for rule, test in rules:
            hits = [sid for sid in ids if test(name, sid)]
            if len(hits) > 1:
                raise AmbiguousMatch(f"{name!r} matches several sections by {rule}: {hits}")
            if hits:
                matched[index] = (hits[0], rule)
                break
    taken: dict[str, list[str]] = {}
    for index, (sid, _rule) in matched.items():
        taken.setdefault(sid, []).append(names[index])
    doubles = {sid: found for sid, found in taken.items() if len(found) > 1}
    if doubles:
        raise AmbiguousMatch("several entries match one section: " + "; ".join(
            f"{sid}: {found}" for sid, found in sorted(doubles.items())))
    return matched

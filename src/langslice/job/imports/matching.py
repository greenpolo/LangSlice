"""Matching the file names of an imported registration to the sections."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from langslice.core.exports.quint_images import base_name, section_number

COPY_NUMBER = re.compile(r"_s\d+$")


class AmbiguousMatch(ValueError):
    """The file's entries cannot be matched to the job's sections one to one."""


def _stem(name: str) -> str:
    return Path(base_name(name)).stem.casefold()


def _same_name(name: str, section_id: str) -> bool:
    return base_name(name) == section_id


def _same_stem(name: str, section_id: str) -> bool:
    return _stem(name) == _stem(section_id)


def _is_copy(name: str, section_id: str) -> bool:
    return COPY_NUMBER.sub("", _stem(name)) == _stem(section_id)


def _same_number(name: str, section_id: str) -> bool:
    number = section_number(name)
    return number is not None and number == section_number(section_id)


RULES = (
    ("exact", _same_name),
    ("stem", _same_stem),
    ("export copy", _is_copy),
    ("section number", _same_number),
)


def match_sections(
    names: Sequence[str],
    section_ids: Iterable[str],
) -> dict[int, tuple[str, str]]:
    """Entry index -> ``(section id, rule)`` (see the module text for the
    rules). :class:`AmbiguousMatch` when an entry matches two sections under
    its rule, or two entries match one section."""
    ids = list(section_ids)
    matched: dict[int, tuple[str, str]] = {}
    for index, name in enumerate(names):
        for rule, test in RULES:
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
        raise AmbiguousMatch(
            "several entries match one section: "
            + "; ".join(f"{sid}: {found}" for sid, found in sorted(doubles.items()))
        )
    return matched

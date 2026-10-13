"""What the agent may not change: the properties a job freezes, per section.

A section has five properties a registration sets:

- ``position``: where it sits along the slicing axis (millimetres);
- ``angles``: the cutting angles of its atlas plane (pitch and yaw);
- ``orientation``: its flip and quarter turn, which go together (one choice
  of how the picture sits before the in-plane alignment);
- ``affine``: the continuous in-plane transform, the six numbers;
- ``deformation``: the deformation on top of the linear placement.

``JobSpec.frozen`` names, per section, the properties that keep the value the
job starts with: ``{"<filename>": ["orientation", "affine"], "*": ["angles"]}``
(``"*"`` is every section; a section's own list and ``"*"`` add up). The value
a frozen property keeps is the one ``JobSpec.inputs`` supplies; one that
supplies none keeps the property's "none": the flat plane (angles), no flip
and no turn (orientation), the host's identity transform (affine), no
deformation. A position has no "none", so a frozen position must be supplied.

A tool that writes only properties frozen on every section is not offered,
and a task whose properties are all frozen on every section is off
(:mod:`langslice.core.spec`, :meth:`~langslice.core.spec.JobSpec.has`). A task
switched off freezes its properties on every section as well (:func:`derived`),
so "the host supplied this" is one rule. This module imports nothing above the
core.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from typing import Any

#: Every property, in the order they are listed everywhere.
PROPERTIES: tuple[str, ...] = ("position", "angles", "orientation", "affine", "deformation")

#: The key of ``frozen`` that names every section.
EVERY = "*"

#: The code of a write refused because it would change a frozen property.
FROZEN = "FROZEN"

#: The properties each task writes (``JobSpec.tasks``).
TASK_PROPERTIES: dict[str, tuple[str, ...]] = {
    "position": ("position", "angles"),
    "transform": ("orientation", "affine"),
    "nonlinear": ("deformation",),
}

#: How a property is said to the agent.
WORDS: dict[str, str] = {
    "position": "position",
    "angles": "cutting angles",
    "orientation": "orientation (flip and quarter turn)",
    "affine": "affine (in-plane transform)",
    "deformation": "deformation",
}

#: How many sections a statement lists by name before it counts them.
LISTED_SECTIONS = 8


def _ordered(names: Iterable[str]) -> tuple[str, ...]:
    held = set(names)
    return tuple(name for name in PROPERTIES if name in held)


def clean(value: Any) -> dict[str, list[str]]:
    """*value* as the canonical ``frozen`` mapping: filename (or ``"*"``) ->
    its properties in :data:`PROPERTIES` order, empty lists dropped.
    ``ValueError`` naming what is wrong."""
    if value is None:
        return {}
    shape = ('frozen must map a section filename, or "*" for every section, to a list '
             f"of properties, any of {list(PROPERTIES)}")
    if not isinstance(value, Mapping):
        raise ValueError(shape)
    out: dict[str, list[str]] = {}
    for key, names in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(shape)
        if not isinstance(names, (list, tuple)) or not all(isinstance(n, str) for n in names):
            raise ValueError(f"frozen[{key!r}] must be a list of properties, any of "
                             f"{list(PROPERTIES)}")
        strange = sorted({n for n in names if n not in PROPERTIES})
        if strange:
            raise ValueError(f"frozen[{key!r}] names unknown properties {strange}; the "
                             f"properties are {list(PROPERTIES)}")
        if names:
            out[key] = list(_ordered(names))
    return out


def for_section(frozen: Mapping[str, Collection[str]], section_id: str) -> frozenset[str]:
    """The properties frozen on one section: its own and every section's."""
    return frozenset(frozen.get(section_id, ())) | frozenset(frozen.get(EVERY, ()))


def everywhere(frozen: Mapping[str, Collection[str]]) -> frozenset[str]:
    """The properties frozen on every section."""
    return frozenset(frozen.get(EVERY, ()))


def sections_of(frozen: Mapping[str, Collection[str]], section_ids: Iterable[str],
                ) -> dict[str, list[str]]:
    """``{section id: its frozen properties}`` for the sections of
    *section_ids* that have any, in that order."""
    out: dict[str, list[str]] = {}
    for section_id in section_ids:
        held = for_section(frozen, section_id)
        if held:
            out[section_id] = list(_ordered(held))
    return out


def unknown_names(frozen: Mapping[str, Collection[str]], section_ids: Collection[str],
                  ) -> list[str]:
    """The filenames *frozen* names that are no section."""
    return sorted(name for name in frozen if name != EVERY and name not in section_ids)


def groups(frozen: Mapping[str, Collection[str]], section_ids: list[str],
           ) -> list[tuple[tuple[str, ...], list[str]]]:
    """The sections grouped by the properties frozen on them, in the order
    the groups first appear: ``[(properties, section ids)]``; sections with
    nothing frozen are left out."""
    grouped: dict[tuple[str, ...], list[str]] = {}
    for section_id, names in sections_of(frozen, section_ids).items():
        grouped.setdefault(tuple(names), []).append(section_id)
    return list(grouped.items())


def words(properties: Iterable[str]) -> str:
    """Properties said to the agent: ``a, b and c``."""
    said = [WORDS[name] for name in properties]
    if len(said) > 1:
        return ", ".join(said[:-1]) + " and " + said[-1]
    return said[0] if said else ""


def statement(frozen: Mapping[str, Collection[str]], section_ids: list[str]) -> str | None:
    """The sentence the agent reads about what is frozen, or None when
    nothing is. Sections sharing a freeze are one clause: ``every section``
    when the freeze covers the stack, their filenames when there are few, and
    their count (the status rows carry ``frozen``) when there are many."""
    clauses: list[str] = []
    total = len(section_ids)
    for properties, ids in groups(frozen, section_ids):
        if len(ids) == total:
            who = "every section"
        elif len(ids) <= LISTED_SECTIONS:
            who = ", ".join(ids)
        else:
            who = f"{len(ids)} of the {total} sections (status marks them in `frozen`)"
        clauses.append(f"{words(properties)} of {who}")
    if not clauses:
        return None
    return ("- Frozen by the user: " + "; ".join(clauses) + ". These keep the values the "
            "job gave them; a write that would change one is refused with FROZEN.")


def message(properties: Collection[str], ids: list[str], total: int) -> str:
    """The sentence of a refusal: what is frozen, on which sections."""
    who = "every section" if total and len(ids) == total else ", ".join(ids)
    return (f"Frozen on {who}: {words(_ordered(properties))}. The job keeps the value it "
            "gave there.")


def derived(tasks_on: Collection[str], *, angles_free: bool) -> frozenset[str]:
    """The properties frozen on every section because their task is off:
    *tasks_on* the tasks that are on, *angles_free* whether the agent may
    still set the cutting angles (``transform.angles`` with Linear on)."""
    held: set[str] = set()
    for task, properties in TASK_PROPERTIES.items():
        if task not in tasks_on:
            held.update(properties)
    if angles_free:
        held.discard("angles")
    return frozenset(held)

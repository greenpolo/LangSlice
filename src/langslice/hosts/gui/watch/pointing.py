"""What the agent's words point at, so the viewer can show it.

Nearly three in four of an agent's messages and reasoning summaries point at
something the stage can show: sections by name ("B_09", "s0426", "slices
62-71"), atlas regions ("the hippocampus", "pontine grey", "CA1"), places
along the cutting axis ("9.4 mm") or the cutting angles ("pitch 7.3°").
:func:`pointed` reads one text against the job's own sections and atlas and
returns them; the page rings the sections, outlines the regions on the
section pictures and marks the places and angles on the axis while those
words are shown.

A section is named by its file name's stem or by a piece of it that no other
section shares (one to three of its ``_``, ``-`` or ``.`` separated parts
holding a digit: ``s0426``, ``B_09``), or as "slice 12", "sections 2, 3 and
62-71": the sections whose file names carry that number ("slice12"), else
the sections at those places in the stack, counted from 1. A region is named by its full name
or its acronym as the atlas lists them, or by a common word for it
(:data:`COMMON_WORDS`) when the atlas has that region.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import PurePosixPath
from typing import Any

from langslice.core.atlas.tree import structure_table

COMMON_WORDS: dict[str, tuple[str, ...]] = {
    "hippocampus": ("HIP",),
    "hippocampal": ("HIP",),
    "dentate gyrus": ("DG",),
    "cortex": ("Isocortex",),
    "cortical": ("Isocortex",),
    "neocortex": ("Isocortex",),
    "ventricle": ("VS",),
    "ventricles": ("VS",),
    "lateral ventricle": ("VL",),
    "third ventricle": ("V3",),
    "fourth ventricle": ("V4",),
    "aqueduct": ("AQ",),
    "cerebellum": ("CB",),
    "cerebellar": ("CB",),
    "striatum": ("STR",),
    "caudoputamen": ("CP",),
    "thalamus": ("TH",),
    "hypothalamus": ("HY",),
    "midbrain": ("MB",),
    "pons": ("P",),
    "pontine": ("P",),
    "medulla": ("MY",),
    "olfactory bulb": ("MOB",),
    "olfactory bulbs": ("MOB",),
    "corpus callosum": ("ccb", "ccg", "ccs"),
    "callosum": ("ccb", "ccg", "ccs"),
    "fornix": ("fx",),
    "fimbria": ("fi",),
    "septum": ("LS",),
    "habenula": ("MH", "LH"),
    "amygdala": ("CTXsp",),
    "superior colliculus": ("SCs", "SCm"),
    "inferior colliculus": ("IC",),
    "white matter": ("fiber tracts",),
}

STOP_ACRONYMS = frozenset(
    {
        "A",
        "I",
        "AN",
        "AS",
        "AT",
        "BE",
        "DO",
        "GO",
        "IN",
        "IS",
        "IT",
        "NO",
        "OF",
        "ON",
        "OR",
        "SO",
        "TO",
        "UP",
        "US",
        "ME",
        "MY",
        "OK",
        "VS",
    }
)
MIN_NAME = 5
MIN_KEY = 3
MAX_REGIONS = 6
MAX_SECTIONS = 24
SEPARATORS = re.compile(r"[_\-.]")
PARAGRAPHS = re.compile(r"\n\s*\n")
NUMBER_RUN = r"\d{1,3}(?:\s*(?:-|–|to)\s*\d{1,3})?"
ORDER_WORDS = re.compile(
    rf"\b(?:slices?|sections?)\s+({NUMBER_RUN}(?:\s*(?:,|and|&)\s*{NUMBER_RUN})*)",
    re.IGNORECASE,
)
NUMBERED_NAME = re.compile(r"(?:slice|section|sec)[_\-]?0*(\d+)", re.IGNORECASE)
RANGE = re.compile(r"(\d{1,3})(?:\s*(?:-|–|to)\s*(\d{1,3}))?")
MILLIMETRES = re.compile(r"(?<![\w.])(-?\d{1,2}(?:\.\d{1,3})?)\s?mm\b")
PITCH = re.compile(r"\bpitch\D{0,12}?(-?\d{1,2}(?:\.\d{1,2})?)\s?°?", re.IGNORECASE)
YAW = re.compile(r"\byaw\D{0,12}?(-?\d{1,2}(?:\.\d{1,2})?)\s?°?", re.IGNORECASE)
ANGLE_WORDS = re.compile(
    r"\b(?:pitch|yaw|oblique|obliquely|tilt|tilted|cutting angles?|cutting plane)\b", re.IGNORECASE
)


@dataclass(frozen=True)
class Pointed:
    """What one text points at: sections (ids, stack order), region acronyms,
    places in mm and the cutting angles it names (None for an angle it does
    not); *angled* when it speaks of the cutting angles at all."""

    sections: list[str]
    regions: list[str]
    positions_mm: list[float]
    pitch: float | None
    yaw: float | None
    angled: bool


def first_paragraph(text: str) -> str:
    """The words the viewer's bubble shows of a message: its first paragraph."""
    return next((block for block in PARAGRAPHS.split(text) if block.strip()), "")


def pointed(text: str, section_ids: Sequence[str], atlas: Any) -> Pointed:
    """Read what *text* points at among a job's sections and its atlas.

    Parameters
    ----------
    text
        The agent's words, as the viewer shows them.
    section_ids
        The job's sections in stack order.
    atlas
        The job's atlas (BrainGlobe's: its ``structures``); None reads no
        regions.

    Returns
    -------
    Pointed
        The sections in stack order (at most :data:`MAX_SECTIONS`), the
        regions in the order the words name them (at most
        :data:`MAX_REGIONS`), the places and the angles.
    """
    sections = _sections(text, tuple(section_ids))
    regions = _regions(text, _lexicon(id(atlas), atlas)) if atlas is not None else []
    places = [float(v) for v in MILLIMETRES.findall(text)]
    angled = ANGLE_WORDS.search(text) is not None
    return Pointed(sections, regions, places, _angle(PITCH, text), _angle(YAW, text), angled)


def _angle(pattern: re.Pattern[str], text: str) -> float | None:
    found = pattern.search(text)
    return float(found.group(1)) if found else None


def _sections(text: str, ids: tuple[str, ...]) -> list[str]:
    """The sections *text* names, in stack order."""
    named: set[str] = set()
    for key, section in _section_keys(ids).items():
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(key)}(?![A-Za-z0-9])", text, re.IGNORECASE):
            named.add(section)
    numbered = _numbered(ids)
    for listed in ORDER_WORDS.findall(text):
        for first, last in RANGE.findall(listed):
            low = int(first)
            high = int(last) if last else low
            for number in range(low, min(high, low + MAX_SECTIONS) + 1):
                if numbered:
                    named.update(numbered.get(number, ()))
                elif 1 <= number <= len(ids):
                    named.add(ids[number - 1])
    return [section for section in ids if section in named][:MAX_SECTIONS]


@lru_cache(maxsize=8)
def _numbered(ids: tuple[str, ...]) -> dict[int, tuple[str, ...]]:
    """The sections whose file names number them ("slice12", "section_3"), by
    that number; empty when the names do not, and "slice N" is then the Nth in
    the stack."""
    out: dict[int, list[str]] = {}
    for section in ids:
        found = NUMBERED_NAME.search(PurePosixPath(section).stem)
        if found:
            out.setdefault(int(found.group(1)), []).append(section)
    return {number: tuple(sections) for number, sections in out.items()}


@lru_cache(maxsize=8)
def _section_keys(ids: tuple[str, ...]) -> dict[str, str]:
    """Every name a section is called by: its stem, and each run of one to
    three of its parts that holds a digit and belongs to it alone."""
    owners: dict[str, set[str]] = {}
    for section in ids:
        stem = PurePosixPath(section).stem
        parts = [part for part in SEPARATORS.split(stem) if part]
        keys = {stem.lower()}
        for size in (1, 2, 3):
            for start in range(len(parts) - size + 1):
                run = parts[start : start + size]
                key = "_".join(run)
                if any(ch.isdigit() for ch in key) and len(key) >= MIN_KEY:
                    keys.add(key.lower())
        for key in keys:
            owners.setdefault(key, set()).add(section)
    return {key: next(iter(found)) for key, found in owners.items() if len(found) == 1}


@lru_cache(maxsize=4)
def _lexicon(_key: int, atlas: Any) -> tuple[tuple[str, str, bool], ...]:
    """``(phrase, acronym, by acronym)`` for every name the atlas's regions go
    by, the longest phrases first."""
    table = structure_table(atlas)
    acronyms = {str(record["acronym"]) for record in table.values()}
    entries: list[tuple[str, str, bool]] = []
    for record in table.values():
        acronym, name = str(record["acronym"]), str(record["name"])
        if len(name) >= MIN_NAME:
            entries.append((name.lower(), acronym, False))
        if acronym not in STOP_ACRONYMS and len(acronym) >= 2 and any(c.isupper() for c in acronym):
            entries.append((acronym, acronym, True))
    for word, wanted in COMMON_WORDS.items():
        for acronym in wanted:
            if acronym in acronyms:
                entries.append((word, acronym, False))
    entries.sort(key=lambda entry: -len(entry[0]))
    return tuple(entries)


def _regions(text: str, lexicon: Iterable[tuple[str, str, bool]]) -> list[str]:
    """The regions *text* names, in the order it names them; a phrase inside a
    longer one already found does not count again."""
    lowered = text.lower()
    taken: list[tuple[int, int]] = []
    found: list[tuple[int, str]] = []
    for phrase, acronym, exact in lexicon:
        haystack = text if exact else lowered
        for match in re.finditer(rf"(?<![A-Za-z0-9]){re.escape(phrase)}(?![A-Za-z0-9])", haystack):
            span = match.span()
            if any(a <= span[0] and span[1] <= b for a, b in taken):
                continue
            taken.append(span)
            found.append((span[0], acronym))
    ordered = [acronym for _at, acronym in sorted(found)]
    return list(dict.fromkeys(ordered))[:MAX_REGIONS]

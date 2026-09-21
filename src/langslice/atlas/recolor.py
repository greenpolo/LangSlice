"""Organized structure colors for atlases whose native palettes mislead people.

Human-review renders key on the Allen convention: hierarchically organized
colors, where one brain division is one hue family and children inherit or
shade their parent's color. (No image model is shown a colored map any more;
nonlinear sends grayscale plates with yellow lines.) Some atlases break this —
all-white placeholders (Osten, Princeton, adult Kim) or real but
hierarchy-uncorrelated colors (the
Waxholm rat family, ADMBA). :func:`color_lut` returns
``{structure_id: (r, g, b)}`` choosing, in order:

1. the atlas's native colors, when they are neither degenerate (a single
   color for everything) nor disorganized (child colors uncorrelated with
   parent colors across a deep-enough tree);
2. the true Allen CCF colors joined by name or acronym (vendored in
   ``allen_colors.json``), when enough of the tree matches — the white atlases
   all reuse Allen's structure tree, so this recovers the real convention, and
   a foreign-but-mammalian tree (the Waxholm rat) lands on its true
   counterparts;
3. a generated Allen-style palette from the structure tree itself: one hue
   per top-level division, shade varying with depth — for foreign trees.

The Allen join (2) matches a structure in this order, first hit wins:

* the structure's own name, normalized on both sides — lowercased,
  punctuation flattened, plurals and British spellings folded, a trailing
  ``, unspecified`` dropped;
* a terminology bridge (:data:`_NAME_BRIDGE`): a generic rat/embryological
  vocabulary map onto Allen's (``mesencephalon`` → ``midbrain``,
  ``olfactory bulb`` → ``main olfactory bulb``), plus "white matter" anywhere
  in a name → Allen's ``fiber tracts``;
* a name variant: the interchangeable cortical tails ``cortex``/``area``/
  ``region``, or the name minus a leading modifier (``secondary visual area``
  → ``visual areas``);
* the acronym — but only when the two names agree that they mean the same
  structure (a shared stem that is not a mere qualifier, or a near-identical
  spelling), so ``V`` (a ventricular system here, the trigeminal motor
  nucleus in Allen) does not join by coincidence;
* the same, on the name with its trailing comma clauses dropped one at a
  time (``zona incerta, dorsal part`` → ``zona incerta``), stopping at a
  clause naming a nucleus, which sits NEXT to the structure before the comma
  rather than inside it.

Whatever the match misses inherits its deepest matched ancestor's color, so
a subregion the Allen tree does not segment still reads as its family; only
structures with no matched ancestor at all fall back to flat gray.

Flat trees (no structure deeper than the top level) always keep their native
colors: with no hierarchy to organize by, no recoloring can help them.
Generic over the structure tree — no atlas or acronym is special-cased.

A palette that is DERIVED (2 or 3 above) then goes through :func:`_organize`,
because "the true Allen color of every leaf" is not a usable palette: Allen
encodes hierarchy in hue, so a cortex-dominated slice renders as a dozen
near-identical greens — distinctions the registration pipeline merges away
anyway (``_family_mapping``, 40 RGB). :func:`_organize` collapses colors
closer than :data:`MERGE_EPS` into one unit (the granularity registration
already runs at) and then pulls the surviving units at least
:data:`MIN_SEPARATION` apart, nudging value and saturation before hue so a
family keeps its identity. Native palettes are left exactly as the atlas
authored them.
"""

from __future__ import annotations

import colorsys
import json
import re
from collections.abc import Iterable, Iterator
from difflib import SequenceMatcher
from functools import lru_cache
from importlib import resources
from typing import Any, Literal, cast

import numpy as np

Rgb = tuple[int, int, int]
RecolorMode = Literal["auto", "always", "never"]

#: Colors closer than this are one registration unit: it is the merge radius
#: ``nonlinear.image_gen_helpers._family_mapping`` folds a classified map
#: down to, so a palette distinction below it is invisible to the pipeline.
MERGE_EPS = 40.0

#: Minimum RGB distance between two colors an organized palette means to
#: distinguish. Above :data:`MERGE_EPS` (so nothing separated merges back)
#: with room for the color drift a generated image adds.
MIN_SEPARATION = 60.0

#: An Allen join must directly match at least this share of the tree to be
#: used; below it the tree is foreign and the palette is generated. Measured:
#: the white Allen-tree atlases score ~0.99, the Waxholm rat family 0.58,
#: truly foreign trees (ADMBA) ~0.09.
MIN_ALLEN_COVERAGE = 0.25

#: Trees at or below this size keep their native colors in "auto" — few
#: structures means few colors on screen, which is organized enough.
MIN_TREE_FOR_RECOLOR = 50

#: Native colors count as organized when the median parent-child color
#: distance is below this fraction of the random-pair distance. Measured on
#: the BrainGlobe corpus: Allen-derived palettes score 0.0, disorganized
#: ones (Waxholm, ADMBA, Dorr) score 0.76-2.1.
DISORGANIZED_RATIO = 0.5

#: Fewer deep parent-child pairs than this and the tree is treated as flat.
MIN_DEEP_PAIRS = 20

#: The generated palette quantizes structures to the deepest ancestor level
#: that keeps at most this many distinct colors (Allen itself uses 81 for
#: 840 structures — few flat colors is the convention).
MAX_FAMILIES = 64

#: ``{(atlas name, mode): colors}``.
_LUT_CACHE: dict[tuple[str, str], dict[int, Rgb]] = {}

#: Word-for-word synonyms, applied to both sides before names are compared:
#: plurals and the British spelling atlases mix freely with Allen's.
_SYNONYM_WORD = {
    "areas": "area",
    "bodies": "body",
    "cells": "cell",
    "commissures": "commissure",
    "complexes": "complex",
    "cortices": "cortex",
    "fibers": "fiber",
    "fibre": "fiber",
    "fibres": "fiber",
    "fields": "field",
    "ganglia": "ganglion",
    "grey": "gray",
    "groups": "group",
    "gyri": "gyrus",
    "laminae": "lamina",
    "layers": "layer",
    "lobules": "lobule",
    "nerves": "nerve",
    "nuclei": "nucleus",
    "peduncles": "peduncle",
    "regions": "region",
    "striae": "stria",
    "systems": "system",
    "tracts": "tract",
}

#: Terminology bridges: a structure name that denotes the same thing Allen
#: names differently. Anatomical vocabulary (classical/embryological against
#: Allen's), not per-atlas special-casing — any tree using these words gets
#: the same treatment.
_NAME_BRIDGE = {
    "gray matter": "Basic cell groups and regions",
    "prosencephalon": "Cerebrum",
    "telencephalon": "Cerebrum",
    "diencephalon": "Interbrain",
    "mesencephalon": "Midbrain",
    "rhombencephalon": "Hindbrain",
    "metencephalon": "Pons",
    "myelencephalon": "Medulla",
    "pallium": "Cerebral cortex",
    "laminated pallium": "Cerebral cortex",
    "non-laminated pallium": "Cortical subplate",
    "subpallium": "Cerebral nuclei",
    "dorsal thalamus": "Thalamus",
    "olfactory bulb": "Main olfactory bulb",
    "anterior olfactory area": "Anterior olfactory nucleus",
    "caudate putamen": "Caudoputamen",
    "cornu ammonis": "Ammon's horn",
    "pontine nuclei": "Pontine gray",
    "tectum": "Midbrain, sensory related",
    "tegmentum": "Midbrain, motor related",
}

#: Words carrying no identity of their own: two names sharing only a "dorsal"
#: or a "nucleus" are not thereby the same structure.
_QUALIFIER = frozenset(
    """of the and a part area region nucleus group complex zone body system
    layer division unspecified dorsal ventral medial lateral anterior
    posterior superior inferior rostral caudal external internal deep
    superficial primary secondary upper lower""".split()
)

#: Leading words that only qualify what follows, so dropping one generalizes
#: to the enclosing structure ("secondary visual area" -> "visual areas").
_MODIFIER = frozenset(
    """dorsal ventral medial lateral anterior posterior superior inferior
    rostral caudal external internal primary secondary""".split()
)

#: Interchangeable last words in cortical naming.
_TAIL = ("cortex", "area", "region")

#: Endings peeled off to let adjective and noun forms of one structure meet
#: ("hippocampal"/"hippocampus", "olivary"/"olive", "thalamic"/"thalamus").
_ENDINGS = ("al", "ar", "ic", "us", "um", "is", "ae", "a", "e", "s", "y")

#: Two names this similar are the same structure misspelled. Measured over
#: the BrainGlobe corpus: real typos ("interpedunclar", "mamilllary") score
#: 0.96+, the closest coincidence (a suprageniculate against a supragenual
#: nucleus) 0.86.
_TYPO_RATIO = 0.9


def _normalize(name: str) -> str:
    """A structure name reduced to what two ontologies can agree on."""
    flat = re.sub(r"[^a-z0-9,]+", " ", str(name).lower())
    words = " ".join(_SYNONYM_WORD.get(w, w) for w in flat.replace(",", " , ").split())
    normalized = words.replace(" ,", ",").strip().strip(",").strip()
    if normalized.startswith("the "):
        normalized = normalized[4:]
    return re.sub(r",\s*unspecified$", "", normalized).strip()


def _stem(word: str) -> str:
    while len(word) > 4:
        for ending in _ENDINGS:
            if word.endswith(ending) and len(word) - len(ending) >= 4:
                word = word[: -len(ending)]
                break
        else:
            break
    return word


def _identity(name: str) -> set[str]:
    """The stems of *name* that say WHICH structure it is."""
    words = _normalize(name).replace(",", " ").split()
    return {_stem(w) for w in words if w not in _QUALIFIER}


def _same_structure(name: str, allen_name: str) -> bool:
    """Whether two names denote one structure: they share an identifying
    stem, or they differ only by a misspelling."""
    if _identity(name) & _identity(allen_name):
        return True
    ratio = SequenceMatcher(None, _normalize(name), _normalize(allen_name)).ratio()
    return ratio >= _TYPO_RATIO


def _tail_swaps(words: list[str]) -> Iterator[str]:
    """The same name with its cortical tail word swapped for the others."""
    if len(words) >= 2 and words[-1] in _TAIL:
        for alt in _TAIL:
            if alt != words[-1]:
                yield " ".join([*words[:-1], alt])


def _variants(normalized: str) -> Iterator[str]:
    """*normalized* and the rewrites that mean the same structure."""
    words = normalized.split()
    yield normalized
    yield from _tail_swaps(words)
    if len(words) > 2 and words[0] in _MODIFIER:
        yield " ".join(words[1:])
        yield from _tail_swaps(words[1:])


@lru_cache(maxsize=1)
def _allen_tables() -> tuple[dict[str, Rgb], dict[str, Rgb], dict[str, str]]:
    """(acronym -> rgb, normalized name -> rgb, acronym -> name) for the
    Allen CCF ontology, with the terminology bridges folded into the names."""
    raw = resources.files("langslice.atlas").joinpath("allen_colors.json").read_text()
    data: dict[str, dict[str, Any]] = json.loads(raw)
    by_acronym = {k: (int(v[0]), int(v[1]), int(v[2])) for k, v in data["acronyms"].items()}
    by_name: dict[str, Rgb] = {}
    for name, triplet in data["names"].items():
        by_name.setdefault(
            _normalize(name), (int(triplet[0]), int(triplet[1]), int(triplet[2]))
        )
    for term, allen_name in _NAME_BRIDGE.items():
        by_name.setdefault(_normalize(term), by_name[_normalize(allen_name)])
    names_of = {k: str(v) for k, v in data["acronym_names"].items()}
    return by_acronym, by_name, names_of


def _allen_color(acronym: str, name: str) -> Rgb | None:
    """The Allen color for a structure named *name*, or ``None``. See the
    module docstring for the order the rules are tried in."""
    by_acronym, by_name, names_of = _allen_tables()
    clauses = _normalize(name).split(",")
    for dropped in range(len(clauses)):
        trimmed = ",".join(clauses[: len(clauses) - dropped]).strip()
        for variant in _variants(trimmed):
            if variant in by_name:
                return by_name[variant]
        if dropped == 0 and acronym in by_acronym:
            if _same_structure(name, names_of.get(acronym, "")):
                return by_acronym[acronym]
        if "nucleus" in clauses[len(clauses) - dropped - 1]:
            break  # a nucleus NEXT to the structure named before the comma
    if "white matter" in _normalize(name):
        return by_name["fiber tract"]
    return None


def _structure_rows(atlas: Any) -> dict[int, dict[str, Any]]:
    """``{id: row}`` from ``atlas.structures``, tolerating rows without an
    explicit ``id`` field (the dict key stands in)."""
    structures = getattr(atlas, "structures", None)
    items = getattr(structures, "items", None)
    rows: dict[int, dict[str, Any]] = {}
    if not callable(items):
        return rows
    pairs = cast("Iterable[tuple[Any, dict[str, Any]]]", items())
    for key, entry in pairs:
        try:
            rows[int(entry.get("id", key))] = entry
        except Exception:
            continue
    return rows


def _native_lut(atlas: Any) -> dict[int, Rgb]:
    lut: dict[int, Rgb] = {}
    for sid, entry in _structure_rows(atlas).items():
        try:
            triplet = entry["rgb_triplet"]
            lut[sid] = (int(triplet[0]), int(triplet[1]), int(triplet[2]))
        except Exception:
            continue
    return lut


def _paths(atlas: Any) -> dict[int, tuple[int, ...]]:
    paths: dict[int, tuple[int, ...]] = {}
    for sid, entry in _structure_rows(atlas).items():
        try:
            paths[sid] = tuple(int(p) for p in entry.get("structure_id_path", ()))
        except Exception:
            continue
    return {sid: p for sid, p in paths.items() if p}


def _is_degenerate(lut: dict[int, Rgb]) -> bool:
    return len(lut) > 1 and len(set(lut.values())) == 1


def _is_disorganized(lut: dict[int, Rgb], paths: dict[int, tuple[int, ...]]) -> bool:
    """Child colors uncorrelated with parent colors, over a deep-enough tree."""
    if len(lut) <= MIN_TREE_FOR_RECOLOR or not paths:
        return False
    root_depth = min(len(p) for p in paths.values())
    pair_distances = []
    for sid, path in paths.items():
        # parent must itself be below the top level, so flat trees drop out
        if len(path) < root_depth + 2 or sid not in lut:
            continue
        parent = path[-2]
        if parent in lut:
            a, b = np.array(lut[sid], float), np.array(lut[parent], float)
            pair_distances.append(float(np.linalg.norm(a - b)))
    if len(pair_distances) < MIN_DEEP_PAIRS:
        return False
    colors = np.array(list(lut.values()), dtype=float)
    rng = np.random.default_rng(0)
    i = rng.integers(0, len(colors), 2000)
    j = rng.integers(0, len(colors), 2000)
    random_mean = float(np.mean(np.linalg.norm(colors[i] - colors[j], axis=1)))
    if random_mean <= 0:
        return False
    return float(np.median(pair_distances)) / random_mean > DISORGANIZED_RATIO


def _allen_join(atlas: Any) -> dict[int, Rgb] | None:
    """True Allen colors, matched structure by structure; the deepest matched
    ancestor colors whatever the match misses (a subregion the Allen tree does
    not segment inherits its family's color). ``None`` when the tree is mostly
    foreign to the Allen ontology."""
    rows = _structure_rows(atlas)
    if not rows:
        return None
    paths = _paths(atlas)
    root_ids = {p[0] for p in paths.values()}
    direct: dict[int, Rgb] = {}
    for sid, entry in rows.items():
        if sid in root_ids:
            # the root must not join (its Allen color is white) nor act as an
            # inheritance source — that painted whole subtrees white once
            continue
        color = _allen_color(str(entry.get("acronym", "")), str(entry.get("name", "")))
        if color is not None:
            direct[sid] = color
    if len(direct) / len(rows) < MIN_ALLEN_COVERAGE:
        return None
    lut = dict(direct)
    for sid, path in paths.items():
        if sid in lut:
            continue
        for ancestor in reversed(path[:-1]):
            if ancestor in direct:
                lut[sid] = direct[ancestor]
                break
        else:
            lut[sid] = (200, 200, 200)  # no Allen counterpart anywhere above
    return lut


def _generated_lut(atlas: Any) -> dict[int, Rgb]:
    """Allen-style palette from the tree alone, by nested hue subdivision.

    Each node receives a slice of the hue wheel proportional to its subtree
    size, subdivided recursively among its children — so a division holding
    most of the brain spreads its children over a wide hue range instead of
    drowning them in one color, while hue distance still tracks tree
    distance at every level (families are hue neighborhoods)."""
    paths = _paths(atlas)
    if not paths:
        return {}

    children: dict[int, set[int]] = {}
    weight: dict[int, int] = {}
    roots: set[int] = set()
    for path in paths.values():
        roots.add(path[0])
        for node in path:
            weight[node] = weight.get(node, 0) + 1
        for parent_node, child in zip(path, path[1:], strict=False):
            children.setdefault(parent_node, set()).add(child)

    hue_depth: dict[int, tuple[float, int]] = {}

    def spread(node: int, lo: float, hi: float, depth: int) -> None:
        hue_depth[node] = ((lo + hi) / 2, depth)
        kids = sorted(children.get(node, ()))
        total = sum(weight[k] for k in kids)
        if not total:
            return
        x = lo
        for kid in kids:
            width = (hi - lo) * weight[kid] / total
            spread(kid, x, x + width, depth + 1)
            x += width

    x = 0.0
    total_weight = sum(weight[r] for r in roots)
    for root in sorted(roots):
        width = weight[root] / total_weight
        spread(root, x, x + width, 0)
        x += width

    # Quantize to a family level: Allen's convention is FEW flat colors (its
    # own table has 81 distinct colors for 840 structures), so descendants
    # inherit their family ancestor's color exactly instead of shading by
    # depth — depth shading on a deep tree washes the whole map into mud.
    max_level = max(len(p) for p in paths.values()) - 1
    family_level = 1
    for level in range(1, max_level + 1):
        count = len({p[min(len(p) - 1, level)] for p in paths.values()})
        if count > MAX_FAMILIES:
            break
        family_level = level

    lut: dict[int, Rgb] = {}
    for sid, path in paths.items():
        family = path[min(len(path) - 1, family_level)]
        h, depth = hue_depth[family]
        if depth == 0:  # the root itself
            lut[sid] = (200, 200, 200)
            continue
        # deterministic s/v alternation so adjacent families differ in more
        # than a small hue step
        s = 0.72 if family % 2 else 0.5
        v = 0.95 if (family // 2) % 2 else 0.78
        r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
        lut[sid] = (int(r * 255), int(g * 255), int(b * 255))
    return lut


def _distance2(a: Rgb, b: Rgb) -> float:
    return float(sum((x - y) ** 2 for x, y in zip(a, b, strict=False)))


def _units(
    lut: dict[int, Rgb], paths: dict[int, tuple[int, ...]]
) -> dict[int, int]:
    """``{structure_id: unit id}`` — the families the palette distinguishes.

    A unit is a set of structures whose base colors are within
    :data:`MERGE_EPS`, named by its shallowest member so a family head (not
    an arbitrary leaf) names and seeds it.
    """
    order = sorted(lut, key=lambda sid: (len(paths.get(sid, ())), sid))
    reps: list[tuple[Rgb, int]] = []
    unit: dict[int, int] = {}
    for sid in order:
        color = lut[sid]
        for rep_color, rep in reps:  # noqa: B007 - rep survives the loop, matched or not
            if _distance2(color, rep_color) <= MERGE_EPS**2:
                break
        else:
            rep = sid
            reps.append((color, sid))
        unit[sid] = rep
    return unit


#: Brightness a palette color may never leave: below it a region reads as the
#: black canvas (the classifier calls anything under 20 background), above it
#: every color is white.
_VALUE_RANGE = (0.35, 1.0)
#: Saturation a CHROMATIC color may never leave — washing one out to gray
#: would collide with the achromatic families (fiber tracts, ventricles),
#: which for their part stay gray because saturation scales rather than shifts.
_SAT_RANGE = (0.18, 1.0)

#: Variations of a color, ordered by how far they stray from it: a small hue
#: rotation is a smaller change than a large brightness drop, so a crowded
#: family fans out inside its own neighbourhood before anything is dimmed.
_SHADE_STEPS: tuple[tuple[float, float, float], ...] = tuple(
    sorted(
        (
            (hue_shift, sat_scale, val_shift)
            for hue_shift in (0.0, 0.05, -0.05, 0.1, -0.1, 0.16, -0.16, 0.25, -0.25, 0.5)
            for sat_scale in (1.0, 1.6, 0.55)
            for val_shift in (0.0, -0.3, 0.25, -0.5, 0.45)
        ),
        key=lambda step: 3.2 * abs(step[0]) + 0.5 * abs(1.0 - step[1]) + abs(step[2]),
    )
)


#: Last resort when a color's own neighbourhood is full: a coarse sweep of the
#: whole gamut. Reached mostly by the gray families — an achromatic seed can
#: otherwise only move along one axis, and a crowded gray axis is how two
#: units end up closer than the merge radius.
_FALLBACK_STEPS: tuple[tuple[float, float, float], ...] = tuple(
    (hue / 12.0, sat, val)
    for val in (0.95, 0.65, 0.45)
    for sat in (0.75, 0.35)
    for hue in range(12)
)


def _shades(color: Rgb) -> Iterator[Rgb]:
    """*color* first, then progressively less similar variants of it."""
    yield color  # unclamped: a color nothing collides with is never touched
    hue, sat, val = colorsys.rgb_to_hsv(*(c / 255.0 for c in color))
    for hue_shift, sat_scale, val_shift in _SHADE_STEPS:
        shifted = sat * sat_scale
        r, g, b = colorsys.hsv_to_rgb(
            (hue + hue_shift) % 1.0,
            min(_SAT_RANGE[1], max(_SAT_RANGE[0], shifted)) if sat > 0.05 else shifted,
            min(_VALUE_RANGE[1], max(_VALUE_RANGE[0], val + val_shift)),
        )
        yield (int(r * 255), int(g * 255), int(b * 255))
    for fallback_hue, fallback_sat, fallback_val in _FALLBACK_STEPS:
        r, g, b = colorsys.hsv_to_rgb(fallback_hue, fallback_sat, fallback_val)
        yield (int(r * 255), int(g * 255), int(b * 255))


def _separate(seeds: dict[int, Rgb], order: list[int]) -> dict[int, Rgb]:
    """Assign each unit a color at least :data:`MIN_SEPARATION` from the rest.

    First come, first served in *order*, so the units that carry the most
    meaning (the shallowest) keep their seed and the crowded ones move. When
    no candidate clears the bar the best one available is taken — a palette
    this dense cannot separate perfectly, and a color as far from its
    neighbours as possible still beats a duplicate.
    """
    taken: list[Rgb] = []
    out: dict[int, Rgb] = {}
    for key in order:
        best: Rgb = seeds[key]
        best_distance = -1.0
        for candidate in _shades(seeds[key]):
            closest = min((_distance2(candidate, t) for t in taken), default=float("inf"))
            if closest >= MIN_SEPARATION**2:
                best = candidate
                break
            if closest > best_distance:
                best, best_distance = candidate, closest
        out[key] = best
        taken.append(best)
    return out


def _organize(atlas: Any, lut: dict[int, Rgb]) -> dict[int, Rgb]:
    """Quantize *lut* to its units and pull those units apart.

    Colors within :data:`MERGE_EPS` collapse into one unit — the granularity
    registration already runs at — and the surviving units are pulled at
    least :data:`MIN_SEPARATION` apart.
    """
    paths = _paths(atlas)
    unit = _units(lut, paths)
    depth = {sid: len(paths.get(sid, ())) for sid in lut}
    seeds: dict[int, Rgb] = {}
    for sid, key in sorted(unit.items(), key=lambda kv: (depth[kv[0]], kv[0])):
        seeds.setdefault(key, lut[sid])
    colors = _separate(seeds, sorted(seeds, key=lambda key: (depth.get(key, 0), key)))
    return {sid: colors[key] for sid, key in unit.items()}


def color_lut(atlas: Any, mode: RecolorMode = "auto") -> dict[int, Rgb]:
    """``{structure_id: (r, g, b)}`` for rendering *atlas*'s annotation."""
    # ponytail: cache keyed by atlas NAME (load_atlas is lru_cached, one name =
    # one object); a nameless atlas (test fakes) is computed fresh every call.
    name = getattr(atlas, "atlas_name", None)
    key = (str(name), mode)
    cached = _LUT_CACHE.get(key) if name is not None else None
    if cached is not None:
        return cached

    native = _native_lut(atlas)
    derived = True
    if mode == "never":
        lut, derived = native, False
    elif mode == "auto" and not (
        _is_degenerate(native) or _is_disorganized(native, _paths(atlas))
    ):
        lut, derived = native, False
    else:
        lut = _allen_join(atlas) or _generated_lut(atlas) or native
    # A palette we derived is ours to make legible; a native one the atlas
    # authored is left exactly as authored.
    if lut and derived:
        lut = _organize(atlas, lut)
    if name is not None:
        _LUT_CACHE[key] = lut
    return lut

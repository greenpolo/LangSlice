"""Canonical anatomical-concept registry for cross-atlas region targeting.

Each *concept* is an atlas-agnostic anatomical idea (e.g. "dentate gyrus
granule cell layer", "Purkinje layer", "isocortex layer 4"). The registry
maps a concept to per-atlas region acronyms plus an optional name-pattern
regex fallback.

Why this exists
---------------
Region-specific texture passes need to fire on the right structures across
every atlas in the training mix (Allen mouse, Kim mouse, Princeton, Osten,
Perens, Waxholm rat, ADMBA developing mouse, ...). Hard-coding Allen
acronyms inside transforms means new atlases break silently. Instead, every
texture pass asks for a concept by name and the registry resolves it.

How resolution works
--------------------
Given (atlas, concept):

1. Match ``atlas.atlas_name`` against the concept's per-atlas table using
   ``fnmatch`` glob patterns (e.g. ``"allen_mouse_*"`` covers 10/25/50 µm
   variants). The first matching pattern wins.
2. For each matched acronym, look it up in ``atlas.structures``; collect the
   structure ID and (when ``include_descendants=True``) every descendant ID.
3. If the table has no match, fall back to ``name_pattern`` — a regex run
   against ``structure["name"]``. Useful for Allen-derived ontologies the
   registry hasn't been hand-curated for yet.
4. If neither resolves, return an empty set. Texture transforms see an
   empty mask and skip the affected pass — graceful degradation.

Adding a concept
----------------
Drop one ``ConceptDef`` into ``CANONICAL_REGIONS``. That's the entire
contract — every texture pass that asks for the concept by name picks it
up automatically on the next regen. No other module needs to change.

Adding atlas support to an existing concept
-------------------------------------------
Add a new key to ``atlas_acronyms`` matching the new atlas's name pattern.
Existing atlases keep working unchanged.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from functools import lru_cache

__all__ = [
    "CANONICAL_REGIONS",
    "ConceptDef",
    "concept_id_set",
    "list_concepts",
    "resolve_concept_acronyms",
]


@dataclass(frozen=True)
class ConceptDef:
    """Definition of one canonical anatomical concept.

    Attributes
    ----------
    name:
        Stable identifier (snake_case). Used as a key into the resolver and
        into ``ctx.tissue_class_masks``.
    description:
        Short human description. Surfaces in QC tooling and helps reviewers
        understand what a texture pass targets.
    atlas_acronyms:
        Mapping ``atlas_name_pattern -> [acronyms]``. Patterns are
        ``fnmatch`` globs (``*`` and ``?``) checked against
        ``atlas.atlas_name`` in dictionary order; the first match wins.
        Acronyms must be valid in the matched atlas's ``structures`` dict.
        Empty lists are allowed — useful for declaring "this concept is
        not represented at this atlas's resolution".
    name_pattern:
        Optional regex applied to ``structure["name"]`` when the per-atlas
        table has no match. Helps cover atlases that share the Allen-style
        English naming convention but aren't explicitly registered yet.
    include_descendants:
        When ``True`` (default), each matched acronym contributes its
        structure ID *and* every descendant ID. Disable for concepts that
        should resolve to one specific node only.
    """

    name: str
    description: str
    atlas_acronyms: dict[str, list[str]] = field(default_factory=dict)
    name_pattern: re.Pattern[str] | None = None
    include_descendants: bool = True


# ---------------------------------------------------------------------------
# Concept registry
# ---------------------------------------------------------------------------
#
# Conventions for atlas patterns:
#   "allen_mouse_*"            — Allen CCFv3 (10/25/50 µm)
#   "kim_mouse_*"              — Kim Lab CCFv3 derivative
#   "princeton_mouse_*"        — Princeton brain registration atlas
#   "osten_mouse_*"            — Osten Lab cell-counting atlas
#   "perens_lsfm_mouse_*"      — Perens iDISCO atlas
#   "ccfv3augmented_mouse_*"   — CCFv3 augmented
#   "allen_mouse_bluebrain_*"  — Bluebrain barrel atlas
#   "admba_*"                  — Allen Developing Mouse (multiple ages)
#   "whs_sd_rat*"              — Waxholm Sprague-Dawley rat
#
# Allen-family atlases share the CCFv3 ontology, so the same acronyms work.
# We use a single ``allen_*`` style key to cover them, plus more specific
# entries when finer-resolution atlases (e.g. allen_mouse_10um) break out
# substructures the coarser variants don't.

CANONICAL_REGIONS: dict[str, ConceptDef] = {
    # -- Hippocampal formation ---------------------------------------------
    "dg_granule_layer": ConceptDef(
        name="dg_granule_layer",
        description=(
            "Dentate gyrus, granule cell layer — extremely tightly packed "
            "small nuclei; reads as a solid bright band in DAPI / Nissl."
        ),
        atlas_acronyms={
            "allen_mouse_*": ["DG-sg"],
            "kim_mouse_*": ["DG-sg"],
            "princeton_mouse_*": ["DG-sg"],
            "osten_mouse_*": ["DG-sg"],
            "perens_lsfm_mouse_*": ["DG-sg"],
            "ccfv3augmented_mouse_*": ["DG-sg"],
            "allen_mouse_bluebrain_*": ["DG-sg"],
        },
        name_pattern=re.compile(
            r"dentate gyrus.*granule|granule cell layer.*dentate", re.I
        ),
    ),
    "ca_pyramidal_layer": ConceptDef(
        name="ca_pyramidal_layer",
        description=(
            "Hippocampal CA1/CA2/CA3 pyramidal cell layer — densely packed "
            "pyramidal neurons forming a thin curving band. Allen 25 µm "
            "does NOT break this out; only resolves on finer atlases."
        ),
        atlas_acronyms={
            "allen_mouse_25um": [],  # not represented at this resolution
            "allen_mouse_50um": [],
            "allen_mouse_10um": ["CA1sp", "CA2sp", "CA3sp"],
            "kim_mouse_*": [],
            "princeton_mouse_*": [],
        },
        name_pattern=re.compile(r"\bCA[123]\b.*pyramidal layer|pyramidal layer.*\bCA[123]\b", re.I),
    ),
    "induseum_griseum": ConceptDef(
        name="induseum_griseum",
        description="Induseum griseum — small dense midline cell strip dorsal to corpus callosum.",
        atlas_acronyms={
            "allen_mouse_*": ["IG"],
            "kim_mouse_*": ["IG"],
            "princeton_mouse_*": ["IG"],
            "osten_mouse_*": ["IG"],
            "perens_lsfm_mouse_*": ["IG"],
            "ccfv3augmented_mouse_*": ["IG"],
        },
        name_pattern=re.compile(r"induseum griseum", re.I),
    ),
    # -- Olfactory bulb -----------------------------------------------------
    "mob_glomerular_layer": ConceptDef(
        name="mob_glomerular_layer",
        description="Main olfactory bulb, glomerular layer — ring of dense cells around glomeruli.",
        atlas_acronyms={
            "allen_mouse_*": ["MOBgl"],
            "kim_mouse_*": ["MOBgl"],
        },
        name_pattern=re.compile(r"main olfactory.*glomerular", re.I),
    ),
    "mob_granule_layer": ConceptDef(
        name="mob_granule_layer",
        description="Main olfactory bulb, granule cell layer — densely packed small nuclei.",
        atlas_acronyms={
            "allen_mouse_*": ["MOBgr"],
            "kim_mouse_*": ["MOBgr"],
        },
        name_pattern=re.compile(r"main olfactory.*granul", re.I),
    ),
    "aob_granule_layer": ConceptDef(
        name="aob_granule_layer",
        description="Accessory olfactory bulb, granular layer.",
        atlas_acronyms={
            "allen_mouse_*": ["AOBgr"],
            "kim_mouse_*": ["AOBgr"],
        },
        name_pattern=re.compile(r"accessory olfactory.*granul", re.I),
    ),
    "nlot_pyramidal_layer": ConceptDef(
        name="nlot_pyramidal_layer",
        description="Nucleus of the lateral olfactory tract, pyramidal layer.",
        atlas_acronyms={
            "allen_mouse_*": ["NLOT2"],
            "kim_mouse_*": ["NLOT2"],
        },
        name_pattern=re.compile(r"lateral olfactory tract.*pyramidal", re.I),
    ),
    "olfactory_bulb": ConceptDef(
        name="olfactory_bulb",
        description=(
            "Whole olfactory bulb (main + accessory) including every sublayer. "
            "Targets the anterior protrusion that physically detaches during "
            "cryosectioning of anterior coronal slices. Layer-specific texture "
            "passes should still use the per-layer concepts "
            "(mob_glomerular_layer, mob_granule_layer, aob_granule_layer)."
        ),
        atlas_acronyms={
            "allen_mouse_*": ["MOB", "AOB"],
            "kim_mouse_*": ["MOB", "AOB"],
            "princeton_mouse_*": ["MOB", "AOB"],
            "osten_mouse_*": ["MOB", "AOB"],
            "perens_lsfm_mouse_*": ["MOB", "AOB"],
            "ccfv3augmented_mouse_*": ["MOB", "AOB"],
            "allen_mouse_bluebrain_*": ["MOB", "AOB"],
        },
        name_pattern=re.compile(r"olfactory bulb", re.I),
    ),
    "islands_of_calleja": ConceptDef(
        name="islands_of_calleja",
        description=(
            "Islands of Calleja — small dense granule-cell clusters scattered "
            "through the olfactory tubercle. Always show a bright DAPI clump "
            "in real sections. Allen 25 µm doesn't break these out as a "
            "distinct structure, so we proxy via the olfactory tubercle "
            "parent (OT) — clump renderer picks one center per OT blob and "
            "lands roughly in the IsC region."
        ),
        atlas_acronyms={
            "allen_mouse_*": ["OT"],
            "kim_mouse_*": ["OT"],
            "princeton_mouse_*": ["OT"],
            "osten_mouse_*": ["OT"],
            "perens_lsfm_mouse_*": ["OT"],
            "ccfv3augmented_mouse_*": ["OT"],
        },
        name_pattern=re.compile(r"island.*calleja|olfactory tubercle", re.I),
    ),
    # -- Cerebellum ---------------------------------------------------------
    # Allen 25 µm only breaks cerebellum down to CB/CBN/CBX — Purkinje and
    # granule layers are not separate structures. These entries resolve to
    # empty at 25 µm and require either a finer atlas or a procedural
    # approximation (e.g. inner edge of CBX for Purkinje line).
    "purkinje_layer": ConceptDef(
        name="purkinje_layer",
        description=(
            "Cerebellar Purkinje cell layer — single line of large flask-"
            "shaped neurons. Not resolved in Allen 25 µm; needs procedural "
            "approximation or a finer atlas."
        ),
        atlas_acronyms={
            "allen_mouse_*": [],  # placeholder until finer mapping or procedural added
        },
        name_pattern=re.compile(r"purkinje layer|cerebellar.*purkinje", re.I),
    ),
    "cerebellar_granule_layer": ConceptDef(
        name="cerebellar_granule_layer",
        description=(
            "Cerebellar cortex, granular layer — extremely dense small "
            "nuclei (the densest cell layer in the brain). Not resolved in "
            "Allen 25 µm; needs procedural approximation or finer atlas."
        ),
        atlas_acronyms={
            "allen_mouse_*": [],
        },
        name_pattern=re.compile(r"cerebellar.*granular", re.I),
    ),
}


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def list_concepts() -> list[str]:
    """Return every registered concept name."""
    return sorted(CANONICAL_REGIONS.keys())


def resolve_concept_acronyms(atlas_name: str, concept: str) -> list[str]:
    """Return the acronym list for ``concept`` in ``atlas_name``.

    First matching ``fnmatch`` pattern wins. Returns ``[]`` if the concept
    isn't registered, or if the atlas doesn't appear in the concept's
    per-atlas table (callers should fall back to ``name_pattern`` in that
    case).
    """
    cd = CANONICAL_REGIONS.get(concept)
    if cd is None:
        return []
    for pattern, acronyms in cd.atlas_acronyms.items():
        if fnmatch.fnmatch(atlas_name, pattern):
            return list(acronyms)
    return []


def _ids_from_atlas(atlas: object, concept: str) -> frozenset[int]:
    """Compute the ID set for ``concept`` against a loaded atlas.

    Used directly for atlases that aren't loadable by name (e.g. test mocks).
    For real BrainGlobe atlases prefer the cached ``concept_id_set``.
    """
    cd = CANONICAL_REGIONS.get(concept)
    if cd is None:
        return frozenset()

    structures = getattr(atlas, "structures", None)
    get_descendants = getattr(atlas, "get_structure_descendants", None)
    if structures is None:
        return frozenset()

    atlas_name: str = getattr(atlas, "atlas_name", "")
    acronyms = resolve_concept_acronyms(atlas_name, concept)

    ids: set[int] = set()

    # Path 1: per-atlas acronym table
    for acronym in acronyms:
        try:
            root = structures[acronym]
        except (KeyError, TypeError):
            continue
        try:
            ids.add(int(root["id"]))
        except (KeyError, TypeError, ValueError):
            pass
        if cd.include_descendants and callable(get_descendants):
            try:
                for descendant_acronym in get_descendants(acronym):
                    try:
                        ids.add(int(structures[descendant_acronym]["id"]))
                    except (KeyError, TypeError, ValueError):
                        continue
            except Exception:
                continue

    # Path 2: name-pattern fallback (only if the atlas table didn't yield anything)
    if not ids and cd.name_pattern is not None:
        try:
            for s in structures.values():
                try:
                    name = str(s.get("name", ""))
                except (AttributeError, TypeError):
                    continue
                if not cd.name_pattern.search(name):
                    continue
                try:
                    ids.add(int(s.get("id")))
                except (TypeError, ValueError):
                    continue
                if cd.include_descendants and callable(get_descendants):
                    try:
                        ac = str(s.get("acronym", ""))
                    except (AttributeError, TypeError):
                        ac = ""
                    if ac:
                        try:
                            for descendant_acronym in get_descendants(ac):
                                try:
                                    ids.add(int(structures[descendant_acronym]["id"]))
                                except (KeyError, TypeError, ValueError):
                                    continue
                        except Exception:
                            continue
        except Exception:
            pass

    return frozenset(ids)


@lru_cache(maxsize=256)
def _ids_for_loadable_atlas(atlas_name: str, concept: str) -> frozenset[int]:
    """Cached resolver keyed on (atlas_name, concept).

    Loads the BrainGlobe atlas by name and computes the ID set. Cache key
    includes ``concept`` so the registry can be edited and a re-import
    invalidates only the affected concept implicitly (the lru_cache lives
    on the function, not the registry).
    """
    from langslice_harness.atlas.core import load_atlas

    atlas = load_atlas(atlas_name)
    return _ids_from_atlas(atlas, concept)


def concept_id_set(atlas: object, concept: str) -> frozenset[int]:
    """Return all annotation IDs for ``concept`` in ``atlas``.

    Uses the cached path when ``atlas.atlas_name`` is loadable by name;
    falls back to direct computation for unnamed / mock atlases.
    """
    atlas_name: str = getattr(atlas, "atlas_name", "")
    if not atlas_name:
        return _ids_from_atlas(atlas, concept)
    try:
        return _ids_for_loadable_atlas(atlas_name, concept)
    except Exception:
        return _ids_from_atlas(atlas, concept)

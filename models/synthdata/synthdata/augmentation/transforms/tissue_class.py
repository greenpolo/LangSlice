"""Tissue-class masks (gray / white matter / ventricle) from atlas annotations.

Procedural Stage B textures are routed by tissue class — gray-matter DAPI looks
nothing like white-matter DAPI, and the same for every other stain. The
annotation slice (per-pixel region IDs from BrainGlobe) is the ground truth for
which class each pixel belongs to.

Allen CCFv3 hierarchy splits the brain into three top-level non-root nodes:
    grey         (id=8)    — every cell-rich region
    fiber tracts (id=1009) — every white-matter tract
    VS           (id=73)   — ventricular system

Any annotation ID descended from one of these roots gets that class.
ID==0 is background (off-tissue).

Atlas-agnostic resolution
-------------------------
``get_tissue_id_sets(atlas)`` resolves the correct root acronyms for each atlas
in two stages:

1. **Per-atlas explicit mapping** (``_ATLAS_TISSUE_ROOTS``) — covers all
   locally cached rodent atlases.  Existing callers of ``allen_mouse_25um``
   are unaffected.

2. **Keyword fallback** — if the atlas name is not in the mapping, the
   structure tree is searched for names matching tissue-class keywords.  A
   ``UserWarning`` is emitted so callers know auto-detection is active.
"""

from __future__ import annotations

import logging
import re
import warnings
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "TISSUE_CLASS_ROOTS",
    "classify_tissue",
    "get_tissue_id_sets",
]


# ---------------------------------------------------------------------------
# Legacy public constant — still valid for allen_mouse_25um
# ---------------------------------------------------------------------------

TISSUE_CLASS_ROOTS: dict[str, str] = {
    "gray_matter": "grey",
    "white_matter": "fiber tracts",
    "ventricle": "VS",
}


# ---------------------------------------------------------------------------
# Per-atlas explicit root-acronym mapping
# ---------------------------------------------------------------------------

# Keys are BrainGlobe atlas names.  Values map tissue class → root acronym
# (as recognised by that atlas's ``structures`` dict).
#
# Allen-based atlases
# -------------------
# All Allen CCFv3 variants (25um, 50um, 10um, princeton, osten, perens,
# ccfv3augmented) share the same hierarchy with the same acronyms.
#
# Kim Lab atlases
# ---------------
# kim_mouse_25um / 50um / 10um use numeric-string acronyms for the root
# structures (e.g. "fiber_tracts" instead of "fiber tracts"; "VentSys"
# instead of "VS").
#
# kim_mouse_isotropic_20um is one level deeper (fiber_tracts at depth 3) but
# uses "grey" / "fiber_tracts" / "VS".
#
# Waxholm Sprague-Dawley Rat
# --------------------------
# Completely different ontology: 'GM', 'wmt', 'V' at depth-3.
#
_ATLAS_TISSUE_ROOTS: dict[str, dict[str, str]] = {
    # --- Allen CCFv3 variants ---
    "allen_mouse_25um": {
        "gray_matter": "grey",
        "white_matter": "fiber tracts",
        "ventricle": "VS",
    },
    "allen_mouse_50um": {
        "gray_matter": "grey",
        "white_matter": "fiber tracts",
        "ventricle": "VS",
    },
    "allen_mouse_10um": {
        "gray_matter": "grey",
        "white_matter": "fiber tracts",
        "ventricle": "VS",
    },
    # --- Princeton / Osten / Perens / CCFv3-augmented — all Allen-derived ---
    "princeton_mouse_20um": {
        "gray_matter": "grey",
        "white_matter": "fiber tracts",
        "ventricle": "VS",
    },
    "osten_mouse_25um": {
        "gray_matter": "grey",
        "white_matter": "fiber tracts",
        "ventricle": "VS",
    },
    "perens_lsfm_mouse_20um": {
        "gray_matter": "grey",
        "white_matter": "fiber tracts",
        "ventricle": "VS",
    },
    "ccfv3augmented_mouse_25um": {
        "gray_matter": "grey",
        "white_matter": "fiber tracts",
        "ventricle": "VS",
    },
    # --- Allen Mouse Bluebrain Barrels (Allen-derived) ---
    "allen_mouse_bluebrain_barrels_10um": {
        "gray_matter": "grey",
        "white_matter": "fiber tracts",
        "ventricle": "VS",
    },
    # --- Kim Lab atlases (numeric-acronym ontology) ---
    "kim_mouse_25um": {
        "gray_matter": "grey",
        "white_matter": "fiber_tracts",
        "ventricle": "VentSys",
    },
    "kim_mouse_50um": {
        "gray_matter": "grey",
        "white_matter": "fiber_tracts",
        "ventricle": "VentSys",
    },
    "kim_mouse_10um": {
        "gray_matter": "grey",
        "white_matter": "fiber_tracts",
        "ventricle": "VentSys",
    },
    "kim_mouse_isotropic_20um": {
        "gray_matter": "grey",
        "white_matter": "fiber_tracts",
        "ventricle": "VS",
    },
    # --- Waxholm Sprague-Dawley Rat ---
    "whs_sd_rat_39um": {
        "gray_matter": "GM",
        "white_matter": "wmt",
        "ventricle": "V",
    },
    # Alias canonicalized by atlas.core
    "whs_sd_rat": {
        "gray_matter": "GM",
        "white_matter": "wmt",
        "ventricle": "V",
    },
}


# ---------------------------------------------------------------------------
# Keyword-fallback patterns used when an atlas is not in the explicit mapping
# ---------------------------------------------------------------------------

_GRAY_MATTER_PATTERNS = re.compile(r"(grey|gray matter|basic cell groups|isocortex|cerebrum)", re.I)
_WHITE_MATTER_PATTERNS = re.compile(
    r"(white matter|fiber tract|fibre tract|corpus callosum|internal capsule"
    r"|callosum|commissure|capsule|funiculus|lemniscus|peduncle|fasciculus"
    r"|fornix|fimbria|stria|tract)",
    re.I,
)
_VENTRICLE_PATTERNS = re.compile(
    r"(ventricle|ventricular system|aqueduct|csf|lateral ventricle"
    r"|third ventricle|fourth ventricle|central canal)",
    re.I,
)

# Concepts whose union forms ``dense_cell_layers`` — these are densely-packed
# cell layers that read as solid bright bands in DAPI/Nissl. Each entry is a
# canonical-region concept registered in ``augmentation.canonical_regions``;
# adding a new dense layer = register a concept + add its name here.
_DENSE_CELL_LAYER_CONCEPTS: tuple[str, ...] = (
    "dg_granule_layer",
    "ca_pyramidal_layer",
    "induseum_griseum",
    "mob_granule_layer",
    "aob_granule_layer",
    "nlot_pyramidal_layer",
    "purkinje_layer",
    "cerebellar_granule_layer",
)

_TISSUE_CLASS_PATTERNS: dict[str, re.Pattern[str]] = {
    "gray_matter": _GRAY_MATTER_PATTERNS,
    "white_matter": _WHITE_MATTER_PATTERNS,
    "ventricle": _VENTRICLE_PATTERNS,
}

_STRUCTURE_MASK_SPECS: dict[str, tuple[tuple[str, ...], re.Pattern[str]]] = {
    "isocortex": (
        ("Isocortex", "ISO", "CTX"),
        re.compile(r"\bisocortex\b|cerebral cortex|cortical plate", re.I),
    ),
    "thalamus": (
        ("TH", "Th", "Thal"),
        re.compile(r"\bthalamus\b|\bthalamic\b", re.I),
    ),
    "hippocampal_formation": (
        ("HPF", "HIP"),
        re.compile(r"hippocampal", re.I),
    ),
    "cortical_subplate": (
        ("CTXsp",),
        re.compile(r"cortical subplate", re.I),
    ),
}


def _find_root_by_keyword(atlas: object, tissue_class: str) -> str | None:
    """Search ``atlas.structures`` for a top-level structure matching keywords.

    Returns the acronym of the shallowest matching structure, or ``None`` if
    nothing is found.
    """
    pattern = _TISSUE_CLASS_PATTERNS[tissue_class]
    structures = getattr(atlas, "structures", None)
    if structures is None:
        return None

    best: tuple[int, str] | None = None  # (path_depth, acronym)
    try:
        items = structures.values() if hasattr(structures, "values") else []
        for s in items:
            if not isinstance(s, dict):
                continue
            name: str = str(s.get("name", ""))
            if not pattern.search(name):
                continue
            depth = len(s.get("structure_id_path", []))
            acronym = str(s.get("acronym", ""))
            if not acronym:
                continue
            if best is None or depth < best[0]:
                best = (depth, acronym)
    except Exception:
        return None

    return best[1] if best is not None else None


def _resolve_tissue_roots(atlas: object) -> dict[str, str]:
    """Return the gray_matter / white_matter / ventricle root acronyms.

    First tries the explicit per-atlas mapping; falls back to keyword search
    and emits a ``UserWarning`` to alert the caller.
    """
    atlas_name: str = getattr(atlas, "atlas_name", "")
    explicit = _ATLAS_TISSUE_ROOTS.get(atlas_name)
    if explicit is not None:
        return dict(explicit)

    warnings.warn(
        f"Atlas '{atlas_name}' is not in the explicit tissue-roots mapping. "
        "Falling back to keyword-based structure search.  Classification may "
        "be less accurate.  Consider adding this atlas to _ATLAS_TISSUE_ROOTS "
        "in transforms/tissue_class.py.",
        UserWarning,
        stacklevel=4,
    )

    resolved: dict[str, str] = {}
    for tissue_class in ("gray_matter", "white_matter", "ventricle"):
        acronym = _find_root_by_keyword(atlas, tissue_class)
        if acronym is None:
            logger.warning(
                "Could not find a '%s' root structure in atlas '%s' via keyword "
                "search; this tissue class will be empty.",
                tissue_class,
                atlas_name,
            )
            # Use a sentinel that can't match any real ID so the set stays empty.
            acronym = f"__missing__{tissue_class}"
        resolved[tissue_class] = acronym
    return resolved


# ---------------------------------------------------------------------------
# Core ID-set helpers (cached)
# ---------------------------------------------------------------------------


def _id_set_from_atlas_obj(atlas: object, root_acronym: str) -> frozenset[int]:
    """Compute the set of annotation IDs under ``root_acronym`` from a loaded atlas."""
    structures = getattr(atlas, "structures", None)
    get_descendants = getattr(atlas, "get_structure_descendants", None)

    if structures is None or not callable(get_descendants):
        logger.warning(
            "Atlas object '%s' does not expose .structures / "
            ".get_structure_descendants(); returning empty set for '%s'.",
            getattr(atlas, "atlas_name", "<unknown>"),
            root_acronym,
        )
        return frozenset()

    try:
        root = structures[root_acronym]
    except (KeyError, TypeError):
        logger.warning(
            "Root acronym '%s' not found in atlas '%s' structures; "
            "returning empty ID set for this tissue class.",
            root_acronym,
            getattr(atlas, "atlas_name", "<unknown>"),
        )
        return frozenset()

    descendants = get_descendants(root_acronym)
    ids: set[int] = {root["id"]}
    for ac in descendants:
        try:
            ids.add(structures[ac]["id"])
        except (KeyError, TypeError):
            pass
    return frozenset(ids)


@lru_cache(maxsize=32)
def _id_set_for_root(atlas_name: str, root_acronym: str) -> frozenset[int]:
    """Cached wrapper: load atlas by name, then compute the ID set."""
    from langslice_harness.atlas.core import load_atlas

    atlas = load_atlas(atlas_name)
    return _id_set_from_atlas_obj(atlas, root_acronym)


def _find_structure_acronym(
    atlas: object,
    *,
    candidate_acronyms: tuple[str, ...],
    name_pattern: re.Pattern[str],
) -> str | None:
    """Find a structure acronym by exact acronym first, then by name keyword."""
    structures = getattr(atlas, "structures", None)
    if structures is None:
        return None

    for acronym in candidate_acronyms:
        try:
            structures[acronym]
            return acronym
        except Exception:
            pass

    try:
        items = structures.values() if hasattr(structures, "values") else []
        best: tuple[int, str] | None = None
        for s in items:
            if not isinstance(s, dict):
                continue
            name = str(s.get("name", ""))
            if not name_pattern.search(name):
                continue
            acronym = str(s.get("acronym", ""))
            if not acronym:
                continue
            depth = len(s.get("structure_id_path", []))
            if best is None or depth < best[0]:
                best = (depth, acronym)
        return best[1] if best is not None else None
    except Exception:
        return None


def _specific_structure_id_sets(atlas: object) -> dict[str, frozenset[int]]:
    result: dict[str, frozenset[int]] = {}
    for key, (candidate_acronyms, name_pattern) in _STRUCTURE_MASK_SPECS.items():
        acronym = _find_structure_acronym(
            atlas,
            candidate_acronyms=candidate_acronyms,
            name_pattern=name_pattern,
        )
        result[key] = _id_set_from_atlas_obj(atlas, acronym) if acronym is not None else frozenset()
    return result


def get_tissue_id_sets(atlas: object) -> dict[str, frozenset[int]]:
    """Return a class -> {annotation IDs} mapping for one atlas.

    For known atlases (in the explicit per-atlas mapping or loadable by name)
    the result is cached per (atlas-name, root-acronym) pair.  For anonymous
    mock objects (e.g. in tests) the lookup falls back to a direct object call
    without caching.

    Existing callers using ``allen_mouse_25um`` are unaffected.
    """
    name: str = getattr(atlas, "atlas_name", "")  # type: ignore[attr-defined]
    roots = _resolve_tissue_roots(atlas)

    # Use the cached load-by-name path only when the atlas name is a real BrainGlobe
    # atlas (i.e. it appears in the explicit mapping OR we can confirm it was loaded
    # via load_atlas).  For fake/mock objects that are already loaded, call the
    # compute function directly on the passed-in object.
    result: dict[str, frozenset[int]] = {}
    for cls, root in roots.items():
        # Skip sentinel "missing" roots produced by the fallback.
        if root.startswith("__missing__"):
            result[cls] = frozenset()
            continue
        # Try to use the cached name-based path first — it avoids redundant
        # re-computation when the same atlas is loaded multiple times under the
        # same name.  If the name isn't a loadable BrainGlobe atlas (e.g. a
        # mock), fall back to computing directly from the supplied object.
        try:
            result[cls] = _id_set_for_root(name, root)
        except Exception:
            result[cls] = _id_set_from_atlas_obj(atlas, root)
    return result


def classify_tissue(
    annotation_slice: np.ndarray,
    atlas: object,
) -> dict[str, np.ndarray]:
    """Per-pixel tissue-class masks for an atlas annotation slice.

    Args:
        annotation_slice: HW int array of region IDs.
        atlas: BrainGlobe atlas exposing ``.atlas_name``, ``.structures``,
            ``.get_structure_descendants(acronym)``.

    Returns dict with keys:
        gray_matter, white_matter, ventricle, tissue, background — all HW bool arrays.

    ``tissue`` is every annotated pixel (``annotation_slice != 0``); ``background`` is
    its complement. The three sub-class masks are mutually exclusive but their
    union is a strict subset of ``tissue``: edge pixels labeled as the root
    structure ("root", id=997) without a sub-classification fall through. The
    texture router should default such pixels to gray-matter.
    """
    id_sets = get_tissue_id_sets(atlas)
    masks: dict[str, np.ndarray] = {}
    for cls, ids in id_sets.items():
        masks[cls] = np.isin(annotation_slice, np.fromiter(ids, dtype=np.int64))
    for cls, ids in _specific_structure_id_sets(atlas).items():
        masks[cls] = np.isin(annotation_slice, np.fromiter(ids, dtype=np.int64))

    # Resolve every concept registered in canonical_regions to its own mask
    # in the dict. Texture passes target concepts by name; the registry hides
    # which atlas-specific acronyms they expand to.
    from ..canonical_regions import (
        concept_id_set,
        list_concepts,
    )

    for concept in list_concepts():
        ids = concept_id_set(atlas, concept)
        masks[concept] = np.isin(annotation_slice, np.fromiter(ids, dtype=np.int64))

    # Backward-compat aggregate: union of every dense-cell-layer concept,
    # so existing callers asking for ``ctx.tissue_class_masks["dense_cell_layers"]``
    # keep working. The aggregate set may be edited by appending or removing
    # entries to ``_DENSE_CELL_LAYER_CONCEPTS``.
    dense_mask = np.zeros_like(
        masks["tissue"] if "tissue" in masks else annotation_slice == 0, dtype=bool
    )
    for concept in _DENSE_CELL_LAYER_CONCEPTS:
        if concept in masks:
            dense_mask |= masks[concept]
    masks["dense_cell_layers"] = dense_mask

    masks["background"] = annotation_slice == 0
    masks["tissue"] = ~masks["background"]
    return masks

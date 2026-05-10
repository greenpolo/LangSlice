# Canonical Regions Registry

`canonical_regions.py` maps anatomical *concepts* (e.g.
`dg_granule_layer`, `purkinje_layer`) to per-atlas region acronyms.
Texture transforms target concepts by name, never raw acronyms — that's
how the pipeline stays atlas-agnostic.

## Why it exists

Without the registry, region-specific texture passes would hard-code
Allen acronyms inside transform classes. Two consequences:

1. **Cross-atlas breakage**: Waxholm rat uses different acronyms (`GM`,
   `wmt`, `V` instead of Allen's `grey`, `fiber tracts`, `VS`). Any
   transform looking up Allen acronyms would silently produce empty
   masks on rat atlases.
2. **Maintenance churn**: Adding a new atlas would require touching
   every region-aware transform. With the registry, one PR adds the
   atlas's acronym list to the relevant `ConceptDef.atlas_acronyms`
   dict and every transform picks it up.

Texture transforms ask `ctx.tissue_class_masks["dg_granule_layer"]` and
get the right mask regardless of which atlas built `ctx`.

## File layout

```
canonical_regions.py
├── @dataclass ConceptDef
│   ├── name              str    — snake_case identifier
│   ├── description       str    — human-readable
│   ├── atlas_acronyms    dict   — fnmatch-pattern → [acronyms]
│   ├── name_pattern      regex  — optional fallback
│   └── include_descendants bool — expand each acronym to its descendants
│
├── CANONICAL_REGIONS: dict[str, ConceptDef]
│   ├── dg_granule_layer
│   ├── ca_pyramidal_layer
│   ├── induseum_griseum
│   ├── mob_glomerular_layer
│   ├── mob_granule_layer
│   ├── aob_granule_layer
│   ├── nlot_pyramidal_layer
│   ├── purkinje_layer            (stub — not in Allen 25 µm)
│   └── cerebellar_granule_layer  (stub — not in Allen 25 µm)
│
├── resolve_concept_acronyms(atlas_name, concept) -> [str]
├── _ids_from_atlas(atlas, concept)              -> frozenset[int]
├── _ids_for_loadable_atlas(atlas_name, concept) -> frozenset[int]  (cached)
└── concept_id_set(atlas, concept)               -> frozenset[int]   (public)
```

## Resolution algorithm

Given `(atlas, concept)`:

1. Find the concept in `CANONICAL_REGIONS`. Return `frozenset()` if not registered.
2. Find the first `atlas_acronyms` key matching `atlas.atlas_name` via
   `fnmatch.fnmatch`. The first match wins (dict insertion order).
3. For each acronym, resolve in `atlas.structures`:
   - Add the structure's ID
   - If `include_descendants=True`, also add IDs for every descendant
     via `atlas.get_structure_descendants(acronym)`
4. **Fallback**: If the per-atlas table yielded nothing AND `name_pattern`
   is set, walk every structure in `atlas.structures.values()` and
   collect IDs whose `name` field matches the pattern.
5. Return the resulting `frozenset[int]`.

If neither path resolves, the concept's mask in
`ctx.tissue_class_masks` is empty — region-aware transforms see this and
skip silently.

## Atlas-pattern conventions

`atlas_acronyms` keys are `fnmatch` glob patterns matched against
`atlas.atlas_name`. Conventional patterns:

| Pattern | Covers |
|---------|--------|
| `allen_mouse_*` | Allen CCFv3 (10/25/50 µm) |
| `kim_mouse_*` | Kim Lab CCFv3 derivative |
| `princeton_mouse_*` | Princeton brain registration atlas |
| `osten_mouse_*` | Osten Lab cell-counting atlas |
| `perens_lsfm_mouse_*` | Perens iDISCO atlas |
| `ccfv3augmented_mouse_*` | CCFv3 augmented |
| `allen_mouse_bluebrain_*` | Bluebrain barrel atlas |
| `admba_*` | Allen Developing Mouse (multiple ages) |
| `whs_sd_rat*` | Waxholm Sprague-Dawley rat |

Allen-family atlases share the CCFv3 ontology, so a single
`"allen_mouse_*"` key covers 10/25/50 µm. When a finer atlas exposes
substructures the coarser variants don't (e.g. `allen_mouse_10um` has
`CA1sp`/`CA2sp`/`CA3sp` while 25 µm doesn't), use a more specific
pattern as a separate dict key.

## Hooked into `tissue_class.classify_tissue`

`tissue_class.py::classify_tissue` calls `concept_id_set` for every
registered concept and writes one mask per concept into the dict it
returns. Plus an aggregate `dense_cell_layers` mask is the union of:

```python
_DENSE_CELL_LAYER_CONCEPTS = (
    "dg_granule_layer",
    "ca_pyramidal_layer",
    "induseum_griseum",
    "mob_granule_layer",
    "aob_granule_layer",
    "nlot_pyramidal_layer",
    "purkinje_layer",
    "cerebellar_granule_layer",
)
```

So `ctx.tissue_class_masks` has both per-concept masks
(`["dg_granule_layer"]` etc.) and the aggregate (`["dense_cell_layers"]`).
Region-aware transforms can target either granularity.

## What resolves where (current state)

For `allen_mouse_25um`:

| Concept | Resolved acronym | Notes |
|---------|------------------|-------|
| `dg_granule_layer` | DG-sg | ✓ The big one — granule cells |
| `induseum_griseum` | IG | ✓ Small midline cell strip |
| `aob_granule_layer` | AOBgr | ✓ Accessory olfactory bulb |
| `nlot_pyramidal_layer` | NLOT2 | ✓ Lateral olfactory tract |
| `ca_pyramidal_layer` | (empty) | Allen 25 µm doesn't break out CA*sp |
| `mob_glomerular_layer` | (empty) | Only `MOB` exists at 25 µm, no sublayers |
| `mob_granule_layer` | (empty) | Same |
| `purkinje_layer` | (empty) | Allen 25 µm only has CB/CBN/CBX |
| `cerebellar_granule_layer` | (empty) | Same |

Empty resolutions are intentional — adding finer atlases will populate them.

## Adding a new concept

```python
# in canonical_regions.py

CANONICAL_REGIONS["my_new_concept"] = ConceptDef(
    name="my_new_concept",
    description="Cell-rich region distinct from generic gray matter.",
    atlas_acronyms={
        "allen_mouse_*":   ["FOO", "BAR"],   # use real acronyms
        "kim_mouse_*":     ["FOO", "BAR"],
        "whs_sd_rat*":     ["RatAcr1", "RatAcr2"],
    },
    name_pattern=re.compile(r"my-region-name|alternative-name", re.I),
    include_descendants=True,
)
```

Mask becomes available at `ctx.tissue_class_masks["my_new_concept"]`
automatically — no edits to `tissue_class.py` needed.

To roll the new concept into the `dense_cell_layers` aggregate, add
`"my_new_concept"` to `tissue_class._DENSE_CELL_LAYER_CONCEPTS`.

## Adding atlas support to an existing concept

Add a new key to `atlas_acronyms`:

```python
CANONICAL_REGIONS["dg_granule_layer"].atlas_acronyms["my_new_atlas"] = ["DGgr"]
```

Note: `ConceptDef` is `@dataclass(frozen=True)`. Edit the literal in the
file rather than mutating at runtime.

## Verifying a concept resolves

```python
PYTHONPATH="models/langslice-gemma-4/data:src" python -c "
from langslice_harness.atlas.core import load_atlas
from augmentation.canonical_regions import concept_id_set
atlas = load_atlas('allen_mouse_25um')
print(concept_id_set(atlas, 'dg_granule_layer'))
# Should print frozenset({632}) — that's DG-sg's atlas ID
"
```

To list which structures resolve in a given atlas:

```python
for s in atlas.structures.values():
    if int(s.get('id', -1)) in ids:
        print(s.get('id'), s.get('acronym'), s.get('name'))
```

## Caching

`_ids_for_loadable_atlas(atlas_name, concept)` is `lru_cache(maxsize=256)`-d.
The cache key is `(atlas_name, concept)`, so editing the registry
invalidates only when the module is reloaded. In the production
pipeline (one process per `synth_dataset write` invocation) this is
fine — the cache fills on first lookup and stays warm for the rest of
the run.

If you edit `CANONICAL_REGIONS` while a long-running process is
holding stale results, restart Python.

## Tests

`canonical_regions` itself has no dedicated test file. Coverage comes
indirectly through `tests/test_augmentation_integration.py` —
`classify_tissue` is exercised in every modality renderer test, and the
shape contract verifies that `ctx.tissue_class_masks` has the expected
keys.

When adding a new concept, manually verify it resolves against
`allen_mouse_25um` (the reference test atlas) by running the inspection
snippet above.

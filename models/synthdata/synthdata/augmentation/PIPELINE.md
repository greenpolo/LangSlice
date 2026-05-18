# Pipeline Architecture

How the code actually flows. Read this when you need to understand
*where* a transform runs, *what state* it sees, and *what contract* it
must honor.

## Top-level entry: `synth_dataset.py`

`synth_dataset.py write` is the canonical regen driver. It:

1. Iterates over a sampling loop, drawing a `SynthSpec` per row (atlas
   choice, plane, position_mm, modality, mode, damage intensity, oblique
   angles, seed).
2. Loads the atlas once via `langslice_harness.atlas.core.load_atlas`.
3. Calls `oblique.get_oblique_slice(atlas, base_position_mm, plane,
   yaw_deg, pitch_deg, roll_deg)` to extract a tilted reference + annotation.
4. Dispatches to the appropriate `render_<modality>_section` function.
5. Writes `images/{seed:08d}.png` + a JSONL row to `manifest.jsonl`.

`SynthSpec` is the contract between sampler and renderer — see
`SynthSpec` dataclass at the top of `synth_dataset.py`.

## The five modality renderers

Each lives in `<modality>_pipeline.py` and exposes a single function:

```python
def render_<modality>_section(
    reference_slice: np.ndarray,   # HW uint8 or float [0,1]
    annotation_slice: np.ndarray,  # HW int32 region IDs
    atlas: object,                 # BrainGlobe atlas
    *,
    seed: int,
    pixel_size_um: float,
    apply_damage: bool = True,
    damage_intensity: str = "medium",
    apply_geometry_warp: bool = True,
    plane: str = "coronal",
    position_mm: float | None = None,
    # plus modality-specific kwargs (mode, gamma_range, etc.)
) -> np.ndarray:                   # HWC float32 [0,1]
    ...
```

Inside, the recipe is always the same three stages:

### Stage A — Build context, render texture

- `classify_tissue(annotation, atlas)` produces a dict of bool masks:
  `gray_matter`, `white_matter`, `ventricle`, `tissue`, `background`,
  plus per-concept masks (`dg_granule_layer`, `induseum_griseum`, etc.)
  registered in `canonical_regions.CANONICAL_REGIONS`, plus the
  aggregate `dense_cell_layers`.
- `atlas_grayscale_density_map(ref, tissue_mask, gamma, floor)` produces
  the per-pixel density modulation map — bright atlas regions = denser
  cells.
- `TransformContext` is constructed wrapping all of the above plus
  `pixel_size_um`, `plane`, `position_mm`, `modality`. This is the
  shared state every transform reads.
- A blank canvas (`np.zeros((h, w, 3), dtype=np.float32)`) is the
  starting image.
- Per-region texture transforms run in modality-specific order (see
  `MODALITIES.md`). They paint into the canvas using the masks.

### Stage B — Tonal pass

- `_apply_tone_shift(canvas, rng)` — color jitter (DAPI: no-op for pure
  blue; brightfield: cream-balance shift; ISH: NBT/BCIP color spread).
- `_apply_brightness_contrast(canvas, rng)` — mild contrast / brightness
  per-call variation. For DAPI specifically, this is *mild* (textures
  are pre-calibrated). For other modalities the gamma can be more aggressive.

### Stage C — Damage layer

`apply_damage_layer(canvas, rng=rng, ctx=ctx, modality="<modality>",
intensity=damage_intensity, geometry=apply_geometry_warp)` applies a
sequenced bundle from `damage_pipeline.py`:

```
HemibrainPreparation       (geometry-only, p=0.08)
VentricleExpansion         (geometry-only, p=0.55, MUST run pre-warp)
BladeStretchHorizontal     (geometry-only, ALWAYS — H-stretch is the defining microtome artifact)
IlluminationGradient       (always, p=0.85)
EmbeddingHalos             (light-bg only, p=0.70)
AffineJitter               (geometry-only, p=0.75)
Microbubbles               (geometry-only, p=0.35)
Debris                     (always, p=0.25)
Folds                      (geometry-only, p=0.45)
PosteriorWingDamage        (geometry-only, p=0.30)
```

Two probabilities/scales here:

- **Modality gating**: `EmbeddingHalos` only on light-background
  modalities (`nissl`, `brightfield`, `ish`); skipped for DAPI/fluorescence.
- **Geometry gating**: when `geometry=False` (set for the bbox-grounding
  bucket), pixel-displacing transforms are skipped while non-coord
  transforms (illumination, halos, debris) still run.
- **Intensity scaling**: `light` / `medium` / `heavy` scales both the
  `p` of each transform and its parameter magnitudes (`_INTENSITY_PROB_SCALE`,
  `_INTENSITY_PARAM_SCALE`).

`ResolutionShift` and `Tears` are intentionally omitted (real microscopy
is in focus; tear edge-bites looked like marker drawings).

## TransformContext — the shared state

Defined in `transforms/base.py`:

| Field | Type | Source | Purpose |
|-------|------|--------|---------|
| `modality` | str | renderer | gates modality-specific ops |
| `annotation_slice` | HW int32 | atlas | region IDs (for damage transforms that mutate region geometry) |
| `density_map` | HW float32 [0,1] | density.py | per-pixel cell-density modulation |
| `tissue_mask` | HW bool | classify_tissue | tissue / not-tissue |
| `pixel_size_um` | float | sampler | render resolution (default 25.0) |
| `tissue_class_masks` | dict[str, HW bool] | classify_tissue | gray_matter, white_matter, ventricle, isocortex, thalamus, hippocampal_formation, cortical_subplate, dense_cell_layers, plus every concept registered in canonical_regions |
| `plane` | str | sampler | coronal / sagittal / horizontal |
| `position_mm` | float | sampler | for region-aware damage gating (e.g., posterior-wing only at AP ≥ 8.5) |
| `counterstain_signal_mask` | HW float32 | counterstain layer | populated by counterstain renderers; consumed by signal renderers to avoid double-staining |
| `tract_orientation` | HW float32 | `_get_or_compute_tract_orientation` | per-pixel WM tract angle (radians); cached on first compute |

Most context fields are mutated by transforms. For example,
`HemibrainPreparation` rewrites `tissue_mask`, `annotation_slice`,
`tissue_class_masks` to reflect the kept hemisphere. Damage transforms
read the *current* state, not the pre-warp original.

## The Transform protocol

```python
def __call__(
    self,
    image: np.ndarray,           # HWC float32 [0, 1]
    *,
    rng: np.random.Generator,
    ctx: TransformContext,
) -> np.ndarray:                 # HWC float32 [0, 1]
    ...
```

Implementations:

- Receive the *current* image, an RNG (pass it through, never reseed),
  and the shared context.
- Return a new HWC float32 array, same H×W (geometry transforms can
  resize but document it; default is shape-preserving).
- Are gated by a `p` probability — if `rng.random() >= self.p`, return
  the input unchanged.
- Mutate `ctx` only when the transform's semantics require it
  (HemibrainPreparation must update masks; non-spatial transforms
  shouldn't touch ctx).

`compose([t1, t2, t3])` returns a Transform that applies in left-to-right
order. Used internally by the modality renderers.

## Mask resolution path

`classify_tissue(annotation, atlas)` is the single source for mask state.
It produces:

```
masks["background"]       = annotation == 0
masks["tissue"]           = ~background

# from get_tissue_id_sets — atlas-resolved root acronyms
masks["gray_matter"]      = annotation ∈ {grey + descendants}
masks["white_matter"]     = annotation ∈ {fiber tracts + descendants}
masks["ventricle"]        = annotation ∈ {VS + descendants}

# from _specific_structure_id_sets — explicit acronym + name fallback
masks["isocortex"]        = annotation ∈ {Isocortex + descendants}
masks["thalamus"]         = ...
masks["hippocampal_formation"] = ...
masks["cortical_subplate"]     = ...

# from canonical_regions registry — one mask per concept
masks["dg_granule_layer"]   = annotation ∈ resolved IDs for this atlas
masks["induseum_griseum"]   = ...
masks["mob_glomerular_layer"] = ...
masks["aob_granule_layer"]    = ...
masks["nlot_pyramidal_layer"] = ...
masks["ca_pyramidal_layer"]   = ...
masks["purkinje_layer"]       = ...
masks["cerebellar_granule_layer"] = ...

# aggregate of dense-layer concepts
masks["dense_cell_layers"]    = union of all DENSE_CELL_LAYER concepts
```

Atlas-specific tables live in `tissue_class._ATLAS_TISSUE_ROOTS` (Allen,
Kim, Princeton, Osten, Perens, ccfv3augmented, allen_mouse_bluebrain_barrels,
WHS rat). Unknown atlases fall through to keyword-based name search with
a UserWarning.

## Coordinate / pixel conventions

- **Canvas dimensions**: `annotation_slice.shape[:2]` from
  `oblique.get_oblique_slice`. Native atlas resolution by default
  (e.g. 320×456 for allen_mouse_25um coronal).
- **Pixel size**: 25 µm/px is the production default. Other resolutions
  work but per-modality density/intensity ranges are calibrated for 25.
- **Coordinate system**: HWC, channel order RGB.
- **Float range**: [0, 1]. Clip after each transform.

## Geometry warping rules

`HemibrainPreparation`, `VentricleExpansion`, `BladeStretchHorizontal`,
`AffineJitter`, `Folds`, and `PosteriorWingDamage` all manipulate pixel
positions or paint position-dependent features.

**Critical ordering constraint**: `VentricleExpansion` MUST run before
`BladeStretchHorizontal` / `AffineJitter` / `Folds`, because those warps
remap canvas pixels but do NOT update `ctx.tissue_class_masks["ventricle"]`.
If ventricle expansion ran after the warps, it would paint at stale pre-warp
ventricle positions. See `damage_pipeline.py` Step 0b comment.

`HemibrainPreparation` updates the masks it shifts; subsequent warps see
the updated context.

## Determinism

Every transform consumes the shared `rng` (a `numpy.random.Generator`).
Fresh RNG state per render seeded by `synth_dataset` from `--seed` plus a
per-row offset. Same `(--seed, --atlases, --n)` produces bit-identical
output. Tests rely on this in `test_augmentation_transforms.py::test_*_determinism`.

Don't reseed the RNG inside a transform. Don't use `np.random` (module-level
RNG) — always go through the passed-in generator.

## Validation harness

`augmentation/validate.py` runs sanity checks on a rendered section
(shape, dtype, range, mask coverage, no-NaN). Hook points:

- After Stage A (texture only)
- After Stage B (tonal applied)
- After Stage C (full pipeline)

The viz scripts in `scripts/viz_*.py` are the visual-QC equivalent —
each renders a comparison grid for one feature against an exemplar.

## Extending the pipeline — minimal checklist

Adding a transform:

1. Implement the class with `__call__(image, *, rng, ctx) -> ndarray`.
2. Match `Transform` protocol; use `if rng.random() >= self.p: return image`
   gating idiom.
3. Add to relevant modality renderer or to `damage_pipeline.apply_damage_layer`.
4. Register in `tests/test_augmentation_transforms.py::_CONTRACT_TRANSFORMS`
   so the shape/dtype/range/determinism contract is enforced.
5. Maintain HWC float32 [0, 1] in/out.

Adding a mask:

1. Add concept to `canonical_regions.CANONICAL_REGIONS`.
2. The mask becomes available at `ctx.tissue_class_masks["<concept>"]`
   automatically — no changes to `classify_tissue` needed.

Adding a modality:

1. Create `<modality>_pipeline.py` matching the existing renderer pattern.
2. Register in `synth_dataset.py` modality dispatch.
3. Add a row class to `tests/test_augmentation_integration.py`
   parametrized list.
4. Decide modality gating in `damage_pipeline._LIGHT_BG_MODALITIES`.

## Known gotchas

- **xarray vs numpy**: `microsim.Simulation.digital_image()` returns
  xarray. Call `.values` to get numpy. Production canvases are pure
  numpy.
- **`_splat_blobs` saturation**: The legacy blob-splat path can saturate
  if density × intensity is too high. The DAPI texture migration
  (`transforms/dapi_texture.py`) replaced the saturating path. Other
  modalities still use `_splat_blobs` — watch for saturation when
  tuning density.
- **`_get_or_compute_tract_orientation` caches on ctx**: it computes
  per-pixel WM tract direction once and stores it on `ctx.tract_orientation`.
  Subsequent transforms reuse it. Don't recompute.
- **Atlas reference type**: `atlas.reference` is uint8 by default.
  Renderers convert to float internally. Don't pass uint8 to numpy ops
  that expect float.
- **Position-mm gating**: Some damage transforms (e.g.,
  `PosteriorWingDamage`) only fire above a position threshold. Check
  the constructor's `posterior_min_position_mm` (default 8.5).

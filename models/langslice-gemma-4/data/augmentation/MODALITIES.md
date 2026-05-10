# Per-Modality Reference

Each of the five modalities has its own renderer. They share the
three-stage skeleton in `PIPELINE.md` but differ in *substrate*,
*texture transforms*, *tonal pass*, and *damage modality gating*.

## DAPI — `dapi_pipeline.py::render_dapi_section`

**Substrate**: pure black (`np.zeros((h, w, 3), dtype=np.float32)`).
DAPI fluorescence is "additive on dark."

**Texture stack** (in order):
1. `DAPIGrayMatterNuclei(p=1.0)`
2. `DAPIWhiteMatterNuclei(p=1.0)` — anisotropic blur aligned with tract orientation
3. `DAPIDenseCellLayers(p=1.0)` — paints solid bright bands in DG-sg, CA pyramidal, MOB-gl, etc.
4. `DAPIBoundaryHighlights(p=1.0)` — pia mater + ventricle wall bright pixels

All four delegate to `transforms/dapi_texture.py::render_dapi_region_texture`
with calibrated `DAPIRegionParams` per region. Cells are sub-pixel at
25 µm/px so the texture comes from per-pixel Poisson cell-count + gamma
intensity + multi-pixel bright clusters.

**Tonal pass**:
- `_apply_tone_shift` is a NO-OP. GT showed pure single-channel blue
  (R=G=0). Re-introducing R/G writes will make highlights look pink/lavender.
- `_apply_brightness_contrast`: mild — brightness 0.85–1.10, contrast
  0.95–1.15. Aggressive gamma compression was needed pre-calibration
  but not anymore.

**Damage modality**: `dapi` — dark-background. `EmbeddingHalos` skipped.

**GT calibration**: `GT_Textures/DAPI/*.png` — see `DAPI_TEXTURE_NOTES.md`.
Target B-channel stats at 25 µm/px: mean 0.20, p95 0.40, max 1.0.

**Channel writes**: Only blue (channel 2). R=G=0 throughout.

## Nissl — `nissl_pipeline.py::render_nissl_section`

**Substrate**: cream/tan canvas painted in tissue regions
(`gray_matter | white_matter`); slide-white outside. Cells SUBTRACT from
the substrate toward purple — opposite of DAPI's additive-on-dark.

`_CREAM_PRESETS` ships four tones: warm-cream thionine, neutral cream,
blue-cream cresyl-violet (Mouse10-like), purple-cream. Picked at random
unless `cream_base=` is pinned. Substrate is shaded by atlas grayscale
via `shade_substrate_by_atlas` so cell-rich regions read slightly darker
even before stained cells appear.

**Texture stack**:
1. `NisslGrayMatterCellBodies(p=1.0, cream=cream_base)` — bimodal size
   distribution (most normal-cell, ~10% giant pyramidal)
2. `NisslWhiteMatterCellBodies(p=1.0, cream=cream_base)` — sparse
   oligodendrocyte cell bodies, no axon-aligned stretching

Both apply cytoplasmic blur — each cell's delta gets blurred slightly so
the dark-purple core fades smoothly into the cream substrate, mimicking
real Nissl-stained cytoplasm under brightfield.

**Tonal pass**:
- `_apply_nissl_tone_shift`: warm/cool drift in cream tone; mimics
  scanner white-balance variation. `warm_pull ∈ [-0.05, 0.10]`,
  `cool_pull ∈ [-0.05, 0.05]`.
- `_apply_brightness_contrast`: brightness 0.85–1.10, contrast 0.85–1.15.

**Damage modality**: `nissl` — light-background. `EmbeddingHalos` runs.

**Channel writes**: All three (RGB), starting from cream substrate.

## Brightfield IHC — `brightfield_pipeline.py::render_brightfield_section`

**Substrate**: tan canvas (similar to Nissl) OR a hematoxylin
counterstain (per-call ~30% probability via `counterstain="auto"`).

**Modes** (`BRIGHTFIELD_MODES`, picked at random unless pinned):
- `pan_neuronal` — NeuN-like, dense in GM
- `sparse_interneuron` — PV/Calbindin-like, ~5–10% of GM cells, no WM
- `myelin` — Woelcke/MBP-like, dense in WM, faint GM

**Stain**: DAB chromogen (warm brown, R>G>B). Sharper edges than Nissl
(no cytoplasmic blur — DAB precipitate is crisp).

**Counterstain branches**:
- **No counterstain (legacy)**: `BrightfieldGrayMatterDAB +
  BrightfieldWhiteMatterDAB` paint directly onto a tan substrate.
- **Hematoxylin counterstain**: `render_hematoxylin_counterstain` builds
  the cellular substrate first (lighter density: 1000–2000/mm² GM,
  100–250/mm² WM — vs standalone hematoxylin 3000–4500/100–450). Then
  `apply_dab_signal` layers DAB on top with reduced intensity scale 0.55
  so brown signal doesn't overwhelm blue nuclei.

**Tonal pass**: `_apply_brightfield_tone_shift` — tan/yellow drift.
Mild brightness/contrast.

**Damage modality**: `brightfield` — light-background.

## Fluorescence — `fluorescence_pipeline.py::render_fluorescence_section`

The most architecturally complex modality — supports 13 modes via
`FLUORESCENCE_MODES` (see `modes.py`). Each mode declares
`(name, weight, counterstain, channels)` and is sampled per-image
according to weight unless pinned.

**Architecture**:
1. **Counterstain** (Stage 1): renderer from `COUNTERSTAIN_REGISTRY`
   produces the substrate. DAPI-based modes get a dark near-black canvas
   with blue nuclear dots; tract-tracing modes use Nissl with a cool
   blue-cream substrate (NeuroTrace look).
2. **Marker channels** (Stage 2): one or more `FluorescenceMarker`
   transforms layer fluorescent signal on top, per `IFChannelSpec`
   defining channel (0=red, 1=green, 2=blue), distribution (sparse /
   moderate), density, intensity, sigma.
3. **Global jitter** (Stage 3): `_apply_brightness_contrast` —
   brightness/contrast 0.85–1.15.

**Generic vs named modes**: Modes whose name starts with `generic_`
sample 1–3 channels from `_GENERIC_IF_PRESETS` at runtime. Named modes
(e.g. `dual_GFP_tdTom`) have explicit `IFChannelSpec` tuples in
`modes.py`.

**Backwards-compat**: `p_green` / `p_red` floats still accepted but
deprecated. Pass `mode=` instead.

**Damage modality**: `fluorescence` — dark-background.

## ISH — `ish_pipeline.py::render_ish_section`

In situ hybridization — counterstain + signal architecture similar to
fluorescence but for chromogenic / DAB / NBT-BCIP combinations.

**Modes** (`ISH_MODES`, weight-sampled per-image):
- `allen_style` (0.30): no counterstain, NBT/BCIP signal — Allen-style
  pale lavender substrate from atlas shading only
- `nbt_nfr` (0.30): nuclear-fast-red counterstain + NBT/BCIP signal
- `dab_hematoxylin` (0.20): hematoxylin counterstain + DAB signal
- `fish` (0.15): DAPI counterstain + fluorescent probe
- `nissl_nbt` (0.05): Nissl counterstain + NBT/BCIP signal

**Architecture**: Stage 1 counterstain (`COUNTERSTAIN_REGISTRY`) →
Stage 2 signal (`SIGNAL_REGISTRY`) → Stage 3 brightness/contrast.

The counterstain populates `ctx.counterstain_signal_mask` which the
signal layer consults so the chromogen wash does not double-stain
nuclei.

**Counterstain registry** (in `counterstain.py`):
- `none` — Allen-style pale lavender substrate
- `nfr` — nuclear fast red, calibrated against Allen ISH P56 NFR scans
- `hematoxylin` — blue-violet nuclei
- `dapi` — wraps `DAPIGrayMatterNuclei + DAPIWhiteMatterNuclei`
- `nissl` — wraps Nissl renderer

**Signal registry** (in `signals.py`):
- `nbt_bcip` — purple substrate wash + cell-restricted intense purple
- `dab` — warm brown precipitate
- `fluorescent_probe` — red/green/yellow puncta

**Damage modality**: `ish` — light-background.

## Per-modality damage gating

`damage_pipeline._LIGHT_BG_MODALITIES = {"nissl", "brightfield", "ish"}`.
For light-background modalities `EmbeddingHalos` runs (warm halo outside
tissue is biologically plausible only on light backgrounds — on
dark-field DAPI/fluorescence a warm halo there would look obviously
wrong).

All other damage transforms (`HemibrainPreparation`, `BladeStretchHorizontal`,
`AffineJitter`, `Microbubbles`, `Folds`, `Tears`, `VentricleExpansion`,
`PosteriorWingDamage`, `Debris`, `IlluminationGradient`) run regardless
of modality, gated only by their own `p` and the global
`apply_damage` / `geometry=` flags.

## Calibration sources

| Modality | GT location | Notes |
|----------|-------------|-------|
| DAPI | `GT_Textures/DAPI/*.png` | 4 reference images, GT-calibrated, see `DAPI_TEXTURE_NOTES.md` |
| Nissl | `exemplars/` (informal) | Cell color presets calibrated against thionine / cresyl-violet imagery |
| Brightfield | None yet | Calibrated against Allen Brain Atlas IHC hand-eye |
| Fluorescence | None yet | Multi-channel, parameter ranges from literature |
| ISH | None yet | NBT/BCIP, NFR, DAB calibrated against Allen ISH P56 scans (per docstrings in `signals.py`) |

When calibrating a new modality, drop GT references into
`GT_Textures/<MODALITY>/` and follow the pattern in
`DAPI_TEXTURE_NOTES.md`.

## Per-modality default damage intensity

`SynthSpec.damage_intensity` is sampled per-row from
`{"light": 0.20, "medium": 0.55, "heavy": 0.18}` (with 0.07 of rows
having `apply_damage=False` for clean baselines). Modality choice doesn't
shift these probabilities — same distribution across all modalities.

## Relevant viz harnesses

For each modality there's at least one visual QC script:

- `scripts/viz_dapi_textures.py`, `viz_dapi_diversity.py`,
  `viz_dapi_wm_diversity.py`, `viz_dapi_gamma_sweep.py`
- `scripts/viz_nissl_compare.py`, `viz_nissl_diversity.py`,
  `viz_damage_nissl_before_after.py`
- `scripts/viz_brightfield_compare.py`, `viz_brightfield_modes.py`,
  `viz_hematoxylin_compare.py`, `viz_hematoxylin_diversity.py`
- `scripts/viz_fluorescence_diversity.py`, `viz_fluorescence_modes.py`
- `scripts/viz_ish_compare.py`, `viz_ish_diversity.py`,
  `viz_ish_modes.py`, `viz_nfr_compare.py`, `viz_nfr_diversity.py`
- `scripts/viz_modality_overview.py` — all five at once
- `scripts/viz_damage.py` — damage layer isolated
- `scripts/viz_oblique.py` — tilted slices
- `scripts/viz_posterior_wing_damage.py` — posterior wing detachment
- `scripts/viz_tissue_class.py` — mask visualization
- `scripts/viz_dapi_gamma_sweep.py` — DAPI gamma parameter sweep

Run them with `PYTHONPATH="models/langslice-gemma-4/data:src" python scripts/viz_<name>.py`.
Output goes to `tmp/outputs/`.

# Synthetic Histology Pipeline — Agent Guide

**You are working on a procedural synthetic histology image generator.** It
produces BrainGlobe-atlas-aligned 2D coronal/sagittal/horizontal sections in
five modalities (DAPI, Nissl, brightfield IHC, fluorescence, ISH) at 25 µm/px,
HWC float32 [0,1]. Output trains a Gemma-4 VLM on AP-position estimation —
**procedural label control is non-negotiable.**

Read this file first. It points you at everything else.

## Five-second mental model

```
synth_dataset.py write
  → for each (atlas, plane, position_mm):
      → get_oblique_slice(atlas, …) → (reference, annotation)
      → classify_tissue(annotation, atlas) → masks dict
      → atlas_grayscale_density_map(reference, tissue) → density
      → render_<modality>_section(reference, annotation, atlas, …)
          ├─ build TransformContext
          ├─ Stage A: per-region texture (texture.py / dapi_texture.py)
          │           — counterstain layer, then signal layer, then highlights
          ├─ Stage B: tone-shift + brightness/contrast
          └─ Stage C: damage layer (damage_pipeline.apply_damage_layer)
      → save PNG to _local/synth_data/images/{seed}.png
      → append row to _local/synth_data/manifest.jsonl
```

## Where things live

| What | File | Read when |
|------|------|-----------|
| **Entry point + manifest writer** | `synth_dataset.py` | Driving regen runs |
| **CLI examples** | `augmentation/cli.py` | One-off section renders |
| **Per-modality renderers** | `augmentation/{dapi,nissl,brightfield,fluorescence,ish}_pipeline.py` | Modality-level changes |
| **Damage layer** | `augmentation/damage_pipeline.py` + `transforms/damage.py` | Bubbles, illumination, blade stretch, hemibrain, posterior wing |
| **Cell texture** | `transforms/texture.py` | Per-modality cell renderers (Nissl bodies, ISH puncta, etc.) |
| **DAPI texture (GT-calibrated)** | `transforms/dapi_texture.py` + `DAPI_TEXTURE_NOTES.md` | DAPI texture tuning |
| **Geometry warps** | `transforms/geometry.py` | Affine/blade-stretch/resolution-shift |
| **Tonal stage** | `transforms/tonal.py` | Per-modality contrast/saturation jitter |
| **Tissue masks** | `transforms/tissue_class.py` | gray_matter, white_matter, ventricle, plus per-concept masks |
| **Region registry** | `augmentation/canonical_regions.py` | Cross-atlas anatomical concepts (DG-sg, Purkinje, etc.) |
| **Counterstain registry** | `augmentation/counterstain.py` + `signals.py` + `modes.py` | ISH/IHC layered rendering |
| **Density priors** | `augmentation/density.py` | Atlas-grayscale-density modulation |
| **Oblique slice extraction** | `augmentation/oblique.py` | Tilted slices (yaw/pitch/roll ≤ 8°) |
| **Validation** | `augmentation/validate.py` | Sanity-check rendered sections |
| **Visualization scripts** | `augmentation/scripts/viz_*.py` | Per-feature visual QC harnesses |
| **Exemplar GT references** | `augmentation/GT_Textures/` + `exemplars/` | Real histology screenshots for calibration |

## The deep docs

- **[PIPELINE.md](PIPELINE.md)** — TransformContext, mask flow, the
  Stage A / B / C breakdown, `Transform` protocol details, gotchas.
- **[MODALITIES.md](MODALITIES.md)** — Per-modality recipe: which texture
  + tonal + counterstain + signal transforms each modality uses, default
  parameter ranges, GT calibration sources.
- **[REGIONS.md](REGIONS.md)** — `canonical_regions` deep dive: how to add
  a concept, atlas-pattern matching with `fnmatch`, regex fallback, what
  resolves where for each atlas family.
- **[RECIPES.md](RECIPES.md)** — Task-oriented playbooks: "add a new
  region-specific texture", "calibrate against new GT", "add an atlas",
  "diagnose a visual artifact".
- **[DAPI_TEXTURE_NOTES.md](DAPI_TEXTURE_NOTES.md)** — Specific
  calibration story for DAPI; pattern for future Nissl/IHC calibrations.
- **[vendored_docs/README.md](vendored_docs/README.md)** — Index of local
  copies of microsim / DeepTrack2 / MicroscPSF-Py docs (so you don't need
  context7 for those).

## Hard rules — read before editing anything

1. **Stay in the synth lane.** When working on synthetic data, the only
   writable surfaces are: `_local/synth_data/`, `models/langslice-gemma-4/data/augmentation/`,
   `models/langslice-gemma-4/data/synth_dataset.py`, and tests for those. Do NOT touch
   `_local/qc_app/`, `data/manifest/`, or any other module — write a handoff
   note instead. Past sessions destroyed hours of curation by drifting outside
   their lane. (See `feedback_synth_lane_only` memory.)

2. **The pipeline is atlas-agnostic.** Renderers must work against any
   BrainGlobe atlas (Allen mouse 10/25/50µm, Kim mouse, Princeton, Osten,
   Perens, ADMBA, Waxholm rat, etc.). The only atlas-specific entries
   live in `tissue_class._ATLAS_TISSUE_ROOTS` and `canonical_regions.CANONICAL_REGIONS`.
   Don't hard-code Allen acronyms anywhere else.

3. **Canvas convention is HWC float32 in [0, 1].** Every transform takes
   and returns this exact shape and dtype. Never ints, never grayscale,
   never [0, 255]. Convert at boundaries only.

4. **Procedural label control is sacred.** Don't introduce probabilistic
   stages that could shift region boundaries. AP labels, tissue class
   masks, and the annotation slice MUST remain pixel-true to the atlas.
   This rules out generative-model post-processing as a primary renderer
   (no GAN/diffusion masking the output).

5. **No tone bleed for DAPI.** GT showed pure single-channel blue (R=G=0).
   `_apply_tone_shift` is a no-op in `dapi_pipeline.py` and the texture
   transforms write only to channel 2. Don't reintroduce R/G writes
   without verifying against GT.

## How to find the right edit point

| Symptom / change | Look in |
|------------------|---------|
| Texture too bright/dim | Region renderer in `transforms/texture.py` or `dapi_texture.py` |
| Cells wrong shape/size | Same as above; or `_splat_blobs` / `_splat_anisotropic_blobs` helpers |
| Mounting bubbles look fake | `transforms/damage.py::Microbubbles` |
| Illumination uniform/wrong | `transforms/damage.py::IlluminationGradient` |
| New region needs distinct texture | Add to `canonical_regions.CANONICAL_REGIONS` + new transform that asks for the concept mask |
| New atlas not classifying tissue right | `transforms/tissue_class._ATLAS_TISSUE_ROOTS` + name-pattern fallback in `_specific_structure_id_sets` |
| Section angle / oblique slicing wrong | `augmentation/oblique.py` — angle cap is `_MAX_ANGLE_DEG` |
| New modality | Add `<modality>_pipeline.py` + register in `synth_dataset.py` |
| Color/tone wrong for an existing modality | `transforms/tonal.py` (per-modality classes) + `<modality>_pipeline.py` |

## The three external libraries

(Installed; vendored docs in `vendored_docs/`.)

- **MicroscPSF-Py** — minimal Gibson-Lanni PSF generator. Returns 2D PSF
  kernel ready for `scipy.ndimage.convolve`. Use when adding optical
  blur/PSF.
- **DeepTrack2** — modular microscopy primitives; every `Feature` has
  `.resolve(np.ndarray)`. Use for runtime additive noise (`dt.noises.Poisson`,
  `dt.noises.Gaussian`). Avoid chaining many `&`-composed scatterers
  (slow / hangs).
- **microsim** — full forward fluorescence simulator. Use as an *offline
  stamp factory* (bake per-fluorophore cell stamps with FPbase spectra,
  paste at runtime). Don't call it on the runtime hot path.

When uncertain which to use: check `vendored_docs/README.md`.

## When you're done

Verify before committing:

```bash
# Augmentation tests must pass (currently 125/125)
pytest tests/test_augmentation_transforms.py tests/test_augmentation_integration.py

# End-to-end regen
PYTHONPATH="models/langslice-gemma-4/data:src" \
    python models/langslice-gemma-4/data/synth_dataset.py write \
    --out _local/synth_data --n 50 --seed 20260507 --atlases allen_mouse_25um

# Histogram audit (replace seed)
python -c "import numpy as np; from PIL import Image; \
    img = np.asarray(Image.open('_local/synth_data/images/<seed>.png')); \
    b = img[..., 2][img.sum(2) > 0]; \
    print(f'B mean={b.mean()/255:.3f} p95={np.percentile(b,95)/255:.3f}')"
```

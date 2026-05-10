# Task Recipes

Playbooks for the tasks that come up most often. Each recipe lists
files to touch, expected outcome, and verification command.

## Recipe — Calibrate a modality texture against new GT

**When**: User drops new GT screenshots into `GT_Textures/<MODALITY>/`
and asks for the synth output to match.

**Steps**:
1. Read the GT and characterize per-channel histograms:
   ```python
   img = np.asarray(Image.open(...))
   for ch_i, name in enumerate('RGB'[:img.shape[2]]):
       arr = img[..., ch_i].astype(np.float32) / 255.0
       print(name, arr.mean(), np.median(arr), np.percentile(arr, 95), arr.max())
   ```
   Note the dominant channel(s) and target percentile values.

2. Build a sweep harness in `tmp/<modality>_iteration/`:
   - `compare.py` with `load_gt`, `save_comparison`, `stats`
   - `iter1_*.py` running the current renderer to baseline
   - Side-by-side comparison panels (GT crop vs candidate)

3. Iterate parameters in the relevant texture transform until histogram
   stats match within ~10% on every percentile (mean / median / p95 /
   max). Target convergence: 3–5 iterations.

4. If saturating with hand-tuned blob splat, switch to per-pixel Poisson
   sampling (the DAPI pattern in `transforms/dapi_texture.py`).

5. Lock parameters into the relevant transform (e.g. `DAPIRegionParams`
   instances for DAPI, or in-class defaults for other modalities).

6. **Verify**:
   - `pytest tests/test_augmentation_transforms.py tests/test_augmentation_integration.py` — 125/125
   - `python models/langslice-gemma-4/data/synth_dataset.py write --out _local/synth_data --n 50 ...`
   - Histogram audit on regenerated PNGs

**Reference**: `DAPI_TEXTURE_NOTES.md` documents the full DAPI calibration story.

## Recipe — Add a region-specific texture pass

**When**: A specific anatomical structure should look distinctive
(e.g. Purkinje cells visible as a single bright line).

**Steps**:
1. Verify the structure resolves in `canonical_regions` for your target
   atlases. If it's not registered, add it (see `REGIONS.md::Adding a
   new concept`). Confirm `concept_id_set(atlas, "<concept>")` returns
   non-empty.

2. Choose your renderer pattern. Two options:
   - **Reuse `render_dapi_region_texture`** with a new `DAPIRegionParams`
     instance — fastest, calibrated against GT statistics.
   - **Custom transform** — when the structure needs visually distinctive
     patterns (e.g. Purkinje cells = single big flask-shaped neurons in
     a line; cerebellar granule = extremely dense small nuclei).

3. Add the new transform to the relevant modality renderer in
   `<modality>_pipeline.py`. Order matters — most region-specific passes
   run AFTER general-tissue passes so they overpaint where needed.

4. Add to `tests/test_augmentation_transforms.py::_CONTRACT_TRANSFORMS`
   so the shape/dtype/range/determinism contract is enforced.

5. **Verify**:
   - Run a small regen and visually QC the targeted structure.
   - Render with `mode=` pinned and `apply_damage=False` to isolate the
     texture in question.

**Reference**: `DAPIDenseCellLayers` is the canonical example of a
region-specific texture pass that paints solid bright bands in
hippocampal dense layers.

## Recipe — Tune a damage transform

**When**: User flags a damage artifact as looking fake (bubbles too
big, illumination too uniform, tears too marker-like).

**Steps**:
1. Open `transforms/damage.py` and find the offending class.

2. Render with **only** that transform pinned at `p=1.0` and everything
   else off:
   ```python
   from augmentation.transforms.damage import Microbubbles
   t = Microbubbles(p=1.0, n_bubbles_range=(2, 2), radius_range=(8, 8))
   out = t(canvas, rng=np.random.default_rng(0), ctx=ctx)
   ```
   The `scripts/viz_damage.py` harness does this for the whole damage layer.

3. Tune the constructor parameters until the visual matches reference.
   Don't change defaults until the user confirms — make the new
   defaults match the new visual target.

4. If the artifact's underlying physics are wrong (not just parameters),
   look at the helper functions:
   - `_make_open_curve_points` — random curve sampling
   - `_build_displacement_field` — TPS warp
   - `_apply_displacement` — pixel remap
   - `_sample_bg_color` — slide-background color from corners
   - `_fray_along_mask_edge` — ragged edge generator

5. Update `damage_pipeline.py` if you're changing the damage layer
   ordering or modality-gating, not just one transform's behavior.

6. **Verify**:
   - Tests still pass (`pytest tests/test_augmentation*`).
   - `scripts/viz_damage.py` shows the new behavior.
   - Regen a sample batch and QC.

**Reference**: The Microbubbles + IlluminationGradient + Tears
re-tuning sessions are documented in the conversation history. Edge
bites were ultimately removed because they always looked like marker
drawings.

## Recipe — Add support for a new BrainGlobe atlas

**When**: User wants to add data from a new species / dev-stage
(rat, ferret, marmoset, ADMBA new age, etc.).

**Steps**:
1. Find the atlas's tissue-class root acronyms. Run:
   ```python
   atlas = load_atlas('<atlas_name>')
   for s in atlas.structures.values():
       if 'grey' in str(s.get('name','')).lower() or 'gray' in str(s.get('name','')).lower():
           print(s.get('id'), s.get('acronym'), s.get('name'))
   # repeat for 'fiber tract', 'ventricle'
   ```

2. Add the atlas to `_ATLAS_TISSUE_ROOTS` in
   `transforms/tissue_class.py`:
   ```python
   _ATLAS_TISSUE_ROOTS["new_atlas"] = {
       "gray_matter": "<acronym>",
       "white_matter": "<acronym>",
       "ventricle": "<acronym>",
   }
   ```
   If you skip this, the keyword fallback runs with a UserWarning —
   acceptable for one-off testing but always add the explicit mapping
   for production atlases.

3. (Optional) Add per-concept acronym mappings to `canonical_regions`
   for region-specific texturing. Even if you skip this, the
   `name_pattern` fallback covers atlases sharing English-language
   anatomical naming.

4. (Optional) Calibrate per-modality texture parameters. Different
   species have different cell densities — Allen mouse cortex is ~5500
   nuclei/mm²; rat is denser. If using the GT-calibrated DAPI renderer,
   adjust `DAPIRegionParams.cells_per_mm2` per atlas family.

5. **Verify**:
   - `classify_tissue` produces non-empty masks for the new atlas.
   - `synth_dataset.py write --atlases <new_atlas>` runs without errors.
   - Visual QC.

**Reference**: `_ATLAS_TISSUE_ROOTS` docstring lists the existing 11
atlases.

## Recipe — Diagnose a visual artifact

**When**: User shows a screenshot of synth output with a visible problem.

**Steps**:
1. Identify the seed in the QC app and locate the file
   (`_local/synth_data/images/<seed>.png`).

2. Read the manifest row:
   ```python
   for line in open('_local/synth_data/manifest.jsonl'):
       r = json.loads(line)
       if r.get('seed') == <seed>: print(json.dumps(r, indent=2))
   ```
   Note `modality`, `mode`, `damage_intensity`, `apply_damage`,
   `apply_geometry_warp`, `position_mm`, oblique angles.

3. Reproduce by calling the renderer directly:
   ```python
   from augmentation.<modality>_pipeline import render_<modality>_section
   from langslice_harness.atlas.core import load_atlas
   from augmentation.oblique import get_oblique_slice
   atlas = load_atlas(r['atlas_name'])
   ref, ann = get_oblique_slice(atlas, base_position_mm=r['position_mm'],
                                plane=r['plane'], yaw_deg=r['yaw_deg'],
                                pitch_deg=r['pitch_deg'], roll_deg=r['roll_deg'])
   out = render_<modality>_section(ref, ann, atlas, seed=r['seed'], pixel_size_um=25.0,
                                   damage_intensity=r['damage_intensity'],
                                   apply_damage=r['apply_damage'])
   ```

4. **Bisect**: Render with `apply_damage=False` first. If artifact
   disappears → it's in the damage layer; check each damage transform
   one by one.

5. **Texture**: If artifact present even with damage off, it's in the
   texture / counterstain / signal stack. Render with `mode=` pinned
   to known-good and see if it persists.

6. **Mask**: If artifact correlates with a region (e.g. only in CA1),
   inspect `ctx.tissue_class_masks` for that mask.

7. **Histograms**: If artifact is "too bright/dim/saturated", run a
   histogram audit — check per-channel mean, median, p95, max against
   GT targets.

8. Once root-caused, fix at the source. Don't add a post-hoc clip /
   workaround unless the underlying issue is genuinely unfixable.

## Recipe — Regenerate the synth batch

**When**: User asks for fresh QC after pipeline changes.

```bash
PYTHONPATH="models/langslice-gemma-4/data:src" \
    python models/langslice-gemma-4/data/synth_dataset.py write \
    --out _local/synth_data \
    --n 300 --seed 20260507 \
    --atlases allen_mouse_25um
```

For multi-atlas variety:
```bash
... --atlases allen_mouse_25um kim_mouse_25um whs_sd_rat
```

The `--seed` is deterministic — same seed produces bit-identical output.
Use a stable canonical seed (e.g. `20260507`) for reproducible QC.

To wipe completely (only when user authorizes; some past sessions
destroyed in-progress curation):

```bash
rm -f _local/synth_data/manifest.jsonl
rm -f _local/synth_data/images/*.png
# then regenerate
```

## Recipe — Run the test suite

```bash
pytest tests/test_augmentation_transforms.py tests/test_augmentation_integration.py -x -q
```

Should print `125 passed`. Anything less means a regression.

For just the contract tests (shape/dtype/range/determinism — fast):
```bash
pytest tests/test_augmentation_transforms.py -x -q
```

For just integration (per-modality end-to-end — slower):
```bash
pytest tests/test_augmentation_integration.py -x -q
```

## Recipe — Use one of the new external libraries

Three libraries are installed (`pip show MicroscPSF-Py deeptrack microsim`):

### MicroscPSF-Py — for runtime PSF blur

```python
import microscPSF.microscPSF as msPSF
import numpy as np
from scipy.ndimage import convolve

mp = dict(msPSF.m_params)
mp["NA"] = 0.45
mp["wvl"] = 0.461
psf_xyz = msPSF.gLXYZParticleScan(mp, pixel_size_um=25.0, n_pixels=5,
                                  pv=np.array([0.0]), zv=0.0)
psf_2d = psf_xyz[0]; psf_2d /= psf_2d.sum()
canvas[..., 2] = convolve(canvas[..., 2], psf_2d, mode='reflect')
```

### DeepTrack2 — for runtime additive noise

```python
import deeptrack as dt
canvas = dt.noises.Poisson(snr=20).resolve(canvas)
canvas = dt.noises.Gaussian(mu=0.0, sigma=0.005).resolve(canvas)
```

Don't chain hundreds of `&`-composed scatterers — feature graph hangs.
Use as standalone primitives only.

### microsim — for offline stamp factory (planned, not yet integrated)

Use to bake per-fluorophore cell stamps offline (DAPI, EGFP, EYFP,
mCherry, td-Tomato, Alexa647). Not on the runtime hot path. See
`vendored_docs/microsim/docs/concept.md` for the simulator structure
when ready to implement.

## Recipe — When stuck, reach for these references

| What you're looking for | Read |
|--------------------------|------|
| "What does this transform do?" | `transforms/<file>.py` docstrings |
| "Where does this run in the pipeline?" | `PIPELINE.md` |
| "What does this modality do?" | `MODALITIES.md` |
| "How do I add a region?" | `REGIONS.md` |
| "How did the DAPI calibration work?" | `DAPI_TEXTURE_NOTES.md` |
| "What does microsim / DT2 / MicroscPSF support?" | `vendored_docs/<lib>/` |
| "What's the user's preference here?" | `~/.claude/projects/.../memory/` |
| "What can I touch / not touch?" | `AGENT_GUIDE.md::Hard rules` |

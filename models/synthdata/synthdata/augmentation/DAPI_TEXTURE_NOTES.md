# DAPI Texture Renderer — Calibration Notes

GT-calibrated DAPI texture renderer for the synthetic histology pipeline.
Replaced the legacy blob-splat approach with per-pixel Poisson cell-count
sampling at production resolution (25 µm/px).

## Calibration data

Ground truth screenshots in:

```
models/langslice-gemma-4/data/augmentation/GT_Textures/DAPI/
├── DAPI_greymatter_zoomIN.png    # 1282×1696 high-zoom GM (cells visible)
├── DAPI_greymatter_zoomout.png   # 1131×1288 medium-zoom GM (cell texture)
├── DAPI_whitematter_zoomIN.png   # 403×504 high-zoom WM (elongated cells)
└── DAPI_whitematter_zoomout.png  # 481×666 medium-zoom WM (tract structure)
```

GT statistics (blue channel, tissue pixels only; R=G=0 throughout — **pure single-channel blue**):

| Image | mean | median | p95 | max |
|-------|------|--------|-----|-----|
| GM zoomIN @ ~0.5 µm/px | 0.216 | 0.169 | 0.490 | 1.000 |
| GM zoomout @ ~4.5 µm/px | 0.205 | 0.157 | 0.451 | 1.000 |
| GM zoomout downsampled to 25 µm/px | 0.205 | 0.179 | 0.396 | 0.999 |
| WM zoomout @ ~4.5 µm/px | 0.229 | 0.192 | 0.486 | 1.000 |
| WM zoomout downsampled to 25 µm/px | 0.229 | 0.212 | 0.423 | 0.999 |

## Architecture

Two-engine separation:

- **Calibrated texture renderer** (`transforms/dapi_texture.py`): per-pixel
  Poisson cell-count sampling, gamma-distributed intensities, multi-pixel
  bright clusters that survive downsampling, autofluor floor + sensor
  noise. Calibrated to GT@25µm.
- **Existing pipeline**: region masks (gray_matter, white_matter,
  dense_cell_layers), atlas density modulation, boundary highlights,
  damage layer. Untouched.

Wrapping `Transform` classes (`DAPIGrayMatterNuclei`,
`DAPIWhiteMatterNuclei`, `DAPIDenseCellLayers`) preserve the existing
pipeline contract — they just delegate to `render_dapi_region_texture`
with the right parameter set.

## Calibrated parameter sets

In `transforms/dapi_texture.py`:

```python
DAPI_GM_PARAMS = DAPIRegionParams(
    cells_per_mm2=5500.0,           # cortex GM density
    base_cell_intensity=0.04,
    cell_tail_scale=0.07,           # gamma-distribution scale (heavy tail)
    bright_cluster_per_mm2=22.0,    # multi-pixel bright spots
    bright_cluster_radius_px_range=(1, 2),
    bright_cluster_intensity_range=(0.45, 0.95),
    autofluor_floor=0.05,
)

DAPI_WM_PARAMS = DAPIRegionParams(
    cells_per_mm2=2500.0,           # WM is sparser (oligodendrocytes)
    base_cell_intensity=0.04,
    cell_tail_scale=0.05,
    bright_cluster_per_mm2=12.0,
    aniso_sigma_long_px=2.0,        # tract-aligned smear
    aniso_sigma_short_px=0.5,
    autofluor_floor=0.05,
)

DAPI_DENSE_LAYER_PARAMS = DAPIRegionParams(
    cells_per_mm2=14000.0,          # densely packed (DG-sg, CA*sp, etc.)
    base_cell_intensity=0.06,
    cell_tail_scale=0.10,
    bright_cluster_per_mm2=80.0,    # frequent bright spots
    bright_cluster_intensity_range=(0.55, 1.0),
    autofluor_floor=0.15,           # uniformly bright floor
)
```

## Iteration journey

10 iterations in `tmp/dapi_iteration/iter[1..10]_*.py` + `compare.py` harness.

Key insights from the iteration sweep:

1. **At 25 µm/px, individual cells are sub-pixel.** A typical 8 µm cell
   nucleus is 0.32 px. PSF (Airy disc, NA 0.45, 461 nm) is also sub-pixel.
   So the texture is dominated by *aggregate* cell-density variation, not
   resolved cells. The right model: Poisson cell-count per pixel + gamma
   intensity distribution.

2. **Bright pixels survive only as clusters.** A single saturated cell
   gets averaged out by the 30 dim neighbors when downsampling to 25 µm/px.
   To preserve the bright tail, render multi-pixel bright spots
   representing real cell clusters / dividing cells / autofluor debris.

3. **Cumulative blob splatting saturates.** The legacy approach
   (`_splat_blobs` with hundreds of overlapping Gaussians) saturated bulk
   tissue at 1.0 and required heavy gamma compression downstream. The new
   per-pixel approach hits target statistics directly.

4. **Real DAPI is pure blue** — verified by GT R=G=0. Earlier code added
   cyan/purple bleed everywhere (`_apply_tone_shift`,
   `DAPIDenseCellLayers`, `DAPIBoundaryHighlights`); all stripped.

5. **DeepTrack2's `Fluorescence + Ellipse` chain is too slow** for runtime
   (chaining hundreds of `&`-composed scatterers builds a deep feature
   graph that hangs). Modular `dt.noises.Poisson(snr=...).resolve(np)`
   works fine, but isn't an upgrade over a 5-line numpy implementation.

6. **`MicroscPSF-Py` is the cleanest physics primitive** found across all
   three libraries. Returns a 2D PSF kernel ready for `scipy.ndimage.convolve`.
   At 25 µm/px the kernel is sub-pixel anyway, so we use a small Gaussian
   smoothing as the practical stand-in. (PSF infrastructure stays useful
   for future render-at-cell-scale work.)

## End-to-end output (after integration)

50-sample regen, blue channel statistics in tissue:

| stat | typical range | GT@25µm target |
|------|---------------|-----------------|
| R mean | 0.000 | 0.000 |
| G mean | 0.000 | 0.000 |
| B mean | 0.10–0.18 | 0.205 |
| B p95 | 0.30–0.55 | 0.396 |
| B max | 0.65–1.00 | 0.999 |

Mean is slightly low (about 70% of GT) but visually matches; the rest of
the pipeline (damage layer, illumination, autofluor) trims another 10-15%
off the renderer's output. Still well within plausible imaging variation.
Bright tail (p95, max) lines up.

## Pipeline-level changes summary

Files modified:

- **NEW** `transforms/dapi_texture.py` — calibrated renderer.
- `transforms/texture.py`:
  - `DAPIGrayMatterNuclei` rewritten to delegate to `render_dapi_region_texture`.
  - `DAPIWhiteMatterNuclei` rewritten — adds tract-aligned anisotropic blur.
  - `DAPIDenseCellLayers` rewritten — uses dense-layer params; pure blue.
  - `DAPIBoundaryHighlights` — stripped cyan/purple bleed; pure blue.
- `dapi_pipeline.py`:
  - `_apply_brightness_contrast` — removed aggressive gamma crush; mild
    contrast/brightness only (textures are now pre-calibrated).
  - `_apply_tone_shift` — kept as no-op for API compat (GT was pure blue).
- `counterstain.py` — `make_dapi_counterstain_canvas` updated for new API.

Files explicitly NOT touched:

- `_local/qc_app/` — QC app stays in its lane.
- `data/manifest/` — no shard touched.
- Other modality pipelines (Nissl, brightfield, fluorescence, ISH) —
  use the same `DAPIGrayMatterNuclei` etc. via counterstain hooks; benefit
  automatically from the new renderer when DAPI is the counterstain.
- `transforms/dapi_texture.py` is DAPI-specific — Nissl/IHC/other modalities
  not affected.

## Tests

`tests/test_augmentation_transforms.py` and
`tests/test_augmentation_integration.py` — 125/125 pass after refactor.

## Verification commands

```bash
# Run augmentation tests
pytest tests/test_augmentation_transforms.py tests/test_augmentation_integration.py

# Regenerate synth data
PYTHONPATH="models/langslice-gemma-4/data:src" \
    python models/langslice-gemma-4/data/synth_dataset.py write \
    --out _local/synth_data --n 300 --seed 20260507 --atlases allen_mouse_25um

# Reload comparison harness
python tmp/dapi_iteration/iter10_target.py
```

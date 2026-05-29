# microscope — code map

Physically-based DAPI microscopy simulation: build a 3-D fluorophore ground truth
(cell nuclei placed in a tissue volume) and image it through the Keyence BZ-X800
optics (microsim PSF + detector model). Self-contained `uv` project, independent of
`langslice_harness`.

## Layout
- `pyproject.toml`, `uv.lock` — uv project; base deps + optional `[gpu]` (cupy) and
  `[cosem]` extras; `numpy` pinned `<2`.
- `src/microscope/` — the package (relative imports).
  - `data/` — reference tables (nucleus sizes, per-region cell density).
  - `assets/` — COSEM nucleus shape bank (`.npy`).
- `out/` — render outputs (gitignored).

## Modules (`src/microscope/`)
- `gpu_engine.py` — optics pipeline (PSF convolve → detector); backend-agnostic
  (`xp=numpy` runs on CPU, `xp=cupy` on GPU).
- `gpu_truth.py` — nucleus template banks + 3-D truth stamping.
- `nuclei.py` — analytic nucleus model + cell-type table.
- `white_matter.py` — white-matter tissue generation.
- `nuclei_sizes.py` — nucleus-size table loader (`data/size_priors.csv`).
- `density_lookup.py` — per-region cell density loader (`data/density_by_region.csv`).
- `lab_optics.py` — Keyence BZ-X800 objective / filter / detector config.
- `render_shape_texture.py` — nucleus texture + the `style_navy` DAPI (blue) colormap.
- `cosem_shapes.py` — (re)generate the COSEM shape bank (needs the `[cosem]` extra).
- `gpu_fft.py` — GPU FFT-convolution / CUDA setup helpers.
- `render_frame.py`, `render_large_patch.py` — GPU render entry points.
- `demo_wm_arch.py` — CPU demo entry point.

## Run
```bash
cd models/synthdata/microscope
uv sync                                     # CPU env
uv run python -m microscope.demo_wm_arch    # CPU demo
# GPU renders:
uv sync --extra gpu
uv run --extra gpu python -m microscope.render_frame
```

Large calibration data and exploratory scripts live outside this package in
`_local/synth_data/_microsim_test/` (gitignored, machine-local).

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
conda env create -f environment.yml
conda activate langslice
pip install -e .

# Run tests
python -m pytest                     # full suite
python -m pytest tests/test_foo.py   # single file
python -m pytest -k "test_name"      # single test by name

# Lint and type-check
python -m ruff check .
python -m basedpyright

# Run the app
langslice version
langslice estimate <image> [--atlas ...] [--model ...]
langslice register <image> --position <mm> [--workflow ...] [--out ...]

# Tauri GUI
cd tauri-gui && pnpm tauri dev
```

## Architecture

LangSlice registers histology slice images to BrainGlobe atlases using Gemini vision-language models for estimation and deterministic local solvers for geometry. The desktop GUI is a Tauri app (Rust + React + Three.js) in `tauri-gui/`.

**Pipeline:** `AP estimation → colored segmentation → Elastix B-spline registration → preview/export`

The code is split into five modules with clear boundaries:

- **`langslice/atlas/`** — BrainGlobe atlas loading, AP/index conversion, coronal slice extraction, colored region and smoothed boundary helpers (`get_colored_region_slice()`, `get_smoothed_boundary_slice()`). Orientation assumptions are centralized in `space.py` and require coronal layout (AP/DV/ML on axes 0/1/2).
- **`langslice/ai/`** — Gemini client configuration (`config.py`), multi-turn AP estimator split across `estimator.py`, `estimator_tools.py`, and `estimator_debug.py`, offline batch helper (`batch_eval.py`). Three auth backends: `ai_studio`, `vertex_api_key`, `vertex_adc`.
- **`langslice/registration/`** — Shared utilities and workflow router (`agents.py`), colored segmentation workflow (`agents_colored_segmentation.py`, default for image-gen models), legacy two-shot workflow (`agents_image_gen.py`), experimental tool-loop workflow (`agents_tool_loop.py`, on hold), deterministic affine and TPS fitting (`solver.py`), orchestration and debug artifacts (`runtime.py`), data classes (`types.py`). Public entry point: `estimate_registration_runtime(...)` in `core.py`. The colored segmentation workflow uses itk-elastix for B-spline registration.
- **`tauri-gui/`** — Tauri desktop app. Rust backend (`src-tauri/`) for atlas loading, reslicing, mesh serving. React + Three.js frontend (`src/`) for 3D visualization, dashboard, split/overlay views. Launched via `cd tauri-gui && pnpm tauri dev`.
- **`langslice/export.py`** — Coronal anchoring math and QUINT/ABBA-compatible single-slice JSON export. `SliceExport.markers` supports VisuAlign `[ox, oy, nx, ny]` pairs from Elastix B-spline control points.

Supporting utilities: `image_prep.py` (normalization, pixel-size detection, VLM downsampling), `agent_trace.py` (structured trace events), `cli.py` (argparse entry point).

## Gemini API — ALWAYS Check Docs First

The google-genai SDK is newer than your training data. **Before writing or modifying ANY code that uses the google-genai SDK**, you MUST look up the relevant API in context7 (`/googleapis/python-genai`). Never assume you know the API surface — check types, method signatures, and supported parameters before writing code. This applies to:

- `langslice/ai/` (client config, estimator, batch)
- `langslice/registration/` (agents, tool loop, image gen)
- Any file importing `google.genai` or `google.genai.types`

Common things to verify: `GenerateContentConfig` fields, `types.Part` constructors, Interactions API parameters, File API methods, thinking config, media resolution options.

## Key Conventions

- AP coordinates are atlas-native millimeters from the anterior edge of the volume.
- The colored segmentation workflow (default for image-gen models) registers grayscale versions of the atlas colored regions (moving) and model output (fixed) via Elastix B-spline, then warps the atlas through the recovered transform. VisuAlign markers are extracted from B-spline control points. The legacy workflows compute affine and TPS results from landmark correspondences; export uses the affine result only.
- itk-elastix is a required dependency for the colored segmentation workflow.
- Debug traces are written only when `LANGSLICE_VLM_DEBUG_DIR` is set.
- Ruff and basedpyright are scoped to specific directories (see `pyproject.toml` `include` lists), not the full package.
- `archive/` and `references/` are read-only unless a task explicitly targets them.
- When behavior changes, update the relevant markdown files (`README.md`, `REPO_MAP.md`, `AGENTS.md`, `docs/`) in the same pass.

## Verify After Edits

Run all three checks after any code change:

```bash
python -m pytest
python -m ruff check .
python -m basedpyright
```

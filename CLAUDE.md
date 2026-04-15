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
langslice estimate-group <img1> <img2> ... [--interval 200] [--atlas ...]
langslice estimate <image> [--atlas ...] [--model ...]
langslice register <image> --position <mm> [--workflow ...] [--out ...]

# Tauri GUI
cd tauri-gui && pnpm tauri dev
```

## Architecture

LangSlice registers histology slice images to BrainGlobe atlases using Gemini vision-language models for estimation and deterministic local solvers for geometry. The desktop GUI is a Tauri app (Rust + React + Three.js) in `tauri-gui/`.

**Pipeline:** `AP estimation → colored segmentation → Elastix affine+B-spline registration → preview/export`

The code is split into modules with clear boundaries:

- **`langslice/atlas/`** — BrainGlobe atlas loading, AP/index conversion, coronal slice extraction, colored region and smoothed boundary helpers (`get_colored_region_slice()`, `get_smoothed_boundary_slice()`). Orientation assumptions are centralized in `space.py` and require coronal layout (AP/DV/ML on axes 0/1/2).
- **`langslice/vlm_config.py`** — Gemini client configuration and backend selection. Three auth backends: `ai_studio`, `vertex_api_key`, `vertex_adc`. Shared by both estimation and registration modules.
- **`langslice/estimation/`** — AP estimation, split by provider:
  - `_types.py` — provider-agnostic result types (`APResult`, `MultiSliceResult`)
  - `google/` — Gemini implementations: `ap_multi_slice.py` (default, tool-use group estimation for 2-8 consecutive slices), `ap_single_slice.py` (single-slice tool-use), `ap_image_gen.py` (image-gen multi-pass), `common.py` (shared loop state + helpers), `tool_definitions.py`, `batch_eval.py`
  - `openai/` — OpenAI-compatible implementations (Responses API, for Ollama/local models)
  - `debug.py` — shared debug artifact writing
- **`langslice/whole_brain/`** — Whole-brain multi-slice AP estimation. Anchors use a two-stage flow: image-gen 2-pass coarse (broad + neighborhood) then nano-banana fine pass within ±0.5mm, with 3-tier fallback (Stage A fail → midpoint, Stage B fail → coarse). All remaining slices are estimated in parallel with 2-pass nano-banana (0.10mm fine resolution, windowed) followed by a confirmation pass (±0.25mm, ~0.06mm spacing). A Huber-loss constrained optimizer fits a monotone curve through all estimates, with local-interval spacing priors and hard minimum-thickness constraints. CLAHE adaptive preprocessing is applied to all slices. Public entry point: `run_brain_estimation(...)` in `pipeline.py`.
- **`langslice/registration/`** — Shared utilities and workflow router (`common.py`), provider-specific workflows:
  - `google/` — Gemini implementations: `warping_image_gen.py` (default, Elastix B-spline), `landmarks_image_gen.py` (legacy two-shot), `landmarks_tool_use.py` (on hold)
  - `openai/` — OpenAI-compatible implementations (Responses API, for Ollama/local models)
  - Shared: `solver.py`, `runtime.py`, `types.py`, `core.py`
- **`langslice/ml/`** — Non-LLM machine learning tools (GPU-accelerated target selection, etc.).
- **`tauri-gui/`** — Tauri desktop app. Rust backend (`src-tauri/`) for atlas loading, reslicing, mesh serving. React + Three.js frontend (`src/`) for 3D visualization, dashboard, split/overlay views. Launched via `cd tauri-gui && pnpm tauri dev`.
- **`langslice/export.py`** — Coronal anchoring math and QUINT/ABBA-compatible single-slice JSON export. `SliceExport.markers` supports VisuAlign `[ox, oy, nx, ny]` pairs from Elastix B-spline control points.

Supporting utilities: `image_prep.py` (normalization, pixel-size detection, VLM downsampling), `agent_trace.py` (structured trace events), `retry.py` (shared retry/heartbeat infrastructure), `cli.py` (argparse entry point).

## Gemini API — ALWAYS Check Docs First

The google-genai SDK is newer than your training data. **Before writing or modifying ANY code that uses the google-genai SDK**, you MUST look up the relevant API in context7 (`/googleapis/python-genai`). Never assume you know the API surface — check types, method signatures, and supported parameters before writing code. This applies to:

- `langslice/vlm_config.py` (client config)
- `langslice/estimation/google/` (Gemini AP estimators, batch)
- `langslice/registration/google/` (Gemini registration workflows)
- Any file importing `google.genai` or `google.genai.types`

Common things to verify: `GenerateContentConfig` fields, `types.Part` constructors, Interactions API parameters, File API methods, thinking config, media resolution options.

## Ollama / OpenAI Provider — DO NOT RESEARCH COMPATIBILITY

Ollama supports the OpenAI Responses API (`client.responses.create()`). Only the non-stateful flavor is supported (no `previous_response_id` or conversation support). The OpenAI provider code in `langslice/estimation/openai/` and `langslice/registration/openai/` is correct as-written. **Do not web-search, ask, or second-guess whether Ollama supports the Responses API. It does. Move on.**

- Config: `langslice/openai_config.py` — defaults to `http://localhost:11434/v1`, API key `"ollama"`, model `gemma4:31b`
- CLI: `--provider openai` on any estimation/registration command
- Ollama management: `langslice ollama status|list|pull|remove`
- Ollama model names use tag format: `gemma4:31b`, `gemma4:26b`, `gemma4:e4b` (not the hyphenated Google names)

### llama-server (direct llama.cpp)

For vision tasks requiring higher image token budgets, use llama-server directly instead of Ollama. Ollama doesn't expose `--image-min-tokens` / `--image-max-tokens` yet.

```bash
# Start llama-server (from C:\LabSoftware\llama-cpp\)
/c/LabSoftware/llama-cpp/llama-bin/llama-server.exe \
  -m /c/LabSoftware/llama-cpp/gemma-4-31B-it-Q4_K_M.gguf \
  --mmproj /c/LabSoftware/llama-cpp/mmproj-BF16.gguf \
  --port 8090 -ngl 99 --ctx-size 65536 \
  --image-min-tokens 1120 --image-max-tokens 1120 \
  --ubatch-size 2048 --batch-size 2048 --jinja

# Point langslice at it
export OPENAI_BASE_URL=http://localhost:8090/v1
export OPENAI_MODEL=gemma-4-31B-it-Q4_K_M
langslice estimate <image> --provider openai
```

Gemma 4 vision token budgets: 70/140 (fast), 280 (default in Ollama), 560 (charts/UI), **1120** (OCR/fine detail — use this for atlas matching). Must set `--ubatch-size` >= `--image-max-tokens` or llama-server crashes.

## Image Resolution for Estimation — DO NOT CHANGE

Atlas slices are sent at their **native coronal resolution** (e.g., 456x320 for `allen_mouse_25um`). Tissue slices are downscaled to match via `get_coronal_long_edge(atlas)`. There is no `atlas_resolution` parameter — resolution is derived automatically from the atlas volume dimensions.

**Individual mode is always better than grid mode for estimation accuracy.** Benchmarked 2026-04-15 on Flash (M01, medium resolution, native atlas):
- Individual: 0.140mm MAE, 100% within 0.5mm
- Grid: 0.283mm MAE, 76% within 0.5mm (2x worse)

Individual gives each atlas slice its own token budget. Grid packs multiple slices into one image, halving effective per-slice resolution and forcing the model to reconstruct spatial relationships across tile boundaries. `send_individually=True` is the default everywhere. The `--grid` CLI flag opts into grid mode for experimentation only.

Do not upscale atlas images. Do not send tissue slices at higher resolution than the atlas. The VLM cannot match detail that doesn't exist in both images — extra pixels become wasted tokens (Wang et al. ICML 2025: interpolated upscaling gives +0.1% accuracy vs +4.2% from genuinely finer patches).

## Key Conventions

- AP coordinates are atlas-native millimeters from the anterior edge of the volume.
- The colored segmentation workflow (default for image-gen models) registers grayscale versions of the atlas colored regions (moving) and model output (fixed) via two-stage Elastix: affine for global alignment then B-spline for local deformations, both using AdvancedNormalizedCorrelation. The atlas is then warped through the recovered transform. VisuAlign markers are extracted from B-spline control points. The legacy workflows compute affine and TPS results from landmark correspondences; export uses the affine result only.
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

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
langslice gui
langslice version
```

## Architecture

LangSlice is a PySide6 desktop app that registers histology slice images to BrainGlobe atlases using Gemini vision-language models for estimation and deterministic local solvers for geometry.

**Pipeline:** `AP estimation → landmark correspondences → affine/TPS solve → preview/export`

The code is split into five modules with clear boundaries:

- **`langslice/atlas/`** — BrainGlobe atlas loading, AP/index conversion, coronal slice extraction. Orientation assumptions are centralized in `space.py` and require coronal layout (AP/DV/ML on axes 0/1/2).
- **`langslice/vlm/`** — Gemini client configuration (`config.py`), multi-turn AP estimator with tool-use loop (`estimator.py`), offline batch helper (`batch_eval.py`). Three auth backends: `ai_studio`, `vertex_api_key`, `vertex_adc`.
- **`langslice/registration/`** — Prompt construction and correspondence parsing (`agents.py`), deterministic affine and TPS fitting (`solver.py`), orchestration and debug artifacts (`runtime.py`), data classes (`types.py`). Public entry point: `estimate_registration_runtime(...)` in `core.py`.
- **`langslice/gui/`** — PySide6 main window orchestrates the full pipeline. Atlas viewer, overlay viewer, settings dialog, trace inspector, and run-metadata dialog are separate widgets.
- **`langslice/export.py`** — Coronal anchoring math and QUINT/ABBA-compatible single-slice JSON export.

Supporting utilities: `image_prep.py` (normalization, pixel-size detection, VLM downsampling), `agent_trace.py` (structured trace events), `cli.py` (argparse entry point).

## Key Conventions

- AP coordinates are atlas-native millimeters from the anterior edge of the volume.
- Registration computes both affine and TPS results, but export uses affine only.
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

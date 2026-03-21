# LangSlice Project Guide

## Overview

LangSlice is a Python desktop application for registering histology slice images to BrainGlobe atlases.
The current runtime is split into:

- AP estimation in `langslice/ai/estimator.py`
- landmark correspondence generation in `langslice/registration/agents.py` (shared utilities and router), with workflow-specific code in:
  - `agents_image_gen.py` — two-shot workflow exclusively for image-gen models (gemini-3-pro-image-preview, gemini-3.1-flash-image-preview)
  - `agents_tool_loop.py` — iterative tool-loop workflow for text-centric models (default)
- deterministic affine and TPS solving in `langslice/registration/solver.py`
- PySide6 workflow orchestration in `langslice/gui/main_window.py`
- QUINT/ABBA-compatible JSON export in `langslice/export.py`

## Active Paths

- `langslice/cli.py` - CLI entry point for `gui` and `version`
- `langslice/atlas/` - BrainGlobe loading, AP coordinate helpers, slice extraction
- `langslice/ai/` - Gemini client setup, AP estimator, offline batch helpers
- `langslice/registration/` - registration agent prompt, runtime wrapper, affine/TPS solving, result types
- `langslice/image_prep.py` - image normalization, pixel-size detection, VLM downsampling
- `langslice/agent_trace.py` - structured trace-event helpers used by AP and registration flows
- `langslice/gui/` - main window, viewers, settings dialog, trace inspector, metadata dialog
- `langslice/export.py` - coronal anchoring math and JSON serialization
- `tests/` - pytest coverage for atlas, export, image prep, registration, and GUI behavior
- `docs/` - maintained project documentation
- `README.md` and `REPO_MAP.md` - top-level user and navigation docs

## Runtime Facts

- The main GUI pipeline is `AP estimate -> registration correspondences -> affine/TPS solve -> preview/export`.
- The manual GUI pipeline skips AP estimation and runs registration from the slider-selected `position_mm`.
- AP coordinates are atlas-native millimeters from the anterior edge of the volume.
- Atlas orientation assumptions are centralized in `langslice/atlas/space.py` and currently require coronal layout with AP/DV/ML on axes `0/1/2`.
- Registration export uses the affine result only; the nonlinear TPS result is computed and stored but not exported.
- Optional debug traces are written only when `LANGSLICE_VLM_DEBUG_DIR` is set.

## Boundaries

- Treat `langslice/`, `tests/`, `docs/`, `README.md`, and `REPO_MAP.md` as the active project.
- Treat `archive/` and `references/` as read-only unless the task explicitly targets them.
- Keep documentation literal to the current code. If behavior changes, update the relevant markdown files in the same pass.

## Verify After Edits

- `python -m pytest`
- `python -m ruff check .`
- `python -m basedpyright`
- `python -m langslice version`

# LangSlice Project Guide

## Overview

LangSlice is a Python package and Tauri desktop application for registering histology slice images to BrainGlobe atlases.
The current runtime is split into:

- AP estimation in `langslice/ai/estimator.py`
- registration workflows in `langslice/registration/agents.py` (shared utilities and router), with workflow-specific code in:
  - `agents_colored_segmentation.py` -- colored segmentation workflow (default for image-gen models): model produces atlas-colored tissue segmentation, Elastix B-spline extracts deformation
  - `agents_image_gen.py` -- legacy two-shot landmark workflow for image-gen models (superseded by colored segmentation)
  - `agents_tool_loop.py` -- iterative tool-loop workflow for text-centric models (experimental, on hold)
- deterministic affine and TPS solving in `langslice/registration/solver.py`
- CLI workflow orchestration in `langslice/cli.py`
- Tauri desktop app in `tauri-gui/`
- QUINT/ABBA-compatible JSON export with VisuAlign markers in `langslice/export.py`

## Active Paths

- `langslice/cli.py` -- CLI entry point for `version`, `register`, and `estimate`
- `langslice/atlas/` -- BrainGlobe loading, AP coordinate helpers, slice extraction, colored region and boundary helpers
- `langslice/ai/` -- Gemini client setup, AP estimator, offline batch helpers
- `langslice/registration/` -- registration workflows, runtime wrapper, affine/TPS solving, result types
- `langslice/image_prep.py` -- image normalization, pixel-size detection, VLM downsampling
- `langslice/agent_trace.py` -- structured trace-event helpers used by AP and registration flows
- `langslice/export.py` -- coronal anchoring math, VisuAlign markers, and JSON serialization
- `tauri-gui/` -- Tauri desktop app (Rust + React + Three.js)
- `tests/` -- pytest coverage for atlas, export, image prep, and registration behavior
- `docs/` -- maintained project documentation
- `README.md` and `REPO_MAP.md` -- top-level user and navigation docs

## Runtime Facts

- The main pipeline is `AP estimate -> colored segmentation -> Elastix B-spline -> VisuAlign markers -> export`.
- The CLI `register` subcommand runs registration at a given AP position.
- The CLI `estimate` subcommand runs AP estimation only.
- AP coordinates are atlas-native millimeters from the anterior edge of the volume.
- Atlas orientation assumptions are centralized in `langslice/atlas/space.py` and currently require coronal layout with AP/DV/ML on axes `0/1/2`.
- The colored segmentation workflow (default for image-gen models) produces an Elastix B-spline transform and VisuAlign markers from B-spline control points.
- The legacy workflows compute affine and TPS results from landmark correspondences; export uses the affine result only.
- Optional debug traces are written only when `LANGSLICE_VLM_DEBUG_DIR` is set.

## Boundaries

- Treat `langslice/`, `tauri-gui/`, `tests/`, `docs/`, `README.md`, and `REPO_MAP.md` as the active project.
- Treat `archive/` and `references/` as read-only unless the task explicitly targets them.
- Keep documentation literal to the current code. If behavior changes, update the relevant markdown files in the same pass.

## Verify After Edits

- `python -m pytest`
- `python -m ruff check .`
- `python -m basedpyright`
- `python -m langslice version`

# Documentation Index

This directory documents the public LangSlice implementation. It is meant to
describe the code that exists now, not private planning notes or historical
session handoffs.

## Start Here

- `README.md` - setup, quickstart, and high-level project behavior.
- `architecture_overview.md` - package boundaries, major modules, and end-to-end control flow.
- `current_workflow.md` - current CLI and Tauri GUI workflows.
- `training_overview.md` - public training layout, entrypoints, and local-data policy.

## Runtime References

- `abba_ap_coordinate_system.md` - atlas-native AP coordinate conventions.
- `registration_plan.md` - current image-generation registration runtime.
- `image_gen_AIstudio_examples/` - small provider examples and reference images.

## Repository Map

- `src/langslice_harness/` - installable Python harness and `langslice` CLI.
- `tauri-gui/` - desktop GUI.
- `web-demo/` - static browser demo.
- `models/langslice-gemma-4/` - Gemma 4 E4B fine-tuning project.
- `models/langslice-traces/`, `models/synthdata/`, `models/training-core/`, `models/data/` - shared model/data packages.
- `tests/` - pytest coverage.
- `docs/` - public documentation.

## Local-Only Material

The repo intentionally does not track private data rows, generated corpora,
model checkpoints, local run outputs, or private planning notes. Keep those in
ignored paths such as `_local/`, `out/`, `data/`, `debug_runs/`, and
`references/`.

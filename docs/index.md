# Documentation Index

This directory documents the public LangSlice implementation. It is meant to
describe the code that exists now, not private planning notes or historical
session handoffs.

## Start Here

- [README](https://github.com/greenpolo/LangSlice#readme) - setup, quickstart, and high-level project behavior.
- [`architecture_overview.md`](architecture_overview.md) - package boundaries, major modules, and end-to-end control flow.
- [`current_workflow.md`](current_workflow.md) - current CLI and Tauri GUI workflows.
- [`training_overview.md`](training_overview.md) - public training layout, entrypoints, and local-data policy.

## Runtime References

- [`registration_plan.md`](registration_plan.md) - current image-generation registration pipeline.
- [`engine_schema.json`](engine_schema.json) - generated JSON Schema bundle for the Python engine API.

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

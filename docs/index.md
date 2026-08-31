# Documentation Index

This directory documents the public LangSlice implementation. It is meant to
describe the code that exists now, not private planning notes or historical
session handoffs.

## Start Here

- [README](https://github.com/greenpolo/LangSlice#readme) - setup, quickstart, and high-level project behavior.
- [`architecture_overview.md`](architecture_overview.md) - package boundaries, major modules, and end-to-end control flow.
- [`current_workflow.md`](current_workflow.md) - current CLI workflows.

## Runtime References

- [`registration_plan.md`](registration_plan.md) - current image-generation registration pipeline.

## Repository Map

- `src/langslice/` - installable Python package and `langslice` CLI.
  - `linear/` - slice-position estimation.
  - `nonlinear/` - generative-image registration.
- `tests/` - pytest coverage.
- `docs/` - public documentation.

Training infrastructure and the SliceBench benchmark live in the separate
LangSlice-Training and SliceBench repositories.

## Local-Only Material

The repo intentionally does not track private data rows, generated corpora,
model checkpoints, local run outputs, or private planning notes. Keep those in
ignored paths such as `_local/`, `out/`, `data/`, `debug_runs/`, and
`references/`.

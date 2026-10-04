# Documentation Index

This directory documents the public LangSlice implementation. It is meant to
describe the code that exists now, not private planning notes or historical
session handoffs.

## Start Here

- [README](https://github.com/greenpolo/LangSlice#readme) - setup, quickstart, and high-level project behavior.
- [`architecture_overview.md`](architecture_overview.md) - package boundaries, major modules, and end-to-end control flow.
- [`current_workflow.md`](current_workflow.md) - current CLI workflows.
- [`agent_cli.md`](agent_cli.md) - the agent CLI: one verb on a job folder, the JSON envelope, exit codes, the job folder's reference card and `langslice.open_job`.
- [`file_formats.md`](file_formats.md) - the job folder's public files: `registration.json`, each section's coordinate, label and residual maps, the QuickNII / VisuAlign exports.
- [`interface_design.md`](interface_design.md) - the target user-facing design: the Positioning, Linear and Nonlinear tasks, their options, and what each host exposes, with what is on main today.

## Runtime References

- [`linear_design.md`](linear_design.md) - the linear agent environment: job spec, state, tools and submit gates.
- [`nonlinear_design.md`](nonlinear_design.md) - the two border-based registration routes (placed-border correction, the production path, and the placement-free atlas-driven route).
- [`nonlinear_image_tool.md`](nonlinear_image_tool.md) - the linear agent's image-model border-correction tool and its prompt review.
- [`registration_plan.md`](registration_plan.md) - current image-generation registration pipeline.

## Hosts

- [`abba_installation.md`](abba_installation.md) - installing and using the Fiji connector in an existing ABBA.
- [`abba_plugin_design.md`](abba_plugin_design.md) - the connector's design, worker protocol and dialog-to-spec mapping.
- [`napari_plugin_design.md`](napari_plugin_design.md) - proposed napari connectors (not built).

## Design Records

- [`visual_context_design.md`](visual_context_design.md) - shelved image-retention experiment and the measurement-first direction.

## Repository Map

- `src/langslice/` - installable Python package and `langslice` CLI.
  - `linear/` - order, position and one in-plane transform per section.
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

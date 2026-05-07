# LangSlice Project Guide

## Overview

LangSlice is an ADK-based Python harness and Tauri desktop application for
registering histology slice images to BrainGlobe atlases.

The active runtime lives in `src/langslice_harness/`. The public CLI command is
still `langslice`, but Python imports should use `langslice_harness.*`.

## Active Paths

- `src/langslice_harness/cli.py` -- CLI entry point for `version`, `register`,
  `estimate`, `estimate-group`, `estimate-brain`, and `ollama`
- `src/langslice_harness/vlm_config.py` -- Gemini backend selection and runtime
  settings
- `src/langslice_harness/openai_config.py` -- OpenAI-compatible model settings
- `src/langslice_harness/atlas/` -- BrainGlobe loading, AP coordinate helpers,
  slice extraction, colored region maps, and borders
- `src/langslice_harness/harness/estimation/` -- ADK AP-estimation agents,
  prompts, tools, validators, and runners
- `src/langslice_harness/estimation/` -- public AP-estimation exports and shared
  helper code
- `src/langslice_harness/harness/registration/` -- image-gen registration
  candidate generation, provider adapters, and optional ADK review loop
- `src/langslice_harness/registration/` -- public registration wrapper, runtime,
  solver helpers, and result types
- `src/langslice_harness/whole_brain/` -- multi-slice whole-brain AP estimation
- `src/langslice_harness/image_prep.py` -- image normalization, pixel-size
  detection, and VLM downsampling
- `src/langslice_harness/export.py` -- QUINT/ABBA-compatible JSON export
- `models/langslice-gemma-4/` -- fine-tuned model project
- `tauri-gui/` -- Tauri desktop app
- `tests/` -- pytest coverage
- `docs/`, `README.md`, and `REPO_MAP.md` -- maintained documentation

## Runtime Facts

- The main pipeline is `AP estimate -> image-gen registration -> Elastix
  B-spline -> VisuAlign markers -> export`.
- Registration has one active path: image-gen registration.
- Registration can run directly or with an optional ADK review loop.
- The ADK review loop receives the generated atlas target, the Elastix-warped
  atlas, and the warped-atlas border overlay.
- AP coordinates are atlas-native millimeters from the anterior edge of the
  volume.
- Atlas orientation assumptions are centralized in
  `src/langslice_harness/atlas/space.py` and currently require coronal layout
  with AP/DV/ML on axes `0/1/2`.
- Optional debug traces are written only when `LANGSLICE_VLM_DEBUG_DIR` is set.

## Boundaries

- Treat `src/langslice_harness/`, `models/`, `tauri-gui/`, `tests/`, `docs/`,
  `README.md`, and `REPO_MAP.md` as the active project.
- Treat `_local/`, `references/`, and generated outputs as local-only material.
- Keep documentation literal to the current code. If behavior changes, update
  the relevant markdown files in the same pass.

## Training-data review (QC app)

When assembling Inventory, SFT, RLVR, Eval, or synthetic data for the QC app,
write outputs to the paths documented in `_local/qc_app/CONTRACTS.md`. The QC
app reloads on mtime change — no app code edit required. Do not invent new
manifest shapes or paths; conform to the contract or extend it explicitly.

## Delegation (cost-saving)

To minimize main-thread token spend, delegate work to other CLIs whenever the
task fits one of these patterns:

- **Deep codebase exploration** (broad reads across many files, tracing call
  paths, structural questions): use the `gemini:explore` skill / `gemini-explorer`
  subagent. Read-only, 1M-token context.
- **Extensive outside research** (library docs, API specs, external best
  practices, anything benefiting from web + Context7): use the `gemini:research`
  skill / `gemini-researcher` subagent. Read-only.
- **Plan execution** -- once a plan exists, delegate each step:
  - **Codex** (`codex:execute` / `multi:codex-execute`) when the step meets a
    complexity threshold: non-trivial logic, math, careful reasoning, multi-file
    refactors with semantic decisions.
  - **Cursor** (`cursor:execute` / `multi:cursor-execute`, Agent mode on Auto)
    when the step is simple and well-defined: long mechanical writes,
    pattern-following across files, bulk edits, boilerplate.

Main-thread Claude stays in the planner/reviewer role: brainstorm, write the
plan, dispatch steps, verify diffs. Avoid doing implementation work directly
when a delegate fits.

## Verify After Edits

- `python -m pytest`
- `python -m ruff check .`
- `python -m basedpyright`
- `python -m langslice_harness version`
- `langslice version`
- `pnpm build` from `tauri-gui/` when GUI TypeScript changes
- `cargo check` from `tauri-gui/src-tauri/` when Rust/Tauri changes

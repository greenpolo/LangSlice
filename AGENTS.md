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
- `models/langslice-gemma-4/` -- fine-tuned Gemma 4 E4B project (SFT + RLVR)
  - `training/sft/` -- SFT trainer (`python -m sft.train_sft`)
  - `training/rlvr/` -- RLVR GRPO trainer (parked 2026-05-09; see `training/rlvr/README.md`)
  - `training/iSFT/` -- expert-iteration SFT driver (active pivot)
  - `training/configs/` -- TOML configs (`sft_default`, `grpo_pilot`, `grpo_phase_b`)
  - `data/sft_examples.jsonl` -- single-slice langslice-native trace corpus
- `tauri-gui/` -- Tauri desktop app
- `tests/` -- pytest coverage
- `docs/` and `README.md` -- maintained documentation

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
  and `README.md` as the active project.
- Treat `_local/`, `references/`, and generated outputs as local-only material.
- Keep documentation literal to the current code. If behavior changes, update
  the relevant markdown files in the same pass.

## Training-data manifest (multi-agent safety)

The training-data manifest has **two architecturally disjoint layers**.
Most agents only touch one of them. The role separation is enforced by
hard rules — mixing roles in one session will silently destroy another
agent's work.

### Layer 1: shards (GT data)

`data/manifest/shards/<plane>/<dataset>.jsonl` — one shard per
`(plane, dataset)` pair. Rows carry GT (position, atlas, species, etc.)
but **never** a `split` field. Per-shard curation lives in
`data/manifest/overrides/<plane>/<dataset>.json` (drops, axis flips,
per-section position overrides, atlas overrides).

### Layer 2: allocations (split membership)

`data/manifest/allocations/<plane>/<split>.jsonl` — 9 files (3 planes ×
3 splits: `eval` / `rlvr` / `sft`). Append-only with tombstone shape;
each line names a `section_id` that lives in some shard for that plane.
Splits are **computed at read time** via `compute_split_for(plane,
section_id)` from `_local/eval/allocations.py`, never stored on the
shard row. A `section_id` may belong to at most one split per plane.

### Two roles, never mixed in one session

- **GT-fix agent.** Edits upstream sources or `overrides/`, then runs
  `rebuild_shard.py`. Never runs `allocate.py`.
- **Allocation agent.** Builds `eval` / `rlvr` / `sft` splits via
  `allocate.py`. Never runs `rebuild_shard.py`, never edits shards,
  never edits overrides.

If you don't know which role you are, stop and ask the user.

### Authoritative docs

- `_local/eval/HOW_TO_FIX_DATA.md` — task-oriented walkthrough; **read this first** before any data fix. Includes the 8 hard rules.
- `_local/eval/SHARDS.md` — architecture reference for the shards / overrides / allocations layout.
- `_local/qc_app/CONTRACTS.md` — what the QC app reads from each layer (Inventory, SFT, RLVR, Eval, Synthetic). The app reloads on mtime change; do not invent new shapes or paths — conform to the contract or extend it explicitly.

### GT-fix CLI

```powershell
# Edit upstream or append to overrides/<plane>/<dataset>.json, then:
python _local/eval/rebuild_shard.py <plane>/<dataset>                  # dry-run, exits 1 on any diff
python _local/eval/rebuild_shard.py <plane>/<dataset> --accept-diff N  # commit; N must match dry-run count
```

Diff gate: `--accept-diff N` must match the *exact* number of changed
rows the dry-run reported. If `N` is bigger than expected, **stop** —
something else changed under you. Never bypass this gate.

### Allocation CLI

```powershell
python _local/eval/allocate.py add <plane>/<split> <section_id> --dataset <name> --added-by <agent_id>
python _local/eval/allocate.py remove <plane>/<split> <section_id> --removed-by <agent_id>
python _local/eval/allocate.py list <plane>/<split>
```

`<plane>` is `coronal` / `sagittal` / `horizontal`; `<split>` is
`eval` / `rlvr` / `sft`. The CLI validates that each `section_id` exists
in the corresponding inventory shard and that it isn't already in
another split for the same plane.

### Cross-shard checks

```powershell
python _local/eval/validate_manifest.py   # read-only; cannot write any shard or allocation
```

### Don't resurrect legacy scripts

`_local/eval/legacy/` is read-only context for what older patchers did.
Their old paths in `_local/eval/` now contain stubs that exit with code
2. Running them re-introduces the multi-agent footgun this architecture
was built to prevent.

## Training (Gemma 4 E4B fine-tune)

The active fine-tune project lives in `models/langslice-gemma-4/`. **v1
scope (hackathon, deadline 2026-05-18) is single-slice agent traces
only.** Bbox grounding, landmark listing, multi-slice morphology, and
programmatic skeletons are designed but **deferred**; do not implement
them without explicit user request.

- **Public training overview:** `docs/training_overview.md`
- **SFT contract:** `models/langslice-gemma-4/training/sft/README.md`
- **Active single-turn RL:** `models/langslice-gemma-4/training/single_turn_rl/README.md`

### SFT data contract

The trainer reads ONE langslice-native JSONL at
`models/langslice-gemma-4/data/sft_examples.jsonl`. Row shape and
constraints are documented in
`models/langslice-gemma-4/training/sft/README.md`. Image paths are
relative to the JSONL's parent directory. The trainer does NOT walk raw
Gemini run folders directly — corpus assembly is upstream.

### Run SFT

```powershell
cd models/langslice-gemma-4/training
python -m sft.train_sft `
  --config configs/sft_default.toml `
  --dataset ../../../models/langslice-gemma-4/data/sft_examples.jsonl `
  --output-dir ../../../out/sft/run0
```

Add `--dry-run` to validate JSONL structure without loading Gemma.

### RLVR (parked)

Multi-turn GRPO RLVR is parked as of 2026-05-09 in favor of expert-iteration
SFT (`training/iSFT/`). The RLVR module + scripts are preserved at
`models/langslice-gemma-4/training/rlvr/`; see that README before un-parking.

## Delegation (cost-saving)

To minimize main-thread token spend, delegate work to other CLIs whenever the
task fits one of these patterns:

- **Deep codebase exploration** (broad reads across many files, tracing call
  paths, structural questions): dispatch a `general-purpose` Claude (Opus)
  subagent. Read-only. Do NOT use any `gemini:*` subagent — Gemini is off the
  table for both exploration and research (hallucinates library names; current
  `multi:gemini-researcher` is broken in this environment).
- **Extensive outside research** (library docs, API specs, external best
  practices, anything benefiting from web + Context7): dispatch a
  `general-purpose` Claude (Opus) subagent and instruct it to use Exa
  (`web_search_exa`, `web_fetch_exa`) and Context7 for docs.
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

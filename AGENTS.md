# LangSlice Project Guide

LangSlice is an ADK-based Python harness plus a Tauri desktop app for
registering histology slice images to BrainGlobe atlases. CLI command is
`langslice`; Python imports use `langslice_harness.*`.

`CLAUDE.md` is a verbatim copy of this file. Edit one, mirror to the other.

## Before writing code (READ THIS FIRST)

LangSlice moves fast and depends on libraries that move fast (Unsloth, TRL,
vLLM, transformers, PEFT, ADK, Gemini SDK). Three subagents are configured
to keep that velocity from costing you correctness.

### 1. Explore the codebase before editing — `Explore` subagent

For anything beyond a single-file tweak, dispatch the **`Explore`** subagent
(profile in `.claude/agents/Explore.md`, runs on Haiku, read-only:
Read/Grep/Glob) to map call sites before changing them. Good Explore
queries:

- "Where is `function_name` defined? Where is it called from?"
- "Which files reference `LANGSLICE_VLM_DEBUG_DIR`?"
- "Find the entry point for the `register` CLI command."
- "List all TOML configs under `models/langslice-gemma-4/training/configs/`."

Don't speculate about paths or call shapes from memory — paths and module
layouts have churned (see `[project_training_core_consolidation_2026_05_22]`
in auto-memory). Memory is a hint; the filesystem is ground truth.

**Skip Explore when:** the path is already known and you just need to Read
it; the answer is a one-shot Grep that's faster inline; the question is
about an external library (dispatch Librarian instead).

### 2. Verify external knowledge — `Librarian` subagent

For anything library-, SDK-, model-, or API-shaped — especially AI/ML —
dispatch the **`Librarian`** subagent (profile in `.claude/agents/Librarian.md`,
runs on Sonnet) to fetch current docs and concrete implementation examples.
Training-data cutoffs lag; model IDs, SDK defaults, and library APIs change
quarterly.

Librarian's toolkit:

- **Context7** for library/framework/SDK/API docs:
  `mcp__context7__resolve-library-id`, then `mcp__context7__query-docs`. Use
  even for well-known libraries (React, PyTorch, HuggingFace, Anthropic SDK,
  Gemini SDK) — catches API drift your training data doesn't.
- **Exa** for web search and changelog hunting:
  `mcp__exa__web_search_exa`, `mcp__exa__web_fetch_exa`. Prefer Exa over
  generic web-search paths.
- `WebFetch` / `WebSearch` as fallback for explicit URLs.

Verify before writing code that depends on: model IDs (Gemini, Claude, Gemma
variants), SDK class/method shapes, training-library knobs (TRL GRPO args,
Unsloth flags, vLLM serving flags), atlas/data tooling (BrainGlobe,
SimpleITK, Elastix). Several painful debugging sessions are recorded under
`[reference_unsloth_*]` and `[reference_gemma4_*]` in auto-memory — failure
modes a docs check would have prevented.

**Skip Librarian when:** refactoring local code; writing one-off scripts;
debugging business logic; questions about general programming concepts that
don't depend on a specific library version.

### 3. Delegating bigger work — `general-purpose` (Opus)

For multi-step work that needs synthesis rather than just retrieval —
executing a slice of a written plan, tracing a pipeline end-to-end, auditing
cross-package consistency, or any open-ended agentic task — dispatch the
built-in **`general-purpose`** subagent (Opus). Main-thread Claude stays
planner/reviewer; verify diffs the subagent produces before integrating.

`codex:codex-rescue` remains available for second-opinion diagnoses or
stuck-state rescue when the general-purpose pass isn't getting unstuck.

**Skip general-purpose when:** the task is pure code lookup (Explore is
faster) or pure docs/API verification (Librarian is faster). Don't reach for
Opus when Haiku/Sonnet can answer.

## Active paths

### Runtime (`src/langslice_harness/`)

- `cli.py` — CLI entry for `version`, `gui`, `register`, `estimate`,
  `estimate-group`, `estimate-brain`, `collect-traces`, `quick-affine`,
  `serve`, `schema`, `ollama`
- `vlm_config.py` — Gemini backend selection and runtime settings
- `openai_config.py` — OpenAI-compatible model settings
- `atlas/` — BrainGlobe loading, AP coordinate helpers, slice extraction,
  colored region maps, borders; orientation assumptions live in `atlas/space.py`
- `harness/estimation/` — ADK AP-estimation agents, prompts, tools, validators,
  runners
- `estimation/` — public AP-estimation exports + shared helpers
- `harness/registration/` — image-gen registration candidates, provider
  adapters, optional ADK review loop
- `registration/` — public registration wrapper, runtime, solver helpers,
  result types
- `whole_brain/` — multi-slice whole-brain AP estimation
- `image_prep.py` — image normalization, pixel-size detection, VLM downsampling
- `export.py` — QUINT/ABBA-compatible JSON export
- `api/`, `comfyui/`, `ml/` — auxiliary surfaces (REST, ComfyUI nodes, ML helpers)
- `training_launchers.py` — exposes `langslice-gemma-sft` and
  `langslice-gemma-rl` console scripts

### Models (`models/`)

- `langslice-gemma-4/` — fine-tuned Gemma 4 E4B project
  - `data/sft_examples.jsonl` — single-slice langslice-native trace corpus
  - `training/sft/` — SFT trainer (model-scoped; entry `train_sft.py`)
  - `training/configs/` — TOML configs (`sft_default`, `grpo_lane_a_default`,
    `grpo_pilot`, phase-specific variants)
  - `training/single_turn_rl/` — thin shim; canonical RL code lives in
    `models/training-core/langslice_training/rl/single_turn/`
  - `inference/` — server-side inference helpers
  - `variants/` — released checkpoint trees (e.g. `langslice-gemma-4-e4b/`)
- `training-core/langslice_training/` — shared training package
  - `sft/`, `rl/{single_turn,multi_turn_env,common}/`, `embeddings/`,
    `curriculum/`, `adaptive/`, `model_io/`, `contracts/`, `corpus/`
  - `corpus/` — synthetic trace-corpus + atlas region-description (renderer-free;
    relocated from the former `synthdata` package)
- Synthetic histology IMAGE generation was extracted to the separate SimSlice
  project; only the renderer-free `corpus/` above remains in LangSlice.
- `langslice-traces/` — agent-trace utilities

### Other top-level

- `tauri-gui/` — Tauri desktop app (TypeScript + Rust)
- `tests/` — pytest coverage (mirrors `src/langslice_harness/` layout)
- `docs/`, `README.md` — maintained documentation

## Runtime facts

- Main pipeline: `AP estimate → image-gen registration → Elastix B-spline →
  VisuAlign markers → export`.
- Registration has one active path (image-gen); can run directly or with an
  optional ADK review loop. The review loop receives the generated atlas
  target, the Elastix-warped atlas, and the warped-atlas border overlay.
- AP coordinates are atlas-native millimeters from the anterior edge of the
  volume.
- Atlas orientation assumptions are centralized in
  `src/langslice_harness/atlas/space.py` and currently require coronal layout
  with AP/DV/ML on axes `0/1/2`.
- Optional debug traces are written only when `LANGSLICE_VLM_DEBUG_DIR` is set.

## Boundaries

- Active surface: `src/langslice_harness/`, `models/`, `tauri-gui/`, `tests/`,
  `docs/`, `README.md`.
- Local-only (do not ship, do not document publicly): `_local/`, `references/`,
  generated outputs, `out/`, `archive/`.
- Keep markdown literal to the code it describes. Behavior change → update the
  relevant doc in the same pass.

## Training-data manifest (multi-agent safety)

The training-data manifest has **two architecturally disjoint layers**.
Most agents only touch one. Role separation is enforced by hard rules —
mixing roles in a single session silently destroys another agent's work.

### Layers

- **Shards (GT data):** `data/manifest/shards/<plane>/<dataset>.jsonl` —
  one shard per `(plane, dataset)` pair. Rows carry GT (position, atlas,
  species, etc.) but **never** a `split` field. Per-shard curation lives in
  `data/manifest/overrides/<plane>/<dataset>.json`.
- **Allocations (split membership):**
  `data/manifest/allocations/<plane>/<split>.jsonl` — 9 files (3 planes ×
  3 splits: `eval` / `rlvr` / `sft`). Append-only with tombstones. Splits
  are computed at read time via `compute_split_for(plane, section_id)`;
  never stored on the shard row.

### Two roles, never mixed in one session

- **GT-fix agent.** Edits upstream sources or `overrides/`, then runs
  `rebuild_shard.py`. Never runs `allocate.py`.
- **Allocation agent.** Builds `eval` / `rlvr` / `sft` splits via
  `allocate.py`. Never runs `rebuild_shard.py`, never edits shards or
  overrides.

If you don't know which role you are, stop and ask the user.

### Authoritative docs (read before any data fix)

- `_local/eval/HOW_TO_FIX_DATA.md` — task-oriented walkthrough + 8 hard rules.
- `_local/eval/SHARDS.md` — architecture reference for shards/overrides/allocations.
- `_local/qc_app/CONTRACTS.md` — what the QC app reads from each layer.
  The app reloads on mtime change; do not invent shapes or paths.

### CLIs

```powershell
# GT-fix: edit upstream or overrides/<plane>/<dataset>.json, then:
python _local/eval/rebuild_shard.py <plane>/<dataset>                  # dry-run, exits 1 on diff
python _local/eval/rebuild_shard.py <plane>/<dataset> --accept-diff N  # commit; N must match exactly

# Allocation:
python _local/eval/allocate.py add    <plane>/<split> <section_id> --dataset <name> --added-by <agent_id>
python _local/eval/allocate.py remove <plane>/<split> <section_id>                  --removed-by <agent_id>
python _local/eval/allocate.py list   <plane>/<split>

# Read-only cross-shard check:
python _local/eval/validate_manifest.py
```

Diff gate: `--accept-diff N` must match the dry-run count *exactly*. If `N`
is bigger than expected, **stop** — something else changed under you.
Never bypass.

### Don't resurrect legacy scripts

`_local/eval/legacy/` is read-only context. The old paths in `_local/eval/`
are now stubs that exit with code 2. Running them re-introduces the
multi-agent footgun this architecture was built to prevent.

## Training (Gemma 4 E4B fine-tune)

`langslice-gemma-4 E4B v1.0` was blessed 2026-05-17 (see
`[project_phase9_v7_v1_release_2026_05_17]`). Post-hackathon, the training
package was consolidated into `models/training-core/langslice_training/`
(merged 2026-05-22, see `[project_training_core_consolidation_2026_05_22]`).

- **Public overview:** `docs/training_overview.md`
- **SFT contract:** `models/langslice-gemma-4/training/sft/README.md`
- **Single-turn RL code:**
  `models/training-core/langslice_training/rl/single_turn/` (canonical entry `train_grpo.py`; no top-level README yet)
- **Multi-turn RL env (parked, preserved internally):**
  `models/training-core/langslice_training/rl/multi_turn_env/`

### SFT data contract

The trainer reads ONE langslice-native JSONL at
`models/langslice-gemma-4/data/sft_examples.jsonl`. Row shape and constraints
are documented in `models/langslice-gemma-4/training/sft/README.md`. Image
paths are relative to the JSONL's parent. The trainer does NOT walk raw
Gemini run folders directly — corpus assembly is upstream.

### Launchers

```powershell
# SFT (canonical)
langslice-gemma-sft `
  --config models/langslice-gemma-4/training/configs/sft_default.toml `
  --dataset models/langslice-gemma-4/data/sft_examples.jsonl `
  --output-dir out/cache_fast/sft/run0

# Add --dry-run to validate JSONL structure without loading Gemma.

# Single-turn GRPO RL (canonical)
langslice-gemma-rl `
  --config models/langslice-gemma-4/training/configs/grpo_lane_a_default.toml `
  --output-dir out/cache_fast/rl/run0
```

### Before depending on training-library APIs

TRL, Unsloth, vLLM, and PEFT all changed shape in the last quarter.
Before adding new args, swapping callbacks, or changing model-loading flow,
dispatch a research subagent against Context7 (`unsloth`, `trl`, `vllm`,
`peft`, `transformers`) and Exa for upstream changelogs. Several painful
debugging sessions are recorded under `[reference_unsloth_*]` and
`[reference_gemma4_*]` in auto-memory — those are the failure modes a docs
check would have prevented.

## Verify after edits

```powershell
python -m pytest
python -m ruff check .
python -m basedpyright
python -m langslice_harness version
langslice version
# When GUI TypeScript changes:
pnpm build      # from tauri-gui/
# When Rust/Tauri changes:
cargo check     # from tauri-gui/src-tauri/
```

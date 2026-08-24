# LangSlice Project Guide

LangSlice is an ADK-based Python harness for registering histology slice
images to BrainGlobe atlases using vision-language models and frontier
image-generation models. CLI command is `langslice`; Python imports use
`langslice.*` (package lives at `src/langslice/`).

`AGENTS.md` is a verbatim copy of this file. Edit one, mirror to the other
(`cp CLAUDE.md AGENTS.md`).

Speak to the user in conceptual and strategic terms, rather than in code, function definitions, or syntactical details. The user is not a software engineer. You should defer to your own knowledge with respect to software-engineering best practices. 

## Before writing code (READ THIS FIRST)

LangSlice moves fast and depends on libraries that move fast (Unsloth, TRL,
vLLM, transformers, PEFT, ADK, the Gemini and OpenAI SDKs). Lean on subagents
so that velocity doesn't cost correctness.

### 1. Explore the codebase before editing — `Explore` subagent

For anything beyond a single-file tweak, dispatch the **`Explore`** subagent
(read-only: Read/Grep/Glob) to map call sites before changing them. Good
queries:

- "Where is `function_name` defined? Where is it called from?"
- "Which files reference `LANGSLICE_VLM_DEBUG_DIR`?"
- "Find the entry point for the `register` CLI command."
- "List all TOML configs under `models/langslice-gemma-4/training/configs/`."

Don't speculate about paths or call shapes from memory — module layouts have
churned. Memory is a hint; the filesystem is ground truth.

**Skip Explore when:** the path is already known and you just need to Read it;
the answer is a one-shot Grep that's faster inline; the question is about an
external library (use a research agent instead).

### 2. Verify external knowledge — `research` / `search` subagents

For anything library-, SDK-, model-, or API-shaped — especially AI/ML —
dispatch the **`research`** subagent (deep: reads docs and repos, has Context7
via `mcp__plugin_context7_context7__*` plus web search) or **`search`** (quick
factual lookup) to fetch current facts. Training-data cutoffs lag; model IDs,
SDK defaults, and library APIs change quarterly.

Verify before writing code that depends on: model IDs (Gemini, GPT, Gemma,
Claude variants), SDK class/method shapes, training-library knobs (TRL GRPO
args, Unsloth flags, vLLM serving flags), atlas/data tooling (BrainGlobe,
SimpleITK, Elastix). Several painful debugging sessions live under
`[reference_unsloth_*]` and `[reference_gemma4_*]` in auto-memory — failure
modes a docs check would have prevented.

**Skip research when:** refactoring local code; writing one-off scripts;
debugging business logic; general programming concepts that don't depend on a
specific library version.

### 3. Delegating bigger work — `general-purpose`

For multi-step work that needs synthesis rather than retrieval — executing a
slice of a written plan, tracing a pipeline end-to-end, auditing cross-package
consistency — dispatch the **`general-purpose`** subagent. Main-thread Claude
stays planner/reviewer; verify diffs the subagent produces before integrating.

**Skip general-purpose when:** the task is pure code lookup (Explore is faster)
or pure docs/API verification (a research agent is faster).

## Active paths

### Runtime (`src/langslice/`)

Two methods live as sibling subpackages with no dependency on each other:

- `linear/` — slice-position estimation. Single-slice ADK agent, prompts,
  tools, validators, session/runner plumbing, trace collection, and
  `linear/whole_brain/` — the whole-brain engine: a node graph
  (`ingest → survey → fix → seed → position → transforms → review → emit`)
  over one `StackState`, with bounded loop-back edges, a JSON checkpoint
  after every node, and results in the same shape as the checkpoint.
  Flips and reorders are recorded as data on the state; user image files are
  never modified. Every agent step is seeded with the whole stack as a
  labelled sequence of per-section images (`_step_common.stack_image_parts`),
  not a thumbnail grid; the contact sheet is still written next to the
  checkpoint but is a human-facing artifact only. `survey.py` is the
  stack-triage agent (damage, hemisphere flips, order, interval breaks in one
  pass) and `fix` rebuilds the sheet and routes back for a re-check;
  `seed` runs an automatic seeder if one is installed
  (only `deepslice.py`, which is not) and otherwise passes the stack through
  unplaced; `position.py` is the agent pass that places the stack, and it is
  deliberately LEAN — no per-slice estimation worker, data-only tool payloads,
  and a prompt that carries the job, the run's facts, one factual line per
  tool and the hard constraints, nothing else. No strategies, no rules of
  thumb, no failure-mode warnings anywhere in the step: full-trace forensics
  showed the per-slice worker's estimates carried ~no signal on real data
  while eating 82% of the wall-clock, and every major benchmark failure traced
  back to advice text the harness injected. Tools report data; the agent
  reasons. Tools: `view_slices`, `fetch_atlas`, `atlas_structures_at` /
  `structure_range` (wrapping `atlas/landmarks.py`), `stack_positions`
  (index, id, `position_mm`, `spacing_to_next_mm` — no nominal comparison, no
  legend), `interpolate_between` (computes, writes nothing), `set_positions`
  (writes, returns the same rows) and `submit_positions`. The gates stay,
  because a constraint stating a fact is not coaching: `submit_positions`
  refuses a submission unless its two `end_anchors` (a structure seen at each
  END of the stack) contain that section's position in their atlas span AND
  that structure's own span is narrow — `MAX_ANCHOR_SPAN_FRACTION` (8%) of the
  atlas's full slicing-axis extent, refused as `STRUCTURE_TOO_BROAD`
  otherwise, so a structure present almost everywhere (cortex, say) cannot
  "prove" a placement. It also refuses `interval_breaks` the agent's own
  written positions do not show (the interval there must exceed 1.5x the
  stack's median written spacing). Refusal messages state the numbers that
  caused them and stop. Both landmark tools and the end-anchor gate are gated
  by `BrainConfig.landmark_tools` (default on; CLI
  `--landmark-tools`/`--no-landmark-tools`), an ablation switch that off,
  drops the tools, the `end_anchors` argument, and the two landmark lines plus
  the end-anchor constraint from the prompt. Interpolation beyond the
  outermost fixed points steps at the interval those points imply, never at
  the nominal one; `transforms.py` proposes one in-plane
  alignment per section — the shared silhouette affine (plain code) for intact
  sections, a per-section interactive agent loop (preview → look → adjust →
  submit) for damaged ones; `review.py` is the final whole-stack consistency
  agent, gets the same lean prompt and data-only seed, and can route back to
  `position` once. `_step_common.py` holds what the
  agent steps share (`render_slice`, `view_slices`, manifest, ADK session
  loop); `render_slice` also applies the display-only fluorescence
  preprocessing (`--preprocess auto|none`, `BrainConfig.preprocess`) and, for
  the paths that SHOW a section to a model (`frame=True`), the tissue crop that
  matches the framing of fetched atlas sections — never for the affine-fitting
  path, whose coordinates the crop would move.
  `--stop-after NODE` runs one step and checkpoints; `--rerun-from
  {position,transforms,review}` rewinds an existing checkpoint's node and
  everything downstream of it (`engine.rewind_state`), notes those steps wrote
  included, then resumes — for re-benchmarking one step without re-paying for
  the agent steps ahead of it, and without seeding the fresh pass with the
  rejected one's numbers.
- `nonlinear/` — generative-image registration: candidate generation, image
  provider adapters, Elastix runtime, optional ADK review loop, affine and
  nonlinear result types, and `quick_affine.py` (silhouette affine preview;
  note the CLI groups `quick-affine` under `linear`). It runs after a linear
  placement step, whether that step is `langslice linear ...` or the user's
  own tool (in ABBA/QUINT workflows, linear placement happens first and
  LangSlice-nonlinear replaces the manual spline/BigWarp deformation step).

Shared, top-level:

- `atlas/` — BrainGlobe loading, position helpers, slice extraction, colored
  region maps, borders, plus `landmarks.py`: what the annotation says exists at
  a level (`structures_at`) and the slicing-axis span over which a structure
  exists (`axis_range_of`, descendants rolled up, presence scan cached per
  atlas+plane). Generic over the structure tree — no acronym is special-cased
- `space.py` — coordinate and orientation conventions
- `affine.py` — shared in-plane affine core: the silhouette (moments) fit of a
  section onto an atlas section, plus the rotation/scale/translate matrix
  builder and the normalized 6-number parameter convention. Used by both
  `linear/whole_brain/transforms.py` and `nonlinear/quick_affine.py`; belongs
  to neither
- `image_prep.py` — image normalization, pixel-size detection, VLM
  downsampling, and foreground framing (`crop_to_tissue`, `crop_to_mask`), used
  by the whole-brain visual path so histology and atlas sections fill their
  frames comparably. Tissue framing crops to the LARGEST connected blob, so a
  fragment or a speck elsewhere on the slide cannot widen the box
- `export.py` — QUINT/ABBA-compatible JSON export
- `providers/` — model access (`vlm_config.py` for Gemini backends,
  `openai_config.py` for OpenAI-compatible backends, `chatgpt.py` for the
  ChatGPT-subscription backend: `langslice login`, the `chatgpt/*` ADK model,
  and `gpt-image-2` image generation)
- `adk/` — ADK harness helpers (`plugins.py`, `model_resolver.py`,
  `sdk_helpers.py`)
- `api/` — Pydantic engine contract, runtime wrappers, and the stdio service
  behind `langslice serve`
- `cli.py` — CLI entry: `langslice linear {estimate, estimate-brain,
  quick-affine}`, `langslice nonlinear {register}`, and top-level `version`,
  `login`, `serve`, `collect-traces`
- `agent_trace.py` — structured trace helpers
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

- `tests/` — pytest coverage (mirrors `src/langslice/` layout)
- `slicebench/` — self-contained position-estimation benchmark
- `docs/`, `README.md` — maintained documentation

## Runtime facts

- Main pipeline: `position estimate (linear) → image-gen registration
  (nonlinear) → Elastix B-spline → VisuAlign markers → export`.
- `linear` and `nonlinear` are independent; `nonlinear` accepts a position
  from any source, not just `langslice linear`.
- Registration has one active path (image-gen); can run directly or with an
  optional ADK review loop. The review loop receives the generated atlas
  target, the Elastix-warped atlas, and the warped-atlas border overlay.
- Positions are atlas-native millimeters from the anterior edge of the volume.
- Atlas orientation assumptions are centralized in `src/langslice/space.py`,
  which derives AP/DV/ML axis indices from the atlas orientation via
  `brainglobe_space` and requires the AP axis to increase anterior→posterior.
- Optional debug traces are written only when `LANGSLICE_VLM_DEBUG_DIR` is set.
- Whole-brain agent sessions write a full-content JSONL trace (what the agent
  was shown, said, called, and got back; images as descriptors, never bytes)
  only when `LANGSLICE_TRACE_DIR` is set — `langslice linear estimate-brain
  --trace-dir PATH` sets it for one run. See `docs/current_workflow.md`.

## Boundaries

- Active surface: `src/langslice/`, `models/`, `slicebench/`, `tests/`,
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

```bash
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

```bash
# SFT (canonical)
langslice-gemma-sft \
  --config models/langslice-gemma-4/training/configs/sft_default.toml \
  --dataset models/langslice-gemma-4/data/sft_examples.jsonl \
  --output-dir out/cache_fast/sft/run0

# Add --dry-run to validate JSONL structure without loading Gemma.

# Single-turn GRPO RL (canonical)
langslice-gemma-rl \
  --config models/langslice-gemma-4/training/configs/grpo_lane_a_default.toml \
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

## Environment & verify after edits

Linux + bash. The project env is a **uv** venv at `.venv` (Python 3.11).
Activate with `source .venv/bin/activate`, or prefix commands with
`uv run`. (Rebuild: `uv venv --python 3.11 .venv && uv pip install -e ".[dev]"`.)

```bash
source .venv/bin/activate
python -m pytest
python -m ruff check .
python -m basedpyright
python -m langslice version
langslice version
```

# LangSlice Project Guide

LangSlice is an ADK-based Python harness for registering histology slice
images to BrainGlobe atlases using vision-language models and frontier
image-generation models. CLI command is `langslice`; Python imports use
`langslice.*` (package lives at `src/langslice/`).

`AGENTS.md` is a verbatim copy of this file, and each package `CLAUDE.md` under
`src/langslice/` has an `AGENTS.md` twin. Edit one, mirror to the other
(`cp CLAUDE.md AGENTS.md`).

Speak to the user in conceptual and strategic terms, rather than in code, function definitions, or syntactical details. The user is not a software engineer. You should defer to your own knowledge with respect to software-engineering best practices. 

## Judging results: look, don't just measure

For nonlinear experiments, generated-image quality is judged primarily by
visual inspection against the original histology. Open the raw model output,
describe anatomical successes and errors, and distinguish it from later
classification or registration. Metrics and human agreement are secondary
diagnostics, not acceptance criteria or a ranking of model quality. Read the
nonlinear package guide for mandatory sentence-by-sentence prompt review.

For other visual pipeline work, open the output and describe what you see as
well as reporting relevant measurements. When the picture and measurements
disagree, report the disagreement as a finding. Evaluating subagents follow the
same standard.

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
- "Which files reference `LANGSLICE_TRACE_DIR`?"

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

- `linear/` — order, position and one in-plane transform per section, as ONE
  agent environment (`langslice linear run FOLDER`): one `StackState`, one
  toolbox built from a `JobSpec`, one job statement, one ADK session that ends
  at `submit` or the turn budget. Tasks (`reorder`/`position`/`transform`) are
  switched on task by task; a task that is OFF builds no tools and takes its
  answer from the host instead. Every write checkpoints and is undoable, order
  and position must agree at submit, and the only nested session is the bounded
  per-section `align_slice`. Spec: `docs/linear_design.md`. Code map, the lean-
  harness rule, the submit gates and the known ceilings:
  `src/langslice/linear/CLAUDE.md` (loads when working there).
- `nonlinear/` — generative-image registration (image model → Elastix → report).
  The one active path, its measured design choices, and the cutting-angle lever:
  `src/langslice/nonlinear/CLAUDE.md` (loads when working there).

Shared, top-level:

- `atlas/` — BrainGlobe loading, slice extraction, colored region maps, borders,
  and the organized-color LUT / `--palette` render styles:
  `src/langslice/atlas/CLAUDE.md` (loads when working there).
- `integrations/` — QUINT JSON export and the ABBA registration plugin:
  `src/langslice/integrations/CLAUDE.md` (loads when working there).
- `oblique.py` — arbitrary-plane sampling out of an atlas volume plus
  (pitch, yaw) fitting (vendored from brainglobe-registration, BSD-3).
  `sample_oblique_annotation` is the label-safe entry point: sampling
  happens in float32, whose mantissa cannot hold the larger Allen ids, so
  it compacts the volume to dense indices first. THE largest lever measured
  on the nonlinear fit — see the `nonlinear/` entry
- `space.py` — coordinate and orientation conventions
- `affine.py` — shared in-plane affine core: the silhouette (moments) fit of a
  section onto an atlas section, plus the rotation/scale/translate matrix
  builder and the normalized 6-number parameter convention. Used by both
  `linear/transform.py` and `nonlinear/quick_affine.py`; belongs
  to neither
- `image_prep.py` — image normalization, pixel-size detection, VLM
  downsampling, and foreground framing (`crop_to_tissue`, `crop_to_mask`), used
  by the linear visual path so histology and atlas sections fill their
  frames comparably. Tissue framing crops to the LARGEST connected blob, so a
  fragment or a speck elsewhere on the slide cannot widen the box
- `providers/` — model ACCESS methods, never task logic. `registry.py` is
  the taxonomy: canonical names pair vendor with auth — `gemini-api` (Google
  API key, `vlm_config.py`), `openai-api` (API key / endpoint,
  `openai_config.py`), `openai-oauth` (subscription OAuth,
  `openai_oauth.py`: `langslice login`, the `openai-oauth/*` ADK model
  strings — legacy `chatgpt/*` accepted — and gpt-image-2), and `none` (no
  model at all: nonlinear's model-free backbone registers the silhouette
  prior itself, so there is nothing to authenticate). The OAuth path is
  NOT the OpenAI API: it talks to the separate Codex backend
  (`chatgpt.com/backend-api/codex`), whose image tool ignores
  `model`/`size`/`quality` and matches the input image's aspect exactly.
  Registration uses the raw `codex/images/edits`
  endpoint (`edit_image`), which delivers the prompt verbatim with no
  routing model in between (the hosted-router Responses path was deleted;
  see the nonlinear section). Default review model:
  `openai_oauth.DEFAULT_REVIEW_MODEL` (`openai-oauth/gpt-5.6-sol`) —
  and future providers (anthropic-api, openrouter-api, qwen-api, ...) are
  added there and nowhere else; legacy spellings google/openai/chatgpt
  resolve as aliases. Task-level semantics stay OUT of providers: the
  registration edit-vs-generate decision is `SegmentationGenerationRequest.mode`
  (nonlinear), and each transport merely translates it (`images.edit`
  endpoint, `action` on the Responses image_generation tool)

### Sibling repos (split out 2026-08-25)

- `../LangSlice-Training` — ALL local-model training infrastructure: the
  `models/` tree (gemma-4 project, training-core, traces, manifest tooling),
  the `langslice-gemma-sft`/`langslice-gemma-rl` launchers, docker training
  env, training tests, and the training-side `_local/` (eval manifest CLIs,
  QC app, synth data, logs). The gemma-4 model there is legacy, slated for
  wholesale replacement.
- `../SliceBench` — the position-estimation benchmark (former `slicebench/`).

Both depend on LangSlice as an editable sibling checkout via
`[tool.uv.sources]`; training additionally depends on SliceBench. Work on
training or benchmark code happens in those repos, not here.

## Runtime facts

- Main pipeline: `position estimate (linear) → image-gen registration
  (nonlinear) → Elastix B-spline → VisuAlign markers → export`.
- `linear` and `nonlinear` are independent; `nonlinear` accepts a position
  from any source, not just `langslice linear`.
- Registration has one active path (image-gen); can run directly or with the
  ADK pilot loop. Each candidate returns the generated atlas target, the
  Elastix-warped atlas, the warped-atlas border overlay, and the Elastix
  error-code report.
- Positions are atlas-native millimeters from the anterior edge of the volume.
- Atlas orientation assumptions are centralized in `src/langslice/space.py`,
  which derives AP/DV/ML axis indices from the atlas orientation via
  `brainglobe_space` and requires the AP axis to increase anterior→posterior.
- Optional debug traces are written only when `LANGSLICE_VLM_DEBUG_DIR` is set.
- Linear agent sessions write a full-content JSONL trace (what the agent
  was shown, said, called, and got back; images as descriptors, never bytes)
  only when `LANGSLICE_TRACE_DIR` is set — `langslice linear run
  --trace-dir PATH` sets it for one run. See `docs/current_workflow.md`.

## Boundaries

- Active surface: `src/langslice/`, `tests/`, `docs/`, `README.md`.
- Local-only (do not ship, do not document publicly): `_local/`, `references/`,
  generated outputs, `out/`, `archive/`.
- Keep markdown literal to the code it describes. Behavior change → update the
  relevant doc in the same pass.

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

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

Two methods live as sibling subpackages, connected through top-level registration
bridges rather than direct imports of each other:

- `linear/` — order, position and one in-plane transform per section, as ONE
  agent environment (`langslice linear run FOLDER`): one `StackState`, one
  toolbox built from a `JobSpec`, one job statement, one ADK session that ends
  at `submit` or the turn budget. Tasks (`reorder`/`position`/`transform`, plus
  the opt-in `nonlinear`: a deformation per section via `fit_deformable`, with
  the image tool unless its provider is `none`) are switched on task by task; a task that
  is OFF builds no tools and takes its answer from the host instead. Users see
  them as Positioning / Linear / Nonlinear: `docs/interface_design.md` is the
  target user-facing design and what each host exposes. Every write checkpoints and is undoable, order
  and position must agree at submit, and in-plane alignment happens in the
  main session through the transform tools (`fit_affine`: by default an
  Elastix intensity affine refining the current placement, or the
  silhouette fit; `adjust_transforms`); there is no nested per-section session. Spec: `docs/linear_design.md`. Code map, the lean-
  harness rule, the submit gates and the known ceilings:
  `src/langslice/linear/CLAUDE.md` (loads when working there).
- `nonlinear/` — generative-image registration (image model → optional Elastix fit → report).
  Exactly two border-based routes (a supplied placement corrected in one model
  call, the production path; or a placement-free route, kept for experiments,
  that draws boundaries against an outlined grayscale atlas template), its
  measured design choices, and the cutting-angle lever: `src/langslice/nonlinear/CLAUDE.md` (loads when
  working there).

Shared, top-level:

- `registration_tool.py` — optional stack-agent image tool, enabled by task
  `nonlinear` with an image provider (not `none`): corrects a supplied linear placement using the fixed prompt plus
  per-slice additional notes. No atlas search, replacement prompt, agent rejection
  or fit. Saves the first result and exact artifacts separately from transforms.
  `registration_handoff.py` supplies calibrated geometry; see
  `docs/nonlinear_image_tool.md`.

- `atlas/` — BrainGlobe loading, slice extraction, colored region maps, borders,
  the organized-color LUT for human-review renders, and one side of a region
  (`"CTX:left"`, the section's displayed side, `sides.py`):
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
- `deformable/` — the deformable-fit engine behind the linear agent's
  `fit_deformable` tool (task `nonlinear`; no host or export adapter reads
  its records yet), whose prepared images and Elastix plumbing also run
  `fit_affine`'s Elastix affine: ANTs SyN (optional
  `registration` extra) or Elastix B-spline residual fit of a linearly placed
  atlas plane onto one section — stain vs reference/ABBA Nissl, model lines
  vs merged borders, ANTs label-map channels, sequential per-structure steps,
  masks for tissue/torn edges/exclusions, and the canonical per-section
  record with plausibility diagnostics. Belongs to neither method:
  `src/langslice/deformable/CLAUDE.md` (loads when working there).
  `oblique.plane_index_coordinates` gives any co-registered volume the
  annotation plane's exact pixel grid
- `image_prep.py` — image normalization, pixel-size detection, VLM
  downsampling, and foreground framing (`crop_to_tissue`, `crop_to_mask`), used
  by the linear visual path so histology and atlas sections fill their
  frames comparably. Tissue framing keeps every blob at least a fifth the
  size of the biggest (both bulbs, cerebellum and brainstem, a torn piece), so
  a speck elsewhere on the slide cannot widen the box.
  `host_preprocess` blends a host's multi-page snapshot (one page per channel)
  into a section's DEFAULT appearance (the worker's `preprocess.preview` writes
  that same image for the host's preview); the pages stay raw channels
  (`read_working_pages`, `channel_planes`) for the agent's picture options (`view.channels`),
  its optional `preprocess` tool (`custom_appearance`: weights, CLAHE, ANTs
  N4/denoise) and fitting. `read_working_image` gives
  each section file's small working copy (the smallest TIFF pyramid level of
  at least 1536 px, a JPEG draft decode, otherwise one downsample to 3072 px);
  the linear renders are drawn from it (`EngineContext.working_source`) and
  count file pixels through its scale, so a whole-slide scan is never decoded
  at full size
- `providers/` — model ACCESS methods, never task logic. `registry.py` is
  the taxonomy: canonical names pair vendor with auth — `gemini-api` (Google
  API key, `vlm_config.py`), `openai-api` (API key / endpoint,
  `openai_config.py`), `openai-oauth` (subscription OAuth,
  `openai_oauth.py`: `langslice login`, the `openai-oauth/*` ADK model
  strings — legacy `chatgpt/*` accepted — and gpt-image-2), and `none` (no
  model at all: nonlinear retains a supplied placement or fits a silhouette
  prior, so there is nothing to authenticate). The OAuth path is
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

## Host connectors

`connectors/` holds what is installed into, or configured in, someone else's
program. Code that runs in LangSlice's own environment lives in
`src/langslice/` instead (`integrations/`, `api/`, `mcp_server/`).

- `connectors/fiji/` — Java/SciJava connector loaded into the user's existing
  ABBA. It starts a separate LangSlice Python environment, provides setup and
  account dialogs, and applies worker results through native ABBA actions.
- `connectors/napari/` — planned napari connectors (`docs/napari_plugin_design.md`).
- `connectors/claude-desktop/` — host configuration for `langslice mcp`
  (`src/langslice/mcp_server/`). This is the linear toolbox served over MCP to
  a host that brings its own model: Claude Desktop, or Claude Code locked to
  this one server. ABBA's Claude mode copies a saved-job prompt, and without a
  host `langslice claude prepare FOLDER` saves the same kind of job (checkpoint
  in the job directory, resumed on reopen) and prints the prompt; `start_job`
  returns a Claude-specific statement and status table, and `show_stack` pages
  deliver the opening images. Authenticated localhost events update ABBA live;
  checkpoints/results remain in `~/.langslice/jobs/<id>/` after disconnection.
  The host owns the loop, so there is no turn budget, nudges or image working
  set. This is the subscription-legal route for Claude; LangSlice never
  handles Claude credentials.
- `src/langslice/api/` — JSON-lines worker protocol, desktop setup/authentication,
  and JVM-free host adapters. Existing `abba_python` launchers remain separate.
- `packaging/`, `environment.yml` — worker distribution and wheel checks.
  Publication status and the accepted installation design are in
  `docs/abba_installation.md` and `docs/abba_plugin_design.md`.

## Runtime facts

- Main pipeline: `linear (order, position, in-plane affine) → image-model
  border correction (nonlinear, needs the linear placement) → deformation
  (not designed yet; off by default) → VisuAlign markers → export`.
- `linear` and `nonlinear` are independent; `nonlinear` accepts a position
  from any source, not just `langslice linear`.
- Registration is exactly two border-based routes, chosen automatically by
  whether a placement is supplied: route "supplied" is one image-model call
  that moves a supplied linear/host placement's drawn boundaries onto the
  tissue; route "atlas" (no placement) fits a local silhouette placement and
  asks the model to draw boundaries from nothing against an outlined
  grayscale atlas template, with an optional second corrective call
  (`passes=2`). Raw model replies, extracted boundaries on original histology,
  and fitted atlas overlays are separate. The initial placement and residual
  fit are composed in exported coordinates. See `docs/nonlinear_design.md`;
  there is no hosted-router retry loop.
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

- Active surface: `src/langslice/`, `tests/`, `docs/`, `README.md`,
  `connectors/`, `packaging/`, and their build/environment configuration.
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

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

The package is laid out by LAYER, not by method (folder move, 2026-10-04).
Each layer imports only the layers below it: `core/` < `job/` < `ops/` <
`doors/` : `agent/` (one level; they may import each other) < `hosts/`.
`providers/` (model access) sits beside them: core, job and ops never import
it, nor `google.*`, `litellm`, `openai` or `mcp`; doors, the agent and hosts
may. import-linter enforces this (`[tool.importlinter]` in `pyproject.toml`,
run by `tests/test_import_layers.py` and CI) with no exceptions listed: a
violation is fixed by moving code, never by an `ignore_imports` entry.
`tests/test_core_imports.py` checks what a fresh interpreter loads.
The two methods are task groups over this one stack, not packages: the
linear agent environment (Positioning / Linear, plus the opt-in Nonlinear
deformation) and the image-model border route (`core/nonlinear/`).

- `core/` — the core library: plain inputs (`core.workspace.Workspace`, the
  `core.state.StackState`, ids, numbers) in, PIL pictures with captions
  burned in, numpy arrays and plain records out; never the job layer, ops,
  a door, the driver, a host or a model client:
  `src/langslice/core/CLAUDE.md` (loads when working there). It holds:
  - the linear method's core: `state.py` (`StackState`/`SliceState`),
    `spec.py` (`JobSpec`: every checkbox a host shows), `workspace.py`,
    `display.py`, `appearance.py`, `atlas_fetch.py`, `atlas_grep.py`,
    `opening.py`, `transform.py` (the silhouette fit and the interactive
    transform arithmetic), `deformation.py` (`fit_deformable`'s machinery),
    `discovery.py`, `deepslice.py`; the pictures the tools send
    (`pictures.py`, `placement.py`: every placement picture and the frame it
    is drawn in), the renders that were `linear/render.py` (`sections.py`,
    `captions.py`, `canvas.py` the physical canvas, `sheets.py`,
    `status.py`, `sizes.py`), `jpeg.py` (the one encoding), `layers.py` (a
    placement picture's atlas labels, border mask and frame,
    `pixel_to_atlas_um` in BrainGlobe µm, and `coordinate_map`), `maps.py`,
    and `handoff.py` (a written linear placement as a section render plus
    the native atlas plane mapped onto it: `prepare_linear_registration`,
    what the trace and every deformable fit start from, and
    `correction_fingerprint`). The map of the whole linear method, file by
    file across the layers, is `src/langslice/agent/CLAUDE.md`.
  - `core/atlas/` — BrainGlobe loading, slice extraction, colored region
    maps, borders, the organized-color LUT for human-review renders, and one
    side of a region (`"CTX:left"`, the section's displayed side,
    `sides.py`): `src/langslice/core/atlas/CLAUDE.md`.
  - `core/deformable/` — the deformable-fit engine behind the linear agent's
    `fit_deformable` tool (task `nonlinear`) and behind the fit of the image
    model's lines in the core border routes and the ABBA registration plugin
    (`core/nonlinear/border_fit.py`; no
    export adapter reads its records yet), whose prepared images and Elastix
    plumbing (the one itk-elastix wrapper, `engines.py`) also run
    `fit_affine`'s Elastix affine: ANTs SyN (optional `registration` extra)
    or Elastix B-spline residual fit of a linearly placed atlas plane onto
    one section — stain vs reference/ABBA Nissl, model lines vs merged
    borders, ANTs label-map channels, sequential per-structure steps, masks
    for tissue/torn edges/exclusions, and the canonical per-section record
    with plausibility diagnostics. Belongs to neither method:
    `src/langslice/core/deformable/CLAUDE.md`.
    `oblique.plane_index_coordinates` gives any co-registered volume the
    annotation plane's exact pixel grid.
  - `core/nonlinear/` — generative-image registration (image model →
    optional fit of its lines by `core/deformable/` → report). Exactly two
    border-based routes (a supplied placement corrected in one model call,
    the production path; or a placement-free route, kept for experiments,
    that draws boundaries against an outlined grayscale atlas template), its
    measured design choices, and the cutting-angle lever:
    `src/langslice/core/nonlinear/CLAUDE.md`. Also here:
    `registration_tool.py`, the optional stack-agent image tool, enabled by
    task `nonlinear` with an image provider (not `none`): corrects a
    supplied linear placement using the fixed prompt plus per-slice
    additional notes. No atlas search, replacement prompt, agent rejection
    or fit. Saves the first result and exact artifacts separately from
    transforms. The image model is an argument
    (`providers.registry.ImageModel`, resolved by a door); the calibrated
    geometry and the correction fingerprint are `core/handoff.py`'s;
    `registration_handoff.py` holds `run_linear_registration`, which takes
    the model as an argument; see `docs/nonlinear_image_tool.md`. A model
    call is only ever the `image_call` a door passes in (refused without
    one); provider names are the core's own table, `core/provider_names.py`.
  - `oblique.py` — arbitrary-plane sampling out of an atlas volume plus
    (pitch, yaw) fitting (vendored from brainglobe-registration, BSD-3).
    `sample_oblique_annotation` is the label-safe entry point: sampling
    happens in float32, whose mantissa cannot hold the larger Allen ids, so
    it compacts the volume to dense indices first. THE largest lever
    measured on the nonlinear fit — see the `core/nonlinear/` entry.
  - `space.py` — coordinate and orientation conventions.
  - `affine.py` — shared in-plane affine core: the silhouette (moments) fit
    of a section onto an atlas section (`silhouette_affine`, the one
    wrapper, cutting angles included), plus the rotation/scale/translate
    matrix builder, the normalized 6-number parameter convention and
    `pixel_center_map` (pixel centres through a resize and offset). Used by
    `core/transform.py`, `core/nonlinear/` and `core/deformable/`; belongs
    to neither method.
  - `image_prep.py` — image normalization, pixel-size detection, VLM
    downsampling, and foreground framing (`crop_to_tissue`,
    `crop_to_mask`), used by the linear visual path so histology and atlas
    sections fill their frames comparably. Tissue framing keeps every blob
    at least a fifth the size of the biggest (both bulbs, cerebellum and
    brainstem, a torn piece), so a speck elsewhere on the slide cannot widen
    the box. `host_preprocess` blends a host's multi-page snapshot (one page
    per channel) into a section's DEFAULT appearance (the worker's
    `preprocess.preview` writes that same image for the host's preview); the
    pages stay raw channels (`read_working_pages`, `channel_planes`) for the
    agent's picture options (`view.channels`), its optional `preprocess`
    tool (`custom_appearance`: weights, CLAHE, ANTs N4/denoise) and fitting.
    `read_working_image` gives each section file's small working copy (the
    smallest TIFF pyramid level of at least 1536 px, a JPEG draft decode,
    otherwise one downsample to 3072 px); the linear renders are drawn from
    it (`core.workspace.Workspace.working_source`) and count file pixels
    through its scale, so a whole-slide scan is never decoded at full size.
  - `landmark_warp.py`, `landmark_elastix.py` — landmark deformations
    (saved spline compatibility).
  - plain shared tables: `provider_names.py` (canonical provider names and
    aliases, re-exported by `providers.registry`) and `media_keys.py` (the
    keys a tool's media travels under, re-exported by `doors.tools`).
  - `abba_affine.py`, `abba_spline.py` — a stored affine or landmark spline
    in ABBA's centred world millimetres: the host rows the linear snapshot
    worker (`doors/api/abba_worker.py`) emits.
- `job/` — the job layer: the `Job` (`job.py`: state, undo/redo, the
  checkpoint `checkpoint.py`, the submit gates, background corrections) and
  the job folder. Everything a job writes lives in `<images>/langslice/`,
  next to the images every host hands LangSlice (`job.json` settings +
  format version, `state.json`, `history/` one file per undo step,
  `sections/<stem>/`, `views/` + `views.jsonl`: every picture the model was
  shown, as the JPEG it received, placement pictures with atlas labels,
  border mask and frame, `exports/`, `logs/`, and the reference card
  `AGENTS.md` = `CLAUDE.md`). Paths in job files are relative to it; old
  layouts are moved in on open, newer ones refused; `--job-dir` /
  `JobSpec.job_dir` puts it elsewhere (a folder holding another image
  folder's job is refused); a read-only image folder falls back to
  `~/.langslice/jobs/<id>/`; saved Claude jobs are found by id through
  `~/.langslice/jobs/<id>.json`. `formats.py` and `quint.py`
  (QUINT/QuickNII/VisuAlign JSON) write the public files:
  `src/langslice/job/CLAUDE.md` (loads when working there).
- `ops/` — the verbs: every write to a stack (positions, order, orientation,
  damage, appearance, notes, undo/redo, the in-plane transform and its fits,
  the image-model trace with the model call passed in, the deformable fit
  and `keep_linear`, submit) and every viewing tool's read, as a function
  on the job: a write is one undo step, each returns a plain record, and
  given the call's display options the core's pictures; no model wording,
  no look-before-commit gates. Each agent tool is argument checking, one
  call here and its wording; `registry.py` maps every verb to its
  operation, read or write, task group and the specs that have it
  (`enabled(spec)`): the one list every door is built from. Imports the
  core and the job layer only, never a door:
  `src/langslice/ops/CLAUDE.md` (loads when working there).
- `doors/` — the doors over the verbs (layered refactor, phase 5).
  `declarations.py` is each verb's ONE declaration (arguments and the
  description a model reads); the agent tools and the MCP tools are built
  from it and the registry. `doors/tools/` is the native agent-tool door
  (`toolbox.py` the tool bodies, `arguments.py`, `view_options.py`,
  `media.py` the ADK message parts); `doors/mcp/` the MCP server
  (`langslice mcp`); `doors/api/` what the MCP door, the CLI and the engine
  service share (the engine contract's Pydantic models, the
  quick-affine/export runtime, setup and credentials, saved Claude
  jobs, the JVM-free linear snapshot worker `abba_worker.py`). The agent CLI for coding
  agents (`doors/cli/`: `langslice job FOLDER VERB`, `langslice ops`,
  `langslice schema`; one JSON envelope on stdout, exit codes 0/2/3/4,
  pictures as file paths, `--dry-run`, `--background`), the script door
  (`import langslice; langslice.open_job(folder)`, `coordinate_map`,
  `load_atlas`; no agent framework loaded) and the job folder's reference
  card (`AGENTS.md` + `CLAUDE.md`, written into every job folder).
  `doors/cli/` also holds every other command, one module per group
  (`langslice/cli.py` keeps the entry point); the host commands (`abba`,
  `serve`) are `hosts/cli.py`, reached by module path
  (`doors.cli.HOST_COMMANDS`): `src/langslice/doors/CLAUDE.md` (loads when
  working there);
  `docs/agent_cli.md`.
- `agent/` — the ADK driver of the linear agent environment (`langslice
  linear run FOLDER`): one `StackState`, one toolbox built from a
  `JobSpec`, one job statement (`prompt.py`), one ADK session
  (`session.py`, `engine.py`) that ends at `submit` or the turn budget,
  plus `trace.py`, `cost.py`, `live.py`, the ADK plugins and model
  resolver. Tasks (`reorder`/`position`/`transform`, plus the opt-in
  `nonlinear`: a deformation per section via `fit_deformable`, with the
  image tool unless its provider is `none`) are switched on task by task; a
  task that is OFF builds no tools and takes its answer from the host
  instead. Users see them as Positioning / Linear / Nonlinear:
  `docs/interface_design.md` is the target user-facing design and what each
  host exposes. Every write checkpoints and is undoable, order and position
  must agree at submit, and in-plane alignment happens in the main session
  through the transform tools (`fit_affine`: by default an Elastix
  intensity affine refining the current placement, or the silhouette fit;
  `adjust_transforms`); there is no nested per-section session. Spec:
  `docs/linear_design.md`. Code map of the whole method across the layers,
  the lean-harness rule, the submit gates and the known ceilings:
  `src/langslice/agent/CLAUDE.md` (loads when working there).
- `providers/` — model ACCESS methods, never task logic. `registry.py` is
  the taxonomy: canonical names pair vendor with auth — `gemini-api` (Google
  API key, `vlm_config.py`), `openai-api` (API key / endpoint,
  `openai_config.py`), `openai-oauth` (subscription OAuth,
  `openai_oauth.py`: `langslice login`, the `openai-oauth/*` ADK model
  strings — legacy `chatgpt/*` accepted — and gpt-image-2), and `none` (no
  model at all: nonlinear retains a supplied placement or fits a silhouette
  prior, so there is nothing to authenticate). `images.py` is the image
  transport (`generate_warped_segmentation_image`, formerly
  `nonlinear/providers.py`). `registry.resolve_image_model` is the one place
  a provider name becomes an image-edit call (`ImageModel`: provider,
  model, `call`); only a door resolves it (the toolbox binding
  `build_tools(image_model=...)`, the CLI and API runtime, the ABBA plugin),
  and the operations (`registration_tool`, `ops.traces`,
  `run_linear_registration`) receive it. The OAuth path is NOT the OpenAI
  API: it talks to the separate Codex backend
  (`chatgpt.com/backend-api/codex`), whose image tool ignores
  `model`/`size`/`quality` and matches the input image's aspect exactly.
  Registration uses the raw `codex/images/edits` endpoint (`edit_image`),
  which delivers the prompt verbatim with no routing model in between (the
  hosted-router Responses path was deleted; see the nonlinear section).
  Default review model: `openai_oauth.DEFAULT_REVIEW_MODEL`
  (`openai-oauth/gpt-5.6-sol`) — and future providers (anthropic-api,
  openrouter-api, qwen-api, ...) are added there and nowhere else; legacy
  spellings google/openai/chatgpt resolve as aliases. Task-level semantics
  stay OUT of providers: the registration edit-vs-generate decision is
  `SegmentationGenerationRequest.mode` (`core/nonlinear/`), and each
  transport merely translates it (`images.edit` endpoint, `action` on the
  Responses image_generation tool).
- `hosts/` — host connectors that run in LangSlice's own environment:
  `hosts/integrations/` (the ABBA registration plugin, the live linear
  mirror, the ABBA viewer and log: `src/langslice/hosts/integrations/CLAUDE.md`)
  `hosts/api/` (the engine service the Fiji connector starts, and the ABBA
  plugin's `nonlinear.abba` worker) and `hosts/cli.py` (`abba`, `serve`):
  `src/langslice/hosts/CLAUDE.md`.
- Compatibility shims, for the sibling repos only (LangSlice imports none;
  import-linter's `no-shims-inside` contract): `linear/` (`JobSpec` & co.,
  `run`, `engine`, `spec`, `state`, `toolbox`, `trace`, `transform`,
  `render`), `atlas/` (+ `core`), `nonlinear/` (+ `image_gen_helpers`,
  `image_gen_registration`, `border_refinement`), `integrations/` (+
  `abba_linear`), `adk/` (the media keys), `space.py`, `oblique.py`,
  `affine.py`, `image_prep.py`, `registration_handoff.py`. Each hands back
  the moved module (or re-exports a package's public names).

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
`src/langslice/` instead (`hosts/`, `doors/mcp/`).

- `connectors/fiji/` — Java/SciJava connector loaded into the user's existing
  ABBA. It starts a separate LangSlice Python environment, provides setup and
  account dialogs, and applies worker results through native ABBA actions.
- `connectors/napari/` — planned napari connectors (`docs/napari_plugin_design.md`).
- `connectors/claude-desktop/` — host configuration for `langslice mcp`
  (`src/langslice/doors/mcp/`). This is the linear toolbox served over MCP to
  a host that brings its own model: Claude Desktop, or Claude Code locked to
  this one server. ABBA's Claude mode copies a saved-job prompt, and without a
  host `langslice claude prepare FOLDER` saves the same kind of job (in the
  job folder next to the sections, resumed on reopen; the id leads there
  through `~/.langslice/jobs/<id>.json`) and prints the prompt; `start_job`
  returns a Claude-specific statement and status table, and `show_stack` pages
  deliver the opening images. Authenticated localhost events update ABBA live;
  checkpoints/results remain in the job folder after disconnection.
  The host owns the loop, so there is no turn budget, nudges or image working
  set. This is the subscription-legal route for Claude; LangSlice never
  handles Claude credentials.
- `src/langslice/hosts/api/` (the engine service) and
  `src/langslice/doors/api/` (its JSON-lines protocol models, desktop
  setup/authentication and the JVM-free host adapters). Existing `abba_python` launchers remain separate.
- `packaging/`, `environment.yml` — worker distribution and wheel checks.
  Publication status and the accepted installation design are in
  `docs/abba_installation.md` and `docs/abba_plugin_design.md`.

## Runtime facts

- Main pipeline: `linear (order, position, in-plane affine) → image-model
  border correction (nonlinear, needs the linear placement) → deformation
  (not designed yet; off by default) → VisuAlign markers → export`.
- `linear` and `nonlinear` are independent; `nonlinear` accepts a position
  from any source, not just `langslice linear`.
- Every way of using LangSlice goes through one job folder and its verbs
  (the agent run, the agent CLI `langslice job`, the library, MCP, the agent
  tools); the one-shot `langslice nonlinear register` / `register.run`
  pipeline was removed 2026-10-04.
- Image-model registration is exactly two border-based routes (the core's
  `core/nonlinear/border_registration.py`): route "supplied" is one
  image-model call that moves a linear/host placement's drawn boundaries onto
  the tissue — on a job, the `trace_borders` verb; route "atlas"
  (placement-free) asks the model to draw boundaries from nothing on the
  clean section against an outlined grayscale atlas template at its
  position, with an optional second corrective call (`passes=2`) — on a job,
  the HIDDEN scripting verb `trace_from_atlas` (`ops/traces.py`,
  `registry.Verb.hidden`: the agent CLI and the library call it by name; no
  listing, agent tool, MCP tool or public doc offers it; kept for
  experiments). Both record the same trace, so the fit of the model's lines
  is `fit_deformable`'s traced fit sections on top of the section's linear
  placement (the deformable package's; the Elastix residual fit was retired
  2026-10-04). Raw model replies, extracted boundaries on original
  histology, and fitted atlas overlays are separate. See
  `docs/nonlinear_design.md`; there is no hosted-router retry loop.
- Positions are atlas-native millimeters from the anterior edge of the volume.
- Atlas orientation assumptions are centralized in `src/langslice/core/space.py`,
  which derives AP/DV/ML axis indices from the atlas orientation via
  `brainglobe_space` and requires the AP axis to increase anterior→posterior.
- A job folder's public files (`registration.json` on every write; per
  section `coords.tif`, `labels.tif`, `labels_fiji.tif` + `labels.csv`,
  `residual.tif`; QuickNII/VisuAlign JSON in `exports/`, at submit and by the
  CLI/library verb `export_maps`) are derived from `state.json` and never read
  back: `docs/file_formats.md`.
- Optional debug traces are written only when `LANGSLICE_VLM_DEBUG_DIR` is set.
- A job's files live in its job folder, `<images>/langslice/`
  (`src/langslice/job/CLAUDE.md`; `--job-dir` moves it, a read-only image
  folder falls back to `~/.langslice/jobs/<id>/`); nothing else is written
  beside the images.
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

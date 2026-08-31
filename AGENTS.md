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
  reasons. Tools: `view_slices`, `fetch_atlas`, `stack_positions`
  (index, id, `position_mm`, `spacing_to_next_mm` — no nominal comparison, no
  legend), `interpolate_between` (computes, writes nothing), `set_positions`
  (writes, returns the same rows) and `submit_positions`. The gates stay,
  because a constraint stating a fact is not coaching: `submit_positions`
  refuses `interval_breaks` the agent's own written positions do not show (the
  interval there must exceed 1.5x the stack's median written spacing) and a
  submission whose positions run the wrong way along the stack's known axis
  direction (`DIRECTION_REVERSED`, a plain code check). Refusal messages state
  the numbers that caused them and stop. There used to be a second gate here —
  `atlas_structures_at`/`structure_range` landmark tools plus a
  `submit_positions` end-anchor requirement — but a benchmarked ablation (both
  test brains) found it made placement worse, not better, so it was deleted
  rather than kept behind a flag. Interpolation beyond the
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
  provider adapters, Elastix runtime, affine and
  nonlinear result types, and `quick_affine.py` (silhouette affine preview;
  note the CLI groups `quick-affine` under `linear`). There is ONE path:
  direct — the handwritten base prompt goes to the image model verbatim
  (openai-oauth: the raw `codex/images/edits` endpoint, no routing model in
  between), one generation per run. The hosted-router session (`router.py`,
  a GPT conversation wielding the `image_generation` tool with self-review
  iteration) was DELETED 2026-08-28 on LSD_910 ground-truth evidence: on the
  anterior benchmark slice the first generation tracked the tissue and the
  router's "improved" retries degraded it into a generic atlas plate
  (truth-Dice 0.54 vs a clean first paint), and its prompt refinement is a
  standing corruption risk — the image model does better with our prompt
  untouched. The canvas pad the router used to choose is now the
  `--canvas-pad` knob (0-1.5, `RegisterRequest.canvas_pad`). The Elastix
  report is the ONLY
  diagnostic — the old `confirm_registration` gate and per-model prompt
  profiles were deleted. One base prompt (the original handwritten text, in
  `model_prompts.py`) serves every image model; the working
  editing copy is `_local/nonlinear_prompts.md`.
  The MODEL-FACING atlas render is drawn at canvas resolution, never
  NEAREST-upscaled: `render.filled_regions` traces each region at atlas
  resolution, low-pass filters the outline and fills the polygon at ~2048px
  (`_generate_colored_region_slice(..., smooth=True)`, the default), so a
  25/39um atlas reaches the model with boundary detail instead of a voxel
  staircase. Fills stay flat and exact-palette (classification is
  nearest-color), each region is stroked as well as filled so neighbours
  overlap instead of leaving background seams, and enclosed pinholes are
  closed. The ELASTIX side asks for `smooth=False` — that render is
  classified back and warped, so it must stay pixel-exact. `_fit_to_canvas`
  picks its resampling filter from the image: NEAREST for a flat region map,
  LANCZOS for the continuous-tone grayscale template.
  `generate_registration_candidate(image_prompt)` runs the whole downstream
  chain immediately (image model → Elastix → pixel classification → borders)
  and returns the images plus an Elastix report: error codes with the numbers
  behind them (`REGION_MISSING`, `REGION_COLLAPSED`, `WARP_FOLDS`,
  `TISSUE_UNCOVERED` — computed on the warped CLASSIFIED result, so color
  drift that still classifies coherently passes clean; pixels the edit left
  untouched (identical to the input canvas) are masked to background before
  classification, because a preserved white background otherwise classifies
  as the atlas root color; there is one real
  failure in this pipeline, Elastix not working on the image, and every code
  is a measured cause of it, not a standalone judgment); passing
  `generated_image` skips the provider call and evaluates an
  externally generated image (offline re-derivations, benchmarks). Under `--palette leaf-borders` the classifier's palette gains the
  hairline color of every region (`darker(color) -> that region's id`),
  because a model that paints the delineation back would otherwise have it
  cut through its own regions as background: measured on the Allen sagittal
  render, hairlines are ~20% of the foreground and land 70-140 RGB from any
  fill color. With the line color known, no border pixel becomes background
  and no new region id appears; the residual (family recovery 0.93 vs a flat
  render) is entirely WHICH of two neighbours owns a shared 2px line, which
  is two orders below the B-spline grid. Nothing else changes: the
  Elastix-side render (`smooth=False`) is byte-identical in both styles. The
  Elastix step registers
  the two label maps as joint RGB at merged-family granularity (one
  AdvancedMeanSquares metric per channel plus the bending penalty; both
  maps merged through a SINGLE family mapping built on their union of ids
  so a family renders one color on both sides). RGB is a benchmarked
  choice, not a leftover: one-binary-channel-per-family (the label-
  registration literature's standard) and clamped per-family signed
  distance maps were both built and measured on the sag140 debris case, and
  every variant came out worse than not deforming at all (family agreement
  0.46-0.54 vs 0.675 identity vs 0.80 RGB) — Elastix's sampled ASGD
  optimizer starves on channels whose gradient lives only in thin boundary
  shells. Known residual flaw: in RGB a mismatched label earns partial
  credit whenever its color sits nearer than black, so preserved debris
  (dark pixels classify to far-away regions' shades, measurably CLOSER to
  the palette than real paint) can pull boundaries slightly; the candidate
  fix is a dense-evaluation engine (NiftyReg 4-D SSD / ANTs label
  registration), not more Elastix channels. Also measured: the
  TransformBendingEnergyPenalty at default-scale weights is INERT against
  0-255 mean-squares values (needs ~1e6-1e8 to bite) — and VisuAlign markers are sampled from the composed
  transformix deformation field at the B-spline control-grid spacing (the
  transform's own resolution; consumer-specific densities belong in the
  integration adapters), never read off the B-spline parameter map, which
  dropped the affine stage and mistook coefficients for displacements. The
  The block's CUTTING ANGLES are an input (`--pitch-deg`/`--yaw-deg`,
  `RegisterRequest`, `generate_registration_candidate`): non-zero angles
  reslice every atlas render — model-facing map, grayscale template,
  Elastix-side map, classification palette, leaf overlay — on that oblique
  plane instead of taking it flat off the voxel grid. Default 0 (flat),
  because nothing upstream fits them yet, but this is the single biggest
  lever in the whole fit and it dwarfs every knob inside Elastix. Measured
  on all 33 LSD_910 hand registrations (block cut at 4 degrees, angles set
  by hand to that): fit-only family dice 0.912 -> 0.945, boundary p95 39px
  -> 9px, better on 25 of 33 slices — and re-deriving 32 saved paintings
  through the fit, +0.022 dice paired, better on 29 of 32, against a
  per-draw sd of 0.02, with the paintings still made from a FLAT reference.
  Under a flat plane the residual is wide BANDS of misplaced anatomy
  (hippocampus, brainstem, one hemisphere ahead of the other); at the right
  angles it collapses to boundary-width lines. `pixel_size_um` puts every
  atlas render at TRUE physical scale on the slice canvas (each render's
  um/px derived from the annotation slice's anatomy, never the letterbox
  padding; anatomy centered via `_anatomy_focus`, and the canvas auto-grows
  through `_pad_to_contain_atlas` when the true-scale anatomy would exceed
  it) so the model sees two comparably sized brains; no pixel size falls
  back to fit-to-canvas. Known residual, measured over all 33 M01 hand
  registrations by physical extent of the GT region maps: the specimen runs
  ~10%% (ML) to ~14%% (DV) smaller than the Allen average, so the true-scale
  atlas lands slightly larger than the tissue and the fit side alone pays
  ~0.004 dice for it — calibration is on for the model-side benefit of
  size-matched references. Settled ablations on the same panel:
  a finer B-spline grid (/48, /72 rather than /36) buys +0.004 dice, the
  fixed-mask dilation is flat from 0 to 32px, and the bending penalty is
  flat at 1e5-1e6 and HARMFUL at 1e7 — the tail is a plane problem, not a
  regularization one. The
  canvas pad (`--canvas-pad`, 0-1.5) grows the working
  canvas by a margin matched to the slice's own background color (never
  black-on-white) so a fragment or hemibrain's COMPLETE painted anatomy can
  exceed the original image bounds; after all padding the canvas silently
  pads one axis if the aspect ratio falls outside what the image path can
  output (`model_prompts.aspect_ratio_limits`, gpt-image-2 =
  1:3..3:1 — pad only, never crop; a mismatched ratio would undo edit-mode
  pixel alignment); markers are reported in
  original-image pixels, unclipped — out-of-bounds correspondences are
  legitimate there, and clipping is an adapter concern. It
  runs after a linear
  placement step, whether that step is `langslice linear ...` or the user's
  own tool (in ABBA/QUINT workflows, linear placement happens first and
  LangSlice-nonlinear replaces the manual spline/BigWarp deformation step).

Shared, top-level:

- `atlas/` — BrainGlobe loading, position helpers, slice extraction, colored
  region maps, borders.
  Also `recolor.py`: organized structure colors for atlases whose native
  palettes mislead image-gen models. `color_lut(atlas)` keeps native colors
  when they are hierarchy-organized, joins the true Allen CCF colors
  (vendored `allen_colors.json`) for trees with
  enough Allen overlap — the all-white atlases (Osten, Princeton, adult Kim)
  and the Waxholm rat family. The join matches by normalized name, a
  terminology bridge (classical/embryological vocabulary onto Allen's, e.g.
  mesencephalon→Midbrain, white matter→fiber tracts), name variants, and
  acronym only when the names corroborate it; unmatched subregions inherit
  their deepest matched ancestor's color (the root never joins or seeds
  inheritance). It generates an Allen-style palette for
  foreign trees: nested hue subdivision (each division's hue range is
  proportional to its subtree), quantized to at most `MAX_FAMILIES` flat
  family colors, because few flat colors IS the Allen convention. Auto-
  detection is data-driven — degenerate = one color for
  everything; disorganized = child colors uncorrelated with parents — and
  flat or small trees always keep native colors. Both colored-region-map
  paths (`atlas/core.py` and `nonlinear/image_gen_helpers.py`, render and
  pixel-classify alike) draw from this one LUT.
  A DERIVED palette (Allen join or generated) is then organized: leaf-level
  Allen colors are not a usable palette — Allen encodes hierarchy in hue, so
  a cortex-dominated slice came out as a dozen near-identical greens the
  pipeline merges away anyway (that was the WHS rat "washed out" bug). Colors
  closer than `MERGE_EPS` (40, `_family_mapping`'s own radius) collapse into
  one unit, and the surviving units are pulled at least `MIN_SEPARATION` (60)
  apart, nudging value/saturation before hue so a family keeps its identity.
  Native palettes are left exactly as the atlas authored them.
  `color_lut` has ONE table and no modes: the `--palette` knob picks a
  model-facing render STYLE, never a color. `"family"` (default) draws flat
  regions, one color per registration unit; `"leaf-borders"` draws the same
  flat colors plus the Allen-Reference-Atlas plate treatment — every leaf
  boundary delineated by a hairline in a darker shade of the region's own
  color (`render.darker`, RGB × `BORDER_DARKEN` = 0.7, which moves HSV value
  only, so hue and saturation still name the region), 2px at a 2048 canvas
  with family boundaries at 1.8× that. Lines are drawn LINE_8 like the fills,
  on the smoothed sub-pixel contours: an anti-aliased line would blend two
  region colors into pixels belonging to neither, and the render has to stay
  classifiable to exact palette colors. Leaf shades were tried here first and
  rejected (Nash: "everything is uniformly worse in our colors") — borders
  express leaves without touching the palette. The style is a PROCESS-WIDE
  setting (`LANGSLICE_ATLAS_PALETTE`, `active_palette()`, `use_palette()`,
  and `langslice nonlinear register --palette`), because the classifier has
  to know whether hairlines were painted (see the nonlinear section);
  `generate_registration_candidate(palette=...)` applies it around one call
- `integrations/` — one module per external registration ecosystem.
  `quint.py`: QUINT/QuickNII/VisuAlign-compatible JSON export (anchoring
  vectors; file-based, formerly top-level `export.py`).
  `abba.py`: LangSlice as an abba-python registration plugin
  (`enable_langslice_registration`, `register_selected_slices`). Implements
  ABBA's `SimpleRegistrationPlugin` socket from Python via JPype: the fixed
  image requested from ABBA is its atlas *coordinate* channels (per-pixel
  AP/DV/ML mm), so the adapter samples the BrainGlobe volumes at exactly
  those coordinates — no offset constants, no axis assumptions (the measured
  ABBA↔brainglobe AP offset is ~0.99, not 1.0; never hardcode it). The
  nonlinear result returns as a serializable invertible thin-plate-spline
  (`InvertibleWrapped2DTransformAs3D` — the plain wrapper is not invertible)
  sampled from the Elastix deformation field, and lands on the slice's
  registration stack like a native step (undo, state save/reload included).
  Reopening a saved state requires the plugin registered first, else ABBA
  drops the step. `install_gui` adds a `Register > LangSlice` menu entry via
  ABBA's registration-plugin UI registry (PyCommandBuilder dialog; needs the
  `pyimagej-scijava-command` artifact, test-scope in ABBA, added as a scyjava
  endpoint before JVM start — `run_gui_session` handles it), and
  `langslice abba` is the CLI launcher: full ABBA GUI with LangSlice in the
  Register menu, one command. Ships in the recommended
  `langslice` conda env — environment.yml carries openjdk 11 + maven, and
  `pip install -e ".[abba]"` adds abba-python (a separate env from the user's
  `abba`/`deepslice` envs). Live-session spikes: `_local/abba_spike/`
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
  `linear/whole_brain/transforms.py` and `nonlinear/quick_affine.py`; belongs
  to neither
- `image_prep.py` — image normalization, pixel-size detection, VLM
  downsampling, and foreground framing (`crop_to_tissue`, `crop_to_mask`), used
  by the whole-brain visual path so histology and atlas sections fill their
  frames comparably. Tissue framing crops to the LARGEST connected blob, so a
  fragment or a speck elsewhere on the slide cannot widen the box
- `providers/` — model ACCESS methods, never task logic. `registry.py` is
  the taxonomy: canonical names pair vendor with auth — `gemini-api` (Google
  API key, `vlm_config.py`), `openai-api` (API key / endpoint,
  `openai_config.py`), `openai-oauth` (subscription OAuth,
  `openai_oauth.py`: `langslice login`, the `openai-oauth/*` ADK model
  strings — legacy `chatgpt/*` accepted — and gpt-image-2). The OAuth path is
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
- `adk/` — ADK harness helpers (`plugins.py`, `model_resolver.py`,
  `sdk_helpers.py`)
- `api/` — Pydantic engine contract, runtime wrappers, and the stdio service
  behind `langslice serve`
- `cli.py` — CLI entry: `langslice linear {estimate, estimate-brain,
  quick-affine}`, `langslice nonlinear {register}`, and top-level `version`,
  `login`, `serve`, `collect-traces`
- `agent_trace.py` — structured trace helpers

### Other top-level

- `tests/` — pytest coverage (mirrors `src/langslice/` layout)
- `docs/`, `README.md` — maintained documentation

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
- Whole-brain agent sessions write a full-content JSONL trace (what the agent
  was shown, said, called, and got back; images as descriptors, never bytes)
  only when `LANGSLICE_TRACE_DIR` is set — `langslice linear estimate-brain
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

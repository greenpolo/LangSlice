# LangSlice `agent/` — LangSlice's own agent driver

Package guide for `src/langslice/agent/`: the ADK driver that runs the
linear agent environment, and the rules that environment follows. The
repo-level `CLAUDE.md` holds the project-wide rules. `AGENTS.md` here is a
verbatim copy — edit one, mirror to the other. The design this implements is
`docs/linear_design.md`; this file is the map of the code, not a second spec.

## Where the linear method lives

The method spans the layers; each package has its own guide:

| Layer | What | Guide |
|---|---|---|
| `core/` | `JobSpec`, `StackState`, `Workspace`, renders, pictures, fits, the deformable engine | `core/CLAUDE.md` |
| `job/` | `Job` (undo, checkpoint, submit gates, image corrections), the job folder | `job/CLAUDE.md` |
| `ops/` | one operation per verb, `ops/registry.py` (`VERBS`, `enabled(spec)`) | `ops/CLAUDE.md` |
| `doors/` | the declarations, the toolbox (`doors/tools/`), MCP, the agent CLI, the library | `doors/CLAUDE.md` |
| `agent/` | this driver: engine, session, job statement, plugins, model resolver, trace, cost | here |

`linear/` holds only re-export shims for sibling repos (`from langslice.linear
import JobSpec, run`, `engine`, `toolbox`, `trace`, ...); LangSlice never
imports them (import-linter's `no-shims-inside` contract).

## One environment, not a pipeline

`langslice linear run FOLDER` is ONE agent environment over a stack of
sections: one `StackState`, one toolbox, one job statement, one ADK session
that ends at `submit` or the turn budget. A single section is a stack of one.
There is no node graph, no per-step agent, no single-section agent and no
sub-session: looking at the whole stack once yields the information for
every task, and splitting it re-pays for that reading.

Every write returns its picture (the computer-use pattern): `adjust_transforms`
writes the transform and returns the result, `orient_slices` returns the
re-oriented sections, `set_positions` returns the placements not already seen
at the same geometry, `fit_affine` always writes. Nothing previews except
`fit_deformable`'s 2–4 `candidates`; everything is undoable.
`adjust_transforms` batches up to four independent sections in one call and
one undo step; a dependent refinement waits for the first picture.

Tasks (`JobSpec.tasks`) are switched on one by one; a task that is OFF builds
no tools and takes its answer from `spec.inputs`. Users see Positioning
(`reorder` + `position`), Linear (`transform`) and Nonlinear (`nonlinear`)
(the user-facing task names). Which verbs a run has is
`ops.registry.enabled(spec)`: `orient_slices`, `fit_affine`,
`adjust_transforms` and `set_cutting_angles` belong to `transform` (a mirror
is the sign of the in-plane affine, so flipping is Linear, never
Positioning); `trace_borders`, `grep_atlas` and `fit_deformable` to
`nonlinear`; `mark_damaged` and `preprocess` are host switches
(`agent_damage`, `agent_preprocessing`); `search_position` is opt-in
(`--bayesian`).

## Files

- `engine.py` — `EngineContext` (`doors.jobs.JobContext` plus the model),
  `build_context(spec)` (the job folder from `job.layout.locate_job_folder`:
  `spec.job_dir`, next to the images, or `~/.langslice/jobs/<id>/` for a
  read-only image folder), `build_seed_message` (the opening strips,
  `doors.tools.media.opening_parts`, and `doors.statement.status_and_notes`),
  `save_opening` / `with_seed_views` (the opening saved as the job's views,
  their paths on the live seed event), `run_session` (the system instruction
  is `prompt.build_job_statement` plus the user's notes from `job.json`,
  `doors.statement.read_notes`; the tools are
  `doors.tools.media.packaged_tools(box.tools)`; the post-submit
  `DEBRIEF_PROMPT` when `spec.debrief`, its answer on `state.debrief`), and
  `run(spec, emit=, atlas_loader=, on_write=, on_event=, on_open=)`:
  `Job.open`, the job folder's card, the toolbox, the session, `close_job`
  (image corrections settled, pictures written) even without a submit,
  `Job.emit_results`. *on_write* is called with the opened state and after
  every checkpoint or reload (`Job.observe`); *on_open* gets the job and
  context first. `ingest` and `apply_host_inputs` are re-exported from
  `job.job` for SliceBench. No post pass: the session is the whole run.
- `session.py` — `build_agent` (the ADK `LlmAgent`; `reasoning` set on any
  resolved model exposing `reasoning_effort`), `build_plugins`,
  `run_agent_session` (the loop: nudges when the model answers without a
  tool, the turn budget `DEFAULT_MAX_ITERATIONS` 60, the budget stops and
  their grace call, the debrief) and `TokenTally` (every call's usage
  printed and traced; `paid` is uncached input plus cached input at
  `CACHED_TOKEN_WEIGHT` 0.13, an input-only proxy). `JobSpec.max_quota_percent`
  (25) ends the session when this run's share of the OAuth usage window
  reaches it; `JobSpec.max_input_tokens` (default None) when one request's
  reported input, cached included, passes it. Either stop gets ONE more
  call, asking for `submit` (`_BUDGET_GRACE`, `_CONTEXT_GRACE`); a stop never
  breaks out of a turn, so the pending tool call is answered first and the
  history stays well formed.
- `prompt.py` — `build_job_statement`: the job, the run facts, ONE factual
  line per tool that exists (`tool_line`, worded for the door that reads it:
  `agent`, `mcp`, `cli`; `OPENING_PLACES` says where the opening pictures
  are; the CLI's trace verbs answer once their call has landed,
  `_CLI_LINES`), the hard constraints, the per-task user notes
  (`task_notes`), the short "Method:" section, and with an image model the
  base trace prompt. `display_facts` reads the run's raw channels and atlas
  channels; `display_lines` describes `view` once (`PICTURE_TOOLS`,
  `ATLAS_CHANNEL_LINES`), with the `view.resolution` range when the caller
  sizes pictures (`auto`). Every door's statement is assembled in
  `doors/statement.py`.
- `plugins.py` — the ADK plugins every session runs with:
  `WorkingSetImages` (in ADK's `ContextFilterPlugin`, below),
  `ToolMediaDeliveryPlugin` (after the filter: marks the placement pictures
  that reached the model as seen), `StrictArgumentsPlugin` (ADK drops unknown
  top-level arguments before a tool runs; this refuses them first, with
  `doors.tools.arguments.argument_refusal`), `ModelCallPacingPlugin`
  (`LANGSLICE_ADK_MODEL_CALL_DELAY_S`) and `RequestCapturePlugin`
  (`LANGSLICE_ADK_CAPTURE_REQUESTS_DIR`).
- `model_resolver.py` — `resolve_adk_model(model)`: `openai-oauth/<model>`
  -> `providers.openai_oauth.OpenAIOAuthLlm`; `gemini-api/<model>` and bare
  Gemini ids -> ADK's own; `gemma-*` -> ADK's `Gemma`; `litellm-proxy:`,
  `openrouter:`, `ollama:` (and a bare `name:tag`) and `gpt-*`/`o*` ids ->
  ADK's `LiteLlm` (the litellm extra; a missing install is a `RuntimeError`
  naming it). `LANGSLICE_ENDPOINT` sends every model string to that
  OpenAI-compatible server. `default_http_options`: timeout and retries for
  every model call. `env_value` / `env_float` read the environment.
- `trace.py` — `SessionTrace`, the full-content JSONL trace (what the agent
  was shown, said, called and got back; images as descriptors, never
  bytes), written only when `LANGSLICE_TRACE_DIR` is set (`--trace-dir`).
- `live.py` — `LiveEvents`, the optional in-memory observer behind
  `engine.run(on_event=...)`: sanitized seed images, assistant text,
  provider-exposed reasoning summaries, tool calls and results with detached
  image bytes, usage, completion and errors. Observer failures never fail
  the run. With an observer ADK streams; partial events are shown but never
  counted twice toward usage, tool limits or the final answer. Encrypted
  reasoning stays in provider replay only. The toolbox emits `tool_start` /
  `tool_end` inside its serialization lock, with resolved `target_ids`, a
  unique execution id and, on `tool_end`, `views` (the pictures the call
  saved); these events never enter model context. `plain` copies JSON
  content without media bytes.
- `cost.py` — `estimate(spec, n_slices, locked)`: the pre-run usage-window
  estimate the worker's `linear.estimate` serves, from measured runs only
  (`_RATES`; a single run or a neighbouring setting gets a widened band).
  Medium, high and auto picture sizes and a Nonlinear-only run get no
  number, with the reason in `basis`. Imports no engine, so a host prices a
  spec without the agent framework.

## The tools

The tools are the toolbox `doors.tools.toolbox.build_tools(state, ctx, spec,
job=, image_model=, gates=, level=, max_view_edge=)` builds: the verbs
`ops.registry.enabled(spec)` names, each declared once in
`doors/declarations.py` (name, arguments, description) and each body
argument checking, one `ops` call and the wording. Every tool is wrapped:
`_serialized` (one lock: ADK runs a turn's calls concurrently, the state has
one writer), `_strict` (unknown or misplaced keys refused, never dropped),
`_saves_views` (every picture saved in the job folder with its layers, the
exact JPEG bytes the door sent) and `_clears_stale_deformations` (a
deformation whose `linear_key` no longer matches is cleared in the same undo
step; the reply says `deformation_cleared`). Tools return plain PIL pictures
under `TOOL_MEDIA_PARTS_KEY`; the ADK driver packages them as JPEG parts
(`doors.tools.media.packaged`), the MCP door as image blocks.

Conventions: sections are addressed by filename or corrected index;
ordinary writes answer with the rows they changed (`ToolBox.answered`) plus
`n_sections`, transform writes with their physical result; `status`,
`undo`, `redo` and `submit` return the whole table. What stays on the
`ToolBox`, the tool door, and is never undone: the look-before-commit gates
(`compared`, `reviewed`), the delivery bookkeeping (seen and pending
placement views, delivery ids, `tool_context`), `transform_history` and the
`ab_reference` of `adjust_transforms`. An undo, redo or reload that moves a
section's position counts as a write to it (`forget_looks`).

## Host controls

The ABBA dialog's controls, all plain `JobSpec` fields:

- **Per-task notes** (`position.notes`, `transform.notes`, `nonlinear.notes`):
  `prompt.task_notes` renders each verbatim under "<task> notes from the
  user:", only when that task is on and the note is non-empty. Global
  `facts` stay in the run facts.
- **`transform.max_parallel`** (1..4, default 4): below 4, `fit_affine` (an
  empty list counts every eligible section) and `adjust_transforms` refuse a
  call naming more sections (`TOO_MANY_SECTIONS`, `max_sections`) and the
  statement states the cap.
- **`agent_damage`** (default true) builds `mark_damaged`. A flag in
  `inputs["damaged"]` is the user's either way: clearing it is refused
  (`DAMAGE_SET_BY_USER`) and the statement lists those sections.
- **`inputs["locked"]`**: sections the user already aligned in-plane. Without
  a supplied transform each gets `job.host_transform()` (`kind` `"host"`,
  identity). `orient_slices`, `fit_affine` and `adjust_transforms` refuse them
  (`LOCKED`); `fit_affine`'s default list skips them; their positions still
  move. The host transform satisfies `MISSING_TRANSFORMS`, and locked sections
  are exempt from `DAMAGED_REQUIRES_MANUAL_TRANSFORM`.
- **`inputs["keep_warp"]`** / **`inputs["nonlinear_skip"]`**: sections kept
  out of the Nonlinear task (the user's own warp kept, or left out by the
  user). `fit_deformable` refuses them per section and `trace_borders`
  refuses the call (`KEEPS_HOST_WARP`, `NONLINEAR_SKIPPED`;
  `Job.nonlinear_refusal`).
- **`image_resolution`** (`low`|`medium`|`high`|`auto`, default `low`):
  `core.sizes.PICTURE_EDGES` gives each level an opening tile size and a
  later-picture size (low 256/512, medium 384/768, high 512/1024; `auto`
  opens at 256 and lets the agent choose `view.resolution` per call, 128 up
  to the driver model's largest image). Only what the agent is SHOWN
  changes: nothing is upsampled past its source, and fits, calibration and
  the image model's inputs do not depend on it. Below `auto` the sizes never
  appear in model-facing text.
- **`TransformSpec.interactive` / `.automatic`** (default true): a host may
  remove `adjust_transforms` or `fit_affine` (Elastix by default, the
  silhouette fit as an option); a host enabling the transform task enables
  at least one. Flip and the hemisphere cue are `transform.flip` /
  `transform.hemisphere_cue`; with flip off, `orient_slices` cannot mirror.

## Pictures and context

The OAuth lane resends the whole history on every call (the Codex backend
refuses stored responses). A cached input token costs ~0.13x an uncached
one, so a token that sits unchanged in the prefix is nearly free and a new
one, or one after a prefix edit, is full price. New images are most of a
run's cost. The rules that follow:

- **Images stay** (`plugins.WorkingSetImages`): every tool image is kept
  until 500 are live, then the oldest media-bearing calls are cut in ONE
  batch to 250 (`DEFAULT_MAX_IMAGES` / `DEFAULT_KEEP_IMAGES`); the cut only
  moves forward and never touches the opening. A cut result says its
  pictures were shown and have been dropped. An agent chooses later steps
  from earlier pictures (its stain, its channels), and a cut breaks the
  cache at the cut point, so a normal run never reaches it.
- **A tool's images ride inside its `function_call_output`** as labelled
  `input_image` parts (`openai_oauth._function_call_output`): a separate user
  message would open a new turn. The request sends `reasoning.context =
  all_turns`, and each turn's encrypted reasoning item is replayed ahead of
  it (`thought_signature`), as the Codex CLI does.
- **Several tool calls per model turn** (`parallel_tool_calls` on the OAuth
  lane): one call's history cost instead of one per tool; the toolbox
  serializes them. One `prompt_cache_key` per session.
- **Rows are compact** (`core.status.compact_rows`): null and empty fields
  are absent from every payload.
- **A placement picture is sent once per geometry.** `view_placement` and
  `set_positions` pictures become seen only when their media reached a model
  request (`ToolMediaDeliveryPlugin`); `set_positions` leaves out a picture
  of the same section, position, orientation and cutting angles already seen
  in a full-canvas atlas-bearing view.
- **Prefix caching is not content addressing.** An image replayed in its
  unchanged history prefix may be cached; a byte-identical copy appended
  later is new input. Never budget a re-sent picture as a cache hit.
- **The opening** is ABBA-style strips (`core/opening.py`): the sections in
  corrected order with the atlas at each one's current position beneath it,
  every tile labelled in its pixels, each strip at the model lane's largest
  image (`doors.tools.view_options.image_limit`) and within its patch budget.

## The Nonlinear task

The job line asks for a deformation per section on top of its linear
placement, with the section's stain (and, with an image model, the borders
it traces) as the evidence. `submit` refuses `MISSING_DEFORMATIONS` until
every section holds a deformation at its current `linear_key` or a
`keep_linear` record (`fit_deformable(slices, keep_linear="reason")`: no
fit, one undo step, cleared by a placement change like a fit; it lives on
the state so it is undoable, checkpointed, visible in `status` and exported).
`trace_borders` is a tool the agent may use or not; no section needs a trace.

- `trace_borders(id, prompt="", include=[], exclude=[])`: one image-model
  call on the supplied linear placement with the agent's per-section edit of
  the base prompt (a profile's own prompt replaces the base prompt for a
  script), showing the model only the chosen regions' borders
  (`registration_tool.shown_labels`; a traced `fit_deformable` adds the
  trace's regions to its own, `ops.deformable.traced_regions`). The image
  model is the door's binding (`build_tools(image_model=)`, else
  `providers.registry.resolve_image_model` from `nonlinear.provider` /
  `nonlinear.image_model`). It runs in the background
  (`core.nonlinear.registration_tool.start_correction`;
  `Job.settle_image_corrections` waits at submit and at session end) and
  returns no picture; the first reply is kept in `SliceState.image_correction`.
  No atlas search, replacement prompt, candidate choice or rejection.
- `grep_atlas(query, section="")` (`core/atlas_grep.py`): a text lookup of
  atlas regions by acronym, name or id, with ancestry and, for a positioned
  section, whether the region is in its plane: for choosing what a fit
  excludes.
- `fit_deformable(slices, include, exclude, start, fit_section, fit_atlas,
  [engine], stiffness, candidates, keep_linear, view)`: `ops.deformable`
  over `core/deformation.py` and `core/deformable/` (their guides hold the
  settings). 2–4 `candidates` preview and write nothing; one setting applies
  as one undo step, reusing an identical cached result. `engine` exists only
  when `nonlinear.engine` is `either`. Every fit runs on a fixed thread
  count and seed: the same inputs give the same warp. A trace still running
  is waited for (`TRACE_WAIT_S`). The description recommends `traced_borders`
  with ANTs at medium for a completed trace (dropped without an image model
  or with Elastix fixed). At most 4 sections and 8 fits per call.
- Provider `none` (`NonlinearSpec.uses_image_model` False) builds
  `grep_atlas` and `fit_deformable` but no `trace_borders`, describes
  `fit_deformable` without traced fit sections (`_STAIN_ONLY_DOC`), requires
  no trace and drops every image-model line from the statement. A door that
  cannot reach the job's image model builds the same toolbox
  (`image_model_connected=False`) and says so (`doors.statement.IMAGE_MODEL_OFF`).
- `view_placement` (built whenever position, transform or nonlinear is on)
  and `set_positions` draw the section resampled by its applied warp in
  every mode that draws it under its placement, unless `view.deformation`
  is `none`.

## Rules that are not negotiable here

**Lean harness.** Tools return data, no interpretation in any payload. The
job statement carries the job, the facts, one line per tool, the constraints
and a short Method per task that is on: place each section on its own
evidence and compare before writing; review the whole stack and re-check
both sides of a gap before reporting a break; inspect each fit or
adjustment against surviving internal anatomy and keep only changes that
improve it; let every region a section still has drive its deformable fit
and compare candidates. Grouping work is the model's choice. Still out:
rules of thumb, failure-mode warnings and region names (the same text runs
against every BrainGlobe atlas, species and plane).

**Gates for cheap models** (`--gates`, `PositionSpec.gated`, off by
default): data-only refusals. `set_positions` refuses a section not compared
since its last write (one compare is enough), and `submit` refuses until
`view_stack` has run after the last write. `--playbook`
(`PositionSpec.playbook`) puts a step-by-step positioning method in the
statement: a complete hypothesis from the opening, one confirmation sweep,
one write, targeted re-checks, a review. Both are tool-door only: a script,
the library or the agent CLI is never gated.

**Gates are constraints, not coaching.** A refusal states the numbers that
caused it; a failed `submit` neither writes nor ends the run. `submit`
refuses `MISSING_POSITIONS`, `ORDER_POSITION_MISMATCH` (positions must run
one way along the corrected order; the offending pairs named),
`STRICT_INTERVAL` (spacing within 10% of the interval and no breaks, with
`--strict-interval`), `INTERVAL_BREAKS_UNSUPPORTED` (a reported break must
exceed 1.5x the median written spacing), `MISSING_TRANSFORMS` (every section,
damaged included), `DAMAGED_REQUIRES_MANUAL_TRANSFORM` (a damaged section
needs a non-identity interactive transform; a note alone does not do; the
check reads the normalized matrix) and, with `nonlinear`, the trace and
deformation gates above. Gates run only for tasks that are on.

**One transform representation.** Elastix, silhouette, interactive: every
stored transform and every fit payload carries `physical` (the knobs
`adjust_transforms` takes — rotation, the two scales, the two shifts,
`shear` — about a pivot in canvas fractions) next to the six normalized
numbers and the calibration used, so a fit's numbers hand straight to the
manual controls and a tweak after a fit keeps its map. A `shear` left out
keeps the current one; an explicit value sets it.

**Corrections are data.** Order, flips, rotations, positions, cutting angles
and transforms are proposals on the state; the user's image files are never
modified.

**Order and position agree only at `submit`.** `reorder_slices` changes
nothing but `index_corrected` (filenames only: an index-addressed reorder
could hit the wrong section next call); the `ORDER_POSITION_MISMATCH` gate
holds the two together. `orient_slices` clears a section's transform: a
transform is defined after the orientation it was fitted under.

**Every write checkpoints and is undoable.** One tool call is one undo step,
so a batch undoes as one. The history (depth 50) is saved in the job folder,
so a resumed run can still undo the steps before it.

## Ceilings worth knowing

- `fit_affine`'s silhouette method measures against the atlas plane at the
  section's cutting angles. Its moments core has a 180-degree ambiguity,
  settled on silhouette IoU alone, and matches long axes, so on round or
  heavily excluded tissue it can turn a section 90 or 180 degrees (the reply
  notes it). The fit's own panel is what catches it. Damaged sections are
  refused unless regions are given; a region fit is one pass per call
  (iterating the tissue selection drifts).
- The Elastix method (the default) refines the CURRENT placement with an
  intensity affine, never a search from scratch: it does not turn sections,
  and moves them by fractions of a millimetre.
- `search_position` is a thin wrapper over `core.oblique.fit_oblique`:
  correct, not tuned, not benchmarked. Without a position it searches the
  whole valid range; the pose fit normalises size away, so the planes at
  the volume's ends (a sliver of brain) win unless the section's pixel size
  is known (`fit_oblique`'s `section_um_per_px` skips planes too small to
  hold the tissue).
- True physical scale is honest, not flattering: a specimen smaller than the
  Allen brain shows the atlas outlines ~10–15% outside the tissue at
  identity. That is anatomy, not a calibration bug.
- `adjust_transforms` resolves its pivot with its own `canvas_geometry`, so a
  call builds the atlas plane twice (an oblique plane resamples twice).

## The Claude host (MCP)

The MCP door gives Claude the same toolbox and the same statement, worded
for door `mcp` (`doors.statement.job_statement`), with the opening as
`show_stack` pages within the host's reply budget, an opening-read gate
armed by `start_job`, and no turn budget, nudges, debrief or working set:
the host owns the loop. `doors/CLAUDE.md` holds the details.

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
| `job/` | `Job` (undo, checkpoint, submit gates, image corrections, background work), the job folder | `job/CLAUDE.md` |
| `ops/` | one operation per verb, `ops/registry.py` (`VERBS`, `enabled(spec)`, `RETIRED`) | `ops/CLAUDE.md` |
| `doors/` | the declarations, the toolbox (`doors/tools/`), MCP, the agent CLI, the library | `doors/CLAUDE.md` |
| `agent/` | this driver: engine, session, job statement, plugins, model resolver, trace, cost | here |

## One environment, not a pipeline

`langslice linear run FOLDER` is ONE agent environment over a stack of
sections: one `StackState`, one toolbox, one job statement, one ADK session
that ends at `submit` or the turn budget. A single section is a stack of one.
There is no node graph, no per-step agent, no single-section agent and no
sub-session: looking at the whole stack once yields the information for
every task, and splitting it re-pays for that reading.

The toolbox: shared tools for the everyday operations (`look` with its four
modes, `zoom`, the two channel tools, `grep_atlas` / `grep_atlas_view`,
`status`, the job-folder tools, `note` / `undo` / `redo` / `submit`) and one
tool per registration method (`position_sections`,
`interactive_transform`, `mark_damage`, `elastix_affine`, `ants_syn`,
`trace_borders`). Every change tool returns its own picture after the write
(the computer-use pattern): the positioning picture for
`position_sections`, the section under its new registration with the atlas
borders on it for `interactive_transform`, `elastix_affine` and `ants_syn`
(zoomed to `restrict_to`), the shaded damage for `mark_damage`. Nothing
previews; everything is undoable, and undo is how two fits are compared.
`interactive_transform` batches up to four independent sections in one
call and one undo step; a dependent refinement waits for the first
picture. Every picture is saved in the job folder with a number, so `zoom`
and the job-folder tools reach it later.

Tasks (`JobSpec.tasks`) are switched on one by one; a task that is OFF builds
no tools and takes its answer from `spec.inputs`. Users see Positioning
(`position`), Linear (`transform`) and Nonlinear (`nonlinear`) (the
user-facing task names). Which verbs a run has is
`ops.registry.enabled(spec)`: `position_sections` with Positioning (or the
cutting angles left to the agent, `transform.angles`; the stack's order
follows the positions, so there is no ordering tool);
`interactive_transform` and `elastix_affine` with `transform` (a mirror is
the sign of the in-plane affine, so flipping is Linear, never Positioning),
each behind its host switch; `ants_syn` and `trace_borders` with
`nonlinear` (`trace_borders` only with an image model); `mark_damage`
behind `agent_damage`; everything else in every run. A retired tool's name
answers with its replacement (`ops.registry.RETIRED`, `plugins.RetiredToolsPlugin`).

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
  `DEBRIEF_PROMPT` when `spec.debrief`, its answer on `state.debrief`;
  `background_message`: when the model ends its turn while background work
  runs, the session waits for the first piece to finish
  (`BackgroundWork.wait_any`) and the finished work's notices and pictures
  are the next message), and
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
  their grace call, the debrief; *background*, called in a thread when a
  turn ends without a submit, returns the next message in place of the
  nudge: the notices of background work that finished) and `TokenTally`
  (every call's usage printed and traced; `paid` is uncached input plus
  cached input at `CACHED_TOKEN_WEIGHT` 0.13, an input-only proxy).
  `JobSpec.max_quota_percent` (25) ends the session when this run's share
  of the OAuth usage window reaches it; `JobSpec.max_input_tokens` (default
  None) when one request's reported input, cached included, passes it.
  Either stop gets ONE more call, asking for `submit` (`_BUDGET_GRACE`,
  `_CONTEXT_GRACE`); a stop never breaks out of a turn, so the pending tool
  call is answered first and the history stays well formed.
- `prompt.py` — `build_job_statement`: the job, the run facts (`run_facts`:
  the range, the protocol, the cutting angles, the sections still at their
  starting positions (not yet placed), the damaged sections and the user's
  damage notes, the locked sections, that the order follows the positions;
  `channel_facts`: the raw channels, the atlas layers here, and where the
  caller sizes pictures, `look`'s `resolution` range), the tools BY NAME
  ONLY (each tool's own description, `doors/declarations.py`, is its only
  description), the hard constraints (with the damaged-section guidance:
  a section with marked regions still needs a transform made for its
  surviving anatomy, by hand or by `elastix_affine`, which leaves the
  marked regions out; `ants_fact` when ANTs is missing), the per-task user
  notes (`task_notes`), the short "Method:" section. `display_facts` reads
  the run's raw channels and atlas channels. Every door's statement is
  assembled in `doors/statement.py`.
- `plugins.py` — the ADK plugins every session runs with:
  `WorkingSetImages` (in ADK's `ContextFilterPlugin`, below),
  `RetiredToolsPlugin` (a call of a
  retired tool's name, which ADK hands the plugins as a placeholder tool
  before its "tool not found" answer, is answered with
  `ops.registry.retired_payload`), `StrictArgumentsPlugin` (ADK drops
  unknown top-level arguments before a tool runs; this refuses them first,
  with `doors.tools.arguments.argument_refusal`), `ModelCallPacingPlugin`
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
(`doors/tools/looking.py`, `changing.py`, `fitting.py`) argument checking,
one `ops` call and the wording. Every tool is wrapped: `_serialized` (one
lock: ADK runs a turn's calls concurrently, the state has one writer, and
the host events), `_announces_work` (the notices of background work that
finished since the last reply, first, under `background`, its pictures
after the reply's own), `_saves_views` (every picture saved in the job
folder with its layers and a number, the exact JPEG bytes the door sent;
the reply's `pictures` lists the numbers in attachment order),
`_clears_stale_deformations` (a deformation whose `linear_key` no longer
matches is cleared in the same undo step; the reply says
`deformation_cleared`) and `_strict` (unknown or misplaced keys refused,
never dropped). Tools return plain PIL pictures under
`TOOL_MEDIA_PARTS_KEY`; the ADK driver packages them as JPEG parts
(`doors.tools.media.packaged`), the MCP door as image blocks.

Conventions: sections are addressed by filename or index (the index is the
stack order, which follows the positions); ordinary writes answer with the
rows they changed (`Door.answered`) plus `n_sections`; `status`, `undo`,
`redo` and `submit` return the whole table, and `status` also the channel
display settings, the preprocessed channel's recipe, the background work
still running and what this run lets the agent change. What stays on the
`ToolBox`, the tool door, and is never undone: the look-before-commit gates
(`compared`, `reviewed`). An undo, redo or reload that moves a section's
position counts as a write to it (`Door.forget_looks`).

## Host controls

The ABBA dialog's controls, all plain `JobSpec` fields:

- **Per-task notes** (`position.notes`, `transform.notes`, `nonlinear.notes`):
  `prompt.task_notes` renders each verbatim under "<task> notes from the
  user:", only when that task is on and the note is non-empty. Global
  `facts` stay in the run facts.
- **`transform.max_parallel`** (1..4, default 4): below 4,
  `interactive_transform` and `elastix_affine` (an empty list counts every
  eligible section) refuse a call naming more sections (`Job.over_cap`)
  and the statement states the cap.
- **`agent_damage`** (default true) builds `mark_damage`. A host's
  `inputs["damaged"]` note is the section's `damage_note` (shown in status
  and the statement); it does not make the section damaged, and it stays
  first in the note whatever the agent marks.
- **`inputs["locked"]`**: sections the user already aligned in-plane. Without
  a supplied transform each gets `job.host_transform()` (`kind` `"host"`,
  identity). `interactive_transform` and `elastix_affine` refuse them
  (`LOCKED`); `elastix_affine`'s default list skips them; their positions
  still move. The host transform satisfies `MISSING_TRANSFORMS`.
- **`inputs["keep_warp"]`** / **`inputs["nonlinear_skip"]`**: sections kept
  out of the Nonlinear task (the user's own warp kept, or left out by the
  user). `ants_syn` refuses them per section and `trace_borders` refuses the
  call (`KEEPS_HOST_WARP`, `NONLINEAR_SKIPPED`; `Job.nonlinear_refusal`).
- **`nonlinear.require_deformation`**: every section needs a deformation;
  `submit` takes no `left_linear` (not declared, refused when sent).
- **`force_view`**: the change tools always return their picture (their
  `view` argument is not declared).
- **`image_resolution`** (`low`|`medium`|`high`|`auto`, default `low`):
  `core.sizes.PICTURE_EDGES` gives each level an opening tile size and a
  later-picture size (low 256/512, medium 384/768, high 512/1024; `auto`
  opens at 256 and declares `look`'s `resolution`, 128 up to the driver
  model's largest image). Only what the agent is SHOWN changes: nothing is
  upsampled past its source, and fits, calibration and the image model's
  inputs do not depend on it. Below `auto` the sizes never appear in
  model-facing text.
- **`TransformSpec.interactive` / `.automatic`** (default true): a host may
  remove `interactive_transform` or `elastix_affine`; a host enabling the
  transform task enables at least one. Flip and the hemisphere cue are
  `transform.flip` / `transform.hemisphere_cue`; with flip off,
  `interactive_transform` cannot mirror.
- **`agent_preprocessing`**: whether the Fiji connector exports every
  channel; the channel tools exist in every run.

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
  cache at the cut point, so a normal run never reaches it. A dropped
  picture can be drawn again: `zoom` takes its number.
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
- **Prefix caching is not content addressing.** An image replayed in its
  unchanged history prefix may be cached; a byte-identical copy appended
  later is new input. Never budget a re-sent picture as a cache hit.
- **The opening** is ABBA-style strips (`core/opening.py`): the sections in
  corrected order with the atlas at each one's current position beneath it
  (labelled `start` while that is the job's starting position), every tile
  labelled in its pixels, each strip at the model lane's largest image
  (`doors.tools.view_options.image_limit`) and within its patch budget.

## The Nonlinear task

The job line asks for a deformation per section on top of its linear
placement, with the section's stain (and, with an image model, the borders
it traces) as the evidence. `submit` refuses `MISSING_DEFORMATIONS` until
every section holds a deformation at its current `linear_key`, or is named
in its `left_linear` with the reason its linear placement stands (unless
the host requires a deformation on every section).

- `ants_syn(sections, restrict_to=, atlas_image=, stiffness=, view=)`:
  ANTs SyN on the preprocessed channel against the template (or Nissl), on
  top of each section's current registration (each fit builds on the one
  before; undo is how to start over), the marked damage regions left out;
  at most 4 sections per call; `ANTS_MISSING` without antspyx.
- `trace_borders(section, prompt="", restrict_to=[])`: LangSlice's own
  method, packaged: the image model traces the atlas borders onto the
  section (the agent's per-section edit of the base prompt, which the
  description carries with the border rules), then ANTs fits the traced
  borders and the deformation is applied as its own undo step, as
  background work (`job.background`): the call returns at once with the
  work's id; the notice comes at the head of a later reply, or as the next
  message when the turn ends; `submit` waits. The first answer at a
  placement and region choice is saved and reused; a call whose fit has
  already landed fits nothing again. The image model is the door's binding
  (`build_tools(image_model=)`, else `providers.registry.resolve_image_model`
  from `nonlinear.provider` / `nonlinear.image_model`).
- `grep_atlas(query, section="")` and `grep_atlas_view(regions,
  positions_mm)` look regions up and show their borders on the atlas, for
  choosing `restrict_to` and `mark_damage` regions.
- Provider `none` (`NonlinearSpec.uses_image_model` False) builds
  `ants_syn` but no `trace_borders`. A door that cannot reach the job's
  image model builds the same toolbox (`image_model_connected=False`) and
  says so (`doors.statement.IMAGE_MODEL_OFF`).
- `look` in mode overlay draws each section with its applied deformation
  unless `warp` is `none`.

## Rules that are not negotiable here

**Lean harness.** Tools return data, no interpretation in any payload. The
job statement carries the job, the facts, the tools by name, the
constraints and a short Method per task that is on: place each section on
its own evidence and compare before writing; review the whole stack and
re-check both sides of a gap before reporting a break; inspect each fit or
adjustment against surviving internal anatomy and keep only changes that
improve it; mark a damaged section's lost regions before fitting it; fit
whole sections first, then regions. Grouping work is the model's choice.
Still out: rules of thumb, failure-mode warnings and region names in the
statement (the same text runs against every BrainGlobe atlas, species and
plane). Each tool's description is the only place it is described.

**Gates for cheap models** (`--gates`, `PositionSpec.gated`, off by
default): data-only refusals. `position_sections` refuses a section not
looked at (`look` in mode overlay or positioning) since its last write
(one look is enough), and `submit` refuses until a `look` in mode
positioning has covered every section after the last write.
`--playbook` (`PositionSpec.playbook`) puts a step-by-step positioning
method in the statement: a complete hypothesis from the opening, one
confirmation sweep, one write, targeted re-checks, a review. Both are
tool-door only: a script, the library or the agent CLI is never gated.

**Gates are constraints, not coaching.** A refusal states the numbers that
caused it; a failed `submit` neither writes nor ends the run. `submit`
waits for the background work, then refuses `MISSING_POSITIONS` (a section
still at the starting position the job gave it counts as missing),
`STRICT_INTERVAL` (spacing within 10% of the interval and no breaks, with
`--strict-interval`), `INTERVAL_BREAKS_UNSUPPORTED` (a reported break must
exceed 1.5x the median written spacing), `MISSING_TRANSFORMS` (every
section, damaged included) and, with `nonlinear`, `MISSING_DEFORMATIONS`.
Gates run only for tasks that are on.

**One transform representation.** Elastix, interactive, imported: every
stored transform and every fit payload carries `physical` (the knobs
`interactive_transform` takes — rotation, the two scales, the two shifts,
`shear` — about a pivot in canvas fractions) next to the six normalized
numbers and the calibration used, so a fit's numbers hand straight to the
manual controls and a tweak after a fit keeps its map. A value left out
keeps the current one.

**Corrections are data.** Positions, flips, rotations, cutting angles and
transforms are proposals on the state; the user's image files are never
modified.

**The order follows the positions.** `position_sections` numbers the stack
by position after every call (`ops.positions.order_by_position`), so there
is no separate ordering and no order-position gate; a changed orientation
keeps the transform's knob values, rebuilt on the new orientation.

**Every write checkpoints and is undoable.** One tool call is one undo step,
so a batch undoes as one; a landed piece of background work is a step of
its own. The history (depth 50) is saved in the job folder, so a resumed
run can still undo the steps before it.

## Ceilings worth knowing

- `elastix_affine` refines the CURRENT placement with an intensity affine,
  never a search from scratch: it does not turn sections, and moves them by
  fractions of a millimetre.
- True physical scale is honest, not flattering: a specimen smaller than the
  Allen brain shows the atlas outlines ~10–15% outside the tissue at
  identity. That is anatomy, not a calibration bug.
- Region fits (`restrict_to`) are fits of those regions only; a later fit
  without `restrict_to` builds on them.

## The Claude host (MCP)

The MCP door gives Claude the same toolbox and the same statement, worded
for door `mcp` (`doors.statement.job_statement`), with the opening as
`show_stack` pages within the host's reply budget, an opening-read gate
armed by `start_job`, and no turn budget, nudges, debrief or working set:
the host owns the loop. `doors/CLAUDE.md` holds the details.

# LangSlice `linear/` — order, position, transform

Package guide for `src/langslice/linear/`. The repo-level `CLAUDE.md` holds the
project-wide rules; this file holds what is specific to this package.
`AGENTS.md` here is a verbatim copy — edit one, mirror to the other.

The design this implements is `docs/linear_design.md`. Read it before changing
shapes; this file is the map of the code, not a second spec.

## One environment, not a pipeline

`langslice linear run FOLDER` is ONE agent environment over a stack of
sections: one `StackState`, one toolbox, one job statement, one ADK session
that ends at `submit` or the turn budget. A single section is a stack of one.
There is no node graph, no per-step agent, and no single-slice agent — looking
at the whole stack once yields the information for every task, and splitting
that into separate sessions threw the shared reading away and re-paid for it.

There are no sub-sessions left. The per-section interactive alignment
(`adjust_transforms`) moved into the main toolbox on 2026-09-06:
GPT-6 Astra moved every parameter at once and finished a section in four
previews, so the fan-out bought nothing and cost the shared reading. Later the
same day preview and set became ONE tool, the computer-use pattern: every
`adjust_transforms` call writes the transform and returns the picture, so the
agent always sees what it did and never spends a turn on a separate look or a
separate commit. `adjust_transforms` batches up to four independent sections
inside this same trajectory and one undo step; a dependent refinement waits
for the first picture. The same rule now holds for every
write outside the transform (audit, 2026-09-06): `orient_slices` returns the
re-oriented sections, `set_positions` returns placements not already seen at
the same geometry, `fit_affine` always writes (no `apply` flag), and
`distribute_spacing` is gone — the agent can space a list of sections in its
head and send one `set_positions`. Nothing previews; everything is undoable.

**GPT-6 Astra's positioning debrief, built 2026-09-06** (full M04, 38
sections, reorder+position: order 38/38, positions median 0.30 mm, bias
-0.17 — a 0.2 mm ladder from a correct anchor; damage 7/7 +1 FP; 13 calls).
Nash kept five of its ten asks:
- images "not delivered": REAL, and not the transport. `adk/plugins.py`
  `trim_stale_tool_images` kept only the newest 24 tool images (sized for
  8-image sweeps); with 16-image `set_positions` batches every call but the
  newest two lost its pixels on every later turn, which is exactly Astra's
  "early results said attached, later ones showed images" — in both runs.
  Budget raised to 128 on 2026-09-06 — then run 4 (2026-09-07) carried 170
  images / 93k input tokens on its last call and 1.3M over the run, and a
  Plus window ended it. Now `WorkingSetImages` (see "Context and tokens"
  below), a dropped call's result says "dropped from context" instead of
  "attached", a tool's images ride INSIDE its `function_call_output` as
  labelled `input_image` parts (`openai_oauth._function_call_output`,
  2026-09-09 — the separate user message they used to follow in opened a
  new turn and threw the replayed reasoning away), and the
  picture-returning writes report `render_failed`.
- `view_placement` (was `compare_placement`, renamed 2026-10-01): the
  section on the physical canvas against the atlas at any positions, every
  `VIEW_MODES` view, zoom, opacity — the "linked viewer" and the "placement
  preview / AP stepper" in one read-only tool. Since 2026-10-01 it draws the
  section under its STORED transform (identity without one), so it shows the
  full current placement. Same renderer as everything.
- `view_stack`: the strip ordered by written position with position and
  spacing in the labels, plus `render.spacing_plot` (PIL, no matplotlib).
- `reorder_slices` took corrected indices for one run; Astra then pointed
  out (run 2) that a reorder changes the indices, so an index-addressed
  reorder can hit the wrong section next call. Filenames only again.
Run 2's new asks, built: `view_stack` pastes the atlas at each placed
section's position beneath it in the SAME image (one picture per section, not
two — the image budget counts); `view_placement` takes a batch of
`{id, positions_mm}` entries, ≤4 pairs per call. Since 2026-09-11,
positioning `side_by_side` returns separate seed-style reference images:
one unchanged section per distinct id, one atlas per pair (up to 8 images),
mapped by zero-based `image_indexes` in each compared row. These are
independently tissue-framed, not a common physical canvas; zoom is refused
and outlines/opacity do not apply. Other modes still return one physical
canvas per pair. Interactive-transform `side_by_side` is unchanged.
Seed, comparison and atlas-fetch paths share encoded reference caches:
section captions retain their first display index/flags across reorder,
filenames are the stable identity, and orientation/preprocessing/size changes
get distinct entries. Atlas entries include exact position, plane and angles.
Reusing bytes avoids recomposition, not new-image input charges. An image
replayed in its original unchanged history prefix is eligible for prompt-cache
reuse; another copy appended after new conversation content is new input,
even when byte-identical. Do not budget a cached-input discount for that copy
merely because the seed contains it. Later requests can reuse its new prefix
if history stays unchanged and the backend retains a matching cache entry.
Rejected: labelled anatomy/landmarks, confidence and verification states,
damage masks, an anatomy-based gap review, a validity-vs-verification audit.

## Files

- `spec.py` — `JobSpec` (+ `ReorderSpec`/`PositionSpec`/`TransformSpec`/
  `NonlinearSpec`). Every checkbox a host shows maps to a field here; nothing
  else is user-facing. Users see Positioning (`reorder` + `position`), Linear
  (`transform`) and Nonlinear (`docs/interface_design.md`). Flip and
  hemisphere cue are `transform.flip` / `transform.hemisphere_cue` since
  2026-09-29 (a mirror is the sign of the in-plane affine): `orient_slices`
  is built with the transform task, a positioning-only run never mentions
  mirroring, and `reorder.flip` / `reorder.hemisphere_cue` survive only as
  aliases `JobSpec.__post_init__` folds in (and mirrors back for old readers;
  `to_dict` writes them under `transform` only). A host transform in
  `inputs.transforms` may be mirrored (negative determinant) and is kept as
  supplied.
  `tasks` is the master switch: a task that is OFF builds no tools and takes
  its answer from `spec.inputs` instead. `inputs["pixel_size_um"]` is the
  calibration override, and `reasoning` (`--reasoning`) rides through
  `session.build_agent` onto any resolved model exposing `reasoning_effort`.
  Host dialog fields (2026-09-28, see "Host controls" below):
  `image_resolution`, `agent_damage`, per-task `notes`,
  `transform.max_parallel`, and `inputs["damaged"]` / `inputs["locked"]`.
- `state.py` — `StackState`/`SliceState`. The checkpoint, the result and the
  thing every tool writes, one JSON shape for all three. `restore()` refills
  the same object in place, because tools close over one state. Every stored
  transform carries `physical` — the five knobs plus `shear`, about a pivot in
  canvas fractions — next to the six normalized numbers, whatever made it.
  There is no `confidence`: nothing downstream read it (Nash, 2026-09-06), and
  the reasoning lives in `adjust_transforms`'s note.
- `checkpoint.py` — atomic JSON write to `<folder>/linear_state.json`.
- `discovery.py` — natural-sorted image discovery.
- `render.py` — `render_slice` (ROTATE first, then FLIP, then the display-only
  `--preprocess auto` enhancement), `stack_image_parts`, the status rows and
  their text form, `caption`, the JPEG `types.Part` encoder, and the PHYSICAL
  overlay: `canvas_geometry` places an atlas section on a section's frame at
  true scale (`atlas um/px / canvas um/px`, anatomy centred, canvas grown to
  hold it — never fit-to-canvas, which is not a calibration), and
  `physical_views` draws the alignment picture on it (family outlines
  from `atlas.render.family_outlines` as yellow 0.5 px lines by default, drawn
  at OUTPUT size. Agent tools expose `border_color` (named color/#RRGGBB) and
  `border_thickness` (0.25–8 output pixels, including fractional widths) alongside template opacity;
  these are display-only and do not change transforms or IoU. The native ABBA
  viewer keeps its own display settings. Includes a 1 mm scale bar and two-line
  caption). It takes either the five physical knobs or a ready 2x3 in the
  section's frame, so the interactive loop and `fit_affine` draw the same
  picture, and returns `(images, silhouette_iou)`. Its view controls:
  `mode` (`overlay`, `side_by_side` — two physical images, `checkerboard`,
  `outlines` — atlas lines plus the section's own silhouette in neutral grey
  on black — and the line-free `section` / `template`), `zoom` ([x0, y0, x1, y1] fractions of the CANVAS, cropped BEFORE
  the resize so it magnifies, with the bar redrawn for the new µm/px),
  `atlas_opacity` (0..1; `template_opacity` until 2026-10-01, the
  `show_template` bool before that) and `outlines`
  (`OUTLINE_LAYERS`: `all` family boundaries, `outer` — the root contour from
  `atlas.render.outer_outline` — or `none`; the caption names the layer when
  it is not `all`).
  `physical_overlay` is the one-image `overlay` wrapper `fit_affine` uses.
  `pivot` (canvas px, `pivot_on_canvas` resolves "canvas"/"tissue"/[fx, fy]
  onto it) and `markers` (the landmark pairs, a cross per section point and a
  ring per atlas point in their own two colors) ride through both.
  Region acronyms and damage masking were asked for and deliberately not
  built. `canvas_um_per_px` follows the file's pixel size down through
  the render's own downsample (`render_scale`, same key as `render_cache`);
  `estimate_um_per_px` is the fallback guess from tissue width. Renders are
  cached on the context and shared: read them, never mutate them. `caption`
  burns a label into a COPY — every image a tool returns gets one, because
  tool images reach the model as bare attachments; never caption an image a
  fit measures.
- `atlas_fetch.py` — `atlas_section` (the one atlas renderer: flat at 0/0
  cutting angles, `oblique.sample_oblique_plane` otherwise) and the
  `view_atlas` tool (was `fetch_atlas`), closed over the run context. Sections
  and atlas sections are framed the same way so apparent scale is not a cue.
- `display.py` — the shared display options (2026-10-01): `mode`, `zoom`,
  `section_image`, `atlas_image` (`ara`/`borders`/`nissl`, nissl only with
  ABBA's cached atlas, `EngineContext.abba_atlas`), `atlas_opacity`, `regions`
  (descendants included, `render.region_polys`, context outlines at
  `REGION_CONTEXT_ALPHA`), `outlines`, `border_color`, `border_thickness`.
  `parse_display` validates them once into a frozen `DisplayOptions`; every
  picture tool takes the same nine names (`with_display_doc` appends the
  shared docstring), `adjust_transforms` per entry. It never writes state, so
  a call's options never change a default. `framed_section` / `framed_atlas`
  draw the tissue-framed pictures (default options = the old pixels exactly);
  the toolbox's `draw_canvas` draws every physical picture.
- `appearance.py` — the section's appearance per target (2026-10-01): `view`
  (what the agent is shown) and `fit` (`fit_image`, what a deformable fit
  reads), each a stack setting plus per-section overrides on
  `StackState.appearance` (undone and checkpointed). `None` is the DEFAULT
  appearance (today's `preprocess` auto, or the host's blend when
  `spec.host_preprocessing` is set); anything else is drawn by
  `render_slice(look=...)` from the raw channels
  (`EngineContext.section_channels`) over the same frame and size. The
  `preprocess` tool (gated by `spec.agent_preprocessing`) is the only writer.
  Computation (silhouette fit, calibration, tissue pivot, `search_position`,
  the image model's input) always reads the default.
- `toolbox.py` — `build_tools(state, ctx, spec)`: every tool, gated by the
  spec, plus the submit gates and the undo/redo snapshot stack. The interactive
  transform lives here: `_Staged` (one section, its calibrated canvas and the
  resolved pivot), `adjust_transforms` (the write AND the look, any positioned
  section; `mode="ab"` draws the new parameters beside what the section
  carried before the call, a silhouette fit included; the same numbers again
  re-draw without an undo step). One to four distinct sections share one undo
  step; each returns one image, or two for `ab`/`side_by_side`, mapped by
  per-result `image_indexes`. Paired landmark tools are removed; interactive
  alignment exposes direct affine adjustments only.
  `transform_history` on the ToolBox is per section and lasts the whole run,
  but is not repeated in tool replies. `commit(*touched)` is what ordinary writes answer with: the status rows of
  the sections it touched plus `n_sections`, never the whole table —
  transform writes return their canonical physical result instead; `status`,
  `undo`, `redo` and `submit` are what return all the rows.
- `transform.py` — the silhouette fit and the arithmetic the interactive tools
  run on, both in PHYSICAL space. `fit_silhouette` matches the whole tissue
  outline against the whole atlas outline (`affine.silhouette_affine`) and
  reports six normalized numbers on the section frame plus `physical` about
  the canvas centre (`_conjugate` moves a 2x3 between the two frames); damaged
  sections are refused before this runs (`toolbox.fit_affine`). `calibrate`
  answers the canvas's
  `--pixel-size-um` (`"host"`), else the file's TIFF/OME tags (`"file"`),
  else the tissue-width guess (`"estimated"`) — it never crashes and never
  silently pretends. The knobs are ABBA's (rotation about the pivot, per-axis
  scales, `translate_x_mm`/`translate_y_mm`), and the recorded transform
  carries them under `"physical"` (with the pivot as canvas fractions) next to
  the six normalized numbers plus the `"calibration"` used.
  `physical_decomposition` is `decompose_affine` minus its translation
  fractions; `similarity_fit` (Umeyama, exact on two points), `affine_fit`
  (least squares) and `physical_params` (a canvas 2x3 back into the five knobs
  about a pivot) are shared affine geometry helpers.
- `prompt.py` — `build_job_statement`: job, run facts, ONE factual line per
  tool that exists, hard constraints. Nothing else.
- `session.py` — the ADK agent builder, the plugins, the loop, and
  `TokenTally`: every call's usage is printed and traced, and
  `JobSpec.max_quota_percent` (25) ends the session when this run's share
  of the provider's usage window reaches it. Optional `max_input_tokens`
  (default `None`) checks one request's reported input, including cached
  tokens, after its response. It is not a cumulative spending guard or a
  preflight size guarantee. Logs separate cumulative input from peak request
  input; 1.3M processed over a run does not mean a 1.3M-token context.
- `engine.py` — `EngineContext`, `ingest`, `apply_host_inputs`, `run_session`,
  `emit_results`, and `run(spec)`. No post pass: the session is the whole run.
- `live.py` — optional in-memory observer for host activity windows.
  `engine.run(on_event=...)` streams sanitized seed images, assistant text,
  provider-exposed reasoning summaries, final tool calls/results with detached
  image bytes, usage, completion and errors. Observer failures cannot fail the
  run. With an observer, ADK uses SSE; partial events are displayed but never
  counted twice toward usage, tool limits or final response collection.
  Encrypted reasoning/signatures stay in provider replay only. The disk trace
  remains independently opt-in.
  Actual execution emits `tool_start`/`tool_end` inside the toolbox serialization
  lock, with resolved filename `target_ids`, sanitized arguments/results and a
  unique execution ID. These host-only events follow real execution order and
  never enter model context; `tool_end` follows checkpoint/mirror writes.
- `deepslice.py`, `trace.py` — the DeepSlice seam (reports `UNAVAILABLE`) and
  the full-content JSONL session trace.
- `cost.py` — `estimate(spec, n_slices, locked)`: the pre-run usage-window
  estimate the worker's `linear.estimate` serves, from measured runs only.
  Refuses medium/high image resolution (nothing measured); single-run or
  unmeasured settings get a widened band. Imports no engine: `linear/__init__`
  loads `run` lazily so hosts can price a spec without the agent framework.

## Host controls (2026-09-28)

The ABBA dialog's controls, all plain `JobSpec` fields (not CLI flags yet):

- **Per-task notes** (`position.notes`, `transform.notes`, `nonlinear.notes`):
  `prompt.task_notes` renders each, verbatim one `- ` line per line, right
  after the job line under "<task> notes from the user:", only when that task
  is on and the note is non-empty. Global `facts` stay in the run facts.
- **`transform.max_parallel`** (1..4, default 4). Below 4, `fit_affine` (an
  empty list counts every eligible section) and `adjust_transforms` refuse a
  call naming more sections with `TOO_MANY_SECTIONS` + `max_sections`, and the
  job statement states the cap. At 4 nothing changes: `adjust_transforms`
  keeps its own four, `fit_affine` any number.
- **`agent_damage`** (default true) builds `mark_damaged`. Whatever it is, a
  flag in `inputs["damaged"]` is the user's: clearing it is refused per entry
  (`DAMAGE_SET_BY_USER`) and the job statement lists those sections.
- **`inputs["locked"]`**: sections the user already aligned in-plane.
  `apply_host_inputs` gives each without a supplied transform
  `toolbox.host_transform()` (`kind` `"host"`, identity params, identity
  `physical`). `orient_slices`, `fit_affine` and `adjust_transforms` refuse
  them per section (`LOCKED`); `fit_affine`'s default target list skips them;
  their positions still move. The host transform satisfies
  `MISSING_TRANSFORMS` and locked sections are exempt from
  `DAMAGED_REQUIRES_MANUAL_TRANSFORM`. The worker never emits orientation or
  transform rows for them.
- **`image_resolution`** (`low`|`medium`|`high`, default `low`):
  `render.IMAGE_RESOLUTION_SCALE` 1.0/1.5/2.0 multiplies what the agent is
  SHOWN, never what is computed. `atlas.render.model_long_edge(scale=)`
  scales both the cap and the atlas-resolution target (still never
  upsampling a section); the framed `render_slice` path (seed, `view_slices`,
  `orient_slices`, separate references, placement pictures, contact-sheet
  thumbnails) passes it; `atlas_fetch.atlas_sized(scale=)` resamples atlas
  images by the same multiple so a section and its atlas keep equal pixels
  per millimetre (no finer atlas detail exists); physical views
  (`view_placement`, `adjust_transforms`, the `fit_affine` panel) are drawn
  by `render.shown_section` from a larger unframed render with the matrix
  (`rescale_section_matrix`) and pivot carried onto it. Unchanged:
  `PREVIEW_LONG_EDGE` working renders, `calibrate`, the silhouette fit, the six
  stored numbers and every payload number, `search_position`, the spacing plot,
  caption font and the image model's inputs. At `low` every path returns the
  same objects as before (an end-to-end hash of a toolbox session matched
  HEAD on 2026-09-28). The multiples never appear in model-facing text.

## Tool consolidation (2026-09-15)

The default full-task toolbox has 15 tools (10 for interactive-only transform
refinement, `orient_slices` included since 2026-09-29; one fewer each with
`agent_damage` off, one more with `agent_preprocessing` on). Renamed
2026-10-01 (Nash: names do not change performance): `fetch_atlas` ->
`view_atlas`, `compare_placement` -> `view_placement`, `fit_position` ->
`search_position`; `template_opacity` -> `atlas_opacity`. `adjust_transforms` handles both one section and batches; the
single-section implementation is private. `mark_damaged` accepts per-entry
`damaged=False` to clear flags. `validate` and `unmark_damaged` are removed;
failed `submit` reports unmet requirements without changing or ending the run.
`reorder_slices(new_order, after="start")` moves a list of filenames as one
block after an anchor or to the start; a full list sets the whole order. Unlisted
sections retain relative order. `move_slice` and the three paired-landmark tools
are removed. Direct adjustments replace the complete transform, including a
historical spline or shear.

## Linear scope and saved spline compatibility (2026-09-15)

The linear agent handles order, position and affine alignment. Paired landmark
viewing, editing and warping were removed from the toolbox and prompt after the
full-stack comparison did not establish a nonlinear quality gain. The dedicated
linear landmark module and editable-point state are removed. Default full-task
and interactive-only tool counts are 15 and 10 (`orient_slices` joined the
transform task on 2026-09-29).

Local anatomical deformation remains the responsibility of the nonlinear workflow.
The optional `nonlinear` task now adds `trace_borders(id, prompt="")` through
the top-level `registration_tool` bridge. It uses the supplied linear placement
and the agent's per-slice edited copy of the base correction prompt.
The call runs in the background (`registration_tool.start_correction` prepares
it; `ToolBox.settle_image_corrections` waits at submit and at session end) and
returns no images. The first image reply is retained in `SliceState.image_correction`. No atlas search,
replacement prompt, candidate selection or anatomical rejection is exposed.
The same task adds `grep_atlas(query, section="")` (`linear/atlas_grep.py`): a text-only
lookup of atlas regions by acronym, name substring or id, with ancestry, descendant
count and, for a positioned section, whether the region is in the atlas plane at its
placement. It is for choosing regions a later deformable fit should exclude.
It does not fit a deformation or modify `transform`. Default task/tool counts stay
unchanged; `DEFAULT_TASKS` is separate from `ALL_TASKS`. Hosts may supply calibrated
`inputs.transforms` or resume saved linear transforms. Submit checks correction
completion and current geometry when the new task is on. Exact artifacts and first
reply caching survive checkpoint undo; no note-only regeneration occurs. A failed
transport that returned no image may be retried.
See `docs/nonlinear_image_tool.md` for the contract and prompt sentence audit.

Historical `transform.spline` checkpoints remain supported: `landmark_warp.py`
evaluates the exact stored Elastix or legacy TPS mapping, and rendering, undo,
resume and calibrated native ABBA export retain that complete transform. The
affine metadata is not applied twice. Later affine adjustments/fits replace the
spline. Obsolete editable-point fields are ignored by `StackState.from_dict`.
Shared spline code is compatibility infrastructure, not an agent tool.

## Context and tokens (2026-09-09)

**Established working set only (2026-09-13).** `WorkingSetImages` retains
its first transform-media stage cut and 256-to-128 high/low image-count cuts.
The acceptance-based completion experiment has been removed: no
`accept_views`, final-view inspection gate, or acceptance-only post-submit
closure remains. `JobSpec.image_retention` / `--image-retention` accepts only
`legacy` for compatibility; obsolete `completion` configurations fail clearly.
No predictive cost trigger is implemented. Preserve image-delivery improvements
and measure new image/text input and cache loss before changing retention.
The shelved design and evaluation separation are recorded in
`docs/visual_context_design.md`.

The OAuth lane resends the whole history on every call (the Codex backend
refuses stored responses; the Codex CLI does the same). What a run COSTS is
not the raw input count: the usage window prices a cached input token at
~0.13x an uncached one (fitted on the x-codex quota headers over run 5's 39
calls, error half a point; OpenAI's published API rate is 0.1x), so a token
that sits unchanged in the prefix is nearly free and a token that is new, or
sits after a prefix edit, is full price. Run 5 (M11, 39 calls): 786k raw,
560k cached, ~298k paid = 28% of a Plus 5-hour window; of the paid part 58%
was NEW IMAGES (each paid once, in full, the call it arrives) and 42% new
text plus prefix breaks. Design rules that follow:
- **Every image at the atlas's resolution, 512 px long edge at most** — one
  rule, `atlas.render.model_long_edge`, sizes every screen from its
  micrometres per pixel (never finer than the atlas, never upsampled, never
  above the cap; `render.atlas_native_long_edge` is it with the section's
  calibration; the fit path is not capped, its parameters are normalized
  against the render; `physical_views(long_edge=None)` is canvas pixels for
  host-side use, every model-facing caller passes a cap). A zoom is a crop
  at that same scale, so it costs only the pixels it shows. `VIEW_LONG_EDGE` 512; atlas images
  (seed strip, `view_atlas`, the atlas half of a write's picture) are sent
  at the atlas's own resolution and only ever shrunk to 512
  (`atlas_fetch.atlas_sized`; a mouse section at 25 um is ~100-200
  tokens — until 2026-09-10 they were upsampled to 512, a quarter of run
  19's input); `view_placement` panels 512 (~250 tokens; the default
  mode is `template`, the atlas alone on the section's canvas, because the
  section is already in the seed — the signature default was
  `side_by_side` until 2026-09-10, so every Astra compare re-sent the
  section at ~515 tokens a pair); `OVERLAY_LONG_EDGE` 768 is only the cap on
  the interactive-transform canvas, which the rule puts at ~450 px on a
  25 um mouse atlas (it was drawn at raw canvas size before 2026-09-10).
- **Images stay** (`adk/plugins.py WorkingSetImages`): every tool image is
  kept until 256 are live, then the oldest media-bearing calls are cut in
  ONE batch to 128 (`DEFAULT_MAX_IMAGES` / `DEFAULT_KEEP_IMAGES`, a cut
  result says "dropped from context"); the cut only moves forward and the
  seed strip is never touched. Measured on M11 at low effort, 2026-09-09:
  keep-all (run 6, killed at call 16) had median 0.10 mm / 30 of 36 within
  0.25 with positions written by call 15; newest-call-only (run 7, 21
  calls, submitted) 0.30 mm / 14 of 36, and its debrief said the "dropped
  from context" results made its comparisons unreliable. Cost: a kept
  image is 0.13x on every later call, so keep-all grows quadratically
  (2.4k -> 8.8k paid a call by call 15), but a batched run submits in ~21
  calls, where keep-all is ~16% of a window against newest-only's 14%.
  A normal run (seed 80, ~50 compares, ~40 write pictures) never reaches
  the cut; it is the safety for a run that goes long; cutting old images
  re-reads everything after the cut once, so it must stay rare.
- **A tool's images ride inside its `function_call_output`** as labelled
  `input_image` parts (`openai_oauth._function_call_output`); the separate
  user message they used to follow in opened a new turn and, under the
  API's default `reasoning.context = current_turn`, threw the replayed
  reasoning away. The request sends `context = all_turns`.
- **Reasoning is replayed.** Each turn's `reasoning` output item (encrypted,
  `store` is false) is kept on the model turn (`thought_signature`, summary
  in a `thought` text part so traces show it) and sent back ahead of that
  turn, the way the Codex CLI does. ~9k tokens by call 39, all cached.
- **Rows are compact** (`render.compact_rows`): null and empty fields are
  absent from every tool payload (a position-only run carried null transform
  fields on every row of every result, a third of the paid text).
- **A placement picture is sent once per geometry.** Successful
  `view_placement` and `set_positions` renders are associated with their
  function result and become seen only when their media survives filtering
  into a later model request. `set_positions` suppresses a picture only when
  the same section, position, orientation and cutting angles were already
  seen in a full-canvas atlas-bearing view; section-only, zoomed and failed
  renders do not count, and same-round compare/write siblings both return
  images.
- **Cost and context are separate.** `session.py TokenTally` prints every call's
  input, cumulative input, peak request input and the legacy input-only
  `paid~` proxy (uncached + 0.13 x cached; excludes output), plus, on the OAuth lane, the
  window percent this run has used; `JobSpec.max_quota_percent` (25,
  `--max-quota-percent`) ends the session when the run's share of the window
  reaches it, after ONE grace call that asks for `submit` (runs 5 and 8
  both died on `validate` with a 500-token submit next). `JobSpec.
  max_input_tokens` (`None` by default, `--max-input-tokens`) instead limits
  one request's reported input, cached included, after its response, with
  one grace submit call. Repeated below-limit requests never trip it, even
  if cumulative input exceeds the limit. A budget stop never breaks out of a turn:
  the pending tool call is answered first, so the history stays
  well-formed for the grace call.
- **Several tool calls per model turn** (`parallel_tool_calls: true` on the
  OAuth lane, 2026-09-09): one call's history cost instead of one per tool.
  ADK runs a turn's calls concurrently (sync tools in a thread pool), so
  `toolbox._serialized` puts every tool under one lock: one state, one
  writer at a time. `openai_oauth.py` sends one
  stable `prompt_cache_key`/`session_id` per session and surfaces the
  `x-codex-*` quota headers.
Not adopted, on Nash's call: server-side compaction (for heavy text; we are
light text, heavy image) and a `Memorize` action (replayed reasoning carries
facts forward already; Codex's `notes` exist to cross a hard context-window
reset, which our runs never hit). The WebSocket transport is a latency lever
only. Prefix caching makes replay of unchanged history cheap, not arbitrary
re-insertion of the same image later. Neither a byte-identical appended copy
nor a new overlay should be budgeted as a cache hit on arrival. The public
OpenAI prompt-caching guide documents prefix reuse, not content-addressed
image discounts; subscription-backend usage remains the measurement source.

## Rules that are not negotiable here

**Gates for the cheap models (`--gates`, `PositionSpec.gated`, 2026-09-09).**
With Luna effectively free on the subscription and Gemini 3.8 Flash on the
API key, Nash's direction is hand-holding for them. The harness does it as
data-only refusals first: gated, `set_positions` refuses a section not
compared since its last write (ONE compare, because Astra's own method
confirms each section at one hypothesised position; the first version
demanded two and blocked that), and `submit` refuses until `view_stack`
has run after the last write. Gates alone did not help Luna (run 10:
median 1.79, it looked without seeing), so `--playbook`
(`PositionSpec.playbook`) puts Astra's run-8 method into the job
statement's Method section, read off its trace: a complete hypothesis of
order and every position from the opening images (it used the interleaved
cutting series), one confirmation sweep four sections per call at one
candidate each, one bulk write, targeted re-checks, damage notes, order,
`view_stack`, submit. That is coaching text and the one
exception to the lean-harness rule below; off for Astra. Run 9 (M11, same harness as
Astra's run 8): Luna wrote all 36 positions in one uncompared call and
submitted at call 7 (median 1.2 mm, 0 of 36 within 0.25); Gemini 3.8 Flash
made 18 compares in 30 calls and never wrote. Astra passes the gates
without noticing. Off by default so the Astra runs stay comparable.

**Lean harness.** Tools return data. No interpretation in any payload. The job statement carries the job, the facts, one line per tool, the constraints and — when positioning is on — a short `Method` section (Nash, 2026-09-07): place each section on its own evidence and compare candidates before writing, review the whole stack afterwards, re-check both sides of a gap before reporting a break, submit. When interactive transforms are enabled, Method also asks the agent to inspect
each fitted/adjusted overlay against surviving internal anatomy and refine each
slice until no further improvement is possible with the available transforms,
keeping only changes that improve alignment; this is prompt guidance, not a
fixed-adjustment-count or submission-review hook.
The default Method no longer prescribes batching (2026-09-11): grouping work is the model's choice; batch-capable tools remain available. The opt-in cheap-model playbook is unchanged. Asked from its own run-3 trace, Astra said it skipped `compare_placement` (now `view_placement`) and `view_stack` by oversight, not wording, and asked for exactly this. Still out: rules of thumb, failure-mode warnings and region names (the same text runs against every BrainGlobe atlas, species and plane). Full-trace forensics found every major
benchmark failure tracking back to advice the harness injected; a per-slice
estimation worker that ate 82% of the wall-clock carried ~no signal and was
deleted; a landmark-tool pass for POSITION estimation benchmarked WORSE and was
deleted rather than kept behind a flag. The later in-plane paired-landmark
extension was also removed; nonlinear registration stays separate.

**One transform representation.** Silhouette, interactive, elastix-someday:
every stored transform and every fit payload carries `physical` (the five
knobs `adjust_transforms` takes, plus `shear`, about a pivot in canvas fractions)
next to the six normalized numbers. The fraction-based `decomposition` left
the fit payload: "+0.04 mm entered, negative fraction reported" cost four
sessions, and GPT-6 Astra could not hand a fit's numbers to the manual
controls. A fit is now a starting point for `adjust_transforms` and the B side
of `mode="ab"`.

**Gates are constraints, not coaching.** A refusal states the numbers that
caused it; failed submission neither writes nor ends the run.
`submit` refuses `MISSING_POSITIONS`,
`ORDER_POSITION_MISMATCH` (positions must run one way along the corrected
order; the offending neighbour pairs are named), `STRICT_INTERVAL` (spacing
within 10% of the interval and no breaks, when `--strict-interval`),
`INTERVAL_BREAKS_UNSUPPORTED` (a reported break must exceed 1.5x the stack's
median written spacing) and `MISSING_TRANSFORMS` (every section carries one,
damaged included). `DAMAGED_REQUIRES_MANUAL_TRANSFORM` additionally rejects
missing, automatic, invalid or identity transforms on damaged sections and asks
for interactive alignment of surviving anatomy. A note alone cannot satisfy it.
The check uses the normalized matrix (identity tolerance 1e-9), not the physical
parameter labels, and runs during `submit`. If interactive tools
are disabled, the refusal names that host setting as the blocker. It does not
measure anatomical alignment quality. Gates only run for tasks that are on.

**Corrections are data.** Order, flips, rotations, positions, cutting angles
and transforms are proposals on the state. The user's image files are never
modified.

**Order and position must agree.** Only at `submit`: `reorder_slices`
changes nothing but `index_corrected`, and the
`ORDER_POSITION_MISMATCH` gate is what holds order and position together.
(Reordering used to CLEAR the positions of the sections that moved; every
debriefed agent called that hazardous, and it made them re-enter numbers they
still believed.) `orient_slices` does still clear a section's transform: a
transform is defined AFTER the orientation it was fitted under.

**Every write checkpoints, every write is undoable.** One tool call is one undo
step, so a batch undoes as one. The undo stack is in memory only (depth 50); a
resumed run starts from the checkpoint, which is the state as it stood.

## Ceilings worth knowing

- `fit_affine`'s silhouette method measures against the atlas plane at the
  stack's cutting angles (`atlas_fetch.atlas_mask` handed to
  `affine.silhouette_affine` as `atlas_mask_at`, 2026-09-10). Until then it
  measured against the FLAT section on a 13-degree brain and said so with
  `flat_atlas_fit`; Astra's run-19 debrief asked for exactly this.
- `physical` on a fit is the five knobs about the canvas centre plus the
  `shear` they cannot express (0.10-0.16 on M05_D_08, not noise), so a preview
  typed from a fit reproduces it only up to that shear. The A/B view does not
  have that gap: its B side is drawn from the six stored numbers
  (`affine.denormalized_affine`), which are exact.
- The moments core has a 180-degree ambiguity — `(1,1)` and `(-1,-1)` are the
  same axes turned round — and settles it on silhouette IoU alone. On INTACT
  M05 sections the right one wins by 0.036-0.092 IoU; a template-correlation
  tie-break would help the cases where silhouette IoU alone is close, but that
  is measured, not built: it would move a benchmarked path (`silhouette_affine`
  is `nonlinear/quick_affine`'s too). The fit's own panel is what catches it;
  look at it. Damaged sections never reach this path: `fit_affine` refuses
  them outright.
- `method="elastix"` and `run_deepslice` answer `UNAVAILABLE`; both are seams,
  not stubs to fill in casually.
- `search_position` (was `fit_position`) is a thin wrapper over `oblique.fit_oblique` — correct, not
  tuned. It has not been benchmarked.
- True physical scale is honest, not flattering: measured on LSD_910 M04 at
  4.9 mm the specimen is 8.26 x 5.73 mm against the atlas section's
  9.05 x 6.68 mm, so the outlines land ~10% (ML) to ~15% (DV) outside the
  tissue at identity. That is the specimen-vs-Allen size difference, the same
  residual the nonlinear side measures, not a calibration bug.
- `adjust_transforms`'s pivot resolution runs its own `canvas_geometry`, so a
  call builds the atlas plane twice (once to place the pivot, once to draw).
  Flat sections are a numpy take; oblique ones resample twice.

### Host transform choices

`TransformSpec.interactive` and `.automatic` default to true. Hosts such as the
ABBA menu may independently remove direct adjustment tools or the
automatic `fit_affine` tool. `elastix` controls the optional backend within the
automatic fitter. The transform task's submit requirement still applies; a
host must enable at least one transform method when enabling that task.

## Claude host briefing

`prompt.run_facts` is shared with `mcp_server/prompt.py`; keep the factual
range, axis direction, protocol and calibration text identical across hosts.
Claude's statement does not reuse the ADK method/playbook. MCP opens saved ABBA
jobs through `api.abba_worker.prepare_linear`, exactly like `linear.run`,
and supplies opening pictures separately with `show_stack` pages. No image
model is available through the Claude connector.

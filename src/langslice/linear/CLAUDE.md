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
positioning `side_by_side` returns separate tissue-framed reference images:
one unchanged section per distinct id, one atlas per pair (up to 8 images),
mapped by zero-based `image_indexes` in each compared row. These are
independently tissue-framed, not a common physical canvas; zoom is refused
and outlines/opacity do not apply. Other modes still return one physical
canvas per pair. Interactive-transform `side_by_side` is unchanged.
Comparison and atlas-fetch paths share cached reference pictures
(`core/pictures.py` `reference_section_picture` / `reference_atlas_picture`,
plain captioned pictures on `Workspace.picture_cache` since 2026-10-03, phase
3b; until then encoded parts on the driver's context; drawn by
`render.reference_slice_picture` / `atlas_fetch.atlas_picture`):
section captions retain their first display index/flags across reorder,
filenames are the stable identity, and orientation/preprocessing/size changes
get distinct entries. Atlas entries include exact position, plane, angles and
size. The opening no longer feeds them: since 2026-10-03 it is strips
(`opening.py`), and since 2026-10-01 a comparison was the later-picture size
anyway, so the two never shared bytes.
Reusing bytes avoids recomposition, not new-image input charges. An image
replayed in its original unchanged history prefix is eligible for prompt-cache
reuse; another copy appended after new conversation content is new input,
even when byte-identical. Do not budget a cached-input discount for that copy
merely because the opening showed it. Later requests can reuse its new prefix
if history stays unchanged and the backend retains a matching cache entry.
Rejected: labelled anatomy/landmarks, confidence and verification states,
damage masks, an anatomy-based gap review, a validity-vs-verification audit.

## Files

**Core and doors (layered refactor, phase 1, 2026-10-03).** The core
modules take a `workspace.Workspace` and return plain PIL pictures, numbers
and text; none imports `google.genai`, ADK, litellm or openai
(`tests/test_core_imports.py` checks each in a fresh interpreter): `workspace`,
`render`, `display`, `transform`, `deformation`, `appearance`,
`atlas_fetch`, `opening`, and `registration_handoff` at the top level. The
doors turn them into what a host reads: `adk/media.py` (the one module that
makes `types.Part`s: JPEG encoding, `packaged` / `package_result` for the
tools' pictures, the opening as parts), the toolbox and `view_options.py`
(argument checking, the model-lane image limit and the wording the model
sees; neither imports `google.genai` since phase 3b, checked by
`tests/test_core_imports.py`), the MCP server (`encode_jpeg` straight into
MCP image blocks, for the opening and for the tools' plain pictures).
`engine.py` and `session.py` are the ADK driver; `engine.run_session` hands
the agent `adk.media.packaged_tools(box.tools)`.

**The job layer (phase 2, 2026-10-03).** `job.py` sits between the core and
the doors (it imports the core only, and `tests/test_core_imports.py` checks
it too): one `Job` owns the `StackState` and `JobSpec`, the host's locked and
damaged sections, undo/redo, the checkpoint and its observers, the submit
gates, the stale-deformation rule, `over_cap`, the deformation
`RecordStore`, the checkpoint and results paths, and the background image
corrections. The native tools (`build_tools(state, ctx, spec, job=...)`; a
job is made around the state when none is passed) and the MCP server
(`mcp_server.server.Session`: the core job, the toolbox over it and the
door's own pages, trace and host channel) both sit on it. The
look-before-commit gates (`compared`/`reviewed`) and the model-delivery
bookkeeping (pending/seen placement views, delivery ids, `tool_context`)
stay on the `ToolBox`, the tool door: a library call on the job is never
gated.

**The operations (phase 3a, 2026-10-03).** The writes live in the top-level
package `src/langslice/ops/` (its own `CLAUDE.md`), one verb group per file:
`positions` (`set_positions`: clamp into the atlas range and write;
`set_cutting_angles`), `order` (`reorder`, `renumber`), `orientation`
(`orient_sections`: flip/quarter turn, drops a transform the change made
stale), `damage` (`mark_damaged`), `appearance` (`set_appearance`,
`planned_settings`), `notes` (`add_note`) and `transforms`
(`interactive_transform`: knobs to the stored record; `fit_transform`:
`fit_affine`'s record; `set_transforms`: any number as one undo step). Each
takes the job (and the workspace where it reads the atlas), takes one undo
step through it and returns a plain record of what changed; a refusal is
`ops.refusal.Refused`. The tool bodies for these verbs are argument checking,
the look-before-commit gates, one ops call, the pictures and the wording.
`preprocess` draws its AFTER pictures from `planned_settings` and writes only
once every picture is drawn.

**Pictures in the core, plain pictures to the doors (phase 3b,
2026-10-03).** The pictures the tools send are built in the new core package
`src/langslice/core/` (its own `CLAUDE.md`): `core/pictures.py` (the
section and atlas pictures of `view_slices`, `orient_slices`, `view_atlas`,
`view_stack`, and the cached reference pictures) and `core/placement.py`
(every placement picture: `draw_canvas`, which returns the panels and the
`CanvasFrame` they were drawn in, `stored_placement`, `current_warp`,
`placement_pictures`, and the interactive transform's `stage` / `Staged` /
`staged_views`). The deformable fit is an operation,
`ops.deformable.fit_deformable` (candidates, include/exclude, start,
traced fit sections; one setting applies as one undo step), and so is
`ops.deformable.keep_linear`; `fit_deformable`'s tool body validates the
arguments (`resolve_choice`, the region checks), calls it and draws the
pictures from what it returns. Every tool returns plain PIL pictures (and,
for `view_stack`, lines of text) under `TOOL_MEDIA_PARTS_KEY`; the ADK driver
packages them as JPEG message parts (`adk.media.packaged`), the MCP server as
image and text blocks (`result_blocks`), so the bytes each host receives are
those it received before (the goldens check).

**The job folder (phase 3c, 2026-10-03).** A job's files live in
`<images>/langslice/` (`src/langslice/job/`, its own `CLAUDE.md`):
`job.json`, `state.json`, `history/` (one file per undo step),
`sections/<stem>/` (deformation records, image corrections, pictures),
`views/`, `views.jsonl`, `exports/`, `logs/`. `Job.open(spec, workspace,
folder=..., results_path=...)` upgrades an old layout first
(`job.migrate`), writes `job.json`, then resumes or ingests;
`engine.EngineContext` carries `job_folder` (`checkpoint_path` and
`layout` are derived). Every path the state stores is relative to the job
folder (state format 2). Every picture a tool returns is saved there with
its layers (`toolbox._saves_views` around every tool, `Job.views`; the
ADK opening through `engine.save_opening`, MCP pages through
`server.save_page`), the exact JPEG bytes the door sent; replies are
unchanged.

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
- `workspace.py` — `Workspace`, the core context: the spec, the image
  folder, the atlas (`atlas_loader`, loaded once), `abba_atlas` (ABBA's
  cached Allen atlas when it matches, looked up once), each section's
  `working_source` (the working copy and its scale, `source_cache`),
  `section_channels` (`channel_cache`), `calibration` (host, then file
  tags), `position_range`, `axis_ends`, `species`, and the `render_cache` /
  `render_scale` caches and the `picture_cache` of captioned reference
  pictures (`core.pictures`). No model: the driver's `engine.EngineContext`
  subclasses it to add that.
- `checkpoint.py` — atomic JSON write of the job folder's `state.json`
  (`default_checkpoint_path`: `<images>/langslice/state.json`), versioned
  (`format_version`, `STATE_FORMAT_VERSION` 2; `upgrade_state(data, root)`
  reads an unversioned checkpoint as version 0, makes version 1's absolute
  paths inside *root* relative, and refuses a newer one), `state_paths`
  (every path a state stores, through one converter), `relative_to`, and
  the global observers (`observe_checkpoints`). `CHECKPOINT_FILENAME`
  (`linear_state.json`) names the old layout's file, for the migration.
- `job.py` — the job layer (above): `Job.open` (migrate an old layout,
  write `job.json`, resume the checkpoint when `spec.resume`, else `ingest`
  + `apply_host_inputs`, then the first checkpoint), `Job.layout` /
  `folder` / `checkpoint_path` / `undo_path`, `Job.portable` (an image
  correction's paths made relative), `Job.views` (`job.views.ViewStore`),
  `Job.close` (flush the pictures), `ingest`, `apply_host_inputs`,
  `host_transform`,
  `emit_results`, the submit gates (`submit_errors` and its parts,
  `missing_deformations` included). Undo is ONE pattern: `before =
  job.snapshot()`, write, `job.commit(before)` (one undo step, then the
  checkpoint). The history is the job folder's `history/`
  (`job.history.History`: `index.json` lists the `undo` and `redo` steps,
  oldest first, each step one whole-state file written once; depth
  `UNDO_DEPTH` 50, steps beyond it deleted), read back by a resumed job and
  emptied by a fresh one; an unreadable one starts empty. `Job.sync` (run inside the toolbox lock before every tool
  call) reloads the state file when its inode, size or mtime changed since
  the job last read or wrote it: a script's edit becomes one undo step, a
  second job's edit comes with its own history index, which is read back;
  an unchanged rewrite is not a step and a file that does not parse (a
  script mid-write) is left for the next call. A step holds the whole
  state, so undo restores the deformation references on the section
  records too (the records on disk are content-addressed and never change).
  Not undone, deliberately: `transform_history` (what was tried, not what
  holds), the seen/pending placement views (what the model received cannot
  be unseen), the render and reference-picture caches (keyed by everything
  they draw, so never stale), running image corrections (they land only on
  a section still waiting at the same geometry). `compared`/`reviewed`:
  an undo, redo or reload that MOVES a section's position counts as a write
  to it, as `set_positions` does: that section's compared views are dropped
  and the stack needs a new `view_stack` (`toolbox.forget_looks`).
  Missing image corrections take the fingerprint function from the door
  (`registration_tool.correction_fingerprint` lives with the provider code
  until phase 4).
- `discovery.py` — natural-sorted image discovery.
- `render.py` — `render_slice` (ROTATE first, then FLIP, then the display-only
  `--preprocess auto` enhancement), `stack_pictures`, the status rows and
  their text form, `caption`, `reference_slice_picture` (the captioned,
  tissue-framed section the comparison tools send), and the PHYSICAL
  overlay: `canvas_geometry` places an atlas section on a section's frame at
  true scale (`atlas um/px / canvas um/px`, anatomy centred, canvas grown to
  hold it — never fit-to-canvas, which is not a calibration), and
  `physical_views` draws the alignment picture on it (family outlines
  from `atlas.render.family_outlines` as yellow 1 px lines by default, drawn
  at OUTPUT size. Agent tools expose `view.border_color` (named color/#RRGGBB) and
  `view.border_thickness` (0.25–8 output pixels, including fractional widths) alongside `view.atlas_opacity`;
  these are display-only and do not change transforms or IoU. The native ABBA
  viewer keeps its own display settings. Includes a 1 mm scale bar and two-line
  caption). It takes either the knobs (`shear` optional; the caption names
  a non-zero one) or a ready 2x3 in the
  section's frame, so the interactive loop and `fit_affine` draw the same
  picture, and returns `(images, silhouette_iou)`. `left` (2026-10-03) is a
  fit's own side split for one-sided `regions` (`transform.FitFrame.left`),
  used instead of resolving the sides through the drawn placement, which has
  none once it turns the midline past 45 degrees. Its view controls:
  `mode` (`overlay`, `side_by_side` — two physical images, `checkerboard`,
  `outlines` — atlas lines plus the section's own silhouette in neutral grey
  on black, over any atlas image the call lists at `atlas_opacity` — and the line-free `section` / `template`), `zoom` ([x0, y0, x1, y1] fractions of the CANVAS, cropped BEFORE
  the resize so it magnifies, with the bar redrawn for the new µm/px),
  `atlas_opacity` (0..1; `template_opacity` until 2026-10-01, the
  `show_template` bool before that) and `outlines`
  (`OUTLINE_LAYERS`: `all` family boundaries, `outer` — the root contour from
  `atlas.render.outer_outline` — or `none`; the caption names the layer when
  it is not `all`).
  `physical_overlay` is the one-image `overlay` wrapper (tests and scripts;
  `fit_affine` draws through `core.placement.draw_canvas` from the fit's
  `FitFrame`).
  `caption` wraps any line wider than its picture (`wrap_caption`: at spaces,
  mid-word for one long word), so a small picture keeps its whole label; a
  caption that fits draws the same pixels as before.
  `pivot` (canvas px, `pivot_on_canvas` resolves "canvas"/"tissue"/[fx, fy]
  onto it) and `markers` (the landmark pairs, a cross per section point and a
  ring per atlas point in their own two colors) ride through both.
  Region acronyms and damage masking were asked for and deliberately not
  built. `canvas_um_per_px` follows the file's pixel size down through
  the render's own downsample (`render_scale`, same key as `render_cache`);
  `estimate_um_per_px` is the fallback guess from tissue width. Renders are
  cached on the workspace and shared: read them, never mutate them. `caption`
  burns a label into a COPY — every image a tool returns gets one, because
  tool images reach the model as bare attachments; never caption an image a
  fit measures.
- `atlas_fetch.py` — `atlas_section` (the one atlas renderer: flat at 0/0
  cutting angles, `oblique.sample_oblique_plane` otherwise), `atlas_picture`
  (one tissue-framed atlas section, sized and captioned; the `view_atlas`
  tool, was `fetch_atlas`, is built in the toolbox, `make_view_atlas`), and
  `reference_atlas` (the opening's evenly spaced atlas positions, at most
  `SEED_ATLAS_MAX_IMAGES` 48, never upsampled). Sections and atlas sections
  are framed the same way so apparent scale is not a cue.
- `opening.py` — the opening images (2026-10-03, Nash: "a strip of atlas
  images and slice images, just like how abba does it"): `opening_items`
  lays the stack out as horizontal strips in corrected order, the sections
  on top and, directly beneath each, the atlas at its CURRENT position and
  the stack's cutting angles (`atlas_tile`, drawn to the section's long edge
  as `view_stack` does, so an olfactory-bulb plane is not a thumbnail).
  Every tile is labelled in its pixels (`tile_label`: `"<index>:
  <filename>"` plus short flags `rot N`/`flipped`/`damaged`, the damage note
  stays in the status table; atlas tiles `atlas <mm> mm`, a section without
  a position gets `no position`), columns are split by a thin grey line,
  and a text before each strip lists its sections; it returns the texts and
  strips in reading order as plain strings and PIL images
  (`adk/media.opening_parts` makes them message parts for the seed). A
  strip's long edge is the model lane's largest image (`limit`, from
  `adk/media.image_limit` on the run's model: `OPENAI_MAX_IMAGE_EDGE`
  2048 for openai-oauth/openai-api, any other lane uses it too, unmeasured;
  `CLAUDE_MAX_IMAGE_EDGE` 1568 for the MCP host; a later picture's cap is
  separate, `CLAUDE_MAX_VIEW_EDGE`, see `view_options.py`), tiles the level's opening
  size (`strip_layout`: as many as fit, shrunk by at most a few pixels so a
  full strip fills the edge): 8/5/4 per strip at low/medium/high on the
  OpenAI lanes, 6/4/3 for Claude. `pack_strips` also keeps every strip
  inside the lane's patch budget (`OPENAI_MAX_IMAGE_PATCHES` 2500 32-px
  patches at detail high, verified 2026-10-03; `CLAUDE_MAX_IMAGE_PATCHES`
  ~1.2 MP), past which the encoder would shrink the strip, labels included:
  a column that would cross it starts the next strip. With no position at
  all the strips are section-only and the atlas reference
  (`reference_atlas`) follows as atlas-only strips; when every section has
  a position the reference is not sent; a partly placed stack gets both.
- `arguments.py` — the shapes of the arguments (2026-10-03): `View` (the
  picture options) and `ViewAuto` (+ `resolution`), and the `entries` /
  `candidates` dicts (`DamageEntry`, `OrientEntry`, `PositionEntry`,
  `PlacementEntry`, `TransformEntry`, `Candidate`, `FixedCandidate`), all
  pydantic-configured typed dicts with `extra="forbid"`, so ADK's and
  FastMCP's schemas name every key and type (the OAuth lane inlines the
  `$defs`, `openai_oauth._inline_refs`). `argument_refusal(func, args)` is the
  one strictness rule: unknown top-level names, and unknown keys inside any
  typed-dict argument, answer `UNKNOWN_ARGUMENTS` with the keys, where a
  stray key belongs (`` `mode` belongs inside `view` ``) and `KEY_NOTES` (a
  `resolution` below "auto": the user fixed the picture size; a candidate
  `engine` when the engine is fixed). Applied by `toolbox._strict` (every
  call, direct ones included), `adk.plugins.StrictArgumentsPlugin` (ADK's
  `FunctionTool` drops unknown top-level arguments before a tool runs; the
  plugin's `before_tool_callback` answers first) and
  `mcp_server.server.strict_arguments` (FastMCP drops them too; checked
  after its JSON pre-parse). `normalize_arguments(func, args)` (2026-10-03)
  makes what the tools accept as sent schema-valid for a door that
  validates first: a whole number in a text argument (`id`, `section`) or in
  `slices` becomes its text (a corrected index; the tools resolve `"2"` and
  `2` alike), a null inside a typed-dict argument is dropped and a null
  `view` is the default. The MCP door applies it after the refusal check;
  ADK hands the values through as sent. `TransformEntry` carries the
  optional `shear` knob (2026-10-03).
- `display.py` / `view_options.py` — `view`, the picture options (2026-10-01
  as nine flat arguments; one `view` object since 2026-10-03, Nash: "Our
  tools have been really messy"; ABBA's image-channel / atlas-channel
  design). The door half, `view_options.py`: `parse_view` validates one call's `view` against the tool's `Profile` (its modes, first
  = default; whether `channels`, the atlas keys and `deformation` apply; zoom
  and deformation modes; per-mode atlas defaults) and `MODE_RULES` (what each
  mode draws, `display.py`) into a frozen `DisplayOptions` (`display.py`, the
  core half with the defaults and the renderers); a key that means nothing for the
  tool or the mode answers `VIEW_KEY_UNUSED` with each reason. Keys: `mode`;
  `channels` — raw channel names (look `{"overlay": names}`: each stretched
  by percentile 1..99.5 on the whole working plane, `render._look_image`; one
  in gray, several added in `appearance.channel_colors`, a channel named
  after a colour keeping it, each dimmed by its `render.fine_detail`
  relative to the most detailed one — on M11_B_03 the flat green
  autofluorescence, stretched alone, washed out the nuclear stain) or one version, `view` (default) / `fit`;
  `atlas_channels` — `ara`, `nissl` (ABBA's cached atlas only,
  `Workspace.abba_atlas`), `borders` (the lines); `atlas_image_picture`
  composes the images (ara alone = the renderers' own reference path, none =
  black, two = green + magenta); `atlas_opacity` (default 0.5 when an image
  is listed in a blending mode); `regions` (descendants included,
  `render.region_polys`, context outlines at `REGION_CONTEXT_ALPHA`; an entry
  may name one side, `"CTX:left"`, which `render.regions_left` resolves per
  picture: on a physical canvas the SECTION's side through the section matrix
  — a mirrored placement shows it on the canvas's other side and the caption
  says so — and on an atlas-only picture the picture's own side; `NO_SIDES`
  on sagittal stacks); `outlines` (`all`/`outer`, only with `borders`);
  `border_color`, `border_thickness` (default `DEFAULT_BORDER_THICKNESS` 1.0
  everywhere, picked by eye on M11_B_03: 0.5 faded into bright tissue, 1.5
  covered ventricle edges); `zoom`; `deformation` (`applied`/`none`,
  placement tools); `resolution` only at image resolution `auto`:
  `view_options.view_schema` (applied to every tool in `build_tools`) swaps `view`'s
  annotation for `ViewAuto` there, and `clamp_resolution` clamps it to
  `render.MIN_RESOLUTION` 128 .. the driver model's largest image with a
  `view.resolution_note` (Nash 2026-10-03: the cap is the model's own
  maximum, not a fixed 1536). The door knows the model and passes the cap
  (`build_tools(max_view_edge=...)`, kept as `ToolBox.max_view_edge`, read
  by `parse_view`, `view_atlas` and the job statement's resolution line):
  the ADK agent's default is `adk.media.view_edge_limit`, the lane's
  largest image edge (2048 px on the OpenAI lanes and any unmeasured lane;
  there a near-square picture past ~1600 px still meets the 2,500-patch
  budget, which shrinks it); the MCP door passes
  `opening.CLAUDE_MAX_VIEW_EDGE` 2000 (a Claude request holding more than
  20 images takes none larger; Claude 4.7+ reads ~2576 px alone, older
  models shrink past 1568). `DisplayOptions.long_edge` is the
  call's picture size; `echo()` is what the call drew (`channels` only
  where they choose the picture, `channels_apply`: not for `preprocess` or
  `fit_deformable`, 2026-10-03). It never writes state,
  so a call's options never change a default. `framed_section` /
  `framed_atlas` draw the tissue-framed pictures (default options = the old
  pixels exactly), `channel_strip` the `view_slices` channels mode (each raw
  channel unmodified, labelled); `core.placement.draw_canvas` draws every
  physical picture (its caption names the channels and any atlas under the
  section, `canvas_label`).
- `appearance.py` — the section's appearance per target (2026-10-01): `view`
  (what the agent is shown) and `fit` (`fit_image`, what a deformable fit
  reads), each a stack setting plus per-section overrides on
  `StackState.appearance` (undone and checkpointed). `None` is the DEFAULT
  appearance (today's `preprocess` auto, or the host's blend when
  `spec.host_preprocessing` is set); anything else is drawn by
  `render_slice(look=...)` from the raw channels
  (`Workspace.section_channels`) over the same frame and size. The
  `preprocess` tool (gated by `spec.agent_preprocessing`) is the only writer;
  it returns each pictured section BEFORE (the target's look before the call,
  drawn first) and AFTER, labelled.
  Computation (silhouette fit, calibration, tissue pivot, `search_position`,
  the image model's input) always reads the default.
- `toolbox.py` — `build_tools(state, ctx, spec, job=None)`: every tool,
  gated by the spec, on the job (state, undo, checkpoint and submit gates
  are the job's), plus the door's own record on the `ToolBox`: the
  look-before-commit gates and the delivery bookkeeping. Every write goes
  through `langslice.ops` (above) and every picture is drawn by
  `langslice.core` (above); the tools return them plain. The interactive
  transform: `core.placement.stage` (one section, its calibrated canvas and
  the resolved pivot; the door checks the knobs and fills a left-out shear
  first), `adjust_transforms` (the write AND the look, any positioned
  section; `mode="ab"` draws the new parameters beside what the section
  carried before the call, a silhouette fit included; the same numbers again
  re-draw without an undo step). Each entry is staged and drawn, its record
  built by `ops.transforms.interactive_transform`, and the call's records
  written once by `ops.transforms.set_transforms`: one to four distinct
  sections share one undo step and one `view`; each returns one image, or two for `ab`/`side_by_side`, mapped by
  per-result `image_indexes`. The knobs are rotation, the two scales, the
  two shifts and, since 2026-10-03 (Nash), an optional `shear` (left out:
  the section's current shear is kept, 0 when it has none; an explicit 0
  drops it; the reply's `physical` shows the shear stored):
  `affine.decompose_affine`'s convention, linear part
  `R(rotation) . [[scale_x, shear*scale_x], [0, scale_y]]`, a unitless slant
  before the rotation in units of `scale_x`, the same number a fit reports,
  so a tweak after a fit keeps its map (an Elastix fit's shear kept: 0.07
  px; set to 0: 8 px, `test_linear_fit_affine_elastix`).
  Paired landmark tools are removed; interactive
  alignment exposes direct affine adjustments only.
  `transform_history` on the ToolBox is per section and lasts the whole run,
  but is not repeated in tool replies. `answered(*touched)` (after an ops
  write) is what ordinary writes answer with: the status rows of
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
  scales, `translate_x_mm`/`translate_y_mm`) plus `shear`, and the recorded transform
  carries them under `"physical"` (with the pivot as canvas fractions) next to
  the six normalized numbers plus the `"calibration"` used.
  The fits return numbers, never pictures (2026-10-03): an ok payload carries
  a `FitFrame` under `FIT_FRAME_KEY` (the working frame, its calibration,
  the fitted matrix on it, and `left`, the side split the fit resolved for
  one-sided regions from the placement it started from), which the caller
  pops and draws from; there is no draw callback.
  `physical_decomposition` is `decompose_affine` minus its translation
  fractions; `similarity_fit` (Umeyama, exact on two points), `affine_fit`
  (least squares) and `physical_params` (a canvas 2x3 back into the knobs,
  shear included, about a pivot) are shared affine geometry helpers.
  `region_silhouette_fit` (2026-10-01) is `fit_affine`'s `include`/`exclude`
  path: atlas regions resolved by the deformable package
  (`deformable.atlas_images.regions_mask`, `deformable.masks`; a one-sided
  entry such as `"CTX:left"` is the section's side, carried onto the atlas
  plane through the CURRENT stored transform, `atlas.sides.native_left`,
  with the plane passed as `plane_at`), the kept footprint (minus
  excluded; with include, within `DEFAULT_NEIGHBOURHOOD_UM` of them) against
  the tissue the section's CURRENT stored transform lays there, one pass of
  `affine.mask_affine` on the TRUE-SCALE canvas (not the stretched fit frame
  of the whole-outline fit). Iterating the tissue selection drifted (D_08: 10
  degrees in 8 passes, never settling), so it is one pass per call. Reports a
  `regions` dict (kept/used fractions, `affine.axis_ratio` of both masks,
  `outline_share`, a note past a 45-degree turn or below `ROUND_AXIS_RATIO`);
  `RegionRefusal` codes `REGIONS_INSIDE_OUTLINE`, `REGIONS_ABSENT`,
  `REGIONS_LEAVE_NOTHING`, `NO_TISSUE_IN_REGIONS`. No regions: the old fit,
  byte-identical (M04 B_05/A_01 params and pictures hashed against 5b873d3).
  `fit_elastix` (2026-10-03) is `fit_affine`'s DEFAULT method, an intensity
  affine that refines the section's CURRENT placement (stored six numbers,
  identity without; never a search from scratch). `elastix_affine` places
  the atlas plane there on the deformable fit grid (`deformation.fit_grid`
  with a stand-in `transform`, so a section without one starts from
  identity; `_start_calibration` keeps the calibration those numbers were
  drawn with), prepares the `fit` appearance against the ARA template
  exactly as a deformable stain fit does (`deformable.prepare_fit` with
  `elastix_settings`: mutual information + edge channel, `include` ->
  `structures`, `exclude` -> `exclude`, one-sided entries included; an
  INTACT section gets no torn-edge band, since the band's rule also marks
  outline the start merely overhangs), runs
  `deformable.engines.run_elastix_affine` (Elastix `AffineTransform` from
  the identity, fixed seed and threads: identical inputs give identical
  numbers) and composes the result onto the start (new matrix = start @
  step, step the engine's mm map in grid pixels). A full affine: the six
  numbers carry its shear exactly and `physical` reports it. `iou` is the
  tissue against the kept atlas footprint (`_overlap`, the silhouette
  fit's region rule) under the fitted placement. The payload is
  `_fit_payload`, shared with `fit_silhouette`, so both methods reply,
  draw, checkpoint and undo identically; the stored `kind` is the method.
  Elastix errors and failed preparation answer `FIT_FAILED` per section.
- `deformation.py` — `fit_deformable`'s machinery (2026-10-01): the fit grid
  (`fit_grid`: `prepare_linear_registration` at `FIT_LONG_EDGE` 1536, the same
  handoff `trace_borders` uses), the image a fit reads (`stain_image`: the
  `fit` appearance — a raw channel is a fit appearance `preprocess` sets, no
  longer a section image of its own; `traced_lines`: a completed
  `trace_borders` result at the CURRENT geometry fingerprint, its
  `extracted_lines.png` mapped onto the grid through `atlas_to_canvas`;
  `TRACE_TIMEOUT` after the call's wait, `TRACE_FAILED` with the error,
  `TRACE_RUNNING` only for a running record no call backs, e.g. after resume),
  `trace_picture` (those lines on the stain at the call's size and style),
  `Choice` (one candidate: section/atlas image, engine, stiffness ->
  `FitSettings` at `DETAIL_LEVEL` "standard", which tests set to "coarse";
  agent `borders` = engine `borders_merged`; a stain fit gets the engine's
  stain defaults, local correlation + edge channel, and with ANTs also the
  automatic tissue/ventricle label channels `labels="auto"`, per the
  2026-10-02 stain ceiling test; `engine_settings` reports the metric,
  correlation radius, edge channel and label map),
  `RecordStore` (results by `cache_key`, a digest of every input incl. the
  linear placement and the start record; 8 in memory, applied ones saved to
  `<folder_of(section)>/<key[:24]>/`, the job's
  `sections/<stem>/deformable/`; `current` is None for a `keep_linear`
  record), `linear_key` /
  `clear_stale`, `summary` (displacement max/median, fold fraction, compact
  flags, the whole-section `SECTION_FLAGS` DISPLACEMENT_OUTSIZED and FOLDS
  listed before the per-region ones so `MAX_FLAGS` never hides them),
  `run_jobs` (one fit in process, several in the spawn pool,
  `USE_PROCESS_POOL`) and `picture` (the borders on the image the fit read at
  the call's picture size, `Style.long_edge`, never past the 1536 px fit
  image; included/`regions` strong, excluded pink).
- `prompt.py` — `build_job_statement`: job, run facts, ONE factual line per
  tool that exists, hard constraints. Nothing else. `display_facts` reads
  the run's raw channels and atlas channels off the workspace for it. `tool_line` words the
  `fit_deformable` line for the run (traced fit sections only with an image
  model). `display_lines` describes `view` ONCE, with the raw channels (and
  that their names may not identify the stain; `view_slices` mode channels
  shows each) and the atlas channels this host has, each with one line
  (`ATLAS_CHANNEL_LINES`); `PICTURE_TOOLS` includes `fit_deformable`.
- `session.py` — the ADK agent builder, the plugins, the loop, and
  `TokenTally`: every call's usage is printed and traced, and
  `JobSpec.max_quota_percent` (25) ends the session when this run's share
  of the provider's usage window reaches it. Optional `max_input_tokens`
  (default `None`) checks one request's reported input, including cached
  tokens, after its response. It is not a cumulative spending guard or a
  preflight size guarantee. Logs separate cumulative input from peak request
  input; 1.3M processed over a run does not mean a 1.3M-token context.
- `engine.py` — `EngineContext` (the `Workspace` plus the checkpoint and
  results paths the job is opened at and the model), `run_session` (the
  agent gets the tools `adk.media.packaged`), and `run(spec)` (`Job.open`, the
  toolbox on it, the session, `Job.emit_results`). `ingest` and
  `apply_host_inputs` are re-exported from `job.py` for the SliceBench
  adapters. No post pass: the session is the whole run.
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
  Refuses medium/high/auto image resolution (nothing measured; the low runs
  predate the 2026-10-01 picture sizes); single-run or
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
  `job.host_transform()` (`kind` `"host"`, identity params, identity
  `physical`). `orient_slices`, `fit_affine` and `adjust_transforms` refuse
  them per section (`LOCKED`); `fit_affine`'s default target list skips them;
  their positions still move. The host transform satisfies
  `MISSING_TRANSFORMS` and locked sections are exempt from
  `DAMAGED_REQUIRES_MANUAL_TRANSFORM`. The worker never emits orientation or
  transform rows for them.
- **`image_resolution`** (`low`|`medium`|`high`|`auto`, default `low`;
  2026-10-01): `render.PICTURE_EDGES` gives each level two long edges, the
  opening images (`render.opening_edge`: each tile of the opening strips,
  sections and atlas, `opening.py`) and
  every later picture (`render.picture_edge`: each panel a tool returns):
  low 256/512, medium 384/768, high 512/1024, auto 256 then the agent's
  `view.resolution` per call (128 up to the driver model's largest image,
  2048 px on the OpenAI lanes, 2000 for a Claude host; 512 when it gives
  none). Only what the
  agent is SHOWN changes. Nothing is upsampled past its source: a framed
  `render_slice` treats `long_edge` as a ceiling over the working copy, a
  physical picture (`core.placement.draw_canvas`) renders the section at the panel size
  divided by the zoom span (`render.shown_section`, the matrix and pivot
  carried onto it by `rescale_section_matrix`) and `physical_views` shows
  the crop at `long_edge` or its own pixels, `atlas_fetch.atlas_sized`
  only shrinks the atlas plane, and `deformation.picture` stops at the
  fit image. The atlas under a section in ONE picture (`stacked`,
  `view_stack`) is drawn to the section's size (`framed_atlas(fill=True)`,
  `stack_pictures`).
  `view_stack` tiles are the opening size (or `view.resolution` at auto), shrunk
  until the sheet is at most `SHEET_MAX_LONG_EDGE` 2048. The old
  `OVERLAY_LONG_EDGE` 768 (the interactive loop's cap) is gone: the
  atlas-voxel rule had already put that canvas at ~450 px, so it now
  follows the later size like every other picture. Unchanged:
  `PREVIEW_LONG_EDGE` working renders, `calibrate`, the silhouette fit, the
  six stored numbers and every payload number, `search_position`, the
  spacing plot, the deformable fit grid (`FIT_LONG_EDGE`) and the image
  model's inputs (fit_affine/adjust_transforms numbers, the image-model
  input and the fit grid on M04 hashed identical before and after,
  2026-10-01). Below `auto` the sizes never appear in model-facing text.

## Tool consolidation (2026-09-15)

The default full-task toolbox has 15 tools (11 for interactive-only transform
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
historical spline; a `shear` left out keeps the current transform's
shear, an explicit value sets it (2026-10-03).

## Linear scope and saved spline compatibility (2026-09-15)

The linear agent handles order, position and affine alignment. Paired landmark
viewing, editing and warping were removed from the toolbox and prompt after the
full-stack comparison did not establish a nonlinear quality gain. The dedicated
linear landmark module and editable-point state are removed. Default full-task
and interactive-only tool counts are 15 and 11 (`orient_slices` joined the
transform task on 2026-09-29).

**The nonlinear task's job (2026-10-01).** A live Astra run with the image
model never called `fit_deformable`: the job line said "use the image model to
correct every section's placed atlas borders" and submit only wanted a trace,
so the deformation was optional. Now the job line is "give every section a
deformation onto the atlas, on top of its linear placement, with the section's
stain [and the borders the image model traces on it] as the evidence", two
Method lines ask it to let every region a section still has drive its fit
(exclude what is lost rather than include a few survivors, damaged sections
included; added 2026-10-03 after M11_B_08/C_08, Nash: "Astra knows, it was
just lazy") and to compare candidates, inspect each fit's borders against
internal anatomy (and the traced borders), apply the best and keep the linear
placement only where no fit improves on it, and `submit` refuses `MISSING_DEFORMATIONS`
(`job.missing_deformations`, part of `submit_errors`) until every section
holds a deformation at its current `linear_key` or a `keep_linear` record.
`fit_deformable(slices, keep_linear="reason")` is that record: no fit,
`SliceState.deformation = {"keep_linear": reason, "linear_key": ...}`, one
undo step, cleared by `clear_stale` like a fit, status row `keep_linear`. It
lives on the deformation field rather than a submit argument so it is
undoable, checkpointed, visible in `status`, exported with the results and
invalidated by a placement change. With an image model the trace gate
(`MISSING_IMAGE_CORRECTIONS`) is reported before the deformation gate.
`nonlinear.provider` `none` (`NonlinearSpec.uses_image_model` False; the
provider is validated against `providers.registry`) builds `grep_atlas` and
`fit_deformable` but no `trace_borders`, strips traced images from
`fit_deformable`'s docstring (`toolbox._STAIN_ONLY_DOC`) and refuses them
(`NO_IMAGE_MODEL`), requires no trace and drops every image-model line and
the base prompt from the job statement. The task-notes heading is
"Nonlinear deformation".

Local anatomical deformation remains the responsibility of the nonlinear workflow.
With an image model the `nonlinear` task also adds `trace_borders(id, prompt="")` through
the top-level `registration_tool` bridge. It uses the supplied linear placement
and the agent's per-slice edited copy of the base correction prompt.
The call runs in the background (`registration_tool.start_correction` prepares
it; `Job.settle_image_corrections` waits at submit and at session end) and
returns no images. The first image reply is retained in `SliceState.image_correction`. No atlas search,
replacement prompt, candidate selection or anatomical rejection is exposed.
The same task adds `grep_atlas(query, section="")` (`linear/atlas_grep.py`): a text-only
lookup of atlas regions by acronym, name substring or id, with ancestry, descendant
count and, for a positioned section, whether the region is in the atlas plane at its
placement. It is for choosing regions a later deformable fit should exclude.
It does not fit a deformation or modify `transform`.

**`fit_deformable` (2026-10-01, task `nonlinear`).** `fit_deformable(slices,
include=[], exclude=[], start="linear"|"current", fit_section="fit"|
"traced_borders"|"traced_lines", fit_atlas=""|"ara"|"borders"|"nissl",
[engine], stiffness="soft"|"medium"|"firm", candidates=[], keep_linear="",
view)` (`fit_section`/`fit_atlas` were `section_image`/`atlas_image` until
2026-10-03, named apart from the picture options; `Choice` and the stored
`steps` use the new names; a fit on the DEFAULT appearance also keys on
what that appearance is drawn from, `--preprocess` and the host's channel
blend, so a resumed or rerun job with other preprocessing does not reuse a
saved warp, 2026-10-03); `view` modes `borders` (default) / `ab`, atlas channels default
`[borders]`, `ara`/`nissl` blending that atlas image, warped, under the lines
(`deformation._blend_atlas`); `channels` and `deformation` do not apply, and
`keep_linear` refuses any fit setting, `view` or `candidates` with `BAD_ARGS`
and the `given` names. Defaults without a trace: the fit
appearance against `ara`, ANTs (when the user left the engine open and it is
installed), medium; the stain fit itself is ANTs local correlation (80 µm
window) + an edge channel + the automatic tissue/ventricle label channels
(Elastix: mutual information + edges, no label channels), chosen by eye in
the 2026-10-02 stain ceiling test (deformable `CLAUDE.md`). Every fit runs
on a fixed thread count with a fixed seed, so the same inputs give the same
warp. With a completed trace the agent chooses; the docstring
states that traced_borders + ANTs + medium is the recommended pairing
(`toolbox._RECOMMENDED_TRACED`, dropped where the engine is fixed to Elastix
or there is no image model). The 2026-10-01 ceiling test (deformable
`CLAUDE.md`) removed `detail` (fixed at standard), the `stiff` level, line
softening as a knob, and raw channels as section images: a channel is a fit
appearance (`preprocess` target `fit`; the docstring points there when
`spec.agent_preprocessing` is on), so an unknown fit section answers
`BAD_FIT_SECTION`. The stain against `borders` is refused (`BAD_ARGS`:
borders are for traced fit sections). `nissl` is described neutrally as a
Nissl-stained reference. Region entries (`include`, `exclude`, `regions`)
may name one side, `"CTX:left"` / `"CTX:right"` (`atlas.sides`): the
section's side as this tool's pictures draw it; `region_names` validates
them (`UNKNOWN_REGIONS` for a bad side, `NO_SIDES` on a sagittal stack) and
`region_overlap` refuses `CTX` with `CTX:left` but not `CTX:left` with
`CTX:right`.
`engine` exists only when `JobSpec.nonlinear.engine` is `"either"` (default);
`"ants"`/`"elastix"` build the same tool without it (`fit_deformable_fixed`,
renamed). ANTs missing answers `UNAVAILABLE` naming the extra. 2–4
`candidates` (each overriding stiffness/fit_section/fit_atlas/
[engine]; another key is refused by the strict check) PREVIEW and write nothing; one setting APPLIES it as one undo step,
reusing an identical cached or saved result (`cached`); the same key again
only re-draws (`written: false`). `include` -> the engine's `structures`
(restricted step + 300 um), `exclude` -> its `exclude`; `start="current"`
passes the applied record as `previous` (refused `NO_DEFORMATION` without
one). `traced_borders` is the ANTs label-map mode (`labels="model"`),
`traced_lines` lines vs borders; both need the trace at this placement
(`NO_TRACE`/`TRACE_STALE`). A trace still running is waited for
(`Job.wait_image_job`, one `deformation.TRACE_WAIT_S` 300 s deadline per
call; the result lands as `settle_image_corrections` lands it, checkpointed,
no undo step), and the reply adds one picture per traced section of its lines
on the stain (`traces` maps them); `TRACE_TIMEOUT` / `TRACE_FAILED` otherwise. Max 4 sections, 8 fits
per call. `SliceState.deformation` holds `record` (the record's folder,
relative to the job folder), `key`,
`linear_key`, `steps` (the chain for `current`), `summary`, `inverse_source`;
the record directory holds the composed field, its inverse, the parent steps
and `provenance` (section id, linear handoff metadata, inputs). Every tool is
wrapped by `toolbox._clears_stale_deformations` (the rule is
`Job.clear_stale_deformations`): after any call, a
deformation whose `linear_key` no longer matches (position, orientation,
cutting angles, transform) is cleared in the same undo step and the reply
carries `deformation_cleared`. `view_placement`/`set_positions` draw the
section resampled by the applied warp (`deformable.warp_section_image`, row
`deformation_drawn`) at the position it was fitted at, in every mode that
draws the section under its placement (`WARPED_PLACEMENT_MODES`: overlay,
checkerboard, outlines, section) unless `view.deformation` is `none`; a held
but unreadable or stale record says `deformation_drawn: false`.
`view_placement` is built whenever position, transform or nonlinear is on
(2026-10-03: Astra asked for a read-only view of the complete
registration). Records are written
at apply time under the results folder and referenced from the results JSON;
export adapters (ABBA, VisuAlign, BrainGlobe) are to read them, none exist.

Default task/tool counts stay
unchanged; `DEFAULT_TASKS` is separate from `ALL_TASKS`. Hosts may supply calibrated
`inputs.transforms` or resume saved linear transforms. With an image model, submit
checks correction completion and current geometry, then the deformations (above).
Exact artifacts and first
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
only a 500-to-250 image-count backstop (the transform-media stage cut was
removed 2026-10-03; see "Images stay" below).
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
- **Picture sizes are the host's level** (`image_resolution`, see Host
  controls): from 2026-09-09 to 2026-10-01 every picture was instead sized
  so one pixel was never finer than the atlas voxel, capped at 512 px
  (`atlas.render.model_long_edge`, deleted), which showed a 6 mm section at
  ~220-360 px everywhere, too small to judge a fit. The opening on M04
  (38 sections) since the strips (2026-10-03), as 32-px patches: no
  positions 5.0k / 10.2k / 15.6k at low / medium / high in 11 / 17 / 21
  images (section strips plus the atlas reference); host positions 4.4k /
  9.6k / 16.2k in 5 / 8 / 10 images. Before (one image per section and per
  atlas section, 81 images, positions or not): 4.5k / 8.8k / 13.1k. The
  extra patches are row padding (a strip row is as tall as its tallest
  tile, and anterior sections stand taller) and, with positions, the atlas
  drawn to the section's size; the per-image count fell 4-16x. Later pictures cost more than
  before (a fit_deformable panel 512 px instead of ~260). Atlas images are
  never upsampled past the plane (until 2026-09-10 they were upsampled to
  512, a quarter of run 19's input); `view_placement`'s default mode is
  `template`, the atlas alone on the section's canvas, because the section
  is already in the opening (the default was `side_by_side` until 2026-09-10,
  so every Astra compare re-sent the section). `physical_views(long_edge=
  None)` is canvas pixels for host-side use; every model-facing caller
  passes a size.
- **Images stay** (`adk/plugins.py WorkingSetImages`): every tool image is
  kept until 500 are live, then the oldest media-bearing calls are cut in
  ONE batch to 250 (`DEFAULT_MAX_IMAGES` / `DEFAULT_KEEP_IMAGES`, a cut
  result says "dropped from context"); the cut only moves forward and the
  opening strips are never touched. Measured on M11 at low effort, 2026-09-09:
  keep-all (run 6, killed at call 16) had median 0.10 mm / 30 of 36 within
  0.25 with positions written by call 15; newest-call-only (run 7, 21
  calls, submitted) 0.30 mm / 14 of 36, and its debrief said the "dropped
  from context" results made its comparisons unreliable. Cost: a kept
  image is 0.13x on every later call, so keep-all grows quadratically
  (2.4k -> 8.8k paid a call by call 15), but a batched run submits in ~21
  calls, where keep-all is ~16% of a window against newest-only's 14%.
  A normal run (opening 5-21 strips, ~50 compares, ~40 write pictures) never reaches
  the cut; it is the safety for a run that goes long; cutting old images
  re-reads everything after the cut once, so it must stay rare. The
  stage-boundary cut (every earlier tool image dropped at the first
  `fit_affine`/`adjust_transforms`) was removed 2026-10-03: in the
  no-image-model runs it discarded the channel strip and preprocess
  pictures both agents had chosen their stain from, and both complained.
  Industry practice, verified that day: Codex CLI never prunes images by
  age (whole images only, at compaction); Claude Code drops its oldest
  batch only when a request would pass the API's image-count or size
  limit. Astra takes 1,500 images a request; the backstop is a third.
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
has run after the last write (an undo, redo or reload that moves a position
counts as a write to it, 2026-10-03). The gates are the tool doors' only:
a script or library call on the job is never gated. Gates alone did not help Luna (run 10:
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
slice, damaged ones included, until no further improvement is possible with the available transforms,
keeping only changes that improve alignment; this is prompt guidance, not a
fixed-adjustment-count or submission-review hook.
The default Method no longer prescribes batching (2026-09-11): grouping work is the model's choice; batch-capable tools remain available. The opt-in cheap-model playbook is unchanged. Asked from its own run-3 trace, Astra said it skipped `compare_placement` (now `view_placement`) and `view_stack` by oversight, not wording, and asked for exactly this. Still out: rules of thumb, failure-mode warnings and region names (the same text runs against every BrainGlobe atlas, species and plane). Full-trace forensics found every major
benchmark failure tracking back to advice the harness injected; a per-slice
estimation worker that ate 82% of the wall-clock carried ~no signal and was
deleted; a landmark-tool pass for POSITION estimation benchmarked WORSE and was
deleted rather than kept behind a flag. The later in-plane paired-landmark
extension was also removed; nonlinear registration stays separate.

**One transform representation.** Elastix, silhouette, interactive:
every stored transform and every fit payload carries `physical` (the
knobs `adjust_transforms` takes, `shear` included since 2026-10-03, about a pivot in canvas fractions)
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
median written spacing), `MISSING_TRANSFORMS` (every section carries one,
damaged included) and, with task `nonlinear`, `MISSING_DEFORMATIONS` (a
deformation or `keep_linear` record at the current placement per section). `DAMAGED_REQUIRES_MANUAL_TRANSFORM` additionally rejects
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
step, so a batch undoes as one. The undo history (depth 50) is saved in the
job folder (`history/`, one file per step), so a resumed run starts from the
state as it stood and can still undo the steps before it (until 2026-10-03
the history was in memory only and a resume began with none).

## Ceilings worth knowing

- `fit_affine`'s silhouette method measures against the atlas plane at the
  stack's cutting angles (`atlas_fetch.atlas_mask` handed to
  `affine.silhouette_affine` as `atlas_mask_at`, 2026-09-10). Until then it
  measured against the FLAT section on a 13-degree brain and said so with
  `flat_atlas_fit`; Astra's run-19 debrief asked for exactly this.
- `physical` on a fit is the knobs about the canvas centre, `shear`
  included (0.10-0.16 on M05_D_08, not noise). Until 2026-10-03
  `adjust_transforms` had no shear knob, so a preview typed from a fit
  reproduced it only up to that shear; now the fit's knobs given back,
  shear included, reproduce it up to their rounding. The A/B view's B side
  is still drawn from the six stored numbers (`affine.denormalized_affine`),
  which are exact.
- The moments core has a 180-degree ambiguity — `(1,1)` and `(-1,-1)` are the
  same axes turned round — and settles it on silhouette IoU alone. On INTACT
  M05 sections the right one wins by 0.036-0.092 IoU; a template-correlation
  tie-break would help the cases where silhouette IoU alone is close, but that
  is measured, not built: it would move a benchmarked path (`silhouette_affine`
  is `nonlinear/quick_affine`'s too). The fit's own panel is what catches it;
  look at it. Damaged sections never reach this path: `fit_affine` refuses
  them outright unless regions are given.
- The moments core matches long axes. With regions on M04_D_08 (both
  hemispheres missing; tissue axis ratio 1.13, taller than wide) excluding
  Isocortex/OLF/ENT turned the section 90 degrees and excluding all of CTX
  180 degrees (kept atlas ratio 1.02), each flagged in the reply's note;
  `include=["MB"], exclude=["CTX"]` gave an upright fit at scales 1.17/1.22,
  close to the agent's own manual 1.02/1.18. The whole-outline fit (refused
  there as damaged) turned it -91 degrees.
- `run_deepslice` answers `UNAVAILABLE`; it is a seam, not a stub to fill in
  casually.
- The Elastix method (2026-10-03, `_local/runs/20261003_elastix_affine`, the
  eight ceiling-test sections from Astra's applied placements, judged by eye
  on borders drawn on the section): it never turned a section (largest turn
  from the start 5 degrees, median movement 0.08-0.31 mm), where the
  silhouette fit turned M04_A_01 (olfactory bulbs) 95 degrees and M04_D_08
  (round tissue, with exclusions) 179 degrees, upside down. Clearly better
  than Astra's placement on M04_A_01 (bulb and frontal outlines),
  M11_C_08 and M11_B_08 (midline and interior centred, the silhouette fit
  there inflated the atlas onto displaced flaps); about the same on
  M04_B_05, M11_D_05, M04_C_08 and M04_D_08; on M11_B_03 slightly worse at
  one lateral ventricle (tens of micrometres). Engine time under 1.2 s a
  section; a whole call ~1-6 s a section (grid render included), sequential.
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
automatic `fit_affine` tool (Elastix by default, silhouette as an option; the
former `transform.elastix` switch was removed 2026-10-03, and saved specs or
hosts that still send it load with the key ignored). The transform task's submit requirement still applies; a
host must enable at least one transform method when enabling that task.

## Claude host briefing

`prompt.run_facts` is shared with `mcp_server/prompt.py`; keep the factual
range, axis direction, protocol and calibration text identical across hosts.
Claude's statement does not reuse the ADK method/playbook. MCP opens saved ABBA
jobs through `api.abba_worker.prepare_linear`, exactly like `linear.run`,
and supplies the opening strips separately with `show_stack` pages
(`opening_pages`: `opening.opening_items` at `CLAUDE_IMAGE_LIMIT`, each
strip encoded by `adk/media.encode_jpeg` straight into an MCP image block,
paged under
`PAGE_BYTES`, a strip and its text kept together). No image
model is available through the Claude connector. The MCP tools are the same
functions with the same `view` and the same strict-argument rule
(`mcp_server.server.strict_arguments`; a nested object Claude Desktop sends
as a JSON string is parsed first); their plain pictures become image blocks
through `encode_jpeg` (`result_blocks`).

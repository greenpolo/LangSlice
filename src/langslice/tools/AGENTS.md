# LangSlice `tools/` — the verbs

Package guide for `src/langslice/tools/`. The repo-level `CLAUDE.md` holds the
project-wide rules. `AGENTS.md` here is a verbatim copy — edit one, mirror to
the other.

Every write to a stack, and every read a viewing tool makes, is a function
here. The agent tools, the MCP server, the agent CLI and scripts call the
same functions, so a tool and a script do exactly the same write and get the same
picture. Each agent tool's body is argument checking, ONE call here and the
wording; `catalog.py` lists which. The package also holds what every tool
body of a run shares: `run.py` (`Run`, the run behind a door: the job, the
workspace, the picture level, the image call), `pipeline.py` (`TOOL_STEPS`,
the steps every tool call goes through in order, the `ToolCall` they
carry, `declared_tool` and the `ToolBox`), `events.py` (each call's
`tool_start` / `tool_end` events and the job folder's event log),
`arguments.py` (the typed
argument shapes and the strictness check), `reply.py` (the reply byte
budget and paging) and `descriptions/` (every model-facing text: the
tools' descriptions, the job statement, the agent CLI's fixes; below).
A tool's body sits beside its operation as `<tool>_tool(run, **arguments)`,
taking the declared arguments as the model sent them (`run.shows(view)`
decides the picture): argument checks, the one operation call, the reply
and its picture (`positions.py`, `transforms.py`, `damage.py`, `masks.py`,
`notes.py`, `history.py` (`undo_tool`, `redo_tool`), `registrations.py`,
`submit.py`, `affine_fit.py`, `deformable.py`, `traces.py`, `exports.py`,
`landmarks.py`, `landmark_warp.py`, and the looking tools' `look.py`,
`zoom.py`, `images.py`, `atlas.py`, `status.py`, `files.py`);
`doors.tools.toolbox.TOOL_BODIES` wires their names, and `build_tools`
(which resolves the image model through the providers) stays in
`doors/tools/`.

## `descriptions/`: the model-facing text

To change what a tool's description says, open its family's module
(`looking.py`, `changing.py`, `fitting.py`, `landmarks.py`,
`bookkeeping.py`), edit that tool's `ToolDescription` and regenerate the
text snapshot (`LANGSLICE_UPDATE_SNAPSHOT=1`).

- `description.py` — the shapes. A `ToolDescription` is a `summary`, the
  `body` paragraphs (one that renders empty goes), one `Argument` per
  argument (its type, default and text; `needs`: the names it needs on, in
  needs syntax; `offered`: the run fact it exists in), `returns`,
  `prompt_heading` (`trace_borders`: the paragraph before the run's base
  image prompt) and `slots`: `${name}` passages a fact of the run words
  (`PerCap`: a door with or without a picture cap; `PerDoor`: the agent
  CLI's own wording; `Distance`: a distance at the job's brain scale).
  Texts are prose at no indentation with the model's line breaks; a
  trailing backslash joins a line too long for the source. A passage that
  names a tool or an option carries needs marks (`job/settings/needs.py`).
  `DescriptionFacts` is what of a run changes the text: `door` (`DOORS`),
  `auto` (the caller sizes pictures), `forced_view`, `positions`,
  `uncapped`, `prompt` (the base image prompt; empty without an image
  model), `switched_off`, `absent` and `plugins` (a plugin's text renders
  by both), `brain_scale`; `from_spec(spec, auto=, image_model=, door=,
  prompt=, uncapped=, brain_scale=)`.
- `render.py` — `declaration(name, facts)` (cached): a native tool's
  description rendered for the run (slots filled, marks rendered, the
  arguments it does not offer left out, `narrowed` dropping the enum values
  and section keys the job switches off) as `doc` (the docstring layout ADK
  and FastMCP read), `text` (below the summary, unindented: `TOOLS.md`),
  `arguments` (each argument's text on one line: the agent CLI's fix of an
  unknown argument) and the signature; a plugin tool from its stub's
  docstring and signature (`argument_texts` reads its `Args`). `declare`
  binds a tool body to its declaration (ADK's `tool_context` added when the
  body takes it), `summary`, `arguments_schema` (pydantic, unknown keys
  refused), `DESCRIPTIONS`, `FULL` (a caller without a job: every argument,
  the coronal base prompt for the OpenAI models).
- `statement.py` — the job statement and `SYSTEM_PROMPT`. `STATEMENT` is
  the whole statement with its slots; the named templates under it are its
  sentences (the job clause per task, the constraints, the method bullets,
  `DEFORMABLE_METHODS`), chosen by the run's `StatementFacts`
  (`from_run(spec, state, workspace, tool_names=, max_resolution=,
  auto=)`). `job_statement(facts, door=, opening=, notes=,
  image_model_off=)` is every door's text: the agent's is the statement and
  the user's notes (the opening pictures and `status_and_notes` follow as
  parts of the first message); MCP's and the CLI's add the door's opening
  paragraph, the image-model line, the notes and the status table.
- `run_facts.py` — the "Run facts:" bullets, one template each: the stack
  and atlas, what the run changes, the raw channels and section images, the
  atlas images, the picture size, the deformable engine.
- `door_text.py` — what the doors add: `opening_for_mcp`,
  `opening_for_cli` (the opening files and the commands, `long_verbs`),
  `IMAGE_MODEL_OFF`, `status_and_notes`, `user_notes_lines`, `read_notes`,
  `NOTES_KEY`, `image_model_state`.
- `fixes.py` — the agent CLI's fix sentence per refusal code.

## The rule

- An operation takes the `job.job.Job` (and the core
  `core.workspace.Workspace` where it reads the atlas, the section files or
  the render caches) plus plain arguments: ids, numbers, dicts.
- A write takes ONE undo step through the job (`before = job.snapshot()`,
  write, `job.commit(before)`: the step and the checkpoint), or none when
  nothing was written, and returns a plain record of what changed (a frozen
  dataclass with `touched`, or plain data). It runs under the job folder's
  write lock (`Job.writing`: lock, sync, apply, commit): the doors hold it
  around every verb except the long ones (`catalog.Verb.long`:
  `elastix_affine`, `fireants_syn`, `trace_borders`, and the scripting
  verb `export_maps`), which compute outside
  it from the state they read and take it themselves to apply, comparing
  each section's `stale.section_inputs` with the value they computed from:
  a changed section is that section's row `STALE_INPUT`
  (`stale.stale_refusal`; `trace_borders` raises it), the others apply.
  `set_transforms` takes it itself too. A read (`status.py`, `atlas.py`, `look.py`)
  writes nothing.
- Pictures are the core's, never drawn here: a read verb, or a write that
  shows its result, takes the call's core `core.pictures.options.DisplayOptions`
  and returns the plain PIL pictures `src/langslice/core/` draws (captions
  burned in, each noted for the job's saved views). Without options a write
  draws nothing (a script). The change
  tools' pictures of what they wrote are `look.show_result`'s, drawn by the
  tool door after the write.
- It never words anything for a model. Job rules (frozen
  properties, host damage marks, the spec's flip switch, the submit gates)
  are the job's and are applied here; the per-call section caps (`refusal.check_section_count`:
  the tool's own cap, its `Verb.limits`, and for the transform tools
  `Job.picture_cap` and the job's `transform.max_parallel`, each None for no
  cap; `TOO_MANY_SECTIONS`) are checked by each tool's body with its other
  arguments.
- A model is passed in, never chosen here: `traces.trace_borders`
  takes the image model (`core.fits.trace.request.ImageModel`: provider, model, `call`)
  the door resolved; the geometry fingerprint is the core's
  (`core.fits.frame.correction_fingerprint`).
- A write to a property the job froze (`JobSpec.frozen`, `job/settings/frozen.py`) is
  refused with the one code `FROZEN` (`ids`, `properties`, `message`):
  `refusal.check_frozen(job, ids, *properties)` raises it for a whole call,
  `refusal.section_refusal(job, id, *properties)` is one section's
  refusal in a per-section answer (`SECTION_BUSY` first).
- A refusal is the core's `Refused(RefusalCode.X, message, **facts)`
  (`core/refusal.py`), raised, nothing written; this package re-exports
  nothing of it. `Refused.payload()` is the `{"status": ..., "error": code,
  ...}` the tool door answers with (`status` "refused" for a gate, the
  opening-read gate's and the job's submit gates, which return theirs as a
  `Refused`, `Job.submit_errors`, that `submit` raises). A per-section
  answer is a list of typed rows, one per section (`affine_fit.AffineRow`,
  `deformable_rows.DeformableRow`, `warps.WarpApplied`, `masks.MaskMade`,
  `registrations.Restored`, `transforms` entries), each holding its
  `Refused` or what was done; the tool's body converts each row to its
  reply once (`affine_fit.affine_row_to_json`,
  `deformable_rows.deformable_row_to_json`; a refused row is
  `refusal.refused_row`: `{"id", "status", "error", ...facts}`), and a
  plugin context converts the rows it hands a plugin. A code is
  always a `RefusalCode` member (a fit's flag a `core.fits.flags.FitFlag`),
  never a string written out.
- A section is named by its filename (`Stack.resolve`: the filename,
  or its stem when no other section shares it; never a number). A name
  that is no section's is `refusal.unknown_sections(state, refs)`:
  `UNKNOWN_SLICE_IDS` with `unknown`, `filenames` and a `message` saying
  sections are named by filename (`core.stack.stack.unknown_sections`).
- Layer: core and job only. Never a door (`doors/`, the built-in agent
  `doors/agent/` included), a host (`hosts/`), a provider (a type named under
  `TYPE_CHECKING` excepted: `traces.py` takes an `ImageModel`), and never
  `google.*`, `litellm` or `openai`. import-linter's contracts
  (`pyproject.toml`, `tests/test_import_layers.py`) check the imports;
  `tests/test_core_imports.py` loads each module in a fresh interpreter and
  checks what it loads.

## Files

- `positions.py` — `position_sections(job, workspace, [{id, position_mm}],
  cutting_angles={pitch_deg, yaw_deg})`: both writes in ONE undo step
  (each value clamped into `workspace.position_range` and reported; new angles give every section the one plane, so a
  stack whose sections carried different angles is flattened, undo
  restoring them, and drop the render cache), then the stack is numbered by
  position (`order_by_position`: placed sections by increasing position take
  the places placed sections held, an unplaced one keeps its own;
  `renumber(order)`, indices only); returns `SectionsPositioned`
  (`written`, `clamped`, `unknown`, `cutting_angles`, `order`,
  `reordered`). Refused: `BAD_ARGS`, `FROZEN` (a section named has its
  position frozen, or angles are asked while any section has its cutting
  angles frozen: they are one plane for the stack; a task that is off
  freezes what it supplies); angles alone are allowed under frozen
  positions, positions under frozen angles.
- `damage.py` — `mark_damage(job, workspace, section, regions, note="",
  options=)`: the section's marked regions (checked against the atlas,
  normalized; `UNKNOWN_REGIONS`, `BAD_ARGS`, `NO_SIDES`, `UNKNOWN_SLICE_IDS`;
  `WRONG_SIDE_PAIR` and the other side refusals for a side the section's
  current placement does not have, checked before the write)
  and note, one undo step (the same mark again writes nothing); empty
  regions clear them and the agent's note. A section is damaged exactly
  when it has marked regions. A host's note (`inputs.damaged`, a note and
  no regions) is the section's `damage_note` alone: it stays first, the
  agent's note after it. With *options*, `core.sections.damage.damage_picture` (a
  picture that fails is `render_failed`, the write standing). Returns
  `DamageRegions`.
- `images.py` — `make_section_image(job, workspace, name, sections,
  given, delete=)`, `given` an `ImageRecipe` (`weights`, `clahe_clip`,
  `clahe_tiles`, `n4`, `denoise`, `contrast`, `gamma`, `invert`; the
  `make_image` tool, `make_image_tool`, normalizes the model's values first): a made section image (`core/sections/images.py`) for the
  sections named (empty: every section), each computed from its own images
  and saved as a file before ONE undo step sets the entries;
  `preprocessed` with every argument at its default writes the opening
  image again; `delete` removes the entries (`preprocessed` cannot be
  deleted). Refused: `BAD_ARGS` (the name: a raw channel's, `none`, `all`,
  a bad spelling; a bad value; an image made from itself), `UNKNOWN_SLICE_IDS`,
  `UNKNOWN_IMAGE` (a weighted image a section lacks; delete of a name no
  named section holds), `RENDER_FAILED`. Returns `ImagesMade` (`name`,
  `sections`, `deleted`, `in_force`: the image as `status` lists it,
  `images`: every name of the stack). The tool `make_image` adds a
  `section` picture of a spread of the sections (`look.show_result`; none
  where `look:section` is off, which also drops its `view`).
  `add_image(job, workspace, record,
  name, array, note)`: a script's image (`(H, W)` on the working grid in the
  file's orientation, `uint8` or 0-1 floats), source `{"script": note}`, one
  undo step; `BAD_ARGS`. `check_image(job, workspace, records, name)` (what a
  fit reads, `UNKNOWN_IMAGE`), `read`, `image_names`, `stack_images`.
- `masks.py` — `make_mask(job, workspace, name, sections, source=,
  section_image=, atlas_image=, sorting=, delete=)`: a section-side mask by
  name (`core/sections/masks.py`) for each section named (empty: every
  section), ONE undo step for the call. `em`: the EM sorting of the section
  image against the atlas image at each section's CURRENT registration;
  `fit`: the sorting the applied deformation saved (`NO_DEFORMATION`,
  `NO_SORTING`). *sorting* takes the five starting points (`SORTING_KEYS`),
  the shape every engine's `sorting` has; `method` and `artifact`, keys only
  STalign's takes, are refused by name (`BAD_ARGS`). A section that cannot get its mask is its row (`NO_POSITION`,
  `INVALID_LINEAR_PLACEMENT`, `NO_DEFORMATION`, `NO_SORTING`, `FIT_FAILED`);
  when none can, the first row's refusal. Refused: `UNKNOWN_SLICE_IDS`,
  `SECTION_BUSY`, `UNKNOWN_IMAGE`, `UNKNOWN_MASK` (delete), `BAD_ARGS`.
  Returns `MasksMade` (`rows`, one `MaskMade` per section: `id`, the
  state's `entry` or the section's `refusal`; `made`). `read` / `section_weights(job,
  record, name)` are what a fit reads: `"none"` gives None, a name gives the
  weights (`UNKNOWN_MASK`, `STALE_MASK` when the section's orientation, the
  grid digest `core.sections.masks.grid_digest`, differs from the one it was
  made on; a changed registration only marks it `earlier_registration`). The
  verb `make_mask`.
- `notes.py` — `add_note(job, text)`.
- `files.py` — read-only file access over the job folder for the native agent
  (`job.browse` keeps every path inside it; `BAD_PATH` otherwise, and
  `NOT_FOUND` for a path that does not exist, with `nearest`, the nearest
  folder above it that does, and `holds`, its names; both raised by
  `browse.resolve`):
  `list_folder(job, path=".", glob="")` -> `Listing`, `search_folder(job, query,
  path=".", glob="")` -> `Found`, `read_folder_file(job, path, offset=0, limit=400)` ->
  `FileText` (the tools `list_files_tool`, `search_files_tool`, `read_file_tool` flush the
  picture store first and reply with the `text`); each has a capped `text` reply that counts what it leaves out. A picture
  file is answered with its `views.jsonl` record (`job.views.records()`), never
  pixels, and a line written with its needs (`job.settings.needs.render`): the number to
  give `zoom`; where the job disables `zoom`, for a picture whose recipe is
  `look`'s, the `look` call that draws its view again (`_look_arguments`: mode,
  sections, positions, images, masks, atlas layers, warp, registrations, from the
  recipe's `args`; said to draw the stack as it is now, and unzoomed for a zoomed
  recipe), offered only while its mode is on; for any other picture nothing.
  `views.jsonl`, `state.json` (the run notes) and `job.json` read as text.
  `trace/` (the agent transcripts, `browse.HIDDEN`) is never listed, searched
  or read: `BAD_PATH`.
- `history.py` — `undo(job, steps=1)` / `redo(job, steps=1)`: that many
  steps under one hold of the job's lock (fewer when the history holds
  fewer); `Stepped` (`done`, `depth`, `step`: a `HistoryStep`, the sections
  and field groups the steps changed together, `describe_step`, `steps`: how
  many were taken).
- `registrations.py` — `restore_registration`'s operations over the job's
  registration store (`job/registrations.py`): `listing(job, section)` ->
  `Listed` (`held`, the number the section holds; `rows`: per registration
  `registration`, `by`, `at`, `transform` (`core.stack.status.transform_entry`: kind
  and knobs), `deformation` (`core.stack.status.deformation_entry`: method, steps,
  settings), `pictures` (`saved_pictures`: the numbers of the `views.jsonl`
  pictures drawn by the tool in `by` whose `registrations` give the section
  that number), `held`, and `not_here` for one whose deformation was fitted at
  another position or cutting angles; the registration the section holds is
  listed first when it never was; no state written); `restore(job,
  [{"id", "registration"}])` -> `RestoredMany` (`rows`, one `Restored` per
  section in call order: `registration`, `replaced`, `written` (False when
  the section holds it already), or that section's `refusal`; `written`): each section's
  orientation, transform and deformation set as its entry holds them, ONE undo
  step for the call, the other sections untouched. Refused per section:
  `UNKNOWN_REGISTRATION` (with `registrations`, the section's numbers),
  `SECTION_BUSY` / `FROZEN` (`check_frozen` with the properties it would
  change: `orientation`, `affine`, `deformation`), `REGISTRATION_ELSEWHERE`
  (its deformation's `linear_key` is not the one the section would have:
  another position or cutting angles), `DEFORMATION_UNREADABLE` (its record
  folder is gone), `BAD_ARGS` (not a number); refused whole, nothing written:
  `BAD_ARGS` (no entries, an entry not `{id, registration}`),
  `UNKNOWN_SLICE_IDS`, `DUPLICATE_SLICE_IDS`. `entry_for(job, section,
  number)` -> `(record, entry, held)`: a registration as it can sit on the
  section now (the refusals above), what `look`'s `registrations` draws. The
  verb `restore_registration`.
- `submit.py` — `submit(job, summary=, notes=, interval_breaks=,
  traces=, workspace=, gate=)`: first every piece of the job's
  background work is waited for (`Job.background.wait_all`: under the door's
  lock, a landing that needs it runs on this thread); with *traces* (the
  image model is in the run) the running traces are waited for and recorded
  (tracing is the agent's choice: no section needs a trace); *interval_breaks*
  names the sections after each gap by filename (`clean_breaks`, kept as
  corrected indices; `UNKNOWN_SLICE_IDS`); then the job's gates
  (`Job.submit_errors`; the deformation gate only where the host set
  `NonlinearSpec.require_deformation`), then the door's
  *gate*; then ONE undo step: the interval
  breaks, the order reversed to run the atlas way (noted), the notes and a
  `submit: <summary>` note, `submitted`; then every queued picture written
  (`job.views.flush`) and, with the workspace, the stack's exports
  (`exports.export_stack`: QuickNII, VisuAlign, `registration.json`, plugin
  layers, each completed trace reply; never the per-section maps; a failure
  is logged, the submit stands). Returns `Submitted` (`exported`, `interval_breaks`
  as filenames). Not a long verb: the door
  holds the job lock for the whole call, the exports included.
- `exports.py` — (the verb's body: `export_maps_tool`) `export_maps(job, workspace, ids=,
  full_resolution=)`: the job folder's derived files from the stack as it
  stands, no undo step (a long verb: the stack read under the job lock
  after a sync, each section's maps computed outside it and written under
  it only when the section's `stale.section_inputs` are unchanged, else
  `skipped` as `STALE_INPUT`; the exports and `registration.json` last,
  under the lock): per placed section the maps
  (`core.maps.section_maps.section_maps` through the applied deformation record, written
  by `job.public_files.write_section_maps`; a section without a position, or
  whose record or file cannot be read, is `skipped` with its reason), then
  `exports/quicknii.json` and `exports/visualign.json` for every placed
  section (`job.quint.job_export`, markers from
  `core.maps.section_maps.residual_markers`), then `registration.json`. With its maps,
  a section whose last trace holds a reply gets `exports/traces/<name>.npz`
  (`core.fits.trace.trace.reply_on_grid` on the maps' grid,
  written by `job.public_files.write_trace_reply`, kind `trace_reply`; one left
  by an earlier export is removed when the section has none now). A job that
  persists nothing writes nothing and lists what it would (`written`
  False). A job whose sections show plugin layers also gets
  `exports/plugins/<tool>/` (`job.plugin_store.export_layers`, kind
  `plugin_layer`). `UNKNOWN_SLICE_IDS`. Returns `Exported`. The only writer
  of the per-section maps. `export_stack(job, workspace)`: the exports and
  `registration.json` under the lock, and each placed section's completed
  trace reply on its working copy's grid (what `submit` writes).
  `section_maps(job, workspace, section, full_resolution=)`: one section's
  `core.maps.section_maps.SectionMaps` in memory, written nowhere (the library's
  `JobHandle.section_maps`); `UNKNOWN_SLICE_IDS`, `ValueError` for a
  section without a placement or an unreadable record or file.
  `applied_record(job, state, record)`: the section's applied deformation
  record, None when linear.
- `points.py` — section points and atlas points through a section's current
  registration, writing nothing (a script's `job.context.to_atlas_um`,
  `from_atlas_um`, `plane_distance_um`, `region_at`): `atlas_points(job,
  workspace, section, xy, frame="file"|"fit")` -> `AtlasPoints` (`um`, NaN
  outside the volume; `native`, `ids`, `inside`; `core.maps.section_maps.native_points`),
  `section_points(job, workspace, section, xyz_um, frame=)` -> `SectionPoints`
  (`xy`, `off_plane_um`: each point projected onto the plane, then
  `core.maps.section_maps.file_points`). Refused: `UNKNOWN_SLICE_IDS`,
  `NO_POSITION`, `INVALID_LINEAR_PLACEMENT`, `DEFORMATION_UNREADABLE`,
  `BAD_ARGS`.
- `transforms.py` — the stored transform: `transform_record` (knobs,
  `shear` optional (0), to the record), `same_transform`, `set_transforms(job,
  {id: record})` (one undo step for the batch; a frozen affine, or a transform that turns
  over a section whose orientation is frozen, refused whole, `FROZEN`;
  `reverses_orientation`: the determinant's sign differs from the sign of
  the transform it replaces, positive without one).
  The shear convention is `affine.decompose_affine`'s.
  `interactive_transform(job, workspace, sections)`: orientation
  (`flip`, `rotate_quarter` 0/90/180/270) and the knobs together, absolute
  values, a value left out keeps what the section has (identity without a
  transform, the stored pivot too; an entry's `pivot`, "canvas" or
  `[fx, fy]`, is the operation's, no tool takes one), ONE undo step for the
  call. A changed orientation keeps the knob values: the transform is
  rebuilt on the new orientation, not dropped, and the entry's `kept` and
  `message` say so; with no transform and no knob given, the orientation
  alone is set. A frozen affine is never rewritten (even under a changed
  orientation); a frozen orientation refuses a changed flip or quarter turn
  and a transform that would mirror the section. Per entry `FROZEN`
  (`properties` says which), `UNKNOWN_SLICE_IDS`, `BAD_ARGS`,
  `BAD_ROTATION`, `OPTION_OFF`, `NO_POSITION`, the stager's codes
  (`core.pictures.placement.stage`). Returns `Adjusted` (one `Adjustment` per entry:
  `refusal`, `staged`, `previous`, `transform`, `written`, `id`,
  `orientation`, `kept`); the verb `interactive_transform`.
- `affine_fit.py` — the Elastix affine: `fit_transform` (a fit's record,
  kind `elastix`), `elastix_affine(job, workspace, sections, restrict_to=(),
  atlas_image="template")` (any of the atlas's images; `BAD_ARGS` for no
  sections; the tool door refuses more than `Job.picture_cap` or `Job.transform_cap`, `TOO_MANY_SECTIONS`): per section `core.fits.affine.fit_elastix`
  refining its current placement against the atlas image, by the *restrict_to* regions only, its marked damage regions
  left out on their own (`core.sections.damage.exclusions`); the named sections;
  `FROZEN` per section (a frozen affine, or a fit that turns over a section
  whose orientation is frozen); every successful fit written as ONE undo step (fits
  computed outside the lock, `STALE_INPUT` for a section that changed);
  a tiny restricted region is `REGION_TOO_SMALL` before the engine runs;
  a fit that turns a section more than `LARGE_TURN_DEG` from its previous
  transform carries `turn_deg`, and one whose scale or shear changes past
  `SCALE_CHANGE_LIMITS` / `SHEAR_CHANGE_LIMIT` from it carries `flags`
  (`plausibility_flags`: `SCALE_CHANGED`, `SHEAR`); each fit runs inside
  `core.fits.engine_log.engine_output(job.engine_log)`. With `restrict_to`, each ok row carries
  `restrict_box`: the regions' bounding box on that section's canvas as
  `[x0, y0, x1, y1]` fractions (`fit_regions.region_box`; `core.pictures.canvas.zoom_box`'s
  zoom). Refused: `UNKNOWN_SLICE_IDS`, `BAD_ARGS`. Returns `AffineFit`
  (`rows`: an `AffineRow` per section, the core's `FittedAffine` or the
  refusal; `fitted`). `affine_row_to_json` is a row's reply: the knobs as
  `transform`, the turn and flags in words as `warning`. `elastix_fit(job, workspace, section, restrict_to=,
  exclude=, atlas_image=, mask=, marked=True)`: one Elastix affine from the
  section's current placement, written nowhere (`core.fits.affine.ElastixFit`;
  a script's `job.context.fit_elastix`): regions checked, the marked regions
  added to `exclude` unless `marked` is False, `mask` narrowing the section
  mask; `UNKNOWN_SLICE_IDS`, `NO_POSITION`, `BAD_ARGS`, `UNKNOWN_REGIONS`,
  `NO_SIDES`, `REGION_TOO_SMALL`, `FIT_FAILED`.
- `deformable.py` — `fit_deformable(job, workspace, records, choice,
  restrict_to=, overrides=)`: one deformation (a resolved
  `deformation.Choice`) on top of each section's applied deformation, or its
  linear placement when it holds none (the step's `start`: `current` or
  `linear`), each section's marked regions excluded
  (`core.sections.damage.exclusions`), applied as ONE undo step. It runs the fit (the fit grid, the
  image the fit reads, a traced fit section waiting for its running trace
  under one `TRACE_WAIT_S` deadline and adding the trace's regions to the
  call's (`traced_regions`; the row's `trace_regions`, and the record keeps
  the regions each fit ran with), the record cache, the engine). Per-section
  problems (`NO_POSITION`, `INVALID_LINEAR_PLACEMENT`, `DEFORMATION_UNREADABLE`,
  `NO_TRACE`/`TRACE_*`, `BAD_SETTINGS`, `FIT_FAILED`, `RECORD_WRITE_FAILED`,
  `STALE_INPUT`) are rows, not refusals. Returns `DeformableFit`: `rows`
  (call order, `deformable_rows.DeformableRow`: the `AppliedFit` or the
  refusal; `deformable_row_to_json` is its reply), `fitted` (each row with its fit and record), `traced` (per
  traced section, the image read and the trace's lines on the fit grid),
  `written`. `fit_pictures.draw_fits(workspace, done, options)` then draws
  each fit (the final borders on the image it read, titled with the
  settings) and each traced section's trace (`pictures`, `traces`,
  `render_failed`, each drawn row's `image_indexes`). Its callers:
  `fireants_syn` and `traces.land_trace`. The
  engine runs inside `core.fits.engine_log.engine_output(job.engine_log)` (what
  it prints goes to `logs/engine.log`); a fit's ok row (not a cached one)
  carries `gpu_rows(record.engine)`: `gpu_wait_s` from `GPU_WAIT_SHOWN_S` of
  waiting for the machine's GPU queue, `retried` in words after the GPU ran
  out of memory. An
  applied record is saved in the section's folder (`job.deformations`:
  `sections/<stem>/deformable/<key>`) and the section's `deformation` holds
  its path relative to the job folder; each step names its `method`
  (`fireants_syn`, or `trace_borders` for a traced fit; a step saved before
  steps named it has none, `job.registrations.made_by` reads it) and the
  choice's settings, a stain fit's `metric`, `edges` and `detail` too; a
  fit's `FIT_FAILED` row says the error's first line and appends the whole
  error to `logs/engine.log` (`core.fits.engine_log.failure_message`, as
  `elastix_affine`'s, `fit_record`'s and `make_mask`'s); the step of a traced fit records the
  trace it read (`trace`: the trace's artifact directory); a traced fit
  section reads the trace's artifacts under the job folder
  (`traced_lines(root=...)`).
  `fireants_syn(job, workspace, sections, restrict_to=, section_image=,
  atlas_image=, mask=, smoothness_um=, metric=, edges=, detail=)`: the FireANTs
  SyN fit, applied as ONE undo step: on a section image
  (`section_image`, default `preprocessed`) against one of the atlas's
  images (`core.atlas.images`; `nissl` resolves per atlas) with a named section
  mask, on top of the applied deformation
  (undo is how to start over); *restrict_to*
  checked against the atlas (`fit_regions.region_entries`); refused, nothing done:
  `ENGINE_MISSING` (`refuse_without_engine`: FireANTs and torch must import, `engine_ready`),
  `BAD_ARGS` (no sections, an unknown
  atlas image, metric, detail or smoothness, a malformed region list;
  more than `MAX_FIREANTS_SYN_SECTIONS` (20, every fitted section held in
  memory until the call's one write, `SECTION_CAP_REASON`) is the tool door's
  `TOO_MANY_SECTIONS`),
  `UNKNOWN_IMAGE`, `UNKNOWN_MASK`, `UNKNOWN_SLICE_IDS`, `UNKNOWN_REGIONS`, `NO_SIDES`. The verb `fireants_syn`;
  its tool `fireants_syn_tool` (the section cap, the reply, the picture).
  `nothing_fitted(rows)` is the top-level error of a fit tool that fitted
  nothing; `check_atlas_image` the atlas-image check both fit tools make.
- `deformable_rows.py` — `DeformableRow` (one section's answer from
  `fit_deformable`: its `settings`, a traced fit's `trace_regions`, the
  `AppliedFit` (engine numbers, summary, runtime, cache, GPU queue, the
  `inputs` / `estimated` blocks) or the `Refused`, `steps`,
  `image_indexes`) and `deformable_row_to_json`, its one conversion to a
  reply row (`fireants_syn`'s `results`, a landed trace's `row`).
- `warps.py` — `warp_from_map(job, workspace, section, native_xy, method=, step=,
  provenance=, weights=, start=, settings=, engine_settings=, engine_version=)`:
  a deformation made elsewhere (a tool plugin's), given as the
  full map from the fit grid's pixel centres to native atlas pixels, built as
  a record (`core.fits.deformation.record_from_map`: the residual over the current
  linear placement, a fixed-point inverse, the usual diagnostics; the engine's
  `native_parameters` and `version`), not written: `MappedWarp` (record, key,
  `linear_key`, step, numbers, `parent_key`). *start* `current`: made on top of
  the section's applied deformation, which becomes the record's `parent` (its
  `step` one more) and whose key the section must still hold. The step carries
  *settings* (the call's, by the tool's names). Refused: `UNKNOWN_SLICE_IDS`,
  `NO_POSITION`, `INVALID_LINEAR_PLACEMENT`, `BAD_FIELD`, `BAD_ARGS` (`current` without a
  deformation), `DEFORMATION_UNREADABLE`.
  `apply_warps(job, warps)`: writes them as ONE undo step under the job's
  lock, each replacing the section's deformation (step `start` `linear`) or,
  with a `parent_key`, keeping its steps before its own (`start` `current`);
  per row `FROZEN`, `STALE_INPUT` (the placement is not the one the warp was
  built on, or the section no longer holds its parent), `RECORD_WRITE_FAILED`.
  Returns `WarpsApplied` (a `WarpApplied` per warp: its numbers or its
  refusal).
  `fit_record(job, workspace, section, settings, start="linear"|"current",
  mask=, em=True, marked=True, section_image="preprocessed")`: one FireANTs
  stain fit with a `FitSettings` of the caller's own on that section image
  (`UNKNOWN_IMAGE`), in this process under the engine lock,
  written nowhere (a script's `job.context.fit_fireants`): the record;
  `mask` through `core.fits.deformable.fit.with_section_mask`, `em` False runs
  once, `marked` adds the section's marked regions to `exclude`. Refused:
  `UNKNOWN_SLICE_IDS`, `ENGINE_MISSING`, `BAD_ARGS`, `NO_POSITION`,
  `INVALID_LINEAR_PLACEMENT`, `FIT_FAILED`.

- `traces.py` — (the tool's body: `trace_borders_tool`, the run's image
  model and picture size) `trace_borders(job, workspace, ref, image_model=, prompt=,
  restrict_to=, options=, workers=)`: LangSlice's own
  nonlinear method packaged. It starts the section's image correction (below)
  and a piece of background work (`Job.background`, kind `TRACE_WORK`
  "trace_borders") and returns at once (`TraceStarted.work`, the work id; a
  section whose packaged trace still runs gets that work, `running`). When
  the reply lands, `land_trace` (the work's callable) fits the traced
  borders (`TRACE_FIT`: traced borders against the atlas borders, FireANTs,
  medium; on top of the applied deformation; by *restrict_to*'s regions only) with
  `fit_deformable` and applies it as its own undo step, its notice saying so
  and that undo removes it while it is the latest step, with the
  displacement, fold fraction and flags (`core.fits.deformation.flag_words`: each
  code once, a region code with its regions); with *options* the fit and the
  trace are drawn and saved with the work. A section whose
  `stale.section_inputs` (with the deformation) moved since the call, or
  whose trace was undone or made at another placement, gets nothing: a
  failed work whose notice starts `STALE_INPUT`; a failed image call is a
  failed work too. Refused before any image call: `ENGINE_MISSING`,
  `BAD_ARGS` / `UNKNOWN_REGIONS` / `NO_SIDES` (*restrict_to*), then the
  trace's own refusals. The trace itself: one section's image correction, showing
  the model only the chosen regions' borders, its marked regions excluded
  (`core.sections.damage.exclusions`; recorded on the result); the prompt sent is
  saved with each attempt. The section's current
  geometry is `core.fits.frame.correction_fingerprint`;
  `registration_tool.start_correction` prepares the edit for *image_model*
  (resolved by the door: the toolbox's `build_tools(image_model=...)`
  binding) and returns the record and the call to run. A packaged call
  whose saved trace at this placement and region choice has already landed
  (a step of the section's applied deformation records that trace) starts
  nothing and fits nothing again (`TraceStarted.landed`). A call already
  running at that geometry is not started again (`running`); otherwise the
  call runs on the job's image executor and the section's
  `border_trace` record is written as one undo step when it changed.
  Prepared outside the lock, started and written under it. Refused:
  `UNKNOWN_SLICE_IDS`, `SECTION_BUSY` (other background work runs on the
  section), `FROZEN` (the section's deformation is frozen),
  `NO_POSITION`, `INVALID_LINEAR_PLACEMENT` (no usable transform; a refusal
  raised while the trace is prepared, such as `NO_SIDES` or `BAD_ARGS` for
  regions that leave nothing on the plane, keeps its own code),
  `BORDER_TRACE_IO_ERROR`, `STALE_INPUT` (the geometry changed while the
  call was prepared). Returns `TraceStarted`.
- `fit_regions.py` — `region_entries(workspace, state, entries)`: a call's
  `restrict_to` (or `exclude`) checked against the atlas and normalized
  (`BAD_ARGS`, `UNKNOWN_REGIONS`, `NO_SIDES`), for every fit tool and a
  script's fit; `region_box` (the regions' bounding box on a section's
  canvas, with `BOX_MARGIN`) and `zoom_window` (that box as a restricted
  fit's picture zoom, only past `MIN_ZOOM_GAIN`).
- `stale.py` — `section_inputs(state, record, deformation=, trace=)`: a
  digest of what a fit of the section reads (its linear placement:
  position, plane, angles, flip, rotation, transform; its made images'
  digests, so an image made again while a long verb runs makes it stale;
  damage: marked regions and a mark naming none; with `deformation` the applied deformation's key, with `trace`
  its image correction); `stale_refusal()`, `STALE_INPUT`.
- `atlas.py` — `search_regions(job, workspace, query="", section="", points=())` (the tool `grep_atlas`): with `points` it adds `region_at_points` (the region a saved picture drew at each `{picture, x, y}`, from its `labels.tif`, an atlas picture's section plane or a crop's source; at most `MAX_POINTS`; `NO_ATLAS_IN_PICTURE`); the region
  hierarchy searched like text (`core/atlas/search.py`), with `in_section`
  per row for a placed section. `draw_regions(job, workspace, regions,
  positions_mm)` (the tool `grep_atlas_view`): the atlas template at each position (clamped into the
  range) at the stack's view angles with the named regions' borders
  highlighted (`"CTX:left"` allowed; no other borders), saved like a
  `look` picture (so it can be zoomed: it registers its recipe renderer
  `grep_atlas_view` with `core.pictures.zoom.RENDERERS`); returns `RegionsView`
  (`pictures`, `entries`, `not_shown`, `regions`, `regions_not_in_plane`,
  `clamped`). Refused: `BAD_ARGS`, `NO_STRUCTURES`, `UNKNOWN_REGIONS`,
  `NO_SIDES`.
- `look.py` — `look_tool(run, mode, sections, images, ...)`, the tool:
  it builds the `core.pictures.look.LookRequest` from the model's
  arguments (section filenames as named, `long_edge` the `resolution` asked
  for, held by `requested_edge`: read only at image resolution `auto`), and
  `look_pictures(job, workspace, request, registrations=, sheet_edge=)`
  checks it and draws; `zoom.py` — `zoom_tool(run, box, picture, region=)`,
  the tool, over `zoom_picture(job, workspace, box=(), picture=None,
  region="")`. Both save their own pictures (`save_pictures`: `job.views.save` with the note's
  recipe and caption), so each has a picture number at once: the reply
  (`Looked`, `Zoomed`) carries `pictures` (PIL) and `entries`
  `[{"id", "caption"}]`, and the door must not save them again. `look` shows
  every picture `core.pictures.look.look` draws, or the first `Job.picture_cap` of them
  (4 at the agent tools and MCP, None at the CLI and the library, set by
  `build_tools`; `look.split_shown`); the rest are `not_shown`, each with the
  arguments that get it. `registrations` (`{filename: number}`, modes
  `section` and `overlay`): each named section drawn under that registration
  of its own (`registrations.entry_for`) on a copy of the state, writing
  nothing; the request's `registrations` words the caption, the picture's note
  `extra["registrations"]` is what `views.jsonl` records, and with no
  `sections` only those sections are drawn (`BAD_ARGS`: another mode, a named
  section not among `sections`). `sheet_edge` (the door's viewer edge; None:
  no sheet): every picture drawn saved as a tile (`ViewStore.save(tiles=True)`),
  then tiled into sheets of at most `core.pictures.sheet.MAX_TILES`
  (`core.pictures.sheet.tile`), saved in one call as pictures of mode `sheet`
  (`SHEET_MODE`) whose recipe lists the tiles; `Looked.pictures` holds the
  sheets, `entries` the tiles (each with its `sheet`), `sheets` the sheets'
  numbers, captions and tiles. `zoom` redraws a box
  of picture `picture` (default: `views.latest(exclude_tool="zoom")`) through
  `core.pictures.zoom.redraw` and saves it as tool `zoom`; `stale` when the stack has
  changed since. On a sheet (`_zoom_sheet`) the box is read on the tile that
  holds most of it (`core.pictures.sheet.tile_under`, `to_tile`) and that tile's picture
  is zoomed (`Zoomed.sheet`, `box`); `region` there is `BAD_REGION_PICTURE`, a
  box over no tile `EMPTY_BOX`. Instead of a box, `region` selects exactly one atlas region
  on a saved overlay, cropped to its bounds with only its border drawn;
  `BAD_REGION_PICTURE` on other modes and `REGION_NOT_IN_PLANE` when absent.
  A picture with no recipe or whose sections are gone is cropped
  from its saved image (`redrawn` False) and its crop recipe names its
  `source` picture, so a crop of a crop is cut from the original. Refused:
  `NO_PICTURE`, `UNKNOWN_PICTURE`, `PICTURE_NOT_SAVED` (no saved image to
  crop, or a dry run's unknown number), `BAD_BOX`, `EMPTY_BOX`, and the look
  refusals (`UNKNOWN_SLICE_IDS`, `UNKNOWN_MODE`, `UNKNOWN_IMAGE`,
  `MIXED_IMAGES`, `TOO_MANY_CHANNELS`, `UNKNOWN_LAYER`,
  `NO_POSITION`, `BAD_WARP`, `BAD_ARGS`, `UNKNOWN_PLUGIN_LAYER`). `look`'s
  `layers` names the layers tool plugins left (`"all"`: every one) to
  draw on `section` and `overlay` pictures (`BAD_ARGS` in another mode);
  `Looked.layers` says per picture which were drawn. `show_result(job, workspace, tool,
  mode, sections, positions_mm=, zooms=, masks=, images=)`: the picture a change tool shows
  of what it wrote, drawn as `look` draws `overlay` (one per section, each
  zoomed to its window) or `positioning`, saved under the change tool's
  name (numbered, captioned, zoomable); at most `Job.picture_cap` shown (None:
  all), the rest `not_shown` with the `look` call that draws them, a failed picture
  `failed` (the write stands).
- `status.py` — `status_tool(run)`, the tool (the rows, `raw_channels`,
  `section_images`, `background_running`, `can_change`), and
  `stack_status(job)` (`StackStatus`: the status rows (each with its
  `transform`, kind and knobs, and `deformation`, method, steps and settings:
  `core.stack.status`), angles,
  breaks as the filenames of the sections after each gap,
  `core.stack.stack.break_ids`; a row with properties the user froze carries
  `frozen: [...]`, one the user gave a damage note `damage_by_user: true`,
  one holding a listed registration its number, `registration`; every write's
  `changed` rows are these rows).
- `catalog.py` — `VERBS`: every verb (agent tool) name -> `Verb(name,
  function, kind "read"/"write", group "Common"/"Positioning"/"Linear"/
  "Nonlinear", tool, limits)`, in the order every door lists them; `tool` is
  the verb's `job.settings.tool_table.ToolDef` (`when`, `long`, `scripting`,
  `image_model`, `options`, `essential`, `needs`: the core's list of
  what a spec can switch, which the registry binds to the operations);
  `enabled(spec, scripting=, image_model=)`: the verbs a run of the
  spec has, less the ones its `disabled` list names. `verbs_for(spec)` is the
  per-spec table: `VERBS` plus one `Verb` per tool of the plugins in
  `spec.plugins` (its function the plugin's handler; `core/plugins.py`,
  `docs/developers/plugins.md`), which `enabled(spec)` reads;
  `VERBS` itself stays the native table (and without plugins `verbs_for` is
  `VERBS`). The gating (`when`):
  `look`, `zoom`, `make_image`, `grep_atlas`, `grep_atlas_view`,
  `status`, the job-folder tools (`list_files`, `search_files`, `read_file`),
  `note`, `undo`, `redo`, `mark_damage` and `submit` in every run;
  `position_sections` with the `position` task or the stack's cutting angles
  left to the agent (`transform.angles`, `positions_on`);
  `interactive_transform` and `elastix_affine` with `transform`; `fireants_syn`
  with `nonlinear`; `make_mask` and `restore_registration` with `transform` or
  `nonlinear`; `trace_borders` with `nonlinear` and an image model (a
  task whose properties are all frozen on every section is not on,
  `JobSpec.has`; a tool that writes only properties frozen on every section,
  `ToolDef.writes`, is off: `JobSpec.switched_off`). A
  tool or option the job disables (`JobSpec.disabled`; `look:<mode>` and
  `interactive_transform:flip` and `:rotate` are the options) is off in every
  run: the operations refuse a disabled `look` mode, a changed quarter turn
  and a changed flip (`OPTION_OFF`; `look`'s names the mode in `option`,
  `interactive_transform`'s row names the argument in its `message`).
  `image_model` False leaves out the verbs that call the image model, for a
  door that cannot reach it (MCP with none connected). A `scripting` verb
  (`export_maps`, a "read": it changes no state) is the agent CLI's and the
  library's only (`build_tools(scripting=True)`), never offered to a model.
  `limits`: the most one
  call takes at every door, by what it counts (`fireants_syn` takes at most
  `MAX_FIREANTS_SYN_SECTIONS` `sections`; a plugin's own); the reference card
  lists them, and `check_section_count`'s `TOO_MANY_SECTIONS` says why in its
  `message`. The picture and transform-section cap of the ADK and MCP doors is
  the job's `picture_cap`, not a limit here. Every door is built from it: `build_tools`
  makes the tools `enabled(spec)` names (the ADK and MCP doors), the MCP
  door's `readOnlyHint` is `kind == "read"`, and the agent CLI
  (`langslice-job ops`, `schema`, `job FOLDER VERB`), the library's job
  methods and the job folder's reference card list it
  (`src/langslice/doors/`).

The pictures are drawn by the core (`src/langslice/core/`, its own
`CLAUDE.md`), and the job saves every one a door shows, with its layers, in
the job folder through one hook (`Job.views.shown`, `src/langslice/job/`).

## Landmark verbs

`landmarks.py`: `set_landmarks(job, workspace, section, pairs=, delete=)`
(returns `LandmarksEdited`, whose `reply()` is the tool's answer) atomically
adds/updates/deletes stable corresponding pairs, ONE undo step for the batch,
and applies no deformation. It resolves the exact saved picture recipe and
explicit atlas/section roles; it refuses captions, gutters, out-of-image
coordinates, missing geometry and stale atlas planes, and, before writing, an
atlas endpoint outside the atlas (`_refuse_outside`: `ATLAS_ENDPOINT_OUTSIDE`,
`pairs` the call's labels, the message giving each the nearest point inside the
atlas at least `INSIDE_MARGIN_PX` from its outline, in the numbers the
endpoint was given in). Section endpoints are
original-image edge fractions; atlas endpoints are native plane pixels.
`landmark_warp.py`: `landmark_triangulation(job, workspace, section)` (returns
`LandmarksApplied`, whose `reply()` is the tool's answer) explicitly applies the full
set, at least three pairs, on its written linear placement. The first application
composes onto the current deformation via its stored inverse. Consecutive
landmark applications replace the landmark method against that original
baseline (`current.parent`), retaining its key and steps in state; a different
intervening method establishes a new baseline. The mesh is the pairs and the
four fixed corners only, as VisuAlign builds it (vertex order: corners, pairs;
`native_parameters.pair_count`); nothing is added to the pairs, so tissue
that should keep its place takes a pair whose endpoints are one point (its
target is its source, the pair targets being in the baseline's section
coordinates). The revision digests the pairs with the mesh's kind, so a warp an
earlier version saved with automatic holds is built again from the same
pairs; its hold fields (`landmark_holds`, `~K` fold vertices) are read and
ignored. The VisuAlign mesh supplies dense
forward and same-connectivity inverse fields to the usual record composition
and diagnostic machinery without running an engine. Unreadable current records,
missing inverses, repeated section points, zero-area source triangles (named by
pair id) and stale atlas endpoints (named by pair id) are refused. Folded
triangles are applied and reported (`folds`: area fraction, pair ids; stored as
`landmark_folds` in the deformation reference, which also holds
`landmark_pairs`, the pair-to-geometry digest `applied` is read from). A
refused endpoint says the pair (call index and id), role, picture, x,y and the
picture's content box. `set_landmarks` replies with one compact record per pair
(`id`, `atlas`, `section`, `state`: applied/moved/pending/stale) and counts and
a `warp` block (the pairs the current warp was built from, those not in it, those
deleted but still shaping it, and that `landmark_triangulation` must run again),
and with no pairs and no delete writes nothing; the apply reply lists
`built_from` and says which displacement figure is the whole deformation; `delete=["all"]` clears the section.
`history.undo/redo` report `step` (`describe_step`: the sections touched
and the changed field groups). `look.show_result` takes
`resolution`; `zoom.zoom_picture`'s entries carry `content_box` and `covers` (the whole
picture's pixels a zoom shows). Every zoom carries a coordinate grid
(`core/pictures/grid.py`) labelled in the whole picture's pixels; on a zoom,
`zoom`'s box and `set_landmarks` endpoints are read in those numbers
(`core.geometry.pixel_frames.source_frame`), so they are the same on the zoom and on
its source picture. Both verbs belong to
Nonlinear. Both refuse a section whose `deformation` is frozen (`FROZEN`,
the whole call; `job/settings/tool_table.py` offers neither tool when it is frozen on every
section, and they need each other). Pair edits leave the current registration applied until another
explicit application or undo. An unchanged application is a no-op.


Every ok row of `fireants_syn` and `elastix_affine` carries `inputs`
(`core.fits.inputs.InputsBlock`: the section image `{name, kind, file, digest}`, atlas image asked
and resolved, mask, atlas-side damage, `restrict_to`, brightness) and
`estimated`; the result's `settings` is the call; `elastix_affine` refuses a
section that holds a deformation (`HAS_DEFORMATION`, that section's row).
`elastix_affine`'s rows keep the knobs as `physical` here; the tool door
answers them under `transform`, as every reply carrying the knobs does.

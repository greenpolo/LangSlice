# LangSlice `ops/` — the verbs

Package guide for `src/langslice/ops/`. The repo-level `CLAUDE.md` holds the
project-wide rules. `AGENTS.md` here is a verbatim copy — edit one, mirror to
the other.

Every write to a stack, and every read a viewing tool makes, is a function
here. The agent tools, the MCP server, the agent CLI and scripts call the
same functions, so a tool and a script do exactly the same write and get the same
picture. Each agent tool's body is argument checking, ONE call here and the
wording; `registry.py` lists which.

## The rule

- An operation takes the `job.job.Job` (and the core
  `core.workspace.Workspace` where it reads the atlas, the section files or
  the render caches) plus plain arguments: ids, numbers, dicts.
- A write takes ONE undo step through the job (`before = job.snapshot()`,
  write, `job.commit(before)`: the step and the checkpoint), or none when
  nothing was written, and returns a plain record of what changed (a frozen
  dataclass with `touched`, or plain data). It runs under the job folder's
  write lock (`Job.writing`: lock, sync, apply, commit): the doors hold it
  around every verb except the long ones (`registry.Verb.long`:
  `elastix_affine`, `ants_syn`, `trace_borders`, and the scripting
  verbs `trace_from_atlas` and `export_maps`), which compute outside
  it from the state they read and take it themselves to apply, comparing
  each section's `inputs.section_inputs` with the value they computed from:
  a changed section is that section's row `STALE_INPUT`
  (`inputs.stale_row`; `trace_borders` raises it, `trace_from_atlas`
  compares the geometry fingerprint), the others apply.
  `set_transforms` takes it itself too. A read (`views.py`, `atlas.py`, `look.py`)
  writes nothing.
- Pictures are the core's, never drawn here: a read verb, or a write that
  shows its result, takes the call's core `core.display.DisplayOptions`
  and returns the plain PIL pictures `src/langslice/core/` draws (captions
  burned in, each noted for the job's saved views). Without options a write
  draws nothing (a script). Where a picture must exist before the write
  (`set_preprocessed`: the BEFORE picture), it is drawn first. The change
  tools' pictures of what they wrote are `look.show_result`'s, drawn by the
  tool door after the write.
- It never words anything for a model and never applies the
  look-before-commit gates (`compared`/`reviewed`): those are the tool
  door's (`submit` takes the door's gate as a callable). Job rules (locked
  sections, host damage marks, the spec's flip switch, the submit gates)
  are the job's and are applied here; the transform cap (`Job.over_cap`)
  is checked by the tool door with its other arguments.
- A model is passed in, never chosen here: `traces.trace_borders` (and
  `traces.trace_from_atlas`) takes the image model (`providers.registry.ImageModel`: provider, model, `call`)
  the door resolved; the geometry fingerprint is the core's
  (`core.handoff.correction_fingerprint`).
- A refusal is `refusal.Refused(code, status="error", **facts)`, nothing
  written; `Refused.payload()` is the `{"status": ..., "error": code, ...}`
  every door answers with (`status` "refused" for a gate, `Refused.of`
  turns a job gate's payload into one).
- A section is named by its filename (`StackState.resolve`: the filename,
  or its stem when no other section shares it; never a number). A name
  that is no section's is `refusal.unknown_sections(state, refs)`:
  `UNKNOWN_SLICE_IDS` with `unknown`, `filenames` and a `message` saying
  sections are named by filename (`core.state.unknown_sections`).
- Layer: core and job only. Never a door (`doors/`), the agent driver
  (`agent/`), a host (`hosts/`), a provider (a type named under
  `TYPE_CHECKING` excepted: `traces.py` takes an `ImageModel`), and never
  `google.*`, `litellm` or `openai`. import-linter's contracts
  (`pyproject.toml`, `tests/test_import_layers.py`) check the imports;
  `tests/test_core_imports.py` loads each module in a fresh interpreter and
  checks what it loads.

## Files

- `positions.py` — `position_sections(job, workspace, [{id, position_mm}],
  cutting_angles={pitch_deg, yaw_deg})`: both writes in ONE undo step
  (each value clamped into `workspace.position_range` and reported; a
  written position is the writer's own, so it drops the section's
  starting-position mark, `position_source` "default", even when the value
  is the starting one; new angles give every section the one plane, so a
  stack whose sections carried different angles is flattened, undo
  restoring them, and drop the render cache), then the stack is numbered by
  position (`order_by_position`: placed sections by increasing position take
  the places placed sections held, an unplaced one keeps its own;
  `renumber(order)`, indices only); returns `SectionsPositioned`
  (`written`, `clamped`, `unknown`, `cutting_angles`, `order`,
  `reordered`). Refused: `BAD_ARGS`, `POSITIONS_SUPPLIED` (the spec has no
  `position` task: the host supplied them), `ANGLES_SUPPLIED` (neither the
  `position` task nor `transform.angles`); angles alone are allowed under
  supplied positions when `transform.angles` is on.
- `damage.py` — `mark_damage(job, workspace, section, regions, note="",
  options=)`: the section's marked regions (checked against the atlas,
  normalized; `UNKNOWN_REGIONS`, `BAD_ARGS`, `NO_SIDES`, `UNKNOWN_SLICE_IDS`)
  and note, one undo step (the same mark again writes nothing); empty
  regions clear them and the agent's note. A section is damaged exactly
  when it has marked regions. A host's note (`inputs.damaged`, a note and
  no regions) is the section's `damage_note` alone: it stays first, the
  agent's note after it. With *options*, `core.damage.damage_picture` (a
  picture that fails is `render_failed`, the write standing). Returns
  `DamageRegions`.
- `appearance.py` — `set_appearance(job, targets, ids, settings)`,
  `planned_settings` (what a write would leave, written nowhere) and
  `preprocess(job, workspace, targets, ids, settings, shown=, options=)`:
  per shown section the first target's BEFORE and AFTER pictures (uncaptioned,
  `BeforeAfter`), drawn before the write; a failure refuses the call
  (`RENDER_FAILED`), nothing written. The target is `preprocessed` (or its
  older name `fit`): the preprocessed channel every fit and the image model
  read, `core/appearance.py`. `set_preprocessed(job, workspace, ids,
  channel_weights=, clahe_clip=, clahe_tiles=, n4=, denoise=, reset=,
  shown=, options=)`: the preprocessed channel's recipe, checked first
  (`UNKNOWN_SLICE_IDS`, `BAD_ARGS`, `CHANNEL_COUNT_MISMATCH`, `UNAVAILABLE`
  without antspyx for N4 or denoising), then `preprocess`'s before and after
  pictures and write. `set_channel_properties(job, workspace, channel,
  contrast_limits=, gamma=, colormap=, reset=)`: a raw channel's display
  properties, stack-wide (`core/channels.py`; contrast limits in file
  intensities; an argument left None keeps its value; `UNKNOWN_CHANNEL`,
  `BAD_ARGS`; no undo step when nothing changed); returns
  `ChannelPropertiesSet` (the properties in force, the sections with the
  channel, its sample type, range and 1st/99.5th percentiles). The verbs
  `set_channel_properties` and `set_preprocessed_channel_properties`.
- `notes.py` — `add_note(job, text)`.
- `files.py` — read-only file access over the job folder for the native agent
  (`job.browse` keeps every path inside it; `BAD_PATH` otherwise):
  `list_files(job, path=".", pattern="")` -> `Listing`, `search_files(job, query,
  path=".", glob="")` -> `Found`, `read_file(job, path, offset=0, limit=400)` ->
  `FileText`; each has a capped `text` reply that counts what it leaves out. A picture
  file is answered with its `views.jsonl` record (`job.views.records()`) and the
  number to give `zoom`, never pixels; `views.jsonl`, `state.json`
  (the run notes) and `job.json` read as text.
- `history.py` — `undo(job)` / `redo(job)`: `Stepped` (`done`, `moved`: the
  sections whose position the step changed, which the tool door's gates
  forget, `depth`); `moved_positions(before, state)`.
- `submit.py` — `submit(job, summary=, notes=, interval_breaks=,
  left_linear=, traces=, workspace=, gate=)`: first every piece of the job's
  background work is waited for (`Job.background.wait_all`: under the door's
  lock, a landing that needs it runs on this thread); with *traces* (the
  image model is in the run) the running traces are waited for and recorded
  (tracing is the agent's choice: no section needs a trace); *interval_breaks*
  names the sections after each gap by filename (`clean_breaks`, kept as
  corrected indices; `UNKNOWN_SLICE_IDS`); *left_linear*
  (`[{id, reason}]`, `clean_left_linear`: `BAD_ARGS` for a malformed list, a
  section named twice or a run without Nonlinear, `UNKNOWN_SLICE_IDS`,
  `DEFORMATION_REQUIRED` when the host set
  `NonlinearSpec.require_deformation`, `HAS_DEFORMATION` for a section
  carrying one at its placement); then the job's gates
  (`Job.submit_errors`, *left_linear* counted as covered), then the door's
  *gate*; then ONE undo step: each *left_linear* section's "linear placement
  stands" record (`{"keep_linear": reason, "linear_key"}` on its
  `deformation`, cleared by a placement change like an applied fit), the interval
  breaks, the order reversed to run the atlas way (noted), the notes and a
  `submit: <summary>` note, `submitted`; then every queued picture written
  (`job.views.flush`) and, with the workspace, every placed section's maps
  and the exports (`exports.export_maps`; a failure is logged, the submit
  stands). Returns `Submitted` (`exported`, `left_linear`, `interval_breaks`
  as filenames). Not a long verb: the door
  holds the job lock for the whole call, the maps included.
- `exports.py` — `export_maps(job, workspace, ids=,
  full_resolution=)`: the job folder's derived files from the stack as it
  stands, no undo step (a long verb: the stack read under the job lock
  after a sync, each section's maps computed outside it and written under
  it only when the section's `inputs.section_inputs` are unchanged, else
  `skipped` as `STALE_INPUT`; the exports and `registration.json` last,
  under the lock): per placed section the maps
  (`core.maps.section_maps` through the applied deformation record, written
  by `job.formats.write_section_maps`; a section without a position, or
  whose record or file cannot be read, is `skipped` with its reason), then
  `exports/quicknii.json` and `exports/visualign.json` for every placed
  section (`job.quint.job_export`, markers from
  `core.maps.residual_markers`), then `registration.json`. A job that
  persists nothing writes nothing and lists what it would (`written`
  False). `UNKNOWN_SLICE_IDS`. Returns `Exported`.
- `transforms.py` — the stored transform: `transform_record` (knobs,
  `shear` optional (0), to the record), `same_transform`, `fit_transform`
  (an Elastix affine fit's record, kind `elastix`), `set_transforms(job,
  {id: record})` (one undo step for the batch; locked sections refused).
  The shear convention is `affine.decompose_affine`'s.
  `interactive_transform(job, workspace, sections)`: orientation
  (`flip`, `rotate_quarter` 0/90/180/270) and the knobs together, absolute
  values, a value left out keeps what the section has (identity without a
  transform, the stored pivot too; an entry's `pivot`, "canvas", "tissue" or
  `[fx, fy]`, is the operation's, no tool takes one), ONE undo step for the
  call. A changed orientation keeps the knob values: the transform is
  rebuilt on the new orientation, not dropped, and the entry's `kept` and
  `message` say so; with no transform and no knob given, the orientation
  alone is set. Per entry `LOCKED`, `UNKNOWN_SLICE_IDS`, `BAD_ARGS`,
  `BAD_ROTATION`, `FLIP_DISABLED`, `NO_POSITION`, the stager's codes
  (`core.placement.stage`). Returns `Adjusted` (one `Adjustment` per entry:
  `error`, `staged`, `previous`, `transform`, `written`, `id`,
  `orientation`, `kept`); the verb `interactive_transform`.
  `elastix_affine(job, workspace, sections=(), restrict_to=(),
  atlas_image="template"|"nissl")`: per section `core.transform.fit_elastix`
  refining its current placement against the atlas image (`ara` read as
  `template`), by the *restrict_to* regions only, its marked damage regions
  left out on their own (`core.damage.exclusions`); the named sections, else
  `fit_targets(job)` (every positioned section the host did not lock);
  `LOCKED` per section; every successful fit written as ONE undo step (fits
  computed outside the lock, `STALE_INPUT` for a section that changed);
  a tiny restricted region is `REGION_TOO_SMALL` before the engine runs;
  a fit that turns a section more than `LARGE_TURN_DEG` from its previous
  transform carries `turn_deg`. With `restrict_to`, each ok row carries
  `restrict_box`: the regions' bounding box on that section's canvas as
  `[x0, y0, x1, y1]` fractions (`region_box`; `core.canvas.zoom_box`'s
  zoom). Refused: `UNKNOWN_SLICE_IDS`, `BAD_ARGS`. Returns `AffineFit`
  (`rows`, `fitted`).
- `deformable.py` — `fit_deformable(job, workspace, records, choice,
  restrict_to=, start=, options=)`: one deformation (a resolved
  `deformation.Choice`) on top of each section's linear placement (*start*
  `linear`), its applied deformation (`current`), or per section whichever it
  holds (`START_LATEST` "latest"), each section's marked regions excluded
  (`core.damage.exclusions`), applied as ONE undo step (an unchanged key
  writes nothing, `written: false`). It runs the fit (the fit grid, the
  image the fit reads, a traced fit section waiting for its running trace
  under one `TRACE_WAIT_S` deadline and adding the trace's regions to the
  call's (`traced_regions`; the row's `trace_regions`, and the record keeps
  the regions each fit ran with), the record cache, the engine). Per-section
  problems (`INVALID_LINEAR_PLACEMENT`, `NO_DEFORMATION`,
  `NO_TRACE`/`TRACE_*`, `BAD_SETTINGS`, `FIT_FAILED`, `RECORD_WRITE_FAILED`,
  `STALE_INPUT`) are rows, not refusals. Returns `DeformableFit`: `rows`
  (call order), `fitted` (each row with its fit and record), `traced` (per
  traced section, the image read and the trace's lines on the fit grid),
  `written`; with `options=`, `pictures(...)` draws each fit (the final
  borders on the image it read, titled with the settings) and each traced
  section's trace (`pictures`, `traces`, `render_failed`, each drawn row's
  `image_indexes`). Its callers: `ants_syn` and `traces.land_trace`. An
  applied record is saved in the section's folder (`job.deformations`:
  `sections/<stem>/deformable/<key>`) and the section's `deformation` holds
  its path relative to the job folder; the step of a traced fit records the
  trace it read (`trace`: the trace's artifact directory); a traced fit
  section reads the trace's artifacts under the job folder
  (`traced_lines(root=...)`).
  `ants_syn(job, workspace, sections, restrict_to=, atlas_image=,
  stiffness=)`: the ANTs SyN fit, applied as ONE undo step:
  on the preprocessed channel (`fit`) against `template` or
  `nissl` (`ara` read as `template`), `soft`/`medium`/`firm`, from
  `START_LATEST` (undo is how to start over); *restrict_to*
  checked against the atlas (`region_entries`); refused, nothing done:
  `ANTS_MISSING` (`refuse_without_ants`: antspyx must import, `ants_ready`),
  `BAD_ARGS` (no sections, more than `MAX_ANTS_SYN_SECTIONS` (4), an unknown
  atlas image or stiffness, a malformed region list), `FIT_ATLAS_UNAVAILABLE`,
  `UNKNOWN_SLICE_IDS`, `UNKNOWN_REGIONS`, `NO_SIDES`. The verb `ants_syn`.

- `traces.py` — `trace_borders(job, workspace, ref, image_model=, prompt=,
  restrict_to=, options=, workers=)`: LangSlice's own
  nonlinear method packaged. It starts the section's image correction (below)
  and a piece of background work (`Job.background`, kind `TRACE_WORK`
  "trace_borders") and returns at once (`TraceStarted.work`, the work id; a
  section whose packaged trace still runs gets that work, `running`). When
  the reply lands, `land_trace` (the work's callable) fits the traced
  borders (`TRACE_FIT`: traced borders against the atlas borders, ANTs,
  medium; from `START_LATEST`; by *restrict_to*'s regions only) with
  `fit_deformable` and applies it as its own undo step, its notice saying so
  and that undo removes it while it is the latest step, with the
  displacement, fold fraction and flags; with *options* the fit and the
  trace are drawn and saved with the work. A section whose
  `inputs.section_inputs` (with the deformation) moved since the call, or
  whose trace was undone or made at another placement, gets nothing: a
  failed work whose notice starts `STALE_INPUT`; a failed image call is a
  failed work too. Refused before any image call: `ANTS_MISSING`,
  `BAD_ARGS` / `UNKNOWN_REGIONS` / `NO_SIDES` (*restrict_to*), then the
  trace's own refusals. The trace itself: one section's image correction, showing
  the model only the chosen regions' borders, its marked regions excluded
  (`core.damage.exclusions`; recorded on the result); the prompt sent is
  saved with each attempt. The section's current
  geometry is `core.handoff.correction_fingerprint`;
  `registration_tool.start_correction` prepares the edit for *image_model*
  (resolved by the door: the toolbox's `build_tools(image_model=...)`
  binding) and returns the record and the call to run. A packaged call
  whose saved trace at this placement and region choice has already landed
  (a step of the section's applied deformation records that trace) starts
  nothing and fits nothing again (`TraceStarted.landed`). A call already
  running at that geometry is not started again (`running`); otherwise the
  call runs on the job's image executor and the section's
  `image_correction` record is written as one undo step when it changed.
  Prepared outside the lock, started and written under it. Refused:
  `UNKNOWN_SLICE_IDS`, `KEEPS_HOST_WARP` / `NONLINEAR_SKIPPED`
  (`Job.nonlinear_refusal`), `INVALID_LINEAR_PLACEMENT`,
  `IMAGE_CORRECTION_IO_ERROR`, `STALE_INPUT` (the geometry changed while the
  call was prepared). Returns `TraceStarted`.
  `trace_from_atlas(job, workspace, refs, image_model=, passes=1,
  workers=)`: the placement-free route "atlas" as a job verb, a HIDDEN
  scripting verb (kept for experiments, called by name, listed nowhere):
  per section `registration_tool.start_atlas_correction` (the model shown
  the clean section and the outlined atlas plane at its position and
  angles, never its placement; `passes` 2 adds the corrective call), the
  record and its artifacts exactly as `trace_borders` writes them, so
  `fit_deformable` with a traced fit section (from the section's written
  linear placement), the submit gate and the maps read it unchanged; no
  verb fits it (a script does). Needs a position and a
  written transform. Prepared outside the lock; under it each section's
  geometry is checked again (`STALE_INPUT` row), its call started, its
  record written; every changed record ONE undo step. Per-section rows
  (`UNKNOWN_SLICE_IDS`, `KEEPS_HOST_WARP` / `NONLINEAR_SKIPPED`,
  `INVALID_LINEAR_PLACEMENT`, `IMAGE_CORRECTION_IO_ERROR`, `STALE_INPUT`,
  `running`); `BAD_ARGS` for no sections or `passes` not 1/2. Returns
  `AtlasTraces` (`rows`, `written`). Both verbs share one prepare step
  (outside the lock) and one apply step (under it).
- `inputs.py` — `section_inputs(state, record, deformation=, trace=)`: a
  digest of what a fit of the section reads (its linear placement:
  position, plane, angles, flip, rotation, transform; its fit appearance;
  damage: marked regions and a mark naming none; with `deformation` the applied deformation's key, with `trace`
  its image correction), `STALE_INPUT`, `stale_row`.
- `atlas.py` — `grep_atlas(job, workspace, query, section="")`: the region
  hierarchy searched like text (`core/atlas_grep.py`), with `in_section`
  per row for a placed section. `grep_atlas_view(job, workspace, regions,
  positions_mm)`: the atlas template at each position (clamped into the
  range) at the stack's view angles with the named regions' borders
  highlighted (`"CTX:left"` allowed; no other borders), saved like a
  `look` picture (so it can be zoomed: it registers its recipe renderer
  `grep_atlas_view` with `core.zoom.RENDERERS`); returns `RegionsView`
  (`pictures`, `entries`, `not_shown`, `regions`, `regions_not_in_plane`,
  `clamped`). Refused: `BAD_ARGS`, `NO_STRUCTURES`, `UNKNOWN_REGIONS`,
  `NO_SIDES`.
- `look.py` — `look(job, workspace, mode, sections=, positions_mm=,
  channels=, atlas_layers=, atlas_opacity=, warp="applied", resolution=)` and
  `zoom(job, workspace, box=(), picture=None, region="")`. Unlike the older read verbs they
  save their own pictures (`save_pictures`: `job.views.save` with the note's
  recipe and caption), so each has a picture number at once: the reply
  (`Looked`, `Zoomed`) carries `pictures` (PIL) and `entries`
  `[{"id", "caption"}]`, and the door must not save them again. `look` shows at
  most `MAX_LOOK_PICTURES` (4) of the pictures `core.look.look` draws; the rest
  are `not_shown`, each with the arguments that get it. `zoom` redraws a box
  of picture `picture` (default: `views.latest(exclude_tool="zoom")`) through
  `core.zoom.redraw` and saves it as tool `zoom`; `stale` when the stack has
  changed since. Instead of a box, `region` selects exactly one atlas region
  on a saved overlay, cropped to its bounds with only its border drawn;
  `BAD_REGION_PICTURE` on other modes and `REGION_NOT_IN_PLANE` when absent.
  A picture with no recipe or whose sections are gone is cropped
  from its saved image (`redrawn` False) and its crop recipe names its
  `source` picture, so a crop of a crop is cut from the original. Refused:
  `NO_PICTURE`, `UNKNOWN_PICTURE`, `PICTURE_NOT_SAVED` (no saved image to
  crop, or a dry run's unknown number), `BAD_BOX`, `EMPTY_BOX`, and the look
  refusals (`UNKNOWN_SLICE_IDS`, `UNKNOWN_MODE`, `UNKNOWN_CHANNEL`,
  `MIXED_CHANNELS`, `TOO_MANY_CHANNELS`, `UNKNOWN_LAYER`, `NO_POSITIONS`,
  `NO_POSITION`, `BAD_WARP`, `BAD_ARGS`). `show_result(job, workspace, tool,
  mode, sections, positions_mm=, zooms=)`: the picture a change tool shows
  of what it wrote, drawn as `look` draws `overlay` (one per section, each
  zoomed to its window) or `positioning`, saved under the change tool's
  name (numbered, captioned, zoomable); at most four shown, the rest
  `not_shown` with the `look` call that draws them, a failed picture
  `failed` (the write stands).
- `views.py` — `status(job)` (`StackStatus`: the status rows, angles,
  breaks as the filenames of the sections after each gap,
  `core.state.break_ids`; a row the user locked carries `locked: true`, one the user gave a
  damage note `damage_by_user: true`).
- `registry.py` — `VERBS`: every verb (agent tool) name -> `Verb(name,
  function, kind "read"/"write", group "Common"/"Positioning"/"Linear"/
  "Nonlinear", when, long, scripting, image_model, hidden, limits)`, in the
  order every door lists them; `enabled(spec, scripting=, image_model=,
  hidden=)`: the verbs a run of the spec has. The gating (`when`): `look`,
  `zoom`, both channel tools, `grep_atlas`, `grep_atlas_view`, `status`, the
  job-folder tools (`list_files`, `search_files`, `read_file`), `note`,
  `undo`, `redo` and `submit` in every run; `position_sections` with the
  `position` task or the stack's cutting angles left to the agent
  (`transform.angles`, `positions_on`); `interactive_transform` with
  `transform` and `transform.interactive`; `elastix_affine` with `transform`
  and `transform.automatic`; `mark_damage` with `agent_damage`; `ants_syn`
  with `nonlinear`; `trace_borders` with `nonlinear` and an image model.
  `image_model` False leaves out the verbs that call the image model, for a
  door that cannot reach it (MCP with none connected). A `scripting` verb
  (`export_maps`, a "read": it changes no state) is the agent CLI's and the
  library's only (`build_tools(scripting=True)`), never offered to a model.
  A `hidden` verb (always a scripting verb; `trace_from_atlas`) is in
  `enabled` only with `hidden=True` (what `build_tools(scripting=True)`
  passes, so the CLI and the library call it by name) and in no listing:
  `listed()` (every verb but the hidden ones) is what `langslice-job ops`,
  `langslice-job schema` without a verb, the job folder's card, the
  library's `verbs` and the CLI's `verbs` lists show. `limits`: the most one
  call takes, by what it counts (`look`, `grep_atlas_view` and
  `set_preprocessed_channel_properties` show at most four `pictures`;
  `interactive_transform` and `ants_syn` take at most four `sections`); the
  reference card lists them. Every door is built from it: `build_tools`
  makes the tools `enabled(spec)` names (the ADK and MCP doors), the MCP
  door's `readOnlyHint` is `kind == "read"`, and the agent CLI
  (`langslice-job ops`, `schema`, `job FOLDER VERB`), the library's job
  methods and the job folder's reference card list it
  (`src/langslice/doors/`). A tool that was replaced keeps answering by its
  old name: `RETIRED` maps each retired name to `(replacement, hint)`
  (`view_slices`, `view_atlas`, `view_placement`, `view_stack` -> `look`;
  `preprocess` -> `set_preprocessed_channel_properties`; `set_positions`,
  `set_cutting_angles`, `reorder_slices` -> `position_sections`;
  `orient_slices`, `adjust_transforms` -> `interactive_transform`;
  `mark_damaged` -> `mark_damage`; `fit_affine` -> `elastix_affine`;
  `fit_deformable` -> `ants_syn`; `search_position` -> none), and every door
  answers a call by one with `retired_payload(name)` (`RETIRED_TOOL`, `use`,
  a message naming the replacement and how to call it; nothing done), every
  time it is called: the ADK plugin `agent.plugins.RetiredToolsPlugin`, the
  agent CLI (`cli/job.py` `call`, `cli/catalog.py` `schema`), the MCP
  server (`mcp/server.py` `LangSliceServer.call_tool`) and the library
  (`JobHandle.__getattr__`'s `AttributeError`). A retired name never becomes
  a verb again. `tests/test_ops_registry.py` and
  `tests/test_doors_declarations.py` check it.

The pictures are drawn by the core (`src/langslice/core/`, its own
`CLAUDE.md`), and the job saves every one a door shows, with its layers, in
the job folder through one hook (`Job.views.shown`, `src/langslice/job/`).

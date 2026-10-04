# LangSlice `ops/` — the verbs

Package guide for `src/langslice/ops/`. The repo-level `CLAUDE.md` holds the
project-wide rules. `AGENTS.md` here is a verbatim copy — edit one, mirror to
the other.

Every write to a stack, and every read a viewing tool makes, is a function
here (layered refactor, phases 3a and 3b, 2026-10-03; the rest of the tool
logic and the read verbs, phase 3d, 2026-10-04). The agent tools, the MCP
server, the agent CLI and scripts (phase 5) call the same
functions, so a tool and a script do exactly the same write and get the same
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
  `fit_affine`, `fit_deformable`, `trace_borders`, and the scripting
  verbs `trace_from_atlas` and `export_maps`), which compute outside
  it from the state they read and take it themselves to apply, comparing
  each section's `inputs.section_inputs` with the value they computed from:
  a changed section is that section's row `STALE_INPUT`
  (`inputs.stale_row`; `trace_borders` raises it, `trace_from_atlas`
  compares the geometry fingerprint), the others apply.
  `set_transforms` and `keep_linear` take it themselves too. A read (`views.py`, `atlas.py`,
  `positions.search_position`) writes nothing.
- Pictures are the core's, never drawn here: a read verb, or a write that
  shows its result, takes the call's core `core.display.DisplayOptions`
  and returns the plain PIL pictures `src/langslice/core/` draws (captions
  burned in, each noted for the job's saved views). Without options a write
  draws nothing (a script). Where a picture must exist before the write
  (`fit_affine`, `adjust_transforms`: a picture that fails refuses that
  section; `preprocess`: the BEFORE picture), it is drawn first.
- It never words anything for a model and never applies the
  look-before-commit gates (`compared`/`reviewed`): those are the tool
  door's (`submit` takes the door's gate as a callable, `set_positions` the
  door's "not already seen" filter as a predicate). Job rules (locked
  sections, host damage flags, the spec's flip switch, the submit gates)
  are the job's and are applied here; the transform cap (`Job.over_cap`)
  is checked by the tool door with its other arguments.
- A model is passed in, never chosen here: `traces.trace_borders` (and
  `traces.trace_from_atlas`) takes the image model (`providers.registry.ImageModel`: provider, model, `call`)
  the door resolved (phase 4); the geometry fingerprint is the core's
  (`core.handoff.correction_fingerprint`).
- A refusal is `refusal.Refused(code, status="error", **facts)`, nothing
  written; `Refused.payload()` is the `{"status": ..., "error": code, ...}`
  every door answers with (`status` "refused" for a gate, `Refused.of`
  turns a job gate's payload into one).
- Layer: core and job only. Never a door (`doors/`), the agent driver
  (`agent/`), a host (`hosts/`), a provider (a type named under
  `TYPE_CHECKING` excepted: `traces.py` takes an `ImageModel`), and never
  `google.*`, `litellm` or `openai`. import-linter's contracts
  (`pyproject.toml`, `tests/test_import_layers.py`) check the imports;
  `tests/test_core_imports.py` loads each module in a fresh interpreter and
  checks what it loads.

## Files

- `positions.py` — `set_positions(job, workspace, [(ref, mm)], options=,
  show=)`: clamp into `workspace.position_range`, write, report
  written/clamped/unknown; with options, the written placements *show* keeps
  are pictured after the write (`views.placement_view`; `view`,
  `not_shown`). `set_cutting_angles(job, workspace, pitch, yaw)` (every
  section gets the one plane, so a stack whose sections carried different
  angles is flattened, one undo step restoring them; drops the render
  cache). `search_position(job, workspace, ref, window_mm, angles=)`
  (a read: `oblique.fit_oblique` around the section's position, holding the
  section's own angles unless `angles`, the best
  position/angles/score; `UNKNOWN_SLICE_IDS`, `NO_POSITION`, `BAD_ARGS`,
  `FIT_FAILED`). `run_deepslice(job, workspace, ids, allow_angle_change=)`
  (the `core/deepslice.py` seam: `UNAVAILABLE`).
- `order.py` — `reorder(job, filenames, after)` (one block, filenames only),
  `renumber(order)` (indices only, no undo step).
- `orientation.py` — `orient_sections(job, entries, workspace=, options=)`:
  flip and quarter turns; a change drops the section's transform; `LOCKED`,
  `FLIP_DISABLED`, `BAD_ROTATION` per entry; with options, the first four
  sections reached pictured as they now stand (`render_failed` per failure).
- `damage.py` — `mark_damaged(job, entries)`; a host flag cannot be cleared.
- `appearance.py` — `set_appearance(job, targets, ids, settings)`,
  `planned_settings` (what a write would leave, written nowhere) and
  `preprocess(job, workspace, targets, ids, settings, shown=, options=)`:
  per shown section the first target's BEFORE and AFTER pictures (uncaptioned,
  `BeforeAfter`), drawn before the write; a failure refuses the call
  (`RENDER_FAILED`), nothing written.
- `notes.py` — `add_note(job, text)`.
- `history.py` — `undo(job)` / `redo(job)`: `Stepped` (`done`, `moved`: the
  sections whose position the step changed, which the tool door's gates
  forget, `depth`); `moved_positions(before, state)`.
- `submit.py` — `submit(job, summary=, notes=, interval_breaks=,
  traces=, workspace=, gate=)`: the job's gates (`Job.submit_errors`), then
  with *traces* (the image model is in the run; *workspace* reads each
  section's geometry) and the nonlinear task every section's completed image
  correction (`MISSING_IMAGE_CORRECTIONS`, reported before a missing
  deformation), then the door's *gate*; then ONE undo step: the interval
  breaks, the order reversed to run the atlas way (noted), the notes and a
  `submit: <summary>` note, `submitted`; then every queued picture written
  (`job.views.flush`) and, with the workspace, every placed section's maps
  and the exports (`exports.export_maps`; a failure is logged, the submit
  stands). Returns `Submitted` (`exported`).
- `exports.py` (formats phase) — `export_maps(job, workspace, ids=,
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
- `transforms.py` — the stored transform: `interactive_transform` (knobs,
  `shear` optional (0), to the record), `same_transform`, `fit_transform`
  (`fit_affine`'s record), `set_transforms(job, {id: record})` (one undo step
  for the batch; locked sections refused). `KNOBS` lists the knobs in
  payload order. The shear convention is `affine.decompose_affine`'s.
  `fit_affine(job, workspace, records, method=, fit_atlas=, include=,
  exclude=, options=)`: the fitter (`elastix` refining the current placement
  against `fit_atlas`, or `silhouette`), `LOCKED`, `DAMAGED` unless regions
  restrict the fit, each fit drawn (`core.placement.fit_picture`) before
  every successful fit is written as one undo step; `AffineFit` (`rows`,
  `fitted`, `pictures`). `fit_targets(job)`: the default sections.
  `adjust_transforms(job, workspace, entries, options=)`: per entry the
  knobs checked (a left-out shear keeps the current one), the section staged
  (`core.placement.stage`), its record built and drawn
  (`core.placement.transform_views`; `ab` adds the stored transform), then
  every changed record written as one undo step; `Adjusted` (one
  `Adjustment` per entry: `error`, `staged`, `previous`, `transform`,
  `written`, `pictures`).
- `deformable.py` (phase 3b) — `fit_deformable(job, workspace, records,
  choices, include=, exclude=, start=)`: the deformation on top of each
  section's linear placement. The door validates the arguments and resolves
  each candidate into a `deformation.Choice`; this runs every fit (the fit
  grid, the image each fit reads, a traced fit section waiting for its
  running trace under one `TRACE_WAIT_S` deadline, the record cache, the
  engines) and, with exactly one choice, applies each section's result as
  ONE undo step (an unchanged key writes nothing, `written: false`).
  Several choices are a preview, nothing written. Per-section problems
  (`INVALID_LINEAR_PLACEMENT`, `NO_DEFORMATION`, `NO_TRACE`/`TRACE_*`,
  `BAD_SETTINGS`, `FIT_FAILED`, `RECORD_WRITE_FAILED`) are rows, not
  refusals. Returns `DeformableFit`: `rows` (call order), `fitted` (each
  row with its fit and record, for the pictures), `traced` (per
  traced section, the image read and the trace's lines on the fit grid),
  `written`; with `options=`, `pictures(...)` draws each fit (the final
  borders on the image it read, titled with the settings; `ab` adds what it
  started from) and each traced section's trace (`pictures`, `traces`,
  `render_failed`, each drawn row's `image_indexes`).
  `keep_linear(job, records, reason)`: the "linear placement
  stands" record, one undo step; `NOTHING_WRITTEN` (each offending section
  under `results`) when one lacks a position or a transform (with Linear
  off, the message adds `handoff.NO_TRANSFORM_LINEAR_OFF`). An applied
  record is saved in the section's folder (`job.deformations`:
  `sections/<stem>/deformable/<key>`) and the section's `deformation`
  holds its path relative to the job folder; a traced fit section reads
  the trace's artifacts under the job folder (`traced_lines(root=...)`).

- `traces.py` — `trace_borders(job, workspace, ref, image_model=, prompt=,
  workers=)`: one section's image correction. The section's current
  geometry is `core.handoff.correction_fingerprint`;
  `registration_tool.start_correction` prepares the edit for *image_model*
  (resolved by the door: the toolbox's `build_tools(image_model=...)`
  binding) and returns the record and the call to run. A call already running at that
  geometry is not started again (`running`); otherwise the call runs on the
  job's image executor and the section's `image_correction` record is
  written as one undo step when it changed. `UNKNOWN_SECTION`,
  `INVALID_LINEAR_PLACEMENT`, `IMAGE_CORRECTION_IO_ERROR`. Returns
  `TraceStarted`.
  `trace_from_atlas(job, workspace, refs, image_model=, passes=1,
  workers=)`: the placement-free route "atlas" as a job verb, a HIDDEN
  scripting verb (kept for experiments, called by name, listed nowhere):
  per section `registration_tool.start_atlas_correction` (the model shown
  the clean section and the outlined atlas plane at its position and
  angles, never its placement; `passes` 2 adds the corrective call), the
  record and its artifacts exactly as `trace_borders` writes them, so a
  traced `fit_deformable` (from the section's written linear placement),
  the submit gate and the maps read it unchanged. Needs a position and a
  written transform. Prepared outside the lock; under it each section's
  geometry is checked again (`STALE_INPUT` row), its call started, its
  record written; every changed record ONE undo step. Per-section rows
  (`UNKNOWN_SECTION`, `INVALID_LINEAR_PLACEMENT`,
  `IMAGE_CORRECTION_IO_ERROR`, `STALE_INPUT`, `running`); `BAD_ARGS` for
  no sections or `passes` not 1/2. Returns `AtlasTraces` (`rows`,
  `written`).
- `inputs.py` — `section_inputs(state, record, deformation=, trace=)`: a
  digest of what a fit of the section reads (its linear placement:
  position, plane, angles, flip, rotation, transform; its fit appearance;
  damage; with `deformation` the applied deformation's key, with `trace`
  its image correction), `STALE_INPUT`, `stale_row`.
- `atlas.py` — `grep_atlas(job, workspace, query, section="")`: the region
  hierarchy searched like text (`core/atlas_grep.py`), with `in_section`
  per row for a placed section.
- `views.py` — the read verbs, one per viewing tool: `status(job)`
  (`StackStatus`: the status rows, angles, breaks), `view_slices(job,
  workspace, records, options, keep_going=)` (`SectionsView`),
  `view_atlas(job, workspace, positions, options)` (`AtlasView`, with
  `regions_not_in_plane`), `view_placement(job, workspace, pairs, options)`
  (`PlacementView`: pictures, `PairShown` per pair with its row and image
  indexes, failed pairs; `placement_view` is the shared body),
  `view_stack(job, workspace, options)` (`StackView`: sheet, plot, rows in
  written-position order). `regions_not_in_plane`. `MAX_VIEW_SLICES` (4).
- `registry.py` — `VERBS`: every verb (agent tool) name -> `Verb(name,
  function, kind "read"/"write", group "Common"/"Positioning"/"Linear"/
  "Nonlinear", alternates, when, long, scripting, image_model, hidden)`, in
  the order every door lists them; `enabled(spec, scripting=, image_model=,
  hidden=)`
  (phase 5): the verbs a run of the spec has (`when`: the task switches and
  host switches that were `build_tools`' if-chain; `image_model` False
  leaves out the verbs that call the image model, `trace_borders`, for a
  door that cannot reach it: MCP with none connected). A `scripting` verb (`export_maps`, a "read":
  it changes no state) is the agent CLI's and the library's only
  (`build_tools(scripting=True)`), never offered to a model, so the agent
  tools and MCP declare exactly what they did. A `hidden` verb (always a
  scripting verb; `trace_from_atlas`) is in `enabled` only with `hidden=True`
  (what `build_tools(scripting=True)` passes, so the CLI and the library
  call it by name) and in no listing: `listed()` (every verb but the hidden
  ones) is what `langslice ops`, `langslice schema` without a verb, the job
  folder's card, the library's `verbs` and the CLI's `verbs` lists show;
  `table()` as plain rows. `fit_deformable`'s alternate is `keep_linear`.
  Every door is built from it (phase 5): `build_tools` makes the tools
  `enabled(spec)` names (the ADK and MCP doors), the MCP door's
  `readOnlyHint` is `kind == "read"`, and the agent CLI (`langslice ops`,
  `schema`, `job FOLDER VERB`), the library's job methods and the job
  folder's reference card list it (`src/langslice/doors/`). A verb is
  never renamed once shipped. `tests/test_ops_registry.py` and
  `tests/test_doors_declarations.py` check it.

The pictures are drawn by the core (`src/langslice/core/`, its own
`CLAUDE.md`), and the job saves every one a door shows, with its layers, in
the job folder through one hook (`Job.views.shown`, `src/langslice/job/`).

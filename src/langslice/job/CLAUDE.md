# LangSlice `job/` — the job folder

Package guide for `src/langslice/job/`. The repo-level `CLAUDE.md` holds the
project-wide rules. `AGENTS.md` here is a verbatim copy — edit one, mirror to
the other.

The job layer: the `Job` (`job.py`: the state, undo, the checkpoint, the
submit gates), where its files live and how they are named, indexed and
filled with pictures, the public files rendered from the state, and the
QUINT (QuickNII/VisuAlign) reading and writing.

## The layer rule

Job layer: imports the core (`core.state`, `core.discovery`,
`core.deformation`, `core.layers`, `core.jpeg`, `core.maps`, `core.atlas`,
...) and itself only. Never an operation (`ops/`), a door (`doors/`), the
agent driver (`agent/`), a host (`hosts/`), a provider or `google.*`,
`litellm`, `openai`. import-linter's contracts (`pyproject.toml`, run by
`tests/test_import_layers.py`) check the imports;
`tests/test_core_imports.py` loads each module in a fresh interpreter and
checks what it loads.

## The job folder

One stack, one job folder, next to its images: `<images>/langslice/`
(`layout.job_folder_for`). Every host puts it beside the images it hands
LangSlice, so a job travels with its images:

| Host | Images | Job folder |
|---|---|---|
| `langslice linear run FOLDER` | `FOLDER` | `FOLDER/langslice/` |
| `langslice mcp`, `start_job(image_folder=...)` | that folder (or the named job folder's) | `<folder>/langslice/`, or the named job folder |
| `langslice mcp`, `start_job(job_id=...)` | the saved job's | the index entry's job folder |
| ABBA worker `linear.run` (Fiji connector) | the connector's snapshot folder, a fresh `~/.langslice/snapshots/abba-*` folder per run | `<snapshots>/langslice/` (or the spec's `job_dir`, or the read-only fallback); the result's `output_dir` names the one used |
| ABBA Claude mode (`mcp.prepare`) | `~/.langslice/snapshots/mcp-*/` | `<snapshots>/langslice/` (id in the index) |

Two exceptions, both from `layout.locate_job_folder` (used by
`Job.open`, `engine.build_context` and the saved ABBA job preparation):

- **An explicit folder.** `JobSpec.job_dir` (`--job-dir PATH` on `linear
  run`, `mcp` and `init`) puts the job folder exactly there, same
  layout, e.g. one per experiment on one dataset folder; the image folder
  is then never written. Two
  jobs never share a folder silently: `layout.check_owner` refuses a folder
  whose `job.json` names a DIFFERENT image folder (`ValueError`, "already
  holds the job of ..."); the same image folder's job is continued.
  `to_dict` leaves `job_dir` out when it is unset.
- **A read-only image folder.** When `<images>/langslice/` cannot be created
  or written (`layout.writable`: a temporary probe file in an existing job folder or
  the image folder; no file remains after the check), the
  job folder is `~/.langslice/jobs/<id>/`, same layout. The id is the
  image folder's path hashed (`index.folder_id`), so a reopen finds it
  again; a saved ABBA job uses its own id. Said once through the
  progress/emit log (`[job] ... cannot be created or written; the job
  folder is ...`) and recorded in the id's index entry under `fallback`
  (`image_folder`, `reason`).

Inside (`layout.py`, names as constants):

```
job.lock             the write lock (lock.py), held while a writer syncs and commits
job.json             settings: the JobSpec under "spec", format_version (1),
                     created_at, image_folder; the user's "notes" (every door
                     gives them to the agent); the agent CLI's "viewer"; a saved
                     ABBA job's job_id and its own fields under "host" (kind,
                     params)
state.json           the checkpoint (StackState, state format 4): THE TRUTH
registration.json    its public rendering (formats.py), rewritten on every checkpoint
history/             undo/redo: index.json + step-NNNNNN.json, one per step
sections/<stem>/     per section; <stem> is the image filename's stem (the whole
                     filename when two images share a stem)
  deformable/<key>/          deformation records (RecordStore): record.json +
                             warp.ome.zarr, the parent a sibling it names
  registrations.jsonl        every registration the section has held, numbered
                             (registrations.py); read back by restore_registration
  images/<name>.png          made section images (core/section_images.py: the
                             opening preprocessed, make_image's, a script's
                             add_image); never deleted, <name>.2.png on a remake
  masks/<name>.npz           section masks (core/section_masks.py); never deleted
  image_correction/<key>/    trace_borders calls, attempt-NN inside
  views/<seq>_<tool>_<mode>/ pictures of this section the model saw
  coords.tif, labels.tif,    the section's maps (formats.py), written only by
  labels_fiji.tif,           export_maps (never by a write or by submit);
  labels.csv,                wherever the section maps into the atlas volume
  residual.tif, maps.json
views/<seq>_<tool>_<mode>/   pictures of several sections or none (atlas, sheets,
                             opening strips, show_stack pages, brief's strips)
views.jsonl          append-only index of every saved picture
views.seq            the picture/call numbers handed out (+ views.seq.lock)
exports/             linear_results.json (the run's result; spec.out overrides
                     the path), spend.json (the agent run's spend account), result.json (a saved host job's final result),
                     quicknii.json and visualign.json (with the maps),
                     traces/<stem>.npz (each traced section's image-model reply
                     on its maps' grid, with the maps),
                     plugins/<tool>/ (copies of the layers plugins left)
plugins/<tool>/      a tool plugin's own files and layers (layers.json, layers/);
                     the only place a plugin tool writes
logs/events.jsonl    one line per open (`resumed`: whether a checkpoint was read)
logs/runs/<id>.json  an agent-CLI background run (`--background`), its stderr in <id>.log
logs/calls.jsonl     one line per agent-CLI call (verb, arguments, outcome, artifacts)
trace/               agent transcripts (`TRACE_DIR`, `JobLayout.trace_dir`): the
                     built-in agent's trace, a coding agent's session copied in by
                     `langslice session-end` (`doors/cli/transcripts.py`)
env/                 the job's own Python environment (`ENV_DIR`; `doors/cli/jobenv.py`)
prompt.txt           a saved ABBA job's copy prompt
AGENTS.md, CLAUDE.md the reference card for coding agents, identical, generated
                     (`doors/card.py`) and rewritten when stale
BRIEF.md             the agent CLI's brief (`langslice-job FOLDER brief`, `init`):
                     the job statement and the opening pictures' paths
```

**The public files (`docs/file_formats.md` has every field).**
`state.json` stays the one working source; `formats.py` renders it for scripts and other programs, in BrainGlobe micrometres (the
atlas's axis order, voxel `i`'s centre at `i * resolution`) and image-file
pixels `[row, col]`, and never reads any of it back. `Job.checkpoint`
writes the state and owes `registration.json`; it is rewritten atomically
once, as the outermost hold of the job folder's lock ends
(`FolderLock.before_release` calls `Job._write_derived`, still holding the
lock), so a write that checkpoints twice in one hold (the write, then the
deformations it made stale) maps the stack once, from the state it leaves
(`formats.write_registration`, through `Job.workspace`, which `Job.open`,
`Job.load` and the tool door set; a failure is logged, never raised): per
section its parameters in public units (atlas, plane, orientation, the six
numbers and their knobs, the applied deformation's record), the file
pixel -> atlas matrix of its linear placement (`core.maps`), why it has none
(`problem`), and its written maps with whether they are still `current`
(`maps.json`'s `parameters_digest`). The maps (`write_section_maps`:
`coords.tif`, `labels.tif`, `labels_fiji.tif` + `labels.csv`,
`residual.tif`, `maps.json`) are written only by `ops.exports.export_maps`
(on demand; a section's `maps` entry is null until then), the exports by it
and at submit (`ops.exports.export_stack`). `Job.persist` False
writes none of it.

**Resuming with other inputs is refused.** `Job.open` with `spec.resume`
on a folder whose checkpoint was made from other supplied inputs
(`JobSpec.inputs` and `JobSpec.frozen`, compared with the checkpoint's own
spec: `job.changed_inputs`) raises `job.InputsChanged` (a `ValueError`) before
anything is written, naming the keys that differ and how to start fresh
(`START_FRESH`: `--fresh`, `resume=False`); resuming would have kept the
old inputs and dropped the new ones. The same inputs resume as before; the
agent CLI's `init` answers `INPUTS_CHANGED` (exit 3). ABBA's Claude mode
(`prepare_saved_job`) never resumes (`prepare_linear` sets `resume` False),
so it is never refused.

**Opening a job without writing.** `Job.open` writes `job.json` and a
first checkpoint. `Job.load(spec, workspace, folder=)` opens an
existing folder as it stands (the checkpoint and its history; nothing
rewritten, `FileNotFoundError` without a checkpoint, another image
folder's job refused): the agent CLI and the library open a job per call
this way (`doors/jobs.py`), so a call beside a running agent changes
nothing until its first write. `Job.persist` False (`Job.load(...,
persist=False)`, the CLI's `--dry-run`) writes nothing at all: no
checkpoint, no history, and `views.DiscardedViews` saves no picture.
`Job.engine_log` is where the engines' own output goes while they fit
(`logs/engine.log`, `core.engine_output`); None for a job that persists
nothing.

**One writer at a time, across processes.** Every write goes lock -> sync -> apply -> commit -> unlock: `Job.writing()` holds
the folder's `lock.FolderLock` (`job.lock`, `filelock`: `fcntl` on Linux
and macOS, `msvcrt` on Windows, released by the OS if a process dies;
reentrant in its thread; `LOCK_TIMEOUT_S` 300 then `JobBusy`) and runs
`Job.sync` first. `commit`, `checkpoint` and the history save take it too;
`undo`/`redo`, `clear_stale_deformations` and the landing of image
corrections (`settle_image_corrections`, `wait_image_job`: the calls run
outside it, their results land under it) are written inside it, and so
is `Job.open`'s read and first checkpoint. A missing or unwritable folder
is used without the lock. After a sync the section records are new
objects: resolve them by id inside the block.

**A read never waits for another process's writer.** `Job.reading()` (the
tool door's guard for a read verb) keeps this process's other threads out
(`FolderLock.threads`) and takes the folder lock only when it is free
(`held_if_free`): then it syncs as a write does. While another process holds
it (a long fit applying, a plugin tool that holds it while it computes), it
reads the state file as that process last saved it (`_peek`: no history
read, no undo step), and keeps the state it held before (`_peeked`): the next
locked `sync` puts that back first and then reloads as usual, so the other
process's write enters this job's history as that job's step.

**Paths are relative.** Every path a job file stores (a deformation's
`record`, an image correction's `artifact_dir` / `artifact_paths`, the views
index) is relative to the job folder (`JobLayout.relative` /
`JobLayout.resolve`; `job.checkpoint.state_paths` walks a state's
paths). The trace identity in a deformation cache key is the stored
(relative) artifact directory, so keys survive a move of the folder.
`StackState.image_folder` itself stays absolute (rewritten by `Job.open`
to where the images are now). `job.json`'s `image_folder` is `".."`
(`layout.IMAGES_ARE_PARENT`) for the default job folder, so the job folder
moves or is renamed with its images; an explicit job folder stores the
absolute path; `registration.json` stores it the same way.
`layout.held_image_folder` resolves it, the CLI and library
(`doors.jobs.read_spec`) open the images there, and when
they are gone say how to reattach (`langslice-job NEW_IMAGES init
--job-dir FOLDER`); `check_owner` lets an explicit folder whose images no
longer exist be taken over by the images it is opened with.

## Files

- `lock.py` — `FolderLock(folder)` (`held(abort=)`, `held_here()`,
  `threads()`, `held_if_free()` (yields whether it got the folder lock without
  waiting), `LOCK_FILE` `job.lock`, `LOCK_TIMEOUT_S`, `JobBusy`): above. A thread
  lock is taken before the file lock, so threads of one process exclude
  each other even where the file lock cannot be used; a waiter given
  `abort` asks it every `ABORT_POLL_S` and raises `LockYielded` when it says
  so (background work giving way, below). `before_release`: callbacks run,
  still holding it, as a thread's outermost hold ends (the job's derived
  files, above; each `Job` registers its own when its `lock` is set).
- `background.py` — `BackgroundWork` (`Job.background`, made on first
  use): long work that runs while the agent carries on. `start(kind,
  sections, land, wait_for_images=)` returns a `Work` at once (`id` `w1`,
  `w2`, ...; `kind`, `sections`, `status` `running`/`done`/`failed`,
  `started`/`finished` wall-clock seconds, `notice`, `pictures`, `images`,
  `captions` (each picture's index caption, from its note), `result`,
  `registrations` (the registration each section holds after the landing,
  `Job.registered_since`)); on a work thread it waits for the sections' running image calls
  (`Job.wait_image_job`, which records each reply), then runs *land*, the
  caller's callable (an operation's fit; this module imports no operation
  and no provider), which writes its own undo step and returns a `Landed`
  (`status`, `text`, `pictures`, `result`); an exception is a `failed`
  work. The pictures *land* draws are saved among the job's pictures with
  their notes (`Job.views.save`), their numbers on the work; *land* runs as
  the work's kind (`Job.acting`), so its registrations are listed as made by
  it. The notice is
  `notice_text`: `"<id> <kind> of <sections> finished|failed: <text>
  Now registration <n> of <section>. Pictures #n, #m."`. `notices()` pops the work finished since the last call
  (each handed out once), `running()` / `running_for(section, kind)` list
  what still runs, `get(id)`, `all()`, `wait_all(timeout=)` waits for every
  piece, `wait_any(timeout=)` until one running piece finishes (for a caller
  without the job's lock: the agent session between turns), `close()` waits
  and stops the threads (`Job.close` calls it).
  Giving way: `submit` waits under the job's lock while a landing needs it,
  so while a lock holder waits in `wait_all`, a work thread that would wait
  for the lock raises `LockYielded` before writing anything
  (`Job.writing` passes `gives_way` as the abort check on work threads) and
  the waiting thread runs that landing itself.
- `layout.py` — `locate_job_folder` (above), `writable`, `check_owner`,
  `JobLayout` (the folder, every name, `section_dir`,
  `deformable_dir`, `masks_dir`, `images_dir`, `image_correction_dir`, `section_views_dir`,
  `relative`/`resolve`, `ensure`, `log_event`), `job_folder_for`,
  `section_dirname`, `read_job_file` (refuses a newer `format_version`),
  `write_job_file` (fields over what it holds; `created_at` kept).
- `registrations.py` — every registration a section has held
  (`docs/file_formats.md`, "Registrations"): `value(state, record)` (its
  orientation, transform and deformation, copied; a deformation whose
  `linear_key` changed is no part of it; None with neither a transform nor a
  deformation), `digest` (orientation, transform, the deformation's record
  key), `made_by` (what a registration held before its list began says made
  it), `Registrations(layout)` (`sections/<stem>/registrations.jsonl`:
  `entries` (cached by the file's stamp; a line cut short skipped), `get`,
  `find` (by digest), `number_of(state, record)`, `add` (under the job's
  lock: the next number, or the entry the registration already has)), `now`,
  `SCRIPT`. Read back, never derived from the state; undo leaves it.
- `plugin_store.py` — what tool plugins keep in the job folder
  (`docs/plugins.md`): `register_layer(layout, record, tool, call, layer,
  persist=)` writes a layer's file under `plugins/<tool>/layers/`, lists it in
  the tool's manifest `layers.json` and puts its reference in the section's
  `layers` (the state: undo hides it, the files stay); `read_manifest`,
  `layer_rows`, `export_layers` (copies the shown layers to
  `exports/plugins/<tool>/`, called by `ops.exports`). `job.json`'s `plugins`
  (the provenance `doors/plugins` records) is written with `write_job_file`.
  `JobLayout.plugins_dir` / `plugin_dir(tool)`. `Job.pushed` counts undo steps
  and `Job.merge_steps(mark)` makes the steps since `mark` one (a plugin tool's
  call is one step however many operations it ran).
- `history.py` — `History`: the undo/redo stacks on disk. A step is written
  once, as its own file, when it enters the history (matched to the job's
  stacks by identity); `index.json` (format 1: `undo`, `redo`, `next`) is
  the only file rewritten per step, and step files it no longer names are
  deleted, so the folder holds at most `UNDO_DEPTH` (50) undo steps plus
  the redo side. A new step never takes the name of an existing file (a
  second job on the same folder). A step of state format 3 is read through
  `History.upgrade` (the job's section image migration) and written again at
  the current format by the next `save` (`upgraded`). An unreadable or newer
  history (index, any step, step files without an index, or a step of
  another state format) starts the session without undo and is never deleted:
  `History.problem` says why (logged, and `Job.open` says it through the
  progress log), and `save` writes and deletes nothing while it is set; a
  fresh job reads a history before emptying it. Only a history that was read is pruned.
- `index.py` — saved host jobs by id: `~/.langslice/jobs/<id>.json`
  (`job_id`, `job_folder`, `host_channel`, `created_at`, and `fallback`
  when the job folder is under `~/.langslice/jobs/` because the image
  folder could not be written; owner-only). `folder_id` (an image folder's
  stable id). The channel token stays here, not in a folder that may be
  shared. `register`, `lookup` (refuses a newer entry).
- `checkpoint.py` — `state.json`: `write_checkpoint` (atomic, versioned,
  `STATE_FORMAT_VERSION` 4), `read_checkpoint(path, upgrade=)`,
  `current_state(data, upgrade=)` (format 3, `UPGRADABLE_FORMAT`, through
  *upgrade*: `core.section_images.Migration`; refuses any other format: a
  newer LangSlice's asks for an update, an older pre-release's for a new job,
  `--fresh` or `resume=False`; history steps are read the same way), `write_json_atomic` (every job file's writer),
  `state_paths` and `relative_to` (a state's stored paths made relative to
  the job folder).
- `job.py` — `Job`: `open` (resume or ingest, the opening `preprocessed`
  image of every section without one, `core.section_images.ensure_opening`,
  outside undo, then the first checkpoint; the open event records
  `resumed`), `load` (as it stands, writing nothing, except once for a job of
  state format 3 or without its opening images: then the migrated state and
  history are written at format 4 under the lock, after reading the state file
  again there (`_write_upgrade`: another process may have migrated and written
  meanwhile; its state is kept); a dry run keeps them in
  memory, though the image files the migration makes are written), `upgrade` (the job's `Migration`, which every read of the state
  file goes through), `workspace` (setting it gives a workspace without a
  `job_folder` this job's folder, where its made images are read), `writing` / `reading` / `sync` (above), `snapshot` / `commit`, `undo` /
  `redo` (each lists the registrations it leaves: `list_registrations(before)`,
  the store `registrations`, made by the tool `acting(tool)` names for the
  thread, else `script`; `registered`, the journal of (thread, section,
  number), and `registered_since(mark)` for a door's reply), `checkpoint` (state, then `registration.json`, then the
  observers: `observe(fn)` calls *fn* with the state after every
  checkpoint and reload, the hosts' live views; one that raises is logged),
  the image-correction jobs (`start_image_job`, `settle_image_corrections`,
  `wait_image_job`: timeout None waits as long as it takes; the call is
  forgotten only once its reply is recorded), `background` (above), `close`
  (the background work, then the pictures), `wire_views` (a saved
  picture's `step` is the undo history's depth), `frozen_for(id)` and
  `frozen_refusal(ids, *properties)` (the `FROZEN` payload when a write to a
  property the job froze on one of the sections would change it: `ids`,
  `properties`, `message`), `emit_results` (atomic).
  `ingest`, `apply_host_inputs` (the host's inputs, each section named
  checked; `spec.frozen` must name sections that are here, a frozen position
  must have been supplied, and a frozen affine without a supplied transform
  carries `host_transform()`, `kind` `"host"`), `changed_inputs` (above;
  `frozen` counts as a changed input). New sections remain unplaced until
  the caller supplies or writes positions; no positions are interpolated
  or extrapolated around supplied ones. Saved positions are preserved on
  resume.
  A host's `inputs.damaged` note becomes the section's `damage_note` only
  (`apply_host_inputs`): a section is damaged when it has marked regions. The submit
  gates, `submit_errors(state, spec, breaks)`, in order and
  only for the tasks that are on: `MISSING_POSITIONS` (no position),
  `STRICT_INTERVAL` or
  `INTERVAL_BREAKS_UNSUPPORTED`; `MISSING_TRANSFORMS`;
  `MISSING_DEFORMATIONS`, only where the host requires a deformation on every
  section (`NonlinearSpec.require_deformation`). The sections whose
  deformation is frozen need no deformation and no image correction.
- `formats.py` — the public files (above): `registration_document`,
  `section_entry`, `parameters` (the truth in public units),
  `applied_deformation`, `maps_status`, `write_registration`,
  `write_section_maps` (float maps as ImageJ hyperstacks, deflate without
  the floating-point predictor, which ImageJ 1.x cannot read;
  `FLOAT_COMPRESSION`), `write_labels` (the uint32 ids, the uint16 Fiji
  index with the atlas colours of indexes 1-255 as its lookup table, the
  csv), `derived_files` (the CLI's artifacts after submit), the file names
  (`SECTION_FILES` by artifact kind, `QUICKNII_FILE`, `VISUALIGN_FILE`).
- `imports.py` — a linear registration made elsewhere, read (never
  written to the job): `read_registration` (QuickNII/VisuAlign/DeepSlice
  JSON, QuickNII/DeepSlice XML, DeepSlice CSV, a job's `registration.json`),
  `match_sections` (exact name, then stem ignoring case, then one `_sNNN`
  section number; `AmbiguousMatch` refuses doubles), `import_placements(source,
  workspace, section_ids=, orientations=)` -> `ImportResult`
  (per-section `ImportedPlacement`: position, per-section pitch and yaw,
  flip, quarter turn, six numbers, pixel size and its source, the imported
  map on the job's file, VisuAlign markers rescaled to the file and raw;
  `transform()` the stored-transform dict, kind `imported`; `unmatched`,
  `missing`, `refused`). Geometry through `job/quint.py`'s inverses and
  `core/import_geometry.py`; `docs/file_formats.md` ("Importing a
  registration made elsewhere"). `registration_inputs(source,
  workspace)` -> `RegistrationInputs`: the result as `JobSpec.inputs`
  (`positions`, `angles` stack-wide when every section's agree to within
  `ANGLE_EPSILON_DEG` (the median), else per section, `orientation`,
  `transforms` with their `physical` knobs, and `pixel_size_um` when neither
  the caller nor the files give one: the imported median, every section
  then placed against it), the `report` the doors show (sections and how
  they matched, `unmatched`, `missing`, `refused`, `markers_imported`
  False, `warnings`: including `MARKERS_NOT_IMPORTED` when the file has
  VisuAlign markers) and the full `ImportResult`; `ValueError` when nothing
  is placed. The doors call it through `doors.jobs.with_registration`.
- `views.py` — `ViewStore` (`Job.views`): every picture the model was
  shown. The hook every door uses is `ViewStore.shown(tool,
  atlas_of=)`: a context manager that collects what `core.layers` notes
  while the door's operation runs; the door hands it the pictures it sends
  (`Shown.show(pictures, arguments=, call_id=)`; after the block `Shown.names`,
  `numbers` and `captions` say what was saved) and the store queues each
  with its note (the atlas, for a placement picture's layers, asked of
  `atlas_of` only then; a block that raises saves nothing; a failed queue is
  logged, never raised). The tool door's `toolbox._saves_views` wraps every
  tool in it, so the ADK and MCP tools save the same way, and a CLI or
  script would too. The ADK driver queues the opening's message images
  (`engine.save_opening`), the MCP door each `show_stack` page's image
  blocks (`server.save_page`), as the bytes sent, through `save`. Per picture a folder `<seq>_<tool>[_<mode>]` (six-digit
  sequence across the job's life, continued on reopen) with `view.jpg` (the
  exact bytes the door sent: the same `core.jpeg.encode_jpeg` on the same
  picture, or the door's own bytes), `view.json` (format 1: tool, call
  number and id, arguments, `caption`, `step`, `recipe`, sections, mode, picture size and bytes, `frame`)
  and, for a placement picture or an `fireants_syn` picture (its note's
  `warp`, `core.layers.warp_layers`), `labels.tif` (uint32, zlib) and
  `borders.png` (8-bit coverage), plus `residual.tif` for an `fireants_syn`
  picture that shows its deformation (`has_frame` decides). One section → `sections/<stem>/views/`,
  else `views/`. One line per picture in `views.jsonl` (`seq`, `name`,
  `path`, `tool`, `call`, `sections`, `mode`, `layers`, and `caption`, `step`,
  `recipe` when set). `recipe` is `{"renderer": name, "args": {...}}`, enough to
  redraw the picture (from the note's `recipe=` / `caption=`, or the `recipes` /
  `captions` lists of `save`); `step` is `save(step=)` or `ViewStore.step_source()`.
  `lookup(seq)` and `latest(exclude_tool="zoom")` return a `PictureRecord` from
  the index. One background
  thread per store, ending when its queue is empty; `flush` (`ops.submit`,
  `Job.emit_results`, `Job.close`, the library after every verb) and
  `flush_all` (interpreter exit) wait for it. `at_exit(fn)` runs a function
  at interpreter exit while threads can still start: Python's threading
  exit hooks, which run before `concurrent.futures` stops taking work (the
  writer's TIFF encoder and an image-model call use thread pools; plain
  `atexit` is too late), `concurrent.futures.thread` imported first so its
  hook runs after ours; `flush_all` and the library's open jobs use it
  (`flush_all` runs once more from `atexit`, for a picture a still-running
  thread queued after the first flush). A failed write is logged and skipped; only the numbering and
  queueing run on the tool's thread. `DiscardedViews` writes nothing (a
  dry run) but reads the pictures the job folder already holds and keeps the
  index of its own in memory after them (its `save` returns the names a saved
  picture would have), with the newest of its own pictures' images, for `lookup`
  / `latest`, so a dry run resolves the picture numbers a real call would. `captured()` collects every
  picture any store queues inside the block (`Saved`: its folder,
  whether it gets layers and a residual, its note's sections and mode, its
  index among the call's pictures; `files()` lists `view.jpg`,
  `view.json` and the layers with their kinds); `artifacts(saved, tool)`
  turns them into the `{"path", "kind", "index", "label"}` entries the agent
  CLI and the library report (and a warning per unsaved picture). The numbering reads the index as it grows
  (each save reads the lines appended since the last), so two stores on one
  folder (a running agent and a CLI call) continue each other's numbers, and the numbers are reserved
  across processes before anything is written (`views.seq`, the next picture
  and last call numbers, under its own file lock `views.seq.lock`;
  `ViewStore._numbering`), so two stores that queue pictures before either
  writes never share one.
- `browse.py` — read-only browsing of the job folder for `ops/files.py`: `resolve`
  (a path relative to the folder, or absolute inside it; `..` and a symlink
  pointing out are refused with `PathRefused`), `list_entries`, `search`
  (regex, else plain text; case-insensitive; text files only), `read_lines`,
  `kind_of` (dir/text/picture/binary), and the caps (`LIST_ENTRIES`,
  `SEARCH_MATCHES`, `READ_LINES`, `TEXT_BYTES`, `LINE_CHARS`). A walk skips
  symlinks leaving the folder.
- `quint.py` — QUINT JSON: `job_export` (QuickNII / VisuAlign JSON of the
  placed sections, each anchoring from its exact pixel -> atlas map, so any
  plane and cutting angle; `exports/quicknii.json` and `visualign.json`,
  through `ops.exports`), the atlas micrometre <-> QuickNII voxel
  conversions and their inverses (read by `imports.py`), the `.cutlas` target per
  atlas (every Allen resolution exported to the 25 um target, its voxel
  grid rescaled; `ccfv3augmented_mouse_25um` to the same target on the Allen
  grid, shifted by `core.atlas.images.AUGMENTED_AP_OFFSET_MM`, in both
  directions).

The frame record and the on-demand coordinate map are the core's
(`core/layers.py`, `core/CLAUDE.md`).

## Landmark state and derived file

Each `SliceState.landmarks` holds its authoritative `next_id` and `pairs`.
Checkpoint refreshes plane-stale flags independently of appearance; the same
write regenerates `sections/<stem>/landmarks.json` through
`formats.write_landmarks`. That JSON is derived and never read back, including
on undo/redo. Pair IDs, both canonical endpoints, original picture numbers and
pixel coordinates, exact endpoint geometry and registration context stay in
state. Landmark applications use normal saved `DeformableRecord`s, so the job's
maps, QUINT exports and ABBA consumers follow the existing residual-field path.

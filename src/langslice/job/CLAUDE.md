# LangSlice `job/` — the job folder

Package guide for `src/langslice/job/`. The repo-level `CLAUDE.md` holds the
project-wide rules. `AGENTS.md` here is a verbatim copy — edit one, mirror to
the other.

The job layer: the `Job` (`job.py` on `checkpoint.py`, `undo.py` and
`border_traces.py`: the state, undo, the checkpoint, ingest;
`host_inputs.py`: the host's inputs; `gates.py`: the submit gates;
`state_file.py`: `state.json`), the job's settings (`settings/`), the
registry of loaded tool plugins (`plugins.py`), where its files live and how
they are named, indexed and filled with pictures, the public files rendered
from the state, and the QUINT (QuickNII/VisuAlign) reading and writing.

## The job's settings (`settings/`)

- `schema.py`: one row per setting (`SETTINGS`, 44 rows of `Setting`): its
  name (`atlas`, a section's `position.thickness_um`, a supplied answer's
  `inputs.positions`), its type and the job's default, the doors that may
  set it (`run`, `init`, `mcp`, `library`, `abba`), its `job.json` key, the
  library's keyword, its command-line flags in the commands' order, the
  values it takes when they are a fixed list, and a door's own default where
  it is not the job's (`door_defaults`, in that door's terms). It also holds
  the form's lists (`ALL_TASKS`, `DEFAULT_TASKS`, `PLANES`,
  `IMAGE_RESOLUTIONS`, `MAX_PARALLEL_TRANSFORMS`, the commands' default
  tasks `CLI_TASKS` and `REGISTRATION_TASKS`) and imports nothing of
  LangSlice. Read by the commands' flags (`doors/commands/flags.py` adds
  them in table order with their help text), and so by `langslice-job
  schema init`; by `JobSpec.from_dict` (the keys `job.json` takes,
  `spec_keys`); by the LangSlice window's prompt generator (each flag's
  spelling, `flag_spellings`). `create_job`'s keywords are checked against
  it by `tests/test_settings_schema.py`, which also pins the table to
  `JobSpec`'s fields, types and defaults.
- `spec.py`: `JobSpec`, the filled form, and its task sections
  (`PositionSpec`, `TransformSpec`, `NonlinearSpec`), validated when made
  (`supplied_angles`, `supplied_pixel_sizes`, the disabled list through
  `tool_table.refusal`, the plugins by loading them). `image_settings()`
  makes the core's `ImageSettings` (`core/workspace.py`) from it;
  `missing_transform()` words what a fit says of a section with no written
  transform (`NO_TRANSFORM`, or `NO_TRANSFORM_LINEAR_OFF` rendered for the
  disabled list when Linear is off). Protocol metadata:
  `PositionSpec.thickness_um` and `interval_um`, and the state's millimetre
  equivalents, are optional (`None` when not supplied); known values are
  positive integers in micrometres, and strict spacing requires an
  interval.
- `tool_table.py` and `needs.py`: what a job can switch off (and, through
  `ToolDef.writes`, what its frozen properties switch off: `off_names`).
  `tool_table.py` lists every tool (`TOOLS`: `ToolDef` name, kind, task
  group, `when`, scripting / image-model / hidden, `options`, `essential`,
  `needs`; `tools/catalog.py` binds each to its operation), the names a
  `JobSpec.disabled` takes (`valid_names`: a tool, or `tool:option`;
  `is_off`, `options_on`) and `refusal(spec)`, the one-line reason a list
  cannot work (an unknown or essential name, every `look` mode off, a task
  with every tool that writes its answer off, a tool without the pictures
  it needs), which the spec raises as a `ValueError` when it is made. The
  kinds and groups and the `look` modes are the core's plain names
  (`core/tool_kinds.py`, `core/pictures/modes.py`). `needs.py`:
  `render(text, off)`, text written with what it needs (`[[?NAMES
  text||other]]`, `[[#NAMES=item||...]]` lists), dropped or swapped for the
  disabled list; with nothing off a text renders to itself without its
  marks, byte for byte. `tool_table.tools_for(spec)` is the per-spec view
  (`TOOLS` plus the tools of `spec.plugins`; `TOOLS` itself without
  plugins), which `valid_names(spec)`, `check_names`, `refusal(spec)`,
  `off_names`, `catalog(spec)` read, so a job's `disabled` list can name a
  plugin tool.
- `frozen.py`: what the agent may not change (`JobSpec.frozen`). The five
  properties a section has (`PROPERTIES`: `position`, `angles`,
  `orientation` (flip and quarter turn together), `affine`, `deformation`),
  the `"*"` key for every section, `clean` (validates and canonicalizes a
  freeze), `for_section` / `everywhere` / `sections_of` / `groups`,
  `TASK_PROPERTIES` (what each task writes), `derived` (what a task that is
  off freezes on every section), the model-facing `statement` (sections
  sharing a freeze are one clause: every section, their filenames, or their
  count) and `message` (a refusal's sentence, whose code is
  `RefusalCode.FROZEN`). A
  frozen property keeps the value `inputs` supplies, else its "none": the
  flat plane, no flip and no turn, the host's identity transform, no
  deformation; a position has none. `JobSpec.has(task)` is False for a task
  whose properties are all frozen on every section, `JobSpec.frozen_for(id)`
  includes what a task that is off supplies, and `JobSpec.switched_off` is
  `disabled` plus the tools that write (`ToolDef.writes`) only properties
  frozen on every section and the mirror under a frozen orientation
  (`tool_table.off_names`): every door and text renders by it.
- `labels.py`: a short label for people per tool and `tool:option`
  (`TOOL_LABELS`, `label`), per task (`TASK_LABELS`, `TASK_ABOUT`) and per
  tool group (`GROUP_LABELS`, with the group's task), for the LangSlice
  window's prompt generator (`hosts/gui/generator.py`, its only importer).
  Never model-facing; `tests/test_gui_generator.py` checks that every tool
  and option has one and no other name does, and that none appears in the
  pinned model-facing text. A new tool or option is a row here too.

`plugins.py`: the job's part of tool plugins (`docs/developers/plugins.md`;
the types are `core/plugins.py`'s). `check_tool` (the one-line reasons a
tool declaration is refused), `tool_def` (a plugin tool as a `ToolDef`:
`when` is its task being on and its `atlases` / `planes` fitting the job;
`not_offered` names the tools they exclude, which `tool_table.off_names`
counts as switched off so no text names them) and the registry of loaded
plugins (`LoadedPlugin`, `loaded`, `of_spec`, `tool_defs`, `tool_specs`;
`set_loader` is how `doors/plugins` registers the code that imports a
plugin: nothing here imports one by name). `JobSpec.plugins` /
`plugin_config` are validated by loading (`tool_table.tools_for`), so a bad
plugin is refused when the spec is made.

## The layer rule

Job layer: imports the core (`core.stack.stack`, `core.sections.discovery`,
`core.fits.deformation`, `core.pictures.layers`, `core.pictures.jpeg`, `core.maps.section_maps`, `core.atlas`,
...) and itself only. Never a tool (`tools/`), a door (`doors/`, the
built-in agent included), a host (`hosts/`), a provider or `google.*`,
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
| ABBA's `Register › LangSlice…` (Fiji connector, `abba.prepare`) | the ABBA project's snapshots, `~/.langslice/abba/<project key>/` | `~/.langslice/abba/<project key>/langslice/`, prepared (below) |

Two exceptions, both from `layout.locate_job_folder` (used by
`Job.open` and `engine.build_context`):

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
  again. Said once through the
  progress/emit log (`[job] ... cannot be created or written; the job
  folder is ...`) and recorded in the id's index entry under `fallback`
  (`image_folder`, `reason`).

Inside (`layout.py`, names as constants):

```
job.lock             the write lock (lock.py), held while a writer syncs and commits
job.json             settings: the JobSpec under "spec", format_version (1),
                     created_at, image_folder; the user's "notes" (every door
                     gives them to the agent); the agent CLI's "viewer"; under
                     "host" a prepared ABBA job's facts (format 3, kind
                     "abba", "abba": the session)
state.json           the checkpoint (Stack, state format 5): THE TRUTH
registration.json    its public rendering (formats.py), rewritten on every checkpoint
history/undo.json    the undo and redo steps (undo_steps.py)
sections/<stem>/     per section; <stem> is the image filename's stem (the whole
                     filename when two images share a stem)
  history.jsonl              every write that changed the section, its row and
                             registration number (history.py); read back by
                             undo, redo and restore_registration
  deformable/<key>/          deformation records (RecordStore): record.json +
                             warp.ome.zarr, the parent a sibling it names;
                             one no history line names is deleted
  images/<name>.png          made section images (core/sections/images.py: the
                             opening preprocessed, make_image's, a script's
                             add_image); never deleted, <name>.2.png on a remake
  masks/<name>.npz           section masks (core/sections/masks.py); never deleted
  border_trace/<key>/        trace_borders calls, attempt-NN inside
  views/<seq>_<tool>_<mode>/ pictures of this section the model saw
  coords.tif, labels.tif,    the section's maps (formats.py), written only by
  labels_fiji.tif,           export_maps (never by a write or by submit);
  labels.csv,                wherever the section maps into the atlas volume
  residual.tif, maps.json
views/<seq>_<tool>_<mode>/   pictures of several sections or none (atlas, sheets,
                             opening strips, show_stack pages, launch's strips)
views.jsonl          append-only index of every saved picture
views.seq            the picture/call numbers handed out (+ views.seq.lock)
exports/             linear_results.json (the run's result; spec.out overrides
                     the path), spend.json (the agent run's spend account),
                     quicknii.json and visualign.json (with the maps) and
                     the section images they name (<stem>_sNNN.png),
                     traces/<stem>.npz (each traced section's image-model reply
                     on its maps' grid, with the maps),
                     plugins/<tool>/ (copies of the layers plugins left),
                     dmc_brainmap/<animal>/ (a DMC-BrainMap animal folder,
                     only on request: tools/dmc_export.py)
plugins/<tool>/      a tool plugin's own files and layers (layers.json, layers/);
                     the only place a plugin tool writes
logs/events.jsonl    one line per open (`resumed`: whether a checkpoint was read), per
                     change of the plugins' record, and a `tool_start` / `tool_end`
                     pair per tool call on every door (`tools/events.py`,
                     `EventLog`; `docs/reference/file_formats.md` has the fields)
logs/runs/<id>.json  an agent-CLI background run (`--background`), its stderr in <id>.log
logs/calls.jsonl     one line per agent-CLI call (verb, arguments, outcome, an error's
                     code and message, artifacts, `reply`)
logs/calls/<n>_<verb>.json  each agent-CLI call's envelope as printed
logs/launches/<n>_<role>.md  the exact launch prompts of `langslice-job FOLDER launch` (`LAUNCHES_DIR`)
trace/               agent transcripts (`TRACE_DIR`, `JobLayout.trace_dir`): the
                     built-in agent's trace, a coding agent's session copied in by
                     `langslice session-end` (`doors/commands/transcripts.py`); the
                     agent's file tools never show or open it (`browse.HIDDEN`)
env/                 the job's own Python environment (`ENV_DIR`; `doors/agent_cli/environment.py`)
AGENTS.md, CLAUDE.md the reference card for coding agents, identical, generated
                     (`doors/reference_card.py`) and rewritten when stale
TOOLS.md             every verb of the job, its description and arguments (by `init`, `launch`)
```

**The public files (`docs/reference/file_formats.md` has every field).**
`state.json` stays the one working source; `public_files.py` renders it for scripts and other programs, in BrainGlobe micrometres (the
atlas's axis order, voxel `i`'s centre at `i * resolution`) and image-file
pixels `[row, col]`, and never reads any of it back. `Job.checkpoint`
writes the state and owes `registration.json`; it is rewritten atomically
once, as the outermost hold of the job folder's lock ends
(`FolderLock.before_release` calls `Job._write_derived`, still holding the
lock), so a write that checkpoints twice in one hold (the write, then the
deformations it made stale) maps the stack once, from the state it leaves
(`public_files.write_registration`, through `Job.workspace`, which `Job.open`,
`Job.load` and the tool door set; a failure is logged, never raised): per
section its parameters in public units (atlas, plane, orientation, the six
numbers and their knobs, the applied deformation's record), the file
pixel -> atlas matrix of its linear placement (`core.maps.section_maps`), why it has none
(`problem`), and its written maps with whether they are still `current`
(`maps.json`'s `parameters_digest`). The maps (`write_section_maps`:
`coords.tif`, `labels.tif`, `labels_fiji.tif` + `labels.csv`,
`residual.tif`, `maps.json`) are written only by `tools.exports.export_maps`
(on demand; a section's `maps` entry is null until then), the exports by it
and at submit (`tools.exports.export_stack`). `Job.persist` False
writes none of it.

**Resuming with other inputs is refused.** `Job.open` with `spec.resume`
on a folder whose checkpoint was made from other supplied inputs
(`JobSpec.inputs` and `JobSpec.frozen`, compared with the checkpoint's own
spec: `job.changed_inputs`) raises `job.InputsChanged` (a `ValueError`) before
anything is written, naming the keys that differ and how to start fresh
(`START_FRESH`: `--no-resume`, `resume=False`); resuming would have kept the
old inputs and dropped the new ones. The same inputs resume as before; the
agent CLI's `init` answers `INPUTS_CHANGED` (exit 3).

**A job the ABBA connector prepared.** `abba.prepare`
(`doors/abba/prepare.py`) writes `job.json` (the settings with ABBA's
inputs, the user's ABBA registrations frozen, and `host` with `kind`
`abba`, `host_inputs.PREPARED_HOST_KIND`) and no checkpoint. Every open of
such a folder, before its first checkpoint and after, goes through
`host_inputs.opening_spec` inside `Job.open`: `prepared_spec(saved,
given_spec, frozen_given=)` takes each setting of the table from the
opening where it differs from the job's default, else from the saved
settings; `inputs` and `host_preprocessing` are always the saved ones (an
input the opening adds, a damage note, kept beside them); `frozen` is the
opening's only when it gave one explicitly (`frozen_given`: the agent CLI's
`init --frozen`, even `'{}'`); the image folder, `job_dir` and `resume` are
the opening's. `Job.open` gives the workspace the settings it opened with
(`Workspace.settings`), and the doors go on with `job.spec`. So `init`,
`linear run`, MCP and `create_job` all open a prepared job with ABBA's
inputs, and a start over (`--no-resume`) keeps them.

**Opening a job without writing.** `Job.open` writes `job.json` and a
first checkpoint. `Job.load(spec, workspace, folder=)` opens an
existing folder as it stands (the checkpoint and its undo steps; nothing
rewritten, `FileNotFoundError` without a checkpoint, another image
folder's job refused): the agent CLI and the library open a job per call
this way (`doors/jobs.py`), so a call beside a running agent changes
nothing until its first write. It stamps the state file and
`history/undo.json` before it reads them (the constructor's `seen`), so a write that
lands during the load is reloaded by its first write, not overwritten.
`Job.persist` False (`Job.load(...,
persist=False)`, the CLI's `--dry-run`) writes nothing at all: no
checkpoint, no history line, no undo file, and `pictures.DiscardedViews`
saves no picture.
`Job.engine_log` is where the engines' own output goes while they fit
(`logs/engine.log`, `core.fits.engine_log`); None for a job that persists
nothing.

**One writer at a time, across processes.** Every write goes lock -> sync -> apply -> commit -> unlock: `Job.writing()` holds
the folder's `lock.FolderLock` (`job.lock`, `filelock`: `fcntl` on Linux
and macOS, `msvcrt` on Windows, released by the OS if a process dies;
reentrant in its thread; `LOCK_TIMEOUT_S` 300 then `JobBusy`) and runs
`Job.sync` first. `commit`, `checkpoint` and the undo file's save take it too;
`undo`/`redo`, `clear_stale_deformations` and the landing of image
corrections (`settle_border_traces`, `wait_image_job`: the calls run
outside it, their results land under it) are written inside it, and so
is `Job.open`'s read and first checkpoint. A missing or unwritable folder
is used without the lock. After a sync the section records are new
objects: resolve them by id inside the block.

**A read never waits for another process's writer.** `Job.reading()` (the
tool door's guard for a read verb) keeps this process's other threads out
(`FolderLock.threads`) and takes the folder lock only when it is free
(`held_if_free`): then it syncs as a write does. While another process holds
it (a long fit applying, a plugin tool that holds it while it computes), it
reads the state file as that process last saved it (`_peek`: no undo file
read, no undo step), and keeps the state it held before (`_peeked`): the next
locked `sync` puts that back first and then reloads as usual, so the other
process's write enters this job's undo steps as that job's step.

**Paths are relative.** Every path a job file stores (a deformation's
`record`, an image correction's `artifact_dir` / `artifact_paths`, the views
index) is relative to the job folder (`JobLayout.relative` /
`JobLayout.resolve`; `job.state_file.state_paths` walks a state's
paths). The trace identity in a deformation cache key is the stored
(relative) artifact directory, so keys survive a move of the folder.
`Stack.image_folder` itself stays absolute (rewritten by `Job.open`
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
  the work's kind (`Job.acting`), so its history lines are made by it. The
  notice is
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
  `JobLayout` (the folder, every name, `section_dir`, `section_history`,
  `undo_file`, `deformable_dir`, `masks_dir`, `images_dir`, `border_trace_dir`, `section_views_dir`,
  `relative`/`resolve`, `ensure`, `log_event`), `job_folder_for`,
  `section_dirname`, `read_job_file` (refuses a newer `format_version`),
  `write_job_file` (fields over what it holds; `created_at` kept).
- `plugin_store.py` — what tool plugins keep in the job folder
  (`docs/developers/plugins.md`): `register_layer(layout, record, tool, call, layer,
  persist=)` writes a layer's file under `plugins/<tool>/layers/`, lists it in
  the tool's manifest `layers.json` and puts its reference in the section's
  `layers` (the state: undo hides it, the files stay); `read_manifest`,
  `layer_rows`, `export_layers` (copies the shown layers to
  `exports/plugins/<tool>/`, called by `tools.exports`). `job.json`'s `plugins`
  (the provenance `doors/plugins` records) is written with `write_job_file`.
  `JobLayout.plugins_dir` / `plugin_dir(tool)`. `Job.pushed` counts undo steps
  and `Job.merge_steps(mark)` makes the steps since `mark` one (a plugin tool's
  call is one step however many operations it ran).
- `history.py` — the record of every write (`docs/reference/file_formats.md`,
  "Section history" and "Undo"). `SectionHistories(layout)`
  (`Job.histories`): each section's `history.jsonl`, one `Row` per write that
  changed it (`seq`, `step`, `by`, `at`, `registration`, `digest`,
  `format_version`, then the section's row of `state.json`); `rows` (cached
  by the file's stamp; a line cut short skipped; an older
  `registrations.jsonl` read in its place), `row(seq)`, `number(id,
  digest)`, `registrations` (the first line of each number),
  `registration_number(state, section)`, `append` (under the job's lock; the
  number the registration has, or the next one when `numbered`; writes an
  older list's lines first and removes it; after a line cut short it starts
  a new line), `collect_records` (deletes the section's record folders no
  line or the section names, parents kept; nothing while an older
  LangSlice's undo steps are in `history/`, which may name records no line
  does). `registration_value` (orientation, transform and the deformation
  while its `linear_key` holds; None with neither), `registration_digest`,
  `guessed_maker` (what a `host` line's registration says made it), `HOST`,
  `SCRIPT`, `set_head(section, fields)` (puts row fields back in place: undo,
  redo and `restore_registration`), `section_row`.
- `undo_steps.py` — `history/undo.json`. `UndoFile(layout)`
  (`Job.undo_file`): format 1, `next_step`, `undo` and `redo` lists of
  `UndoStep` (`step`, `by`, `at`, `sections` {id: seq before},
  `stack_before` (the stack's fields but the notes), `notes_before`
  (`NotesBefore`: `keep` the first notes, then `then`; the change only, so
  the file grows with what each write changed)); `load` returns `UndoLists`
  (an unreadable or newer file: no steps, its `notice` says why and
  `writable` False, so `save` leaves it; an older LangSlice's `index.json` /
  `step-*.json` (`older_undo_steps`): no steps, the notice once per
  process, the files left on disk), `save`; `combined_steps` (the earliest
  line per section and field wins, the notes' changes composed:
  `merge_steps`, `amend`). No cap on the steps.
- `recent.py` — the jobs opened on this computer,
  `~/.langslice/recent_jobs.jsonl` (`folder`, `image_folder`, `atlas`, `at`;
  one line per `Job.open`), for the GUI's job list: `remember`, `read`,
  `recent_file` (`LANGSLICE_RECENT_JOBS`, `RECENT_ENV`: another file, or
  `off`; the test suite sets it off); past `KEEP_BYTES` rewritten with the
  newest line of each of the newest `KEEP_JOBS` folders. Never fails an open.
- `index.py` — job folders kept under `~/.langslice/jobs/` because their
  image folder could not be written: `~/.langslice/jobs/<id>.json`
  (`job_id`, `job_folder`, `created_at`, `fallback`; owner-only).
  `folder_id` (an image folder's stable id), `register`, `lookup`
  (refuses a newer entry).
- `state_file.py` — `state.json`, and the one place a `Stack` becomes JSON
  and comes back: `stack_to_json` (the stack's fields, one row per section,
  the cutting angles once on the stack while every section shares them,
  `layers` / `masks` / `images` left out of a row while empty),
  `stack_from_json` (unknown keys refused, a row without its own angles
  given the stack's), `serialized_mixed_angles`; `write_checkpoint` (atomic, versioned,
  `STATE_FORMAT_VERSION` 5), `read_checkpoint(path, upgrade=)`,
  `current_state(data, upgrade=)` (every format through *upgrade*, the
  job's `migrations.FolderMigration`; a newer LangSlice's format is refused
  and asks for an update, an older one no migration reads (`older_format`)
  for a new job, `--no-resume` or `resume=False`; without *upgrade* only the
  current format reads), `stack_fields` / `set_stack_fields` (the stack's
  own fields without its sections; an undo step's `stack_before` is them
  but the notes),
  `json_object` (a JSON text's object, or None), `write_json_atomic` (every
  job file's writer),
  `state_paths` and `relative_to` (a state's stored paths made relative to
  the job folder).
- `migrations.py` — old job folders read as the current format: the table
  `STATE_MIGRATIONS` (rows of `Migration`: a name, the format it reads, the
  step to the next format; `state_3_to_4` makes a format-3 state's
  `appearance` recipes into section image files and entries, drops the raw
  channels' display properties, names a mask's `channel` `section_image`).
  `state_4_to_5` names a section's `image_correction` `border_trace`; an
  old trace stays in `sections/<name>/image_correction/` with the paths
  stored to it, new traces go under `border_trace/`.
  `FolderMigration(workspace, layout)` is a job's `upgrade`, which every
  read of the state file and of a history line of an older format goes
  through: it applies the
  rows in order from a file's format, refuses one no row reads
  (`state_file.older_format`), then drops at every format the keys no
  reader uses (`drop_removed_keys`: `REMOVED_STACK_KEYS` `appearance`,
  `REMOVED_SECTION_KEYS` `position_source` and `damaged`), logging one
  warning per key the first time (`dropped`, with the sections that held
  it). It keeps one memo of the images it made across the checkpoint and
  the history lines (`seed` takes a current state's entries); `used` says
  it changed what it read, `applied` the rows it ran. Without a workspace
  (`viewer_stack`, the GUI's read of `state.json`) it makes no image and
  writes nothing: a format-3 section is read without its images.
  `lines_from_registration_list` reads an older LangSlice's
  `sections/<stem>/registrations.jsonl` (`OLD_REGISTRATION_LIST`) as the
  section's history lines, numbers kept; an older `history/index.json` and
  step files are not read (`undo_steps.UndoFile` warns once per process).
  `without_removed_settings(spec)` leaves out of saved settings what
  LangSlice no longer has (`REMOVED_SETTINGS`, the `REMOVED_TASKS` task
  `reorder`) and names them; the doors read `job.json` through it
  (`doors.jobs.read_spec`, the reference card) and the job logs them
  (`warn_removed_settings`). The module text names the two old shapes the
  core reads itself (a deformation record's format, its fit settings'
  `stiffness`) and why.
- `job.py` — `Job`, assembled from one class per seam, each built on the
  one below: `checkpoint.py` `JobCheckpoint` (the state and its files,
  `checkpoint`, the observers, `histories` and `undo_file`,
  `wire_views`), `undo.py` `JobUndo` (`writing` / `reading` /
  `sync`, `snapshot` / `commit` / `amend`, `undo` / `redo`), `border_traces.py`
  `JobBorderTraces` (the image-correction jobs and `background`, the
  background work that waits for them, whose threads give way to a lock
  holder through `_lock_abort`); `job.py` adds `open`,
  `load`, the rules and the results. Its methods: `open` (resume or ingest, the opening `preprocessed`
  image of every section without one, `core.sections.images.ensure_opening`,
  outside undo, then the first checkpoint; the open event records
  `resumed`; a folder the migrations changed adds a `migrated` event; the
  job added to the recent jobs list, `recent.remember`), `load` (as it stands, writing nothing, except once for a job
  the migrations change (an older state format, a removed key or setting)
  or without its opening images: then the migrated state is
  written at the current format, and `job.json` without its removed
  settings, under the lock, after reading the state file again there
  (`_write_upgrade`: another process may have migrated and written
  meanwhile; its state is kept), with a `migrated` event; a dry run, or a
  folder this process cannot write, keeps them in memory, though the image files the migration makes are written; the
  removed settings are logged either way), `upgrade` (the job's `FolderMigration`, which every read of the state
  file goes through), `workspace` (setting it gives a workspace without a
  `job_folder` this job's folder, where its made images are read), `writing` / `reading` / `sync` (above), `snapshot` / `commit`
  (a history line per section the write changed, by the tool `acting(tool)`
  names for the thread, else `script` (`actor`); one undo step; the record
  folders no line names deleted), `amend` (the stale deformations cleared
  after a write, joined to its step), `undo` / `redo` (put the lines a step
  names back, `history.set_head`, and append lines of their own),
  `number_registration(section)` (the number of the registration it holds,
  numbered by a `host` line when it never was), `record_opening` (a fresh
  job's opening lines); `registered`, the journal of (thread, section,
  number) of each registration a write numbered or returned to, and
  `registered_since(mark)` for a door's reply), `checkpoint` (state, then `registration.json`, then the
  observers: `observe(fn)` calls *fn* with the state after every
  checkpoint and reload, the hosts' live views; one that raises is logged),
  `registration_numbers(ids)` (the numbered registration each section holds,
  what a saved picture records), the image-correction jobs (`start_image_job`, `settle_border_traces`,
  `wait_image_job`: timeout None waits as long as it takes; the call is
  forgotten only once its reply is recorded), `background` (above), `close`
  (the background work, then the pictures), `wire_views` (a saved
  picture's `step` is the undo history's depth), `frozen_for(id)` and
  `frozen_refusal(ids, *properties)` (the `FROZEN` `Refused` when a write to a
  property the job froze on one of the sections would change it: `ids`,
  `properties`, `message`; `busy_refusal(ids)` the `SECTION_BUSY` one),
  `emit_results` (atomic).
  `ingest`; in `host_inputs.py`, `apply_host_inputs` (the host's inputs, each section named
  checked; `spec.frozen` must name sections that are here, a frozen position
  must have been supplied, and a frozen affine without a supplied transform
  carries `host_transform()`, `kind` `"host"`), `changed_inputs` (above;
  `frozen` counts as a changed input). New sections remain unplaced until
  the caller supplies or writes positions; no positions are interpolated
  or extrapolated around supplied ones. Saved positions are preserved on
  resume.
  A host's `inputs.damaged` note becomes the section's `damage_note` only
  (`apply_host_inputs`): a section is damaged when it has marked regions. The submit
  gates, `submit_errors(state, spec, breaks)` (the first gate's `Refused`,
  status `refused`, or None), in order and only for the tasks that are on: `MISSING_POSITIONS` (no position),
  `STRICT_INTERVAL` or
  `INTERVAL_BREAKS_UNSUPPORTED`; `MISSING_TRANSFORMS`;
  `MISSING_DEFORMATIONS`, only where the host requires a deformation on every
  section (`NonlinearSpec.require_deformation`). The sections whose
  deformation is frozen need no deformation and no image correction.
- `public_files.py` — the public files (above): `registration_document`,
  `section_entry`, `parameters` (the truth in public units),
  `applied_deformation`, `maps_status`, `write_registration`,
  `write_section_maps` (float maps as ImageJ hyperstacks, deflate without
  the floating-point predictor, which ImageJ 1.x cannot read;
  `FLOAT_COMPRESSION`), `write_labels` (the uint32 ids, the uint16 Fiji
  index with the atlas colours of indexes 1-255 as its lookup table, the
  csv), `derived_files` (the CLI's artifacts after submit, the image
  copies `quicknii.json` names among them, `quint_copies`), the file names
  (`SECTION_FILES` by artifact kind, `QUICKNII_FILE`, `VISUALIGN_FILE`).
- `imports/readers.py` — a linear registration made elsewhere, read (never
  written to the job): `read_registration` (QuickNII/VisuAlign/DeepSlice
  JSON, QuickNII/DeepSlice XML, DeepSlice CSV, a job's `registration.json`),
  `match_sections` (exact name, then stem ignoring case, then the stem
  without a final `_sNNN` (`export copy`: an export's image copy), then one `_sNNN`
  section number; `AmbiguousMatch` refuses doubles), `import_placements(source,
  workspace, section_ids=, orientations=)` -> `ImportResult`
  (per-section `ImportedPlacement`: position, per-section pitch and yaw,
  flip, quarter turn, six numbers, pixel size and its source, the imported
  map on the job's file, VisuAlign markers rescaled to the file and raw;
  `transform()` the stored-transform dict, kind `imported`; `unmatched`,
  `missing`, `refused`). Geometry through `job/quint.py`'s inverses and
  `core/maps/imported.py`; `docs/reference/file_formats.md` ("Importing a
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
- `pictures.py` — `ViewStore` (`Job.views`): every picture the model was
  shown. The hook every door uses is `ViewStore.shown(tool,
  atlas_of=)`: a context manager that collects what `core.pictures.layers` notes
  while the door's operation runs; the door hands it the pictures it sends
  (`Shown.show(pictures, arguments=, call_id=)`; after the block `Shown.names`,
  `numbers` and `captions` say what was saved) and the store queues each
  with its note (the atlas, for a placement picture's layers, asked of
  `atlas_of` only then; a block that raises saves nothing; a failed queue is
  logged, never raised). The tool pipeline's `_save_pictures` step runs every
  tool in it, so the ADK and MCP tools save the same way, and a CLI or
  script would too. The ADK driver queues the opening's message images
  (`engine.save_opening`), the MCP door each `show_stack` page's image
  blocks (`server.save_page`), as the bytes sent, through `save`. Per picture a folder `<seq>_<tool>[_<mode>]` (six-digit
  sequence across the job's life, continued on reopen) with `view.jpg` (the
  exact bytes the door sent: the same `core.pictures.jpeg.encode_jpeg` on the same
  picture, or the door's own bytes), `view.json` (format 1: tool, call
  number and id, arguments, `caption`, `step`, `recipe`, sections, mode, picture size and bytes, `frame`)
  and, for a placement picture or an `fireants_syn` picture (its note's
  `warp`, `core.pictures.layers.warp_layers`), `labels.tif` (uint32, zlib) and
  `borders.png` (8-bit coverage). One section → `sections/<stem>/views/`,
  else `views/`. One line per picture in `views.jsonl` (`seq`, `name`,
  `path`, `tool`, `call`, `sections`, `mode`, `layers`, and `caption`, `step`,
  `recipe` when set). `recipe` is `{"renderer": name, "args": {...}}`, enough to
  redraw the picture (from the note's `recipe=` / `caption=`, or the `recipes` /
  `captions` lists of `save`); `step` is `save(step=)` or `ViewStore.step_source()`.
  `lookup(seq)` and `latest(exclude_tool="zoom")` return a `PictureRecord` from
  the index. Each picture of sections records `registrations`, the number of
  the registration each section was drawn under (`view.json` and the index
  line; `PictureRecord.registrations`): the note's `extra["registrations"]`
  (a `look` of a registration the section does not hold; empty for a
  proposal not written, which records none), else what
  `registration_source` answers (`Job.wire_views` sets it to
  `Job.registration_numbers`, the numbers the sections hold now).
  `save(tiles=True)`: a sheet's tiles, their JPEGs listed as artifacts of kind
  `tile` (`Saved.view_kind`), not `view`. One background
  thread per store, ending when its queue is empty; `flush` (`tools.submit`,
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
- `browse.py` — read-only browsing of the job folder for `tools/files.py`: `resolve`
  (a path relative to the folder, or absolute inside it; `..` and a symlink
  pointing out are refused, `Refused` `BAD_PATH`; one that does not exist,
  `NOT_FOUND` with `nearest`, the nearest folder above it that exists, and
  `holds`, at most `NEAREST_NAMES` of its names), `list_entries`, `search`
  (regex, else plain text; case-insensitive; text files only), `read_lines`,
  `kind_of` (dir/text/picture/binary), and the caps (`LIST_ENTRIES`,
  `SEARCH_MATCHES`, `READ_LINES`, `TEXT_BYTES`, `LINE_CHARS`). A walk skips
  symlinks leaving the folder. `HIDDEN` (`TRACE_DIR`, the agent transcripts;
  `is_hidden`): left out of every listing, search, walk and a missing path's
  `holds`, and a path in it (or a symlink into it) refused `BAD_PATH`
  whether or not it exists.
- `quint.py` — QUINT JSON: `job_export(rows, atlas, write_markers=)`
  (QuickNII / VisuAlign JSON of the placed sections, each anchoring from
  its exact pixel -> atlas map, so any plane and cutting angle, made
  perpendicular by `perpendicular_anchoring` (the nearest anchoring with
  `u` and `v` at right angles in the same plane, same centre: QuickNII and
  VisuAlign orthonormalize on load, Nutil and PyNutil do not, so all four
  then read one placement); with `write_markers` (`visualign.json`) the
  rows' deformation markers and the dropped shear carried onto that
  anchoring by `markers_on_anchoring` (`anchoring_pixel_map`; a section
  without a deformation gets a coarse grid, `SHEAR_MARKER_DIVISIONS`, only
  when the shear moves a corner more than `SHEAR_TOLERANCE_PX`);
  `exports/quicknii.json` and `visualign.json`, through `tools.exports`),
  the atlas micrometre <-> QuickNII voxel
  conversions and their inverses (read by `imports/readers.py`), the `.cutlas` target per
  atlas (every Allen resolution exported to the 25 um target, its voxel
  grid rescaled; `ccfv3augmented_mouse_25um` to the same target on the Allen
  grid, shifted by `core.atlas.images.AUGMENTED_AP_OFFSET_MM`, in both
  directions).
- `quint_images.py` — the PNG copies the QUINT files name, written beside
  them in `exports/` (QuickNII and VisuAlign open images from the file's
  folder, PNG or JPEG only): `section_numbers` (each filename's one `_sNNN`
  when every section of the stack has one and none repeats, else the stack order from 1: the
  files' `nr`), `copy_filename` (`<stem>_sNNN.png`, three digits, or
  `<stem>.png` when the stem carries that number), `write_copies` (each
  section's `preprocessed` image at the working copy's size, unoriented;
  the copies the previous `quicknii.json` named and this export does not
  are removed, `public_files.quint_copies`). The files keep the image
  file's `width` and `height`.

- `dmc_brainmap.py` — a section's linear map in DMC-BrainMap's terms
  (napari-dmc-brainmap 0.2.0b2's SHARPy-track `registration.json`):
  `dmc_frame` (its `xyz_dict` axes and bregma voxel for an atlas and plane,
  as its `get_xyz` / `get_bregma`; `DMC_BREGMA` its table), `dmc_placement`
  (from `pixel_to_atlas_um`: the whole-voxel depth shifts nearest the
  section's slopes, their angles, the centre voxel that best matches its
  depth, `z_mm`, and the point pairs of a 5 x 5 grid (finer when fewer than
  five fall on the atlas plane) placed half a voxel and half a resized pixel
  on for DMC's truncations; `ValueError` for a plane DMC refuses),
  `dmc_depth` (DMC's sheared plane), `dmc_decimals`, `natural_key` (natsort's
  order, which numbers DMC's images). Read back through DMC's own reader in
  `tests/test_dmc_export.py` (replicated there; DMC is not a dependency).

The frame record and the on-demand coordinate map are the core's
(`core/pictures/layers.py`, `core/CLAUDE.md`).

## Landmark state and derived file

Each `Section.landmarks` holds its authoritative `next_id` and `pairs`.
Checkpoint refreshes plane-stale flags independently of appearance; the same
write regenerates `sections/<stem>/landmarks.json` through
`public_files.write_landmarks`. That JSON is derived and never read back, including
on undo/redo. Pair IDs, both canonical endpoints, original picture numbers and
pixel coordinates, exact endpoint geometry and registration context stay in
state. Landmark applications use normal saved `DeformableRecord`s, so the job's
maps, QUINT exports and ABBA consumers follow the existing residual-field path.

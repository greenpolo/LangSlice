# LangSlice `job/` — the job folder

Package guide for `src/langslice/job/`. The repo-level `CLAUDE.md` holds the
project-wide rules. `AGENTS.md` here is a verbatim copy — edit one, mirror to
the other.

The job layer (layered refactor, phase 3c, 2026-10-03; folder move
2026-10-04). The `Job` itself (state, undo, checkpoint, gates) is
`job/job.py` (formerly `linear/job.py`) over `job/checkpoint.py` (formerly
`linear/checkpoint.py`), both described in the linear agent environment's
guide (`src/langslice/agent/CLAUDE.md`, "Files"); `job/quint.py` (formerly
`integrations/quint.py`) is the QUINT/QuickNII/VisuAlign writer
`ops.exports` calls, described in `src/langslice/hosts/integrations/CLAUDE.md`.
The rest of this package is where the job's files live and how they are
named, upgraded, indexed and filled with pictures.

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
| `langslice claude prepare FOLDER` | `FOLDER` | `FOLDER/langslice/` (id in the index) |
| `langslice mcp`, `start_job(image_folder=...)` | that folder | `<folder>/langslice/` |
| `langslice mcp`, `start_job(job_id=...)` | the saved job's | the index entry's job folder |
| ABBA worker `linear.run` (Fiji connector) | the connector's snapshot folder, a fresh `langslice-abba-*` temporary folder per run | `<snapshots>/langslice/` (or the spec's `job_dir`, or the read-only fallback); the result's `output_dir` names the one used |
| ABBA Claude mode (`claude.prepare`) | `~/.langslice/snapshots/claude-*/` | `<snapshots>/langslice/` (id in the index) |
| `langslice abba --linear` (Python launcher) | a fresh `langslice-abba-*` run folder under the spec's folder (or the system temporary folder) | `<run folder>/langslice/` |

Two exceptions, both from `layout.locate_job_folder` (used by
`Job.open`, `engine.build_context` and the Claude job preparation):

- **An explicit folder.** `JobSpec.job_dir` (`--job-dir PATH` on `linear
  run`, `mcp` and `claude prepare`) puts the job folder exactly there, same
  layout, e.g. one per benchmark arm on one dataset folder; the image folder
  is then never written (no migration from beside the images either). Two
  jobs never share a folder silently: `layout.check_owner` refuses a folder
  whose `job.json` names a DIFFERENT image folder (`ValueError`, "already
  holds the job of ..."); the same image folder's job is continued.
  `to_dict` leaves `job_dir` out when it is unset.
- **A read-only image folder.** When `<images>/langslice/` cannot be created
  or written (`layout.writable`: a probe file in an existing folder, else
  write access to the image folder; nothing is created by the check), the
  job folder is `~/.langslice/jobs/<id>/`, same layout. The id is the
  image folder's path hashed (`index.folder_id`), so a reopen finds it
  again; a saved Claude job uses its own id. Said once through the
  progress/emit log (`[job] ... cannot be created or written; the job
  folder is ...`) and recorded in the id's index entry under `fallback`
  (`image_folder`, `reason`). Old files beside a read-only image folder are
  not migrated.

Inside (`layout.py`, names as constants):

```
job.lock             the write lock (lock.py), held while a writer syncs and commits
job.json             settings: the JobSpec under "spec", format_version (1),
                     created_at, image_folder; a saved Claude job's job_id and
                     its own fields under "host" (kind, params, notes, trace_dir)
state.json           the checkpoint (StackState, state format 2): THE TRUTH
registration.json    its public rendering (formats.py), rewritten on every checkpoint
history/             undo/redo: index.json + step-NNNNNN.json, one per step
sections/<stem>/     per section; <stem> is the image filename's stem (the whole
                     filename when two images share a stem)
  deformable/<key>/          applied deformation records (RecordStore)
  image_correction/<key>/    trace_borders calls, attempt-NN inside
  views/<seq>_<tool>_<mode>/ pictures of this section the model saw
  coords.tif, labels.tif,    the section's maps (formats.py), written at submit
  labels_fiji.tif,           and by export_maps, never on an ordinary write;
  labels.csv, tissue.png,    over the section's filled outline (tissue.png: the
  residual.tif, maps.json    threshold's tissue estimate, not used to cut them)
views/<seq>_<tool>_<mode>/   pictures of several sections or none (atlas, sheets,
                             opening strips, show_stack pages)
views.jsonl          append-only index of every saved picture
views.seq            the picture/call numbers handed out (+ views.seq.lock)
exports/             linear_results.json (the run's result; spec.out overrides
                     the path), result.json (a saved host job's final result),
                     quicknii.json and visualign.json (with the maps)
logs/events.jsonl    one line per open and per migration
logs/runs/<id>.json  an agent-CLI background run (`--background`), its stderr in <id>.log
prompt.txt           a saved Claude job's copy prompt
AGENTS.md, CLAUDE.md the reference card for coding agents, identical, generated
                     (`doors/card.py`) and rewritten when stale
```

**The public files (formats phase, 2026-10-04; `docs/file_formats.md` has
every field).** `state.json` stays the one working source; `formats.py`
renders it for scripts and other programs, in BrainGlobe micrometres (the
atlas's axis order, voxel `i`'s centre at `i * resolution`) and image-file
pixels `[row, col]`, and never reads any of it back. `Job.checkpoint`
rewrites `registration.json` atomically after the state
(`formats.write_registration`, through `Job.workspace`, which `Job.open`,
`Job.load` and the tool door set; a failure is logged, never raised): per
section its parameters in public units (atlas, plane, orientation, the six
numbers and their knobs, the applied deformation's record), the file
pixel -> atlas matrix of its linear placement (`core.maps`), why it has none
(`problem`), and its written maps with whether they are still `current`
(`maps.json`'s `parameters_digest`). The maps (`write_section_maps`:
`coords.tif`, `labels.tif`, `labels_fiji.tif` + `labels.csv`, `tissue.png`,
`residual.tif`, `maps.json`) and the exports are written by
`ops.exports.export_maps`, at submit and on demand. `Job.persist` False
writes none of it.

**Resuming with other inputs is refused.** `Job.open` with `spec.resume`
on a folder whose checkpoint was made from other supplied inputs
(`JobSpec.inputs`, compared with the checkpoint's own spec:
`job.changed_inputs`) raises `job.InputsChanged` (a `ValueError`) before
anything is written, naming the keys that differ and how to start fresh
(`START_FRESH`: `--fresh`, `resume=False`); resuming would have kept the
old inputs and dropped the new ones. The same inputs resume as before; the
agent CLI's `init` answers `INPUTS_CHANGED` (exit 3). A door that saves a
job to be opened later checks at save time with
`refuse_changed_inputs(layout, spec)` (the same refusal, `inputs_changed`):
`claude prepare` (`doors.api.claude_jobs._write_job`, when `spec.resume`).
`claude prepare --fresh` marks the saved job `fresh` (job.json `host`);
the MCP door's first open of it (`server.open_folder_job`) runs `Job.open`
without resume, as `linear run --fresh`, and clears the mark. ABBA's
Claude mode (`prepare_claude`) never resumes (`prepare_linear` sets
`resume` False), so it is never refused.

**Opening a job without writing (phase 5).** `Job.open` writes `job.json`
and a first checkpoint. `Job.load(spec, workspace, folder=)` opens an
existing folder as it stands (the checkpoint and its history; nothing
rewritten, `FileNotFoundError` without a checkpoint, another image
folder's job refused): the agent CLI and the library open a job per call
this way (`doors/jobs.py`), so a call beside a running agent changes
nothing until its first write. `Job.persist` False (`Job.load(...,
persist=False)`, the CLI's `--dry-run`) writes nothing at all: no
checkpoint, no history, and `views.DiscardedViews` saves no picture.

**One writer at a time, across processes (phase 5 follow-up).** Every
write goes lock -> sync -> apply -> commit -> unlock: `Job.writing()` holds
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
absolute path. `layout.held_image_folder` resolves it (a default folder
written with an absolute path that no longer exists: its parent), the
CLI and library (`doors.jobs.read_spec`) open the images there, and when
they are gone say how to reattach (`langslice job NEW_IMAGES init
--job-dir FOLDER`); `check_owner` lets an explicit folder whose images no
longer exist be taken over by the images it is opened with.

## Files

- `lock.py` — `FolderLock(folder)` (`held()`, `LOCK_FILE` `job.lock`,
  `LOCK_TIMEOUT_S`, `JobBusy`): above.
- `layout.py` — `locate_job_folder` (above), `writable`, `check_owner`,
  `JobLayout` (the folder, every name, `section_dir`,
  `deformable_dir`, `image_correction_dir`, `section_views_dir`,
  `relative`/`resolve`, `ensure`, `log_event`), `job_folder_for`,
  `section_dirname`, `read_job_file` (refuses a newer `format_version`),
  `write_job_file` (fields over what it holds; `created_at` kept).
- `history.py` — `History`: the undo/redo stacks on disk. A step is written
  once, as its own file, when it enters the history (matched to the job's
  stacks by identity); `index.json` (format 1: `undo`, `redo`, `next`) is
  the only file rewritten per step, and step files it no longer names are
  deleted, so the folder holds at most `UNDO_DEPTH` (50) undo steps plus
  the redo side. A new step never takes the name of an existing file (a
  second job on the same folder). Why per-step files: the phase-2
  `linear_undo.json` rewrote every whole state on every call (megabytes per
  call on a 40-section stack at depth 50). An unreadable or newer history
  (index, any step, or step files without an index) starts the session
  without undo and is never deleted: `History.problem` says why (logged,
  and `Job.open` says it through the progress log), and `save` writes and
  deletes nothing while it is set; a fresh job reads a history before
  emptying it. Only a history that was read is pruned.
- `index.py` — saved host jobs by id: `~/.langslice/jobs/<id>.json`
  (`job_id`, `job_folder`, `host_channel`, `created_at`, and `fallback`
  when the job folder is under `~/.langslice/jobs/` because the image
  folder could not be written; owner-only). `folder_id` (an image folder's
  stable id). The
  channel token stays here, not in a folder that may be shared. `register`,
  `lookup` (refuses a newer entry), `legacy_dir`.
- `migrate.py` — opening an old layout upgrades it.
  `migrate_beside_images(layout)` (called by `Job.open` and by a Claude
  job's preparation): `linear_state.json`, `linear_undo.json`,
  `linear_results.json`, `deformable/<file>/<key>/` and
  `nonlinear/<file>/<key>/` beside the images move into the job folder
  (state to `state.json` at format 2, history split per step, records and
  calls into their section folders, results to `exports/`, every stored
  path rewritten to its new relative place, including each moved call's
  saved `result.json`). Skipped (logged) when the job folder already holds a
  checkpoint. `migrate_saved_job(root, id)` (called by
  `doors.api.claude_jobs.load_job`): a phase-2 saved job under
  `~/.langslice/jobs/<id>/` moves into the job folder next to its images
  (plus `prompt.txt` and `result.json` → `exports/`), its `job.json`
  becomes the folder's (`host`), the index entry is written and the old
  directory removed; refused when that job folder already holds a
  checkpoint. A checkpoint from a newer LangSlice is refused before
  anything moves. Each migration is a line in `logs/events.jsonl`.
  Resumable (review fix 2026-10-04): a journal, `migration.json` in the job
  folder, lists every planned move before anything moves and marks each one
  done; the converted checkpoint, history and results are written while the
  old files still exist, the old files go after, the journal last, and the
  next open finishes an interrupted migration from it (a saved job's too).
  A move across file systems copies under `.partial`, renames into place,
  marks done, and only then removes the original. A target that already
  exists without the mark is compared file by file (size and SHA-256): the
  same, the original goes; different, both stay, the state keeps the
  original's path and the event lists it under `conflicts`.
- `formats.py` — the public files (above): `registration_document`,
  `section_entry`, `parameters` (the truth in public units),
  `applied_deformation`, `maps_status`, `write_registration`,
  `write_section_maps` (float maps as ImageJ hyperstacks, deflate without
  the floating-point predictor, which ImageJ 1.x cannot read;
  `FLOAT_COMPRESSION`), `write_labels` (the uint32 ids, the uint16 Fiji
  index with the atlas colours as its lookup table, the csv),
  `derived_files` (the CLI's artifacts after submit), `write_json`, the
  file names (`SECTION_FILES` by artifact kind, `QUICKNII_FILE`,
  `VISUALIGN_FILE`).
- `views.py` — `ViewStore` (`Job.views`): every picture the model was
  shown. The hook every door uses (phase 3d) is `ViewStore.shown(tool,
  atlas_of=)`: a context manager that collects what `core.layers` notes
  while the door's operation runs; the door hands it the pictures it sends
  (`Shown.show(pictures, arguments=, call_id=)`) and the store queues each
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
  number and id, arguments, sections, mode, picture size and bytes, `frame`)
  and, for a placement picture or a `fit_deformable` picture (its note's
  `warp`, `core.layers.warp_layers`), `labels.tif` (uint32, zlib) and
  `borders.png` (8-bit coverage), plus `residual.tif` for a `fit_deformable`
  picture that shows its deformation (`has_frame` decides). One section → `sections/<stem>/views/`,
  else `views/`. One line per picture in `views.jsonl` (`seq`, `name`,
  `path`, `tool`, `call`, `sections`, `mode`, `layers`). One background
  thread per store, ending when its queue is empty; `flush` (`ops.submit`,
  `Job.emit_results`, `Job.close`) and `flush_all` (interpreter exit) wait
  for it. A failed write is logged and skipped. Measured (2026-10-03,
  synthetic atlas): numbering and queueing 0.2-0.3 ms per call on the tool
  thread; writing 14 ms (512 px) to 32 ms (1024 px) per placement picture
  in the background; the 18 s golden recording and the test suite take as
  long as before. A 1024 px placement picture: `view.jpg` ~40 KB,
  `labels.tif` ~10 KB, `borders.png` ~1 KB, `view.json` ~2.5 KB (a real
  atlas's labels compress less). `captured()` (phase 5) collects every
  picture any store queues inside the block (`Saved`: its folder,
  whether it gets layers and a residual; `files()` lists `view.jpg`,
  `view.json` and the layers with their kinds): the agent CLI's `artifacts`. The numbering reads the index as it grows
  (each save reads the lines appended since the last), so two stores on one
  folder (a running agent and a CLI call) continue each other's numbers, and the numbers are reserved
  across processes before anything is written (`views.seq`, the next picture
  and last call numbers, under its own file lock `views.seq.lock`;
  `ViewStore._numbering`), so two stores that queue pictures before either
  writes never share one.

The frame record and the on-demand coordinate map are the core's
(`core/layers.py`, `core/CLAUDE.md`).

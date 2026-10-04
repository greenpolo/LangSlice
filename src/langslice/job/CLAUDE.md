# LangSlice `job/` — the job folder

Package guide for `src/langslice/job/`. The repo-level `CLAUDE.md` holds the
project-wide rules. `AGENTS.md` here is a verbatim copy — edit one, mirror to
the other.

The files of the job layer (layered refactor, phase 3c, 2026-10-03). The
`Job` itself (state, undo, checkpoint, gates) is `linear/job.py` until the
rename-only commit moves it; this package is where its files live and how
they are named, upgraded, indexed and filled with pictures.

## The layer rule

Job layer: imports the core (`linear.state`, `linear.checkpoint`,
`linear.discovery`, `core.layers`, `core.jpeg`) only. Never an operation
(`ops`), a door (`linear.toolbox`, `linear.view_options`, `adk`,
`mcp_server`, `api`) or `google.*`, `litellm`, `openai`.
`tests/test_core_imports.py` loads each module in a fresh interpreter and
checks both.

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
| ABBA worker `linear.run` (Fiji connector) | the connector's snapshot folder, a fresh `langslice-abba-*` temporary folder per run | `<snapshots>/langslice/`; the result's `output_dir` |
| ABBA Claude mode (`claude.prepare`) | `~/.langslice/snapshots/claude-*/` | `<snapshots>/langslice/` (id in the index) |
| `langslice abba --linear` (Python launcher) | a fresh `langslice-abba-*` run folder under the spec's folder (or the system temporary folder) | `<run folder>/langslice/` |

Inside (`layout.py`, names as constants):

```
job.json             settings: the JobSpec under "spec", format_version (1),
                     created_at, image_folder; a saved Claude job's job_id and
                     its own fields under "host" (kind, params, notes, trace_dir)
state.json           the checkpoint (StackState, state format 2)
history/             undo/redo: index.json + step-NNNNNN.json, one per step
sections/<stem>/     per section; <stem> is the image filename's stem (the whole
                     filename when two images share a stem)
  deformable/<key>/          applied deformation records (RecordStore)
  image_correction/<key>/    trace_borders calls, attempt-NN inside
  views/<seq>_<tool>_<mode>/ pictures of this section the model saw
views/<seq>_<tool>_<mode>/   pictures of several sections or none (atlas, sheets,
                             opening strips, show_stack pages)
views.jsonl          append-only index of every saved picture
exports/             linear_results.json (the run's result; spec.out overrides
                     the path), result.json (a saved host job's final result)
logs/events.jsonl    one line per open and per migration
prompt.txt           a saved Claude job's copy prompt
```

Reserved, not written yet (the formats phase): `registration.json` at the
top, and `residual.tif`, `coords.tif`, `labels.tif`, `labels_fiji.tif` +
`labels.csv` in each section folder.

**Paths are relative.** Every path a job file stores (a deformation's
`record`, an image correction's `artifact_dir` / `artifact_paths`, the views
index) is relative to the job folder (`JobLayout.relative` /
`JobLayout.resolve`; `linear.checkpoint.state_paths` walks a state's
paths). The trace identity in a deformation cache key is the stored
(relative) artifact directory, so keys survive a move of the folder.
`StackState.image_folder` itself stays absolute.

## Files

- `layout.py` — `JobLayout` (the folder, every name, `section_dir`,
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
  starts empty (logged).
- `index.py` — saved host jobs by id: `~/.langslice/jobs/<id>.json`
  (`job_id`, `job_folder`, `host_channel`, `created_at`; owner-only). The
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
  `api.claude_jobs.load_job`): a phase-2 saved job under
  `~/.langslice/jobs/<id>/` moves into the job folder next to its images
  (plus `prompt.txt` and `result.json` → `exports/`), its `job.json`
  becomes the folder's (`host`), the index entry is written and the old
  directory removed; refused when that job folder already holds a
  checkpoint. A checkpoint from a newer LangSlice is refused before
  anything moves. Each migration is a line in `logs/events.jsonl`.
- `views.py` — `ViewStore` (`Job.views`): every picture the model was
  shown. The tool door (`toolbox._saves_views`, around every tool) queues
  each returned picture with what `core.layers` noted about it; the ADK
  driver queues the opening's message images (`engine.save_opening`), the
  MCP door each `show_stack` page's image blocks (`server.save_page`), as
  the bytes sent. Per picture a folder `<seq>_<tool>[_<mode>]` (six-digit
  sequence across the job's life, continued on reopen) with `view.jpg` (the
  exact bytes the door sent: the same `core.jpeg.encode_jpeg` on the same
  picture, or the door's own bytes), `view.json` (format 1: tool, call
  number and id, arguments, sections, mode, picture size and bytes, `frame`)
  and, for a placement picture, `labels.tif` (uint32, zlib) and
  `borders.png` (8-bit coverage). One section → `sections/<stem>/views/`,
  else `views/`. One line per picture in `views.jsonl` (`seq`, `name`,
  `path`, `tool`, `call`, `sections`, `mode`, `layers`). One background
  thread per store, ending when its queue is empty; `flush` (submit,
  `Job.emit_results`, `Job.close`) and `flush_all` (interpreter exit) wait
  for it. A failed write is logged and skipped. Measured (2026-10-03,
  synthetic atlas): numbering and queueing 0.2-0.3 ms per call on the tool
  thread; writing 14 ms (512 px) to 32 ms (1024 px) per placement picture
  in the background; the 18 s golden recording and the test suite take as
  long as before. A 1024 px placement picture: `view.jpg` ~40 KB,
  `labels.tif` ~10 KB, `borders.png` ~1 KB, `view.json` ~2.5 KB (a real
  atlas's labels compress less).

The frame record and the on-demand coordinate map are the core's
(`core/layers.py`, `core/CLAUDE.md`).

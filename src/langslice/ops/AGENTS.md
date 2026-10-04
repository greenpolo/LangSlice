# LangSlice `ops/` — the verbs

Package guide for `src/langslice/ops/`. The repo-level `CLAUDE.md` holds the
project-wide rules. `AGENTS.md` here is a verbatim copy — edit one, mirror to
the other.

Every write to a stack is a function here (layered refactor, phases 3a and
3b, 2026-10-03).
The agent tools, the MCP server and, later, the per-operation CLI and
scripts call the same functions, so a tool and a script do exactly the same
write.

## The rule

- An operation takes the `linear.job.Job` (and the core
  `linear.workspace.Workspace` where it reads the atlas, the section files or
  the render caches) plus plain arguments: ids, numbers, dicts.
- It takes ONE undo step through the job (`before = job.snapshot()`, write,
  `job.commit(before)`: the step and the checkpoint), or none when nothing
  was written, and returns a plain record of what changed (a frozen
  dataclass with `touched`, or plain data).
- It never draws a picture, never words anything for a model and never
  applies the look-before-commit gates (`compared`/`reviewed`): those are the
  tool door's. Job rules (locked sections, host damage flags, the spec's
  flip switch) are the job's and are applied here.
- A refusal is `refusal.Refused(code, **facts)`, nothing written;
  `Refused.payload()` is the `{"status": "error", "error": code, ...}` every
  door answers with.
- Layer: core and job only. Never a door (`linear.toolbox`,
  `linear.view_options`, `adk`, `mcp_server`) and never `google.*`, `litellm`
  or `openai`. `tests/test_core_imports.py` loads each module in a fresh
  interpreter and checks both.

## Files

- `positions.py` — `set_positions(job, workspace, [(ref, mm)])`: clamp into
  `workspace.position_range`, write, report written/clamped/unknown.
  `set_cutting_angles(job, workspace, pitch, yaw)` (drops the render cache).
- `order.py` — `reorder(job, filenames, after)` (one block, filenames only),
  `renumber(order)` (indices only, no undo step).
- `orientation.py` — `orient_sections(job, entries)`: flip and quarter turns;
  a change drops the section's transform; `LOCKED`, `FLIP_DISABLED`,
  `BAD_ROTATION` per entry.
- `damage.py` — `mark_damaged(job, entries)`; a host flag cannot be cleared.
- `appearance.py` — `set_appearance(job, targets, ids, settings)` and
  `planned_settings` (what a write would leave, written nowhere: the door
  draws its AFTER picture first).
- `notes.py` — `add_note(job, text)`.
- `transforms.py` — the stored transform: `interactive_transform` (knobs,
  `shear` optional (0), to the record; the tool door fills a left-out
  shear from the current transform), `same_transform`, `fit_transform`
  (`fit_affine`'s record), `set_transforms(job, {id: record})` (one undo step
  for the batch; locked sections refused). `KNOBS` lists the knobs in
  payload order. The shear convention is `affine.decompose_affine`'s.
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
  row with its fit and record, for the door's pictures), `traced` (per
  traced section, the image read and the trace's lines on the fit grid),
  `written`. `keep_linear(job, records, reason)`: the "linear placement
  stands" record, one undo step; `NOTHING_WRITTEN` (each offending section
  under `results`) when one lacks a position or a transform. An applied
  record is saved in the section's folder (`job.deformations`:
  `sections/<stem>/deformable/<key>`) and the section's `deformation`
  holds its path relative to the job folder; a traced fit section reads
  the trace's artifacts under the job folder (`traced_lines(root=...)`).

The pictures are not here: they are the core's (`src/langslice/core/`,
its own `CLAUDE.md`), and the job saves every one the model is shown, with
its layers, in the job folder (`src/langslice/job/`, phase 3c).

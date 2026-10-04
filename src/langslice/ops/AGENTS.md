# LangSlice `ops/` — the verbs

Package guide for `src/langslice/ops/`. The repo-level `CLAUDE.md` holds the
project-wide rules. `AGENTS.md` here is a verbatim copy — edit one, mirror to
the other.

Every write to a stack is a function here (layered refactor, phase 3a,
2026-10-03).
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
  `shear` optional, to the record), `same_transform`, `fit_transform`
  (`fit_affine`'s record), `set_transforms(job, {id: record})` (one undo step
  for the batch; locked sections refused). `KNOBS` lists the knobs in
  payload order. The shear convention is `affine.decompose_affine`'s.

Not here yet (phase 3b): the placement pictures, the deformable fit
(`toolbox.fit_deformable_impl`) and saving renders to the job folder.

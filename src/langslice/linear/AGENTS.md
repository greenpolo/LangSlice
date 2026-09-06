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

The one exception is `align_slice`: a bounded interactive sub-session for ONE
section, run in-process as an async tool of the main agent (`transform.py`).

## Files

- `spec.py` — `JobSpec` (+ `ReorderSpec`/`PositionSpec`/`TransformSpec`). Every
  checkbox a host shows maps to a field here; nothing else is user-facing.
  `tasks` is the master switch: a task that is OFF builds no tools and takes
  its answer from `spec.inputs` instead. `inputs["pixel_size_um"]` is the
  calibration override, and `reasoning` (`--reasoning`) rides through
  `session.build_agent` onto any resolved model exposing `reasoning_effort`.
- `state.py` — `StackState`/`SliceState`. The checkpoint, the result and the
  thing every tool writes, one JSON shape for all three. `restore()` refills
  the same object in place, because tools close over one state.
- `checkpoint.py` — atomic JSON write to `<folder>/linear_state.json`.
- `discovery.py` — natural-sorted image discovery.
- `render.py` — `render_slice` (ROTATE first, then FLIP, then the display-only
  `--preprocess auto` enhancement), `stack_image_parts`, the status rows and
  their text form, `caption`, the JPEG `types.Part` encoder, and the PHYSICAL
  overlay: `canvas_geometry` places an atlas section on a section's frame at
  true scale (`atlas um/px / canvas um/px`, anatomy centred, canvas grown to
  hold it — never fit-to-canvas, which is not a calibration), and
  `physical_overlay` draws the ONE alignment picture on it (family outlines
  from `atlas.render.family_outlines` as 1 px neutral-grey hairlines drawn at
  OUTPUT size — ABBA's border look; coloured 2 px rimmed lines were tried and
  rejected as "too thick, colors are weird" — optional 35% template, 1 mm scale bar,
  two-line caption). It takes either the five physical knobs or a ready 2x3
  in the section's frame, so the interactive loop and `fit_affine` draw the
  same picture. `canvas_um_per_px` follows the file's pixel size down through
  the render's own downsample (`render_scale`, same key as `render_cache`);
  `estimate_um_per_px` is the fallback guess from tissue width. Renders are
  cached on the context and shared: read them, never mutate them. `caption`
  burns a label into a COPY — every image a tool returns gets one, because
  tool images reach the model as bare attachments; never caption an image a
  fit measures.
- `atlas_fetch.py` — `atlas_section` (the one atlas renderer: flat at 0/0
  cutting angles, `oblique.sample_oblique_plane` otherwise) and the
  `fetch_atlas` tool, closed over the run context. Sections and atlas sections
  are framed the same way so apparent scale is not a cue.
- `toolbox.py` — `build_tools(state, ctx, spec)`: every tool, gated by the
  spec, plus the submit gates and the undo/redo snapshot stack.
- `transform.py` — the silhouette fit and the interactive alignment
  sub-session, both in PHYSICAL space. `calibrate` answers the canvas's
  micrometres per pixel and where it came from: the host's
  `--pixel-size-um` (`"host"`), else the file's TIFF/OME tags (`"file"`),
  else the tissue-width guess (`"estimated"`) — it never crashes and never
  silently pretends. The sub-session's knobs are ABBA's (rotation about the
  centre, per-axis scales, `translate_x_mm`/`translate_y_mm`), and the
  recorded transform carries them under `"physical"` next to the six
  normalized numbers plus the `"calibration"` used.
- `prompt.py` — `build_job_statement`: job, run facts, ONE factual line per
  tool that exists, hard constraints. Nothing else.
- `session.py` — the ADK agent builder, the plugins, and the loop.
- `engine.py` — `EngineContext`, `ingest`, `apply_host_inputs`, `run_session`,
  `run_post_pass`, `emit_results`, and `run(spec)`.
- `signals.py`, `deepslice.py`, `trace.py` — interval arithmetic, the DeepSlice
  seam (reports `UNAVAILABLE`), and the full-content JSONL session trace.

## Rules that are not negotiable here

**Lean harness.** Tools return data. No advice, no interpretation, no strategy
in any payload or prompt. The job statement carries the job, the facts, one
line per tool and the constraints — no strategy menu, no rules of thumb, no
failure-mode warnings, no region names (the same text runs against every
BrainGlobe atlas, species and plane). Full-trace forensics found every major
benchmark failure tracking back to advice the harness injected; a per-slice
estimation worker that ate 82% of the wall-clock carried ~no signal and was
deleted; a landmark-tool pass benchmarked WORSE and was deleted rather than
kept behind a flag.

**Gates are constraints, not coaching.** A refusal states the numbers that
caused it and stops; `validate` runs the same gates without submitting.
`submit` refuses `MISSING_POSITIONS`,
`ORDER_POSITION_MISMATCH` (positions must run one way along the corrected
order; the offending neighbour pairs are named), `STRICT_INTERVAL` (spacing
within 10% of the interval and no breaks, when `--strict-interval`) and
`INTERVAL_BREAKS_UNSUPPORTED` (a reported break must exceed 1.5x the stack's
median written spacing). Gates only run for tasks that are on.

**Corrections are data.** Order, flips, rotations, positions, cutting angles
and transforms are proposals on the state. The user's image files are never
modified.

**Order and position must agree.** Only at `submit`: `reorder_slices` and
`move_slice` change nothing but `index_corrected`, and the
`ORDER_POSITION_MISMATCH` gate is what holds order and position together.
(Reordering used to CLEAR the positions of the sections that moved; every
debriefed agent called that hazardous, and it made them re-enter numbers they
still believed.) `orient_slices` does still clear a section's transform: a
transform is defined AFTER the orientation it was fitted under.

**Every write checkpoints, every write is undoable.** One tool call is one undo
step, so a batch undoes as one. The undo stack is in memory only (depth 50); a
resumed run starts from the checkpoint, which is the state as it stood.

## Ceilings worth knowing

- `fit_affine`'s silhouette method measures against the FLAT atlas section even
  when the stack carries cutting angles (`langslice.affine.silhouette_affine`
  builds its own atlas silhouette off the voxel grid). The payload says so with
  `flat_atlas_fit: true`.
- `method="elastix"` and `run_deepslice` answer `UNAVAILABLE`; both are seams,
  not stubs to fill in casually.
- `fit_position` is a thin wrapper over `oblique.fit_oblique` — correct, not
  tuned. It has not been benchmarked.
- True physical scale is honest, not flattering: measured on LSD_910 M04 at
  4.9 mm the specimen is 8.26 x 5.73 mm against the atlas section's
  9.05 x 6.68 mm, so the outlines land ~10% (ML) to ~15% (DV) outside the
  tissue at identity. That is the specimen-vs-Allen size difference, the same
  residual the nonlinear side measures, not a calibration bug.
- The post pass (`--no-subagents`) runs alignment sessions one at a time.

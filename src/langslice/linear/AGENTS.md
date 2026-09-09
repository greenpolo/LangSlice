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

There are no sub-sessions left. The per-section interactive alignment
(`adjust_transform`, `landmarks`) moved into the main toolbox on 2026-09-06:
GPT-6 Astra moved every parameter at once and finished a section in four
previews, so the fan-out bought nothing and cost the shared reading. Later the
same day preview and set became ONE tool, the computer-use pattern: every
`adjust_transform` call writes the transform and returns the picture, so the
agent always sees what it did and never spends a turn on a separate look or a
separate commit. `copy_transform` went with it — re-sending the same call for
another section is trivial for the agent. The same rule now holds for every
write outside the transform (audit, 2026-09-06): `orient_slices` returns the
re-oriented sections, `set_positions` returns each written section beside the
atlas at its new position, `fit_affine` always writes (no `apply` flag), and
`distribute_spacing` is gone — the agent can space a list of sections in its
head and send one `set_positions`. Nothing previews; everything is undoable.

**GPT-6 Astra's positioning debrief, built 2026-09-06** (full M04, 38
sections, reorder+position: order 38/38, positions median 0.30 mm, bias
-0.17 — a 0.2 mm ladder from a correct anchor; damage 7/7 +1 FP; 13 calls).
Nash kept five of its ten asks:
- images "not delivered": REAL, and not the transport. `adk/plugins.py`
  `trim_stale_tool_images` kept only the newest 24 tool images (sized for
  8-image sweeps); with 16-image `set_positions` batches every call but the
  newest two lost its pixels on every later turn, which is exactly Astra's
  "early results said attached, later ones showed images" — in both runs.
  Budget raised to 128 on 2026-09-06 — then run 4 (2026-09-07) carried 170
  images / 93k input tokens on its last call and 1.3M over the run, and a
  Plus window ended it. Now `WorkingSetImages` (see "Context and tokens"
  below), a dropped call's result says "dropped from context" instead of
  "attached", a tool's images ride INSIDE its `function_call_output` as
  labelled `input_image` parts (`openai_oauth._function_call_output`,
  2026-09-09 — the separate user message they used to follow in opened a
  new turn and threw the replayed reasoning away), and the
  picture-returning writes report `render_failed`.
- `compare_placement` (position stage): the section on the physical canvas
  against the atlas at any positions, every `VIEW_MODES` view, zoom, opacity
  — the "linked viewer" and the "placement preview / AP stepper" in one
  read-only tool. `physical_views` at identity, same renderer as everything.
- `view_stack`: the strip ordered by written position with position and
  spacing in the labels, plus `render.spacing_plot` (PIL, no matplotlib).
- `reorder_slices` took corrected indices for one run; Astra then pointed
  out (run 2) that a reorder changes the indices, so an index-addressed
  reorder can hit the wrong section next call. Filenames only again.
Run 2's new asks, built: `view_stack` pastes the atlas at each placed
section's position beneath it in the SAME image (one picture per section, not
two — the image budget counts); `compare_placement` takes a batch of
`{id, positions_mm}` entries, ≤4 pairs per call (one image each).
Rejected: labelled anatomy/landmarks, confidence and verification states,
damage masks, an anatomy-based gap review, a validity-vs-verification audit.

## Files

- `spec.py` — `JobSpec` (+ `ReorderSpec`/`PositionSpec`/`TransformSpec`). Every
  checkbox a host shows maps to a field here; nothing else is user-facing.
  `tasks` is the master switch: a task that is OFF builds no tools and takes
  its answer from `spec.inputs` instead. `inputs["pixel_size_um"]` is the
  calibration override, and `reasoning` (`--reasoning`) rides through
  `session.build_agent` onto any resolved model exposing `reasoning_effort`.
- `state.py` — `StackState`/`SliceState`. The checkpoint, the result and the
  thing every tool writes, one JSON shape for all three. `restore()` refills
  the same object in place, because tools close over one state. Every stored
  transform carries `physical` — the five knobs plus `shear`, about a pivot in
  canvas fractions — next to the six normalized numbers, whatever made it.
  There is no `confidence`: nothing downstream read it (Nash, 2026-09-06), and
  the reasoning lives in `adjust_transform`'s note.
- `checkpoint.py` — atomic JSON write to `<folder>/linear_state.json`.
- `discovery.py` — natural-sorted image discovery.
- `render.py` — `render_slice` (ROTATE first, then FLIP, then the display-only
  `--preprocess auto` enhancement), `stack_image_parts`, the status rows and
  their text form, `caption`, the JPEG `types.Part` encoder, and the PHYSICAL
  overlay: `canvas_geometry` places an atlas section on a section's frame at
  true scale (`atlas um/px / canvas um/px`, anatomy centred, canvas grown to
  hold it — never fit-to-canvas, which is not a calibration), and
  `physical_views` draws the alignment picture on it (family outlines
  from `atlas.render.family_outlines` as 1 px neutral-grey hairlines drawn at
  OUTPUT size — ABBA's border look; coloured 2 px rimmed lines were tried and
  rejected as "too thick, colors are weird" — 1 mm scale bar, two-line
  caption). It takes either the five physical knobs or a ready 2x3 in the
  section's frame, so the interactive loop and `fit_affine` draw the same
  picture, and returns `(images, silhouette_iou)`. Its view controls:
  `mode` (`overlay`, `side_by_side` — one stitched image, `checkerboard`,
  `outlines` — atlas lines plus the section's own silhouette in a second grey
  on black — and the line-free `section` / `template`), `zoom` ([x0, y0, x1, y1] fractions of the CANVAS, cropped BEFORE
  the resize so it magnifies, with the bar redrawn for the new µm/px),
  `template_opacity` (0..1, replaced the `show_template` bool) and `outlines`
  (`OUTLINE_LAYERS`: `all` family boundaries, `outer` — the root contour from
  `atlas.render.outer_outline` — or `none`; the caption names the layer when
  it is not `all`).
  `physical_overlay` is the one-image `overlay` wrapper `fit_affine` uses.
  `pivot` (canvas px, `pivot_on_canvas` resolves "canvas"/"tissue"/[fx, fy]
  onto it) and `markers` (the landmark pairs, a cross per section point and a
  ring per atlas point in their own two colors) ride through both.
  Region acronyms and damage masking were asked for and deliberately not
  built. `canvas_um_per_px` follows the file's pixel size down through
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
  spec, plus the submit gates and the undo/redo snapshot stack. The interactive
  transform lives here: `_Staged` (one section, its calibrated canvas and the
  resolved pivot), `adjust_transform` (the write AND the look, any positioned
  section; `mode="ab"` draws the new parameters beside what the section
  carried before the call, a silhouette fit included; the same numbers again
  re-draw without an undo step) and `landmarks` (read-only).
  `transform_history` on the ToolBox is per section and lasts the whole run. `commit(*touched)` is what every write answers with: the status rows of
  the sections it touched plus `n_sections`, never the whole table —
  `status`, `undo`, `redo` and `submit` are what return all the rows.
- `transform.py` — the silhouette fit and the arithmetic the interactive tools
  run on, both in PHYSICAL space. `fit_silhouette` matches the whole tissue
  outline against the whole atlas outline (`affine.silhouette_affine`) and
  reports six normalized numbers on the section frame plus `physical` about
  the canvas centre (`_conjugate` moves a 2x3 between the two frames); damaged
  sections are refused before this runs (`toolbox.fit_affine`). `calibrate`
  answers the canvas's
  `--pixel-size-um` (`"host"`), else the file's TIFF/OME tags (`"file"`),
  else the tissue-width guess (`"estimated"`) — it never crashes and never
  silently pretends. The knobs are ABBA's (rotation about the pivot, per-axis
  scales, `translate_x_mm`/`translate_y_mm`), and the recorded transform
  carries them under `"physical"` (with the pivot as canvas fractions) next to
  the six normalized numbers plus the `"calibration"` used.
  `physical_decomposition` is `decompose_affine` minus its translation
  fractions; `similarity_fit` (Umeyama, exact on two points), `affine_fit`
  (least squares) and `physical_params` (a canvas 2x3 back into the five knobs
  about a pivot) are what `landmarks` measures with.
- `prompt.py` — `build_job_statement`: job, run facts, ONE factual line per
  tool that exists, hard constraints. Nothing else.
- `session.py` — the ADK agent builder, the plugins, the loop, and
  `TokenTally`: every call's usage is printed and traced, and
  `JobSpec.max_quota_percent` (25) ends the session when this run's share
  of the provider's usage window reaches it, and `max_input_tokens` (2M) is
  the raw safety. Run 4 on M04 spent 1.3M input tokens (93k on the last
  call, 170 images resent) and a Plus window ended it; never again.
- `engine.py` — `EngineContext`, `ingest`, `apply_host_inputs`, `run_session`,
  `emit_results`, and `run(spec)`. No post pass: the session is the whole run.
- `deepslice.py`, `trace.py` — the DeepSlice seam (reports `UNAVAILABLE`) and
  the full-content JSONL session trace.

## Context and tokens (2026-09-09)

The OAuth lane resends the whole history on every call (the Codex backend
refuses stored responses; the Codex CLI does the same). What a run COSTS is
not the raw input count: the usage window prices a cached input token at
~0.13x an uncached one (fitted on the x-codex quota headers over run 5's 39
calls, error half a point; OpenAI's published API rate is 0.1x), so a token
that sits unchanged in the prefix is nearly free and a token that is new, or
sits after a prefix edit, is full price. Run 5 (M11, 39 calls): 786k raw,
560k cached, ~298k paid = 28% of a Plus 5-hour window; of the paid part 58%
was NEW IMAGES (each paid once, in full, the call it arrives) and 42% new
text plus prefix breaks. Design rules that follow:
- **Every image at the atlas's resolution, 512 px long edge at most**
  (`render.atlas_native_long_edge`: with a known pixel size a show render
  lands at the atlas's µm/px; the fit path is not capped, its parameters
  are normalized against the render). `VIEW_LONG_EDGE` 512, seed/atlas 512,
  `compare_placement` panels 512 (~260 tokens an image, ~530 a stitched
  pair); the interactive-transform canvas keeps `OVERLAY_LONG_EDGE` 768.
- **Images stay** (`adk/plugins.py WorkingSetImages`): every tool image is
  kept until 256 are live, then the oldest media-bearing calls are cut in
  ONE batch to 128 (`DEFAULT_MAX_IMAGES` / `DEFAULT_KEEP_IMAGES`, a cut
  result says "dropped from context"); the cut only moves forward and the
  seed strip is never touched. Measured on M11 at low effort, 2026-09-09:
  keep-all (run 6, killed at call 16) had median 0.10 mm / 30 of 36 within
  0.25 with positions written by call 15; newest-call-only (run 7, 21
  calls, submitted) 0.30 mm / 14 of 36, and its debrief said the "dropped
  from context" results made its comparisons unreliable. Cost: a kept
  image is 0.13x on every later call, so keep-all grows quadratically
  (2.4k -> 8.8k paid a call by call 15), but a batched run submits in ~21
  calls, where keep-all is ~16% of a window against newest-only's 14%.
  A normal run (seed 80, ~50 compares, ~40 write pictures) never reaches
  the cut; it is the safety for a run that goes long; cutting old images
  re-reads everything after the cut once, so it must stay rare.
- **A tool's images ride inside its `function_call_output`** as labelled
  `input_image` parts (`openai_oauth._function_call_output`); the separate
  user message they used to follow in opened a new turn and, under the
  API's default `reasoning.context = current_turn`, threw the replayed
  reasoning away. The request sends `context = all_turns`.
- **Reasoning is replayed.** Each turn's `reasoning` output item (encrypted,
  `store` is false) is kept on the model turn (`thought_signature`, summary
  in a `thought` text part so traces show it) and sent back ahead of that
  turn, the way the Codex CLI does. ~9k tokens by call 39, all cached.
- **Rows are compact** (`render.compact_rows`): null and empty fields are
  absent from every tool payload (a position-only run carried null transform
  fields on every row of every result, a third of the paid text).
- **The budget is the window.** `session.py TokenTally` prints every call's
  usage with `paid~` (uncached + 0.13 x cached) and, on the OAuth lane, the
  window percent this run has used; `JobSpec.max_quota_percent` (25,
  `--max-quota-percent`) ends the session when the run's share of the window
  reaches it. `JobSpec.max_input_tokens` (2M, `--max-input-tokens`) stays as
  the raw safety for lanes that report no quota. `openai_oauth.py` sends one
  stable `prompt_cache_key`/`session_id` per session and surfaces the
  `x-codex-*` quota headers.
Not adopted, on Nash's call: server-side compaction (for heavy text; we are
light text, heavy image) and a `Memorize` action (replayed reasoning carries
facts forward already; Codex's `notes` exist to cross a hard context-window
reset, which our runs never hit). The WebSocket transport is a latency lever
only. Prefix caching is what makes "re-send the same image" cheap; there is
no content-addressed reuse, so an overlay composed onto a cached picture is
a new picture at full price, which at 512 px is ~260 tokens.

## Rules that are not negotiable here

**Lean harness.** Tools return data. No interpretation in any payload. The job statement carries the job, the facts, one line per tool, the constraints and — when positioning is on — a short `Method` section (Nash, 2026-09-07): place each section on its own evidence and compare candidates before writing, work in batches (several sections per compare or write call — the "one at a time" wording went on 2026-09-09, it bought 39 calls on M11), review the whole stack afterwards, re-check both sides of a gap before reporting a break, validate, submit. Asked from its own run-3 trace, Astra said it skipped `compare_placement` and `view_stack` by oversight, not wording, and asked for exactly this. Still out: rules of thumb, failure-mode warnings and region names (the same text runs against every BrainGlobe atlas, species and plane). Full-trace forensics found every major
benchmark failure tracking back to advice the harness injected; a per-slice
estimation worker that ate 82% of the wall-clock carried ~no signal and was
deleted; a landmark-tool pass for POSITION estimation benchmarked WORSE and was
deleted rather than kept behind a flag (the `landmarks` tool of the transform
task is a different animal: it measures pairs the agent picks, and reports
millimetres, never a recommendation).

**One transform representation.** Silhouette, interactive, elastix-someday:
every stored transform and every fit payload carries `physical` (the five
knobs `adjust_transform` takes, plus `shear`, about a pivot in canvas fractions)
next to the six normalized numbers. The fraction-based `decomposition` left
the fit payload: "+0.04 mm entered, negative fraction reported" cost four
sessions, and GPT-6 Astra could not hand a fit's numbers to the manual
controls. A fit is now a starting point for `adjust_transform` and the B side
of `mode="ab"`.

**Gates are constraints, not coaching.** A refusal states the numbers that
caused it and stops; `validate` runs the same gates without submitting.
`submit` refuses `MISSING_POSITIONS`,
`ORDER_POSITION_MISMATCH` (positions must run one way along the corrected
order; the offending neighbour pairs are named), `STRICT_INTERVAL` (spacing
within 10% of the interval and no breaks, when `--strict-interval`),
`INTERVAL_BREAKS_UNSUPPORTED` (a reported break must exceed 1.5x the stack's
median written spacing) and `MISSING_TRANSFORMS` (every section carries one,
damaged included — the agent aligns those by hand or explains in the note).
Gates only run for tasks that are on.

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
- `physical` on a fit is the five knobs about the canvas centre plus the
  `shear` they cannot express (0.10-0.16 on M05_D_08, not noise), so a preview
  typed from a fit reproduces it only up to that shear. The A/B view does not
  have that gap: its B side is drawn from the six stored numbers
  (`affine.denormalized_affine`), which are exact.
- The moments core has a 180-degree ambiguity — `(1,1)` and `(-1,-1)` are the
  same axes turned round — and settles it on silhouette IoU alone. On INTACT
  M05 sections the right one wins by 0.036-0.092 IoU; a template-correlation
  tie-break would help the cases where silhouette IoU alone is close, but that
  is measured, not built: it would move a benchmarked path (`silhouette_affine`
  is `nonlinear/quick_affine`'s too). The fit's own panel is what catches it;
  look at it. Damaged sections never reach this path: `fit_affine` refuses
  them outright.
- `method="elastix"` and `run_deepslice` answer `UNAVAILABLE`; both are seams,
  not stubs to fill in casually.
- `fit_position` is a thin wrapper over `oblique.fit_oblique` — correct, not
  tuned. It has not been benchmarked.
- True physical scale is honest, not flattering: measured on LSD_910 M04 at
  4.9 mm the specimen is 8.26 x 5.73 mm against the atlas section's
  9.05 x 6.68 mm, so the outlines land ~10% (ML) to ~15% (DV) outside the
  tissue at identity. That is the specimen-vs-Allen size difference, the same
  residual the nonlinear side measures, not a calibration bug.
- `adjust_transform`'s pivot resolution runs its own `canvas_geometry`, so a
  call builds the atlas plane twice (once to place the pivot, once to draw).
  Flat sections are a numpy take; oblique ones resample twice.

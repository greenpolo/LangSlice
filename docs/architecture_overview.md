# Architecture Overview

LangSlice is one installable Python package, `langslice`. Local-model training
lives in the sibling `LangSlice-Training` repository and the position benchmark
in `SliceBench`; both depend on this package.

## Package Layout

The two registration methods are sibling subpackages with no dependency on each
other:

- `src/langslice/linear/` -- order, position and one in-plane affine per
  section: the job spec, the stack state and its JSON checkpoint, the one
  toolbox, the job statement, the ADK session, and the run engine.
- `src/langslice/nonlinear/` -- generative-image registration: placed-border
  correction (with color-map initialization when no placement is supplied),
  image provider adapters, Elastix runtime, affine/nonlinear result types, and
  the silhouette-based `quick_affine` preview.

The remaining top-level modules are shared by both:

- `src/langslice/atlas/` -- BrainGlobe atlas loading, slice extraction, and
  colored region maps.
- `src/langslice/space.py` -- coordinate and orientation conventions.
- `src/langslice/affine.py` -- the shared in-plane affine core: the silhouette
  (image-moments) fit of a section onto an atlas section, the
  rotation/scale/translate matrix builder, and the normalized six-number
  parameter convention. Used by the linear transform tools and by
  `quick_affine`.
- `src/langslice/image_prep.py` -- image normalization, metadata detection, and downsampling.
- `src/langslice/integrations/` -- integration layers for external registration software: `quint.py` (QUINT/QuickNII/VisuAlign JSON export), `abba.py` (abba-python registration plugin).
- `src/langslice/providers/` -- Gemini and OpenAI-compatible model configuration.
- `src/langslice/adk/` -- ADK plugins, model resolution, and SDK helpers.
- `src/langslice/api/` -- Pydantic engine contract, runtime wrappers, and the
  stdio service used by non-Python clients.
- `src/langslice/cli.py` -- the `langslice` command.

## Where The Methods Fit

`linear` and `nonlinear` can be used separately. `nonlinear` takes a slice
position as input and does not care where that position came from, so it can
follow `langslice linear run` or a placement made in another tool. In
QUINT/ABBA-style workflows, linear placement happens first in the host tool and
LangSlice-nonlinear stands in for the manual spline/BigWarp deformation step.

## Engine Contract

The Python package is the source of truth for LangSlice runtime behavior. The
engine contract is defined with Pydantic models in `src/langslice/api/models.py`.

`langslice serve --stdio` runs the newline-delimited JSON engine service. It
accepts `version`, `register.run`, `quick_affine.run`, and `export.run`, plus
`setup.status`, `setup.login`, `setup.api_key`, `linear.run`, and `nonlinear.abba`
for the independent Fiji connector. It emits progress/log/data event envelopes
and returns either result or error envelopes. The connector starts a worker in
the selected Python environment; it does not require `abba_python`. See
[the connector design](abba_plugin_design.md) and
[installation instructions](abba_installation.md).

## Linear: Order, Position, Transform

`langslice linear run` is ONE agent environment over a folder of sections: one
`StackState`, one toolbox, one job statement, and one ADK session that ends at
`submit` or the turn budget. A host fills a `JobSpec` -- which of `reorder`,
`position` and `transform` are on, plus the knobs each one exposes -- and the
toolbox is built from it: a task that is off contributes no tools, and its
answer comes from `spec.inputs` instead.

Every write tool checkpoints the whole state, so a run that dies resumes with
the state it had (the agent is re-seeded, not replayed), and every write is
undoable in memory (`undo`/`redo`, one tool call = one step). The results file
uses the checkpoint's schema, so the CLI, the checkpoint and any host adapter
read one shape. Corrections (order, flips, rotations), positions, cutting
angles and per-section transforms are proposals -- the host applies them, and
the user's image files are never modified.

Interactive alignment runs in the main session, sharing the context that placed
and ordered the stack. `adjust_transforms` writes and shows one to four sections
in one undo step, including before/after or side-by-side views. A dependent
refinement waits for its first picture. The linear agent handles affine
alignment; local deformation belongs to the separate nonlinear image-generation
workflow. An agent-callable bridge to image generation remains undecided.

Picture-returning writes show what they did: `orient_slices` the re-oriented
sections, `set_positions` placements not already seen at the same position,
orientation and cutting angles in a full-canvas atlas-bearing view, and the
transform tools their overlays. A section-only or zoomed comparison does not
suppress the full placement. A compare and position write planned in the same
model round both return their images because neither sibling result was
visible when planned. There is no
preview-then-apply anywhere; every write is undoable instead. Transform
results send the useful physical parameters once; normalized matrices,
derived representations and adjustment history stay local.

## Nonlinear: Border Refinement

1. Prefer the atlas placement supplied by the linear agent or host tool.
   Without placement, generate and register a color map first.
2. Send rough yellow atlas borders over histology plus the identical clean
   histology to the image model; ask it to adjust the lines to the tissue.
3. Extract corrected yellow lines and overlay them on the original photograph.
4. Fit the residual deformation and warp the placed atlas labels.
5. Compose the initial affine or nonlinear placement with that residual for
   native-atlas correspondences and VisuAlign markers.
6. Return separate raw, corrected-border and fitted-atlas review artifacts.

Supplied placement requires one image-generation call; standalone requires two.
ABBA shares the border-correction core and retains its host placement.
The top-level linear handoff adapter does not introduce a dependency between
the sibling method packages. See [the design](nonlinear_design.md).

## Debugging

Set `LANGSLICE_VLM_DEBUG_DIR` for run artifacts. Set
`LANGSLICE_ADK_CAPTURE_REQUESTS_DIR` for redacted ADK request captures.

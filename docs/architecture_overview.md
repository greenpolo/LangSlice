# Architecture Overview

LangSlice is one installable Python package, `langslice`. Local-model training
lives in the sibling `LangSlice-Training` repository and the position benchmark
in `SliceBench`; both depend on this package.

## Package Layout

The two registration methods are sibling subpackages that never import each
other; top-level bridge modules connect them:

- `src/langslice/linear/` -- order, position and one in-plane affine per
  section: the job spec, the stack state and its JSON checkpoint, the one
  toolbox, the job statement, the ADK session, and the run engine.
- `src/langslice/nonlinear/` -- generative-image registration: exactly two
  border-based routes (placed-border correction when a placement is supplied;
  otherwise a silhouette-placed, model-drawn-boundary route against an
  outlined grayscale atlas template), image provider adapters, Elastix
  runtime, affine/nonlinear result types, and the silhouette-based
  `quick_affine` preview.

The remaining top-level modules are shared by both:

- `src/langslice/atlas/` -- BrainGlobe atlas loading, slice extraction, and
  colored region maps.
- `src/langslice/space.py` -- coordinate and orientation conventions.
- `src/langslice/affine.py` -- the shared in-plane affine core: the silhouette
  (image-moments) fit of a section onto an atlas section, the
  rotation/scale/translate matrix builder, and the normalized six-number
  parameter convention. Used by the linear transform tools and by
  `quick_affine`.
- `src/langslice/oblique.py` -- arbitrary-plane (cutting-angle) sampling of an
  atlas volume and the (pitch, yaw) fitter.
- `src/langslice/registration_handoff.py` -- turns a linear section state into
  the calibrated placement the nonlinear border correction takes.
- `src/langslice/registration_tool.py` -- the linear agent's optional
  image-model border-correction tool, built on that handoff.
- `src/langslice/image_prep.py` -- image normalization, metadata detection,
  downsampling, and the host multichannel blend (`host_preprocess`).
- `src/langslice/integrations/` -- integration layers for external registration software: `quint.py` (QUINT/QuickNII/VisuAlign JSON export), `abba.py` (abba-python registration plugin).
- `src/langslice/providers/` -- model access: Gemini API keys, OpenAI API keys,
  and ChatGPT subscription sign-in (`openai-oauth`), named in `registry.py`.
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
`setup.status`, `setup.login`, `setup.api_key`, `linear.run`, `linear.estimate`,
`preprocess.preview` and `nonlinear.abba` for the independent Fiji connector. It emits progress/log/data event envelopes
and returns either result or error envelopes. The connector starts a worker in
the selected Python environment; it does not require `abba_python`. See
[the connector design](abba_plugin_design.md) and
[installation instructions](abba_installation.md).

## Linear: Order, Position, Transform

`langslice linear run` is ONE agent environment over a folder of sections: one
`StackState`, one toolbox, one job statement, and one ADK session that ends at
`submit` or the turn budget. A host fills a `JobSpec` -- which of `reorder`,
`position`, `transform` and the optional `nonlinear` are on, plus the knobs
each one exposes -- and the toolbox is built from it: a task that is off
contributes no tools, and its answer comes from `spec.inputs` instead. Hosts
present these tasks to users as Positioning (`reorder` + `position`), Linear
(`transform`) and Nonlinear; see [the interface design](interface_design.md).

Every write tool checkpoints the whole state, so a run that dies resumes with
the state it had (the agent is re-seeded, not replayed), and every write is
undoable (`undo`/`redo`, one tool call = one step; the history is saved beside
the checkpoint and survives a resume). One job object (`linear/job.py`) owns
the state, undo, checkpoint and submit gates; the agent's tools and the MCP
server both sit on it, and it picks up a state file a script changed on disk. The results file
uses the checkpoint's schema, so the CLI, the checkpoint and any host adapter
read one shape. Corrections (order, flips, rotations), positions, cutting
angles and per-section transforms are proposals -- the host applies them, and
the user's image files are never modified.

Interactive alignment runs in the main session, sharing the context that placed
and ordered the stack. `adjust_transforms` writes and shows one to four sections
in one undo step, including before/after or side-by-side views. A dependent
refinement waits for its first picture. The linear agent handles affine
alignment. With the optional `nonlinear` task it also gets
`trace_borders`, which sends one section's placed atlas borders to the
image model for correction and keeps the first reply; it fits no deformation
and changes no transform. See [the image-tool contract](nonlinear_image_tool.md).

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
   Without placement, fit a local silhouette placement instead.
2. Send rough yellow atlas borders over histology plus the identical clean
   histology to the image model (route "supplied"); or, with no placement,
   send the clean histology plus an outlined grayscale atlas template and
   ask the model to draw the boundaries from nothing (route "atlas", with an
   optional second corrective call).
3. Extract corrected yellow lines and overlay them on the original photograph.
4. Optionally fit the residual deformation (Elastix B-spline or affine) and
   warp the placed atlas labels. The default is `deformation="none"`: the
   residual is identity while the deformation algorithm is still being designed.
5. Compose the initial affine placement with that residual for native-atlas
   correspondences and VisuAlign markers.
6. Return separate raw, corrected-border and fitted-atlas review artifacts.

Each route uses one image-generation call, except route "atlas" with an
optional second audit call. Route "supplied" is the production path: nonlinear
correction needs a linear placement first. Route "atlas" remains for
experiments. ABBA shares the border-correction core and
retains its host placement. The top-level linear handoff adapter does not
introduce a dependency between the sibling method packages. See
[the design](nonlinear_design.md).

## Debugging

Set `LANGSLICE_VLM_DEBUG_DIR` for run artifacts. Set
`LANGSLICE_ADK_CAPTURE_REQUESTS_DIR` for redacted ADK request captures. Set
`LANGSLICE_TRACE_DIR` (or `langslice linear run --trace-dir`, or the worker's
`trace_dir`) for a full-content trace of each linear agent session.

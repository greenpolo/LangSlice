# Architecture Overview

LangSlice is one installable Python package, `langslice`, plus model projects
under `models/`.

## Package Layout

The two registration methods are sibling subpackages with no dependency on each
other:

- `src/langslice/linear/` -- order, position and one in-plane transform per
  section: the job spec, the stack state and its JSON checkpoint, the one
  toolbox, the job statement, the ADK session, and the run engine.
- `src/langslice/nonlinear/` -- generative-image registration: candidate
  generation, image provider adapters, Elastix runtime, optional ADK review,
  affine/nonlinear result types, and the silhouette-based `quick_affine`
  preview.

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
accepts request envelopes such as `version`, `register.run`,
`quick_affine.run`, and `export.run`, emits progress/log event envelopes, and
returns either result or error envelopes.

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

The per-section interactive alignment is not a nested session: since
2026-09-06 `adjust_transform` and `landmarks` sit in the main toolbox, so the
agent aligns a section whenever it wants, in the same context that placed it.
Each `adjust_transform` call writes the transform and returns the picture of
the result, so acting and looking are one step.

Every write returns the picture of what it did: `orient_slices` the
re-oriented sections, `set_positions` each section beside the atlas at its
new position, `fit_affine` and `adjust_transform` the overlay. There is no
preview-then-apply anywhere; every write is undoable instead.

## Nonlinear: Image-Gen Registration

Registration has one active method: image-gen registration.

1. Prepare a histology slice, atlas color map, and atlas reference image.
2. Ask the image-generation provider to generate an atlas-colored target aligned to the histology.
3. Register the generated target to the atlas color map with itk-elastix.
4. Warp the atlas RGB through the recovered transform.
5. Extract VisuAlign markers from B-spline control points.
6. Return the generated atlas target, warped atlas, and warped-border overlay.

Direct mode returns the first candidate. Agentic mode runs a hosted-router
conversation (openai-oauth only): the task prompt and input images go to the
hosted GPT model with the image_generation tool attached, the harness runs
Elastix on each generated image and sends the report back as a follow-up
message, and the loop accepts the first candidate whose report is clean
(capped at four attempts; otherwise the fewest-codes candidate wins).

## Debugging

Set `LANGSLICE_VLM_DEBUG_DIR` for run artifacts. Set
`LANGSLICE_ADK_CAPTURE_REQUESTS_DIR` for redacted ADK request captures.

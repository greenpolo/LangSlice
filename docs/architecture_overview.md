# Architecture Overview

LangSlice is one installable Python package, `langslice`, plus model projects
under `models/`.

## Package Layout

The two registration methods are sibling subpackages with no dependency on each
other:

- `src/langslice/linear/` -- slice-position estimation: ADK agents, prompts,
  tools, validators, runners, and trace collection.
- `src/langslice/linear/whole_brain/` -- multi-slice whole-brain position
  estimation pipeline.
- `src/langslice/nonlinear/` -- generative-image registration: candidate
  generation, image provider adapters, Elastix runtime, optional ADK review,
  affine/nonlinear result types, and the silhouette-based `quick_affine`
  preview.

The remaining top-level modules are shared by both:

- `src/langslice/atlas/` -- BrainGlobe atlas loading, slice extraction, and
  colored region maps.
- `src/langslice/space.py` -- coordinate and orientation conventions.
- `src/langslice/image_prep.py` -- image normalization, metadata detection, and downsampling.
- `src/langslice/export.py` -- QUINT/ABBA-compatible JSON export.
- `src/langslice/providers/` -- Gemini and OpenAI-compatible model configuration.
- `src/langslice/adk/` -- ADK plugins, model resolution, and SDK helpers.
- `src/langslice/api/` -- Pydantic engine contract, runtime wrappers, and the
  stdio service used by non-Python clients.
- `src/langslice/cli.py` -- the `langslice` command.

## Where The Methods Fit

`linear` and `nonlinear` can be used separately. `nonlinear` takes a slice
position as input and does not care where that position came from, so it can
follow `langslice linear estimate` or a placement made in another tool. In
QUINT/ABBA-style workflows, linear placement happens first in the host tool and
LangSlice-nonlinear stands in for the manual spline/BigWarp deformation step.

## Engine Contract

The Python package is the source of truth for LangSlice runtime behavior. The
engine contract is defined with Pydantic models in `src/langslice/api/models.py`.

`langslice serve --stdio` runs the newline-delimited JSON engine service. It
accepts request envelopes such as `version`, `estimate.run`, `register.run`,
`quick_affine.run`, and `export.run`, emits progress/log event envelopes, and
returns either result or error envelopes.

## Linear: Position Estimation

Single-slice and group position estimation run through ADK. The agent can fetch atlas
images and must submit a structured estimate. Native Gemini requests can use the
File API for target images, and fetched atlas images are returned as ADK native
media tool results, so they persist in session history and stay visible on
every later turn. This is the only estimation path, and it supports all planes
(coronal, sagittal, horizontal).

## Linear: Whole-Brain Estimation

Whole-brain estimation discovers a folder of slices, estimates anchor slices with
the tool-use estimator, interpolates positions for non-anchor slices, estimates
each non-anchor slice independently over the full atlas range with the same
tool-use estimator, and fits a constrained monotonic position curve.

## Nonlinear: Image-Gen Registration

Registration has one active method: image-gen registration.

1. Prepare a histology slice, atlas color map, and atlas reference image.
2. Ask the image-generation provider to generate an atlas-colored target aligned to the histology.
3. Register the generated target to the atlas color map with itk-elastix.
4. Warp the atlas RGB through the recovered transform.
5. Extract VisuAlign markers from B-spline control points.
6. Return the generated atlas target, warped atlas, and warped-border overlay.

Direct mode returns the first candidate. Agentic mode lets an ADK review agent
inspect up to three candidates before confirming one.

## Debugging

Set `LANGSLICE_VLM_DEBUG_DIR` for run artifacts. Set
`LANGSLICE_ADK_CAPTURE_REQUESTS_DIR` for redacted ADK request captures.

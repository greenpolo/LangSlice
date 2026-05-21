# Architecture Overview

LangSlice is organized around one installable Python harness package,
`langslice_harness`, plus a Tauri desktop app and model projects.

## Package Layout

- `src/langslice_harness/atlas/` -- BrainGlobe atlas loading, coordinate conversion, and slice extraction.
- `src/langslice_harness/harness/estimation/` -- ADK slice-position estimation agents, prompts, tools, validators, plugins, and runners.
- `src/langslice_harness/harness/estimation/image_gen.py` -- image-gen position estimation.
- `src/langslice_harness/registration/` -- public registration wrapper, runtime, solver, and result types.
- `src/langslice_harness/harness/registration/` -- image-gen registration candidate generation and optional ADK review.
- `src/langslice_harness/whole_brain/` -- multi-slice position estimation pipeline.
- `src/langslice_harness/image_prep.py` -- image normalization, metadata detection, and downsampling.
- `src/langslice_harness/export.py` -- QUINT/ABBA-compatible JSON export.
- `src/langslice_harness/api/` -- Pydantic engine contract, runtime wrappers,
  schema export, and stdio service used by non-Python clients.

The CLI command remains `langslice`, but the Python import package is
`langslice_harness`.

## Engine Contract

The Python harness is the source of truth for LangSlice runtime behavior. The
engine contract is defined with Pydantic models and exported to JSON Schema plus
generated TypeScript types:

```bash
python scripts/generate_engine_contract.py
langslice schema --out docs/engine_schema.json
```

`langslice serve --stdio` runs the newline-delimited JSON engine service. It
accepts request envelopes such as `version`, `estimate.run`, `register.run`,
`quick_affine.run`, and `export.run`, emits progress/log event envelopes, and
returns either result or error envelopes.

## Position Estimation

Single-slice and group position estimation run through ADK. The agent can fetch atlas
images and must submit a structured estimate. Native Gemini requests can use the
File API for target images, and a persistent multimodal plugin keeps fetched
atlas images visible across turns.

Image-gen position estimation is available for visual sweep/zoom style estimation and
is used by the whole-brain pipeline.

## Image-Gen Registration

Registration has one active method: image-gen registration.

1. Prepare a histology slice, atlas color map, and atlas reference image.
2. Ask the image-generation provider to generate an atlas-colored target aligned to the histology.
3. Register the generated target to the atlas color map with itk-elastix.
4. Warp the atlas RGB through the recovered transform.
5. Extract VisuAlign markers from B-spline control points.
6. Return the generated atlas target, warped atlas, and warped-border overlay.

Direct mode returns the first candidate. Agentic mode lets an ADK review agent
inspect up to three candidates before confirming one.

## Whole-Brain Estimation

Whole-brain estimation discovers a folder of slices, estimates anchor slices,
interpolates positions for non-anchor slices, runs windowed image-gen estimation,
and fits a constrained monotonic position curve.

## Desktop App

`tauri-gui/` contains the Rust backend and React frontend. The GUI invokes the
Python engine protocol for position estimation, registration, and quick-affine
preview. Export remains on the legacy `langslice register --out ...` path until
the engine contract grows an affine-aware export request. The Rust side handles
atlas loading, reslicing, mesh serving, image thumbnails, and process lifecycle.

## Web Demo

`web-demo/` is a static browser demo with a deliberately limited runtime. It
keeps generated TypeScript engine contract types and declares the subset of
engine methods it supports, but Python remains the canonical runtime.

## Debugging

Set `LANGSLICE_VLM_DEBUG_DIR` for run artifacts. Set
`LANGSLICE_ADK_CAPTURE_REQUESTS_DIR` for redacted ADK request captures.

# Registration pipeline

Registration has one active method: image-gen registration.

## Active Files

- `src/langslice_harness/registration/core.py` -- public wrapper.
- `src/langslice_harness/registration/runtime.py` -- orchestration and debug artifacts.
- `src/langslice_harness/registration/solver.py` -- deterministic affine/TPS helpers retained for export math and tests.
- `src/langslice_harness/registration/types.py` -- result and annotation data classes.
- `src/langslice_harness/harness/registration/image_gen_registration.py` -- candidate pipeline.
- `src/langslice_harness/harness/registration/providers.py` -- image generation provider adapters.
- `src/langslice_harness/harness/registration/runner.py` -- optional ADK review loop.

## Pipeline

1. Load the histology slice and atlas slice at the chosen AP coordinate.
2. Generate an atlas-colored image aligned to the histology.
3. Register the generated target to the atlas color map with itk-elastix.
4. Warp the atlas through the recovered transform.
5. Extract VisuAlign-compatible markers from the B-spline control points.
6. Return all review outputs: generated atlas target, warped atlas, and warped-border overlay.

## Modes

- `direct` -- generate one candidate and return it.
- `agentic` -- expose the candidate generator to an ADK review agent, capped at three candidates.

## Notes

The runtime exposes one registration path: image-gen registration.

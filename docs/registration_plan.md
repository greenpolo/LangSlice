# Registration pipeline

Registration has one active method: image-gen registration. It lives in
`src/langslice/nonlinear/` and runs after a slice position is known, from
`langslice linear ...` or from any other placement step.

## Active Files

- `src/langslice/nonlinear/runtime.py` -- orchestration and debug artifacts.
- `src/langslice/nonlinear/types.py` -- affine/nonlinear result and annotation data classes.
- `src/langslice/nonlinear/image_gen_registration.py` -- candidate pipeline.
- `src/langslice/nonlinear/providers.py` -- image generation provider adapters.
- `src/langslice/nonlinear/router.py` -- optional hosted-router session.
- `src/langslice/nonlinear/quick_affine.py` -- silhouette-based affine preview.

## Pipeline

1. Load the histology slice and atlas slice at the chosen AP coordinate.
2. Generate an atlas-colored image aligned to the histology.
3. Register the generated target to the atlas color map with itk-elastix.
4. Warp the atlas through the recovered transform.
5. Extract VisuAlign-compatible markers from the B-spline control points.
6. Return all review outputs: generated atlas target, warped atlas, and warped-border overlay.

## Modes

- `direct` -- generate one candidate and return it.
- `agentic` -- hosted-router conversation (openai-oauth only); Elastix reports
  are fed back as follow-up messages, capped at four candidates.

## Notes

The runtime exposes one registration path: image-gen registration.

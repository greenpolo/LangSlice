# LangSlice `nonlinear/ — generative-image registration`

Package guide for `src/langslice/nonlinear/`. The repo-level `CLAUDE.md` holds the
project-wide rules; this file holds what is specific to this package. `AGENTS.md`
here is a verbatim copy — edit one, mirror to the other.

- `nonlinear/` — generative-image registration: candidate generation, image
  provider adapters, Elastix runtime, affine and
  nonlinear result types, and `quick_affine.py` (silhouette affine preview;
  note the CLI groups `quick-affine` under `linear`). There is ONE path:
  direct — the handwritten base prompt goes to the image model verbatim
  (openai-oauth: the raw `codex/images/edits` endpoint, no routing model in
  between), one generation per run. The hosted-router session (`router.py`,
  a GPT conversation wielding the `image_generation` tool with self-review
  iteration) was DELETED 2026-08-28 on LSD_910 ground-truth evidence: on the
  anterior benchmark slice the first generation tracked the tissue and the
  router's "improved" retries degraded it into a generic atlas plate
  (truth-Dice 0.54 vs a clean first paint), and its prompt refinement is a
  standing corruption risk — the image model does better with our prompt
  untouched. The canvas pad the router used to choose is now the
  `--canvas-pad` knob (0-1.5, `RegisterRequest.canvas_pad`). The Elastix
  report is the ONLY
  diagnostic — the old `confirm_registration` gate and per-model prompt
  profiles were deleted. One base prompt (the original handwritten text, in
  `model_prompts.py`) serves every image model; the working
  editing copy is `_local/nonlinear_prompts.md`.
  The MODEL-FACING atlas render is drawn at canvas resolution, never
  NEAREST-upscaled: `render.filled_regions` traces each region at atlas
  resolution, low-pass filters the outline and fills the polygon at ~2048px
  (`_generate_colored_region_slice(..., smooth=True)`, the default), so a
  25/39um atlas reaches the model with boundary detail instead of a voxel
  staircase. Fills stay flat and exact-palette (classification is
  nearest-color), each region is stroked as well as filled so neighbours
  overlap instead of leaving background seams, and enclosed pinholes are
  closed. The ELASTIX side asks for `smooth=False` — that render is
  classified back and warped, so it must stay pixel-exact. `_fit_to_canvas`
  picks its resampling filter from the image: NEAREST for a flat region map,
  LANCZOS for the continuous-tone grayscale template.
  `generate_registration_candidate(image_prompt)` runs the whole downstream
  chain immediately (image model → Elastix → pixel classification → borders)
  and returns the images plus an Elastix report: error codes with the numbers
  behind them (`REGION_MISSING`, `REGION_COLLAPSED`, `WARP_FOLDS`,
  `TISSUE_UNCOVERED` — computed on the warped CLASSIFIED result, so color
  drift that still classifies coherently passes clean; pixels the edit left
  untouched (identical to the input canvas) are masked to background before
  classification, because a preserved white background otherwise classifies
  as the atlas root color; there is one real
  failure in this pipeline, Elastix not working on the image, and every code
  is a measured cause of it, not a standalone judgment); passing
  `generated_image` skips the provider call and evaluates an
  externally generated image (offline re-derivations, benchmarks). Under `--palette leaf-borders` the classifier's palette gains the
  hairline color of every region (`darker(color) -> that region's id`),
  because a model that paints the delineation back would otherwise have it
  cut through its own regions as background: measured on the Allen sagittal
  render, hairlines are ~20% of the foreground and land 70-140 RGB from any
  fill color. With the line color known, no border pixel becomes background
  and no new region id appears; the residual (family recovery 0.93 vs a flat
  render) is entirely WHICH of two neighbours owns a shared 2px line, which
  is two orders below the B-spline grid. The experimental `--palette
  family-flat` goes further: `_plane_families` collapses leaves onto the
  `_family_mapping` partition BEFORE painting, and the classifier's whole
  palette becomes the family colors plus their line shades, keyed to each
  family's representative id — so `_merge_classified` is already the identity
  on what it returns. Nothing else changes: the
  Elastix-side render (`smooth=False`) is byte-identical in every style. The
  Elastix step registers
  the two label maps as joint RGB at merged-family granularity (one
  AdvancedMeanSquares metric per channel plus the bending penalty; both
  maps merged through a SINGLE family mapping built on their union of ids
  so a family renders one color on both sides). RGB is a benchmarked
  choice, not a leftover: one-binary-channel-per-family (the label-
  registration literature's standard) and clamped per-family signed
  distance maps were both built and measured on the sag140 debris case, and
  every variant came out worse than not deforming at all (family agreement
  0.46-0.54 vs 0.675 identity vs 0.80 RGB) — Elastix's sampled ASGD
  optimizer starves on channels whose gradient lives only in thin boundary
  shells. Known residual flaw: in RGB a mismatched label earns partial
  credit whenever its color sits nearer than black, so preserved debris
  (dark pixels classify to far-away regions' shades, measurably CLOSER to
  the palette than real paint) can pull boundaries slightly; the candidate
  fix is a dense-evaluation engine (NiftyReg 4-D SSD / ANTs label
  registration), not more Elastix channels. Also measured: the
  TransformBendingEnergyPenalty at default-scale weights is INERT against
  0-255 mean-squares values (needs ~1e6-1e8 to bite) — and VisuAlign markers are sampled from the composed
  transformix deformation field at the B-spline control-grid spacing (the
  transform's own resolution; consumer-specific densities belong in the
  integration adapters), never read off the B-spline parameter map, which
  dropped the affine stage and mistook coefficients for displacements. The
  The block's CUTTING ANGLES are an input (`--pitch-deg`/`--yaw-deg`,
  `RegisterRequest`, `generate_registration_candidate`): non-zero angles
  reslice every atlas render — model-facing map, grayscale template,
  Elastix-side map, classification palette, leaf overlay — on that oblique
  plane instead of taking it flat off the voxel grid. Default 0 (flat),
  because nothing upstream fits them yet, but this is the single biggest
  lever in the whole fit and it dwarfs every knob inside Elastix. Measured
  on all 33 LSD_910 hand registrations (block cut at 4 degrees, angles set
  by hand to that): fit-only family dice 0.912 -> 0.945, boundary p95 39px
  -> 9px, better on 25 of 33 slices — and re-deriving 32 saved paintings
  through the fit, +0.022 dice paired, better on 29 of 32, against a
  per-draw sd of 0.02, with the paintings still made from a FLAT reference.
  Under a flat plane the residual is wide BANDS of misplaced anatomy
  (hippocampus, brainstem, one hemisphere ahead of the other); at the right
  angles it collapses to boundary-width lines. `pixel_size_um` puts every
  atlas render at TRUE physical scale on the slice canvas (each render's
  um/px derived from the annotation slice's anatomy, never the letterbox
  padding; anatomy centered via `_anatomy_focus`, and the canvas auto-grows
  through `_pad_to_contain_atlas` when the true-scale anatomy would exceed
  it) so the model sees two comparably sized brains; no pixel size falls
  back to fit-to-canvas. Known residual, measured over all 33 M01 hand
  registrations by physical extent of the GT region maps: the specimen runs
  ~10%% (ML) to ~14%% (DV) smaller than the Allen average, so the true-scale
  atlas lands slightly larger than the tissue and the fit side alone pays
  ~0.004 dice for it — calibration is on for the model-side benefit of
  size-matched references. Settled ablations on the same panel:
  a finer B-spline grid (/48, /72 rather than /36) buys +0.004 dice, the
  fixed-mask dilation is flat from 0 to 32px, and the bending penalty is
  flat at 1e5-1e6 and HARMFUL at 1e7 — the tail is a plane problem, not a
  regularization one. The
  canvas pad (`--canvas-pad`, 0-1.5) grows the working
  canvas by a margin matched to the slice's own background color (never
  black-on-white) so a fragment or hemibrain's COMPLETE painted anatomy can
  exceed the original image bounds; after all padding the canvas silently
  pads one axis if the aspect ratio falls outside what the image path can
  output (`model_prompts.aspect_ratio_limits`, gpt-image-2 =
  1:3..3:1 — pad only, never crop; a mismatched ratio would undo edit-mode
  pixel alignment); markers are reported in
  original-image pixels, unclipped — out-of-bounds correspondences are
  legitimate there, and clipping is an adapter concern. It
  runs after a linear
  placement step, whether that step is `langslice linear ...` or the user's
  own tool (in ABBA/QUINT workflows, linear placement happens first and
  LangSlice-nonlinear replaces the manual spline/BigWarp deformation step).

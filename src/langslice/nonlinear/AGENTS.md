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
  editing copy is `_local/nonlinear_prompts.md`. The model's inputs are the
  histology slice followed by THREE colored region maps bracketing the
  estimated position at ±125um (`REFERENCE_OFFSETS_MM`, human placement
  error), anterior → posterior — the grayscale template input was dropped
  2026-09-01 with the prompt updated to match; the slice itself goes through
  the shared `image_prep.adaptive_preprocess` first (`--preprocess auto`,
  the default on BOTH the linear and nonlinear CLIs / API requests) so the
  model sees structural detail, not one dim raw channel. CORONAL planes
  black out the ventricular system (`_annotation_slice` +
  `_VENTRICLE_KEYWORDS`, re-instated 2026-09-01 — ventricles are holes in
  coronal histology; sagittal keeps them per Nash's original instruction):
  the ids become background at the shared annotation entry point, so
  render, classifier palette, Elastix side and ledger stay consistent, and
  the prompt tells the model the black holes are holes.
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
  angles it collapses to boundary-width lines. Every atlas render is FIT TO
  CANVAS — a hard boundary, pinned by `tests/test_reference_scale_boundary.py`:
  `_fit_to_canvas` takes no scale, and the physical-placement helpers were
  deleted. `pixel_size_um` is recorded in metadata only. True-physical
  placement (on 2026-08-29 to 2026-09-05, never measured on the model side
  until Nash saw the paintings) drew the atlas 20-30%% larger than the
  shrunken LSD tissue (specimens run ~10-14%% under the Allen average) and
  the model copied the oversized plate instead of repainting the tissue:
  painting dice 0.52 vs 0.62 fit-to-canvas on A_02+A_04 (4 Codex draws
  each), visibly worse, on BOTH lanes; the fit side never needed it either
  (0.912 fit-to-canvas vs 0.908 physical, 33 slices). Model paint is classified
  in PAINT MODE (`_classify_pixels_to_region_ids(..., paint=True)`): the model
  often paints translucently, so tissue brightness modulates a region's
  lightness; the lightness axis is down-weighted (0.25) in the color distance
  so hue/saturation decide. Measured 2026-09-05 on 12 draws: dice 0.583 ->
  0.606, despeckle churn 0.067 -> 0.044, the translucent draw 0.158 -> 0.090.
  Unmixing against the input pixel and a gray-color gate both measured no gain
  and were dropped. The leaf review render keeps the ventricles (blackout
  off) and its warped id map is saved as `warped_leaf_ids.npz` — the input to
  landmark scoring against hand registrations (`_local/nonlinear_eval/landmarks.py`,
  Nash's visual-landmark list). Settled ablations on the same panel:
  a finer B-spline grid (/48, /72 rather than /36) buys +0.004 dice, the
  fixed-mask dilation is flat from 0 to 32px, and the bending penalty is
  flat at 1e5-1e6 and HARMFUL at 1e7 — the tail is a plane problem, not a
  regularization one. The image model's quality tier is not a lever either
  (2026-09-05, 12 draws/arm on the same three M01 slices, byte-identical
  inputs): the subscription lane runs gpt-image-2 at quality `auto` — it
  chose the medium token band (~1300-1600 output image tokens) for our
  edits — and scored painting dice 0.537; the public API at the same pixel
  budget scored 0.461 (low), 0.526 (medium), 0.481 (high). Paying for
  `high` made the anterior slice WORSE (plate-copied striatum, more
  cortical confetti). `openai-api` honors `thinking_level` as the tier
  and pins `size` to the subscription lane's pixel budget (1024x1536
  worth of pixels at the canvas aspect) so the lanes stay comparable. The
  canvas pad (`--canvas-pad`, 0-1.5) grows the working
  canvas by a margin matched to the slice's own background color (never
  black-on-white) so a fragment or hemibrain's COMPLETE painted anatomy can
  exceed the original image bounds; after all padding the canvas silently
  pads one axis if the aspect ratio falls outside what the image path can
  output (`model_prompts.aspect_ratio_limits`, gpt-image-2 =
  1:3..3:1 — pad only, never crop; a mismatched ratio would undo edit-mode
  pixel alignment); markers are reported in
  original-image pixels, unclipped — out-of-bounds correspondences are
  legitimate there, and clipping is an adapter concern.
  DRAWS (`--draws`, `RegisterRequest.draws`, default 1 = today's single
  generation) asks the image model for K independent paintings of the SAME
  request and registers their per-pixel MAJORITY vote (ties go to the lowest
  draw index, so exactly TWO kept draws degenerate to the first of them — ask
  for three or more), repainted from the voted ids through the atlas LUT on black so
  Elastix, the ledger and every overlay still see one painting; the raw draws
  are kept as `generated_segmentation_draw{i}.png` and
  `generated_segmentation.png` stays the voted result. Measured on the
  hand-registered slices (3 slices x 8 draws): the vote beats the mean single
  draw by 0.04-0.08 family dice and lands near the BEST draw of the set — it
  buys best-of-K without having to know which draw is best. The TRANSLUCENCY
  gate (`max_off_palette`, default 0.08) drops the draws that half-preserve
  the tissue texture instead of painting flat color: the share of painted
  foreground (max channel >= 20) that classifies as off-palette background,
  taken BEFORE the preserved mask and despeckle, runs 0.01-0.03 on clean
  draws and 0.10-0.17 on translucent ones, so the threshold sits between the
  two populations. A draw over it leaves the vote unless that would drop
  every draw (then they all stay — something has to register), and with one
  draw the number is only recorded; kept/dropped indices and every draw's
  fraction go in the candidate metadata. DEFORMATION (`--deformation`,
  default `bspline`) picks the Elastix stages: `affine` emits only the affine
  map in BOTH parameter builders, so the fit is the affine stage alone.
  Measured on the same slices, the B-spline stage driven by generated
  paintings lands BELOW the affine stage alone (0.65 vs 0.735 flat, 0.74 vs
  0.80 oblique) — the paintings' boundaries are not accurate enough to bend
  toward. Nothing downstream assumes a B-spline map: the deformation field
  comes from transformix (composing whatever stages exist, verified to carry
  the affine displacement in full) and the VisuAlign markers are sampled off
  that field. INIT (`--init`, `RegisterRequest.init`, default `atlas` =
  today's path: the section is the canvas and the model paints it against the
  three atlas maps) chooses what the painting starts from. `silhouette`
  builds the PRIOR first (`prior.py`): the atlas plane at (position, pitch,
  yaw) placed on the section's own outline by the shared moments fit
  (`affine._moments_pose`/`_affine_from_pose`), soft-label smoothed to take
  the voxel staircase off the upscaled plane, and painted by the SAME
  `render.paint_labels` call the model-facing reference makes — flat palette
  fills, ventricles black, darker fill-change lines at the same width rule —
  canvas-sized, saved as `input_prior.png`. Only the two ROTATIONS are tried
  (the reflections score the same silhouette IoU on a near-symmetric section
  and land anatomy on the wrong hemisphere); the one with the higher tissue
  IoU wins and both numbers go in the metadata beside `init` and `provider`.
  With a real provider, `init=silhouette` sends the prior as Image 1 and the
  preprocessed section as Image 2 (`reference_images_override`; the ±125um
  atlas band is NOT sent) under `model_prompts.prior_refinement_prompt` —
  move each boundary onto the visible transition, change nothing else — and
  the preserved-background mask is OFF for that call, because unchanged
  pixels on a prior canvas are paint the model kept, not unpainted slide
  (production decides this per call from `init`; the module flag
  `PRESERVED_BACKGROUND_MASKING` remains for offline arms). `provider="none"`
  (canonical in `providers/registry.py`, and it requires `init=silhouette`)
  calls no model at all: the prior IS the painting, and with
  `--deformation affine` the entire downstream chain — Elastix, classified
  warp, markers, overlays, ledger, report, exports — runs on the placement
  alone. Measured against the LSD_910 hand registrations, the placement
  scores 0.82 mean family dice on the fitted cutting plane, better than every
  image-model configuration measured, and as the model's canvas it raises the
  painting floor from ~0.5 to ~0.8 — which is why the prior exists on both
  sides of the model. The classifier is untouched by any of it: every prior
  pixel, fill or line, is an exact palette color, so classification is
  lossless. The overnight run of 2026-09-05 (report in
  `_local/nonlinear_eval/runs/night_report_2026-09-05.md`) is the evidence
  behind these knobs: on A_02/A_04/A_08 with 8 draws per arm the B-spline
  stage driven by any painting scored BELOW its own affine stage (0.65 vs
  0.735 flat, 0.74 vs 0.80 oblique), the affine-placed atlas beat the
  8-draw vote in every family on every slice, and no fit variant (coarser
  grid, bending x10, vote target, edge-only metric, consensus masks)
  recovered the loss; with the prior as canvas the model's edits landed
  0.04-0.13 below the prior even after consensus and morphological
  filtering, though it visibly perceives ventricles. The family-Dice
  metric against the spline-warped hand registrations saturates near
  0.87-0.90 for a good affine (boundary precision ~50um), so judging
  refinements finer than that needs a finer ground truth. The production
  backbone (`--provider none --init silhouette --deformation affine`)
  scored 0.874/0.808/0.751 with the fitted angles and 0.813/0.797/0.668
  flat, versus 0.741/0.672/0.655 for the best image-model arm. It
  runs after a linear
  placement step, whether that step is `langslice linear ...` or the user's
  own tool (in ABBA/QUINT workflows, linear placement happens first and
  LangSlice-nonlinear replaces the manual spline/BigWarp deformation step).

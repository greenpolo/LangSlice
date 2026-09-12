# LangSlice `nonlinear/ — generative-image registration`

Package guide for `src/langslice/nonlinear/`. The repo-level `CLAUDE.md` holds the
project-wide rules; this file holds what is specific to this package. `AGENTS.md`
here is a verbatim copy — edit one, mirror to the other.

- `nonlinear/` — generative-image registration: candidate generation, image
  provider adapters, Elastix runtime, affine and
  nonlinear result types, and `quick_affine.py` (silhouette affine preview;
  note the CLI groups `quick-affine` under `linear`).
  The atlas geometry these draw from lives in `atlas/render.py`, not here:
  the annotation slice (`_annotation_slice` is an alias of
  `atlas.render.annotation_slice`), the family mapping (`_family_mapping` of
  `family_mapping`), `region_contours`, and the `border_color` / `darker` /
  `BORDER_DARKEN` / `is_dark_background` shade rules, which
  `nonlinear/render.py` re-exports. `linear` draws its physical overlay from
  the same functions, so both methods put a boundary in the same place. There is ONE path:
  direct — the handwritten prompt goes to the image model verbatim
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
  profiles were deleted.
- THE LINEUP (what the model is sent, in order). Image 1 is ONE colored atlas
  region map of the target plane — flat Allen-organized colors, one per
  registration unit, drawn pixel-exact off the annotation
  (`_generate_colored_region_slice`), NEAREST-upscaled to a
  `MODEL_MAP_MIN_LONG_EDGE` (1024) long edge and then letterboxed onto black
  to the SECTION's aspect (`letterbox_to_aspect`). THIS is the image the model
  edits (`SegmentationGenerationRequest.slice_image`, the first image every
  transport sends). Image 2 is the grayscale atlas template of the same plane
  (`_model_facing_template`, restretched on the plate's own 99.5th
  percentile), upscaled and letterboxed identically. Image 3 is the section
  itself, at the working canvas (long edge 2048), unpadded and unaltered.
  The three therefore share one frame, which is what the prompt pins the
  answer to. Nothing else reaches the model: no neighbouring planes at
  ±125um, no borders drawn over the fills, no blacked-out ventricles, no
  CLAHE, no silhouette-affine prior painted onto the canvas, and never the
  section repainted in place. This is the APRIL LINEUP, restored 2026-09-12
  after it beat the elaborated production lineup on the LSD_910 panel; the
  earlier design is recoverable from git, not from a flag.
- The ANSWER comes back in the frame it was given, except on lanes with fixed
  output frames, which return the nearest legal aspect with our frame
  letterboxed inside it: `crop_to_aspect` (tolerance 0.5%) center-crops that
  excess off before the painting is resampled onto the section frame. Never
  stretch — a mismatched ratio resampled whole slides every painted boundary
  off the tissue. Nothing is masked out by comparison with the input: the
  model edits the ATLAS MAP, so pixels it left untouched are regions it had
  no reason to move. (The old `_preserved_background_mask`, which erased
  untouched pixels as background, belonged to the retired repaint-the-section
  contract and would now erase correct anatomy.)
- Every atlas render in the run is fit-to-canvas at its own proportions —
  the model-facing pair by letterbox, the Elastix-side map by `_fit_to_canvas`
  (uniform scale, centered, NEAREST) — so the map Elastix registers sits in
  exactly the geometry the model was shown. There is no physical-scale
  placement here any more (`pixel_size_um`, `_pad_to_contain_atlas`,
  `_anatomy_focus`): true-scale plates were measured as something the model
  COPIES rather than deforms, and the April lineup was benchmarked without
  them. Physical placement lives on in `linear/`, where the user, not a
  model, is looking at the overlay.
  `generate_registration_candidate(image_prompt)` runs the whole downstream
  chain immediately (image model → Elastix → pixel classification → borders)
  and returns the images plus an Elastix report: error codes with the numbers
  behind them (`REGION_MISSING`, `REGION_COLLAPSED`, `WARP_FOLDS`,
  `TISSUE_UNCOVERED` — computed on the warped CLASSIFIED result, so color
  drift that still classifies coherently passes clean; there is one real
  failure in this pipeline, Elastix not working on the image, and every code
  is a measured cause of it, not a standalone judgment); passing
  `generated_image` skips the provider call and evaluates an
  externally generated image (offline re-derivations, benchmarks). The
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
  credit whenever its color sits nearer than black, so debris (dark pixels
  classify to far-away regions' shades, measurably CLOSER to
  the palette than real paint) can pull boundaries slightly; the candidate
  fix is a dense-evaluation engine (NiftyReg 4-D SSD / ANTs label
  registration), not more Elastix channels. Also measured: the
  TransformBendingEnergyPenalty at default-scale weights is INERT against
  0-255 mean-squares values (needs ~1e6-1e8 to bite) — and VisuAlign markers are sampled from the composed
  transformix deformation field at the B-spline control-grid spacing (the
  transform's own resolution; consumer-specific densities belong in the
  integration adapters), never read off the B-spline parameter map, which
  dropped the affine stage and mistook coefficients for displacements.
- THE PROMPT is one text per model family, in `model_prompts.py`, chosen by
  PROVIDER: `V14` for `gemini-api` (written to Google's image-prompting
  guide, positive framing throughout) and `V14_EDIT_MOUSE` for the GPT image
  lanes (OpenAI's guide: indexed inputs, invariants listed, the species and
  atlas named). Both are byte-identical to the benchmark's own copies in
  `_local/nonlinear_eval/prompts.py` (`v14`, `v14edit_mouse`) — that is how
  they were measured, and `tests/test_nonlinear_prompts.py` compares them
  whenever that local tree is present. They are coronal-worded; another
  plane gets the same text with that one word swapped. Tune against real
  runs, not per-model prose.
- The block's CUTTING ANGLES are an input (`--pitch-deg`/`--yaw-deg`,
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
  angles it collapses to boundary-width lines. Settled ablations on the same
  panel: a finer B-spline grid (/48, /72 rather than /36) buys +0.004 dice,
  the fixed-mask dilation is flat from 0 to 32px, and the bending penalty is
  flat at 1e5-1e6 and HARMFUL at 1e7 — the tail is a plane problem, not a
  regularization one. The
  canvas pad (`--canvas-pad`, 0-1.5) grows the working
  canvas by a margin matched to the slice's own background color (never
  black-on-white) so a fragment or hemibrain's COMPLETE painted anatomy can
  exceed the original image bounds; the atlas renders follow it, because they
  are letterboxed to whatever aspect the padded canvas ends up with. After
  all padding the canvas silently
  pads one axis if the aspect ratio falls outside what the image path can
  output (`model_prompts.aspect_ratio_limits`, gpt-image-2 =
  1:3..3:1 — pad only, never crop; a ratio the lane cannot return would come
  back letterboxed and have to be cropped); markers are reported in
  original-image pixels, unclipped — out-of-bounds correspondences are
  legitimate there, and clipping is an adapter concern. It
  runs after a linear
  placement step, whether that step is `langslice linear ...` or the user's
  own tool (in ABBA/QUINT workflows, linear placement happens first and
  LangSlice-nonlinear replaces the manual spline/BigWarp deformation step).

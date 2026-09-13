# LangSlice `nonlinear/ — generative-image registration`

Package guide for `src/langslice/nonlinear/`. The repo-level `CLAUDE.md` holds the
project-wide rules; this file holds what is specific to this package. `AGENTS.md`
here is a verbatim copy — edit one, mirror to the other.

## Current direction and review standard

The April input arrangement is the production baseline. The second-prompt /
reply approach is an active experiment to improve anatomical outlines. Historical
scores have not ruled it out; it is not yet a production second turn. Preserve
its prompts, exact inputs, raw replies and conversation history.

Judge generated outputs primarily by visual inspection against actual histology.
Inspect raw replies before classification, resizing or registration, then inspect
model boundaries against the original section. Describe concrete anatomical
successes, errors and uncertainty. Metrics, downstream registration and agreement
with human annotations cannot establish model-output quality. Historical
metric-based verdicts below are experiment history, not settled conclusions.
A useful model output can be damaged by the subsequent fit. Check a redrawn
photograph's boundaries against the original tissue before accepting alignment.

## Image-generation prompts: sentence-by-sentence review

Before changing or sending an experimental prompt, scrutinize EVERY sentence:

- State its intended effect and plausible alternative interpretations. Remove
  ambiguity, redundant demands and contradictions with earlier sentences; look
  for wording that is simultaneously restrictive and vague.
- Identify each input by its actual order, content and role. Specify which image
  is edited, what changes, what is preserved and the intended output frame.
- For border refinement, distinguish moving boundaries to visible tissue
  transitions from copying or overlaying a reference. Calling reference lines
  "correct" can imply transferring them unchanged.
- For a reply, verify the actual history and attachments. Say "your previous
  image" only when it really is the model's earlier reply in the supplied
  history. Record constructed history explicitly.
- Prefer positive descriptions of the desired result and small, targeted edits.
  Preserve separate Gemini and OpenAI base wordings; change one idea at a time
  and save the exact text and inputs used, not just a prompt nickname.

This review applies even to a one-sentence follow-up. Inspect the resulting
images before declaring a prompt improvement or rejecting an approach.

The multi-image provider helper preserves every returned image in reply order,
skips text parts during extraction, and requests TEXT plus IMAGE for multi-image
requests at 1K or below (IMAGE only at higher tiers). The single-image helper
continues to return the last image. This support does not itself create a second
conversational turn.

- `nonlinear/` — generative-image registration: candidate generation, image
  provider adapters, Elastix runtime, affine and
  nonlinear result types, and `quick_affine.py` (silhouette affine preview;
  note the CLI groups `quick-affine` under `linear`).
  The atlas geometry these draw from lives in `atlas/render.py`, not here:
  the annotation slice (`_annotation_slice` wraps
  `atlas.render.annotation_slice`; its ventricle blackout is OFF by default
  now — the model deforms an atlas plate rather than painting the section,
  so a ventricle is a region to place and a landmark for the fit, not a hole
  with nothing to paint), the family mapping (`_family_mapping` of
  `family_mapping`), `region_contours`, and the `border_color` / `darker` /
  `BORDER_DARKEN` / `is_dark_background` shade rules, which
  `nonlinear/render.py` re-exports. `linear` draws its physical overlay from
  the same functions, so both methods put a boundary in the same place. There is ONE path:
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
  profiles were deleted. THE LINEUP (what the model is sent, in order) is the APRIL
  LINEUP, restored 2026-09-12 after it beat the elaborated production
  lineup on the LSD_910 panel. Image 1 is ONE colored atlas region map of
  the target plane — flat Allen-organized colors, one per registration unit,
  drawn pixel-exact off the annotation (`_generate_colored_region_slice`,
  no smoothing, no delineation lines, ventricles kept),
  NEAREST-upscaled to a `MODEL_MAP_MIN_LONG_EDGE` (1024) long edge and then
  letterboxed onto black to the SECTION's aspect (`letterbox_to_aspect`).
  THIS is the image the model edits
  (`SegmentationGenerationRequest.slice_image`, the first image every
  transport sends). Image 2 is the grayscale atlas template of the same
  plane (`_model_facing_template`, restretched on the plate's own 99.5th
  percentile), upscaled and letterboxed identically. Image 3 is the section
  at the working canvas, through the shared `image_prep.adaptive_preprocess`
  first (`--preprocess auto`, the default on BOTH the linear and nonlinear
  CLIs / API requests — the April benchmark ran with it, and without it a
  dim fluorescence section reaches the model as a near-black field). The
  three share one frame, which is what the prompt pins the answer to.
  Nothing else reaches the model: no neighbouring planes at ±125um, no
  borders drawn over the fills, no blacked-out ventricles, no
  silhouette-prior canvas, and never the section repainted in place. The
  earlier design is recoverable from git, not from a flag.
  THE ANSWER comes back in the frame it was given, except on lanes with
  fixed output frames, which return the nearest legal aspect with our frame
  letterboxed inside it: `crop_to_aspect` (tolerance 0.5%) center-crops that
  excess off, then `_onto_canvas` puts the draw on the canvas grid. Never
  stretch — a mismatched ratio resampled whole slides every painted boundary
  off the tissue. Nothing is masked out by comparison with the input either:
  the model edits the ATLAS MAP, so pixels it left untouched are regions it
  had no reason to move. (The old `_preserved_background_mask`, which erased
  untouched pixels as background, belonged to the retired
  repaint-the-section contract and would now erase correct anatomy.)
  THE PROMPT is one text per model family, in `model_prompts.py`, chosen by
  PROVIDER: `V14` for `gemini-api` (Google's image-prompting guide, positive
  framing throughout) and `V14_EDIT_MOUSE` for the GPT image lanes (OpenAI's
  guide: indexed inputs, invariants listed, species and atlas named). Both
  are byte-identical to the benchmark's own copies in
  `_local/nonlinear_eval/prompts.py` (`v14`, `v14edit_mouse`) — that is how
  they were measured, and `tests/test_nonlinear_prompts.py` compares them
  whenever that local tree is present. They are coronal-worded; another
  plane gets the same text with that one word swapped.
  `generate_registration_candidate(image_prompt)` runs the whole downstream
  chain immediately (image model → Elastix → pixel classification → borders)
  and returns the images plus an Elastix report: error codes with the numbers
  behind them (`REGION_MISSING`, `REGION_COLLAPSED`, `WARP_FOLDS`,
  `TISSUE_UNCOVERED` — computed on the warped CLASSIFIED result, so color
  drift that still classifies coherently passes clean; there is one real
  failure in this pipeline, Elastix not working on the image, and every code
  is a measured cause of it, not a standalone judgment); passing
  `generated_image` skips the provider call and evaluates an
  externally generated image (offline re-derivations, benchmarks). The atlas RENDER STYLES (`atlas.recolor`: `family`,
  `leaf-borders`, `family-flat`) no longer reach any model. The lineup fixes
  what it is shown — one flat, undelineated map — so registration PINS the
  style for the whole call (`_pinned_registration_palette`, "family"):
  a floating style would change the render without changing what the
  classifier expects of it. The styles and their measurements stay for
  renders people look at, and the classifier still knows each region's line
  shade (`darker(color) -> that region's id`), which on an undelineated map
  only ever rescues a drifted dark pixel into its own region. The
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
  angles it collapses to boundary-width lines. EVERY atlas render is fit-to-canvas at its own
  proportions — the model-facing map by letterbox, the Elastix-side map by
  `_fit_to_canvas` (uniform scale, centered, NEAREST) — so the map Elastix
  registers sits in exactly the geometry the model was shown. There is no
  TRUE-PHYSICAL placement on this path any more (`pixel_size_um`,
  `_pad_to_contain_atlas`, `_anatomy_focus`, `_physical_scale_for` are
  gone; `tests/test_reference_scale_boundary.py` now pins the one-geometry
  rule instead). The measurement behind that, 2026-09-05 on M01 A_02: the
  atlas plane is 7.80 mm wide and the tissue 7.46 mm, so true scale draws
  the plate at 1.05x the tissue width where fit-to-canvas (the whole plane,
  margins included) draws it at 0.69x — and the image MODEL reacts to the
  plate's size, copying the true-scale plate on 7 of 8 draws (painting dice
  0.621) but editing in place on 5 of 8 with the smaller one (0.669, best
  draws 0.76). A plate the size of the tissue reads as the answer, a small
  one as a legend. The April lineup was benchmarked without physical
  placement, and calibration by pixel size lives on in `linear/`, where a
  person reads the overlay. Model paint is classified
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
  worth of pixels at the canvas aspect) so the lanes stay comparable. The WORKING CANVAS is the image path's own output
  frame (`prepare_canvas(native_canvas=True)`, the default, via
  `model_prompts.native_output_size`): the layout is worked out at the
  long-edge rule, scaled to fit that frame and padded out to it exactly, so
  the section is resampled ONCE, straight from the original to its final
  size, and the model works on the grid its answer comes back on. Nash
  2026-09-11: every input the model must rescale to its output is a pixel
  the output cannot carry, paid for twice (input tokens in, a resample out).
  The frames are measured, not guessed — the Codex lane returns a fixed
  ~1.573 Mpx budget at the input aspect (read off 19 outputs), Gemini
  returns one fixed frame per (aspect, size tier) (`_GEMINI_FRAMES`), and
  the openai-api lane is sent the same budget on its 16-px grid
  (`_api_edit_size`) so the lanes stay comparable. `canvas_long_edge` pins
  the long edge instead (the pre-2026-09-11 rule, and what the benchmark's
  April arm used: 2048). The atlas renders follow whatever aspect the canvas
  ends up with, because they are letterboxed to it. The
  canvas pad (`--canvas-pad`, 0-1.5) grows the working
  canvas by a margin matched to the slice's own background color (never
  black-on-white) so a fragment or hemibrain's COMPLETE painted anatomy can
  exceed the original image bounds; after all padding the canvas silently
  pads one axis if the aspect ratio falls outside what the image path can
  output (`model_prompts.aspect_ratio_limits`, gpt-image-2 =
  1:3..3:1 — pad only, never crop; a ratio the lane cannot return would come
  back letterboxed and have to be cropped); markers are reported in
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
  taken BEFORE despeckle, runs 0.01-0.03 on clean
  draws and 0.10-0.17 on translucent ones, so the threshold sits between the
  two populations. A draw over it leaves the vote unless that would drop
  every draw (then they all stay — something has to register), and with one
  draw the number is only recorded; kept/dropped indices and every draw's
  fraction go in the candidate metadata. The ELASTIX STAGE (`elastix=`, default `"rgb"`) picks
  which fit runs: `"rgb"` is the measured default described above;
  `"april-borders"` is the April 2026 stage (B-spline only, on single-pixel
  border images, mean squares, grid 64, bending 10) kept selectable as an
  experiment arm. DEFORMATION (`--deformation`,
  default `bspline`) picks the Elastix stages: `affine` emits only the affine
  map in BOTH parameter builders, so the fit is the affine stage alone.
  Measured on the same slices, the B-spline stage driven by generated
  paintings lands BELOW the affine stage alone (0.65 vs 0.735 flat, 0.74 vs
  0.80 oblique) — the paintings' boundaries are not accurate enough to bend
  toward. Nothing downstream assumes a B-spline map: the deformation field
  comes from transformix (composing whatever stages exist, verified to carry
  the affine displacement in full) and the VisuAlign markers are sampled off
  that field. THE SILHOUETTE PRIOR (`prior.py`) is what
  `provider="none"` registers instead of a painting — there is no `init`
  knob any more, because Image 1 is the atlas map and the prior cannot also
  be it. The prior is the atlas plane at (position, pitch,
  yaw) placed on the section's own outline by the shared moments fit
  (`affine._moments_pose`/`_affine_from_pose`), soft-label smoothed to take
  the voxel staircase off the upscaled plane, and painted by the SAME
  `render.paint_labels` call the model-facing reference makes — flat palette
  fills, ventricles black, darker fill-change lines at the same width rule —
  canvas-sized, saved as `input_prior.png`. Only the two ROTATIONS are tried
  (the reflections score the same silhouette IoU on a near-symmetric section
  and land anatomy on the wrong hemisphere); the one with the higher tissue
  IoU wins and both numbers go in the metadata beside `provider`.
  `provider="none"` (canonical in `providers/registry.py`)
  calls no model at all: the prior IS the painting, and with
  `--deformation affine` the entire downstream chain — Elastix, classified
  warp, markers, overlays, ledger, report, exports — runs on the placement
  alone. Measured against the LSD_910 hand registrations, the placement
  scores 0.82 mean family dice on the fitted cutting plane, better than every
  image-model configuration measured before the April lineup — which makes it
  the backbone every model run has to beat. (As a CANVAS for the model it
  raised the painting floor from ~0.5 to ~0.8, which is what the April
  lineup's Image 1 now occupies; that arm is in git, not in a flag.) The
  classifier is untouched by any of it: every prior
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
  backbone (`--provider none --deformation affine`)
  scored 0.874/0.808/0.751 with the fitted angles and 0.813/0.797/0.668
  flat, versus 0.741/0.672/0.655 for the best image-model arm. It
  runs after a linear
  placement step, whether that step is `langslice linear ...` or the user's
  own tool (in ABBA/QUINT workflows, linear placement happens first and
  LangSlice-nonlinear replaces the manual spline/BigWarp deformation step).

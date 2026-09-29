# LangSlice `nonlinear/` — generative-image registration

Package guide; `AGENTS.md` is a verbatim twin of this file.
See `docs/nonlinear_design.md` for the design and coordinate contracts.

## Supported design

The stack agent's optional `nonlinear` task uses a supplied linear placement via
top-level `registration_tool.py`. Its only image tool is
`correct_slice_borders(id, additional_notes="")`: fixed placed-border correction
prompt plus optional notes, one retained reply per geometry, raw and extracted
images returned separately, no fit, atlas search, prompt replacement or rejection.
See `docs/nonlinear_image_tool.md` for its contract and prompt sentence review.
The standalone atlas route below is not available through this agent tool.

Exactly two border-based routes, chosen by whether a placement is supplied.
No model is ever shown a colored region map: the model-facing atlas is a
grayscale plate with thin yellow family borders, so atlas colormap quality
never matters.

- **Route "supplied"** — `initial_atlas_to_slice` is given (linear agent, ABBA
  host, or the top-level `registration_handoff` bridge). Image 1 is the tissue
  with the rough yellow family borders drawn on it, Image 2 the identical clean
  tissue; ONE model call with `prompts.border_refinement_prompt(plane)`, then
  the residual border fit. Production; not under test.
- **Route "atlas"** — no placement supplied. Image 1 is the clean tissue,
  Image 2 the outlined grayscale atlas (`outlined_atlas_template`: grayscale
  plate plus yellow family borders, oriented by `image_axes` then the explicit
  `atlas_mirror_lr`). ONE model call with `prompts.pass1_atlas_prompt(plane,
  provider)`; with `passes=2` a second call with `prompts.pass2_atlas_prompt`
  sends Image 1 clean tissue, Image 2 pass 1's extracted lines redrawn on the
  tissue, Image 3 the outlined atlas. The rough placement for the fit is the
  local silhouette-moments fit (`prior.place_plane_on_tissue_with_matrix`); the
  model's lines then go through the SAME residual fit as route "supplied".
  Metadata: `prior["source"] = "silhouette_moments_atlas_route"`, `passes`,
  `prior["atlas_route_model_calls"]`. Pass 2 is optional: on clean coronal
  sections it moves lines by about a pixel; its value on large sagittal and
  heavily damaged sections is untested. Kept for experiments, not the
  production direction (Nash, 2026-09-23): the silhouette placement it starts
  from is broken by exactly the outline damage that matters, and no local
  correction recovers a global placement error. Nonlinear needs linear first.
- `provider="none"` — model-free diagnostic: a supplied placement or the
  silhouette placement, zero model calls, zero residual.
- A caller-supplied `generated_image` with no placement replays route "atlas"
  with zero model calls.

How corrected borders become a deformation is not designed yet. The agreed
order is: design the deformation algorithm, then settle the border output format
it consumes, then test (`docs/interface_design.md`). Until then the residual fit
is off by default and the model call is judged on its raw lines.

Atlas labels and grayscale references must use the same position, plane, cutting
angles and orientation. Reflection is explicit (`atlas_mirror_lr`), never guessed
from a symmetric silhouette. Supplied matrices map the oriented native annotation
grid to original input-image pixel centers. Carry resize, padding, initial
affine/nonlinear mapping and residual deformation through exports; never export
the residual alone.

## File map

- `image_gen_registration.py` — model-facing canvas geometry
  (`prepare_canvas`, aspect helpers), `outlined_atlas_template`, and
  `generate_registration_candidate`, the public dispatcher (thin; routing lives
  in `border_registration.py`, imported lazily to avoid a circular import).
- `border_registration.py` — `generate_border_registration_candidate`: route
  selection, rough placement, the atlas-route model calls, the shared residual
  fit, coordinate composition (`composed_correspondences`,
  `composed_native_map`) and the candidate contract.
- `border_refinement.py` — `refine_borders`: the correction request,
  yellow-line extraction (`extract_thinned_lines`, `yellow_mask`, `thin`),
  residual Elastix fit and nearest-neighbor label warp. `deformation="none"`
  (the CLI default since 2026-09-22) skips the fit entirely and returns an
  identity residual: the model call is judged on its raw lines while its
  design is open, and the fit stage is not under evaluation.
  `integrations/abba.py` shares this core directly.
- `prompts.py` — the three prompt functions; the OpenAI-GPT or Gemini wording
  is selected by `canonical_provider(provider)`.
- `prior.py` — `place_plane_on_tissue_with_matrix`, the silhouette-moments
  placement.
- `providers.py` — `SegmentationGenerationRequest` /
  `generate_warped_segmentation_image`: the one-call-in, one-image-out
  transport adapter every prompt call goes through. `mode` (default `"edit"`)
  is the edit-vs-generate task semantic each transport translates its own way.
- `model_prompts.py` — canvas/aspect facts only (`image_model_family`,
  `aspect_ratio_limits`, `gemini_aspect_for`, `native_output_size`).
- `image_gen_helpers.py` — Elastix and geometry helpers for the residual fit.
- `render.py` — review-grade rendering only.
- `runtime.py` — `estimate_registration`: orchestration, debug artifacts,
  trace events.
- `registration_handoff.py` (top-level) — prepares supplied linear geometry.
  `registration_tool.py` uses it for the opt-in annotation tool, leaving residual
  fitting and transformation export to the separate registration stage.

## Visual review is essential

Inspect raw model replies BEFORE resizing, extraction or registration. Next inspect
extracted boundaries on the original photograph, then fitted atlas borders on that
same photograph. Describe concrete anatomical successes, errors and uncertainty.
A model may redraw tissue; its redrawn photograph is not the evaluation reference.
A useful model correction can be damaged by the fit. Metrics and fit error codes
cannot establish anatomical quality; report disagreement with visual evidence.

Keep rough overlay, exact prompt, raw reply, original-photo correction overlay,
fitted overlay and labels separately. Preserve existing local experiments, but do
not publish local-only paths or make historical metric verdicts design requirements.

## Prompt review: every sentence

Before changing or sending a prompt:

- State each sentence's intended effect and plausible alternative interpretations.
  Remove ambiguity, redundancy and contradictions. No sentence is exempt.
- Identify actual attachment order, content and role. Specify what is edited,
  preserved, and the output frame.
- Never write a visibility-conditioned rule ("no tissue, no line", "border what
  you see"). A visible slide feature (bubble, stain, debris) must NOT get a
  line, and an indistinct region MUST still get its atlas line. The atlas image
  alone decides which boundaries exist. The only exclusion a prompt may state
  is tissue physically torn away or missing from the section.
- Distinguish moving lines onto visible tissue from copying supplied lines.
  Calling rough lines "correct" can imply retaining them unchanged.
- Verify actual history before referring to an earlier reply. A second request
  supplies constructed images, not conversational memory.
- Follow the vendor guide for the lineage: OpenAI GPT-image (roles by number and
  purpose, "change only X" plus an explicit preserve list, exclusions allowed)
  or Google Gemini (positive framing only, "change only X, keep everything else
  exactly the same"). State the intended use (an annotation overlay on a
  photograph), that the yellow lines are the only new thing in the image, and
  that every tissue pixel and its texture is preserved.
- Save exact text and attachments. Inspect the result before declaring an
  improvement.

Provider adapters only translate requests; task semantics and prompt selection
stay in nonlinear.

## Known limits

One draw per model call. Yellow tissue can contaminate line extraction; the
border fit does not itself certify region identity or topology. Inverse renders are not computed on either route
(`inverse_warp_status="not_computed"`); do not invent them or silently report
identity. Route "atlas" cannot detect a mirrored section; the caller must pass
`atlas_mirror_lr`. Full details and output frame meanings are in the design doc.

# LangSlice `core/nonlinear/` — generative-image registration

Package guide for `src/langslice/core/nonlinear/` (the `nonlinear/` package
until the folder move, 2026-10-04); `AGENTS.md` is a verbatim twin of this
file. The image transport it calls, formerly `nonlinear/providers.py`, is
`src/langslice/providers/images.py`; `registration_tool.py`,
`registration_handoff.py` and `agent_trace.py`, formerly top-level, live
here; the `nonlinear register` command is the door `doors/cli/register.py`.
See `docs/nonlinear_design.md` for the design and coordinate contracts.

## Supported design

The stack agent's optional `nonlinear` task uses a supplied linear placement via
`core/nonlinear/registration_tool.py`. Its only image tool is
`trace_borders(id, prompt="")`: placed-border correction with the agent's
edited copy of the base prompt (blank = base), one retained reply per geometry,
raw and extracted images returned separately, no fit, atlas search or rejection.
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
  (`deformation="deformable"`) the fit of its lines (`border_fit.py`).
  Production; not under test.
- **Route "atlas"** — no placement supplied. Image 1 is the clean tissue,
  Image 2 the outlined grayscale atlas (`outlined_atlas_template`: grayscale
  plate plus yellow family borders, oriented by `image_axes` then the explicit
  `atlas_mirror_lr`). ONE model call with `prompts.pass1_atlas_prompt(plane,
  provider)`; with `passes=2` a second call with `prompts.pass2_atlas_prompt`
  sends Image 1 clean tissue, Image 2 pass 1's extracted lines redrawn on the
  tissue, Image 3 the outlined atlas. The rough placement for the fit is the
  local silhouette-moments fit (`prior.place_plane_on_tissue_with_matrix`); the
  model's lines then go through the SAME fit as route "supplied".
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

The fit of the model's corrected lines is the deformable package's (Nash
2026-10-01: `deformable/` replaces the old residual fit; done 2026-10-04):
`border_fit.fit_border_lines`, route B, the extracted line mask against the
colour-family borders the model was shown, Elastix B-spline lines-vs-borders
by default (`traced_lines`; `engine="ants"` adds the lines as named regions,
`traced_borders`), on a `deformable.Placement` of the atlas plane on the
model's canvas (`border_fit.placement_on_canvas`: the canvas pixel size from
the placement's scale; `border_registration.native_to_oriented_map` folds
`image_axes` and the mirror into the matrix, since the fit reads the native
plane). `deformation="deformable"` fits, `"none"` (the CLI default) fits
nothing and returns an identity residual, as does the model-free diagnostic;
the Elastix `"bspline"`/`"affine"` residual fit and its report are gone.
Elastix is the default because it ships with LangSlice (the result does not
change with the optional ANTs install) and on the 2026-10-04 before/after
check (a stubbed reply drawn at a known placement on a real coronal section,
the supplied placement shrunk 6 %, turned 3° and shifted) it followed the
reply's lines closest, outline included; the ANTs named-region mode left the
ventrolateral outline short there (its regions are named after a placement
that far off). The model call is still judged on its raw lines first.

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
  selection, rough placement, the atlas-route model calls, the fit (through
  `border_fit`), coordinate composition (`composed_correspondences`,
  `composed_native_map`, `marker_spacing_px`), `native_to_oriented_map` and
  the candidate contract. Candidate metadata: `deformation`, `fit` (the fit's
  settings, engine, diagnostics and flags, or `fit_skipped`),
  `fit_elapsed_s`; the debug folder adds `fit_report.json` and the saved
  `deformable/` record.
- `border_refinement.py` — `refine_borders`: the correction request and the
  yellow-line extraction (`extract_thinned_lines`, `yellow_mask`, `thin`),
  the rough and corrected overlays; it fits nothing. `hosts/integrations/abba.py`
  shares this core directly.
- `border_fit.py` — `fit_border_lines` (the deformable fit of the lines;
  `BorderFit`: the field in canvas pixels, fitted labels, the fitted borders
  drawn smoothly on the canvas, the record, metadata) and
  `placement_on_canvas`. The ABBA plugin passes its own label grid as
  `native=` (placed by an identity).
- `prompts.py` — the three prompt functions; the OpenAI-GPT or Gemini wording
  is selected by `canonical_provider(provider)`.
- `prior.py` — `place_plane_on_tissue_with_matrix`, the silhouette-moments
  placement.
- `providers/images.py` (in the providers package since the folder move) —
  `generate_warped_segmentation_image`: the one-call-in,
  one-image-out transport adapter every prompt call goes through, by the
  request's provider. `SegmentationGenerationRequest` lives in `types.py`
  (re-exported here) so callers build requests without the transport. `mode`
  (default `"edit"`) is the edit-vs-generate task semantic each transport
  translates its own way. A door resolves a provider name to this call once
  (`providers.registry.resolve_image_model`) and passes it in as
  `image_call` (`refine_borders`, `generate_registration_candidate`,
  `estimate_registration`); without one they fall back to this adapter.
- `model_prompts.py` — canvas/aspect facts only (`image_model_family`,
  `aspect_ratio_limits`, `gemini_aspect_for`, `native_output_size`).
- `image_gen_helpers.py` — label-map helpers: families (`_merge_classified`),
  crisp label borders, ventricle ids, line widths. No fit.
- `render.py` — review-grade rendering only.
- `runtime.py` — `estimate_registration`: orchestration, debug artifacts,
  trace events.
- `core/handoff.py` — prepares supplied linear geometry
  (`prepare_linear_registration`, re-exported by the top-level
  `core/nonlinear/registration_handoff.py`, which also holds `run_linear_registration`:
  route "supplied" for one section, the image model an argument).
  `core/nonlinear/registration_tool.py` uses it for the opt-in annotation tool, taking the
  image model as an argument too, and leaves fitting to `fit_deformable` and
  transformation export to the separate registration stage.

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

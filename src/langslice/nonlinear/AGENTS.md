# LangSlice `nonlinear/` — generative-image registration

Package guide; `AGENTS.md` is a verbatim twin of this file.
See `docs/nonlinear_design.md` for the current design and coordinate contracts.

## Supported design

Prefer an existing linear-agent or host alignment. Draw its atlas family borders
in yellow on the histology and send that image plus the identical clean histology
to the image model. Ask it to move the borders to match tissue, not make a colormap.
Fit the residual deformation and compose it with the complete initial placement.

Without supplied alignment, standalone registration uses TWO image-generation
calls: the existing color-map generation and registration, then correction of the
resulting placed borders. A silhouette-only initialization is NOT this default.
`provider="none"` is the explicit model-free diagnostic and makes no model calls.

- `image_gen_registration.generate_registration_candidate`: public dispatcher;
  borders default, explicit `registration_mode="colormap"` for initialization
  and historical experiments.
- `border_registration.py`: placement precedence, two-stage orchestration,
  coordinate composition, artifacts and candidate contract.
- `border_refinement.py`: two-image correction request, yellow-line extraction,
  residual fit and nearest-neighbor label warp. ABBA shares this core.
- `registration_handoff.py` (top-level): independent bridge from linear state.
  It does not install a new tool into the linear agent.
- `prior.py`: moments placement for the explicit model-free route.
- `model_prompts.py`: existing provider-specific color-map initialization prompts.
- `image_gen_helpers.py`: classification, Elastix and geometry helpers.

Atlas labels and grayscale references must use the same position, plane, cutting
angles and orientation. Reflection is explicit, never guessed from a symmetric
silhouette. Supplied matrices map the oriented native annotation grid to original
input-image pixel centers. Carry resize, padding, initial affine/nonlinear mapping
and residual deformation through exports; never export the residual alone.

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
  Remove ambiguity, redundancy and contradictions.
- Identify actual attachment order, content and role. Specify what is edited,
  preserved, and the output frame.
- Distinguish moving lines onto visible tissue from copying supplied lines.
  Calling rough lines “correct” can imply retaining them unchanged.
- Verify actual history before referring to an earlier reply. The standalone
  second request supplies constructed images, not implicit conversational memory.
- Prefer small targeted edits and positive descriptions. Save exact text and
  attachments. Inspect the result before declaring an improvement.

The correction prompt retains the accepted rough-plus-clean wording, with plane
parameterized and without a species-only assumption. Provider adapters only
translate requests; task semantics and prompt selection stay in nonlinear.

## Known limits

The default route supports one draw per stage. Yellow tissue can contaminate line
extraction; the border fit does not itself certify region identity or topology.
Dense first-stage coordinates use documented nearest-edge extension off canvas.
Inverse renders are not computed on this route; do not invent them or silently
report identity. Replay without supplied placement is rejected to avoid accidental
remote initialization. Full details and output frame meanings are in the design doc.

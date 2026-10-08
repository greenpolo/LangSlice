# LangSlice `core/nonlinear/` — generative-image border correction

Package guide for `src/langslice/core/nonlinear/`; `AGENTS.md` is a verbatim
twin of this file. The image transport it calls is
`src/langslice/providers/images.py`. Every way of using LangSlice goes through
the job folder and its verbs. See `docs/registration.md` for tool behavior and `docs/file_formats.md`
for coordinate contracts.

## Supported design

The stack agent's optional `nonlinear` task uses a supplied linear placement via
`registration_tool.py`. Its only image tool is `trace_borders(section, prompt="", restrict_to=())`:
placed-border correction with the agent's edited copy of the base prompt
(blank = base), one retained reply per geometry, raw and extracted images
returned separately, no atlas search or rejection. The fit of the extracted
lines is the traced ANTs fit (`ops.traces.land_trace`), in `core/deformation.py` and the
`core/deformable/` package. The prompt sentence review is below.

Exactly two border-based routes, chosen by whether a placement is supplied.
No model is ever shown a colored region map: the model-facing atlas is a
grayscale plate with thin yellow family borders.

- **Route "supplied"** — a written linear placement exists.
  `start_correction` draws the placed family borders on the tissue (Image 1
  the clean tissue and Image 2 the borders for a GPT-lineage prompt, the
  other way round for the Gemini wording; `_GPT_ATTACHMENTS` /
  `_GEMINI_ATTACHMENTS`) and makes ONE model call with
  `prompts.border_correction_tool_prompt(plane, provider)` or the agent's
  edit of it. A model profile's own prompt and attachment order
  (`ImageModel.prompt` / `photograph_first`, `providers/profiles.py`) replace
  the base prompt (`profile_prompt`) and join the call key; an untested
  profile marks its request and result `untested` (`profile_marks`).
- **Route "atlas"** — the job's HIDDEN scripting verb `trace_from_atlas`
  (`ops/traces.py`; the agent CLI and the library call it by name, no listing
  shows it), `start_atlas_correction`. Image 1 is the clean tissue, Image 2
  the outlined grayscale atlas (`outlined_atlas_template`: grayscale plate
  plus yellow family borders). ONE model call with
  `prompts.pass1_atlas_prompt(plane, provider)`; with `passes=2` a second call
  with `prompts.pass2_atlas_prompt` sends Image 1 clean tissue, Image 2 pass
  1's extracted lines redrawn on the tissue, Image 3 the outlined atlas. The
  calls are `draw_from_atlas`. The reply is recorded as `trace_borders`
  records its own (`trace_route: "atlas"`, `passes`, `model_calls`;
  `request.json`'s `atlas_to_canvas` is the section's written linear
  placement, never shown to the model). It needs a written transform.

Both routes keep the call-key folders and attempts, take the image model as
an argument, and leave fitting to the ops layer.

Atlas labels and grayscale references use the same position, plane, cutting
angles and orientation. Reflection is explicit, never guessed from a
symmetric silhouette. Supplied matrices map the oriented native annotation
grid to original input-image pixel centers. Resize, padding, the initial
affine and the residual deformation are all carried through exports; the
residual alone is never exported.

## File map

- `registration_tool.py` — `start_correction` (route "supplied"),
  `start_atlas_correction` and `draw_from_atlas` (route "atlas", `AtlasDrawing`),
  the call-key folders and attempts, `profile_prompt`, `profile_marks`,
  `prompt_diff`.
- `image_gen_registration.py` — model-facing canvas geometry (`prepare_canvas`,
  aspect helpers) and `outlined_atlas_template`.
- `border_refinement.py` — the yellow lines of a trace: `smooth_border_overlay`
  and `border_overlay` draw them on a section, `extract_thinned_lines`
  (`yellow_mask`, `thin`) reads the model's lines off a raw reply. It fits
  nothing and calls no model.
- `prompts.py` — the prompt functions; the OpenAI-GPT or Gemini wording is
  selected by `canonical_provider(provider)` (`core/provider_names.py`).
- `types.py` — `SegmentationGenerationRequest` / `GeneratedSegmentation`, the
  request and reply of the one-call-in, one-image-out transport
  (`providers/images.py`). `mode` (default `"edit"`) is the edit-vs-generate
  semantic each transport translates its own way; `size_tier` is the
  output tier the transport passes on. A door resolves a provider name to
  this call once (`providers.registry.resolve_image_model`) and passes it in
  as the `ImageModel`; a model call without one is refused, so the core never
  imports the transport.
- `image_frames.py` — canvas and aspect facts only (`image_model_family`,
  `aspect_ratio_limits`, `gemini_aspect_for`, `native_output_size`).
- `image_gen_helpers.py` — label-map helpers: families (`_merge_classified`),
  crisp label borders, line widths.

## Visual review is essential

Inspect raw model replies BEFORE resizing, extraction or registration. Next inspect
extracted boundaries on the original photograph, then fitted atlas borders on that
same photograph. Describe concrete anatomical successes, errors and uncertainty.
A model may redraw tissue; its redrawn photograph is not the evaluation reference.
A useful model correction can be damaged by the fit. Metrics and fit error codes
cannot establish anatomical quality; report disagreement with visual evidence.

Keep rough overlay, exact prompt, raw reply, original-photo correction overlay,
fitted overlay and labels separately. 

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

## Base prompt sentence review

`border_correction_tool_prompt` returns `border_refinement_prompt` for the
Gemini lineage (Image 1 the placed borders, Image 2 the clean photograph);
OpenAI providers get `_SUPPLIED_GPT` (the same sentences, attachments swapped).
The notes heading, "Additional notes for this slice (supplement the task
above)", is subordinate scope and adds no attachment, frame or partition. Notes
describe specimen-specific displacement, damage or slide artifacts; they are
never a replacement task or a visibility-based border-selection policy. The exact
notes and the effective prompt are always saved. Sentence by sentence, with the
interpretation each was checked against:

| Sentence | Purpose and alternative interpretation checked |
| --- | --- |
| Image 1 is the photograph with roughly aligned borders. | Establishes the edit target; "rough" avoids treating the placement as anatomically final. |
| Image 2 is the same photograph without lines. | Supplies unobscured tissue in the identical frame; it is not an independent atlas plate. |
| Edit Image 1 so each line follows its region edge in Image 2. | Requests corrected anatomical placement rather than tracing any visible artifact. |
| Move a line by sliding or bending it. | Describes changes to lines, not movements of tissue pixels. |
| Keep an already correct line. | Allows partial correction without asking the model to redraw every border. |
| Infer indistinct boundaries from neighboring structures and the supplied arrangement. | Indistinct surviving anatomy retains boundaries; visibility is not a permission to omit them. |
| Preserve Image 2's photograph, tissue, background, size, position and frame. | Defines an annotation overlay and excludes synthesized tissue as the intended output. |
| Keep the same boundary set and thin bright yellow style. | The supplied atlas arrangement decides region identity; artifacts do not introduce regions. |
| Output one photograph with corrected boundaries replacing the rough ones. | Requests one corrected annotation, with no labels, alternate views or accumulated rough lines. |

## Known limits

One draw per model call. Yellow tissue can contaminate line extraction; the
border fit does not itself certify region identity or topology. Route "atlas"
cannot detect a mirrored section. Full details and output frame meanings are
in the design doc.

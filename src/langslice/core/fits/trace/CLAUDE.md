# `core/fits/trace/`: the image-model border correction

The `trace_borders` tool draws the placed atlas borders on a section (thin
yellow, regions at `core.atlas.levels.MODEL_DEPTH`), asks an image model to move
the lines onto the tissue, extracts the lines from the reply and fits them with
FireANTs. No model is ever shown a colored region map. The prompts are
`prompts.py`; the image model is an argument, never imported here.

## Design rules

- One border route, from a written linear placement. One model call per
  placement and region choice, one reply kept per call key; the raw reply and
  the extracted lines are kept apart and neither is the evaluation reference.
- Reflection is explicit, never guessed from a symmetric silhouette. The
  canvas turn and crop are folded into one matrix, so the rough overlay is
  drawn through it and the model's lines are read back through it exactly.
- A running attempt holds a lock on its call key; another process asking the
  same call is refused `BUSY`, and an attempt whose process died releases it.
- The pictures sent are not kept; they are drawn again from the section and
  the request. Exact prompt text and attachment roles are always saved.
- Provider adapters only translate requests. Task semantics and the choice of
  wording (GPT-image lineage or Gemini lineage) stay here.

## Visual review is essential

Inspect the raw reply BEFORE resizing, extraction or registration; then the
extracted lines on the original photograph; then the fitted borders on that
photograph. Describe concrete anatomical successes, errors and uncertainty. A
model may redraw tissue; its redrawn photograph is not the reference. A useful
correction can be damaged by the fit. Metrics and fit codes cannot establish
anatomical quality: report any disagreement with what you see.

## Prompt rules: every sentence

Before changing or sending a prompt:

- State each sentence's intended effect and its plausible alternative
  readings. Remove ambiguity, redundancy and contradiction; no sentence is
  exempt.
- Name the real attachment order, content and role; say what is edited, what
  is preserved and the output frame.
- Never write a visibility-conditioned rule ("no tissue, no line", "border what
  you see"). A visible slide feature (bubble, stain, debris) must NOT get a
  line, and an indistinct region MUST still get its atlas line: the atlas image
  alone decides which boundaries exist. The only exclusion a prompt may state
  is tissue physically torn away or missing.
- Distinguish moving lines onto visible tissue from copying supplied lines;
  calling rough lines "correct" can imply keeping them unchanged.
- Check the real history before referring to an earlier reply: a second
  request supplies constructed images, not conversational memory.
- Follow the vendor guide for the lineage. OpenAI GPT-image: roles by number
  and purpose, "change only X" with an explicit preserve list, exclusions
  allowed. Gemini: positive framing only, "change only X, keep everything else
  exactly the same". State the intended use (an annotation overlay on a
  photograph), that the yellow lines are the only new thing, and that every
  tissue pixel and its texture is preserved.
- Per-slice notes ("Additional notes for this slice (supplement the task
  above)") are subordinate: specimen-specific displacement, damage or slide
  artifacts. They never replace the task or select borders by visibility, and
  add no attachment, frame or partition.
- Inspect the result before declaring an improvement.

The base prompt asks for exactly this: edit Image 1 (the photograph with rough
borders) so each line follows its region edge in Image 2 (the same photograph
without lines); slide or bend a line, keep a correct one, infer indistinct
boundaries from neighbouring structures; preserve the photograph, tissue,
background, size, position and frame; keep the same boundary set and the thin
bright yellow style; output one photograph with the corrected lines replacing
the rough ones. Each sentence was checked against the readings above; change
none without repeating that check.

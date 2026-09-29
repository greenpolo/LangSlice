# Image-model correction tool after linear registration

The stack agent can correct atlas borders after linear alignment by enabling the
optional `nonlinear` task. The tool is `correct_slice_borders(id,
additional_notes="")`. Its two agent-controlled fields are the slice identifier
and a supplementary text field for that slice. There is no atlas search, region
selection, replacement prompt, candidate selection or anatomical rejection tool.

The existing supplied-placement prompt owns the task. Image 1 is the original
rendered photograph with the linearly placed atlas borders. Image 2 is the same
clean photograph in the same frame. Optional notes are appended to that fixed
prompt. The image model makes one correction; the first reply is retained and
returned along with its extracted yellow borders on the unchanged photograph.

This stage produces border-annotation images. It does not fit a deformation,
modify the linear transform, or export a completed nonlinear registration.
Deformation fitting remains a separate stage.

## Running it

To add correction after positioning and linear alignment in one agent session:

```bash
langslice linear run sections/ --tasks reorder,position,transform,nonlinear
```

To use positions and transforms from the folder's existing linear checkpoint:

```bash
langslice linear run sections/ --tasks nonlinear
```

Hosts can instead provide `inputs.positions`, `inputs.transforms`, cutting angles
and calibration through `JobSpec`. The CLI accepts `--positions` and `--transforms`
as JSON files or inline mappings keyed by filename. Transform records use the
existing linear checkpoint format, including calibration. A historical spline is
refused rather than silently reduced to its stored affine baseline. The new task
is opt-in: the default remains reorder, position and transform. Image transport
defaults to `openai-oauth`; `--image-provider` and `--image-model` select the host's
image settings independently of the agent's `--model`.

The notes field is an agent tool argument. A host may also pass user notes for
this task as `JobSpec.nonlinear.notes`; they appear under the task in the job
statement. The ABBA Registration dialog shows the Nonlinear task but keeps it
disabled ("Not yet available in ABBA") until the deformation stage exists, and
the connector's older `nonlinear.abba` worker method is not called by the dialog.

## Retention and state

`SliceState.image_correction` records the first result and artifact paths separately
from `transform`. The artifacts live in `nonlinear/` beside the results JSON and
contain exact image attachments, their hashes, full prompt, additional notes,
placement provenance, raw reply, extracted lines and lines on the original.

Calls at the same source image, geometry and image settings return the first saved
image reply, even if the notes change or an undo removes its checkpoint reference.
Changing the linear placement requires a new correction; the previous artifacts
remain. There is no automatic regeneration after a model reply. Transport failures
are recorded without hidden retries; a later tool call may retry a failed transport
that returned no image. Each attempt's artifacts remain separate. An interrupted
request with an unknown outcome is reported explicitly.

Submit checks that every section has a completed correction for its current
placement. It checks completion and geometry, not anatomical quality. An empty or
unhelpful drawing is still retained, shown and counted as a completed model call.
Raw output and extracted lines remain separate because the model can redraw tissue.

## Prompt sentence review

`border_correction_tool_prompt` leaves the accepted `border_refinement_prompt`
byte-identical when notes are blank. The existing nine sentences are reviewed below
in order; attachment roles match the actual request described above.

| Sentence | Purpose and alternative interpretation checked |
| --- | --- |
| Image 1 is the photograph with roughly aligned borders. | Establishes the edit target; “rough” avoids treating the placement as anatomically final. |
| Image 2 is the same photograph without lines. | Supplies unobscured tissue in the identical frame; it is not an independent atlas plate. |
| Edit Image 1 so each line follows its region edge in Image 2. | Requests corrected anatomical placement rather than tracing any visible artifact. |
| Move a line by sliding or bending it. | Describes changes to lines, not movements of tissue pixels. |
| Keep an already correct line. | Allows partial correction without asking the model to redraw every border unnecessarily. |
| Infer indistinct boundaries from neighboring structures and the supplied arrangement. | Indistinct surviving anatomy retains boundaries; visibility is not a permission to omit them. |
| Preserve Image 2's photograph, tissue, background, size, position and frame. | Defines an annotation overlay and excludes synthesized tissue as the intended output. |
| Keep the same boundary set and thin bright yellow style. | The supplied atlas arrangement decides region identity; artifacts do not introduce regions. |
| Output one photograph with corrected boundaries replacing the rough ones. | Requests one corrected annotation, with no labels, alternate views or accumulated rough lines. |

The optional heading, “Additional notes for this slice (supplement the task
above),” establishes subordinate scope and introduces no different attachment,
frame or anatomical partition. The agent checks each sentence of its notes for
conflicts before calling the tool. Notes describe specimen-specific displacement,
damage or slide artifacts; they are not a replacement task or a visibility-based
border-selection policy. Exact notes and effective prompt are always saved.

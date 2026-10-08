# Nonlinear: the agent's deformation toolbox

The `nonlinear` task gives each linearly placed section a deformation. It
needs a linear placement first, made by the agent or supplied with the job
(`--positions`, `--transforms`, `--registration`).

The agent works the way an expert drives a registration library. It looks at
the section, decides which regions can steer a fit and which are torn or
missing (`mark_damage`), sets the preprocessed channel so those regions stand
out, runs ANTs with the settings it chooses, inspects the picture and fits
again on top where needed. The image model is one optional tool in that box:
the agent decides which sections, if any, it traces. A run ends with every
section carrying an applied deformation, or named in `submit`'s `left_linear`
with the reason its linear placement stands (unless the host requires a
deformation on every section).

## The tools

The task adds `ants_syn` and, when an image model is connected,
`trace_borders`. The shared tools do the rest: `set_preprocessed_channel_properties`
sets the channel the fits and the image model read, `grep_atlas` and
`grep_atlas_view` choose regions, `look` mode `overlay` shows the section
under its registration, `mark_damage` records lost regions
([linear_design.md](linear_design.md)).

| Tool | What it does |
|---|---|
| `ants_syn` | fits an ANTs SyN deformation of the atlas onto one to four sections, on their current registration (below) |
| `trace_borders` | optional, background: an image model draws the atlas borders onto the tissue, then ANTs fits what it drew (below) |

## `ants_syn`

`core/deformation.py` calls the engine in `src/langslice/core/deformable/`:
ANTs SyN (antspyx, a LangSlice dependency below Python 3.14). It bends the
atlas onto each section's preprocessed channel, starting from the section's
current registration: its linear placement, and its deformation when it has
one, so each fit builds on the one before and undo goes back. Fit the whole
section first, then refine regions. It needs a position and a transform and is
one undoable write. Arguments (`langslice-job schema ants_syn`):

| Argument | Choice |
|---|---|
| `sections` | one to four filenames |
| `restrict_to` | regions (acronyms, names or ids, descendants included, optionally one side: `"CTX:left"`); only the atlas within 300 um of them is fitted. Empty fits by every region |
| `atlas_image` | `template` (the atlas's reference template, default) or `nissl` (a Nissl template aligned to the CCFv3, Allen mouse atlases only, below) |
| `stiffness` | `soft`, `medium` (default) or `firm` |
| `view` | false returns no picture |

The section's marked damage regions are left out of the atlas side
automatically. The reply gives, per section, the displacement (median and
max, mm), the fold fraction and plausibility flags (regions compressed,
expanded, vanished or folded beyond limits; displacement outsized for the
section); the picture shows the section under its new registration; with
`restrict_to`, those regions' borders are drawn thick and the others faint,
zoomed to their box when that magnifies at least 1.5 times.

Masks cover the tissue (widened past its outline) minus a band along torn
edges, and the atlas footprint minus excluded regions. The record stores a
displacement field from section to placed atlas in millimetres, the warped
labels clipped to tissue, and diagnostics. The flags report; they never
enforce. Fits are deterministic (fixed seed and thread count). Records are
kept in the job folder (`sections/<name>/deformable/`). Any later change to a
section's position, orientation, cutting angles or linear transform clears
its deformation (the write's reply lists the sections). Engine details:
`src/langslice/core/deformable/CLAUDE.md`.

`nissl` is the Blue Brain population-averaged Nissl template, aligned to the
CCFv3 (Piluso et al., *Imaging Neuroscience* 2025,
[doi:10.1162/imag_a_00565](https://doi.org/10.1162/imag_a_00565)), read from
BrainGlobe's `ccfv3augmented_mouse_25um` (downloaded on first use) and sampled
on any Allen mouse (CCFv3) atlas: that atlas is the CCFv3 grid extended 0.35 mm
at the front (`core/deformable/nissl.py`). The Allen Institute's own Nissl
volume, the one ABBA displays, is misaligned with the CCFv3 annotation and is
not used.

## `trace_borders` (optional)

`trace_borders(section, prompt="", restrict_to=[])`: an image model corrects
the placed atlas borders onto one section's tissue, and ANTs then fits what it
drew. Two images accompany the prompt: the clean photograph (the section's
preprocessed channel), and the same photograph in the same frame with the
linearly placed atlas borders drawn in thin yellow. A preprocessed-channel
recipe other than the default is part of the call key, so a changed recipe
makes a new call. OpenAI providers get the clean photograph first; every other
provider the bordered one first. The model slides or bends each line onto the
tissue edge it belongs to and returns one photograph in the same frame. No
image model is shown a colored atlas region map.

- The section needs a position and a linear transform. `restrict_to` (region
  entries as in `ants_syn`, sides allowed) chooses the regions whose borders
  the model is shown and that are fitted: only those (all when empty). The
  section's marked damage regions are left out of what the model is shown
  automatically; excluded regions join the background, so their edge with the
  shown regions becomes an outline (`registration_tool.shown_labels`).
- The agent may edit the base prompt for the section (`prompt`; blank sends it
  unchanged). The base prompt, the prompt sent and their word diff are saved
  with each attempt; user notes for the task (`JobSpec.nonlinear.notes`)
  appear in the job statement.
- The call runs in the background and returns at once with an id; up to 8 run
  at a time. When it finishes, ANTs fits the traced borders against the atlas
  borders (medium stiffness, on top of the section's current registration,
  by the traced regions only) and applies the result as its own undo step; the
  next tool reply starts with a notice giving the fit's numbers, and its
  pictures follow the reply's own, each with its number and caption (the
  fitted borders, the `restrict_to` regions thick; the trace on the section).
  `status`
  lists work still running and `submit` waits for it. A section whose
  placement changed meanwhile, or whose trace was undone, gets no fit and the
  notice says `STALE_INPUT`.
- The first reply at a placement is kept: calls with the same image,
  geometry, regions and settings return the saved reply, and a fit already
  applied is not repeated; changing the linear placement needs a new trace and
  the old artifacts stay. No automatic retries.
- The yellow lines are extracted from the raw reply (HSV threshold, crop to the
  canvas aspect, thinning to one pixel) and drawn on the untouched photograph,
  never on the model's redrawn tissue. The raw reply, the extracted lines and
  the lines on the photograph stay separate files
  (`sections/<name>/image_correction/<call key>/attempt-NN/`, with the exact
  attachments, their hashes, the placement and the prompts);
  `SliceState.image_correction` points to them.
- The image model is the job's (`--image-provider`, `--image-model`; default
  `openai-oauth`; `none` offers no `trace_borders`, and a door that cannot
  reach the model leaves it out). A model profile (own prompt, attachment
  order, or a model of your own) is described in [library.md](library.md); its
  traces are marked `"untested": true`.

Prompt text is in `src/langslice/core/nonlinear/prompts.py`. Rules a prompt
edit keeps: every sentence is audited for a second reading; no line is made
conditional on visibility (the atlas alone decides which borders exist; the
one exception is tissue torn away); the OpenAI wording uses "change only X"
plus a preserve list, the Gemini wording positive framing only.

## What the deformation becomes

- The job folder's maps hold the complete mapping, linear plus residual
  (`coords.tif`, `labels.tif`, `residual.tif`), and `exports/visualign.json`
  holds VisuAlign markers on a regular grid of the residual
  ([file_formats.md](file_formats.md)).
- In ABBA, an applied deformation arrives as a BigWarp warp step on top of the
  LangSlice affine step: landmark pairs in ABBA's centred world frame, a
  thin-plate spline on a grid from 9x9 to 33x33 that stops growing once it is
  within 5 micrometres of the deformation and is otherwise sent at 33x33 with
  its measured error; a spline that folds at every grid is refused
  (`core/abba_warp.py`, [abba_plugin_design.md](abba_plugin_design.md)).

## Placement contract

`langslice.core.handoff.prepare_linear_registration` turns a section's linear
state into the calibrated placement the trace and the fit take: an invertible
3x3 affine from the oriented native atlas grid's pixel centres to the
photograph's pixels, keeping shear, calibration and orientation. Mirroring is
explicit (the job's `flip`, or a mirrored transform); it is never inferred from
a nearly symmetric silhouette, and one flag is applied identically to every
atlas render.

## Review

Inspect the section and the fitted atlas overlay against the section's
internal anatomy; for a traced section, inspect the raw reply first, then the
extracted lines on the original tissue, then the fitted overlay (a good reply
can be degraded by the fit). Metrics and fit flags complement the visual
check but do not prove anatomical correctness. Known limits: the yellow-line
extractor can confuse naturally yellow tissue with drawn lines; the fit is not
region-identity-aware and does not certify topology.

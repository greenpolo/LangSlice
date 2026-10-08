# Nonlinear: the agent's deformation toolbox

The `nonlinear` task gives each linearly placed section a deformation. It
needs a linear placement first, made by the agent or supplied with the job
(`--positions`, `--transforms`, `--registration`).

The agent works the way an expert drives a registration library. It looks at
the section, decides which regions can steer a fit and which are torn or
missing, adjusts the section image so those regions stand out, runs ANTs or
Elastix with the settings it chooses, previews candidates and applies the
best. The image model is one optional tool in that box: the agent decides
which sections, if any, it traces. A run ends with every section carrying an
applied deformation, or a `keep_linear` reason saying its linear placement
stands.

## The tools

| Tool | What it does |
|---|---|
| `preprocess` | sets a recipe (channel weights, CLAHE, N4 bias correction, denoising), stack-wide or per section, for what the agent views (`target` `view`) or for the section's preprocessed channel (`fit`), or both; the preprocessed channel is what `fit_affine`'s Elastix method, `fit_deformable` and the image model read, and without a recipe it is the default rendering; returns before and after pictures |
| `grep_atlas` | looks regions up in the atlas hierarchy by acronym, name or id: ancestry, descendants, and whether the region appears in the atlas plane at a section's placement |
| `view_placement` | shows a section in its current registration; the shared picture options highlight chosen `regions` and draw atlas images (`template`, `nissl`, `borders`) under the section |
| `fit_deformable` | fits a deformation of the placed atlas onto sections (below) |
| `trace_borders` | optional: an image model draws the atlas borders onto the tissue, for a fit to read (below) |

## `fit_deformable`

`core/deformation.py` calls the engine in `src/langslice/core/deformable/`:
ANTs SyN (antspyx, a LangSlice dependency below Python 3.14) or an Elastix
B-spline with a bending penalty. It is the only deformation step; no deformation is fitted unless it
is called. Main arguments (`langslice-job schema fit_deformable`):

| Argument | Choice |
|---|---|
| `fit_section` | `fit`: the section's preprocessed channel (stain); `traced_borders`: a trace's lines turned into named regions (ANTs); `traced_lines`: a trace's lines against the atlas borders |
| `fit_atlas` | for `fit`: `template` (the atlas's reference template, default) or `nissl` (a Nissl template aligned to the CCFv3, Allen mouse atlases, below); for a traced fit section: `borders` |
| `include`, `exclude` | regions (acronyms or ids, descendants included, optionally one side: `"CTX:left"`); only included regions plus a 300 µm margin are fitted; excluded regions (for example tissue missing from the section) are removed from the atlas side |
| `start` | `linear`, or `current` to compose onto the applied deformation (region-by-region steps) |
| `stiffness`, `engine` | `soft`, `medium`, `firm`; `ants` or `elastix` |
| `candidates` | 2 to 4 variants preview and write nothing; one setting applies it (undoable; identical fits are reused) |
| `keep_linear` | a reason recording that the section's linear placement stands |

The engine is the user's choice (`nonlinear.engine`: `ants` or `elastix`) or,
left at `either`, the agent's per call.

Masks cover the tissue (widened past its outline) minus a band along torn
edges, and the atlas footprint minus excluded regions. The record stores a
displacement field from section to placed atlas in millimetres, the warped
labels clipped to tissue, and diagnostics: per-region area ratios, folds,
displacement, and flags such as `DISPLACEMENT_OUTSIZED`. The flags report; they
never enforce. Fits are deterministic (fixed seed and thread count). Records
are kept in the job folder (`sections/<name>/deformable/`). Any later change
to a section's linear placement clears its deformation. Engine details:
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

`trace_borders(id, prompt="")`: an image model corrects the placed atlas
borders onto one section's tissue. Two images accompany the prompt: the clean
photograph (the section's preprocessed channel), and the same photograph in
the same frame with the linearly placed atlas borders drawn in thin yellow. A
preprocessed-channel recipe other than the default is part of the call key,
so a changed recipe makes a new call. OpenAI providers get the clean
photograph first; every other provider the bordered one first. The model
slides or bends each line onto the tissue edge it belongs to and returns one
photograph in the same frame. No image model is shown a colored atlas region
map.

- `include` and `exclude` (region entries as in `fit_deformable`, sides
  allowed) choose the regions whose borders the model is shown: only the
  included regions (all when empty), without the excluded ones, which join
  the background so their edge with the shown regions becomes an outline
  (`registration_tool.shown_labels`). A traced `fit_deformable` adds the
  trace's regions to its own (its exclude plus the trace's; its include,
  else the trace's), so the atlas side matches what the model was shown,
  and reports them as `trace_regions`.
- Damage also reaches the model through the prompt: the OpenAI base prompt
  tells it to leave out borders over missing tissue (the general wording has
  no such sentence), and the agent may edit the base prompt for the section
  (blank sends it unchanged). The base prompt, the prompt sent and their
  word diff are saved with each attempt; user notes for the task
  (`JobSpec.nonlinear.notes`) appear in the job statement.
- The call runs in the background and returns at once; up to 8 run at a time.
  `submit` and a traced `fit_deformable` wait for running calls.
- The first reply at a placement is kept: calls with the same image,
  geometry, regions and settings return the saved reply; changing the linear
  placement needs a new trace and the old artifacts stay. No automatic retries; a transport
  failure that returned no image may be retried by a later call.
- The yellow lines are extracted from the raw reply (HSV threshold, crop to the
  canvas aspect, thinning to one pixel) and drawn on the untouched photograph,
  never on the model's redrawn tissue. The raw reply, the extracted lines and
  the lines on the photograph stay separate files
  (`sections/<name>/image_correction/<call key>/attempt-NN/`, with the exact
  attachments, their hashes, the placement and the prompts);
  `SliceState.image_correction` points to them.
- It fits no deformation and changes no transform. A fit reads the trace with
  `fit_section` `traced_borders` (ANTs at medium stiffness is the recommended
  pairing) or `traced_lines`.
- The image model is the job's (`--image-provider`, `--image-model`; default
  `openai-oauth`; `none` offers no `trace_borders`). A model profile (own
  prompt, attachment order, or a model of your own) is described in
  [library.md](library.md); its traces are marked `"untested": true`.

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

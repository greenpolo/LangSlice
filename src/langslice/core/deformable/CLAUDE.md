# LangSlice `core/deformable/` — library deformable fit of a placed atlas plane

Package guide for `src/langslice/core/deformable/`; `AGENTS.md` is a verbatim
twin of this file. A core sub-package, like `core/affine.py`: it belongs to
neither the linear method nor `core/nonlinear/` and imports neither. It is the
engine behind the deformable fits (`fireants_syn` and the packaged
`trace_borders`' fit, through `core/deformation.py`, task `nonlinear`) and
behind `elastix_affine` (`core/transform.elastix_affine`: `prepare_fit`
builds its images and masks, `engines.run_elastix_affine` fits). The job folder's maps and VisuAlign
markers read an applied record (`core/maps.py`); the ABBA connector receives
it as landmark pairs (`core/abba_warp.py`). `engines.py` is LangSlice's one
itk-elastix registration wrapper; `engines.preload()` imports torch, FireANTs,
SimpleITK and the ITK Elastix classes without running a fit or touching the GPU
(the agent CLI's warm process forks after it). On Windows, `import_fireants` gives
FireANTs' import a stand-in `fcntl` (it imports the Unix-only module for a
multi-GPU file lock LangSlice never takes).

## What it does

Given a section image, a linear placement (`geometry.Placement`: atlas name,
position, plane, pitch/yaw, the 3x3 `atlas_to_section` map from native
atlas-plane pixel centres to section pixel centres, section mm/px) and one
`settings.FitSettings`, it fits a residual deformation with a LIBRARY engine
and returns a `record.DeformableRecord`. No custom solver.

- Route A (no image model): the stain image (`section_image="stain"`) against
  an atlas image — one of the atlas's own images (`core.atlas.images`:
  BrainGlobe's `template` and any additional references the atlas ships,
  by their BrainGlobe names; temporarily, an Allen CCFv3 atlas also gets
  `ccfv3augmented_nissl`, the Blue Brain averaged Nissl from
  `ccfv3augmented_mouse_25um` at its 0.35 mm AP offset). Metric
  (`stain_metric`, default `mutual_information`, FireANTs' global mutual
  information over the in-mask pixels; `engines._channel_loss`). The
  alternative, `local_correlation`, is neighbourhood cross-correlation over a
  box window of radius `CORRELATION_RADIUS_UM` (80 µm, rounded to working
  pixels: 4 at standard, 2 at coarse), each window counting only its in-mask
  pixels (as ITK's); the Elastix affine has no local correlation and uses
  `ELASTIX_STAIN_METRIC` (mutual information). Plus, by
  default (`stain_edges`), an EDGE channel: the gradient
  magnitude after a `EDGE_SIGMA_UM` (30 µm) Gaussian of the stain and of the
  atlas image (`fit.edge_image`, scaled by its 99th percentile in each mask),
  a second metric of the same kind at `EDGE_CHANNEL_WEIGHT` 1.0. The
  magnitude ignores which side of an edge is brighter, so fluorescent and
  brightfield stains and the template share pial surface, ventricle walls
  and layer boundaries. Excluded regions are inpainted from their
  surroundings before the atlas edges are taken (`fit._fill_from_surroundings`):
  the blanked region's rim otherwise pulled tissue in.
  Optionally SEQUENTIAL: each call with `structures=(...)` (acronyms/ids,
  descendants included, optionally one-sided) restricts both masks to those
  structures plus
  `neighbourhood_um`, and with `previous=record` composes onto that record;
  `record.parent` is the step before. The step's own field (and its inverse,
  re-inverted: `fit._invert_step`) is tapered to none over
  `fit.RESTRICTED_TAPER_MM` beyond the neighbourhood (`PreparedFit.step_weight`;
  `step_details.taper_um`), since the masks bound the metric, not the field.
- Route B (image model): the model's extracted line mask (`lines=`, on the
  section grid, e.g. `trace_borders`' `extracted_lines.png` with its
  `atlas_to_canvas`) against `borders_merged` (the regions the model was shown:
  `core.atlas.levels.labels_at` at `MODEL_DEPTH`) or `borders`; both softened
  by the same Gaussian ridge (`settings.LINE_SOFTENING_UM`, 60 µm in the
  adult mouse, times the brain scale).
  Mean squares (`core/deformation.Choice` adds the lines as named regions,
  `labels="model"`, for `traced_borders`). The crossed pairings are refused
  (`metric_for`): lines
  against an atlas image, and the stain against borders (the pairing that
  stays at the linear placement).
- Label-map channels (`labels=`, SyN only): `model` names each area the
  model's lines enclose after the placed merged region it overlaps most
  (`regions.named_regions`, small loops around bubbles dropped, joint
  one-to-one renaming) and pairs each region's indicator with the atlas's:
  one mean-squares term per label pair (the construction of
  `ants.label_image_registration`), beside the image pair's and the edge
  channel's terms in one weighted loss (`engines._channel_loss`).
  The areas are named inside the model's outline (`regions.model_extent`:
  what the outermost line encloses, gaps up to ~0.2 mm closed), not inside
  the placed footprint; outside the outline is not section, and an `outline`
  channel pairs the enclosed area with the atlas's foreground. An outline
  enclosing less than half or more than twice the placed atlas area is open
  or broken: the fit falls back to the footprint (`step_details.model_outline`).
- The deformable engine (`engines.run_engine`): FireANTs' `SyNRegistration`
  (torch, a core dependency, imported lazily with an install hint; on the
  GPU when torch sees one, `engines.fit_device`) on the working-grid images,
  the masks riding as each image's last channel (FireANTs' masked mode,
  resized but never blurred by the pyramid, warped with their image, no
  gradient through them). Settings: shrink factors
  2^(n-1)..1 for the detail's iteration list (`[8, 4, 2, 1]` at standard),
  `smooth_grad_sigma` = update sigma / voxel, `smooth_warp_sigma` = total
  sigma / voxel, the edge channel an extra channel. Optimizer: FireANTs' SGD
  with ANTs' step rule (`engines._global_step_sgd`: the gradient scaled so
  its largest vector moves `FIREANTS_LEARNING_RATE` (0.8) half-pixels of the
  level; Adam's per-pixel scaling moved excluded regions as far as any).
  A fit of traced lines (`section_image="lines"`) is run at ANTs SyN's
  settings: half the step (`TRACED_LEARNING_RATE`,
  0.4 = ANTs' 0.2 voxel), the whole warp smoothed `TRACED_TOTAL_SIGMA_FACTOR`
  (2x) wider than its smoothness says, and ITK's multi-metric rule
  (`_channel_loss(balanced=True)`: each channel's gradient scaled to unit
  length before its weight, the length estimated from the warped images at
  each step). At the stain fit's settings it folded and labelled worse than
  ANTs on Carey (seven brains, saved image-model replies).
  Returns millimetre fields both ways through the SyN midpoint: section ->
  atlas = fwd o rev^-1, atlas -> section = rev o fwd^-1, each half inverted by
  a damped fixed point (`engines._invert`, 200 sampling iterations; FireANTs'
  own inversion is an optimization that is not reproducible on the GPU).
  Any other `FitSettings.engine` is refused there: `engine="fireants"` names the
  SyN engine,
  `engine="elastix"` exists for the Elastix affine's inputs (its metric, no
  label channels), never a deformation.
- DETERMINISM: identical inputs give identical fields, bit for bit (tests
  `test_identical_inputs_give_identical_fields_in_and_across_processes`,
  `test_masks_are_deterministic`). FireANTs runs inside
  `engines._deterministic`: torch's deterministic algorithms (warn-only: its
  grid sampler's backward is flagged for the image's gradient, which no fit
  takes), cuDNN off (its grid sampler's backward is not deterministic), a
  fixed seed `engines.RANDOM_SEED`, and on the CPU `FIT_THREADS` (8) torch
  threads; the process's own settings come back afterwards.
  `CUBLAS_WORKSPACE_CONFIG` is set before torch's first use when nothing set
  it. The Elastix affine's `RandomSeed` is the same seed; Elastix's own
  `SetNumberOfThreads` segfaults in itk-elastix 0.25.4, so the call sets
  ITK's global default (`FIT_THREADS`) for its duration instead. A fit takes
  about 0.7 s per run on an RTX 5090.
- `engines.run_elastix_affine` (the linear affine, not a deformation): an
  Elastix `AffineTransform` from the identity on the same prepared inputs
  (intensity pair, edge pair, masks; `AutomaticTransformInitialization` off,
  so it refines the placement the moving image was drawn at), mutual
  information, `ELASTIX_AFFINE_RESOLUTIONS` 3 x `ELASTIX_AFFINE_ITERATIONS`
  500, the same seed and thread count; the edge channel a second image pair
  (`AddFixedImage`/`AddMovingImage`, masks per pair). Returns
  `AffineResult.matrix_mm` (section mm -> placed-atlas mm, the field's
  direction); a non-finite or singular matrix raises. `_elastix_register` is
  the one Elastix call (images, masks, extra pairs, the global thread count).
- `fit.run_fit(inputs, settings)` is a prepared fit's engine work on plain
  arrays: one run. No fit sorts the section by itself: `prepare_fit(...,
  section_weights=)` takes a section mask's weights (float, the section grid;
  `core/section_masks.py`), thresholds them at `EM_MATCH_THRESHOLD` (0.5) and
  ANDs them into the fixed mask. `run_prepared` runs one in this process.
  `with_section_mask(prepared, mask)` narrows a prepared fit's fixed mask to
  a caller's own (bool on the section grid; a script's `fit_fireants(mask=)`,
  the Elastix affine's `mask=`).
  `fit_prepared` runs the prepared fits of a call (different sections or settings):
  on a GPU one after another in the caller, on the CPU in a spawn process
  pool (`cpu_count // FIT_THREADS` workers at most, one per fit); a failing
  candidate is a `CandidateFailure`, never an exception. A GPU fit runs in the
  machine's one GPU queue (`engines.gpu_turn`: a file lock on `GPU_LOCK`,
  `LANGSLICE_GPU_LOCK` or `~/.cache/langslice-stalign/gpu.lock`, the lock the
  STalign, emlddmm and SPOmiAlign engines take too; it yields a `GpuTurn`
  whose `waited_s` is how long the lock was waited for), so fits of several jobs
  never share the GPU's memory; one that still finds it full (a process outside
  the queue) waits and fits again, `OOM_RETRIES` times. The `EngineResult`
  carries both (`gpu_wait_s`, `oom_retries`), and `finish_fit` puts them in
  the record's `engine` when not zero (a fit's row reads them,
  `ops.deformable.gpu_rows`).

## Brain size

Every distance in this package (in mm or µm: working pixels, the smoothness
sigmas, edge sigma, correlation radius, line softening, mask and exclusion
margins, `TRACE_MOVING_MARGIN_MM`, the restricted taper, the regions module's
gap closing and smallest loop, the EM sorting's block and background distance, the
record's `OUTSIZED_MEDIAN_MM` and `MIN_FLAG_AREA_MM2`) is the adult mouse
brain's, and a fit multiplies it by the atlas's brain scale
(`core.atlas.size.brain_scale`: the cube root of the atlas's annotated volume
over the adult Allen mouse's, rounded to 0.05; exactly 1.0 for every atlas of
the adult CCFv3 brain, about 1.7 for `whs_sd_rat_39um`, 0.7 for DeMBA P4),
areas by its square. `prepare_fit` / `prepare_warp` read it once
(`PreparedFit.brain_scale`) and hand it on: `EngineInputs.brain_scale` (the
smoothness sigmas, recorded in the engine's native parameters with
`brain_scale`), the EM sorting's scale (`masks.block_px`, `masks.em_parameters`),
`footprint_masks(..., scale)`, `regions.model_extent` / `named_regions`
(`brain_scale=`), `record.diagnose(..., brain_scale=)`. The working grid is
`DETAIL[...].working_um` times the scale, so a rat section is fitted as a
mouse section drawn smaller: the same working pixels across the brain, the
same pixel quantities everywhere. Kept in micrometres: the EM's grid
(`EM_GRID_UM`, the sampling of the slide's holes and debris). An explicit
`FitSettings.neighbourhood_um` is taken as given; the tools pass
`settings.neighbourhood_um_for(scale)`. `tests/test_brain_scale.py` pins
that a brain of scale 1 prepares exactly the inputs it did before scaling,
and the tests' fake atlases are all scale 1 (`tests/synthetic_stack.py`
`PATCHES`).

## Settings (`settings.py`)

Physical units: `smoothness_um` (the update Gaussian sigma of SyN in µm, adult
mouse, times the brain scale; the whole warp's sigma is
`TOTAL_SIGMA_FRACTION` = 0.25 of it; default 80, bounds 5-2000; the former
`stiffness` is a `ValueError`, and a saved record's `stiffness` reads as
soft 50 / medium 80 / firm 120); `detail` coarse/standard/fine (working µm and
pyramid depth/iterations — the extra levels are SyN's capture range; `fine`
is 10 µm with five levels, 4-5x slower than `standard`);
`atlas_image` (default `nissl`, resolved against the atlas where the fit
samples it: `atlas.images.default_image`), `section_image`, `exclude`,
`labels`, `structures`, `neighbourhood_um`, `preprocess`, and for stain fits
`stain_metric` (`mutual_information` | `local_correlation`) and
`stain_edges` (lines ignore both). `FitSettings.metric` resolves the metric
from pairing, engine and stain metric (`metric_for`); lines against
an atlas image and the stain against `borders`/`borders_merged` are refused.
The step size is fixed.

Stain defaults, and why: with the EM section mask (now made by `make_mask`),
mutual information
recovered more of the human raters' own corrections than local correlation
on Carey 2026 brightfield and LSD_910 immunofluorescence, did less harm where
the raters moved little, and was within noise of it on pl_projections when
the fit read the DAPI counterstain (where indicator channels were blended in, it was pulled
toward their bright labelling). The edge channel is on by default because
the intensity metric on its own crossed enlarged ventricles. There is no `stiff`
setting (it left enlarged ventricles unfilled); `fine` detail was measured
4-5x slower with no visible gain and exists for a caller who asks for it; line
softening is the constant `LINE_SOFTENING_UM` = 60 (adult mouse). The default
stain atlas image is the atlas's Nissl where it has one (`nissl`), else its
template; the agent picks whichever of the atlas's images looks most
like its sections (on `ccfv3augmented_mouse_25um` the template is the Blue
Brain averaged Nissl and `single_animal_nissl` is an additional reference).
On synthetic sections (flat regions, no texture) local correlation recovers a
known warp worse than mutual information, so the mechanism tests pin mutual
information.

**One side of a region.** Any `exclude` or `structures` entry may name one
side, `"CTX:left"` / `"CTX:right"` (`core.atlas.sides`): left and right of the
SECTION as displayed (the fit grid: oriented render, rotation and flip
applied), carried to the native plane through the placement's
`atlas_to_section` (`plane_images.placement_left`). Exclusion is therefore a
native-plane MASK (`plane_images.regions_mask`), not an id set:
`PreparedFit.excluded_mask` blanks the moving image, `excluded` (and the
record's `excluded_ids`) holds only whole-region ids (`whole_region_ids`),
and area diagnostics drop excluded pixels, so a region excluded on one side
is still judged on the other. On a section with cortex torn off
one side, excluding RSP/VIS/PTLp by name also drops the intact hemisphere;
`:left` keeps it.

## Masks (`masks.py`)

One module builds every mask. FOOTPRINT masks (`footprint_masks`: every fit; a
restricted fit's are cut to the regions' neighbourhood): fixed = placed atlas footprint widened by `MASK_MARGIN_MM`
(0.2); moving = that footprint minus the damage zone, the excluded regions
widened by `EXCLUSION_MARGIN_MM` (blanking alone left an edge that still
pulled tissue). Excluded regions are blanked on the native plane for EVERY
atlas image kind before placement (one-sided entries: that half of the
region only).

Route B with the model's outline: fixed = the footprint plus the outline's
area widened by `MASK_MARGIN_MM`, minus unnamed areas inside the outline (the
strip between the outline and the placed atlas stays in, so atlas left there
costs); moving = the placed atlas widened by `TRACE_MOVING_MARGIN_MM` (1.0;
drawn borders, nothing to fence off) minus the damage zone, so an outline
beyond the atlas can draw it out.

The EM sorting (`em_weights`; made by `core/section_masks.em_sorting`, which
`make_mask` calls, never by a fit): the section's P(match) is a section mask;
a fit given one compares only pixels with P(match) >= `EM_MATCH_THRESHOLD`
(0.5), ANDed into the footprint's fixed mask (`prepare_fit(section_weights=)`);
the atlas (moving) mask is not touched, and the damage zone stays atlas-side.
`em_sorting` places the atlas image through the section's CURRENT registration
(its applied deformation included), takes the placed footprint widened by
`MASK_MARGIN_MM`, runs `em_weights` on the 20 µm working grid and upsamples
P(match) to the fit grid. The classes are estimated over the whole section
(the EM needs tissue and background to learn what each looks like): `muB`
from the section's own background beyond 500 µm of the footprint, a contrast
map per 0.8 mm block (both adult mouse, times the brain scale). `sorting`
overrides `match_sd`, `background_sd`, `artifact_sd`, `background_mean`,
`artifact_mean` (brightness 0-1; `masks.SORTING_KEYS`) that are otherwise
estimated. `em_weights` is a port of
emlddmm's three-class EM (Tward et al. 2025, doi 10.1038/s41467-025-65317-7)
with the warp held: matching J ~ N(f(I_warped), sigmaM)
with f linear per `masks.block_px(scale)` block (`EM_BLOCK_PX`, 16, times the
brain scale) of the `EM_GRID_UM` (50 µm) grid,
artifact ~ N(muA, sigmaA), background ~ N(muB, sigmaB), `EM_ITERATIONS` (20)
E/M alternations, emlddmm's ridge. Two departures from emlddmm's defaults,
commented in the code: sigmas all 0.5 x std(J), fixed (its (1, 2, 5) x std
flagged nothing, not even a 0.3 mm hole), and priors fixed at (0.9, 0.05,
0.05) (updated priors ran away on DAPI, where dim tissue has the background's
intensity). muB is the image's own background, muA the opposite extreme
quantile of the tissue. On flat, textureless synthetic tissue a hole of 0.5 mm
or more inside one uniform region is absorbed by its block's contrast offset
and kept.

## The record (`record.py`)

`field_mm[y, x]` = [dx, dy] mm: section point (x, y)·mm_per_px corresponds to
placed-atlas point + field (brainglobe-registration's direction); native =
inv(atlas_to_section) @ (pixel + field/mm_per_px), the order `core/maps.native_points` composes. A sequential record's field is
the total. Also: inverse field (or None), engine name/version/runtime/native
parameters, working grid, warped leaf labels (unclipped), the placed
footprint mask the fixed mask starts from, `left_out` (section pixels an engine's
own weighting dropped, on the section grid, None otherwise; for review
pictures, never read back into a fit), `weights` (the engine's own P(match)
sorting, float16 on the section grid, None when it estimated nothing: the
source of `make_mask`'s `fit` kind), excluded ids, `native_to_volume_index` (plane pixel → atlas
volume index, so exports need no atlas object), `save()`/`load()`,
and a free-form `provenance` dict saved with the metadata (the linear tool
stores section id, linear handoff metadata and its inputs there). The job
folder's maps (`coords.tif`, `labels.tif`, `residual.tif`, written by
`export_maps`) and the VisuAlign markers are composed from it by
`core/maps.py` (`native_points`).

On disk (`save(directory)` / `load(directory)`, `docs/reference/file_formats.md`):
`record.json` (format `RECORD_VERSION` 2, written last: a folder without it
is incomplete) and `warp.ome.zarr` (`ome_warp.py`: a Zarr v3 store in the
OME-Zarr 0.6 layout; the fields `(2, rows, cols)` `(dy, dx)` float16, Blosc
zstd + shuffle, read back as `(rows, cols, 2)` float32; `warped_labels`,
`footprint`, `left_out`, `weights` (float16) lossless; the root's `ome.scene` states section mm
-> placed atlas mm (a `bijection` of the two `displacements`) -> atlas µm (a
`sequence`: scale, the placement's inverse affine, the plane-to-volume
affine, the resolution), written from the record and never read back). No
other format is read. The parent is BY REFERENCE: `record.json`'s `parent`
is the parent's folder name (`saved_as`), a sibling folder; `save` saves a
parent that is not there first (one without `saved_as` is a `ValueError`),
`load` reads the chain from the siblings. `save` rounds the fields to
float16 precision in memory first (`round_to_storage`,
`ome_warp.at_storage_precision`), and `core.deformation.RecordStore.put`
does the same for every record it holds, so a record in memory equals the
one read back and a fit built on either is the same (SyN refinements are
sensitive to their start: a 0.24 µm change of a soft whole-section
refinement's start moved its result by up to 1 mm on a rat section).
`RecordStore` saves under `deformable/<key[:24]>` and deletes nothing, so
undo and every child find their records.

Diagnostics are reported, never enforced: per-region area ratio
warped/placed from the Jacobian (`N_R / Σ_{p∈R} J`), raster ratio, fold
fraction in the footprint, inverse consistency, and flags REGION_COMPRESSED /
EXPANDED / VANISHED / FOLDED, FOLDS and DISPLACEMENT_OUTSIZED. A fit's reply lists them (`core.deformation.summary`: whole-section flags first,
at most `MAX_FLAGS`, each region flag with its acronym); a notice words them in one
line, each code once and a region code with its regions (`core.deformation.flag_words`,
also `langslice.plugin_api.flag_words`). Limits are the
named constants `TISSUE_AREA_RATIO_LIMITS` (0.5, 2),
`VENTRICLE_AREA_RATIO_LIMITS` (0.1, 10) for `VS` and descendants,
`MIN_FLAG_AREA_MM2`, `FOLD_FRACTION_LIMIT`, and for DISPLACEMENT_OUTSIZED
(`displacement_report`, also in `diagnostics["displacement"]`) a maximum
displacement in the footprint above `OUTSIZED_MAX_FRACTION` (0.1) of the footprint's
longest extent, or a median above `OUTSIZED_MEDIAN_MM` (0.6, adult mouse;
times the brain scale, `MIN_FLAG_AREA_MM2` times its square). The median
limit sits above every observed median.

DISPLACEMENT_OUTSIZED counts only what the dense fits add beyond the agent's
landmark warp, so a deliberate large landmark move is never flagged. A
landmark application (`finish_fit(..., landmark_warp=True)`) measures nothing
(its base is itself); any other record measures `field - base` where the base
is the field of the most recent `landmark_triangulation` record in its chain
(`record.landmark_base_field`, walking `parent`), and a chain without one is
measured from the linear placement as before. Landmarks applied over a fit
make that combined field the base for later fits. `displacement["max_mm"]`
etc. and `beyond_landmark_warp` say which was measured. Folds and region area
ratios are always measured on the full field.

## Visual review

Judge fits by drawing the final borders on the ORIGINAL section next to the
linear-only borders, region by region; diagnostics and synthetic recovery are
secondary. Use `render.draw_warped_borders(image, record, atlas, highlight=...,
warped=...)` (exported from the package): it composes the linear placement and
the residual field and draws per-region blurred indicators sampled bilinearly
on a 3x supersampled grid (the approach of `core.atlas.render.placed_border_coverage`,
which is affine-only, so the composed sampling lives here), one shared line per
edge, antialiased, cased (`core=` black or white, edged by the other: see
`core.canvas.composite_cased`; the marked regions' second ink is plain). `highlight` (acronyms/ids, descendants
included, one-sided entries drawn on that side of the record's placement)
draws those regions' edges strongly, `core.atlas.render.HIGHLIGHT_WIDTH`
times as thick, over a faint outline of the other regions; `marked` draws a second set (regions excluded from a
fit) in `MARKED_COLOR`, and `outlines` (`all`/`none`) limits the
rest (`warped_border_layers` returns every layer; `drawn_border_coverage`
the lines a call draws, as a saved picture's borders layer). `resampled_record` carries a
record onto a smaller or cropped grid for pictures; `warp_section_image`
resamples a section render into its placed-atlas frame through the stored
inverse field, so a picture drawn under the
linear placement shows the full registration; `warp_source_points` gives the
section points it shows at any points of its result (a zoom reads the image
file there, `core/native.py`). Never judge borders traced from `record.labels`: those
are nearest-sampled from the 25 um atlas grid and look staircased on fine
section pixels (3.5 section px per step at 7 um/px), which is the whole of the
zig-zag along hippocampal arcs in label-map mode (the residual field there
has under 1 um of high-frequency content). Smoothing is 1.4 atlas px:
1.0 leaves a ripple along shallow edges (the annotation's own steps), 2.0
turns thin regions into dotted blobs.

## Known limits

Label-map mode is SyN-only. Named model regions inherit the
linear placement's naming. Composed sequential inverses carry each step's
approximation.

## A map made elsewhere

`fit.warp_from_native_map(atlas, placement, native_xy, detail, engine=)`: a
full map (section-grid pixel centres to native atlas pixels, `(H, W, 2)`)
as a record, the residual `field_mm = (atlas_to_section @ native - pixel) *
mm_per_px` over *placement*, its inverse by `_invert_step`'s fixed point
(guess: minus the forward field), `prepare_warp` and `finish_fit` as for the
landmark warp below, `engine["name"]` *engine*. A tool plugin's
deformations go through it (`core.deformation.record_from_map`,
`ops.deformable.warp_from_map`).

## Landmark record adapter

`ops.landmarks.landmark_triangulation` uses `prepare_warp` (the atlas side
only: native plane, placed footprint, ventricles, atlas facts; no images,
masks or settings) and `finish_fit` to build the same residual-field record
and report as other nonlinear methods, with fields from `core.triangulation`
instead of an engine. Its record has no `settings` (null) and the engine name
`landmark_triangulation`; source/target
vertices and connectivity are saved in `engine.native_parameters`. A
composition converts each atlas endpoint through the baseline's stored inverse,
then composes the resulting correction with its field. On the first application,
the current record becomes the baseline `parent`. Consecutive landmark
applications reuse that parent and replace the landmark result; an intervening
other method becomes a new baseline. The state reference keeps the baseline
key and steps, while job undo history preserves each previously applied result.
Degenerate/folded mesh triangles are rejected before
application; biological plausibility diagnostics retain their reporting-only
semantics.

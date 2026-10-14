# LangSlice `core/deformable/` — library deformable fit of a placed atlas plane

Package guide for `src/langslice/core/deformable/`; `AGENTS.md` is a verbatim
twin of this file. A core sub-package, like `core/affine.py`: it belongs to
neither the linear method nor `core/nonlinear/` and imports neither. It is the
engine behind the deformable fits (`ants_syn`, whose name predates the
FireANTs engine, and the packaged
`trace_borders`' fit, through `core/deformation.py`, task `nonlinear`) and
behind `elastix_affine` (`core/transform.elastix_affine`: `prepare_fit`
builds its images and masks, `engines.run_elastix_affine` fits). The job folder's maps and VisuAlign
markers read an applied record (`core/maps.py`); the ABBA connector receives
it as landmark pairs (`core/abba_warp.py`). `engines.py` is LangSlice's one
itk-elastix registration wrapper.

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
  (`stain_metric`, default `local_correlation`): neighbourhood
  cross-correlation over a box window of radius `CORRELATION_RADIUS_UM`
  (80 µm, rounded to working pixels: 4 at standard, 2 at coarse), each window
  counting only its in-mask pixels (as ITK's; `engines._channel_loss`); the
  Elastix affine has no local correlation and uses `ELASTIX_STAIN_METRIC`
  (mutual information); `mutual_information` is the alternative. Plus, by
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
  by the same Gaussian ridge (`settings.LINE_SOFTENING_UM`, fixed at 60 µm).
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
- The deformable engine (`engines.run_engine`): FireANTs' `SyNRegistration`
  (torch, a core dependency, imported lazily with an install hint; on the
  GPU when torch sees one, `engines.fit_device`) on the working-grid images,
  the masks riding as each image's last channel (FireANTs' masked mode,
  resized but never blurred by the pyramid, warped with their image, no
  gradient through them). Settings from the former ANTs ones: shrink factors
  2^(n-1)..1 for the detail's iteration list (`[8, 4, 2, 1]` at standard),
  `smooth_grad_sigma` = update sigma / voxel, `smooth_warp_sigma` = total
  sigma / voxel, the edge channel an extra channel. Optimizer: FireANTs' SGD
  with ANTs' step rule (`engines._global_step_sgd`: the gradient scaled so
  its largest vector moves `FIREANTS_LEARNING_RATE` (0.8) half-pixels of the
  level; Adam's per-pixel scaling moved excluded regions as far as any).
  Returns millimetre fields both ways through the SyN midpoint: section ->
  atlas = fwd o rev^-1, atlas -> section = rev o fwd^-1, each half inverted by
  a damped fixed point (`engines._invert`, 200 sampling iterations; FireANTs'
  own inversion is an optimization that is not reproducible on the GPU).
  Any other `FitSettings.engine` is refused there: `engine="ants"` names the
  SyN engine (the record's engine version names FireANTs),
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
- `fit.run_fit(inputs, settings, em_damage_zone)` is a prepared fit's
  engine work on plain arrays: one run, or for a whole-section stain fit
  (`PreparedFit.em_damage_zone` set) a run with the footprint masks, the EM
  masks from the atlas image it warped (`masks.em_masks`) and a second run
  from the same start, whose result counts (its native parameters gain
  `section_mask`: the EM's settings, its fitted means and widths, the
  fraction of the footprint left out, the first run's time; `left_out` the
  dropped section pixels). `run_prepared` runs one in this process.
  `fit_prepared` runs 1–8 prepared fits (different sections or settings):
  on a GPU one after another in the caller, on the CPU in a spawn process
  pool (`cpu_count // FIT_THREADS` workers at most, one per fit); a failing
  candidate is a `CandidateFailure`, never an exception.

## Settings (`settings.py`)

Named ladders, physical units: `stiffness` soft/medium/firm
(`settings.STIFFNESS`: update/total Gaussian sigma in mm, converted to
voxels); `detail` coarse/standard (working µm and pyramid depth/iterations —
the extra levels are SyN's capture range); `atlas_image`, `section_image`, `exclude`,
`labels`, `structures`, `neighbourhood_um`, `preprocess`, and for stain fits
`stain_metric` (`local_correlation` | `mutual_information`) and
`stain_edges` (lines ignore both). `FitSettings.metric` resolves the metric
from pairing, engine and stain metric (`metric_for`); lines against
an atlas image and the stain against `borders`/`borders_merged` are refused.
The step size is fixed.

Stain defaults, and why: local correlation at 80 µm radius followed visible
structure better than mutual information but on its own still crossed
enlarged ventricles, so the edge channel is on by default. There is no `stiff`
setting (it left enlarged ventricles unfilled), no `fine` detail (4-5x
slower, worse outlines) and the linear tool always runs `standard`; line
softening is the constant `LINE_SOFTENING_UM` = 60. `template` is the default
stain atlas image; the agent picks whichever of the atlas's images looks most
like its sections (on `ccfv3augmented_mouse_25um` the template is the Blue
Brain averaged Nissl and `single_animal_nissl` is an additional reference).
On synthetic sections (flat regions, no texture) local correlation recovers a
known warp worse than mutual information, so the mechanism tests pin mutual
information.

**One side of a region.** Any `exclude` or `structures` entry may name one
side, `"CTX:left"` / `"CTX:right"` (`core.atlas.sides`): left and right of the
SECTION as displayed (the fit grid: oriented render, rotation and flip
applied), carried to the native plane through the placement's
`atlas_to_section` (`atlas_images.placement_left`). Exclusion is therefore a
native-plane MASK (`atlas_images.regions_mask`), not an id set:
`PreparedFit.excluded_mask` blanks the moving image, `excluded` (and the
record's `excluded_ids`) holds only whole-region ids (`whole_region_ids`),
and area diagnostics drop excluded pixels, so a region excluded on one side
is still judged on the other. On a section with cortex torn off
one side, excluding RSP/VIS/PTLp by name also drops the intact hemisphere;
`:left` keeps it.

## Masks (`masks.py`)

One module builds every mask. FOOTPRINT masks (`footprint_masks`: the
Elastix affine, restricted steps, traced lines, and a whole-section fit's
first run): fixed = placed atlas footprint widened by `MASK_MARGIN_MM`
(0.2); moving = that footprint minus the damage zone, the excluded regions
widened by `EXCLUSION_MARGIN_MM` (blanking alone left an edge that still
pulled tissue). Excluded regions are blanked on the native plane for EVERY
atlas image kind before placement (one-sided entries: that half of the
region only).

EM masks (`em_masks`, a whole-section stain fit's second run): the section
mask is P(match) >= `EM_MATCH_THRESHOLD` (0.5) over the whole working grid,
the atlas mask everything except the damage zone. `em_weights` is a port of
emlddmm's three-class EM (Tward et al. 2025, doi 10.1038/s41467-025-65317-7)
with the warp fixed at the first run's: matching J ~ N(f(I_warped), sigmaM)
with f linear per `EM_BLOCK_PX` (16) block of the `EM_GRID_UM` (50 µm) grid,
artifact ~ N(muA, sigmaA), background ~ N(muB, sigmaB), `EM_ITERATIONS` (20)
E/M alternations, emlddmm's ridge. Two departures from emlddmm's defaults,
commented in the code: sigmas all 0.5 x std(J), fixed (its (1, 2, 5) x std
flagged nothing, not even a 0.3 mm hole), and priors fixed at (0.9, 0.05,
0.05) (updated priors ran away on DAPI, where dim tissue has the background's
intensity). muB is the image's own background, muA the opposite extreme
quantile of the tissue. `EM_PARAMETERS` (every setting) is part of the
deformable cache key, as is the engine (`core.deformation.cache_key`). On
flat, textureless synthetic tissue a hole of 0.5 mm or more inside one
uniform region is absorbed by its block's contrast offset and kept.

## The record (`record.py`)

`field_mm[y, x]` = [dx, dy] mm: section point (x, y)·mm_per_px corresponds to
placed-atlas point + field (brainglobe-registration's direction); native =
inv(atlas_to_section) @ (pixel + field/mm_per_px), the order `core/maps.native_points` composes. A sequential record's field is
the total. Also: inverse field (or None), engine name/version/runtime/native
parameters, working grid, warped leaf labels (unclipped), the placed
footprint mask the fixed mask starts from, `left_out` (the section pixels an
EM section mask dropped, on the section grid, None otherwise; for review
pictures, never read back into a fit), excluded ids, `native_to_volume_index` (plane pixel → atlas
volume index, so exports need no atlas object), `save()`/`load()` (parent chain under `parent/`),
and a free-form `provenance` dict saved with the metadata (the linear tool
stores section id, linear handoff metadata and its inputs there). The job
folder's maps (`coords.tif`, `labels.tif`, `residual.tif`) and the VisuAlign
markers are composed from it by `core/maps.py` (`native_points`).

Diagnostics are reported, never enforced: per-region area ratio
warped/placed from the Jacobian (`N_R / Σ_{p∈R} J`), raster ratio, fold
fraction in the footprint, inverse consistency, and flags REGION_COMPRESSED /
EXPANDED / VANISHED / FOLDED, FOLDS and DISPLACEMENT_OUTSIZED. Limits are the
named constants `TISSUE_AREA_RATIO_LIMITS` (0.5, 2),
`VENTRICLE_AREA_RATIO_LIMITS` (0.1, 10) for `VS` and descendants,
`MIN_FLAG_AREA_MM2`, `FOLD_FRACTION_LIMIT`, and for DISPLACEMENT_OUTSIZED
(`displacement_report`, also in `diagnostics["displacement"]`) a maximum
displacement in the footprint above `OUTSIZED_MAX_FRACTION` (0.1) of the footprint's
longest extent, or a median above `OUTSIZED_MEDIAN_MM` (0.6). The median
limit sits above every observed median.

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

# LangSlice `core/deformable/` — library deformable fit of a placed atlas plane

Package guide for `src/langslice/core/deformable/`; `AGENTS.md` is a verbatim
twin of this file. A core sub-package, like `core/affine.py`: it belongs to
neither the linear method nor `core/nonlinear/` and imports neither. It is the
engine behind the linear agent's `fit_deformable` tool (`core/deformation.py`,
task `nonlinear`) and behind `fit_affine`'s default Elastix method
(`core/transform.elastix_affine`: `prepare_fit` builds its images and masks,
`engines.run_elastix_affine` fits). The job folder's maps and VisuAlign
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
  an atlas image — `template` (BrainGlobe reference), or `nissl` (the Nissl
  template aligned to the CCFv3, `nissl.py`; Allen mouse atlases). Metric
  (`stain_metric`, default `local_correlation`): ANTs' neighbourhood
  cross-correlation `CC` over a window of radius `CORRELATION_RADIUS_UM`
  (80 µm, rounded to working pixels: 4 at standard, 2 at coarse); Elastix has
  no local correlation and uses `ELASTIX_STAIN_METRIC` (mutual information)
  and says so in `engine["notes"]`; `mutual_information` is the alternative. Plus, by default (`stain_edges`), an EDGE channel: the gradient
  magnitude after a `EDGE_SIGMA_UM` (30 µm) Gaussian of the stain and of the
  atlas image (`fit.edge_image`, scaled by its 99th percentile in each mask),
  a second metric of the same kind at `EDGE_CHANNEL_WEIGHT` 1.0. The
  magnitude ignores which side of an edge is brighter, so fluorescent and
  brightfield stains and the template share pial surface, ventricle walls
  and layer boundaries. Excluded regions are inpainted from their
  surroundings before the atlas edges are taken (`fit._fill_from_surroundings`):
  the blanked region's rim otherwise pulled tissue in (Elastix).
  Optionally SEQUENTIAL: each call with `structures=(...)` (acronyms/ids,
  descendants included, optionally one-sided) restricts both masks to those
  structures plus
  `neighbourhood_um`, and with `previous=record` composes onto that record;
  `record.parent` / `record.undo()` is the step before.
- Route B (image model): the model's extracted line mask (`lines=`, on the
  section grid, e.g. `trace_borders`' `extracted_lines.png` with its
  `atlas_to_canvas`) against `borders_merged` (the same color-family set the
  model was shown, `core.atlas.render.family_labels`) or `borders`; both softened
  by the same Gaussian ridge (`settings.LINE_SOFTENING_UM`, fixed at 60 µm).
  Mean squares. `settings.traced_settings(engine)` is the traced-lines
  setting in one call: ANTs with the lines as named regions too
  (`labels="model"`), or Elastix lines against borders; `engine=None` picks
  ANTs when installed. The crossed pairings are refused (`metric_for`): lines
  against `template`/`nissl`, and the stain against borders (the pairing that
  stays at the linear placement).
- Label-map channels (`labels=`, ANTs only): `model` names each area the
  model's lines enclose after the placed merged region it overlaps most
  (`regions.named_regions`, small loops around bubbles dropped, joint
  one-to-one renaming) and pairs each region's indicator with the atlas's;
  `auto` adds the stain's tissue footprint and empty interior holes near
  placed ventricles (the linear tool adds `auto` to every ANTs stain fit,
  `core/deformation.Choice`). Built as `ants.registration`
  `multivariate_extras` (the same per-label MeanSquares construction
  `ants.label_image_registration` uses, which hard-codes `SyN[0.2,3,0]` and
  so would ignore stiffness); the edge channel is the first extra.
- Engines (`engines.py`): ANTs `SyNOnly` with an identity initial transform
  (antspyx, optional `registration` extra, imported lazily with an install
  hint), or Elastix B-spline with a bending-energy penalty (itk-elastix,
  core dependency). Both see the same working-grid images, masks, spacing and
  origin and return millimetre fields. ANTs supplies its own inverse; Elastix
  has none, so its inverse is a fixed-point approximation, said so in
  `inverse_source` and `engine["notes"]`. Elastix's edge channel is a
  second image pair (`AddFixedImage`/`AddMovingImage`, masks per pair); its
  multi-image rules want pyramids, interpolators and samplers numbering one
  per metric, the bending penalty included, and so an image pair per metric:
  the intensity pair is passed again for the penalty.
- DETERMINISM: identical inputs give identical fields, bit for bit (test
  `test_identical_inputs_give_identical_fields`: twice in process and in a
  pool worker). Fixed seed `engines.RANDOM_SEED` (antsRegistration
  `--random-seed` through `ants.config._random_seed`, restored after the
  call; Elastix `RandomSeed`, which its random sampler reads). Fixed thread
  count `engines.FIT_THREADS` (8), because a multithreaded metric's sums
  depend on the split (ANTs fields moved ~2 µm between 1 and 8 threads,
  Elastix ~1e-9 mm). ANTs' bundled ITK reads
  `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS` once per process at first use, so
  `engines.import_ants` (also used by `image_prep.ants_enhance`) sets it,
  runs one tiny filter and restores the environment; pool workers set it in
  `engines.ants_worker`. If something outside LangSlice loaded ANTs first the
  count is unknown: `threads` is None and a note says so. Elastix's own
  `SetNumberOfThreads` segfaults in itk-elastix 0.25.4, so the call sets
  ITK's global default for its duration instead. A single fit takes seconds at 8 threads and roughly
  doubles with each halving.
- `engines.run_elastix_affine` (the linear affine, not a deformation): an
  Elastix `AffineTransform` from the identity on the same prepared inputs
  (intensity pair, edge pair, masks; `AutomaticTransformInitialization` off,
  so it refines the placement the moving image was drawn at), mutual
  information, `ELASTIX_AFFINE_RESOLUTIONS` 3 x `ELASTIX_AFFINE_ITERATIONS`
  500, the same seed and thread count. Returns `AffineResult.matrix_mm`
  (section mm -> placed-atlas mm, the field's direction); a non-finite or
  singular matrix raises. `_elastix_register` is the one Elastix call both
  runners share (images, masks, extra pairs, the global thread count). The
  affine stays Elastix-only; ANTs is a deformable option.
- Optional ANTs preprocessing of the stain (`preprocess=("n4", "denoise")`).
- `fit_prepared` runs 1–8 prepared fits (different sections or settings) in
  a spawn process pool (atlas work in the caller, engine calls in workers,
  `cpu_count // FIT_THREADS` workers at most, one per fit); a failing
  candidate is a `CandidateFailure`, never an exception.

## Settings (`settings.py`)

Named ladders, physical units: `stiffness` soft/medium/firm (ANTs
update/total Gaussian sigma in mm, converted to voxel variances; Elastix
control spacing in mm and bending weight, scaled by
`ELASTIX_MEAN_SQUARES_BENDING_SCALE` for mean squares); `detail`
coarse/standard (working µm and pyramid depth/iterations — the extra ANTs
levels are its capture range); `atlas_image`, `section_image`, `exclude`,
`labels`, `structures`, `neighbourhood_um`, `preprocess`, and for stain fits
`stain_metric` (`local_correlation` | `mutual_information`) and
`stain_edges` (lines ignore both). `FitSettings.metric` resolves the metric
from pairing, engine and stain metric (`metric_for`); lines against
`template`/`nissl` and the stain against `borders`/`borders_merged` are refused.
The step size is fixed.

Stain defaults, and why: local correlation at 80 µm radius followed visible
structure better than mutual information but on its own still crossed
enlarged ventricles, so the edge channel is on by default and the tool adds
the automatic label channels to every ANTs stain fit (they give a continuous
pial outline and fill enlarged ventricles; their known error: small
lateral-ventricle pieces pulled into a dorsal third-ventricle hole, since the
ventricle channel pairs all ventricles with all holes). There is no `stiff`
setting (it left enlarged ventricles unfilled), no `fine` detail (4-5x
slower, worse outlines) and the linear tool always runs `standard`; line
softening is the constant `LINE_SOFTENING_UM` = 60. `template` is the default
stain atlas image; the aligned `nissl` is untested on real sections (the
misaligned Allen Nissl it replaced sat 40-80 µm inside fluorescent tissue).
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

Fixed: tissue (`image_prep.foreground_mask` at a 1024-px proxy, holes
filled) widened by `MASK_MARGIN_MM`, minus the torn-edge band. Default torn
rule: tissue outline lying more than `TORN_EDGE_DEPTH_MM` inside the placed,
non-excluded atlas footprint, widened by `TORN_EDGE_BAND_MM`; `torn_band=`
overrides it. Moving: placed footprint widened by the margin, minus excluded
regions and `EXCLUSION_MARGIN_MM` around them (blanking alone left an edge
that still pulled tissue). Excluded regions are blanked on the native plane
for EVERY atlas image kind before placement (one-sided entries: that half
of the region only).

## The record (`record.py`)

`field_mm[y, x]` = [dx, dy] mm: section point (x, y)·mm_per_px corresponds to
placed-atlas point + field (brainglobe-registration's direction); native =
inv(atlas_to_section) @ (pixel + field/mm_per_px), the order `core/maps.native_points` composes. A sequential record's field is
the total. Also: inverse field (or None), engine name/version/runtime/native
parameters, working grid, warped leaf labels CLIPPED TO TISSUE, tissue and
torn-band masks, excluded ids, `native_to_volume_index` (plane pixel → atlas
volume index, so exports need no atlas object), `native_coordinates()`,
`volume_coordinates()`, `save()`/`load()` (parent chain under `parent/`),
and a free-form `provenance` dict saved with the metadata (the linear tool
stores section id, linear handoff metadata and its inputs there). The job
folder's maps (`coords.tif`, `labels.tif`, `residual.tif`) and the VisuAlign
markers are composed from it by `core/maps.py` (`native_points`).

Diagnostics are reported, never enforced: per-region area ratio
warped/placed from the Jacobian (`N_R / Σ_{p∈R} J`), raster ratio, fold
fraction in tissue, inverse consistency, and flags REGION_COMPRESSED /
EXPANDED / VANISHED / FOLDED, FOLDS and DISPLACEMENT_OUTSIZED. Limits are the
named constants `TISSUE_AREA_RATIO_LIMITS` (0.5, 2),
`VENTRICLE_AREA_RATIO_LIMITS` (0.1, 10) for `VS` and descendants,
`MIN_FLAG_AREA_MM2`, `FOLD_FRACTION_LIMIT`, and for DISPLACEMENT_OUTSIZED
(`displacement_report`, also in `diagnostics["displacement"]`) a maximum
displacement in tissue above `OUTSIZED_MAX_FRACTION` (0.1) of the tissue's
longest extent, or a median above `OUTSIZED_MEDIAN_MM` (0.6).
Elastix fits are the ones that trip DISPLACEMENT_OUTSIZED in practice (a
stretch over a displaced flap, or the raw blue channel against Nissl); ANTs
fits stay well under it. The median limit sits above every observed median:
0.4 mm would flag Elastix fits whose borders look as plausible as ANTs's.

## The aligned Nissl (`nissl.py`)

`NisslAtlas` serves the `nissl` image on any asr atlas covering the Allen
CCFv3 extent (13.2 x 8.0 x 11.4 mm, `compatible`): the reference image of
BrainGlobe's `ccfv3augmented_mouse_25um` v1.0, the Blue Brain
population-averaged Nissl template (Piluso et al. 2025), loaded once per
process through the workspace's atlas loader (downloaded on first use; a
different shape is refused). That atlas is the CCFv3 grid extended along AP:
the CCFv3 starts `AP_OFFSET_MM` (0.35 mm, 14 voxels) into it, where tissue
outlines agree with `allen_mouse_25um` at Dice 0.997
(`test_the_augmented_atlas_holds_the_ccfv3_at_the_offset`). `sample_plane`
maps the run atlas's plane (`oblique.plane_index_coordinates`, any
resolution, cutting angles included) to millimetres, adds the offset, samples
trilinearly, and zeroes pixels outside the run atlas's brain (the averaged
template glows faintly beyond the tissue). The Allen Institute's own Nissl
(ABBA's) is misaligned with the CCFv3 annotation and is not read.

## Visual review

Judge fits by drawing the final borders on the ORIGINAL section next to the
linear-only borders, region by region; diagnostics and synthetic recovery are
secondary. Use `render.draw_warped_borders(image, record, atlas, highlight=...,
warped=...)` (exported from the package): it composes the linear placement and
the residual field and draws per-region blurred indicators sampled bilinearly
on a 3x supersampled grid (the approach of `core.atlas.render.placed_border_coverage`,
which is affine-only, so the composed sampling lives here), one shared line per
edge, antialiased, clipped to tissue. `highlight` (acronyms/ids, descendants
included, one-sided entries drawn on that side of the record's placement)
draws those regions' edges strongly over a faint outline of the
colour-family regions; `marked` draws a second set (regions excluded from a
fit) in `MARKED_COLOR`, and `outlines` (`all`/`outer`/`none`) limits the
rest (`warped_border_layers` returns every layer; `drawn_border_coverage`
the lines a call draws, as a saved picture's borders layer). `resampled_record` carries a
record onto a smaller or cropped grid for pictures; `warp_section_image`
resamples a section render into its placed-atlas frame through the inverse
field (fixed-point inverse when none is stored), so a picture drawn under the
linear placement shows the full registration. Never judge borders traced from `record.labels`: those
are nearest-sampled from the 25 um atlas grid and look staircased on fine
section pixels (3.5 section px per step at 7 um/px), which is the whole of the
zig-zag along hippocampal arcs in label-map mode (the residual field there
has under 1 um of high-frequency content). Smoothing is 1.4 atlas px:
1.0 leaves a ripple along shallow edges (the annotation's own steps), 2.0
turns thin regions into dotted blobs.

## Known limits

Label-map mode is ANTs-only. The torn-edge rule also marks real outline
wherever the linear placement overhangs tissue by more than its depth, and
covers most of the outline on heavily damaged sections; pass `torn_band` or
`exclude` there. `auto` ventricle detection only finds empty holes near
placed ventricles (none at caudal levels). Named model regions inherit the
linear placement's naming. Composed sequential inverses carry each step's
approximation.

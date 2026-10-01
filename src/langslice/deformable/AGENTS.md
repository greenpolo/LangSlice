# LangSlice `deformable/` — library deformable fit of a placed atlas plane

Package guide; `AGENTS.md` is a verbatim twin of this file. Shared top-level
package, like `affine.py`: it belongs to neither `linear/` nor `nonlinear/`
and imports neither. It is the engine behind the linear agent's
`fit_deformable` tool (`linear/deformation.py`, task `nonlinear`); no CLI,
host or export adapter uses it yet.

## What it does

Given a section image, a linear placement (`geometry.Placement`: atlas name,
position, plane, pitch/yaw, the 3x3 `atlas_to_section` map from native
atlas-plane pixel centres to section pixel centres, section mm/px) and one
`settings.FitSettings`, it fits a residual deformation with a LIBRARY engine
and returns a `record.DeformableRecord`. No custom solver.

- Route A (no image model): the stain image (`section_image="stain"`) against
  an atlas image — `ara` (BrainGlobe reference), or `nissl` (ABBA's cached
  Allen Nissl, ABBA hosts only, `atlas_images_for_host`). Metric
  (`stain_metric`, default `local_correlation`): ANTs' neighbourhood
  cross-correlation `CC` over a window of radius `CORRELATION_RADIUS_UM`
  (80 µm, rounded to working pixels: 4 at standard, 2 at coarse); Elastix has
  no local correlation and uses `ELASTIX_STAIN_METRIC` (mutual information)
  and says so in `engine["notes"]`; `mutual_information` is the pre-2026-10-02
  metric. Plus, by default (`stain_edges`), an EDGE channel: the gradient
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
  model was shown, `atlas.render.family_labels`) or `borders`; both softened
  by the same Gaussian ridge (`settings.LINE_SOFTENING_UM`, fixed at 60 µm).
  Mean squares. The crossed pairings are refused (`metric_for`): lines
  against `ara`/`nissl`, and the stain against borders (the ceiling test's
  worst pairing: it stayed at the linear placement).
- Label-map channels (`labels=`, ANTs only): `model` names each area the
  model's lines enclose after the placed merged region it overlaps most
  (`regions.named_regions`, small loops around bubbles dropped, joint
  one-to-one renaming) and pairs each region's indicator with the atlas's;
  `auto` adds the stain's tissue footprint and empty interior holes near
  placed ventricles (the linear tool adds `auto` to every ANTs stain fit,
  `linear/deformation.Choice`). Built as `ants.registration`
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
  ITK's global default for its duration instead. Single-fit runtime at 8
  threads ~5 s (local correlation + edges, M04_B_05), 9 s at 4, 15 s at 2.
- Optional ANTs preprocessing of the stain (`preprocess=("n4", "denoise")`).
- `fit_candidates` runs 1–8 settings in a spawn process pool (atlas work in
  the caller, engine calls in workers, `cpu_count // FIT_THREADS` workers at
  most, one per fit); a failing candidate is a
  `CandidateFailure`, never an exception. `fit_prepared` is the same pool
  over already prepared fits (different sections or section images).

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
`ara`/`nissl` and the stain against `borders`/`borders_merged` are refused.
`FitSettings.from_dict` loads records saved before 2026-10-02 (no
`stain_metric`/`stain_edges` keys) as mutual information without edges,
which is what they were fitted with. The step size is fixed.

The 2026-10-02 stain ceiling test (`_local/runs/20261001_ceiling_test_stain2`,
the same eight sections, 15 stain settings each, judged by eye) set the
stain defaults. Local correlation at 40 µm radius was unstable (the one ANTs
DISPLACEMENT_OUTSIZED, stray bands); at 80 and 120 µm (indistinguishable)
interior lines followed visible structure better than mutual information
(olfactory-bulb cores, thalamic nuclei) but on its own still crossed
enlarged ventricles and missed a midline slit; the edge channel fixed most
of that and put a Nissl outline back on the edge (~1.8x the runtime). The
automatic label channels were the clearest gain (continuous pial outline,
enlarged ventricles filled), hence the tool's ANTs stain default; their one
error: small lateral-ventricle pieces pulled into a dorsal third-ventricle
hole (the ventricle channel pairs all ventricles with all holes). Elastix
mutual information blew up again on M11_B_08 (1.07 mm); with edges it
stayed under the flag but still stretched cortex over the displaced flap,
as did AdvancedNormalizedCorrelation (one global correlation) + edges. On
the synthetic sections (flat regions, no texture) local correlation
recovers a known warp worse than mutual information (error 0.39 vs 0.12 of
the warp at standard), so the mechanism tests pin mutual information, and
Elastix exclusion leaks more with edges (0.29 vs 0.12 of the free fit's
movement inside the excluded region).

The 2026-10-01 ceiling test (eight LSD_910 sections, 400 fits, judged by
eye) trimmed the ladders: `stiff` was never the best-looking fit and left
enlarged ventricles unfilled, so it is gone; detail never changed which fit
looked best — `coarse` looked the same at ~1 s, `fine` (10 µm) was 4-5x
slower with worse outline numbers and is gone, and the linear tool always
runs `standard`; line softening 30 and 60 µm looked the same and 120 µm
slightly worse with more Elastix folds, so it is the constant
`LINE_SOFTENING_UM` = 60 (`FitSettings.from_dict` drops the old
`line_softening_um` key of records saved before). On that fluorescent data
the Nissl reference's pial outline sat 40-80 µm inside the tissue's bright
surface rim with every engine and stiffness, where `ara` followed the edge
(outline error 19-25 µm against 37-63 µm for Nissl, 65 µm linear); Nissl
stays available, untested on brightfield Nissl stains.

**One side of a region.** Any `exclude` or `structures` entry may name one
side, `"CTX:left"` / `"CTX:right"` (`atlas.sides`): left and right of the
SECTION as displayed (the fit grid: oriented render, rotation and flip
applied), carried to the native plane through the placement's
`atlas_to_section` (`atlas_images.placement_left`). Exclusion is therefore a
native-plane MASK (`atlas_images.regions_mask`), not an id set:
`PreparedFit.excluded_mask` blanks the moving image, `excluded` (and the
record's `excluded_ids`) holds only whole-region ids (`whole_region_ids`),
and area diagnostics drop excluded pixels, so a region excluded on one side
is still judged on the other. Checked on M11_C_08 (cortex torn off one
side): excluding RSP/VIS/PTLp by name also dropped the intact hemisphere;
`:left` kept it.

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
inv(atlas_to_section) @ (pixel + field/mm_per_px), the same order as
`border_registration.composed_native_map`. A sequential record's field is
the total. Also: inverse field (or None), engine name/version/runtime/native
parameters, working grid, warped leaf labels CLIPPED TO TISSUE, tissue and
torn-band masks, excluded ids, `native_to_volume_index` (plane pixel → atlas
volume index, so exports need no atlas object), `native_coordinates()`,
`volume_coordinates()`, `save()`/`load()` (parent chain under `parent/`),
and a free-form `provenance` dict saved with the metadata (the linear tool
stores section id, linear handoff metadata and its inputs there). Export
adapters (ABBA, VisuAlign, BrainGlobe) are to read this record; none exist.

Diagnostics are reported, never enforced: per-region area ratio
warped/placed from the Jacobian (`N_R / Σ_{p∈R} J`), raster ratio, fold
fraction in tissue, inverse consistency, and flags REGION_COMPRESSED /
EXPANDED / VANISHED / FOLDED, FOLDS and DISPLACEMENT_OUTSIZED. Limits are the
named constants `TISSUE_AREA_RATIO_LIMITS` (0.5, 2),
`VENTRICLE_AREA_RATIO_LIMITS` (0.1, 10) for `VS` and descendants,
`MIN_FLAG_AREA_MM2`, `FOLD_FRACTION_LIMIT`, and for DISPLACEMENT_OUTSIZED
(`displacement_report`, also in `diagnostics["displacement"]`) a maximum
displacement in tissue above `OUTSIZED_MAX_FRACTION` (0.1) of the tissue's
longest extent, or a median above `OUTSIZED_MEDIAN_MM` (0.6). The ceiling
test's one blow-up (Elastix on the raw blue channel against Nissl, M11_C_08:
max 1.31 mm, median 0.46 mm, no fold or area flag; the atlas outline shrank
~1 mm inside the cortex) raised nothing before; applied to all 400 stored
fits the flag fires on six, all Elastix: that blow-up, Elastix medium on
M11_B_08 (1.08 mm, the atlas pulled onto a displaced flap), and four soft or
120 µm-softened Elastix fits (one on M11_B_08, three on M04_D_08, where the
ceiling test could not tell any candidate apart by eye). No ANTs fit fires (largest max 0.78 mm on
an 8.9 mm section). The median limit sits above every observed median
(largest 0.52 mm): 0.4 mm would have flagged Elastix fits on M11_C_08 whose
borders looked as plausible as ANTs's.

## ABBA's Allen volume (`abba_atlas.py`)

Reads `~/cached_atlas/ccf2017-mod65000-border-centered-mm-bc.h5` (+
`mouse_brain_ccfv3p1.xml`) lazily with h5py: the pyramid level no coarser
than the atlas voxel, only the slab the plane crosses, sampled at
`oblique.plane_index_coordinates` (BrainGlobe voxel centres mapped onto the
10 µm grid). ML order: h5 index k = BrainGlobe ML index k. The module
docstring gives the evidence (XML affine keeps data in native order; ABBA's
Nissl correlates with the Allen Institute's own `ara_nissl_50.nrrd` with a
positive antisymmetric part). Unverifiable by images: BrainGlobe's own
packaging of the symmetric volume. An XML with a different affine is refused.

## Visual review

Judge fits by drawing the final borders on the ORIGINAL section next to the
linear-only borders, region by region; diagnostics and synthetic recovery are
secondary. Use `render.draw_warped_borders(image, record, atlas, highlight=...,
warped=...)` (exported from the package): it composes the linear placement and
the residual field and draws per-region blurred indicators sampled bilinearly
on a 3x supersampled grid (the approach of `atlas.render.placed_border_coverage`,
which is affine-only, so the composed sampling lives here), one shared line per
edge, antialiased, clipped to tissue. `highlight` (acronyms/ids, descendants
included, one-sided entries drawn on that side of the record's placement)
draws those regions' edges strongly over a faint outline of the
colour-family regions; `marked` draws a second set (regions excluded from a
fit) in `MARKED_COLOR`, and `outlines` (`all`/`outer`/`none`) limits the
rest (`warped_border_layers` returns every layer). `resampled_record` carries a
record onto a smaller or cropped grid for pictures; `warp_section_image`
resamples a section render into its placed-atlas frame through the inverse
field (fixed-point inverse when none is stored), so a picture drawn under the
linear placement shows the full registration. Never judge borders traced from `record.labels`: those
are nearest-sampled from the 25 um atlas grid and look staircased on fine
section pixels (3.5 section px per step at 7 um/px), which was the whole of the
zig-zag once seen along hippocampal arcs in label-map mode (the residual field
there has under 1 um of high-frequency content). Smoothing is 1.4 atlas px:
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

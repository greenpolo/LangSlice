# LangSlice `deformable/` — library deformable fit of a placed atlas plane

Package guide; `AGENTS.md` is a verbatim twin of this file. Shared top-level
package, like `affine.py`: it belongs to neither `linear/` nor `nonlinear/`
and imports neither. It is the engine a future agent tool (`fit_deformable`)
will call; no tool, CLI or host wiring exists yet.

## What it does

Given a section image, a linear placement (`geometry.Placement`: atlas name,
position, plane, pitch/yaw, the 3x3 `atlas_to_section` map from native
atlas-plane pixel centres to section pixel centres, section mm/px) and one
`settings.FitSettings`, it fits a residual deformation with a LIBRARY engine
and returns a `record.DeformableRecord`. No custom solver.

- Route A (no image model): the stain image (`section_image="stain"`) against
  an atlas image — `ara` (BrainGlobe reference), or `nissl` (ABBA's cached
  Allen Nissl, ABBA hosts only, `atlas_images_for_host`). Mutual information.
  Optionally SEQUENTIAL: each call with `structures=(...)` (acronyms/ids,
  descendants included) restricts both masks to those structures plus
  `neighbourhood_um`, and with `previous=record` composes onto that record;
  `record.parent` / `record.undo()` is the step before.
- Route B (image model): the model's extracted line mask (`lines=`, on the
  section grid, e.g. `trace_borders`' `extracted_lines.png` with its
  `atlas_to_canvas`) against `borders_merged` (the same color-family set the
  model was shown, `atlas.render.family_labels`) or `borders`; both softened
  by the same Gaussian ridge (`line_softening_um`). Mean squares.
- Label-map channels (`labels=`, ANTs only): `model` names each area the
  model's lines enclose after the placed merged region it overlaps most
  (`regions.named_regions`, small loops around bubbles dropped, joint
  one-to-one renaming) and pairs each region's indicator with the atlas's;
  `auto` adds the stain's tissue footprint and empty interior holes near
  placed ventricles. Built as `ants.registration` `multivariate_extras` (the
  same per-label MeanSquares construction `ants.label_image_registration`
  uses, which hard-codes `SyN[0.2,3,0]` and so would ignore stiffness).
- Engines (`engines.py`): ANTs `SyNOnly` with an identity initial transform
  (antspyx, optional `registration` extra, imported lazily with an install
  hint), or Elastix B-spline with a bending-energy penalty (itk-elastix,
  core dependency). Both see the same working-grid images, masks, spacing and
  origin and return millimetre fields. ANTs supplies its own inverse; Elastix
  has none, so its inverse is a fixed-point approximation, said so in
  `inverse_source` and `engine["notes"]`.
- Optional ANTs preprocessing of the stain (`preprocess=("n4", "denoise")`).
- `fit_candidates` runs 1–8 settings in a spawn process pool (atlas work in
  the caller, engine calls in workers); a failing candidate is a
  `CandidateFailure`, never an exception.

## Settings (`settings.py`)

Named ladders, physical units: `stiffness` soft/medium/firm/stiff (ANTs
update/total Gaussian sigma in mm, converted to voxel variances; Elastix
control spacing in mm and bending weight, scaled by
`ELASTIX_MEAN_SQUARES_BENDING_SCALE` for mean squares); `detail`
coarse/standard/fine (working µm and pyramid depth/iterations — the extra
ANTs levels are its capture range); `atlas_image`, `section_image`,
`line_softening_um` (default 60; flat between 40 and 150 µm on synthetic
lines, 20 µm worse — the ceiling test decides), `exclude`, `labels`,
`structures`, `neighbourhood_um`, `preprocess`. Metric and step size are
fixed per pairing (`metric_for`); lines against `ara`/`nissl` is refused.

## Masks (`masks.py`)

Fixed: tissue (`image_prep.foreground_mask` at a 1024-px proxy, holes
filled) widened by `MASK_MARGIN_MM`, minus the torn-edge band. Default torn
rule: tissue outline lying more than `TORN_EDGE_DEPTH_MM` inside the placed,
non-excluded atlas footprint, widened by `TORN_EDGE_BAND_MM`; `torn_band=`
overrides it. Moving: placed footprint widened by the margin, minus excluded
regions and `EXCLUSION_MARGIN_MM` around them (blanking alone left an edge
that still pulled tissue). Excluded regions are blanked on the native plane
for EVERY atlas image kind before placement.

## The record (`record.py`)

`field_mm[y, x]` = [dx, dy] mm: section point (x, y)·mm_per_px corresponds to
placed-atlas point + field (brainglobe-registration's direction); native =
inv(atlas_to_section) @ (pixel + field/mm_per_px), the same order as
`border_registration.composed_native_map`. A sequential record's field is
the total. Also: inverse field (or None), engine name/version/runtime/native
parameters, working grid, warped leaf labels CLIPPED TO TISSUE, tissue and
torn-band masks, excluded ids, `native_to_volume_index` (plane pixel → atlas
volume index, so exports need no atlas object), `native_coordinates()`,
`volume_coordinates()`, `save()`/`load()` (parent chain under `parent/`).

Diagnostics are reported, never enforced: per-region area ratio
warped/placed from the Jacobian (`N_R / Σ_{p∈R} J`), raster ratio, fold
fraction in tissue, inverse consistency, and flags REGION_COMPRESSED /
EXPANDED / VANISHED / FOLDED and FOLDS. Limits are the named constants
`TISSUE_AREA_RATIO_LIMITS` (0.5, 2), `VENTRICLE_AREA_RATIO_LIMITS` (0.1, 10)
for `VS` and descendants, `MIN_FLAG_AREA_MM2`, `FOLD_FRACTION_LIMIT`.

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
included) draws those regions' edges strongly over a faint outline of the
colour-family regions. Never judge borders traced from `record.labels`: those
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

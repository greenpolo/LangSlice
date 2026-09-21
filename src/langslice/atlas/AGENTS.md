# LangSlice `atlas/ — atlas access and colored region maps`

Package guide for `src/langslice/atlas/`. The repo-level `CLAUDE.md` holds the
project-wide rules; this file holds what is specific to this package. `AGENTS.md`
here is a verbatim copy — edit one, mirror to the other.

- `atlas/` — BrainGlobe loading, position helpers, slice extraction, colored
  region maps, borders.
  `render.py` is the geometry BOTH methods draw from, and it belongs to
  neither: `annotation_slice` (the display-oriented annotation at a position,
  resliced obliquely when the block carries cutting angles), `atlas_um_per_px`,
  `family_mapping` (region id -> its merged color family's representative),
  `region_contours` + `_smooth_closed` (smoothed per-region polygons, holes
  included, confetti dropped), `family_outlines` (one `(family color,
  polyline)` per family region, in atlas-native pixels), and the shade rules
  `border_color` / `darker` / `BORDER_DARKEN` / `is_dark_background`. It moved
  here from `nonlinear/render.py` and `nonlinear/image_gen_helpers.py` when
  `linear`'s physical overlay started drawing the same lines: two methods
  disagreeing about where a boundary is would be a bug neither could see.
  `nonlinear.render` re-exports the shade rules and `region_contours`, so it
  stays the one import for review rendering.
  Also `recolor.py`: organized structure colors for atlases whose native
  palettes are visually confusing. Nonlinear's two border-based routes never
  show an image-generation model any colored atlas render — the model-facing
  atlas reference on route "atlas" is a grayscale plate with thin yellow
  boundaries only (`nonlinear.image_gen_registration.outlined_atlas_template`).
  This LUT still colors renders people (not models) look at: the atlas
  package's own colored region maps, linear's physical overlay, and
  nonlinear's human-review warped-atlas overlay
  (`nonlinear.image_gen_helpers._classified_to_rgb`). `color_lut(atlas)` keeps native colors
  when they are hierarchy-organized, joins the true Allen CCF colors
  (vendored `allen_colors.json`) for trees with
  enough Allen overlap — the all-white atlases (Osten, Princeton, adult Kim)
  and the Waxholm rat family. The join matches by normalized name, a
  terminology bridge (classical/embryological vocabulary onto Allen's, e.g.
  mesencephalon→Midbrain, white matter→fiber tracts), name variants, and
  acronym only when the names corroborate it; unmatched subregions inherit
  their deepest matched ancestor's color (the root never joins or seeds
  inheritance). It generates an Allen-style palette for
  foreign trees: nested hue subdivision (each division's hue range is
  proportional to its subtree), quantized to at most `MAX_FAMILIES` flat
  family colors, because few flat colors IS the Allen convention. Auto-
  detection is data-driven — degenerate = one color for
  everything; disorganized = child colors uncorrelated with parents — and
  flat or small trees always keep native colors. `atlas/core.py`'s colored
  region-map render and nonlinear's human-review warped-atlas render
  (`nonlinear/image_gen_helpers.py`'s `_classified_to_rgb`) both draw from
  this one LUT; border fitting warps the existing label map by
  nearest-neighbor sampling and never classifies model-output pixels.
  A DERIVED palette (Allen join or generated) is then organized: leaf-level
  Allen colors are not a usable palette — Allen encodes hierarchy in hue, so
  a cortex-dominated slice came out as a dozen near-identical greens the
  pipeline merges away anyway (that was the WHS rat "washed out" bug). Colors
  closer than `MERGE_EPS` (40, `_family_mapping`'s own radius) collapse into
  one unit, and the surviving units are pulled at least `MIN_SEPARATION` (60)
  apart, nudging value/saturation before hue so a family keeps its identity.
  Native palettes are left exactly as the atlas authored them.
  `color_lut` has ONE table and no modes. Region fills are outlined by a
  line in a darker shade of the region's own color (`render.darker`, RGB ×
  `BORDER_DARKEN` = 0.7, which moves HSV value only, so hue and saturation
  still name the region); lines are drawn LINE_8 like the fills, on the
  smoothed sub-pixel contours, so no pixel blends two region colors. These
  renders are for people (this package's own colored region maps, linear's
  overlay, nonlinear's review-only warped-atlas render); no model is shown
  any of them.

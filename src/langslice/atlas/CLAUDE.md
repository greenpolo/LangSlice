# LangSlice `atlas/ — atlas access and colored region maps`

Package guide for `src/langslice/atlas/`. The repo-level `CLAUDE.md` holds the
project-wide rules; this file holds what is specific to this package. `AGENTS.md`
here is a verbatim copy — edit one, mirror to the other.

- `atlas/` — BrainGlobe loading, position helpers, slice extraction, colored
  region maps, borders.
  Also `recolor.py`: organized structure colors for atlases whose native
  palettes mislead image-gen models. `color_lut(atlas)` keeps native colors
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
  flat or small trees always keep native colors. Both colored-region-map
  paths (`atlas/core.py` and `nonlinear/image_gen_helpers.py`, render and
  pixel-classify alike) draw from this one LUT.
  A DERIVED palette (Allen join or generated) is then organized: leaf-level
  Allen colors are not a usable palette — Allen encodes hierarchy in hue, so
  a cortex-dominated slice came out as a dozen near-identical greens the
  pipeline merges away anyway (that was the WHS rat "washed out" bug). Colors
  closer than `MERGE_EPS` (40, `_family_mapping`'s own radius) collapse into
  one unit, and the surviving units are pulled at least `MIN_SEPARATION` (60)
  apart, nudging value/saturation before hue so a family keeps its identity.
  Native palettes are left exactly as the atlas authored them.
  `color_lut` has ONE table and no modes: the `--palette` knob picks a
  model-facing render STYLE, never a color. `"family"` (default) draws flat
  regions, one color per registration unit; `"leaf-borders"` draws the same
  flat colors plus the Allen-Reference-Atlas plate treatment — every leaf
  boundary delineated by a hairline in a darker shade of the region's own
  color (`render.darker`, RGB × `BORDER_DARKEN` = 0.7, which moves HSV value
  only, so hue and saturation still name the region), 2px at a 2048 canvas
  with family boundaries at 1.8× that. Lines are drawn LINE_8 like the fills,
  on the smoothed sub-pixel contours: an anti-aliased line would blend two
  region colors into pixels belonging to neither, and the render has to stay
  classifiable to exact palette colors. Leaf shades were tried here first and
  rejected (Nash: "everything is uniformly worse in our colors") — borders
  express leaves without touching the palette. A third style,
  `"family-flat"`, is an EXPERIMENTAL arm and not the default: it collapses
  the leaves onto their registration families before painting
  (`image_gen_helpers._plane_families`, exactly the `_family_mapping`
  partition Elastix and the benchmark already score at) and delineates that
  flat paint with the same darker hairlines, so the model is shown the
  granularity it is graded on and nothing finer. Its classifier maps each
  family color and its line shade to the family's representative id, which
  `_merge_classified` keeps unchanged — measured on Allen coronal 3.9 and
  9.0 mm, render→classify→merge lands on the SAME map as `"family"` to the
  pixel, and the family colors on those planes stay 40.6 / 42.1 RGB apart.
  The style is a PROCESS-WIDE
  setting (`LANGSLICE_ATLAS_PALETTE`, `active_palette()`, `use_palette()`,
  and `langslice nonlinear register --palette`), because the classifier has
  to know whether hairlines were painted (see the nonlinear section);
  `generate_registration_candidate(palette=...)` applies it around one call

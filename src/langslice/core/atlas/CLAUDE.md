# LangSlice `core/atlas/` — atlas access and colored region maps

Package guide for `src/langslice/core/atlas/`. The repo-level `CLAUDE.md` holds the
project-wide rules; this file holds what is specific to this package. `AGENTS.md`
here is a verbatim copy — edit one, mirror to the other.

- `core.py` — BrainGlobe loading (`load_atlas`, `canonicalize_atlas_name`),
  position helpers (`position_mm_to_index`, `index_to_position_mm`,
  `get_position_range_mm`), the reference plate (`get_reference_slice`) and
  `get_root_mask`. The public surface is deliberately small: what LangSlice
  itself reads.
- `sides.py` — one side of a region, `"CTX:left"`: the grammar
  (`split_side`, `overlapping`), the native plane's two ML halves
  (`ml_halves`: ML volume index per pixel from `oblique.plane_index_coordinates`
  on the axis `space.atlas_space_context` derives; split where BrainGlobe
  splits a symmetric atlas, at `round(n_ml / 2)`, or by the atlas's own
  `hemispheres` volume when its metadata says it is not symmetric) and
  `native_left` (which half the placement puts on the DISPLAYED section's
  left: the ML gradient carried through the placement's linear part; a
  mirror swaps the halves; `SideError` `NO_SIDES` on sagittal planes,
  `SIDES_AMBIGUOUS` when the midline is turned past 45 degrees). BrainGlobe's
  own "left"/"right" hemisphere values are never used: its `hemispheres`
  docstring and code disagree, and on a symmetric atlas no label agreement
  could settle which is which.
- `render.py` — the geometry both methods draw from, belonging to neither:
  `annotation_slice` (the display-oriented annotation at a position,
  resliced obliquely when the block carries cutting angles), `atlas_um_per_px`,
  `family_mapping` (region id -> its merged color family's representative),
  `family_labels` (a label map rewritten to those representatives),
  `placed_border_coverage` (antialiased placed boundaries; the image tool's
  overlay and the deformable fit's border images), `region_contours` +
  `_smooth_closed` (smoothed per-region polygons, holes included, confetti
  dropped), `family_outlines` / `outer_outline` (one `(family color,
  polyline)` per family region, in atlas-native pixels), and the shade rules
  `border_color` / `is_dark_background`. Every picture draws its boundaries
  from these, so two views never disagree about where a boundary is.
- `recolor.py` — `color_lut(atlas)`: organized structure colors for atlases
  whose native palettes are visually confusing. Its consumer is the family
  grouping (`render.family_mapping`) behind the placement overlay's outlines,
  whose lines are drawn in one color (yellow by default); no model is shown a
  colored atlas render. Native colors are kept when they are
  hierarchy-organized. Otherwise the true Allen CCF colors (vendored
  `allen_colors.json`) are joined for trees with enough Allen overlap (the
  all-white atlases such as Osten, Princeton and adult Kim, and the Waxholm
  rat family): by normalized name, a terminology bridge (classical or
  embryological vocabulary onto Allen's, e.g. mesencephalon -> Midbrain,
  white matter -> fiber tracts), name variants, and acronym only when the
  names corroborate it; unmatched subregions inherit their deepest matched
  ancestor's color (the root never joins or seeds inheritance). Foreign trees
  get a generated Allen-style palette: nested hue subdivision (each
  division's hue range proportional to its subtree), quantized to at most
  `MAX_FAMILIES` flat family colors, because few flat colors is the Allen
  convention. Auto-detection is data-driven (degenerate = one color for
  everything; disorganized = child colors uncorrelated with parents), and
  flat or small trees keep native colors. A derived palette is then
  organized: colors closer than `MERGE_EPS` (40, `family_mapping`'s own
  radius) collapse into one unit, and the surviving units are pulled at least
  `MIN_SEPARATION` (60) apart, nudging value/saturation before hue so a
  family keeps its identity. Native palettes are left exactly as the atlas
  authored them.

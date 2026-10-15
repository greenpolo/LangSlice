# LangSlice `core/atlas/` — atlas access and colored region maps

Package guide for `src/langslice/core/atlas/`. The repo-level `CLAUDE.md` holds the
project-wide rules; this file holds what is specific to this package. `AGENTS.md`
here is a verbatim copy — edit one, mirror to the other.

- `core.py` — BrainGlobe loading (`load_atlas`, `canonicalize_atlas_name`;
  BrainGlobe's notices, such as its atlas-version check, printed on stderr,
  never the caller's stdout; with `LANGSLICE_ATLAS_SKIP_LATEST_CHECK` set,
  offline: only an atlas already in the local cache loads, in either
  brainglobe-atlasapi layout, `<name>_v*/` or 3.x's
  `brainglobe-atlasapi/atlases/<name>/`),
  position helpers (`position_mm_to_index`, `index_to_position_mm`,
  `get_position_range_mm`), the reference plate (`get_reference_slice`) and
  `get_root_mask`. The public surface is deliberately small: what LangSlice
  itself reads.
- `disk_cache.py` — volumes derived from an atlas, kept on disk between
  processes (`docs/architecture.md`, "Atlas cache"): `cached_volume(atlas,
  name, build)` reads `<cache>/<atlas>_v<version>/format<FORMAT>/<name>.npy`
  memory-mapped, else builds it and saves it under a temporary name renamed
  into place; `cache_root` (`LANGSLICE_ATLAS_CACHE`, `off`, else
  `~/.langslice/atlas-cache`), `atlas_folder` (None without an atlas version
  or past `MAX_VOXELS`). A write or read failure falls back to building in
  memory. `core.oblique` caches the template's spline coefficients and the
  annotation's dense indices through it; the tests turn it off
  (`tests/conftest.py`).
- `images.py` — the atlas's own grayscale images (`atlas_images`,
  `atlas_volume`): BrainGlobe's `template` and every additional reference the
  atlas ships, under BrainGlobe's names, plus `ccfv3augmented_nissl` for an
  Allen CCFv3 atlas (no Nissl of its own): the Blue Brain Nissl template of
  `ccfv3augmented_mouse_25um` resampled onto its grid.
- `sides.py` — one side of a region, `"CTX:left"`: the grammar
  (`split_side`), the native plane's two ML halves
  (`ml_halves`: ML volume index per pixel from `oblique.plane_index_coordinates`
  on the axis `space.atlas_space_context` derives; split where BrainGlobe
  splits a symmetric atlas, at `round(n_ml / 2)`, or by the atlas's own
  `hemispheres` volume when its metadata says it is not symmetric) and
  `native_left` (a `SideMask`: which half the placement puts on the
  DISPLAYED section's left, or its top when the midline runs closer to
  horizontal there, `names` `LEFT_RIGHT` or `TOP_BOTTOM`: the ML gradient
  carried through the placement's linear part; a mirror swaps the halves;
  `SideError` `NO_SIDES` on sagittal planes, `SIDES_AMBIGUOUS` at 45
  degrees (within `DIAGONAL_TOLERANCE`), `WRONG_SIDE_PAIR` from `restrict` for
  a side outside the picture's pair, its message naming the pair). BrainGlobe's
  own "left"/"right" hemisphere values are never used: its `hemispheres`
  docstring and code disagree, and on a symmetric atlas no label agreement
  could settle which is which.
- `render.py` — the geometry both methods draw from, belonging to neither:
  `annotation_slice` (the display-oriented annotation at a position,
  resliced obliquely when the block carries cutting angles), `atlas_um_per_px`,
  `region_boundaries` (where neighbouring pixels of a label map differ: the
  one boundary computation every pixel boundary picture uses) and
  `stroke_boundaries` (those edges widened and area-averaged into antialiased
  coverage), `placed_border_coverage` (antialiased placed boundaries; the
  image tool's overlay and the deformable fit's border images), `region_contours` +
  `_smooth_closed` (smoothed per-region polygons, holes included, confetti
  dropped; each outline runs along the pixel edges its region shares with
  its neighbours, `_on_pixel_edges`, not through its boundary pixels'
  centres, so a border two regions share is one line), `region_outlines` (the polylines of every region at a
  depth, `levels.labels_at`), in
  atlas-native pixels, and the shade rule
  `is_dark_background`. Every picture draws its boundaries
  from these, so two views never disagree about where a boundary is.
- `tree.py` — the structure table and walks over its hierarchy, the one
  reader of `atlas.structures`: `structure_table` (`{id: record}`, records
  read in place, never copied, so BrainGlobe never loads a lazy mesh), `path`,
  `ancestry`, `ancestor_at`, `descendants` and `with_descendants`.
- `levels.py` — `labels_at(labels, atlas, depth)`: each id mapped to its
  ancestor at that depth of BrainGlobe's `structure_id_path` (root = 0; an id
  with a shorter path stays itself, 0 stays 0), cached per atlas and depth.
  The one region grouping for every atlas: agent overlays draw contours at
  `OVERLAY_DEPTH` (6; `look`'s `depth`), the image model is shown and traces
  regions at `MODEL_DEPTH` (6, fixed). Atlas colours are BrainGlobe's own
  `rgb_triplet` (the exported label table, `job.formats.write_labels`);
  every picture draws its lines in one colour.

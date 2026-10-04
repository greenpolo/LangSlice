# Job folder file formats

The public files a LangSlice job folder (`<images>/langslice/`) holds for
scripts and other programs. `state.json` is the job's one working source and
the truth; every file below is DERIVED from it and never read back as input.
Change a registration through the verbs (`langslice job FOLDER VERB`, or
`langslice.open_job(...)` in Python), never by editing these files.

Code: `src/langslice/core/maps.py` (the geometry), `src/langslice/job/formats.py`
(the files), `src/langslice/ops/exports.py` (the `export_maps` verb),
`src/langslice/integrations/quint.py` (QuickNII / VisuAlign JSON).

## Conventions

- **Atlas coordinates** are BrainGlobe micrometres in the atlas's own axis
  order, given by its orientation string (`atlas.orientation`; every packaged
  BrainGlobe atlas is `asr`: axis 0 anterior to posterior, axis 1 superior to
  inferior, axis 2 right to left). Voxel `i`'s CENTRE is at
  `i * resolution_um`, so an atlas id is read at
  `annotation[round(x0 / r0), round(x1 / r1), round(x2 / r2)]`.
- **Image pixels** are `[row, col]` of the image FILE as stored (no rotation or
  flip applied: those are part of the mapping), pixel centres at integers,
  row 0 at the top.
- A **flat plane** (cutting angles 0/0) lies on the atlas slab nearest its
  position, exactly as every picture draws it, so the slicing-axis coordinate
  is a multiple of the resolution. With cutting angles the plane is oblique
  and continuous.
- A section without an in-plane transform is placed by the identity (as its
  pictures show it); `registration.json` says so (`mapping`).

## `registration.json` (job folder root)

Rewritten atomically on every write to the state (with every checkpoint).
Format 1.

| Field | Meaning |
|---|---|
| `format_version` | 1 |
| `written_by` | the LangSlice version |
| `note`, `convention` | what the file is; the conventions above, in words |
| `atlas` | `name`, `version`, `orientation`, `shape` (voxels per axis), `resolution_um` (per axis) |
| `plane` | `coronal`, `sagittal` or `horizontal` |
| `cutting_angles_deg` | the stack's `pitch` and `yaw` |
| `image_folder` | the images' folder (absolute) |
| `submitted` | whether the run was submitted |
| `sections` | one entry per section, in corrected order (below) |
| `exports` | relative paths of `exports/quicknii.json` and `exports/visualign.json` when written |

Each section:

| Field | Meaning |
|---|---|
| `id` | the image filename |
| `order` | its corrected index |
| `folder` | its folder, relative to the job folder (`sections/<name>`) |
| `parameters` | THE REGISTRATION, in public units: `atlas` (`name`, `version`); `plane` (`name`, `position_mm`, `position_um`, `pitch_deg`, `yaw_deg`); `orientation` (`rotation_deg`: counter-clockwise quarter turns, applied first; `flip`: left-right, applied after); `affine` (null without a transform; else `kind`, `params`: the six normalized numbers `[a, b, tx, c, d, ty]` on the oriented section render, x as a fraction of its width and y of its height; `physical`: rotation, scales, shear, translations in mm, pivot; `mirrored`); `deformation` (null; `{"kind": "none", "reason"}` when the linear placement was kept; or `{"kind": "residual", "record", "key", "steps", "inverse_source"}`, `record` being the deformation record's folder relative to the job folder); `damaged` |
| `image` | `file`, `size` (`[width, height]` of the file), `pixel_size_um` (of the file), `pixel_size_source` (`host`, `file` or `stored`) |
| `pixel_to_atlas_um` | 3x3: a file pixel `[row, col, 1]` -> atlas micrometres (3 rows, the atlas axes), the LINEAR placement; null without a placement |
| `mapping` | `linear`, `linear (identity in-plane: no transform written)`, `linear + residual` (the maps hold the complete mapping), or null |
| `problem` | why there is no matrix (`no position`, an unreadable file, ...), else null |
| `parameters_digest` | SHA-256 of the parameters, the matrix and the image facts |
| `maps` | null before any maps were written; else `files` (kind -> relative path), `grid_size`, `full_resolution`, the grid's `pixel_to_atlas_um`, and `current` (false once the section changed after the maps were written) |

## Per-section maps (`sections/<name>/`)

Written at `submit` and by the `export_maps` verb (CLI and library only), for
every section with a position. Not on every write (they are large). The grid
is the section's WORKING COPY (the image every picture is drawn from: a
whole-slide TIFF's smallest pyramid level of at least 1536 px, otherwise the
file downsampled to at most 3072 px), or with `full_resolution` the file's
own pixels. A grid pixel `[row, col]` maps to the file by pixel centres
(`langslice.affine.pixel_center_map`); `maps.json` gives the grid's own
`pixel_to_atlas_um`.

"Tissue" is the deformable fit's tissue rule (`deformable.masks.tissue_masks`:
foreground against the slide background, interior holes filled) on the
working copy. A tear or a gap open to the outside is not tissue.

| File | Content |
|---|---|
| `coords.tif` | float32, ImageJ hyperstack, axes CYX, shape `(3, rows, cols)`: channel k is atlas axis k in micrometres. NaN outside the tissue and outside the atlas volume. Calibrated in µm per grid pixel; the channel labels are `axis0_um`..`axis2_um`; the ImageJ Info property holds the grid facts as JSON. |
| `labels.tif` | uint32, `(rows, cols)`: the atlas annotation id under each pixel (nearest neighbour on the atlas plane, as the pictures' labels layer and the deformable record do), 0 where `coords.tif` is NaN and where the atlas has no region. Deflate with the horizontal predictor; not an ImageJ file (Fiji turns uint32 into float32, which loses the larger Allen ids). |
| `labels_fiji.tif` | uint16, ImageJ, `(rows, cols)`: a dense index of the section's regions (1..n in the order of their atlas ids, 0 none), with an ImageJ lookup table giving each index its atlas colour (display range 0-255; a section with more than 255 regions shows the rest without their colour). Opens in Fiji as is. |
| `labels.csv` | `index,id,acronym,name,r,g,b`: the dense index -> atlas id, acronym, name and RGB colour (the atlas structure tree's). |
| `residual.tif` | only with an applied deformation. float32, ImageJ, axes CYX, `(2, rows, cols)`: `(d_row, d_col)` in grid pixels, defined on the whole grid, such that `atlas_um = pixel_to_atlas_um @ [row + d_row, col + d_col, 1]` (`maps.json`'s matrix). Its source is the deformation record named in `registration.json`. |
| `maps.json` | `format_version` 1, `section`, `grid_size` (`[width, height]`), `full_resolution`, `um_per_px`, `pixel_to_atlas_um` (the grid's), `parameters_digest` (what they were written from; `registration.json` compares it), `tissue_found`, `deformation_record`, `files`. |

Float maps are deflate-compressed WITHOUT TIFF's floating-point predictor:
ImageJ 1.x cannot decode it. Measured 2026-10-04 on LSD_910 M02 (whole-slide
TIFFs, 12540 x 8417 px, working copy about 3072 x 2060): `coords.tif` 26-28
MB, `residual.tif` 46 MB, `labels.tif` 0.2 MB, `labels_fiji.tif` 0.1 MB per
section, about 0.6 s per section; at full resolution `coords.tif` 426 MB,
`residual.tif` 750 MB, 13 s.

Agreement (tests/test_job_formats.py, synthetic atlas): `labels.tif` equals
the atlas read at `coords.tif` on over 99% of tissue pixels (exact but for
ties at half voxels); `coords = matrix @ (p + residual)` to 0.01 µm; a
placement picture's coordinate map and `registration.json`'s matrix agree to
0.05 µm; a `fit_deformable` picture's (its residual layer included) and the
section maps' to under 1 µm on a 31 µm/px section.

## Pictures

Each picture a door showed is saved under `views/` (`job/views.py`); a
placement picture and, since this phase, a `fit_deformable` picture carry
`labels.tif`, `borders.png` and a frame in `view.json`.
`langslice.coordinate_map(".../view.json")` gives every pixel's atlas
micrometres. A `fit_deformable` picture showing its deformation adds
`residual.tif` (float32, `(2, rows, cols)`, `(d_row, d_col)` in picture
pixels, zero in the caption band); its frame's `residual` names it, and
`coordinate_map` applies it.

## Exports (`exports/`)

Written with the maps. Both are QuickNII-format JSON (`name`, `target`,
`aligner`, `slices`: `filename`, `anchoring`, `width`, `height`, `nr`,
`markers`), one slice per placed section, `width`/`height` the image file's
pixels and `nr` the corrected index + 1.

- `quicknii.json`: the linear anchoring of every section, `markers` empty.
  The anchoring `o, u, v` is taken from the section's exact
  `pixel_to_atlas_um` (any plane, any cutting angle): the image's top-left
  corner and its two corner-to-corner vectors, converted to QuickNII voxel
  space (x left to right, y posterior to anterior, z inferior to superior;
  voxel edges) through `brainglobe_space`. The conversion is SliceBench's,
  checked there against DeepSlice and human QUINT registrations on the Allen
  CCFv3 25 µm atlas; for other atlases the QuickNII target is assumed to
  share the BrainGlobe voxel grid. `target` is the `.cutlas` name for known
  atlases.
- `visualign.json`: the same, plus VisuAlign markers of each applied
  deformation: `[x, y, nx, ny]` on a regular grid (1/36 of the long edge, at
  least 32 px), in the image's continuous pixel coordinates (top-left corner
  at 0). A section point `(nx, ny)` corresponds to the point `(x, y)` of the
  linearly anchored plane: the direction of VisuAlign's triangulation as the
  QUINT team's reference code reads it (`triangulate` on `(nx, ny)`, mapped to
  `(x, y)`). Each marker reproduces the section's maps exactly at its own
  point (tests). Not yet opened in VisuAlign itself.

## Edited maps and labels

An edited `coords.tif`, `labels.tif` or painted label image changes no
registration: nothing reads them back. Bringing one in needs a fit that takes
it as its target, e.g. a `fit_deformable` section input that reads a label
image from a file and pairs it with the atlas regions (as `traced_borders`
does with the image model's lines); that input does not exist yet.

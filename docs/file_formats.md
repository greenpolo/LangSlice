# Job folder file formats

A job stores settings in `job.json` and its working state in `state.json`.
`registration.json`, maps and pictures are derived outputs. Change the job
through its tools; editing an output does not change the registration.
A registration file can also be explicitly imported when creating a job.

## The job folder

```text
<images>/langslice/
  job.json              settings (the JobSpec) and the format version
  state.json            the checkpoint: the truth
  registration.json     its public rendering
  history/              undo and redo steps
  sections/<name>/      per section (<name>: the filename's stem): its maps,
                        applied deformation records (deformable/),
                        image-model traces (image_correction/), its pictures (views/)
  views/                pictures of several sections
  views.jsonl           the picture index: number, tool, history step, sections, caption
  exports/              linear_results.json, quicknii.json, visualign.json
  logs/                 the agent CLI's calls and background runs
  AGENTS.md, CLAUDE.md  the reference card for coding agents
  BRIEF.md              the agent CLI's brief
```

The job folder is `<images>/langslice/` unless `--job-dir` says otherwise (a
read-only image folder falls back to `~/.langslice/jobs/<id>/`). Output artifact
paths are relative to the job folder. The default image folder reference is
`..`; a separately located job records an absolute image path.

## Conventions

- **Atlas coordinates** are BrainGlobe micrometres in the atlas's own axis
  order, given by its orientation string (`atlas.orientation`). For `asr`,
  axis 0 runs anterior to posterior, axis 1 superior to inferior, and axis 2
  right to left. Voxel `i`'s centre is at
  `i * resolution_um`, so an atlas id is read at
  `annotation[round(x0 / r0), round(x1 / r1), round(x2 / r2)]`.
- **Image pixels** are `[row, col]` of the image file as stored (no rotation or
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
| `cutting_angles_deg` | the stack's `pitch` and `yaw`; null when the sections' angles differ (a registration supplied per section), each section's `parameters.plane` then giving its own |
| `image_folder` | the images' folder: `".."` (the job folder's parent) for the default job folder beside the images, so the file stays right when the folder moves with them; the absolute path for an explicit job folder |
| `submitted` | whether the run was submitted |
| `sections` | one entry per section, in the stack's order (below) |
| `exports` | relative paths of `exports/quicknii.json` and `exports/visualign.json` when written |

Each section:

| Field | Meaning |
|---|---|
| `id` | the image filename |
| `order` | its index in the stack's order |
| `folder` | its folder, relative to the job folder (`sections/<name>`) |
| `parameters` | Registration settings in public units (below) |
| `image` | `file`, `size` (`[width, height]` of the file), `pixel_size_um` (of the file), `pixel_size_source` (`host`, `file`, or `estimated`: neither gives one, so the scale is the one every placement picture draws the section at, estimated from its tissue width at its current position) |
| `pixel_to_atlas_um` | 3x3: a file pixel `[row, col, 1]` -> atlas micrometres (3 rows, the atlas axes), the linear placement; null without a placement |
| `mapping` | `linear`, `linear (identity in-plane: no transform written)`, `linear + residual` (the maps hold the complete mapping), or null |
| `problem` | why there is no matrix (`no position`, an unreadable file, ...), or, with a matrix, that the pixel size is unknown (the scale `estimated`, as the pictures draw it); else null |
| `parameters_digest` | SHA-256 of the parameters, the matrix and the image facts |
| `maps` | null before any maps were written; else `files` (kind -> relative path), `grid_size`, `full_resolution`, the grid's `pixel_to_atlas_um`, and `current` (false once the section changed after the maps were written) |

### Section parameters

| Key under `parameters` | Content |
|---|---|
| `atlas` | `name`, `version` |
| `plane` | `name`, `position_mm`, `position_um`, `pitch_deg`, `yaw_deg` |
| `orientation` | Counter-clockwise quarter turn `rotation_deg`, followed by left-right `flip` |
| `affine` | Null without a transform; otherwise `kind`, `params`, `physical` and `mirrored` |
| `deformation` | Null, `{"kind":"none","reason":"..."}` for a section left linear, or a residual record with `kind`, `record`, `key`, `steps`, `inverse_source` |
| `damaged_regions` | Atlas regions excluded from fits, optionally with a side (`"CTX:left"`) |
| `damaged` | True exactly when `damaged_regions` is nonempty |

Affine `params` are `[a,b,tx,c,d,ty]` on the oriented section render:
`x' = a*x + b*y + tx`, `y' = c*x + d*y + ty`, with x normalized by width
and y by height. `physical` records rotation, scales, shear, translations
in millimetres and pivot. A residual's `record` is its folder relative to
the job folder; coordinate maps include that deformation as well as the affine.

## Per-section maps (`sections/<name>/`)

Written at `submit` and by the `export_maps` verb (CLI and library only), for
every section with a position. Not on every write (they are large). The grid
is the section's working copy (a whole-slide TIFF's smallest pyramid level of at least 1536 px, otherwise the
file downsampled to at most 3072 px), or with `full_resolution` the file's
own pixels. A grid pixel `[row, col]` maps to the file by pixel centres
(`langslice.core.affine.pixel_center_map`); `maps.json` gives the grid's own
`pixel_to_atlas_um`.

The maps cover the section's footprint, its filled outline: the deformable
fit's foreground rule (`core.deformable.masks.tissue_masks`, against the slide
background) on the working copy, closed over gaps up to 0.3 mm wide
(`core.maps.FOOTPRINT_CLOSING_MM` 0.15 mm, a radius) and with every hole
filled. Nothing inside the outline is cut: dim fibre tracts, enlarged
ventricles, tears and fissures all get coordinates and labels; only the
slide outside it is NaN / 0. The threshold's own tissue estimate is written
beside the maps as `tissue.png`, for a script that wants to mask with it.

| File | Content |
|---|---|
| `coords.tif` | float32, ImageJ hyperstack, axes CYX, shape `(3, rows, cols)`: channel k is atlas axis k in micrometres. NaN outside the footprint and outside the atlas volume. Calibrated in µm per grid pixel; the channel labels are `axis0_um`..`axis2_um`; the ImageJ Info property holds the grid facts as JSON. |
| `labels.tif` | uint32, `(rows, cols)`: the atlas annotation id under each pixel (nearest neighbour on the atlas plane, as the pictures' labels layer and the deformable record do), 0 where `coords.tif` is NaN and where the atlas has no region. Deflate with the horizontal predictor; not an ImageJ file (Fiji turns uint32 into float32, which loses the larger Allen ids). |
| `tissue.png` | uint8, `(rows, cols)`: 255 where the foreground rule finds tissue, 0 elsewhere (the raw rule: no hole filling, no closing, so dim tissue and gaps inside the footprint may be 0). Not used to cut the maps. |
| `labels_fiji.tif` | uint16, ImageJ, `(rows, cols)`: a dense index of the section's regions (1..n in the order of their atlas ids, 0 none), with an ImageJ lookup table giving each index its atlas colour (display range 0-255; a section with more than 255 regions shows the rest without their colour). Opens in Fiji as is. |
| `labels.csv` | `index,id,acronym,name,r,g,b`: the dense index -> atlas id, acronym, name and RGB colour (the atlas structure tree's). |
| `residual.tif` | only with an applied deformation. float32, ImageJ, axes CYX, `(2, rows, cols)`: `(d_row, d_col)` in grid pixels, defined on the whole grid, such that `atlas_um = pixel_to_atlas_um @ [row + d_row, col + d_col, 1]` (`maps.json`'s matrix). Its source is the deformation record named in `registration.json`. |
| `maps.json` | `format_version` 1, `section`, `grid_size` (`[width, height]`), `full_resolution`, `um_per_px`, `pixel_to_atlas_um` (the grid's), `parameters_digest` (what they were written from; `registration.json` compares it), `tissue_found`, `footprint_fraction` and `tissue_fraction` (of the grid), `deformation_record`, `files`. |

Float maps are deflate-compressed without TIFF's floating-point predictor,
which ImageJ 1.x cannot decode. At working size a section's `coords.tif` is
tens of MB; at full resolution of a whole-slide scan, hundreds.

## Pictures

Each picture a door showed is saved as a folder `<number>_<tool>[_<mode>]/`
under `views/` (several sections) or `sections/<name>/views/` (one;
`job/views.py`), holding `view.jpg` (the bytes the model got) and `view.json`
(tool, arguments, caption, history step, and the recipe `zoom` redraws it
from), and listed in `views.jsonl`. A
placement picture and an `ants_syn` picture carry
`labels.tif` (the atlas id under every pixel below the caption, no tissue
rule), `borders.png` and a frame in `view.json`.
`langslice.coordinate_map(".../view.json")` gives every pixel's atlas
micrometres. An `ants_syn` picture showing its deformation adds
`residual.tif` (float32, `(2, rows, cols)`, `(d_row, d_col)` in picture
pixels, zero in the caption band); its frame's `residual` names it, and
`coordinate_map` applies it.

## Exports (`exports/`)

Written with the maps. Both are QuickNII-format JSON (`name`, `target`,
`aligner`, `slices`: `filename`, `anchoring`, `width`, `height`, `nr`,
`markers`), one slice per placed section, `width`/`height` the image file's
pixels and `nr` that index + 1.

- `quicknii.json`: the linear anchoring of every section, `markers` empty.
  The anchoring `o, u, v` is taken from the section's exact
  `pixel_to_atlas_um` (any plane, any cutting angle): the image's top-left
  corner and its two corner-to-corner vectors, converted to QuickNII voxel
  space (x left to right, y posterior to anterior, z inferior to superior;
  voxel edges). Allen mouse jobs at any supported resolution export to
  `ABA_Mouse_CCFv3_2017_25um.cutlas` with its anchoring rescaled to that
  target's voxels (voxel-edge coordinates scale with the voxel size; the
  volumes span the same millimetres). The rat target is DeepSlice's name
  (`WHS_Rat_v4_39um.cutlas`); for other atlases the QuickNII target is
  assumed to share the BrainGlobe voxel grid (untested). `target` is the
  `.cutlas` name for known atlases.
- `visualign.json`: the same, plus VisuAlign markers of each applied
  deformation: `[x, y, nx, ny]` on a regular grid (1/36 of the long edge, at
  least 32 px), in the image's continuous pixel coordinates (top-left corner
  at 0). A section point `(nx, ny)` corresponds to the point `(x, y)` of the
  linearly anchored plane: the direction of VisuAlign's triangulation as the
  QUINT team's reference code reads it (`triangulate` on `(nx, ny)`, mapped to
  `(x, y)`). Each marker reproduces the section's maps exactly at its own
  point.

## Importing a registration

Use `langslice-job FOLDER init --registration FILE`, `langslice linear run
FOLDER --registration FILE`, or `langslice.create_job(FOLDER,
registration=FILE)`. The file's linear placement becomes supplied positions,
cutting angles, orientation and affine transforms. Tasks default to
`nonlinear`; explicitly enable linear tasks to revise that placement.

| Format | What is read |
|---|---|
| QuickNII / VisuAlign JSON or XML | Target atlas and each section's anchoring, dimensions and filename |
| DeepSlice CSV, JSON or XML | Section anchorings and dimensions; CSV assumes the mouse target |
| LangSlice `registration.json` | Each section's `pixel_to_atlas_um` and source image size |

Matching tries the basename first, then the stem ignoring case, then a
unique QuickNII `_sNNN` section number. Ambiguous matches are refused.
The import report names placed sections, unmatched entries, missing sections,
refused placements and warnings. Missing placements must be resolved before
submission. A file that places no section is refused.

A registration on resized copies transfers by image fractions. Calibration
comes from the caller, then file metadata; otherwise the imported mapping
supplies an estimated pixel size. Per-section cutting angles are preserved.
An incompatible atlas target or a plane tilted more than 45 degrees from
the job's slicing plane is refused. Flat planes render at the nearest atlas
slab, up to half a voxel from the stored continuous position.

`registration` cannot be combined with separately supplied positions,
transforms, angles or orientation. Only the linear placement is imported:
VisuAlign markers and saved LangSlice residual deformations are not applied
as a new job's deformation. VisuAlign markers produce a warning.

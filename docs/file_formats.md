# Job folder file formats

A job stores settings in `job.json` and its working state in `state.json`.
`registration.json`, maps and pictures are derived outputs. Change the job
through its tools; editing an output does not change the registration.
A registration file can also be explicitly imported when creating a job.

## The job folder

```text
<images>/langslice/
  job.json              settings (the JobSpec, with its `disabled` tools and options, its `frozen` properties and its tool `plugins` when it has any), the format version, and the provenance of the plugins it ran with (`plugins`)
  state.json            the checkpoint: the truth
  registration.json     its public rendering
  history/              undo and redo steps
  sections/<name>/      per section (<name>: the filename's stem): its maps,
                        landmarks.json (readable corresponding pairs),
                        applied deformation records (deformable/),
                        image-model traces (image_correction/), its pictures (views/)
  views/                pictures of several sections
  views.jsonl           the picture index: number, tool, history step, sections, caption
  exports/              linear_results.json, spend.json, quicknii.json, visualign.json, traces/ (image-model replies), plugins/<tool>/ (copies of plugin layers)
  plugins/<tool>/       a tool plugin's own files and the layers it left (layers.json, layers/)
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
| `source` | the code that wrote it: `version`, `commit` (the git commit of the LangSlice checkout, null for an installed package) and `uncommitted_changes` (whether that checkout had uncommitted edits to tracked files), read once per process |
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
| `image` | `file`, `size` (`[width, height]` of the file), `pixel_size_um` (of the file), `pixel_size_source` (`host`, `file`, or `uncalibrated`: neither gives one, so the scale is the one every placement picture draws the section at, the image width mapped onto the atlas plane's width at its current position) |
| `pixel_to_atlas_um` | 3x3: a file pixel `[row, col, 1]` -> atlas micrometres (3 rows, the atlas axes), the linear placement; null without a placement |
| `mapping` | `linear`, `linear (identity in-plane: no transform written)`, `linear + residual` (the maps hold the complete mapping), or null |
| `problem` | why there is no matrix (`no position`, an unreadable file, ...), or, with a matrix, that the pixel size is unknown (`uncalibrated`, as the pictures draw it); else null |
| `parameters_digest` | SHA-256 of the parameters, the matrix and the image facts |
| `maps` | null before any maps were written; else `files` (kind -> relative path), `grid_size`, `full_resolution`, the grid's `pixel_to_atlas_um`, and `current` (false once the section changed after the maps were written) |

### Section parameters

| Key under `parameters` | Content |
|---|---|
| `atlas` | `name`, `version` |
| `plane` | `name`, `position_mm`, `position_um`, `pitch_deg`, `yaw_deg` |
| `orientation` | Counter-clockwise quarter turn `rotation_deg`, followed by left-right `flip` |
| `affine` | Null without a transform; otherwise `kind`, `params`, `physical` and `mirrored` |
| `deformation` | Null, or a residual record with `kind`, `record`, `key`, `steps`, `inverse_source` |
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

The maps cover every pixel that maps into the atlas volume.

| File | Content |
|---|---|
| `coords.tif` | float32, ImageJ hyperstack, axes CYX, shape `(3, rows, cols)`: channel k is atlas axis k in micrometres. NaN outside the atlas volume. Calibrated in µm per grid pixel; the channel labels are `axis0_um`..`axis2_um`; the ImageJ Info property holds the grid facts as JSON. |
| `labels.tif` | uint32, `(rows, cols)`: the atlas annotation id under each pixel (nearest neighbour on the atlas plane, as the pictures' labels layer and the deformable record do), 0 where `coords.tif` is NaN and where the atlas has no region. Deflate with the horizontal predictor; not an ImageJ file (Fiji turns uint32 into float32, which loses the larger Allen ids). |
| `labels_fiji.tif` | uint16, ImageJ, `(rows, cols)`: a dense index of the section's regions (1..n in the order of their atlas ids, 0 none), with an ImageJ lookup table giving each index its atlas colour (display range 0-255; a section with more than 255 regions shows the rest without their colour). Opens in Fiji as is. |
| `labels.csv` | `index,id,acronym,name,r,g,b`: the dense index -> atlas id, acronym, name and RGB colour (the atlas structure tree's). |
| `residual.tif` | only with an applied deformation. float32, ImageJ, axes CYX, `(2, rows, cols)`: `(d_row, d_col)` in grid pixels, defined on the whole grid, such that `atlas_um = pixel_to_atlas_um @ [row + d_row, col + d_col, 1]` (`maps.json`'s matrix). Its source is the deformation record named in `registration.json`. |
| `maps.json` | `format_version` 1, `section`, `grid_size` (`[width, height]`), `full_resolution`, `um_per_px`, `pixel_to_atlas_um` (the grid's), `parameters_digest` (what they were written from; `registration.json` compares it), `deformation_record`, `files`. |

Float maps are deflate-compressed without TIFF's floating-point predictor,
which ImageJ 1.x cannot decode. At working size a section's `coords.tif` is
tens of MB; at full resolution of a whole-slide scan, hundreds.

## Pictures

Each picture a door showed is saved as a folder `<number>_<tool>[_<mode>]/`
under `views/` (several sections) or `sections/<name>/views/` (one;
`job/views.py`), holding `view.jpg` (the bytes the model got) and `view.json`
(tool, arguments, caption, history step, and the recipe `zoom` redraws it
from), and listed in `views.jsonl`. A
placement picture and an `fireants_syn` picture carry
`labels.tif` (the atlas id under every pixel below the caption, no tissue
rule), `borders.png` and a frame in `view.json`.
`langslice.coordinate_map(".../view.json")` gives every pixel's atlas
micrometres. An `fireants_syn` picture showing its deformation adds
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

### `traces/<name>.npz`

Written with the maps, one per mapped section whose last `trace_borders`
call holds a completed reply (`<name>` as for `sections/<name>/`: the
filename's stem). A section without one loses the file an earlier export
left. It holds what the image model returned, before any fit, on the grid
of the section's maps (`maps.json`'s `grid_size`): the image file's own
frame, with no rotation or flip applied, at the working copy's size or,
with `full_resolution`, the file's own pixels. A NumPy `.npz` archive:

| Key | Content |
|---|---|
| `borders` | bool, `(rows, cols)`: the model's lines as LangSlice extracts them from its reply (yellow pixels, thinned): the lines the trace's fit is fitted to |
| `reply` | uint8 RGB, `(rows, cols, 3)`: the model's raw reply, cropped to the canvas's aspect and resized to it as the lines are, then mapped to the grid; black outside the canvas |
| `info` | JSON text: `section`, `grid_size` (`[width, height]`), `full_resolution`, `frame`, `grid_from_canvas` (3x3 on `[x, y, 1]`: the model's canvas to the grid's pixel centres), `artifact_dir` (the attempt's folder under `sections/<name>/image_correction/`, relative to the job folder), `attempt`, `provider`, `model`, `prompt_edited`, `profile` and `untested` (an untested profile), `include` / `exclude` (the regions `restrict_to` chose), `placement_current` (whether the section's placement is still the one traced), `written_by` |

The map from the canvas to the grid is the images' own (the canvas's
padding and resize of the oriented section render, then the render's quarter
turn, flip and resize of the file); the section's placement does not enter
it, so a reply made at an earlier placement is still drawn where the model
put it on the tissue.

### `spend.json`

Written by `langslice linear run` when the session ends (also when it fails),
for the person or agent monitoring the run: `lane` and `model`; `tokens`
(`input_uncached`, `input_cached`, `output`, `reasoning`, the model's reported
counts summed over the run); `quota_used_percent` (the usage window's rise since
the run's first reading, null when the provider reports no window) and
`quota_reading` (the last reading); `usd_estimate` (list prices times tokens,
null for a model without a known price); `note`.

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


## Landmarks

`state.json` stores each section's authoritative `landmarks` object:
`next_id` and a list of `pairs`. Every pair has a stable `id`,
`section_fraction_xy` (edge fractions of the original, unrotated, unflipped
image) and `atlas_native_xy` (pixel centres in that section's native atlas
plane). `plane` records atlas, plane, position and cutting angles; `stale`
compares it with the section's current plane. The endpoint records `section`
and `atlas` include `canonical_xy`, input `picture` number, input `pixel_xy`,
its `registration_context` and the exact saved `geometry` used to resolve it.
Appearance does not enter the correspondence's staleness test.

`sections/<stem>/landmarks.json` is a readable derived copy, regenerated on
every checkpoint, including undo and redo. It has `format_version: 1`,
`section`, `next_id`, and `pairs`. Editing this file does not change the job.
A landmark application records its revision in the section's deformation
reference, with `landmark_pairs` (each applied pair id with a digest of its two
endpoints, what `applied` / `pending` is read from) and `landmark_folds`
(`area_fraction` and the pair ids of each folded triangle, empty when none); the saved `DeformableRecord` includes its source/target vertices
and mesh connectivity in `engine.native_parameters`, and its full pair set
in `provenance`, with `automatic_holds_xy` (the automatic holds' section pixels
on the working grid), `hold_spacing_px` and `hold_clearance_spacings`; the
mesh vertices of `native_parameters` are the four fixed corners, then the
pairs, then the holds (`pair_count`, `hold_count`), each hold's target equal to its
source. The section's deformation reference lists the holds as `landmark_holds`
(edge fractions of the original image, like a pair's `section_fraction_xy`), which
the result picture draws; `landmark_folds` names a hold vertex of a fold `~K`
(the K-th hold). `landmark_baseline_key` and `landmark_baseline_steps` in the
section's deformation reference identify the registration before the current
landmark method; its record's `parent` holds that baseline. Consecutive landmark
applications replace the landmark method on that baseline. Undo history
separately preserves the previously applied landmark result. Maps and exports
consume the composed residual field through
the same path as other applied deformations.

Saved look/zoom recipes carry `endpoint_panels`: role (`atlas` or `section`),
section filename, content `bounds`, and exact pixel maps. Framed section
pictures use `pixel_to_file_fraction`; overlays carry `pixel_to_render`,
render size, orientation and the saved warp key; atlas panels carry
`pixel_to_native` and native size. All of these matrices act on `[x, y, 1]`;
they are separate from the `[row, col, 1]` convention of public atlas maps.
`registration_context` identifies the atlas plane and registration drawn.
Side-by-side crops preserve their panel offsets and each panel's caption
exclusion. Repeated saved-image crops preserve the original panel geometry
under `source_endpoint_panels` and carry the adjusted `endpoint_panels`.

When MCP shrinks a numbered picture to fit its reply budget, the delivered
JPEG gets a new picture number and geometry at its delivered size. The
original picture remains unchanged; `extra.transport_source_picture` names
it. Look pictures retain their native-detail redraw recipe. Other resized
pictures use `transport_snapshot` and subsequent zooms crop that snapshot.

## Plugin layers

A tool plugin leaves layers on sections (`docs/plugins.md`). Each tool keeps
its files under `plugins/<tool>/`; nothing else of the job is written by it.
`plugins/<tool>/layers.json` is the manifest of every layer the tool ever made:
`tool`, `next` (the number of the next layer file) and `layers`, each with `id`
(`<tool>/<number>`), `section`, `name`, `kind`, `frame`, `file` (relative to the
job folder), `tool`, `call` (the call that made it) and `created`. The layers a
section shows now are its `layers` in `state.json` (the same fields, without
`section`; the key is absent while a section has none), so undoing the call that
made a layer hides it and the files stay.

The files are `plugins/<tool>/layers/<section>_<number>_<name>.<ext>`:

| Kind | File | Content |
|---|---|---|
| `points` | `.csv` | header `x,y,label,value` for `section_px`, `x_um,y_um,z_um,label,value` for `atlas_um`; `label` and `value` empty where absent |
| `lines` | `.json` | `{"kind": "lines", "frame", "features": [...]}`; a feature is a GeoJSON `LineString`, or a `Polygon` (one ring, first point repeated) for a closed line, with `properties.label` when labelled |
| `labels` | `.png` | 16-bit grey, the integer ids; `.npz` (`labels`, `uint32`) when an id is 65536 or more |
| `scalar` | `.npz` | `values` (`float32`, 2-D) and `level` (the level drawn as an outline) |
| `picture` | `.png` | the picture as given |

Frames: `section_px` is `[x, y]` pixels of the section's image file (pixel
centres at integers, `(0, 0)` the centre of the first pixel; a raster covers
the whole file whatever its size, in the file's orientation: not turned or
flipped); `atlas_um` is `[x, y, z]` in BrainGlobe
micrometres in the atlas's axis order (the convention above). `export_maps`
copies the layers the sections show to `exports/plugins/<tool>/` with a
`layers.json` listing them (`file` the name inside that folder).

`job.json`'s `plugins` lists, per plugin of the most recent run that built its
tools: `ref`, `module`, `factory`, `version`, `package_version`, `api_version`,
`sources` (SHA-256 of each source file that declared its tools) and `tools`:
`name`, `kind`, `group`, `task`, `writes`, `long`, `limits`, `version`, `meta`
(when given) and `declaration` (`name`, `description`, `arguments`: the tool as
the model reads it). `logs/events.jsonl` has a `plugins` line whenever that
record changes.

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
  sections/<name>/      per section (<name>: the filename's stem): its maps
                        (written by export_maps), landmarks.json (readable
                        corresponding pairs), deformation records (deformable/),
                        every registration it has held (registrations.jsonl),
                        its made section images (images/<name>.png), its masks
                        (masks/<name>.npz), image-model traces
                        (image_correction/), its pictures (views/)
  views/                pictures of several sections
  views.jsonl           the picture index: number, tool, history step, sections, caption
  exports/              linear_results.json, spend.json, quicknii.json, visualign.json, traces/ (image-model replies), plugins/<tool>/ (copies of plugin layers)
  plugins/<tool>/       a tool plugin's own files and the layers it left (layers.json, layers/);
                        plugins/script/: the layers a script left through job.context
  logs/                 the agent CLI's calls (calls.jsonl, one line per call, and
                        calls/<n>_<verb>.json, each call's whole envelope as printed),
                        launches/<n>_<role>.md (the exact launch prompts of `langslice-job FOLDER launch`)
                        and background runs, events.jsonl (job events: opens, plugins,
                        every tool call on every door, a script_write line per write
                        a script's job.context landed);
                        engine.log (what the registration engines printed while they fit);
                        env.lock, held while env/ is made
  trace/                agent transcripts: the built-in agent's trace (also when it is sent elsewhere), a coding agent's session copied in by its harness plugin, and its launch check (<harness>_<session>.launch.json)
  env/                  the job's own Python environment (`langslice-job FOLDER env`), removed when the harness session that submitted the job ends
  scripts/, scratch/    a coding agent's own files; scripts/requirements.txt lists the packages env/ holds
  AGENTS.md, CLAUDE.md  the reference card for coding agents
  TOOLS.md              every verb of the job with its description and arguments (written by init and launch)
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
| `maps` | null until `export_maps` wrote the section's maps (`submit` writes none); else `files` (kind -> relative path), `grid_size`, `full_resolution`, the grid's `pixel_to_atlas_um`, and `current` (false once the section changed after the maps were written) |

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
the job folder ([deformation records](#deformation-records-sectionsnamedeformablekey));
coordinate maps include that deformation as well as the affine.
Each of its `steps` says how a step was made: a native fit's `method`
(`fireants_syn`, or `trace_borders` for the fit of a trace; a step a job saved
before steps named their method has none and is read as one of these),
`start` (`linear` or `current`), settings (`fit_section`, `fit_atlas`, `mask`,
`smoothness_um`, and for a stain fit `metric`, `edges` and `detail`; a step
saved before these were recorded lacks them) and regions; a landmark
application's `method` `landmark_triangulation`; a deformation a tool plugin
wrote (`docs/plugins.md`), `{"method": "plugin:<tool>", "start": "linear" |
"current", "note"?, "settings"?}` (`current`: made on top of the deformation
the section held, whose steps come first and whose record is its record's
`parent`; `settings`: the call's, by the tool's names), whose record has the
engine name `plugin:<tool>`, the engine's own numbers and version where the
plugin gave them (`engine.native_parameters`, `engine.version`), the
`inverse_source` `fixed-point inversion of the supplied map`, and in its
`provenance` `start`, `settings` and the `plugin` (`tool`, `call`, `version`,
`note`, `meta`). One a script wrote through the library's `job.context`
([your own method](library.md#your-own-method)) is the same with `script` for
`plugin:<tool>` and a provenance `script` (`call`, `note`, `meta`, `argv0`: the
path of the running script); `fit_fireants` gives its call's settings by
`fireants_syn`'s names and FireANTs' own numbers. A transform written by a
tool plugin has the `kind` `plugin:<tool>`, by a script `script` (and its
`note`; from `fit_elastix`, the call's settings under `fit` in `state.json`).
A `script_write` line in `logs/events.jsonl` (`property`, `sections`, `notes`,
`argv0`, `call`) marks each write a script landed.

## Per-section maps (`sections/<name>/`)

Written only by the `export_maps` verb (CLI and library), for every section
with a position; never on a write or at `submit` (they are large). The
library's `job.section_maps(section)` returns the same arrays in memory
without writing them ([library](library.md#reading-results)). The grid
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

## Deformation records (`sections/<name>/deformable/<key>/`)

Every deformation a section held (each fit, landmark application and
plugin warp; `<key>` the first 24 hex digits of the record's input digest)
is a folder that nothing deletes, so undo can return to it. It holds:

- `record.json`, written last (a folder without it is incomplete):
  `version` 2, `step` (0 for a deformation on the linear placement),
  `parent` (the folder name of the record this step was fitted on, a sibling
  in the same `deformable/` folder, or null), `placement` (atlas, position,
  plane, angles, `atlas_to_section`: native atlas-plane pixel centres to
  fit-grid pixel centres, `section_mm_per_px`), `section_size` (the fit
  grid, `[width, height]`: the oriented section render the fit ran on, not
  the image file), `settings`, `field_convention`, `inverse_source`,
  `labels`, `excluded_ids`, `engine`, `diagnostics`, `atlas` (name,
  orientation, `resolution_um`, `native_size`, `native_to_volume_index`)
  and `provenance`.
- `warp.ome.zarr`: the arrays, a Zarr v3 store in the OME-Zarr 0.6 layout,
  every array one chunk compressed with Blosc zstd and byte shuffle:

| Path | Content |
|---|---|
| `coordinateTransformations/field/s0` | float16 `(2, rows, cols)`: `(dy, dx)` in mm on the fit grid. A section point `p = (row, col) * section_mm_per_px` corresponds to the placed atlas point `p + field[:, row, col]`. Stored as float16 (rounding under 0.25 µm on displacements below 0.5 mm; LangSlice reads it as float32) |
| `coordinateTransformations/inverse_field/s0` | the same for the inverse, on the placed-atlas grid: placed point `q` corresponds to section point `q + inverse_field(q)`; absent when the record has none |
| `warped_labels/s0` | uint32 `(rows, cols)`: the leaf atlas ids under the composed warp |
| `footprint/s0` | bool `(rows, cols)`: the placed atlas footprint |
| `left_out/s0` | bool `(rows, cols)`: the section pixels an engine's own sorting left out; absent otherwise |
| `weights` | float16 `(rows, cols)` in [0, 1]: the engine's own tissue sorting (P(match)) on the fit grid, `make_mask(source="fit")` copies it; absent when the engine estimated none |

Each group is an OME-Zarr multiscale image of one level whose `s0` scale is
the grid's millimetres per pixel (1 on the displacement axis, `type`
`displacement`). The root group's `ome.scene` states the whole mapping with
the OME-NGFF 0.6 coordinate transformations, so another program can read it
without LangSlice: coordinate systems `section` and `placed_atlas` (`y`, `x`
in millimetres) and `atlas` (`axis0`..`axis2` in micrometres, the atlas's
own axis order, voxel `i`'s centre at `i * resolution`); a `bijection`
`section` -> `placed_atlas` whose forward and inverse are the
`displacements` of the two fields (the forward `displacements` alone when
there is no inverse); a `sequence` `placed_atlas` -> `atlas` of a `scale`
(mm to placed pixels), an `affine` (placed pixels to native atlas-plane
pixels: the inverse of `atlas_to_section`), an `affine` from the 2-D plane
into the 3-D volume (`native_to_volume_index`) and a `scale` by the
resolution; and an `identity` from each image group's `section` system to
the scene's. The metadata is written from `record.json` and never read back.

## Registrations (`sections/<name>/registrations.jsonl`)

Every registration a section has held, one JSON line each, numbered from 1
per section; `restore_registration` puts one back. A section's registration
is its orientation, transform and deformation (a deformation whose linear
placement has changed is no part of it). A line is appended when an undo
step, an undo or a redo leaves the section a registration it never held, and
never rewritten; undo leaves the file as it is. Fields: `format` (1), `id`
(the number), `by` (the tool whose write left it, `script` for a script's
own write; for one held before the section's list began, which a later write
lists, what its content says made it), `at` (local time, `YYYY-MM-DD
HH:MM:SS`; null for one held before the list began), `digest` (of the
orientation, the transform and the deformation's record key: two writes that
leave the same registration share its number), `position_mm` and
`cutting_angles_deg` (where the section was), then the registration: `flip`,
`rotation_deg`, `transform` (as `state.json` holds it) and `deformation` (the
section's `deformation` entry, naming its record under `deformable/`, or
null). A status row's `registration` is the number of the one the section
holds. `restore_registration`'s list adds, per registration, the pictures its
write saved: the `views.jsonl` pictures drawn by the tool named in `by` whose
`registrations` give the section that number. The file is read back, as the deformation records are; a line cut
short is skipped.

## Pictures

Each picture a door showed is saved as a folder `<number>_<tool>[_<mode>]/`
under `views/` (several sections) or `sections/<name>/views/` (one;
`job/views.py`), holding `view.jpg` (the bytes the model got) and `view.json`
(tool, arguments, caption, history step, the recipe `zoom` redraws it from,
and `registrations`: the number of the registration each of its sections was
drawn under, `{filename: number}`, for the sections that held a listed one or
a `look` drew under one), and listed in `views.jsonl` (with the same
`registrations`). A `look` sheet is a picture of mode `sheet` whose recipe
(`renderer` `sheet`) lists each tile's `picture`, its `box` on the sheet and
the picture's `size`. A
placement picture and an `fireants_syn` picture carry
`labels.tif` (the atlas id under every pixel below the caption, no tissue
rule), `borders.png` and a frame in `view.json`.
`langslice.coordinate_map(".../view.json")` gives every pixel's atlas
micrometres. An `fireants_syn` picture showing its deformation adds
`residual.tif` (float32, `(2, rows, cols)`, `(d_row, d_col)` in picture
pixels, zero in the caption band); its frame's `residual` names it, and
`coordinate_map` applies it. A script's preview of a proposal not written
(`preview_overlay`, `preview_positioning`) is tool `script`
(`<number>_script_overlay`, `<number>_script_positioning`), its caption
saying `NOT written`, its recipe a `transport_snapshot` (a zoom crops the
saved picture); a linear proposal's overlay keeps its layers and frame, a
deformation proposal's has none.

## Events (`logs/events.jsonl`)

One JSON object per line, appended as things happen, each with `kind` and
`at` (UTC, ISO 8601, seconds). Readers in another process (the LangSlice GUI)
follow the job through it.

| `kind` | When | Other fields |
|---|---|---|
| `open` | the job is opened (`Job.open`: created or resumed) | `resumed` |
| `plugins` | the tool plugins' record in `job.json` changed | `plugins` |
| `script_write` | a script's `job.context` write landed | `property`, `sections`, `notes`, `argv0`, `call` |
| `tool_start` | a tool call begins, on any door | `name`, `execution_id`, `door`, `target_ids`, `args` |
| `tool_end` | that call ends | the same, plus `step`, `views`, `failed`, `status` |

The tool lines are written for every door: `door` is `agent` (the built-in
agent, `langslice linear run` and the ABBA worker), `mcp`, `cli` (the agent
CLI) or `library` (a script). `execution_id` pairs a call's two lines.
`target_ids` are the sections the call names (every section for `status`,
`undo`, `redo`, `submit` and a stack-wide call), as filenames. `args` are the
call's arguments as JSON; past 2048 bytes they are that JSON cut to 2048
bytes, as a string, and the line adds `args_truncated: true`. `step` is the
undo history's depth after the call, as a picture's `step` in `views.jsonl`.
`views` are the `view.jpg` paths of the pictures the call saved, relative to
the job folder (written in the background: a file can appear just after the
line). `failed` is true when the reply's status is `error` or `refused`, or the
tool raised; `status` is the reply's status word (`ok` when it has none,
`error` when the tool raised). The reply itself and picture bytes are never
written. A dry run (`--dry-run`) writes no line.

## Recent jobs (`~/.langslice/recent_jobs.jsonl`)

Outside the job folder: every `Job.open` (a run, `langslice-job FOLDER init`,
`create_job`, an MCP `start_job` that makes or resumes a job) appends one line
with `folder` (the job folder), `image_folder`, `atlas` and `at` (UTC, ISO
8601), so the GUI can list the jobs opened on this computer. Past 256 KB the
file is rewritten with the newest line of each of the newest 500 job folders.
`LANGSLICE_RECENT_JOBS` names another file, or `off` for none.

## Settings (`~/.langslice/settings.json`)

Outside the job folder: what the LangSlice window's Settings tab saves
([the LangSlice window](gui.md)). No key or sign-in is in it.

```json
{
 "version": 1,
 "connection": "openai-oauth",
 "runs_in": "langslice",
 "agent": {
  "openai-oauth": {"model": "openai-oauth/gpt-6.1-sol", "reasoning": null},
  "anthropic-api": {"model": "anthropic-api/claude-opus-5-5", "reasoning": "high"}
 }
}
```

`connection` is one of `openai-oauth`, `openai-api`, `anthropic-api`,
`gemini-api`, or null. `runs_in` is `app`, `langslice`, or null (not chosen);
nothing reads it yet. `agent` keeps a model per connection, spelled as
`langslice linear run --model` takes it, and a `reasoning` effort (`low`,
`medium`, `high`, `xhigh`, `max`, or null for the provider's own).
`langslice linear run` started with neither `--model` nor `--reasoning` runs
on the chosen connection's entry. A value LangSlice does not know reads as
unset. `LANGSLICE_SETTINGS` names another file, or `off` for none.

## Exports (`exports/`)

Written at `submit` and by `export_maps`. Both are QuickNII-format JSON (`name`, `target`,
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
  volumes span the same millimetres). `ccfv3augmented_mouse_25um` (the Allen
  25 um volume with 38 more planes in front and 24 behind) exports to the
  same target on the Allen grid: positions are 0.350 mm (14 planes) further from
  the front than in the Allen volume, and an anchoring is read back onto the
  augmented atlas with the same offset, so one QuickNII or DeepSlice file
  placed on `allen_mouse_25um` and on the augmented atlas differs by 0.350 mm
  in position and in nothing else. The rat target is DeepSlice's name
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

Written at `submit` (on the section's working-copy grid) and with the maps by
`export_maps` (on the maps' grid), one per placed section whose last `trace_borders`
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
upright turn, crop to the placed atlas plus a margin, padding and resize of
the oriented section render, then the render's quarter
turn, flip and resize of the file); the section's placement does not enter
it, so a reply made at an earlier placement is still drawn where the model
put it on the tissue.

### `spend.json`

Written by `langslice linear run` (and the ABBA worker's run) after every model
call, so it holds the account so far while the run goes, and once more when
the session ends (also when it fails), for the person or agent monitoring the
run: `lane` and `model`; `tokens`
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
| DeepSlice CSV, JSON or XML | Section anchorings and dimensions; CSV assumes the mouse target (also for jobs on `ccfv3augmented_mouse_25um`, read on the Allen grid) |
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
in `provenance`; the mesh vertices of `native_parameters` are the four fixed
corners, then the pairs (`pair_count`). A reference or record saved by an
earlier version may also carry automatic holds (`landmark_holds`,
`hold_count`, `automatic_holds_xy`, fold vertices named `~K`); they are read
and ignored. `landmark_baseline_key` and `landmark_baseline_steps` in the
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

A tool plugin leaves layers on sections (`docs/plugins.md`), and so does a
script through `job.context` (as tool `script`). Each tool keeps
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

## Section images (`sections/<stem>/images/<name>.png`)

One 8-bit gray PNG per made section image a section holds (`make_image`, a
script's `add_image`, and the opening `preprocessed`; see `registration.md`):
the working copy's size, in the image file's orientation (the quarter turn and
flip are applied where it is drawn). Files are never deleted: undo only hides
the state reference, and a name made again with other pixels gets a new file
(`<name>.2.png`, ...). `state.json` holds the entry under the section's
`images`, absent while empty: `source` (`{"recipe": {...}}` with the recipe's
`weights` (`"auto"` or `{image name: weight}`), `clahe_clip`, `clahe_tiles`,
`n4`, `denoise`, `contrast` (`[low, high]` or null), `gamma` (or null),
`invert`; `{"script": note}`; or `{"opening": "adaptive" | "host" | "plain"}`,
with the host's `settings` for `host`), `file`, `digest` (8 hex digits of the
pixels and their shape), `made` (ISO time) and, for a recipe with automatic
weights, `weights_used` (the weights it resolved to, by raw channel). The
opening `preprocessed` file is the automatic path run on the working copy;
like every made image, each reader reads the file, resized to its own size.

State format 4 holds these entries; a job of format 3 (the stack's
`appearance`: recipes and raw channels' display properties) is read through
the migration when it opens (`core.section_images.Migration`): each recipe is
made into a file and an entry per section (a section's own override
honoured), `preprocessed` without a recipe becomes the opening image, the
display properties are dropped, a mask's `channel` is named `section_image`,
and the state and its history steps are written again as format 4.
Deformation records are not rewritten. A recipe image is made once on the
working copy, so a fit on a migrated job reads that file: it is not the image
the older LangSlice ran the recipe on at the fit's size, and a refit does not
reproduce the job's earlier fit. The first open of a large job takes a while
(about 2-3 minutes for 20-30 sections whose recipe denoises); later opens do not.

## Masks (`sections/<stem>/masks/<name>.npz`)

One file per mask a section holds (`make_mask`; see `registration.md`). Arrays:
`weights` float16 `(rows, cols)` in [0, 1] on the section's fit grid, `grid`
`[width, height]`, `made` the ISO time. Files are never deleted: undo only hides
the state reference, and a mask made again under a name that exists gets a new
file (`<name>.2.npz`). `state.json` holds the entry under the section's
`masks`, absent while empty: `kind` (`em` or `fit`), `section_image`, `atlas_image`
(resolved), `params` (the EM starting points used), `inputs` (the digest of the section's
orientation, which the mask's pixels sit on), `registration` (the digest of the
registration it was made at), `file` and `left_out` (the fraction of the placed
footprint under 0.5). A deformation record's `provenance` holds the `inputs` and
`estimated` blocks of the fit that made it.

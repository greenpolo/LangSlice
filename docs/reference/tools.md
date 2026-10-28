# Tools and options

The registration {term}`agent` works only through LangSlice's {term}`tools <tool>`.
This page lists every tool, what it does, its arguments, the parts of it a
job can switch off, and what it needs while it is on. You do not call these
tools yourself; your agent does. Read this page to know what the agent could
do in a run, and to choose what to switch off in the
[job settings](settings.md#disabled).

Every way into LangSlice gives the agent the same tools under the same names:
the {term}`MCP` server, the {term}`agent CLI`, the Python library and the
built-in agent (`langslice linear run`). The exact arguments of a tool in a
given job are printed by `langslice-job FOLDER schema TOOL`, and the job
folder's `TOOLS.md` lists them all.

## Summary

The code groups the tools under the {term}`task` that brings them. Common
tools are in every run.

One table per group. "Off" says whether the tool can be switched off; an
option is switched off as `tool:option` (for example `look:atlas`).

**Common**

| Tool | Writes | In a run when | Off | Options |
|---|---|---|---|---|
| [`look`](#look) | no | always | no | `section`, `atlas`, `overlay`, `positioning` |
| [`zoom`](#zoom) | no | always | yes | |
| [`make_image`](#make_image) | yes | always | yes | |
| [`grep_atlas`](#grep_atlas) | no | always | yes | `points` |
| [`grep_atlas_view`](#grep_atlas_view) | no | always | yes | |
| [`status`](#status) | no | always | no | |
| [`list_files`, `search_files`, `read_file`](#list_files-search_files-read_file) | no | always | yes | |
| [`mark_damage`](#mark_damage) | yes | always | yes | |
| [`make_mask`](#make_mask) | yes | Linear or Nonlinear on | yes | |
| [`note`](#note) | yes | always | yes | |
| [`undo`](#undo-and-redo) | yes | always | no | |
| [`redo`](#undo-and-redo) | yes | always | yes | |
| [`restore_registration`](#restore_registration) | yes | Linear or Nonlinear on | yes | |
| [`submit`](#submit) | yes | always | no | |
| [`export_maps`](#export_maps) | files only | agent CLI and library only | yes | |

**Positioning, Linear and Nonlinear**

| Tool | Group | In a run when | Off | Options |
|---|---|---|---|---|
| [`position_sections`](#position_sections) | Positioning | Positioning on, or Linear on with `transform.angles` | yes | |
| [`interactive_transform`](#interactive_transform) | Linear | Linear on | yes | `flip`, `rotate` |
| [`elastix_affine`](#elastix_affine) | Linear | Linear on | yes | |
| [`fireants_syn`](#fireants_syn) | Nonlinear | Nonlinear on | yes | `metric`, `edges`, `detail` |
| [`set_landmarks`](#set_landmarks) | Nonlinear | Nonlinear on | with `landmark_triangulation` | |
| [`landmark_triangulation`](#landmark_triangulation) | Nonlinear | Nonlinear on | with `set_landmarks` | |
| [`trace_borders`](#trace_borders) | Nonlinear | Nonlinear on and an image model connected | yes | |

Every tool of these three groups writes.

The Positioning task is `position` in the settings, Linear is `transform` and
Nonlinear is `nonlinear` ([tasks](settings.md#tasks)). A job's
[tool plugins](settings.md#plugins) add their own tools after these.

## Which tools a run has

Four things decide which tools the agent gets.

1. **Tasks.** A task that is off brings none of its tools, and its
   properties are frozen on every section: Positioning writes position and
   cutting angles, Linear writes orientation and the affine, Nonlinear writes
   the deformation. A task whose properties are all frozen on every section
   counts as off.
2. **What is frozen.** A tool that writes only properties frozen on every
   section is not offered: `position_sections`, `interactive_transform`,
   `elastix_affine`, `fireants_syn`, `landmark_triangulation`,
   `trace_borders` and `restore_registration`, by what they write.
   `set_landmarks` goes with `landmark_triangulation`. A frozen orientation
   on every section takes off `interactive_transform`'s flip and quarter
   turn. See [`frozen`](settings.md#frozen).
3. **The `disabled` list.** It switches off single tools and options on top
   of what the tasks give. See [`disabled`](settings.md#disabled) and the
   rules [below](#switching-tools-and-options-off).
4. **The image model and the door.** `trace_borders` needs the Nonlinear
   task, an image model in the job (`nonlinear.provider` not `none`) and,
   on MCP and the agent CLI, that model's key or login on this computer.
   `export_maps` and `look`'s `sheet` argument exist only on the agent CLI
   and in the library, never for a model on MCP or the built-in agent.

`look`, `status`, `undo` and `submit` are in every run and cannot be
switched off.

## Switching tools and options off

A name in a job's `disabled` list is a tool (`zoom`) or a tool and one of its
options (`look:positioning`). The valid names are:

| Name | What it removes |
|---|---|
| `zoom`, `make_image`, `grep_atlas`, `grep_atlas_view`, `list_files`, `search_files`, `read_file`, `note`, `redo` | the tool |
| `position_sections`, `interactive_transform`, `elastix_affine`, `fireants_syn`, `trace_borders`, `mark_damage`, `make_mask`, `restore_registration` | the tool, as its task or switch would; with `make_mask` off, the `mask` argument of `elastix_affine` and `fireants_syn` and `look`'s `masks` go too |
| `set_landmarks`, `landmark_triangulation` | the landmark pair; each needs the other, so both must be switched off together |
| `export_maps` | the scripting verb |
| `look:section`, `look:atlas`, `look:overlay`, `look:positioning` | that mode of `look` |
| `grep_atlas:points` | the `points` argument of `grep_atlas` |
| `interactive_transform:flip` | mirroring: the `flip` key of its sections |
| `interactive_transform:rotate` | the quarter turn: the `rotate_quarter` key of its sections |
| `fireants_syn:metric`, `fireants_syn:edges`, `fireants_syn:detail` | that argument of `fireants_syn`, which then keeps its default |

The tools of the job's tool plugins are valid names too, for that job.

A list that cannot work is refused when the job is made, with one line
saying why. The checks, in order:

- every name is a tool or option above (or a plugin tool), and none is
  `look`, `status`, `undo` or `submit`;
- `look` keeps at least one mode;
- a task that is on keeps at least one tool that writes its answer:
  `position_sections` for Positioning; `interactive_transform` or
  `elastix_affine` for Linear; `fireants_syn`, `landmark_triangulation` or
  `trace_borders` for Nonlinear;
- a tool that is on keeps the pictures it needs (the "Needs" line of each
  tool below).

What a job switches off is absent everywhere the agent looks: the tool is not
built, an option is not in the tool's arguments, and no sentence of the job
statement, a tool description, the `status` reply or a refusal names it. A
call that names a switched-off `look` mode or quarter turn anyway is refused
with `OPTION_OFF`, and a flip with `FLIP_DISABLED`. A reply that would use a
disabled option uses another: with `look:positioning` off,
`position_sections` returns an overlay of each written section.

Switching a tool off saves the agent's context; it is not a guard. A script's
own writes through the library's `job.context` are not checked against
`disabled`. What a job freezes holds for every tool and every script.

## Limits per call

| Where | Limit |
|---|---|
| The built-in agent and MCP | `look`, `grep_atlas_view`, `make_image` and `make_mask` show at most 4 pictures per call; the rest are listed under `not_shown` with the call that draws them. `interactive_transform` and `elastix_affine` take at most 4 sections per call. |
| The agent CLI and the library | No picture limit. `interactive_transform` and `elastix_affine` take any number of sections. |
| Every door | `fireants_syn` takes at most 20 sections per call (each fitted section is held in memory, about 170 MB, until the call writes them all). A job's [`transform.max_parallel`](settings.md#transform-settings) below 4 caps both transform tools at that number. |

A call over a limit is refused with `TOO_MANY_SECTIONS` and nothing is done.

## Conventions every tool uses

- **Sections** are named by image filename; the extension may be left off,
  and a stem unique within the job also works. Never by number.
- **Positions** are millimetres along the atlas's slicing axis. In a coronal
  atlas with BrainGlobe's `asr` orientation they run from the anterior edge
  of the volume toward the posterior. The stack's order follows the
  positions.
- **Cutting angles** are pitch and yaw in degrees; 0 and 0 is the atlas's
  flat plane.
- **Regions** are named by acronym, name or atlas id, descendants included.
  A region may be named with one half of the brain, `"CTX:left"`. The halves
  come from the atlas midline carried through the section's placement, never
  from the tissue, so a mirrored placement swaps them. Where the midline runs
  closer to horizontal on the picture (a horizontal section shown with the
  anterior-posterior axis across), the halves are `top` and `bottom`. A
  sagittal plane lies in one hemisphere and has no sides (`NO_SIDES`).
  Naming the pair a picture does not have is refused with
  `WRONG_SIDE_PAIR`, whose message names the pair to use; a side on a
  midline at 45 degrees is refused with `SIDES_AMBIGUOUS`.
- **Pictures** are numbered, captioned and saved in the job folder. A
  later tool names a picture by its number, and points on it by `x`, `y`
  pixels from its top-left corner. Picture size changes what is shown, never
  a fit or a stored transform. Image files are never modified. Where the
  agent chooses the picture size, `resolution` applies to every mode: a
  section is never drawn past its source pixels, an atlas plane at most four
  times past its voxels. The footer of a picture drawn at other than canvas
  pixels gives both scales: `canvas` (micrometres per canvas pixel) and
  `view` (micrometres per picture pixel, the number the caption calls
  um/px).
- **Scale.** A section's scale comes from the supplied pixel size (one for
  the stack, or one per file) or the file's TIFF or OME tags, else from the
  section image's width mapped onto the atlas plane's width (reported as
  `uncalibrated`). Positioning pictures and the paired opening tiles use the
  same micrometres per pixel for section and atlas, so tissue can appear
  smaller than the atlas. The stored affine contributes its isotropic scale,
  `sqrt(abs(det))`. Each section thumbnail is the whole image and each atlas
  thumbnail is framed around the atlas anatomy; neither is stretched to fill
  its tile.
- **Brain size.** The fits' distances were set on adult mouse sections. For
  another brain every distance of `elastix_affine`, `fireants_syn`,
  `make_mask` and the image-model trace is multiplied by the atlas brain's
  size factor: the cube root of its annotated brain volume over the adult
  Allen mouse's, rounded to 0.05. Measured factors: every adult CCFv3 atlas
  1, `whs_sd_rat_39um` 1.7, `swc_female_rat_50um` 1.4, DeMBA P4 0.7, P10
  0.9, P14 0.95. Distances on this page are the adult mouse's. What is
  scaled: the working pixels (20 µm in the adult mouse, 34 µm in the rat),
  so a rat section is fitted as a mouse section drawn smaller;
  `smoothness_um`; the edge channel's blur and the local-correlation
  window; the outline and damage margins; the restricted neighbourhood and
  its taper; a mask's EM contrast block and how far from the tissue it
  samples background; the image-model line softening, gap closing and
  smallest region; the displacement and smallest-region limits of the
  plausibility flags; and the positioning picture's smallest ruler margin
  and span. The EM's grid stays at the sampling at which holes, tears and
  debris are seen. A fit's reply reports its engine numbers at the job's
  scale (`core/atlas/size.py`).
- **`view`.** Every tool that changes the registration returns a picture of
  the result unless called with `view` false. A job with
  [`force_view`](settings.md#pictures) does not offer `view`, so the
  pictures always come.
- **Undo.** Every write is one undo step and is saved at once.
- **Background work and conflicts.** Work running in the background
  (`trace_borders`, a plugin's) holds its sections: any write to them is
  refused `SECTION_BUSY`, naming the work, until it lands. Long fits compute
  outside the job's write lock and check their inputs again before they
  apply: a section whose inputs changed meanwhile is `STALE_INPUT`, and the
  other sections' results still apply.

## Common tools

### `look`

Draws pictures of sections and the atlas. Writes nothing.

| Argument | Default | What it does |
|---|---|---|
| `mode` | (required) | `section`: each section alone, as oriented. `atlas`: the atlas plane at each of `positions_mm`, or at each named section's own position and cutting angles. `overlay`: each section under its current registration with atlas borders on it. `positioning`: sections and atlas planes along the slicing axis, as ABBA lays them out, sections joined by a line to their position; at most 6 sections per picture. |
| `sections` | every section | Filenames. |
| `positions_mm` | none | Atlas positions to draw (`atlas`, `positioning`). |
| `images` | `preprocessed` | One [section image](#make_image) alone in gray, or up to 6 raw channels together, each in its own colour. |
| `atlas_layers` | per mode | Atlas images (`template` and any other reference image the atlas ships) and `borders`. Default: borders in `overlay`, template in `atlas` and `positioning`. |
| `atlas_opacity` | 0.5 | How strongly an atlas image shows under the section in `overlay`. |
| `warp` | `applied` | `applied` draws each section with its deformation, `none` with its linear placement only. |
| `depth` | 6 | Atlas ontology depth whose region boundaries `borders` draws (0 is the root, as in BrainGlobe). |
| `masks` | none | Masks made with `make_mask`, outlined where their weight is 0.5. |
| `layers` | none | Layers left by tool plugins or scripts, or `["all"]`. |
| `registrations` | none | `{filename: number}`: draw a section under an earlier [registration](#restore_registration) of its own, writing nothing. |
| `resolution` | 0 (job's size) | Long edge in pixels. Offered only when the job's [`image_resolution`](settings.md#pictures) is `auto`. |
| `sheet` | false | Tile the pictures into sheets of up to 16. Agent CLI and library only. |

Atlas borders are drawn as cased lines: a black core with a white edge on
light tissue, a white core with a black edge on dark tissue; the caption
says which.

**Switch off:** cannot be switched off. Options: `look:section`,
`look:atlas`, `look:overlay`, `look:positioning` (one mode each; one must
stay). Without `overlay`, `atlas_opacity` and `warp` go; without `atlas` and
`positioning`, `positions_mm` goes; without `section` and `overlay`,
`layers` and `registrations` go.

**Details.** New jobs leave sections without supplied positions unplaced; no
position is inferred from filename order or the nominal interval. The opening
shows unplaced sections beside a separate overview of the atlas. In
`positioning` mode an unplaced section appears at the right, labelled "no
position", with no line to the ruler; `positions_mm` draws candidate atlas
planes above it.

`look(mode="atlas", sections=[...])` draws one atlas picture per section, at
that section's own position and cutting angles (captioned `atlas at 5.00 mm
... (s0.png's cutting angles)`); `positions_mm` given with `sections` is
refused `BAD_ARGS` unless it is the sections' own positions.

`registrations={filename: number}` (with `overlay` or `section`) draws each
named section under that earlier registration and writes nothing. The caption
says `registration 3 (not held)` (or `(held)`), the picture index records the
registration drawn, and a zoom of the picture draws it again without calling
it stale. With `sections` empty only the named sections are drawn; a named
section missing from `sections`, or another mode, is `BAD_ARGS`, and the
registration's own refusals apply (`UNKNOWN_REGISTRATION`,
`REGISTRATION_ELSEWHERE`, `DEFORMATION_UNREADABLE`).

`images` with several raw channels draws each in its own colour with automatic
contrast per section: black at the background and full colour at the tissue's
95th percentile on a dark background, the 1st and 99.5th percentiles on a
light one. There are no display settings: a stretched, inverted or
gamma-corrected view is a made image, which a fit can read too. Every caption
names the image drawn and what it is (its recipe, a script's note, or how the
opening image was made).

`borders` draws the boundaries between regions at one `depth` of the atlas
ontology; a highlighted region is outlined from the leaf annotation, whatever
its depth. A picture takes a black core with a white outline when its own
pixels under the lines are light (mean CIELAB lightness 50 or more), else a
white core with a black outline; the caption names it (`borders black,
white-edged`). Faint context lines and thick highlighted regions keep the same
arrangement. The image model's pictures keep plain yellow lines. `masks`
outlines each mask where its weight is 0.5 on `section` and `overlay`
pictures; the caption says that the ringed patches are what the mask leaves
out of a fit, and names each mask and its source.

`sheet=True` (agent CLI and library) also tiles the pictures drawn into sheets
of up to 16, each at most the viewer's largest picture: each tile is one
picture as drawn, scaled down whole, under a header naming its number. The
sheet is captioned with every tile's caption and saved as a picture of its own
(mode `sheet`), whose recipe lists each tile's picture and box. Every picture
is still saved and numbered (on the agent CLI its JPEG is an artifact of kind
`tile`, the sheet the one in `picture_files`); the reply's `pictures` lists
the sheets with their tiles and `tiled` the pictures they tile.

### `zoom`

Draws a pixel box, or one atlas region, of an earlier picture again at the
size of the picture. Writes nothing. A section or overlay picture is read
again from the original image file, at the file's own resolution. A
positioning picture is drawn from the working copies (the file reduced to at
most 3072 pixels). Every zoom carries a dashed green grid labelled with the
pixel numbers of the whole picture it was cut from; those numbers are the
coordinates of a further zoom and of a landmark endpoint.

| Argument | Default | What it does |
|---|---|---|
| `box` | none | `[x0, y0, x1, y1]` pixels of the picture. |
| `picture` | 0 (newest picture that is not a zoom) | The picture's number. |
| `region` | none | One atlas region on an overlay: crops to it with a small margin and draws only its border. Give `box` or `region`, not both. |

A zoom of a picture whose sections have changed since draws them as they
were and says `stale`.

**Switch off:** `zoom`. Without `look:overlay`, `region` goes.

**Details.** Without `picture`, the newest picture that is not a zoom is used.
A zoom of a tall or narrow box is padded with black beside its picture and
captioned below; the reply's `content_box` `[x0, y0, x1, y1]` is where the
picture itself lies, and `covers` `[x0, y0, x1, y1]` is the part of the whole
picture it shows. The grid's step is a round number from 1 to 5000 pixels
that gives at most twelve lines along the longer side, labelled at every
edge. `region` takes one acronym or numeric id, descendants included,
optionally with a side, and uses the saved picture's placement, deformation
and display settings; the region must occur in that plane. `zoom` on a sheet
takes the box on the tile holding most of it and zooms that tile's own
picture (the reply's `sheet` and `box`, the box in that picture's pixels);
`region` is refused on a sheet, and a box over no tile is `EMPTY_BOX`.

### `make_image`

Makes, replaces or deletes a named {term}`section image`: a grayscale picture
of each section that `look` draws and every fit reads by name. A section has
two kinds of section image. **Raw** images are the planes of the image file
(`red`, `green`, `blue` for a colour file, `gray`, or the channel names a host
gives, else `ch1`, `ch2`, and so on), never edited. **Made** images are made
here, by a script, or when the job opens, and saved as
`sections/<stem>/images/<name>.png`.

`preprocessed` is the made image LangSlice writes for every section when the
job is created, and the image every fit, the image model and `look` read by
default. With [`preprocess`](settings.md#pictures) `auto` it is each raw
channel weighted by its squared tissue coverage, CLAHE (clip 4, 8 by 8
tiles), brightened toward a mid-gray tissue mean; with `none`, the file as
read, in gray.

| Argument | Default | What it does |
|---|---|---|
| `name` | (required) | The image's name; not a raw channel's. An existing name is replaced. |
| `sections` | every section | Sections to make it for. |
| `weights` | each raw channel by how much tissue it lights | The images to blend and their weights, by name: `{"dapi": 1, "gfp": 0.3}`. |
| `clahe_clip` | 4 | Local contrast enhancement (CLAHE) clip limit, 0 (none) to 40. |
| `clahe_tiles` | 8 | CLAHE tiles per side, 1 to 32. |
| `n4` | false | Correct uneven illumination (N4). |
| `denoise` | false | Patch-based denoising. |
| `contrast` | none | `[low, high]`, fractions of the blended image's range, stretched to the full range. |
| `gamma` | 0 (linear) | 0.1 to 10. |
| `invert` | false | Bright as dark. |
| `delete` | false | Remove the image. `preprocessed` cannot be deleted. |
| `view` | true | Return up to 4 sections drawn as `look`'s `section` mode draws the image. |

The steps run in this order on each source: N4, denoising, CLAHE; then the
weighted blend, brightened to a mid-gray tissue mean; then `contrast`,
`gamma`, `invert`. Files are never deleted: a name made again with other
pixels gets a new file (`<name>.2.png`), so undo finds the old one.

**Switch off:** `make_image`. The raw channels and `preprocessed` stay.
Without `look:section`, `view` goes.

**Details.** For an ABBA job that sends preprocessing settings,
`preprocessed` is the host's channel blend. It is made once on the working
copy and saved as its file; from then on every reader (the fits, the image
model, `look`, zooms, masks) reads that file, resized to the size it works
at, so a fit reply's file and digest are exactly what the engine read.
`make_image("preprocessed")` with every argument at its default writes the
same file again. A recipe image is likewise computed once on the working copy
and read back from its file. `weights` empty is every raw channel by its
tissue coverage; an image cannot be its own source, and a raw channel's name
cannot be made. Naming `sections` remakes the image for them alone under the
same name. One undo step per call. Through the agent CLI and the library
`view` returns every section. Refusals: `BAD_ARGS`, `UNKNOWN_SLICE_IDS`,
`UNKNOWN_IMAGE`, `RENDER_FAILED`.

A script adds an image it computed with
`job.context.add_image(id, name, array, note="")`
([library](../developers/library.md#your-own-method)): an `(H, W)` array on
the section's working grid in the file's orientation, `uint8` or floats from
0 to 1, recorded with the note as its source, one undo step.

Every fit reply's `inputs.section_image` is `{name, kind, file, digest}` (for
a raw channel, the section's image file and no digest); the picture
`elastix_affine` and `fireants_syn` return is drawn on that image. A fit's
cache key holds the image's digest, so an image made again while a fit
computes makes that section `STALE_INPUT`. `status` lists the raw channels
and the made images (`section_images`: each name with its recipe, a script's
note or the opening image's recipe, `varies` with the sections that differ,
and `only` / `except` where it exists on some sections). No image changes
geometry: every image is held at the working copy's size.

The job statement says what the section images are and how `preprocessed` was
made, that a channel whose signal is sparse shows little of the anatomy the
atlas draws, which atlas images there are and which one the fits' default
`nissl` is, and that a fit compares one section image with one atlas image
and does best when both show as much internal structure as possible and look
alike.

### `grep_atlas`

Looks regions up in the atlas hierarchy by acronym, name or id. Text only.

| Argument | Default | What it does |
|---|---|---|
| `query` | empty | Text matched against acronyms and names, or an exact acronym or id. |
| `section` | none | A placed section: each row then says whether the region appears in that section's atlas plane. |
| `points` | none | Up to 40 `{picture, x, y}` points of saved pictures: the region drawn at each, and the regions containing it. A picture of the section alone draws no atlas and is refused (`NO_ATLAS_IN_PICTURE`). |

Returns at most 40 rows: acronym, id, name, ancestry and descendant count.

**Switch off:** `grep_atlas`. Option: `grep_atlas:points`.

**Details.** With `points` (`query` may be left empty), each point's answer is
the region's acronym, name and id, and `within`, the regions containing it,
nearest first (the root left out, at most six), or `region: null` with a note
when the point is outside the atlas or outside the picture's image content.
On a zoom, x and y are the numbers its grid prints and give the same answer as
on the whole picture. It reads what the picture drew: the saved `labels.tif`
of an overlay or a zoom of one, the section's atlas plane for an atlas picture
drawn with `look(mode="atlas", sections=[...])`, and the source picture of a
crop. An atlas picture not tied to a section is refused with
`NO_ATLAS_IN_PICTURE`, as is a picture of the section alone.

### `grep_atlas_view`

Draws the atlas at given positions with only the named regions' borders.
Writes nothing.

| Argument | Default | What it does |
|---|---|---|
| `regions` | (required) | Regions, descendants included; `"CTX:left"` names one side. |
| `positions_mm` | (required) | Positions; one outside the atlas is clamped into its range. |

**Switch off:** `grep_atlas_view`.

### `status`

Returns the stack as it stands. Writes nothing. Takes no arguments. The reply
holds:

- one row per section: filename, position, spacing to the next placed
  section, flip and quarter turn, transform, damaged regions and note,
  deformation and how it was made, the number of the registration it holds,
  masks, and frozen properties;
- the cutting angles and the interval breaks;
- the raw channels and the section images made;
- the background work still running;
- what this run lets the agent change.

**Switch off:** cannot be switched off.

**Details.** Every status row (here, and the rows of the sections a write
changed) gives the section's `transform`: its `kind`, the values
`interactive_transform` takes (`rotation_deg`, `scale_x`, `scale_y`, `shear`,
`translate_x_mm`, `translate_y_mm`, `pivot`) and `mirrored`; and the
`deformation` it holds: `method` (its last step's), `steps` (every step's
method, first first) and `settings` (what the last step was made with, by the
tools' names: a native fit's `section_image`, `atlas_image`, `mask`,
`smoothness_um`, `metric`, `edges`, `detail` and `restrict_to`; a plugin's or
a script's call settings). A section with landmark pairs carries `landmarks:
{total, pending, stale, warp}` (`pending` counts moved pairs too; `warp` says
in words whether the applied warp matches the pairs), and `can_change` names
the landmark tools when they are on. Replies give a transform's values under
`transform` everywhere.

### `list_files`, `search_files`, `read_file`

Read the job folder. Write nothing. The agent transcripts (`trace/`) are
never listed, searched or read. A picture file is answered with its entry in
the picture index (`views.jsonl`), never its pixels.

| Tool | Arguments (default) |
|---|---|
| `list_files` | `path` (`.`), `glob` (none): at most 200 entries |
| `search_files` | `query` (required; a regular expression, or plain text), `path` (`.`), `glob` (none): at most 60 matches |
| `read_file` | `path` (required), `offset` (0), `limit` (400; at most 2000 lines) |

**Switch off:** each by its own name.

**Details.** A path that does not exist is `NOT_FOUND`, naming the nearest
folder above it and what it holds; a path under `trace/` is refused
(`BAD_PATH`). A picture file's answer says how to see it again: `zoom` by its
number or, in a job without `zoom`, the `look` call that draws the view again
on the stack as it is now.

### `mark_damage`

Records the atlas regions a section has lost: regions whose tissue is missing
from the section, or lies folded over other anatomy, including one
hemisphere. Every fit and the image-model trace leave those regions out, so
intact tissue inside a mark goes unfitted. Small tears, bubbles, stains,
low contrast and folds within an intact outline are not marked. A section is
damaged exactly when it has marked regions. A note supplied in the job's
[`inputs.damaged`](settings.md#inputs) is information for the agent; only
`mark_damage` excludes regions.

| Argument | Default | What it does |
|---|---|---|
| `section` | (required) | One filename. |
| `regions` | (required) | Replace the section's marked regions; empty clears them. |
| `note` | empty | What is wrong with the tissue. A note the user gave stays first. |

Returns a picture of the marked regions shaded on the section beside the same
regions on the atlas.

**Switch off:** `mark_damage`.

**Details.** Its description tells the agent to leave intact tissue shifted
near its place unmarked, for the warp to fit; to move tissue shifted further
with landmarks; and that marking a whole large region when most of it is
missing keeps the surrounding anatomy safe.

### `make_mask`

Makes, replaces or deletes a named {term}`mask` of sections: a weight from 0
to 1 per section pixel, saved as `sections/<stem>/masks/<name>.npz`. A fit
that takes the mask by name leaves out the pixels under 0.5. The mask sorts
the section's pixels into tissue that matches the atlas image, artifact and
background, after Tward et al. 2025 (three-class EM). It is the only way a
fit leaves out a hole, a tear or a stain. The damage marks are separate and
always apply.

| Argument | Default | What it does |
|---|---|---|
| `name` | (required) | The mask's name. |
| `sections` | every section | Each section gets its own mask at its own current registration. |
| `source` | `em` | `em`: sort now. `fit`: copy the sorting the section's applied deformation saved. |
| `section_image` | `preprocessed` | The section image to sort (`em` only). |
| `atlas_image` | `nissl` | The atlas image to compare with (`em` only). |
| `sorting` | defaults | Starting points `match_sd`, `background_sd`, `artifact_sd`, `background_mean`, `artifact_mean` (`em` only). |
| `delete` | false | Remove the mask. |
| `view` | true | Return up to 4 sections with the mask's 0.5 outline. |

A mask stays usable after fits and moves (`status` marks it
`earlier_registration`); after the section is turned or flipped a fit
refuses it (`STALE_MASK`) until it is made again.

**In a run when** Linear or Nonlinear is on. **Switch off:** `make_mask`,
which also removes the `mask` argument of the fits and `look`'s `masks`.
**Needs:** `look:section`.

**Details.** A mask has one of two kinds. `em` is made at each section's
current registration: the atlas image placed through its transform and
applied deformation, on a 20 µm working grid. `fit` is copied from the sorting
the section's applied deformation saved when its engine estimated one
(`NO_DEFORMATION` or `NO_SORTING` when there is none). There is no threshold
mask and no drawn mask. `elastix_affine` and `fireants_syn` cut the weights at
0.5 and combine them with the section's own mask; a tool plugin's engine uses
them as its own limits say. With `mask="none"` every pixel inside the placed
atlas outline counts. A section that cannot get a mask (no position; no
deformation for `fit`) is that section's row and the others are written.
`status` lists each section's masks (name, kind, `stale`,
`earlier_registration`).

### `note`

Appends one line (`text`) to the run notes, saved with the results.

**Switch off:** `note`.

### `undo` and `redo`

`undo` takes back the last write; `redo` puts back what `undo` took back. A
tool call, or a finished piece of background work, is one step. Both take
`steps` (default 1): that many at once, or all the history holds. The history
is saved and survives a restart.

**Switch off:** `redo`. `undo` cannot be switched off.

**Details.** The reply's `undone` (or `redone`) holds the `sections` the steps
touched and the `fields` they changed, then those sections' status rows and the
remaining depth; with more than one step asked, `steps` (how many were taken)
and, when fewer, a `message`. Undo returns a section to an earlier fit only
while nothing else was written since; `restore_registration` returns one
section alone.

### `restore_registration`

Puts sections back on a {term}`registration <registration number>` they
held before: orientation, transform and deformation together. Each
registration a section holds gets a number, kept whatever is written after
it. Undo is one history for the whole job; this tool returns single sections
to any earlier registration and leaves the others as they are.

| Argument | Default | What it does |
|---|---|---|
| `section` | none | List this section's registrations (the tool that made each, when, its transform and deformation, its pictures). Writes nothing. |
| `sections` | none | `[{"id": filename, "registration": number}]`: put these back, one undo step. |
| `view` | true | Return each section's overlay. |

A deformation fitted with the section at another position or cutting angles
is refused (`REGISTRATION_ELSEWHERE`) until the section is back there.

**In a run when** Linear or Nonlinear is on. **Switch off:**
`restore_registration`. **Needs:** `look:overlay`.

**Details.** Each registration is numbered per section
([`registrations.jsonl`](file_formats.md#registrations-sectionsnameregistrationsjsonl)).
The section's status row says which it holds (`registration`), and so does
the reply to the write that left it. The listing gives, per number, the tool
that made it and when, its `transform`, its `deformation` (`method`, `steps`,
`settings`), `pictures` (the numbers of the pictures the write that left it
saved), the one held, and `not_here` for one fitted at another position or
cutting angles. A section that cannot go back is its own row (`results`) and
the others are written: `UNKNOWN_REGISTRATION` for a number it never held
(the row lists its numbers), `REGISTRATION_ELSEWHERE` (above), `FROZEN` for a
frozen property it would change, `SECTION_BUSY` while background work runs on
it. When no section can go back, the first row's code leads the reply. A
registration held before its section's list began (a job made by an older
LangSlice) is listed when a write replaces it, or when the tool lists the
section.

### `submit`

Ends the run. Waits for background work, checks the run's requirements,
then writes the QuickNII and VisuAlign exports and `registration.json`. It
writes no per-section maps (see `export_maps`). Refused, with nothing changed,
while something is missing:

- Positioning on: every section needs a position. With
  [`strict_interval`](settings.md#position-settings), positions may deviate
  at most 10% from the interval. A reported interval break must exceed 1.5
  times the median spacing.
- Linear on: every section needs a transform, damaged ones included.
- Nonlinear on: no deformation is required, unless the job sets
  [`require_deformation`](settings.md#nonlinear-settings); then every
  section needs one at its current placement (`MISSING_DEFORMATIONS`),
  except sections whose deformation is frozen. An image-model trace is never
  required.

| Argument | What it does |
|---|---|
| `summary` | One or two sentences on what the agent did. |
| `notes` | Short observations worth carrying forward. |
| `interval_breaks` | Filenames of the sections after a gap the agent concludes is real. |

**Switch off:** cannot be switched off.

### `export_maps`

Writes the job folder's derived files from the stack as it stands, submitted
or not: each placed section's coordinate and label maps (in
`sections/<stem>/`), the QuickNII and VisuAlign exports, `registration.json`,
and each traced section's image-model reply. Arguments: `sections` (default
every section) and `full_resolution` (default false: the working copy's grid;
true: the image file's own pixels). Changes no registration. Offered on the
agent CLI and in the library only. See
[file formats](file_formats.md#per-section-maps-sectionsname).

**Switch off:** `export_maps`.

## Positioning tools

### `position_sections`

Sets where sections sit along the slicing axis, and the stack's cutting
angles. A value outside the atlas range is clamped into it and reported. The
stack's order follows the positions. A changed position or angle clears the
deformation of each section it moves.

| Argument | Default | What it does |
|---|---|---|
| `sections` | none | `[{"id": filename, "position_mm": number}]`. Offered only with the Positioning task. |
| `cutting_angles` | unchanged | `{"pitch_deg": number, "yaw_deg": number}`: one plane for the whole stack. |
| `view` | true | Return a positioning picture of the written sections. |

How a positive angle tilts the plane, with the edges as the pictures show
them (a negative angle tilts it the other way):

| Plane | Positive pitch puts at a larger position | Positive yaw puts at a larger position |
|---|---|---|
| coronal | the top edge (than the bottom) | the right edge (than the left) |
| sagittal | the left edge (than the right) | the bottom edge (than the top) |
| horizontal | the right edge (than the left) | the top edge (than the bottom) |

An edge D mm from the picture's centre moves about D x tan(angle) mm along
the slicing axis: 10 degrees moves an edge 4 mm from the centre about 0.7 mm.

Sections without a supplied position stay unplaced until the agent writes
them; no position is inferred from filename order or the nominal interval.
`cutting_angles` is refused while any section has its angles frozen.

**In a run when** Positioning is on, or Linear is on with
[`transform.angles`](settings.md#transform-settings) (then only
`cutting_angles`). **Switch off:** `position_sections`. **Needs:**
`look:positioning` or `look:overlay`.

## Linear tools

### `interactive_transform`

Sets sections' orientation and in-plane transform by hand and shows the
result. Values are absolute: a value given replaces the section's current
one, and a value left out keeps it.

| Key of each section | What it does |
|---|---|
| `id` | Filename. |
| `flip` | Mirror the section left-right. |
| `rotate_quarter` | 0, 90, 180 or 270 degrees counter-clockwise. Applied before the flip. |
| `rotation_deg` | Degrees counter-clockwise about the canvas centre. |
| `scale_x`, `scale_y` | Multipliers; 1.0 keeps the size. |
| `shear` | A slant applied before the rotation. |
| `translate_x_mm`, `translate_y_mm` | Shift right and down. |

`view` (default true) returns each section under its new transform with the
atlas borders. A changed orientation keeps the other values: the transform is
rebuilt on the new orientation. Changing the orientation or transform clears
the section's deformation.

**In a run when** Linear is on. **Switch off:** `interactive_transform`.
Options: `interactive_transform:flip`, `interactive_transform:rotate`.
**Needs:** `look:overlay`.

### `elastix_affine`

Refines each section's current placement automatically with
[Elastix](https://elastix.dev), matching a section image against an atlas
image. It adjusts from where the section is and does not search from scratch;
it does not mirror a section. The section's marked damage regions are left
out. A section that holds a deformation is refused (`HAS_DEFORMATION`).

| Argument | Default | What it does |
|---|---|---|
| `sections` | (required) | Filenames. |
| `restrict_to` | every region | Fit by these regions only: the atlas within 300 µm of them. |
| `section_image` | `preprocessed` | The section image to match. |
| `atlas_image` | `nissl` | One of the atlas's images; `nissl` is its Nissl image where it has one, else its template. |
| `mask` | `none` | A mask made with `make_mask`. |
| `edges` | true | Also match the edges (gradient magnitude) of both images, at equal weight. |
| `view` | true | Return each fitted section's overlay. |

A fit that changes the scale along either axis by more than a factor of 0.8
to 1.25 is flagged `SCALE_CHANGED`, one that changes the shear by more than
0.15 `SHEAR`, and one that turns the section more than 45 degrees says so. A
region too small for Elastix is refused (`REGION_TOO_SMALL`).

**In a run when** Linear is on. **Switch off:** `elastix_affine`.
**Needs:** `look:overlay`.

**Details.** Each ok row carries the `inputs` it read and `estimated: null`.
Each flag is also worded in the row's `warning`; a section that holds only
part of the brain (a hemisphere) can be stretched over atlas it does not hold.
For a deformed section, undo the deformation first or fit a warp on it. A call that fits nothing answers with the code and sentence every
section was refused for when they share one (`HAS_DEFORMATION`, `FROZEN`),
else `NOTHING_FITTED`, each row naming its own; `fireants_syn` answers the
same way. A restricted fit highlights those regions' borders and may zoom its
picture to them; other borders stay faint. A region is too small when its
support outside the damage marks spans fewer than four native atlas pixels in
either in-plane direction or holds fewer than sixteen pixels, checked before
any padding or fitting; use a larger region or `interactive_transform`.
Under a frozen `orientation`, a fit keeps the section's handedness (the sign
of its transform's determinant), and a fit that would mirror it is refused; a
frozen `affine` is refused by name.

## Nonlinear tools

### `fireants_syn`

Fits the atlas onto sections with ANTs SyN (symmetric normalization), run by
[FireANTs](https://github.com/rohitrango/FireANTs) on the GPU when there is
one, else on the CPU. Each fit starts from the section's current registration,
so fits build on each other; undo goes back. A piece of tissue displaced more
than about 0.4 mm from its place is not moved by SyN; landmarks move it. Needs
a position and a transform.

| Argument | Default | What it does |
|---|---|---|
| `sections` | (required) | 1 to 20 filenames, fitted one after another. |
| `restrict_to` | every region | Refine these regions only: a 0.3 mm neighbourhood around them is fitted, and the change fades to none over a further 0.3 mm. |
| `section_image` | `preprocessed` | The section image to match. |
| `atlas_image` | `nissl` | As for `elastix_affine`. |
| `mask` | `none` | A mask made with `make_mask`. |
| `smoothness_um` | 80 | Smoothing of each update of the displacement field, in µm; the whole field is smoothed a quarter as much. Larger bends less. |
| `metric` | `mutual_information` | Or `local_correlation`: how the two images are compared. |
| `edges` | true | Also match the edges of both images, at equal weight. |
| `detail` | `standard` | `coarse`, `standard` or `fine` (10 µm working pixels, five pyramid levels): finer is slower. |
| `view` | true | Return each section's overlay. |

The reply gives displacement, fold fraction and plausibility flags (regions
compressed, expanded, vanished or folded beyond limits; outsized
displacement), which report concerns and reject nothing. Changing a section's
position, orientation, cutting angles or transform clears its deformation.

**In a run when** Nonlinear is on. **Switch off:** `fireants_syn`. Options:
`fireants_syn:metric`, `fireants_syn:edges`, `fireants_syn:detail` (the
argument goes and keeps its default). **Needs:** `look:overlay`.

**Details.** The engine is FireANTs' SyN with ANTs' step rule: mutual
information (or local correlation) plus an edge channel, four pyramid levels
(five at `fine`), `smoothness_um` as the Gaussian smoothing of each step and a
quarter of it for the whole warp; the step size, the iterations per level and
the pyramid are fixed. About a second per section on a GPU. Identical inputs
give identical fields. A fit runs once and measures the section inside the
placed atlas outline widened by 0.2 mm (for a restricted fit, only within the
regions' neighbourhood), less the section pixels a named mask's weights put
under 0.5; `elastix_affine` and the fit of traced borders do the same. Marked
damage is left out, with a margin. A restricted fit's masks limit where it is
measured, not where its smooth field reaches: the change fades to none over a
further 0.3 mm and the tissue beyond keeps its earlier fit.

`atlas_image` takes any of the atlas's images: BrainGlobe's `template` and
every additional reference the atlas ships, under BrainGlobe's names. The
alias `nissl`, the default, is the atlas's Nissl image where it has or
borrows one (the template of `ccfv3augmented_mouse_25um`, which is a Nissl;
an additional reference whose name contains "nissl"), else its template. An
Allen CCFv3 atlas, which ships no Nissl (one whose BrainGlobe metadata names
the `allen-adult-mouse-ccf-space` coordinate space, or names none and has its
grid), is also offered `ccfv3augmented_nissl`, the Blue Brain averaged Nissl
template of `ccfv3augmented_mouse_25um`, downloaded on first use; `nissl`
resolves to it. The reply names the image it resolved to, with
`atlas_image_note` where `nissl` was honoured by the Nissl template of
`ccfv3augmented_mouse_25um`. The job statement lists the images, quotes
BrainGlobe's description of the atlas, and asks the agent to fit against the
image whose contrast is closest to the section's and that lines up with the
borders.

The reply carries `settings` (the call) and, per section, `inputs` (the
section image `{name, kind, file, digest}`; the atlas image as asked and as
resolved; the mask, always an object: `name` (`"none"` for no mask), `kind`,
`stale`, `earlier_registration`; the damage regions and the margin left
around them; the regions restricted to; the brightness scaling in words) and
`estimated` (null: FireANTs sorts nothing itself); the deformation record's
provenance holds the same blocks. The reply names what was used and nothing
about how well it worked. The outsized-displacement flag counts only what the
fit adds beyond the section's landmark warp, so a deliberate landmark move is
never flagged. Up to 20 sections per call because every fitted section is held
in memory until the call writes them all as one undo step (20 sections peaked
at 6.4 GB).

A fit that waited a second or more for the machine's GPU queue (one fit at a
time across every job on the machine) says how long in its row
(`gpu_wait_s`); one that found the GPU's memory taken by another process,
waited and ran again says so (`retried`). What the engines print goes to the
job's `logs/engine.log`, never into the reply. A fit whose engine fails is
that section's row `FIT_FAILED`: its message is the error's first line (a GPU
out of memory said in words: running the call again fits it from the start)
and says that `logs/engine.log` holds the whole error. The same holds for
`elastix_affine`, `make_mask` and the library's `fit_fireants` and
`fit_elastix`.

### `set_landmarks`

Adds, updates and deletes {term}`landmark` pairs of one section: each pair is
a point on the atlas and the point on the section that corresponds to it. It
applies no deformation; `landmark_triangulation` does.

| Argument | Default | What it does |
|---|---|---|
| `section` | (required) | Filename. |
| `pairs` | none | `{atlas: {picture, x, y}, section: {picture, x, y}}`; an update adds `id` and either endpoint. At most 4 with `view` true. |
| `delete` | none | Pair ids, or `["all"]`. |
| `view` | true | One side-by-side crop per edited pair: atlas template at left, `preprocessed` section at right. |

Endpoints are pixels of a numbered picture: an overlay or section picture for
the section point, an overlay or `look(mode="atlas", sections=[...])` picture
for the atlas point. An atlas endpoint outside the atlas is refused with the
nearest point inside it. Pair ids are never reused. A change of the
section's position or cutting angles keeps the pairs and marks their atlas
endpoints stale.

**In a run when** Nonlinear is on. **Switch off:** only together with
`landmark_triangulation`. **Needs:** `look:atlas` or `look:overlay`;
`look:section` or `look:overlay`; `landmark_triangulation`.

**Details.** Endpoints are x,y pixels of a numbered picture, from the top-left
of the whole image, panel offsets included. On a zoom, x,y are the numbers its
grid prints, so a number read off any grid works whether the call names the
zoom or the whole picture; a point the zoom does not show is refused with the
range it covers. An atlas endpoint outside the atlas is refused before
anything is written (`ATLAS_ENDPOINT_OUTSIDE`), naming the pair and giving
the nearest point 1.5 atlas pixels inside the outline. A generic atlas
picture drawn by `positions_mm` has no section and cannot supply endpoints.
When MCP shrinks pictures to fit its reply budget, the delivered sizes get new
picture numbers with exact geometry; earlier numbers still refer to the
originals.

Omitted endpoints of an update stay as they were; `delete=["all"]` clears the
section's pairs as one step. Every reply lists the section's pairs (`id`,
`atlas` and `section` as `{picture, x, y}`, and a `state`) with their counts:
`applied` (in the landmark warp now applied, with these endpoints), `moved`
(in it, with other endpoints now), `pending` (saved, not in it) or `stale`
(placed against an atlas plane the section has left; `stale` wins). A `warp`
block gives `built_from`, `not_applied`, `deleted_still_in_warp` and a `note`
that `landmark_triangulation` must run again. A call with no pairs and no
`delete` writes nothing and returns this listing. The reply names the region
drawn at each atlas endpoint it placed (`atlas_endpoint_regions`). A refused
endpoint names the pair (its index and id), the endpoint, the picture and the
x,y, and says what to do: a point outside the image content gives the
picture's content box, a picture without geometry for the section or drawn at
an earlier atlas plane says which picture to draw again, and
`TOO_MANY_LANDMARK_VIEWS` says to resend with `view=false` or in calls of
four. A section whose `deformation` is frozen is refused (`FROZEN`) by both
landmark tools.

On the agent CLI, `pairs` goes through `--args`, as a JSON object or a file:

```bash
langslice-job JOB set_landmarks --args @pairs.json     # {"section": "s01.tif", "pairs": [...]}
langslice-job JOB set_landmarks --args '{"section": "s01.tif", "delete": ["L2"]}'
langslice-job JOB zoom --picture 12 --box 960,230,1300,940   # or --box '[960,230,1300,940]'
```

The `view` crop carries the atlas region borders as `look` draws them. A zoom
of it is a saved-image crop and reports `redrawn=False`. Later section views
show open-centre circles, atlas views diamonds, and overlays both, with
matching ids, outlined in white and black. Markers are drawn after
compositing and never enter fit images or image-model inputs; older pictures
keep their original markers. Editing pairs, or marking damage, does not make
an older picture `stale`; a change to the placement or the registration does.
Appearance changes leave pairs valid; section endpoints are stored in the
original image's normalized frame and survive orientation changes, and
linear transform changes keep both endpoints.

The tool's description advises placing both endpoints of a pair on one
overlay (or a zoom of it) where possible, the atlas point where a border is
drawn and the section point where that border belongs; after a warp, an
overlay shows the section warped onto the fixed atlas and a point placed
there is read through the warp. Endpoints are best read off a close zoom with
grid lines 10 to 20 pixels apart. Pairs are spread over the outline and the
internal anatomy (the midline, a ventricle, a fibre tract). An atlas region
with no tissue on the section takes no pair (`mark_damage` records it); a
region only partly present keeps its pairs on the part that is present;
tissue with no atlas counterpart takes none.

### `landmark_triangulation`

Applies the section's saved landmark pairs as a piecewise affine warp, built
as [VisuAlign](https://www.nitrc.org/projects/visualign/) builds it: a
Delaunay triangulation of the pairs' section points plus four fixed corners
10% outside the image. Needs at least three pairs and a linear placement.
Each pair moves the tissue around it out to the neighbouring pairs and the
corners, so tissue that is already right takes pairs too: a pair with both
endpoints on one point of an overlay keeps that spot where it is. Folded
triangles are applied, as VisuAlign applies them, and reported. The warp is
composed onto the current deformation; a later application replaces the
earlier landmark warp.

Arguments: `section` (required), `view` (default true), and `resolution` when
the job's `image_resolution` is `auto`.

**In a run when** Nonlinear is on. **Switch off:** only together with
`set_landmarks`. **Needs:** `look:overlay`; `set_landmarks`.

**Details.** The triangulation is of the pairs' section points in their pixel
aspect ratio, with the four corners 10% outside the frame, and a piecewise
affine pull-back to the matching atlas points (`core/triangulation.py`). A
pair whose two endpoints are one point of an overlay corresponds to that spot
against the baseline the warp is applied on, so it keeps it where it is. A
pair on the outline with no pair beside it drags distant anatomy through the
large triangles that reach the corners. LangSlice adds no points of its own. A
state whose landmark warp carries automatic holds (`landmark_holds`, fold
vertices named `~K`) still opens: those fields are ignored. The inverse uses
the same triangles. Repeated section points (the reply names the pairs) and
zero-area source triangles are refused.

A triangle whose atlas points wind the other way from its section points (a
fold) is applied and the forward map is unchanged. Where target triangles
overlap, the inverse takes the unfolded triangles first and then the folded
ones, each in mesh order, so the result is deterministic. The reply reports
`folds`: `area_fraction` (the share of the frame covered by folded
triangles), `pairs` (the ids at their corners), the first triangles with
their pair ids, and `triangle_count`; the result picture outlines each folded
triangle in magenta. It lists `built_from` and, since `displacement_mm` is the
whole deformation now applied, `step_displacement_mm`, the change this
application made (`displacement_note` says so).

The first application composes the landmark warp onto the current
deformation. Later applications replace that landmark warp on the same
earlier baseline, so editing or deleting a pair removes its influence. A
FireANTs or trace fit in between becomes the baseline for the next
application. A missing baseline inverse or an unreadable deformation is
refused. Reapplying an unchanged set writes nothing. Editing or deleting
pairs leaves the current registration in force until the next application;
undo restores the registration before an application.

### `trace_borders`

Asks the {term}`image model` to draw the placed atlas borders onto the
section's anatomy, then fits a FireANTs deformation to the drawn lines. The
model receives two images in the same frame: the section's `preprocessed`
image, turned upright and cropped to the placed atlas plus 15%, and the same
image with thin yellow atlas borders. Needs a position and a transform. The
section's marked damage regions are left out.

| Argument | Default | What it does |
|---|---|---|
| `section` | (required) | Filename. |
| `prompt` | the base prompt | The full image prompt for this section, edited from the base prompt. |
| `restrict_to` | every region | Trace and fit these regions only. |

The base prompt is LangSlice's for the provider, or the job's own
[`nonlinear.image_prompt`](settings.md#nonlinear-settings). The work runs in
the background: the tool returns at once, later replies deliver the result,
`status` lists the work still running, and `submit` waits for it. A write to
the section is refused (`SECTION_BUSY`) until the work lands. The first answer
at a placement and region choice is saved and reused. The job folder keeps
the prompts, the raw reply, the extracted lines and the fit.

**In a run when** Nonlinear is on, the job's `nonlinear.provider` is not
`none`, and (on MCP and the agent CLI) the image model's key or login is on
this computer. **Switch off:** `trace_borders`, or set
`nonlinear.provider` to `none`. **Needs:** `look:overlay`.


**Details.** The section is turned upright on the model's frame from its
linear placement: the atlas's dorsal side (the anterior end on a horizontal
section) points straight up, and on a coronal or horizontal section the atlas
midline runs down the middle. A sagittal section is only turned. It is a turn
and shift, never a mirror. The crop is the placed atlas's extent plus 15% of
its larger side on every side, never past the section image, and on a coronal
or horizontal section widened to stay symmetric about the midline, padded
with the slide's background colour where it runs past the image. Turn, crop
and scale are one resample, and the model's lines are read back through them.

The model's outer line sets the section's extent: the fit pulls the atlas in
or out to it, wherever the linear placement left the atlas edge. The fit of
traced lines steps half as far as a stain fit, smooths the whole warp twice
as wide, and gives every traced region and the lines an equal pull (each
one's gradient scaled to the same length, as ANTs' multi-metric rule does),
so no single large region or stray line drives the warp. A completed fit is
its own undo step. Identical inputs reuse the saved reply; a changed
placement or section image needs a new trace, and a result whose inputs
changed while it ran is refused `STALE_INPUT`. The prompt tells the model
that the atlas decides which borders exist, even where tissue boundaries are
faint, and that only physically missing tissue justifies leaving one out.

Inspect the raw model reply, then the extracted lines on the original
photograph, then the fitted overlay: a model may redraw tissue, a fit may
degrade a useful trace, yellow tissue can confuse the line extractor, and fit
numbers do not certify region identity. `export_maps` exports each traced
section's last reply and its lines on the maps grid
([`exports/traces/`](file_formats.md#tracesnamenpz)). With
`nonlinear.provider` `none` the FireANTs and landmark tools stay. Custom image
models and prompts are in the
[library guide](../developers/library.md#image-models).

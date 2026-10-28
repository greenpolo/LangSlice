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
- **Pictures** are numbered, captioned and saved in the job folder. A
  later tool names a picture by its number, and points on it by `x`, `y`
  pixels from its top-left corner. Picture size changes what is shown, never
  a fit or a stored transform. Image files are never modified.
- **Brain size.** The fits' distances were set on adult mouse sections. For
  another brain every distance of `elastix_affine`, `fireants_syn`,
  `make_mask` and the image-model trace is multiplied by the atlas brain's
  size factor: the cube root of its annotated brain volume over the adult
  Allen mouse's, rounded to 0.05. Measured factors: every adult CCFv3 atlas
  1, `whs_sd_rat_39um` 1.7, `swc_female_rat_50um` 1.4, DeMBA P4 0.7, P10
  0.9, P14 0.95. Distances on this page are the adult mouse's.
- **`view`.** Every tool that changes the registration returns a picture of
  the result unless called with `view` false. A job with
  [`force_view`](settings.md#pictures) does not offer `view`, so the
  pictures always come.
- **Undo.** Every write is one undo step and is saved at once.

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

### `grep_atlas`

Looks regions up in the atlas hierarchy by acronym, name or id. Text only.

| Argument | Default | What it does |
|---|---|---|
| `query` | empty | Text matched against acronyms and names, or an exact acronym or id. |
| `section` | none | A placed section: each row then says whether the region appears in that section's atlas plane. |
| `points` | none | Up to 40 `{picture, x, y}` points of saved pictures: the region drawn at each, and the regions containing it. A picture of the section alone draws no atlas and is refused (`NO_ATLAS_IN_PICTURE`). |

Returns at most 40 rows: acronym, id, name, ancestry and descendant count.

**Switch off:** `grep_atlas`. Option: `grep_atlas:points`.

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

### `note`

Appends one line (`text`) to the run notes, saved with the results.

**Switch off:** `note`.

### `undo` and `redo`

`undo` takes back the last write; `redo` puts back what `undo` took back. A
tool call, or a finished piece of background work, is one step. Both take
`steps` (default 1): that many at once, or all the history holds. The history
is saved and survives a restart.

**Switch off:** `redo`. `undo` cannot be switched off.

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

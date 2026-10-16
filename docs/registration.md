# Registration tools

LangSlice gives an agent a stack of sections, an atlas and tools for looking,
placing and fitting. The agent inspects the anatomy, chooses tools and
regions, reviews their pictures and revises the registration. Image-model
border tracing is optional; affine and FireANTs fits can use the stain directly.

The tools have the same names across [CLI, MCP and agent sessions](agents.md)
and the [Python library](library.md). Use `langslice-job schema VERB --job
FOLDER` for exact arguments and limits.

## Tasks and supplied inputs

| Task | Tools | Result |
|---|---|---|
| `position` | `position_sections` | Atlas positions, section order and cutting angles |
| `transform` | `interactive_transform`, `elastix_affine` | In-plane alignment of each section |
| `nonlinear` | `fireants_syn`, `set_landmarks`, `landmark_triangulation`, optionally `trace_borders` | Deformation on top of a linear placement |

Default tasks are `position,transform`. A disabled task takes its answer
from supplied inputs and removes its tools. `transform.angles` also enables
`position_sections` for cutting angles when Positioning is off. A disabled
task is the same as freezing its properties on every section
([freezing](#freezing-what-the-agent-may-not-change)): `position` writes
position and angles, `transform` orientation and affine, `nonlinear`
deformation.

## Switching tools and options off

A job's `disabled` list removes single tools and tool options on top of what
its tasks give (`--disable NAME[,NAME...]`, `disabled=[...]` in the library
and in `job.json`'s spec). A name is a tool, or `tool:option`:

| Name | Removes |
|---|---|
| `zoom`, `set_channel_properties`, `set_preprocessed_channel_properties`, `grep_atlas`, `grep_atlas_view`, `list_files`, `search_files`, `read_file`, `note`, `redo` | the tool |
| `position_sections`, `interactive_transform`, `elastix_affine`, `fireants_syn`, `trace_borders`, `mark_damage` | the tool, as its task or host switch would |
| `set_landmarks`, `landmark_triangulation` | the landmark pair: each needs the other, so both are disabled together |
| `export_maps` | the scripting verb |
| `look:section`, `look:atlas`, `look:overlay`, `look:positioning` | that `look` mode |
| `grep_atlas:points` | the region lookup at points of saved pictures: the `points` argument of `grep_atlas` |
| `interactive_transform:flip` | mirroring: the `flip` key of its sections |
| `interactive_transform:rotate` | the quarter turn: the `rotate_quarter` key of its sections |

The tools of a job's [plugins](plugins.md) are valid names too, for that job.
`look`, `status`, `undo` and `submit` cannot be disabled. An unknown name is
refused when the job is made, with the valid names listed. A list that cannot
work is refused with one line saying why:

- a task that is on keeps at least one tool that writes its answer
  (`position_sections`; `interactive_transform` or `elastix_affine`;
  `fireants_syn`, `landmark_triangulation` or `trace_borders`);
- `look` keeps at least one mode;
- `set_landmarks` and `landmark_triangulation` need each other; the first
  also needs a way to see atlas points (`look:atlas` or `look:overlay`) and
  section points (`look:section` or `look:overlay`), the second `look:overlay`;
- a tool needs the pictures it returns: the linear and nonlinear fitting
  tools and `interactive_transform` need `look:overlay`, and
  `position_sections` needs `look:positioning` or `look:overlay`.

What a job disables is absent from every door: the tool is not built, an
option is not in the tool's schema (`look`'s `mode` offers the modes that are
on; `flip` is not a key of the sections), and a call that names one anyway is
refused (`OPTION_OFF` from `look`). A reply that would use a disabled option
uses another: `position_sections` returns an overlay of each written section
when `look:positioning` is off. The job statement, tool descriptions,
reference card, `status` reply and `error.fix` texts leave out every
sentence that needs what is off, or word it another way.

Sections are addressed by filename, or by a stem unique within the job.
Positions are millimetres along the atlas's slicing axis; in a coronal atlas
with BrainGlobe's `asr` orientation, they run from the anterior edge toward
the posterior. Ordering follows the positions. Cutting angles are pitch and
yaw in degrees. A host can supply positions, per-section or stack-wide
angles, orientation, transforms and pixel size, or import a
[registration file](file_formats.md#importing-a-registration).

Notes supplied through `damaged` are information for the agent; actual
damage exclusions are atlas regions marked with `mark_damage`.

## Freezing what the agent may not change

A job is configured by three settings: what is already known
(`inputs`, above), what is frozen, and which tools exist (`disabled`, above).
`frozen` names, per section, the properties the agent may not change. A
section has five:

| Property | What it is | Kept when `inputs` supplies none |
|---|---|---|
| `position` | Where it sits along the slicing axis | Nothing: a frozen position must be supplied in `inputs.positions` |
| `angles` | Its cutting angles (pitch, yaw) | The flat plane |
| `orientation` | Its flip and quarter turn, together | No flip, no turn |
| `affine` | Its continuous in-plane transform | The host's identity transform (`kind` `host`) |
| `deformation` | The deformation on top of its linear placement | None |

```json
{"frozen": {"s_007.tif": ["orientation", "affine"], "*": ["angles"]}}
```

A filename names one section, `"*"` every section; a section's own list and
`"*"` add up. From the command line, `--frozen JSON`; in a library call,
`frozen=`. The properties and filenames are checked when the job is made.

Three common freezes:

- Positions, cutting angles and orientation known, the continuous affine and
  the deformation left to the agent:
  `{"*": ["position", "angles", "orientation"]}` with `inputs.positions`,
  `inputs.angles` and `inputs.orientation` and the tasks `transform,nonlinear`.
- A section the host already aligned and does not want touched:
  `{"s_007.tif": ["orientation", "affine"]}` (its transform is the supplied
  one, else the identity).
- A section left out of Nonlinear: `{"s_007.tif": ["deformation"]}`.

Every write that would change a frozen property is refused, in every door,
with `FROZEN` naming the sections and properties; nothing is written for the
refused section, and the others in a per-section call still are
(`position_sections` and `set_transforms` refuse the whole call). The
partial cases:

- A fit under a frozen `orientation` keeps the section's handedness: the
  sign of its transform's determinant stays the sign of the transform it
  replaces (positive without one, negative for a supplied mirrored
  transform). `elastix_affine` and `interactive_transform` refuse a fit or
  a scale that would mirror the section.
- A frozen `affine` is never rewritten: `elastix_affine` refuses it by name, `interactive_transform`
  refuses a changed knob (the same numbers again write nothing, and the
  orientation may change under it when it is free). A section whose
  `deformation` is free is still deformed on its supplied transform.
- The cutting angles are one plane for the stack, so `position_sections`
  refuses its `cutting_angles` while any section has `angles` frozen.
- A frozen `deformation` needs no deformation to submit, even where the host
  requires one on every section.

A task whose properties are all frozen on every section is off
(`position`: position and angles; `transform`: orientation and affine;
`nonlinear`: deformation), and a task that is switched off freezes its
properties on every section: "the host supplied this" is one rule. A tool
that writes only properties frozen on every section is not offered
(`position_sections`, `interactive_transform`, `elastix_affine`, `fireants_syn`,
`landmark_triangulation` and `trace_borders` by what they write; `set_landmarks`
goes with `landmark_triangulation`, whose deformation its pairs serve), `interactive_transform`'s `flip` and
`rotate_quarter` go with a frozen `orientation` on every section, and no text names what is gone. The job
statement says what is frozen, grouping sections that share a freeze (`every
section`, their filenames, or their count when many), and each `status` row
of a section with something frozen carries `frozen`.

Saved jobs from before this setting read their `inputs.locked`,
`inputs.keep_warp` and `inputs.nonlinear_skip` as the freezes they stood for
(`orientation` and `affine`; `deformation`; `deformation`). A job resumed with
a different `frozen` is refused like one resumed with other `inputs`.

## Looking and calibration

| Tool | Use |
|---|---|
| `look` | `section`: tissue alone; `atlas`: atlas planes; `overlay`: tissue with registered atlas borders; `positioning`: sections and atlas along the slicing axis |
| `zoom` | Redraw a saved picture's pixel box, or an overlay's single atlas region |
| `grep_atlas` | Find region acronyms, names and ids; with `points`, the region drawn at given points of saved pictures |
| `grep_atlas_view` | Show the atlas with only the selected regions' borders |
| `status` | Read placements, settings, constraints and pending work |
| `list_files`, `search_files`, `read_file` | Inspect the job folder |

Every picture is numbered, captioned and saved. `zoom(picture=N,
box=[x0,y0,x1,y1])` redraws its selected area; omitting `picture` uses the
latest non-zoom picture. Section and overlay zooms reread the source image
at its native resolution; positioning zooms use the working copies. A zoom
of a tall or narrow box is padded with black beside its picture and
captioned below; the reply's `content_box` `[x0, y0, x1, y1]` is where the
picture itself lies. Every zoom carries a thin dashed green coordinate grid
whose labels are the pixel numbers of the whole picture the zoom was cut from
(a round step, 1 to 5000 px, that gives at most twelve lines along the longer
side; labels at every edge), and the reply's `covers` `[x0, y0, x1, y1]` is the
part of that picture it shows. Those numbers are also the coordinates of a
further zoom's box on the zoom and of a `set_landmarks` endpoint on the zoom or
on the whole picture alike.

`grep_atlas(points=[{"picture": N, "x": 640, "y": 410}])` (at most 40 points;
text only, nothing written; `query` may then be left empty) says which atlas
region is drawn at each point: its acronym, name and id, and `within`, the
regions containing it, nearest first (the root left out, at most six), or
`region: null` with a note when the point is outside the atlas or outside the
picture's image content. x and y are read as everywhere else, so on a zoom they
are the numbers its grid prints and give the same answer as on the whole
picture. It reads what the picture drew: the saved `labels.tif` of an overlay
or a zoom of one (the fixed atlas the section is warped onto), the section's
atlas plane for an atlas picture drawn with `look(mode="atlas",
sections=[...])`, and the source picture of a crop. A picture of the section
alone draws no atlas and is refused with `NO_ATLAS_IN_PICTURE`, as is an atlas
picture not tied to a section. `grep_atlas:points` switches the argument off.

For an overlay, `zoom(region="CA1:left")` crops to the entire selected
region with a small margin and shows only that region's border. It accepts
one acronym or numeric id, including descendants, optionally with a
[side](#sides). It uses the saved picture's placement, deformation and display
settings. The region must occur in that plane; `region` and `box` are
mutually exclusive.

### Sides

A region may be named with one half of the brain, `"CTX:left"`. The halves
come from the atlas's midline carried through the section's placement, never
from the tissue, so a mirrored placement swaps them. Where the midline runs
closer to vertical on the picture, the halves are `left` and `right`; where it
runs closer to horizontal, as on a horizontal section shown with the
anterior-posterior axis across, they are `top` and `bottom`. Naming the pair
the picture does not have is refused with `WRONG_SIDE_PAIR`, whose message
names the pair to use; a side on a midline at 45 degrees is refused with
`SIDES_AMBIGUOUS`. A
sagittal plane lies in one hemisphere and has no sides (`NO_SIDES`).

New jobs leave sections without supplied positions unplaced. The opening
shows those sections alongside a separate overview of the atlas. No positions
are inferred from filename order or the nominal interval. In positioning
mode, unplaced sections appear at the right, labelled "no position", with
no line to the ruler; `positions_mm` selects candidate atlas planes above it.
Supplied positions and saved registrations are preserved.

Section scale comes from supplied calibration (one size for the stack, or one per file) or TIFF/OME metadata, else the
section image's width is mapped onto the atlas plane's width (reported as `uncalibrated`). Positioning pictures and paired
opening tiles use the same micrometres per pixel for section and atlas
panels. Tissue can therefore appear smaller than the atlas. The stored
affine contributes its isotropic scale, `sqrt(abs(det))`. Each section
thumbnail is the whole image; each atlas thumbnail is framed around the
atlas anatomy. Neither is stretched to fill the tile.

Overlay pictures show the applied deformation unless `warp="none"`. Their atlas borders are the boundaries between regions at one depth of the atlas ontology (`depth`, default 6; 0 = root, as in BrainGlobe's `structure_id_path`); a highlighted region is outlined from the leaf annotation, whatever its depth. Every agent picture draws its atlas borders as cased lines, so they read on dark and pale tissue alike: a core line in black or white with a thin outline of the other colour (about one picture pixel) on both sides. Each picture takes one arrangement, a black core with a white outline when the picture's own pixels under the lines are light (mean CIELAB lightness 50 or more) and a white core with a black outline when they are darker; the caption names it (`borders black, white-edged`). Faint context lines and thick highlighted regions keep the same arrangement, the outline scaling with the core. The image model's pictures keep plain yellow lines. The footer of a picture drawn at other than canvas pixels gives both scales: `canvas` (micrometres per canvas pixel) and `view` (micrometres per picture pixel, the number the reply's caption calls um/px).
Where the agent chooses the picture size, `resolution` applies to every mode:
a section is never drawn past its source pixels, an atlas plane at most four
times past its voxels (as an atlas zoom is). Picture resolution affects
display only, never fit geometry or stored transforms. Source image files are never modified.

## Channels and damage

`set_channel_properties` changes raw-channel display contrast, gamma and
colormap across the stack. Automatic contrast is computed per section.
These settings affect pictures, not fits.

`set_preprocessed_channel_properties` changes the channel blend, CLAHE,
N4 correction and denoising used by the fits and image model, stack-wide or
per section. It returns before/after pictures. Its description and the
job statement's Method tell the agent to register on the counterstain
(DAPI, Hoechst, Nissl, NeuroTrace) alone, enhanced, with every labelling
channel at weight 0. Inspect that channel with
`look(channels=["preprocessed"])`. Both channel tools are available in every
job; ABBA's agent-preprocessing switch controls which raw channels it exports.

`mark_damage` records atlas regions physically lost or badly displaced,
including one hemisphere when needed. Those regions are excluded from
elastix, `fireants_syn` and image-model tracing. Small tears, bubbles and folds do not
by themselves justify excluding an entire region. A job can disable this tool
with `mark_damage` in its [disabled list](#switching-tools-and-options-off).

## Positioning and linear alignment

`position_sections` writes positions and cutting angles. Sections without
supplied positions stay unplaced until the agent writes them. The agent can
revisit placement after inspecting linear or nonlinear fits; changing a
section's position clears its deformation.

`interactive_transform` sets flip, quarter turn, rotation, scales, shear and
shifts for several sections at once (up to four through LangSlice's own agent
and MCP; any number through the CLI and the library, within the job's
`transform.max_parallel`). Values are absolute; omitted values keep
their current setting. Rotation and left-right flip orient the section
before its in-plane affine.

`elastix_affine` takes as many sections per call as `interactive_transform` does (more is refused with `TOO_MANY_SECTIONS`, an empty list with `BAD_ARGS`), and refines each placement against one of the atlas's images. `restrict_to` selects the anatomy that steers it. Region selections
include descendants and may specify a [side](#sides), such as `"CTX:left"`.
A restricted fit highlights those borders and may zoom its result picture
to them; other borders remain faint. Use region `zoom` for an isolated border.

A region too small for elastix is refused with `REGION_TOO_SMALL`: its
non-excluded support must span at least four native atlas pixels in each
in-plane dimension and contain at least sixteen pixels. The check happens
before padding or fitting. No fit is run or written for that section; use
a larger region or `interactive_transform`.

## FireANTs deformation

`fireants_syn` fits the atlas to the section's preprocessed stain, starting from
its current linear placement and any applied deformation. Each fit builds
on the previous one and can be undone. It accepts one to four sections,
`stiffness="soft"|"medium"|"firm"` and optional `restrict_to` regions.
The restricted atlas mask includes a 300 µm neighbourhood around them;
marked damage is excluded automatically. A restricted fit moves only that
neighbourhood: the masks limit where the fit is measured, not where its
smooth field reaches, so the step's displacement fades to none over a further
0.3 mm and the tissue beyond keeps its earlier fit.

The engine is the SyN (symmetric diffeomorphic) registration of
[FireANTs](https://github.com/rohitrango/FireANTs), on the GPU when there is
one (about a second per section) and on the CPU otherwise, with ANTs' step rule: mutual information plus an edge channel,
four pyramid levels, the stiffness as Gaussian smoothing of each step and of
the whole warp. Identical inputs give identical fields.

A stain fit runs twice, over the whole section or restricted to regions. The
first run measures the section inside the placed atlas outline widened by
0.2 mm (for a restricted fit, only within the regions' neighbourhood). Then each section pixel is
classed as matching the atlas image the first run warped, as an artifact, or
as background (a three-class EM after Tward et al. 2025, on a 50 µm grid with
a linear contrast per 0.8 mm block). The second run starts again from the
same placement, measures every section pixel that matches with probability
at least 0.5 anywhere in the picture, and lets the atlas reach everywhere
except the marked damage: holes, tears and debris drop out, and the outline
no longer bounds where tissue may go. A restricted fit classes the pixels
over the whole section (the classes need tissue and background to learn
from) and then keeps both masks within its regions' neighbourhood; its
field still fades to nothing over 0.3 mm beyond the neighbourhood. The fitted record keeps the pixels left
out, for review pictures; the agent's replies and pictures do not change.
The Elastix affine and fits of traced borders keep the outline masks and
run once.

Both registration engines accept any of the atlas's own images as
`atlas_image`: BrainGlobe's `"template"` (the default) and every additional
reference the atlas ships, under BrainGlobe's names (for example
`ccfv3augmented_mouse_25um`, whose template is the Blue Brain averaged Nissl,
also ships a single-animal Nissl). The job statement lists them, and `look`
draws any of them as an atlas layer. One addition: an Allen CCFv3
atlas (which ships no Nissl) is also offered `"ccfv3augmented_nissl"`, the
Blue Brain averaged Nissl template from `ccfv3augmented_mouse_25um`, which is
the CCFv3 grid extended 0.35 mm at the front; it downloads on first use.

The reply reports displacement, fold fraction and plausibility flags, and
shows the fitted overlay. These diagnostics report concerns rather than
rejecting fits. Judge the overlay against internal tissue anatomy.
Changing a section's position, orientation, cutting angles or linear
transform clears its deformation; the reply lists affected sections.

## Landmark triangulation

`set_landmarks` saves explicit atlas/section corresponding pairs. It adds,
updates and deletes a batch as one undo step and leaves the registration
unchanged. Accumulate correspondences across the whole section, including
its outline and internal anatomy, then call `landmark_triangulation` to apply
the complete set.

When MCP shrinks pictures to fit its reply budget, it assigns new saved picture
numbers to the delivered sizes and preserves their exact coordinate geometry.
Earlier picture numbers still refer to their original images.

Both endpoints use **x,y pixels of an explicitly numbered returned picture**,
from the top-left of the whole image. Panel offsets are included. On a zoom, x,y
are the numbers its grid prints (pixels of the whole picture it was cut from),
so a number read off any grid works whether the call names the zoom or the whole
picture; a point the zoom does not show is refused with the range it covers in
those numbers. `look` in
`section` or `overlay` mode provides section points; `overlay` also provides
atlas points. For an atlas-only view tied to a section, use
`look(mode="atlas", sections=["s01.tif"])`: it draws that section's own
position and cutting angles. Generic atlas browsing with `positions_mm`
has no section context and cannot supply landmark endpoints.

```python
job.set_landmarks(
    section="s01.tif",
    pairs=[{"atlas": {"picture": 12, "x": 180, "y": 140},
            "section": {"picture": 13, "x": 190, "y": 145}}],
)
job.set_landmarks(section="s01.tif", pairs=[
    {"id": "L1", "section": {"picture": 15, "x": 215, "y": 160}}], view=False)
job.set_landmarks(section="s01.tif", delete=["L2"], view=False)
job.landmark_triangulation(section="s01.tif")
```

Pair IDs persist through edits and are never reused. Omitted endpoints of an
update stay as they were. `delete=["all"]` clears the section's pairs as one
step. Every reply lists the section's pairs, one compact record each (`id`,
`atlas` and `section` as `{picture, x, y}`, and a `state`) with their counts:
`applied` (in the landmark warp applied now, with these endpoints), `moved` (in
it, but with other endpoints now), `pending` (saved, not in it) or `stale`
(placed against an atlas plane the section has left; `stale` wins over the
others). A `warp` block says which pairs the current warp was built from
(`built_from`), which pairs are not in it as they are (`not_applied`), which
were deleted but still shape it (`deleted_still_in_warp`), and in a `note` that
`landmark_triangulation` must run again to apply the set. A call with no pairs
and no `delete` writes nothing and returns this listing. `status` rows carry
`landmarks: {total, pending, stale, warp}` for a section with pairs (`pending`
counts `moved` too; `warp` says in words whether the applied warp matches the
pairs), and `can_change` names the landmark tools when they are on.

Place both endpoints of a pair on the same overlay (or a zoom of it) where you
can: the atlas point where a border is drawn, the section point where that
border belongs in the tissue. After a warp, overlays show the section warped
onto the fixed atlas borders and a point placed there is read through the
warp; the overlay's caption says so ("the section is warped onto the atlas,
which keeps its place"). Read endpoints off a close zoom, its grid lines 10 to 20
pixels apart: read off a coarse grid, a point can miss the border meant by 20
pixels or more. Spread the pairs over the outline and the internal anatomy: a pair on
the outline moves the outline, and a pair on an internal landmark (the midline,
a ventricle, a fibre tract) moves the anatomy inside. No pair is needed to keep
a spot where it is: `landmark_triangulation` adds automatic holds between the
pairs. An atlas region with no tissue on the section (torn or missing) takes no
pair; `mark_damage` records it. `mark_damage` marks whole regions, so mark a
region only when it is missing; a region only partly present keeps its pairs on
the part that is present. Tissue with no atlas counterpart (a detached fragment,
debris) takes no pair either.

The reply names the region drawn at each atlas endpoint the call placed
(`atlas_endpoint_regions`, by pair id, from the section's current atlas plane).
An atlas endpoint outside the atlas, where no brain is drawn, is saved, listed in
`outside_atlas` and named in a `warning`: for a pair on the outline the atlas
endpoint belongs on the drawn border or just inside it.

On the command line a list of objects such as `pairs` goes through `--args`,
as a JSON object or a file, and a list of four numbers such as a zoom's box
is one flag:

```bash
langslice-job JOB set_landmarks --args @pairs.json     # {"section": "s01.tif", "pairs": [...]}
langslice-job JOB set_landmarks --args '{"section": "s01.tif", "delete": ["L2"]}'
langslice-job JOB zoom --picture 12 --box 960,230,1300,940   # or --box '[960,230,1300,940]'
```

A refused endpoint names the pair (its index in the call and id), the
endpoint role, the picture number and the x,y, and says what to do: a point
outside the image content gives the picture's content box ("image content is
x 0-1399, y 0-1089 in picture 29"), a picture without geometry for the section
or one drawn at an earlier atlas plane says which picture to draw again, and
`TOO_MANY_LANDMARK_VIEWS` says how many pairs were sent and to resend them with
`view=false` or in calls of four. The default `view=True` shows one contextual side-by-side crop per
added/edited pair: atlas **template** at left, **preprocessed** section at
right, with the selected ID labeled. The atlas panel carries the atlas region
borders as `look` draws them, so the border an atlas point sits on is visible. At most four pairs can be inspected
this way per call; `view=False` permits larger batches and independent
`look`/`zoom` inspection. A zoom of a contextual pair is a saved-image crop and reports
`redrawn=False`.

The exact saved geometry resolves cropping, resizing, rotation, flips,
multiple panels and the inverse of a deformation drawn on an overlay.
Caption bands, gutters and points outside either image are refused without
clamping. Subsequent section views show open-centre circles, atlas views
show diamonds, and overlays show both with matching IDs. White and black
outlines and backed labels keep them visible across channel colors. Markers
are drawn after compositing and never enter fit images or image-model inputs.
Old pictures and zooms retain their original marker set. Markers are
annotations: editing pairs alone does not make an older picture or zoom read
as drawn from a changed stack (`stale`), and neither does marking damage, which
only fits read; a change to the placement or the registration does.

Changing the atlas position or cutting plane keeps the pairs and marks their
atlas correspondences stale. Update those atlas endpoints from a current
view before applying them. Appearance changes leave correspondences valid;
section endpoints are stored in the original image's normalized frame and
survive orientation changes. Linear transform changes retain both endpoints.

A section whose `deformation` is frozen takes neither tool: both refuse it
with `FROZEN`.

`landmark_triangulation` requires at least three pairs and a written linear
placement. The agent is responsible for anatomical coverage. Before the mesh is
built it adds **automatic holds** between the pairs: a regular lattice over the
section frame, spaced one seventh of its longer side (about 35 points on a
typical frame), keeping only the lattice points on the placed atlas footprint
(the deformable fit's fixed mask) and leaving out every lattice point within 0.7 spacings of the
section point of any pair, so a pair moves its own neighbourhood freely and the
holds fill only the gaps between pairs. Each hold corresponds to itself against
the baseline the warp is applied on (the linear placement, or the earlier
deformation it composes onto): it keeps that spot where it is. Without them a
pair on the outline would drag distant anatomy through the large triangles that
reach the four fixed corners. Holds are recomputed from the current pairs on
every application; they are not pairs, are listed nowhere and cannot be edited.
The reply says in words that holds were added, with their count
(`automatic_holds`, `holds_note`), and the result picture draws them as small
white dots with a black ring and no label. The mesh is a Delaunay
triangulation of the pairs' and holds' section points in their pixel aspect
ratio, with VisuAlign's four fixed corners 10% outside the frame, and a
piecewise affine pull-back to the matching atlas points. Only this tool adds
holds: the VisuAlign mesh itself (`core/triangulation.py`) is exact for any
point set it is given. Its inverse uses the same
triangle connectivity. Repeated section points (the reply names the pairs) and
zero-area source triangles are refused as invalid geometry.

Folds are accepted, as VisuAlign accepts them (expert registrations contain
them): a triangle whose atlas points wind the other way from its section
points is applied, and the forward map is unchanged. Where target triangles
overlap, the inverse takes the unfolded triangles first and then the folded
ones, each in mesh order, so the result is deterministic and a fold never
overrides sound anatomy. The reply reports `folds`: `area_fraction` (the share
of the section frame covered by folded triangles), `pairs` (the pair IDs at
their corners, so they can be fixed), the first triangles with their pair
IDs (an automatic hold at a corner is named `automatic hold`, with no id), and
`triangle_count`. The result picture outlines each folded triangle in
magenta between its vertices' section points, holds included (a triangle at the frame's edge
shows only its non-corner edges). Displacement and existing biological
plausibility diagnostics (which flag a fold fraction past a threshold) are
reported for review. The reply lists `built_from` (the pair IDs the warp was made
of) and, since `displacement_mm` is the whole deformation now applied (every step
since the linear placement, this one included), `step_displacement_mm` is only
the change this application made over the warp it replaced or was applied onto
(`displacement_note` says so). The result picture is `look`'s overlay at the job's
picture size; where the agent chooses the size, `resolution` works as for
`look`. The first application
composes the landmark correction onto the current deformation. Consecutive
landmark applications replace that landmark correction using the same
earlier baseline, so editing or deleting a pair removes its prior influence.
An intervening FireANTs or trace fit becomes the baseline for the next landmark
application. A missing baseline inverse or unreadable current deformation is
refused. Reapplying an unchanged set writes nothing. Editing or deleting
points leaves the current registration in force until explicit application;
undo restores the registration before that application.

## Optional image-model traces

`trace_borders(section="s01.tif", restrict_to=["CA1"], prompt="")` asks an
image model to move the placed atlas borders onto the tissue, then fits an
FireANTs deformation to the extracted lines. It requires a linear placement.
The model receives two images in the same frame: the preprocessed section
and that section with thin yellow atlas borders. The section is turned
upright on that frame from its linear placement: the atlas's dorsal side
(the anterior end on a horizontal section) points straight up, and on a
coronal or horizontal section the atlas midline runs down the middle of the
image; a sagittal section is only turned. It is a turn and shift, never a
mirror. The image is then cropped to the placed atlas's extent plus 15% of
its larger side on every side, never past the section image, and on a
coronal or horizontal section widened to stay symmetric about the midline;
where that runs past the section image it is padded with the slide's
background colour. Turn, crop and scale are drawn in one resample, and the
model's lines are read back through them. `restrict_to` selects the
borders shown and fitted; empty uses all eligible regions. Marked damage
is excluded. The model's outer line sets the section's extent: the fit pulls
the atlas in or out to it, wherever the linear placement left the atlas edge.
The fit of traced lines steps half as far as a stain fit, smooths the whole
warp twice as wide, and gives every traced region and the lines an equal
pull (each one's gradient scaled to the same length, as ANTs' multi-metric
rule does), so no single large region or stray line drives the warp.

The agent can edit the base prompt for a section; blank uses it unchanged.
The base prompt is LangSlice's for the provider, or the job's own
(`nonlinear.image_prompt`, `linear run --image-prompt FILE`).
The atlas determines which borders exist even when tissue boundaries are
indistinct. Only physically missing tissue justifies omitting them.

The tool runs in the background. `status` lists pending work, later tool
replies deliver completed results, and `submit` waits for them. A completed
fit is its own undo step. Identical inputs reuse the saved reply; a changed
placement or preprocessing recipe requires a new trace. Results whose
inputs changed while they ran are refused as `STALE_INPUT`.

Inspect the raw model reply, then the extracted lines on the original
photograph, then the fitted overlay. A model may redraw tissue, and a fit
may degrade a useful trace. Yellow tissue can confuse the line extractor;
fit metrics do not certify region identity or topology. The job saves the
attachments, prompts, raw reply, extracted lines and fit separately, and
`export_maps` (and `submit`) export each section's last reply and its lines
on the section's maps grid, before any fit
([`exports/traces/`](file_formats.md#tracesnamenpz)).

`trace_borders` is offered only with Nonlinear enabled and an image model
connected. `provider="none"` disables it while keeping FireANTs and landmark tools.
Custom models and prompts are described in the [library guide](library.md#image-models).

## Writes, review and submission

Writes checkpoint the job and form undo steps; `undo` and `redo` persist
across sessions. Their reply says which step it took back or forward: `undone`
(or `redone`) holds the `sections` it touched and the `fields` it changed,
then the status rows of those sections and the remaining depth. Change tools return review pictures unless `view=False`;
a host's `force_view=True` (`linear run --force-view`) forces `view=True`, and the argument is not offered. It defaults to false, so
the agent may suppress a picture. Long fits compute outside
the write lock and recheck inputs before applying. A conflicting edit
produces `STALE_INPUT` for that section; unaffected results still apply.

`submit` checks the enabled tasks before writing maps and exports:

- Positioning requires a written position for every section. Strict spacing
  permits at most 10% deviation from the interval. Reported interval breaks
  must exceed 1.5 times the median spacing.
- Linear requires a transform for every section, including damaged ones.
- Nonlinear asks for no deformation by default: the agent gives a section one
  where it improves on the linear placement. A host setting
  `nonlinear.require_deformation=True` (`linear run --require-deformation`)
  requires a deformation at the current placement on every section; the job
  statement then says so, and `submit` is refused with `MISSING_DEFORMATIONS`
  until each has one. Host-excluded sections are exempt. An image-model trace
  is never required.

Results are [coordinate maps, labels and registration exports](file_formats.md).

`--playbook` adds optional positioning guidance using candidate planes,
region lookup, incremental placement and stack review. It defaults off. Thickness and nominal interval are optional protocol facts,
with no lab-specific defaults. When either is omitted, the prompt marks it
as not supplied and asks the agent to infer positions and spacing from
anatomy, estimate missing protocol values where evidence supports them,
and record estimates or uncertainty in its notes. Omission does not block
a run or require a question to the user. Spacing is enforced only when
`strict_interval` is enabled, which requires a supplied interval.

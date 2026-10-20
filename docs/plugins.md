# Tool plugins

A tool plugin adds tools to a job's toolbox when the job is made. An added tool
is a tool like the native ones: it has its own name, description and
arguments, is listed with them, is called the same way (the built-in agent,
the agent CLI, the Python library), and can read and write the job through the
same operations. LangSlice supplies the seam; the code of the tool lives in the
plugin.

A plugin has two parts. The **adapter** is a small Python module that runs in
LangSlice's process and imports `langslice.plugin_api` only. The **engine** is
the method itself, usually in its own environment; the adapter calls it as a
subprocess and exchanges files or JSON with it. The engine never imports
LangSlice.

## A plugin module

A plugin names a function, `module:factory`, that returns a list of `ToolSpec`.
`tests/fixtures/example_plugin.py` is a complete small one (a read tool and
two write tools on synthetic data, one of them writing a deformation):

```python
from langslice.plugin_api import ToolContext, ToolResult, ToolSpec


def nudge_sections(sections: list[str], dx: float = 0.01) -> dict:
    """Shift sections sideways in the plane by dx of their width.

    Args:
        sections: Filenames of the sections to shift.
        dx: The shift as a fraction of the section's width.
    """
    ...


def _nudge(ctx: ToolContext, sections: list[str], dx: float = 0.01) -> ToolResult:
    rows = [{"id": name, "params": [1, 0, dx, 0, 1, 0]} for name in sections]
    done = ctx.set_transforms(rows)
    return ToolResult(done.reply(), [ctx.overlay_picture(done.written[0])] if done.written else [])


def tools() -> list[ToolSpec]:
    return [ToolSpec("nudge_sections", nudge_sections, _nudge, kind="write",
                     group="Linear", task="transform", writes=("affine",))]
```

`langslice.plugin_api` is the one import path; `PLUGIN_API_VERSION` (1) changes
when the names in it change incompatibly. A module may set its own
`PLUGIN_API_VERSION`, and a `ToolSpec` has `api_version`; a plugin written for
another version is refused. The same names are importable from
`langslice.doors.plugins`.

### `ToolSpec`

| Field | Meaning |
|---|---|
| `name` | The tool's name: an identifier; not a native tool's nor another plugin tool's |
| `stub` | A function whose signature is the tool's arguments and whose docstring is its description, written like the stubs in `langslice/doors/declarations.py`. The description may use [needs marks](registration.md#switching-tools-and-options-off) (`[[?look:overlay text]]`) |
| `handler` | `handler(ctx, **arguments)`, returns a `ToolResult` |
| `kind` | `read` or `write` |
| `group` | `Common`, `Positioning`, `Linear` or `Nonlinear`: where a host lists it |
| `task` | `position`, `transform`, `nonlinear` or `None`: the tool exists only while that task is on; `angles`: while the agent may write the cutting angles, as `position_sections` does (the position task on, or the transform task leaving the angles to the agent) |
| `writes` | The section properties it may write (`position`, `angles`, `orientation`, `affine`, `deformation`). A write outside this set is refused; a tool that writes only properties frozen on every section is not offered |
| `long` | Computes outside the job's write lock and applies under it (see [long tools](#long-tools-and-stale_input)) |
| `limits` | The most one call takes: `{"sections": 4}` refuses a longer `sections` argument (`TOO_MANY_SECTIONS`); `{"pictures": 2}` sends at most two pictures |
| `needs` | What the tool needs while it is on: groups of names, one of each on (`(("look:overlay",),)`), as for native tools |
| `version` | The plugin's own version, kept in the provenance |
| `meta` | JSON values kept with the tool in the provenance |
| `atlases` | Shell-style patterns for the job's atlas name (`("allen_mouse_*um", "ccfv3augmented_mouse_25um")`); the tool is offered on a job whose atlas matches one. Empty: every atlas |
| `planes` | The section planes it is offered on (`("coronal",)`); empty: every plane |

A tool whose `atlases` or `planes` do not fit the job is absent, as a disabled
tool is: it is not declared or listed, its options are gone, and every sentence
that names it in the other tools' descriptions (written with [needs
marks](registration.md#switching-tools-and-options-off), `[[?other_tool ...]]`)
is dropped. The provenance records the two fields.

A plugin's description is also rendered by the tools the job's tasks leave out: a
sentence written `[[?elastix_affine ...]]` in a plugin's stub stands only where the
Linear task is on and `elastix_affine` is not disabled or frozen away. (The native
tools' texts keep naming a tool of a task that is off, as their text snapshot pins.)

The checks run when the factory is called: a name that clashes (with a native
tool, another plugin tool, or `script`, the name a script's writes are
recorded under), an unknown
`task`, `planes` that are no plane, `atlases` that are not name patterns, `writes` outside the five properties, `writes` on a read tool, a bad
limit, a stub with no docstring, another API version. Each is refused with one
line.

### `ToolResult` and errors

A handler returns `ToolResult(reply, pictures)`. `reply` is a dict of JSON
values; the tool's reply is `{"status": "ok", **reply}` (a handler may set
`status`). `pictures` are PIL images (or arrays) with their captions burned in
(`ctx.caption(image, text)`); they are sent with the reply and saved in the job
folder, numbered, like every picture a tool shows, so `zoom` and `read_file`
reach them. Raising `ToolError(code, message, **details)` refuses the call with
`{"status": "error", "error": code, "message": message, **details}`; any other
exception is a reply `PLUGIN_FAILED` naming it, and the job carries on.

## `ToolContext`

One per call; a handler must not keep it. A script gets the same context as
`job.context` ([your own method](library.md#your-own-method)): its writes are
recorded as `script` instead of `plugin:<tool>`, each outermost write is one
undo step, its reads sync the job first, and `work_dir` and `start_work` are
refused (`NOT_FOR_SCRIPTS`).

**The job.** `ctx.spec` (the job's spec as a read-only mapping), `ctx.job_dir`,
`ctx.work_dir()` (`<job>/plugins/<tool>/`, made on use), `ctx.task_on(task)`.

**Reading.** All in LangSlice's documented units and frames.

| Call | Gives |
|---|---|
| `ctx.sections()`, `ctx.section(id)` | `SectionView`: `id`, `index`, `file_size`, `pixel_size_um`, `pixel_size_source`, `position_mm`, `angles`, `flip`, `rotation_deg`, `affine` (`kind`, the six normalized numbers `matrix`, `knobs`, `mirrored`), `deformation`, `damaged_regions`, `damage_note`, `layers`, `frozen` |
| `ctx.channels(id)` | The names of the section's raw channels in file order (`red`, `green`, `blue` or `gray` for a colour file; the host's names or `ch1`, `ch2`... for a multi-page file): what `channels=[name]` takes |
| `ctx.section_raster(id, level=, channels=, raw=False)` | `SectionRaster`: `array` (the section as its orientation sets it, RGB `uint8`, reduced to `level`: a long edge in pixels or `low` / `medium` / `high` / `auto`) with its frame: see [what a raster shows](#what-a-raster-shows) for what `channels` and `raw` give; `to_section_px(xy)` and `from_section_px(xy)` convert `[x, y]` between the array's pixels and `section_px`, `mask_to_section_px(raster)` turns a raster computed on the array into the file's orientation |
| `ctx.section_image(id, ...)` | Only the `array` of `section_raster` (same arguments). **Its pixels are not `section_px`** (see [frames](#frames)) |
| `ctx.atlas()` | `AtlasView`: `name`, `resolution_um`, `orientation`, `species`, `plane`, `shape` |
| `ctx.fit_frame(id)` | `FitFrame`: the section on its fit grid (`image`), the atlas plane at its position and angles on the native atlas grid (`atlas["template"]`, `["annotation"]`, and any other of the atlas's images asked for with `atlas=`, by its BrainGlobe name), `atlas_to_section` (3x3: native atlas pixels to fit-grid pixels with no in-plane transform), `placed_atlas_to_section` (the same map under the section's stored transform, its current linear placement; None without a transform), `excluded` (the native plane's pixels of the section's marked damage regions, sides included, which every native fit leaves out; None when none are marked), `damage_zone` (`excluded` widened by the margin the native fits also leave out of their comparison, 0.1 mm scaled to the atlas's brain size), `stain` with `stain=True` (the section's preprocessed channel on the grid of `image`, the picture the native fits read), `applied_map` (the native atlas pixel of every fit-grid pixel through the section's current registration, its deformation included, `(H, W, 2)` as `deformation_from_map` takes it: where an engine that builds on a landmark warp or an earlier fit starts; None without a deformation), `plane` (the plane in atlas micrometres: `center_um`, `dx`, `dy`, `px_um`, `shape`, and `native_to_um`, the 3x3 taking a native pixel centre `[x, y, 1]` to its atlas micrometres, voxel centres at index times resolution) |
| `ctx.atlas_slice(id, kind=)` | One of those atlas planes |
| `ctx.section_inputs(id)` | A digest of what a computation on the section reads (placement, appearance, damage marks); for [`expect=`](#long-tools-and-stale_input) |
| `ctx.read_registration(path, sections=(), at=None, registered_on="file")` | A registration file made elsewhere (QuickNII, VisuAlign, DeepSlice, `registration.json`) as `placements`, nothing written; see [reading a registration](#reading-a-registration) |
| `ctx.layers(id)` | The layers the section shows |
| `ctx.atlas_plane(position_mm, pitch_deg=0, yaw_deg=0, kinds=)` | `AtlasPlane`: the atlas plane at any position and angles, on its native grid (`atlas`, `plane` as `fit_frame` gives them) |
| `ctx.atlas_volume(kind)` | A whole atlas volume (`annotation` or one of the atlas's images), read only |
| `ctx.to_atlas_um(id, xy, frame="file")`, `ctx.from_atlas_um(id, xyz_um, frame="file")`, `ctx.plane_distance_um(id, xyz_um)`, `ctx.region_at(id, xy, frame="file")` | Section points (`[x, y]` pixels of the file, or `frame="fit"` of the fit grid) to atlas micrometres through the current registration, its deformation included, and back (projected onto the section's plane, through the deformation's inverse); how far atlas points lie off the plane; the region (`id`, `acronym`, `name`) under each point |

**Converting.** `ctx.transform_from_matrix(id, matrix)` takes the 2x3 map from
the fit grid's pixels to native atlas pixels that registration engines return
and gives a `TransformProposal`: the transform as an Elastix fit would store it
(`params`, the knobs `interactive_transform` takes about the section's pivot,
`record`), refining the section's current placement.
`ctx.transform_from_params(id, params)` does the same for six normalized
numbers `[a, b, tx, c, d, ty]`. `proposal.row()` is a row for `set_transforms`.

`ctx.angles_from_plane(normal_um, point_um)` takes a plane in atlas
micrometres (a normal and a point on it, the atlas's axis order, as
`fit_frame.plane` gives one: its normal is `dx` x `dy`) and gives the
placement that says it in the stack's conventions: `position_mm` (where the
plane crosses the volume's in-plane centre), `pitch_deg` and `yaw_deg`.
Nothing is written; a plane tilted past what a cutting angle can say is
`BAD_PLANE`.

`ctx.write_frame(id, folder, format="tif")` writes an outside engine's
inputs (the section, its preprocessed channel and the atlas plane placed on
the fit grid, the atlas plane on its native grid, `frame.json`; spacing 1,
origin 0); `ctx.map_from_itk(id, source, fixed="section", atlas="placed")`
reads its answer (an ITK/ANTs displacement field or transform file, a
SimpleITK object, an array of pixel displacements) as the full map
`deformation_from_map` takes, and `ctx.transform_from_itk(...)` an affine as a
`TransformProposal` ([an outside engine's files](library.md#an-outside-engines-files)).
`ctx.fit_fireants(id, settings=, mask=, start=, em=, marked=)` and
`ctx.fit_elastix(id, restrict_to=, exclude=, atlas_image=, mask=, marked=)`
run the native engines with settings and a section mask of the caller's own
and return a `DeformationProposal` / `TransformProposal`, not written
([the native engines](library.md#the-native-engines-with-settings-of-your-own)).

`ctx.deformation_from_map(id, native_xy, note="", meta=None)` takes a
deformable method's result: `native_xy` is a float array `(H, W, 2)`, `H` by
`W` the fit grid's `image`, giving for each fit-grid pixel centre `(x, y)` the
native atlas pixel `(x, y)` it corresponds to. It is the whole map, linear
placement and deformation composed, in the frames `transform_from_matrix`
takes; an engine started from the section's current placement
(`placed_atlas_to_section`) composes that placement into its answer. It gives a
`DeformationProposal`: `summary` (the largest and median displacement over
the placed atlas footprint in millimetres, the fold fraction and the
plausibility flags, as for a native fit) and `inverse_consistency_mm`.
`proposal.row()` is a row for `set_deformations`. The map is stored as every
deformation is: the residual over the section's current linear placement
(`field_mm = (placed_atlas_to_section @ native - pixel) * mm_per_px`, which
gives the map back exactly, to float32 rounding), with an inverse LangSlice
computes by fixed-point iteration (`inverse_consistency_mm` is how far the
two disagree). It replaces whatever deformation the section holds rather than
building on it: the map is already the whole registration. `note` and `meta`
(JSON values) are kept in the record's provenance.

**Writing.** Each write goes through the operation a native tool uses.

| Call | Writes | Needs in `writes` |
|---|---|---|
| `ctx.set_positions(rows, cutting_angles=None, expect=None)` | rows `{"id", "position_mm"}`; positions are clamped into the atlas range and reported; the stack is renumbered by position | `position` (and `angles` with `cutting_angles`) |
| `ctx.set_transforms(rows, expect=None)` | rows `{"id", "transform"}` (a `TransformProposal`'s `record`) or `{"id", "params", "note"?}` | `affine` |
| `ctx.set_deformations(rows, expect=None)` | rows `{"id", "map", "note"?, "meta"?}` (the map `deformation_from_map` takes) or `{"id", "deformation"}` (a `DeformationProposal`); each replaces the section's deformation; ok rows carry the proposal's `summary` | `deformation` |
| `ctx.mark_damage(rows, expect=None)` | rows `{"id", "regions", "note"?}`, as `mark_damage` | any write tool |
| `ctx.add_layer(id, layer)` | a [layer](#layers) | any write tool |

A write returns a `WriteResult`: `ok` is false when the call was refused whole
(`error`, `message`); otherwise `rows` has one row per section asked
(`status` `ok` or `error` with the code: `FROZEN`, `STALE_INPUT`,
`UNKNOWN_SLICE_IDS`, ...) and `written` the ids written. `result.reply()` is
those as reply fields. A read tool that writes, or a write outside `writes`, is
refused (`READ_TOOL_WRITES`, `WRITE_NOT_DECLARED`).

A deformation is written as a native fit's is: the record is saved in the
section's folder (`sections/<name>/deformable/<key>/`, its engine
`plugin:<tool>`), the section's `deformation` names it with one step
`{"method": "plugin:<tool>", "start": "linear", "note"?}`, and `look`,
`export_maps`, `registration.json`, the VisuAlign markers and the ABBA
connector read it like any other. A section without a position or transform is
that section's row `INVALID_LINEAR_PLACEMENT`, as a native fit refuses it; a
map of another shape, not float or not finite is `BAD_FIELD`. Maps are
converted before the job's write lock is taken.

Everything a write tool's call writes is one undo step: `undo` reverts it
whole, the checkpoint and `registration.json` follow, a property the job froze
is refused for that section, and a write that changes a section's placement
clears its deformation as it does for a native tool. `with ctx.writing():`
holds the job's write lock around several writes.

**Showing.** `ctx.overlay_picture(id, proposal=None)` is the overlay `look`
draws, as it stands or with a proposal not written yet (`{"transform": ...,
"position_mm": ..., "angles": (pitch, yaw)}`, or `{"deformation": ...}` alone,
a `DeformationProposal` or a map, drawn on the current placement);
`ctx.positioning_picture(ids, proposal=None, resolution=None, part=None)` is
the positioning picture `position_sections` shows (`look` mode `positioning`:
the sections in position order under their atlas planes, captioned) as a list of
PIL images, one per part of a long stack (`part=` picks one, 0-based; the
caption says which part). With `proposal={id: {"position_mm": mm, "angles":
(pitch, yaw)}}` (either key, any of the sections) it is drawn at the proposed
positions and angles, ordered by them, and carries one more caption band, `proposal: ...
NOT written`; the job is not touched. Where the job disables `look:positioning`
it is refused (`POSITIONING_OFF`): give the tool `needs=(("look:positioning",),)`
to be absent there. `ctx.caption(image, text)`.

### What a raster shows

`section_raster` is always the section as its orientation sets it (quarter
turn, then flip) and reduced to `level`. What the pixels are depends on
`channels` and `raw`:

| Call | Pixels |
|---|---|
| no `channels` | The default appearance: with `preprocess="auto"` the CLAHE and DAPI-weighted blend LangSlice's pictures show (a host's own blend with `host_preprocessing`); with `preprocess="none"` the file |
| `channels=["preprocessed"]` | The channel the native fits read |
| `channels=[names]` | Those raw channels in their colours with the display properties set on them (`set_channel_properties`) |
| `raw=True` | The file as an image viewer shows it, 8-bit RGB: no preprocessing, no channel properties, no contrast changes. For a PNG or JPEG this is the delivered image, oriented |
| `raw=True, channels=[name]` | That one raw channel as gray RGB (three equal planes), as read from the file whatever display properties are set. More than one channel, or `preprocessed`, is `BAD_ARGS`; so is a name the section does not have |

Pixels come from LangSlice's working copy (a file over 3072 pixels on the long
edge is already reduced to it) and never above the source size, so
`raw=True` with `level` at or above the file's long edge returns the
delivered pixels exactly for a file of that size or smaller; a larger file
or a smaller `level` is resampled with Lanczos.

### Reading a registration

`ctx.read_registration(path, sections=(), at=None, registered_on="file")` reads a file from QuickNII,
VisuAlign, DeepSlice or `registration.json` for the job's sections (default all)
and returns `{"placements", "refused": {id: reason}, "missing": [id],
"unmatched": [names in the file]}`. It writes nothing. A placement:

| Key | Meaning |
|---|---|
| `id`, `position_mm`, `pitch_deg`, `yaw_deg` | The section's plane in the job's terms (atlas millimetres from the anterior edge; the job's own cutting-angle convention). On an augmented-atlas job the file is read on the Allen grid and the position includes the 0.350 mm offset |
| `rotation_deg`, `flip` | The section's **current** quarter turn and flip: each section is read in the orientation the agent set, never one chosen by the reader |
| `params` | The six normalized numbers `[a, b, tx, c, d, ty]` of the section's in-plane transform **for that oriented render**, at that plane: what `set_transforms` takes as `{"id", "params"}`. None (with `params_note`) when the job has no pixel size for the section |
| `in_plane` | The six numbers as rotation, scales and shear on the render |
| `pixel_size_um`, `pixel_size_source` | The calibration the numbers are relative to (`imported` means the job had none and the file's maps implied one) |
| `out_of_plane_um` | How far the file's map lies off the plane it was read on (about 0 on the file's own plane) |

`registered_on="oriented"` says the file's entries were made on the section as its
orientation turns and flips it (the image `section_raster` gives a method), not on the
image file as stored: the entry's width, height and fractions then refer to the oriented
image, and the numbers come out for the same orientation the section holds. A tool that
hands an engine `section_raster` and reads its answer back uses it; with the default
`"file"` an engine's answer about a flipped or turned section would come back with the
flip in its numbers.

`at={id: {"position_mm": mm, "angles": (pitch, yaw)}}` reads a section's map on a
plane of your choosing (a section's final position and angles, e.g. a stack's
propagated angle), not the file's own: `params` are then the file's map projected
onto that plane (least squares) and `out_of_plane_um` says how much was
discarded. `params` are only right for the plane they were read at, so a tool
that writes a registration's placement and transform together does it as:

```python
read = ctx.read_registration(path, at=planes)        # planes = the final placement per section
with ctx.writing():                                    # one undo step
    ctx.set_positions(rows, cutting_angles={"pitch_deg": p, "yaw_deg": y})
    ctx.set_transforms([{"id": r["id"], "params": r["params"]} for r in read["placements"]])
```

Positions and angles go first: `set_transforms` builds its record on the placement
the section holds when it is called (a section without a position is that
row's `NO_POSITION`). `set_positions` takes one stack-wide angle pair
(`cutting_angles`). A frozen affine is that row's `FROZEN`, a transform write clears
the deformation, and `undo` takes the whole block back.

### Long tools and `STALE_INPUT`

A tool declared `long=True` is not run under the job's write lock. It reads
`ctx.section_inputs(id)` for each section first, computes (an engine can take
minutes while the agent or other commands keep working), then writes with
`expect={id: digest}` (for a deformation, the digest without
`deformation=True` covers the placement the map was computed on; with it,
the deformation it built on too). A section whose placement, appearance or damage changed
meanwhile comes back as that section's row `STALE_INPUT`, nothing written for
it; the others are written. A long call's writes are one undo step too, unless
another write landed on the job between two of them: then they stay separate
steps, so `undo` never takes back a write that was not the call's.

### Background work

`ctx.start_work(sections, land)` starts `land` as a piece of the job's
background work on `sections` and returns its id (`w3`) at once; the tool
returns meanwhile (a tool that does is declared `long=True`). On a work
thread `land(ctx)` gets a context of its own, computes and writes through it
(`with ctx.writing():` makes its writes one undo step; pass `expect=` digests
taken at the start), and returns a `ToolResult` whose `reply["notice"]` is
the notice's text, the rest of the reply its result, and whose pictures are
saved and numbered with the job's. A `ToolError` or refusal it raises fails
the work with its message. As for `trace_borders`: the next tool reply
carries the notice under `background` with the pictures; `status` lists the
work still running; `submit` waits for it; the agent session waits for it
when the model ends its turn; an agent-CLI call that started work waits for
it before the process ends and puts the notice on its reply.

While the work runs, any write to its sections, by any tool, native or
plugin, is refused `SECTION_BUSY` naming the work (a call over several
sections gives that section's row); so is starting work on a section that
already has some. The work's own landing is not refused.

## Layers

Anything the agent or the user should see that is not a section property is a
layer: `ctx.add_layer(id, Layer...)`. A layer has a name (default: the tool's)
and a kind with a fixed file and frame; a layer of the same name on a section
replaces the earlier one.

| Constructor | Content | Frames | File |
|---|---|---|---|
| `Layer.points(points, labels=, values=, frame=, name=)` | `N` x 2 (or 3) coordinates | `section_px`, `atlas_um` | `.csv` |
| `Layer.lines(lines, closed=, labels=, frame=, name=)` | polylines or polygons | `section_px`, `atlas_um` | `.json` |
| `Layer.labels(raster, frame=, name=)` | integer label raster | `section_px`, `oriented` | 16-bit `.png`, `.npz` past 65535 |
| `Layer.scalar(raster, frame=, level=, name=)` | float raster | `section_px`, `oriented` | `.npz` |
| `Layer.picture(image, name=)` | a finished picture | none | `.png` |

### Frames

`section_px` is `[x, y]` pixels of the section's image file as stored: full
size, not turned, not flipped (pixel centres at integers; `(0, 0)` is the
centre of the first pixel). The array `ctx.section_image` / `section_raster`
returns is **not** in that frame: it is reduced, turned by the section's quarter
turn and flipped as its orientation sets it. A point found in that array must go
through `raster.to_section_px(...)` before it is a `Layer.points` or
`Layer.lines` coordinate (`from_section_px` goes the other way). A label or
scalar raster covers the whole file whatever its size, so it needs no scaling
but must be in the file's orientation; `frame` is required to say which it is:
`frame="section_px"` for a raster already in the file's orientation, or
`frame="oriented"` for one computed on the array of `section_raster`, which
`add_layer` turns into the file's orientation. `atlas_um` coordinates are `[x, y, z]`
BrainGlobe micrometres in the atlas's axis order. The file formats are in
[file formats](file_formats.md#plugin-layers).

The **fit grid** is `ctx.fit_frame(id).image`: the section turned and flipped
as its orientation sets it, at the size LangSlice's fits run at (a long edge of
1536 pixels, never larger than the file), pixel centres at integers. **Native
atlas pixels** are `[x, y]` pixels of the atlas plane at the section's
position and cutting angles (`fit_frame(id).atlas`, `plane` says where it lies
in atlas micrometres). `atlas_to_section` and `placed_atlas_to_section` map
native atlas pixels to fit-grid pixels; `transform_from_matrix` and
`deformation_from_map` take maps the other way.

`look` takes `layers=[names]` (or `["all"]`) for modes `section` and `overlay`
on every job (a script leaves layers too, under `plugins/script/`). Points and lines are drawn with a dark halo, label
and scalar rasters as outlines (a scalar at its `level`), never as translucent
fills. `section_px` layers follow the section through the picture (and its
applied deformation in an overlay); `atlas_um` layers are placed on the
overlay's atlas plane, or in a section picture through the section's linear
placement. A picture layer is not drawn on other pictures. The reply's `layers`
says, per picture and layer, whether it was drawn (and why not). A zoom of such a
picture draws the same layers again. `status` lists each section's layers
(`name`, `kind`, `frame`, `tool`).

A layer is part of the checkpoint: undoing the call that made it hides it,
redoing shows it. Its files stay under `plugins/<tool>/`. `export_maps` copies
the layers the sections show to `exports/plugins/<tool>/` with a manifest.

## Loading

| Where | How |
|---|---|
| Job spec | `JobSpec.plugins`: `["module:factory", ...]`, saved in `job.json`; `JobSpec.plugin_config`: options for a factory, by reference (the factory is called with the options object when it has one) |
| `langslice linear run` | `--tool-plugin MODULE:FACTORY` (repeatable); `--tool-plugin-config 'MODULE:FACTORY=JSON-or-file'` |
| Python | `langslice.create_job(folder, plugins=[...], plugin_config={...})`: references, factory functions defined at module level, or `ToolSpec` objects |
| Reopening | `open_job(folder)` and `langslice-job FOLDER VERB` read `plugins` from `job.json` and load them again |

A plugin that cannot be loaded, or whose tools clash, is refused with one line
when the spec is made. The plugin tools follow the native ones in every
listing: the built-in agent's tools and job statement, `langslice-job ops` and
`schema` (in a job folder), the job's `AGENTS.md` card, the library's `verbs`,
and `status`' `can_change`. They are named in `disabled` like native tools
(`--disable nudge_sections`), are absent while their task is off, and plugin
writers do not count toward a task's requirement that one of its own tools
stays on. An agent CLI reply from a plugin tool is the plugin's reply as it is.

A job without plugins is unchanged: no plugin text is in any declaration, reply
or picture.

## Provenance

A tool's description is an experimental variable, so a job records what it ran
with. When the tools are built, `job.json` gets `plugins`: per plugin its
reference, module, factory, `version`, package version, `api_version`, the
SHA-256 of the source files that declared its tools, and per tool its kind,
group, task, `writes`, `limits`, `meta` and `declaration`: the name,
description and argument schema as the model sees them. A change is also a
`plugins` line in `logs/events.jsonl`. The built-in agent's trace
(`<job>/trace` or `--trace-dir`) has a `plugins` record after `session` with
the same content; plugin tool calls and replies are traced like native ones.

## Where plugins load

A plugin is loaded, in the process that opens the job, by every door that opens
a job naming it: the library, `langslice linear run`, the agent CLI, the MCP
server and the ABBA worker. A host's cost estimate ignores the plugin fields. A
plugin keeps its files under `plugins/<tool>/` of the job (`ctx.work_dir()`).

## The MCP adapter

`langslice.doors.plugins.mcp:tools` is a plugin that makes LangSlice a client of
an MCP server on this machine. Its options (`plugin_config`) name the server,
by command LangSlice starts or by a streamable-HTTP URL, and the **allow-list**
of its tools; only those are offered, each as `<server>__<tool>` (characters
other than letters, digits and underscores in the remote name become `_`),
with the server's description and input schema as its declaration. Several
servers can be given as `{"servers": [...]}`.

```json
{"server": "napari",
 "command": ["python", "-m", "my_napari_server"],
 "timeout_s": 120,
 "tools": {
   "add_points": {
     "kind": "write",
     "inputs": {"image_path": {"send": "section_image", "section": "section"}},
     "fixed": {"viewer": "main"},
     "results": [{"to": "layer", "section": "section", "from": "points",
                  "kind": "points", "frame": "section_px", "name": "napari_points"}]
   }}}
```

| Option | Meaning |
|---|---|
| `kind` | `read` or `write` (default `write` when `results` is given, else `read`); also `group`, `task`, `long`, `limits`, `needs`, `version` as in `ToolSpec` |
| `inputs` | Arguments LangSlice fills instead of the model: `{"arg": {"send": "section_image" or "placement", "section": <argument naming the section>}}`. `section_image` writes a PNG under `plugins/<tool>/inputs/` and sends its path (`"inline": true`: the image as base64); `placement` sends the section's position, angles, orientation and transform. The argument that names the section is added to the declaration when the server has none of that name |
| `fixed` | Arguments always sent with a fixed value |
| `results` | How the answer (its structured content, or its text read as JSON) reaches the job: `{"to": "layer", "from": path, "kind": "points" or "lines", "frame":, "name":, "section":}`, `{"to": "transform", "from": path}` (six numbers), `{"to": "position", "from": path}` (millimetres); `from` is a dotted path into the answer |

The server's text, structured data and images are in the reply. The remote tool
cannot touch job state: it receives only what `inputs` gives it and its answer
reaches the job only through `results`, via the same context writes as any
plugin (frozen properties, undo, `registration.json`). The server's
descriptions are text from outside the project: they are shown as given (with
`[[` made `[ [` so they cannot act as a rendering mark), recorded verbatim in
the provenance (`meta.remote_description`, `meta.remote_input_schema`), and
cannot change the job's settings or the tools offered. The connection is kept
for the life of the process, so a stateful server (a viewer) keeps its state
between calls.

## Errors a plugin tool can reply

| Code | Meaning |
|---|---|
| `TOO_MANY_SECTIONS` | More sections than `limits["sections"]` |
| `READ_TOOL_WRITES`, `WRITE_NOT_DECLARED` | A write by a read tool, or of a property not in `writes` |
| `FROZEN`, `STALE_INPUT`, `UNKNOWN_SLICE_IDS`, `BAD_ARGS` | Rows of a write, as for native tools |
| `NO_POSITION`, `INVALID_LINEAR_PLACEMENT` | A section without a position (a read or conversion that needs one), or without a usable linear placement (a deformation needs a position and a transform) |
| `BAD_MATRIX`, `BAD_FIELD` | A degenerate or non-finite linear map; a deformation map that is not a finite float `(H, W, 2)` array on the fit grid |
| `BAD_LAYER`, `UNKNOWN_PLUGIN_LAYER` | A malformed layer; `look` naming a layer no drawn section has |
| `PLUGIN_FAILED` | The handler raised something other than `ToolError` |
| `NOT_FOR_SCRIPTS` | `work_dir` or `start_work` from a script's `job.context` |
| `ENGINE_MISSING`, `FIT_FAILED`, `REGION_TOO_SMALL`, `UNKNOWN_REGIONS`, `NO_SIDES` | `fit_fireants` / `fit_elastix`: the engine does not import, failed, or the regions given cannot be fitted |
| `DEFORMATION_UNREADABLE` | A point conversion through a deformation whose record or inverse cannot be read |
| `MCP_TOOL_ERROR`, `MCP_TIMEOUT`, `MCP_UNAVAILABLE` | The adapter's server answered with an error, did not answer, or is not running |

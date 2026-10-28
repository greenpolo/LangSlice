# Python library

The library lets a coding agent write its own registration workflow: open a
job, call tools, inspect their pictures and choose the next step. It uses
the same operations, checkpoints and outputs as the [agent interfaces](../agents/index.md).
Importing `langslice` loads no agent framework or model client.

## Open a job and call tools

```python
import langslice

with langslice.open_job("/scans/brain1") as job:
    print(job.verbs)
    print(job.status()["rows"])
    reply = job.look(mode="overlay", sections=["s07.tif"])
    print(reply["pictures"], reply["artifacts"])
    images = reply.images  # PIL images, in reply order
    detail = job.zoom(region="CA1:left")
```

`open_job` accepts a job folder or its image folder. Each enabled verb is a
method with the same name and arguments as the agent tool. Discover them
with `job.verbs` and `langslice-job schema VERB --job FOLDER`.
Sections are named by filename or a unique filename stem.

A reply is a JSON-safe dict: the full tool result plus `artifacts` (saved
file paths, kinds and picture numbers). `reply.images` is an attribute,
not a JSON key. Refusals are replies too: check `status` and `error` and,
for batch calls, the individual section results. Pictures are saved before
the method returns. `look(resolution=N)` is bounded by source pixels, with
no model-viewer cap.

Each write checkpoints the job and is undoable with `undo()` and `redo()`.
`restore_registration(sections=[{"id": ..., "registration": n}, ...])` puts
sections back to registrations they held earlier (their status rows'
`registration`), whatever was written since, as one undo step;
`restore_registration(section=...)` lists one section's, and `look(...,
registrations={id: n})` draws one without writing it
([writes](../reference/tools.md#restore_registration)). `look(...,
sheet=True)` tiles the pictures into sheets
([looking](../reference/tools.md#look)).
The handle reloads other processes' saved changes before each call; a read
never waits for another process's write lock.
Registration requirements still apply. See [tools and options](../reference/tools.md).

`trace_borders`, `fireants_syn` and a plugin's long tools (`stalign_lddmm`) start
background work. Subsequent replies deliver its completion notice; `job.wait(timeout=None)`
waits for it and returns what the agent CLI's call returns: `background` (each finished
work's notice), `work` (id, kind, sections, status), `running`, and the pictures under
`artifacts` and `images`; leaving the `with` block or calling `close()` waits for
pending traces and picture writes. `submit(...)` checks the job and writes
the QuickNII and VisuAlign exports, not the per-section maps.
`export_maps(full_resolution=False)` writes the maps of placed sections at
any point, submitted or not; `True` uses the source pixels.
Each traced section's image-model reply is exported with its maps, to
`exports/traces/<name>.npz` ([format](../reference/file_formats.md#tracesnamenpz)).

## Create a job

```python
with langslice.create_job(
    "/scans/brain1",
    atlas="allen_mouse_25um",
    tasks=["position", "transform", "nonlinear"],
    pixel_size_um=0.65,
) as job:
    opening = job.look(mode="positioning")
```

Image formats are PNG, JPEG and TIFF. `create_job` creates or continues
`<images>/langslice/`; `job_dir=` chooses another location. Changed supplied
inputs are refused on resume; `fresh=True` starts over. Source images remain
untouched.

| Argument | Meaning |
|---|---|
| `atlas`, `plane` | BrainGlobe atlas and `coronal`, `sagittal` or `horizontal` plane |
| `tasks` | Enabled tasks: `position`, `transform`, `nonlinear` |
| `positions` | `{filename: position_mm}` along the atlas slicing axis |
| `transforms` | `{filename: [a,b,tx,c,d,ty]}` or saved transform dictionaries; normalized in-plane affine on the oriented section |
| `angles` | `{"pitch": degrees, "yaw": degrees}`, or `{filename: {"pitch": ..., "yaw": ...}}` |
| `orientation` | Per filename: a `rotation_deg` quarter turn and `flip` boolean; rotation then left-right flip |
| `pixel_size_um` | Source pixel size in µm: one number for every file, or `{filename: um}` (a file it does not name uses its metadata); otherwise file metadata, else uncalibrated (image width mapped onto the atlas plane's width) |
| `inputs` | Other `JobSpec.inputs`, including `order`, `damaged`, `channel_names` |
| `frozen` | What the agent may not change, per section: `{filename: ["orientation", "affine"], "*": ["angles"]}`; properties `position`, `angles`, `orientation`, `affine`, `deformation`; see [freezing](../reference/settings.md#frozen) |
| `disabled` | Tools and `tool:option` names the job does without, for example `["zoom", "look:positioning"]`; see [switching tools off](../reference/tools.md#switching-tools-and-options-off) |
| `image_model` | Optional provider, profile or callable for `trace_borders`; omitted creates a job without one |
| `plugins`, `plugin_config` | [Tool plugins](plugins.md) to add to the job's toolbox: `module:factory` references, factory functions defined at module level, or `ToolSpec` objects; options for a factory by reference. Loading runs their code. The references are saved with the job, so reopening it loads the same plugins |

Set `tasks` explicitly for a job that must determine positions. The library
default is `transform,nonlinear`, or `nonlinear` alone when every section has
a supplied transform. Other `JobSpec` fields can be passed as keywords,
such as `position={"interval_um": 100}`. Passing a `JobSpec` as the first
argument uses it directly; only `image_model`, `atlas_loader` and `emit`
may accompany it.

### Import a linear registration

```python
with langslice.create_job(
    "/scans/brain1",
    registration="/scans/brain1/quicknii.json",
) as job:
    print(job.imported)
```

Supported files are QuickNII/VisuAlign JSON or XML, DeepSlice CSV/JSON/XML,
and LangSlice `registration.json`. Positions, cutting angles, orientation
and affine transforms become supplied inputs; tasks default to `nonlinear`.
VisuAlign's nonlinear markers are not imported. `registration=` cannot be
combined with the placement arguments above. Check `job.imported` for
unmatched, missing or refused sections and warnings. See
[file matching and import constraints](../reference/file_formats.md#importing-a-registration).

## Image models

```python
model = langslice.image_model("openai-oauth")
with langslice.create_job(
    "/scans/brain2", registration="/scans/brain2/quicknii.json", image_model=model,
) as job:
    reply = job.trace_borders(section="s07.tif", restrict_to=["CA1"])
```

The job must enable Nonlinear. Provider names are `openai-oauth`,
`openai-api` and `gemini-api`; credentials are loaded as for the CLI.
Resolving a profile makes no model request. `model=` overrides the provider's
image-model default.

A profile pairs the model with its prompt and attachment order:

```python
base_prompt = langslice.default_prompt("openai-oauth", plane="coronal")
model = langslice.image_model(
    "openai-oauth",
    prompt_file="prompts/lab.txt",  # or prompt="..."
    photograph_first=True,
    name="lab",
)
```

`{plane}` in a custom prompt is replaced with the section plane. A job's own
base prompt can also be set in its settings,
`nonlinear={"image_prompt": "..."}` (`langslice linear run --image-prompt
FILE`), for any model without a prompt of its own; it is recorded as an
untested `custom` profile, and a profile with another prompt of its own is
refused.
`photograph_first=True` sends the clean photograph first and its bordered
copy second; `False` reverses them. Omitted uses the provider's order:
clean first for OpenAI, bordered first otherwise. Custom prompts or
attachment orders are recorded as `untested`, with their profile name and
prompt hash. Reopen such a job with the profile again to enable tracing;
without it, `open_job` leaves `trace_borders` off.

`image_model` also accepts a function or object with a `call` method. It
receives a request with `prompt`, `slice_image` (Image 1) and
`reference_images[0]` (Image 2), both PIL images. Return a PIL image, array
or `GeneratedSegmentation`: the photograph in the same frame, with corrected
borders in thin bright yellow. LangSlice extracts the lines and fits them
against the atlas. Custom callables are recorded as untested profiles too.

Each attempt saves the attachments, raw reply, base and sent prompts, and
their word diff. A profile's prompt and attachment order are part of the
cache key. Without an explicit profile, `open_job` uses the saved provider
when its credentials are present; otherwise it leaves tracing off.

## Your own method

A method of the script's own (an affine from a feature matcher, a
displacement field from SimpleITK, an outside engine's answer) goes into the
job through `job.context`. It is the context a [tool plugin](plugins.md#toolcontext)'s
handler gets, handed to the script: the same reads (`fit_frame`,
`section_raster`, `atlas_slice`, `section_inputs`, `read_registration`, ...),
conversions (`transform_from_matrix`, `deformation_from_map`, ...) and writes
(`set_positions`, `set_transforms`, `set_deformations`, `mark_damage`,
`add_layer`), through the operations the native tools use. `help(job.context)`
lists them. For a script:

- each write is recorded as `script`: a transform's `kind`, a deformation's
  step `method` and record engine, and the record's provenance `script`
  (`call`, `note`, `meta`, and `argv0`, the path of the running script); each
  write that lands is a `script_write` line in `logs/events.jsonl`;
- each outermost write is one undo step; `with job.context.writing():` makes
  several one step;
- each read first picks up what another writer saved (a running agent, a CLI
  call), so `section_inputs` digests are taken from the job as it stands;
- a write to a property the job freezes, or that a task that is off
  supplies, is that section's row `FROZEN`; a placement write drops the
  deformations it made stale in the same undo step; the job's `disabled` list
  does not gate a script's writes (it switches tools off to save the agent's
  context, [switching tools off](../reference/tools.md#switching-tools-and-options-off));
- `work_dir()` and `start_work()` are refused (`NOT_FOR_SCRIPTS`): a script
  keeps its files in `scripts/` and `scratch/` and is its own computation.

A long computation takes `digest = c.section_inputs(id)` first and passes
`expect={id: digest}` to the write: a section whose placement, made images or
damage changed meanwhile is that row's `STALE_INPUT`, nothing written for it.
Without `expect=` the write lands on the section as it now stands.

`c.preview_overlay(id, proposal, resolution=None)` draws the overlay `look`
draws (atlas borders on the section) with a proposal NOT written
(`{"deformation": map or DeformationProposal}`, or `{"transform": ...,
"position_mm": ..., "angles": (pitch, yaw)}`) and a caption band saying so,
and saves it as a numbered picture of the job (tool `script`, mode
`overlay`) before it returns: a `SavedPicture` (`path` of its `view.jpg`,
`index`, the picture number `zoom` takes, `caption`, `image`). `zoom` crops
it from the saved picture. A linear proposal's picture keeps its layers
(`labels.tif`, `borders.png`, the frame); a deformation's has none, its
record not being in the job. `c.preview_positioning(proposal, sections=None,
resolution=None, part=None)` does the same for the positioning picture,
`{id: {"position_mm", "angles"}}`. The written result is `job.look(...)`.

### A worked example

A multi-resolution B-spline with mutual information (SimpleITK, a core
dependency) stands for "your engine": mutual information because the stain
and the atlas template differ in brightness.

```python
import numpy as np, cv2, SimpleITK as sitk, langslice

job = langslice.open_job("/scans/brain1")
c, sid = job.context, "s07.tif"
digest = c.section_inputs(sid)                       # taken before computing: for expect=
f = c.fit_frame(sid, stain=True)                     # the section and its atlas plane
h, w = f.image.shape[:2]
placed = f.placed_atlas_to_section                   # 3x3: native atlas px -> fit-grid px (current placement)
atlas_on_grid = cv2.warpAffine(f.atlas["template"], placed[:2], (w, h), flags=cv2.INTER_LINEAR)

fixed = sitk.GetImageFromArray(f.stain[..., 0].astype(np.float32))      # the section, fit grid
moving = sitk.GetImageFromArray(atlas_on_grid.astype(np.float32))       # the atlas, same grid
reg = sitk.ImageRegistrationMethod()
reg.SetMetricAsMattesMutualInformation(32)           # stain and template differ in brightness
reg.SetMetricSamplingStrategy(reg.RANDOM); reg.SetMetricSamplingPercentage(0.2, seed=1)
reg.SetOptimizerAsLBFGSB(numberOfIterations=100)
reg.SetShrinkFactorsPerLevel([4, 2, 1]); reg.SetSmoothingSigmasPerLevel([2, 1, 0])
reg.SetInitialTransform(sitk.BSplineTransformInitializer(fixed, [8, 8]), inPlace=True)
reg.SetInterpolator(sitk.sitkLinear)
found = reg.Execute(fixed, moving)                   # a B-spline: section point -> placed-atlas point
native_xy = c.map_from_itk(sid, found)               # the full map: fit-grid px -> native atlas px

print(c.preview_overlay(sid, {"deformation": native_xy}).path)        # look before writing
done = c.set_deformations([{"id": sid, "map": native_xy, "note": "SimpleITK B-spline, MI"}],
                          expect={sid: digest})
print(done.reply())                                                     # rows: ok with the warp's numbers, or why not
job.look(mode="overlay", sections=[sid])                                # the written result, numbered
```

`map_from_itk` does the bookkeeping that is the usual mistake by hand (ITK
arrays are `[y, x]` with `(dx, dy)` last, a transform maps fixed points to
moving ones, the placed atlas is not the native plane); the preview is where
a wrong map shows. The fit compares the whole grid, so where the atlas has
anatomy the section lost (a tear, missing ventral tissue, tissue past the
image's edge) it pulls the nearby tissue toward the empty outlines, and at a
cut image edge it smears the edge's pixels outward; leave such parts out with
a fixed mask (`reg.SetMetricFixedMask`).

### An outside engine's files

`c.write_frame(id, folder, format="tif")` writes what an engine outside
LangSlice registers: on the fit grid `section` (RGB), `stain` (the
preprocessed channel the native fits read) and `placed_template` /
`placed_annotation` (the atlas plane drawn there at the section's current
placement), on the native atlas grid `template` and `annotation`, and
`frame.json` (the sizes, `placed_atlas_to_section`, `atlas_to_section`,
`native_to_um`, the fit grid's millimetres per pixel). `format` is `tif`,
`nii.gz`, `nii`, `mha` or `nrrd`; every image has spacing 1 and origin 0, so
an engine's answer is in pixels.

`c.map_from_itk(id, source, fixed="section", atlas="placed", spacing=1.0)`
reads the answer back as the full map `set_deformations` takes: a
displacement-field image (`.nii.gz`, `.mha`, `.nrrd`), a transform file
(`.mat`, `.txt`, `.tfm`, `.h5`, as ANTs and ITK write them), a SimpleITK
transform or vector image, or a `(H, W, 2)` array of `(dx, dy)` pixels. It is
read in ITK's convention (a fixed point to its moving point); `fixed="atlas"`
says the engine had the atlas fixed (the map is inverted on the fit grid),
`atlas="native"` that it registered against `template` rather than
`placed_template`. `c.transform_from_itk(id, source, ...)` reads an affine
(ANTs' `0GenericAffine.mat`, either way round) as a `TransformProposal`;
a transform that is not affine is `BAD_MATRIX`. For Elastix, `transformix
-def all` writes the deformation field file.

### Points, regions and atlas planes

`c.to_atlas_um(id, xy, frame="file")` takes `(N, 2)` `[x, y]` pixels of the
image file (`frame="fit"`: of the fit grid) to `(N, 3)` atlas micrometres
through the section's current registration, its deformation included (NaN
outside the atlas). `c.from_atlas_um(id, xyz_um, frame="file")` goes back,
each atlas point projected onto the section's plane and carried through the
deformation's inverse; `c.plane_distance_um(id, xyz_um)` says how far off the
plane each lies. `c.region_at(id, xy, frame="file")` gives `{"id", "acronym",
"name"}` under each point, None outside the atlas.

`c.atlas_plane(position_mm, pitch_deg=0, yaw_deg=0, kinds=("template",
"annotation"))` samples the atlas plane at any placement (an `AtlasPlane`:
`atlas`, the arrays on the native grid, and `plane`, with `native_to_um`, as
`fit_frame` gives it), so a script can search positions and angles itself.
`c.atlas_volume(kind)` is a whole atlas volume (`annotation` or an atlas
image), read only.

### The native engines with settings of your own

`c.fit_fireants(id, restrict_to=(), exclude=(), section_image="preprocessed",
atlas_image="nissl", mask="none", smoothness_um=80.0,
metric="mutual_information", edges=True, detail="standard", start="linear",
marked=True, note="")` runs FireANTs SyN, the engine `fireants_syn` runs, and
returns a `DeformationProposal` NOT written. Its settings are `fireants_syn`'s,
by its names and with its defaults (`restrict_to` fits those regions'
neighbourhood only; `exclude` adds regions to leave out). `mask` is `"none"` or
the NAME of a mask of the section that `make_mask` made (`c.mask(id, name)`
reads one: `UNKNOWN_MASK`, `STALE_MASK`); an array is refused (`BAD_ARGS`, make
the mask with `make_mask` so the job holds it). `start="current"` fits on top
of the applied deformation and the written step keeps the chain; `marked=False`
keeps the section's marked damage regions in the fit. The proposal's provenance
holds the fit's `inputs` and `estimated` blocks, as a tool's reply does.
`c.fit_elastix(id, restrict_to=(), exclude=(), section_image="preprocessed",
atlas_image="nissl", mask="none", edges=True, marked=True, note="")` runs the
Elastix affine `elastix_affine` runs, with its names and defaults, from the
current placement, and returns a `TransformProposal` (refused
`HAS_DEFORMATION` on a section that holds a deformation).
`set_deformations` / `set_transforms` write them as the script's: the
FireANTs call's settings as the deformation step's `settings` (what `status`
and the registration list show) and in the record's provenance, the engine's
own numbers in its `engine.native_parameters`, the start and runtime in
provenance `meta`; the Elastix call's settings under the transform record's
`fit`. `c.image_names(id)` lists the section images a fit can read, and
`c.fit_frame(id, image="preprocessed")` gives one of them on the fit grid
(`frame.section_image`, and `frame.section_image_ref`: `{name, kind, file,
digest}`; `stain=True` is the same for `preprocessed`); `atlas=("nissl",)`
asks for the atlas's Nissl plane.
`c.deformation_from_map(..., weights=)` stores an engine's own sorting (float
`(H, W)` on the fit grid) with the record, where `make_mask(source="fit")` finds it;
`start="current"` marks a map made on top of the held deformation (read
through `frame.applied_map`), so the written step keeps the section's chain;
`settings=` (the call's, by the tool's names), `engine_settings=` and
`engine_version=` are kept as a native fit keeps its own
([plugins](plugins.md#toolcontext)).

### An image of your own for the fits

A script hands the fits and the pictures an image it computed (a colour
deconvolution of a brightfield stain, a cell-density map, an edge map) with
`c.add_image(id, name, array, note="")`: `array` is `(H, W)` on the section's
working grid, `c.working_grid(id)`, in the image file's orientation, before the
section's quarter turn and flip. Compute it from `c.raw_plane(id, name)` (one raw
channel, `uint8` `(H, W)`) or `c.working_rgb(id)` (the working copy as read,
`(H, W, 3)`), which are in that frame; never from `c.section_raster(...)`, which
is turned and flipped as the section is (a flipped section's raster handed back
would be a mirror image). It is `uint8`, or floats from 0 to 1. The image is saved as `sections/<stem>/images/<name>.png`
with `note` as its source, one undo step; then `job.look(mode="section",
sections=[id], images=[name])` draws it and every fit reads it as
`section_image=name`. A haematoxylin channel, for example:

```python
from skimage.color import rgb2hed

rgb = c.working_rgb(sid)                     # the file's orientation, working size
hema = rgb2hed(rgb)[..., 0]
hema = (hema - hema.min()) / (hema.max() - hema.min())
c.add_image(sid, "hema", hema, note="haematoxylin, skimage rgb2hed")
```

### Marks of your own on the pictures

`c.add_layer(id, Layer.points(...) | Layer.lines(...) | Layer.labels(...) |
Layer.scalar(...) | Layer.picture(...))` (`from langslice.plugin_api import
Layer`; [layers](plugins.md#layers)) leaves the script's detections (cells,
edges, matched points) on a section; its file goes under `plugins/script/`,
`status` lists it and `job.look(mode="overlay", sections=[id],
layers=[name])` draws it, as outlines and marks, never fills. Undoing the
write hides it.

### Already in reach

A proposal's numbers without writing it: `c.deformation_from_map(id,
xy).summary` (displacement, fold fraction, flags), the same numbers a native
fit's reply carries. The current composed map: `c.fit_frame(id).applied_map`,
`job.section_maps(id)`. Raw and preprocessed rasters at any size:
`c.section_raster(id, level=, images=["preprocessed"])`, `raw=True`. The
damage masks every native fit leaves out: `c.fit_frame(id).excluded`,
`.damage_zone`. A registration made elsewhere: `c.read_registration(path)`.
The atlas object: `job.workspace.atlas`, `langslice.load_atlas(name)`. Run
notes: `job.note(text=...)`.

`read_registration` reads any file the script names: a benchmark that keeps a
ground-truth registration inside or beside the job folder would be readable
through it, so answer keys are kept out of the job folder.

## Reading results

`langslice.coordinate_map(view_json)` returns a saved picture's pixel-to-atlas
coordinates as a `(rows, cols, 3)` float32 array in atlas micrometres,
including its deformation.

`job.section_maps(section, full_resolution=False)` returns one section's
maps in memory, writing nothing: the arrays `export_maps` writes, as a
`langslice.core.maps.SectionMaps`. `coords` is `(rows, cols, 3)` float32
atlas micrometres (NaN outside the atlas volume), `labels` `(rows, cols)`
uint32 atlas ids, `residual` `(rows, cols, 2)` float32 `(d_row, d_col)` in
grid pixels over the whole grid (None without a deformation),
`pixel_to_atlas_um` the grid's 3x3 matrix on `[row, col, 1]`, so that
`coords = pixel_to_atlas_um @ [row + d_row, col + d_col, 1]`, plus
`grid_size` (`(width, height)`), `um_per_px` and `full_resolution`. The grid
is the section's working copy, or with `full_resolution=True` the image
file's own pixels. A section without a placement raises `ValueError` saying
why. `job.export_stack()` writes what `submit` writes after its step, without
submitting (the QuickNII and VisuAlign exports, `registration.json`, the
image-model trace replies; no per-section maps) and returns the files
written. `langslice.load_atlas(name)` loads a cached
BrainGlobe atlas. The handle exposes `folder`, `imported`, `image_model`
and, for lower-level work, `job`, `state` and `workspace`.

[File formats](../reference/file_formats.md) defines the coordinate conventions, map
arrays and exports. Change a registration through the tools; editing
exported files does not change the job.

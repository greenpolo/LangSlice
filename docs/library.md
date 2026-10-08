# Python library

The library lets a coding agent write its own registration workflow: open a
job, call tools, inspect their pictures and choose the next step. It uses
the same operations, checkpoints and outputs as the [agent interfaces](agents.md).
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
The handle reloads other processes' saved changes before each call.
Registration requirements still apply; agent look-before-write gates do
not. See [registration tools](registration.md).

`trace_borders` starts background work. Subsequent replies deliver its
completion notice; leaving the `with` block or calling `close()` waits for
pending traces and picture writes. `submit(...)` checks the job and writes
maps and exports. `export_maps(full_resolution=False)` exports placed
sections at any point without submitting; `True` uses the source pixels.

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
| `pixel_size_um` | Source pixel size; otherwise file metadata, then a tissue-width estimate |
| `inputs` | Other `JobSpec.inputs`, including `order`, `locked`, `damaged`, `keep_warp`, `nonlinear_skip`, `channel_names` |
| `image_model` | Optional provider, profile or callable for `trace_borders`; omitted creates a job without one |

Set `tasks` explicitly for a job that must determine positions. The library
default is `transform,nonlinear`, or just `nonlinear` when every section has
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
[file matching and import constraints](file_formats.md#importing-a-registration).

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

`{plane}` in a custom prompt is replaced with the section plane.
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

## Reading results

`langslice.coordinate_map(view_json)` returns a saved picture's pixel-to-atlas
coordinates as a `(rows, cols, 3)` float32 array in atlas micrometres,
including its deformation. `langslice.load_atlas(name)` loads a cached
BrainGlobe atlas. The handle exposes `folder`, `imported`, `image_model`
and, for lower-level work, `job`, `state` and `workspace`.

[File formats](file_formats.md) defines the coordinate conventions, map
arrays and exports. Change a registration through the tools; editing
exported files does not change the job.

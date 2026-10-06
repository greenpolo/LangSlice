# The Python library: scripted registration in a pipeline

For a microscopy pipeline that registers sections from a script, without the
LangSlice agent: the image model traces the atlas borders onto each section
from its linear placement, a deformable fit turns the traced lines into a
deformation, and the results are written as per-section coordinate and label
maps plus QuickNII / VisuAlign exports.

Everything here runs through the same job folder and the same operations as
every other way into LangSlice (the agent, the agent CLI, ABBA): a script
calls the verbs an agent would, in a fixed order. `import langslice` is light
and loads no agent framework or model client; each public name is imported on
first use (`tests/test_core_imports.py` checks it).

Code: `src/langslice/doors/library.py` (`create_job`, `open_job`),
`src/langslice/doors/pipeline.py` (`register_section`, `register_job`),
`src/langslice/providers/profiles.py` (`image_model`, `default_prompt`).

## Install

LangSlice needs Python 3.10 or newer. From a source checkout:

```bash
conda env create -f environment.yml   # everything; or: pip install ".[registration,mcp,abba]"
```

Atlases are BrainGlobe's, downloaded on first use into `~/.brainglobe/`.

The image model's credentials: `langslice login` for `openai-oauth` (a
ChatGPT subscription sign-in), or an API key in the environment
(`OPENAI_API_KEY` for `openai-api`, `GEMINI_API_KEY` or `GOOGLE_API_KEY` for
`gemini-api`), in a `.env` file, or saved by the ABBA setup dialog.
`create_job` and `open_job` load them the way the CLI does. A model of your
own (below) needs none of this.

## The public API

| Name | What it does |
|---|---|
| `langslice.image_model(source, ...)` | the image model profile the border trace calls |
| `langslice.default_prompt(provider, plane)` | LangSlice's own trace prompt, the text to start a prompt of your own from |
| `langslice.register_section(image, ...)` | one section, start to finish |
| `langslice.create_job(images, ...)` | the job of a folder of sections (or a `JobSpec`), from supplied placements or a registration file made elsewhere; a job handle |
| `langslice.register_job(job, ...)` | the scripted registration over a job's sections |
| `langslice.open_job(folder, ...)` | an existing job; the same handle, every verb a method |
| `langslice.RegistrationError` | raised by `register_section` when its section could not be registered |
| `langslice.coordinate_map(view_json)`, `langslice.load_atlas(name)` | a saved picture's atlas coordinates; a cached BrainGlobe atlas |

## Choosing an image model

`langslice.image_model(source, *, model=None, prompt=None, prompt_file=None,
photograph_first=None, name=None)` returns the model the trace calls. `source`
is one of:

- **A provider name**: `"openai-oauth"`, `"openai-api"` or `"gemini-api"`
  `model` names the image
  model; without it the provider's default (`gpt-image-2` on `openai-oauth`).
  Without a prompt this is the provider's **built-in profile**: LangSlice's
  prompt for that provider.
- **A model of your own** (bring your own model): a plain function, or any
  object with a `call` method (and, optionally, `provider` and `model`
  attributes). It is called once per section with a request whose
  `prompt` is the text to send, `slice_image` is Image 1 and
  `reference_images[0]` is Image 2 (both PIL images of the same size), and
  returns the edited image: a PIL image, an array, or a
  `langslice.core.nonlinear.types.GeneratedSegmentation`. Its reply must be
  the photograph with the corrected borders drawn in thin bright yellow, in
  the same frame: LangSlice extracts the yellow lines from it.
- **An `ImageModel`** (`langslice.providers.registry.ImageModel`), returned
  as it is when nothing else is given.

Resolving a provider name contacts nothing; the model is called only when a
section is traced.

## Profiles: a model and the prompt written for it

Different image models need different prompts, so a profile pairs a model
with its prompt. The built-in profiles use LangSlice's own prompts:
the GPT wording for `openai-oauth` and `openai-api` (the clean photograph is
Image 1, the photograph with the placed borders Image 2), the general wording
for every other provider (the placed borders first). To make your own:

```python
import langslice

print(langslice.default_prompt("openai-oauth"))     # start from LangSlice's text
model = langslice.image_model(
    "gemini-api", model="my-image-model",
    prompt_file="prompts/lab_v1.txt",               # or prompt="..." as text
    photograph_first=True,                          # the order your prompt describes
    name="lab-v1",
)
model = langslice.image_model(my_edit_function, prompt_file="prompts/lab_v1.txt")
```

- `prompt` (text) or `prompt_file` (a UTF-8 text file) is the base prompt;
  `{plane}` in it becomes the section's plane (`coronal`, `sagittal`,
  `horizontal`). Edit the file to change the prompt; nothing else needs to
  change.
- `photograph_first` is the attachment order your prompt describes: True,
  Image 1 is the clean photograph and Image 2 the same photograph with the
  placed borders; False, the other way round; None (default), the order of
  the provider's own prompt (photograph first for the OpenAI providers,
  borders first for every other provider and for a model of your own).
- `name` names the profile in every record (default `"custom"`).

**Untested profiles are marked.** A profile with a prompt or attachment order
of its own, or a model of your own, is not one LangSlice's prompts were written for:
every border trace it makes records `"untested": true` and `"profile":
"<name>"` in the trace record (each attempt's `request.json` and
`result.json` under `sections/<name>/image_correction/`, and the section's
`image_correction` in `state.json`), and `job.json` records the profile under
`image_model` (`provider`, `model`, `profile`, `tested`, the prompt's
SHA-256). The result objects below carry it as `SectionOutput.untested`. A
built-in profile adds nothing to the record.

**What is saved.** Every trace attempt keeps the prompt sent (`prompt.txt`),
the base prompt it was edited from (`base_prompt.txt`: the profile's prompt
for a custom profile) and their word diff (`prompt_diff.txt`), with the two
images sent and the model's raw reply. A job's verb `trace_borders(id,
prompt=...)` still takes a lightly edited copy of the base prompt for one
section; the diff is then against the profile's prompt. A profile's own
prompt and attachment order are part of the key a saved reply is reused
under, so a changed prompt is never answered by an old reply; the same inputs
at the same placement reuse the first reply.

A job made with an untested profile and reopened with `open_job` without
`image_model=` offers no `trace_borders`, so it is never traced again with a
different prompt by accident; pass the profile again to trace. Without
`image_model=`, a job whose provider has no key or login on this machine
offers no `trace_borders` either (the check the MCP door and the agent CLI
make: `doors.api.setup.image_model_connected`, offline, presence only), so
`register_job` fits such a job to the stain.

## One section: `register_section`

```python
result = langslice.register_section(
    "/scans/brain1/s07.tif",                # a file path, or an array
    atlas="allen_mouse_25um", position_mm=6.2,
    pitch_deg=1.0, yaw_deg=-0.5,         # optional cutting angles
    transform=[1.02, 0.01, -0.01, 0.0, 0.98, 0.01],   # optional in-plane affine
    flip=False, rotation_deg=0,          # optional orientation
    pixel_size_um=0.65,                  # optional; else read from the file
    image_model=model,
    folder="/work/brain1_s07",              # optional; default a new temporary folder
    output="lean",                       # optional; "full" by default
    arrays=True,                         # optional: load the maps as arrays
)
section = result.sections[0]
section.coords, section.labels, section.maps     # file paths
section.arrays["coords"], section.arrays["labels"]
```

The placement:

- `atlas`, `plane` (default `"coronal"`), and `position_mm`: atlas
  millimetres from the anterior edge of the volume (BrainGlobe's
  orientation, the plane's axis).
- `pitch_deg`, `yaw_deg`: the section's cutting angles (default 0).
- `transform`: LangSlice's normalized in-plane affine, six numbers `[a, b, tx,
  c, d, ty]` on the oriented section: x as a fraction of its width, y of its
  height (`langslice.core.affine`; `registration.json` states it under
  `params_convention`). A transform dictionary as `registration.json` stores
  it is accepted too.
- `flip` (left-right) and `rotation_deg` (counter-clockwise quarter turns,
  0/90/180/270): rotation first, then flip, both before the transform.
- `pixel_size_um`: the image file's pixel size; without it, read from the
  file's metadata, else estimated from the tissue width.

What runs, on a job made for this one section (`create_job` on the section's
folder, then `register_job`):

- **With a transform:** `trace_borders` (the image model), then
  `fit_deformable` (the traced lines against the atlas borders, Elastix,
  medium stiffness), then `submit`, which writes the maps and the exports.
- **With a position only:** first `fit_affine`, the automatic linear
  alignment the agent's linear tool runs (Elastix by default, refining from
  no transform; `affine_method="silhouette"` fits the outline instead), then
  the same three steps.
- **Without an image model** (`image_model=None`): no trace; the deformable
  fit reads the section's stain instead.

`fit=` replaces the deformable fit's settings (any `fit_deformable`
argument, e.g. `{"fit_section": "traced_borders", "engine": "ants"}` with the
ANTs extra installed); `full_resolution=True` writes the maps on the image
file's own pixels instead of its working copy.

The input image:

- A **path** is hard-linked into `folder` (copied where the file system cannot
  link), under its own name; `name=` renames it.
- An **array** `(rows, cols)` or `(rows, cols, 3)` is written into `folder`
  as a lossless TIFF, `section.tif` by default (`name=` must end in `.tif`),
  with its dtype kept.

`folder` must hold no other section images. The job folder is
`<folder>/langslice` (or `job_dir=`). If the section cannot be registered,
`register_section` raises `RegistrationError` naming the step and the reason
(`.problems`); the job folder stays for a look.

## Many sections: the pipeline path

For a stack, make one job over the folder and run the same sequence on every
section:

```python
job = langslice.create_job(
    "/scans/brain1",                            # a folder of section images
    atlas="allen_mouse_25um",
    positions={"s01.tif": 5.9, "s02.tif": 6.1},           # by filename, mm
    transforms={"s01.tif": [1, 0, 0, 0, 1, 0]},           # optional, per section
    angles={"pitch": 1.0, "yaw": -0.5},                   # optional, the stack's
    orientation={"s02.tif": {"flip": True}},              # optional
    pixel_size_um=0.65,
    image_model=model,
    output="lean",
)
result = langslice.register_job(job)
for section in result.sections:
    print(section.id, section.coords, section.problem)
```

`register_job(job, *, sections=None, affine_method="elastix", fit=None,
full_resolution=False, arrays=False)` runs, per section: `fit_affine` when it
has no transform; `trace_borders` (started for every section, the calls run
concurrently in the background); `fit_deformable`, applied, four sections per
call (each fit waits for its section's trace); then `submit` once every
section is done. A section that fails a step is listed in `result.problems`
with the step and the reason, and the job is then not submitted: the maps of
every placed section are still written (`export_maps`). `result.ok` is True
when the job was submitted with no problems.

`create_job` arguments:

- `images`: the folder of section images (`.png`, `.jpg`, `.jpeg`, `.tif`,
  `.tiff`), or a `JobSpec` (then only `image_model`, `atlas_loader` and
  `emit` may be given besides).
- `positions`, `transforms`, `angles`, `orientation`, `pixel_size_um`: the
  registration you supply, as for `register_section`. `angles` is
  `{"pitch": deg, "yaw": deg}` for the whole stack, or
  `{filename: {"pitch": deg, "yaw": deg}}` to give each section its own
  plane (a section it does not name is flat); `inputs=` takes any
  other key of `JobSpec.inputs` (`order`, `damaged`, `locked`,
  `channel_names`).
- `tasks`: default `["nonlinear"]` when every section has a supplied
  transform, else `["transform", "nonlinear"]` (adds `fit_affine`). Any other
  task list makes the job of that list (the agent's tasks included).
- `registration`: a registration file made elsewhere, instead of the four
  placement arguments (below).
- `image_model`: as above; None makes a job without one.
- `job_dir`, `output`: the job-folder settings (below).
- `fresh=True` starts the folder's job over instead of continuing it
  (continuing with other supplied inputs is refused, naming the keys).
- Any other `JobSpec` field as a keyword (`preprocess="none"`,
  `position={"interval_um": 100}`, ...).

The handle (`create_job` and `open_job` return the same) has every verb the
job's tasks give as a method, with the agent tools' names and arguments
(`langslice-job ops`, `langslice-job schema VERB`): `job.fit_affine(slices=[...])`,
`job.trace_borders(id=...)`, `job.fit_deformable(slices=[...], ...)`,
`job.submit(summary=..., notes=[], interval_breaks=[])`,
`job.export_maps()`, `job.status()`, `job.undo()`. A script that needs another
sequence calls the verbs itself; `register_job` is only the default sequence.

Each method returns a plain dict of JSON values (`json.dumps(reply)` works):
the verb's reply as the agent CLI gives it under `result`, in full; every
status row with every field (null where a section has none: the last placed
section's `delta_to_next_mm`); and `artifacts`, the files of its pictures as
the CLI lists them (`path`, `kind`, `index`: the number the reply's
`image_indexes` give, `label` for the `view` JPEG). A method returns once its
pictures are written. The pictures themselves, as PIL images in the order a
model would receive them, are on the reply's `images` attribute
(`reply.images`), which is not a key.

`trace_borders` returns once its image-model call has started; the call lands
in the background. `job.close()` (or leaving a `with langslice.open_job(...)
as job:` block) waits for it and for any picture still being written; a
script that exits without closing has every job it opened closed for it at
exit.

## Starting from an existing registration

A stack already registered linearly elsewhere (QuickNII, VisuAlign,
DeepSlice, or an earlier LangSlice job) keeps that registration, and only the
image-model nonlinear step runs on top:

```python
job = langslice.create_job(
    "/scans/brain1",
    atlas="allen_mouse_25um",
    registration="/scans/brain1/quicknii.json",   # or .xml, a DeepSlice .csv, a registration.json
    pixel_size_um=0.65,                        # optional (below)
    image_model=model,
)
print(job.imported["sections"], job.imported["warnings"])
result = langslice.register_job(job)
```

The file is read and its entries matched to the folder's images: the same
file name, else the name without extension (ignoring case: a registration
made on PNG copies of TIFF scans), else one QuickNII section number
`_sNNN`. Each matched section gets the file's position, its own cutting
angles, its orientation and its in-plane transform, exactly: the new job's
`registration.json` maps every section file as the file does (a QuickNII
anchoring places the image by fractions of its width and height, so a
registration made on smaller copies carries over). The formats and what is
read: [file_formats.md](file_formats.md), "Importing a registration made elsewhere".

- `tasks` defaults to `["nonlinear"]`: the imported placement is kept as it
  is; `register_job` then traces and fits each section. Pass `tasks=` to
  let the automatic alignment or the agent change it.
- `registration=` cannot be combined with `positions`, `transforms`,
  `angles` or `orientation` (`ValueError`).
- The pixel size: `pixel_size_um`, else the files' own; when neither gives
  one, the size the imported registration implies is used (and said).
- `job.imported` is the import report: the file's `format`, each placed
  section (`id`, the file's `entry`, how it matched, `position_mm`,
  `pitch_deg`, `yaw_deg`, `rotation_deg`, `flip`), `unmatched` entries,
  `missing` sections (no placement: give them one or leave them out),
  `refused` sections with the reason (another atlas target, a plane tilted
  more than 45 degrees from the job's), the pixel size, and `warnings`
  (also passed to `emit`).
- VisuAlign markers are not imported: only the file's linear registration
  is, and a warning says so; LangSlice's nonlinear step replaces them.
- An ambiguous file (two entries for one section, or one entry naming two)
  and a file that places no section are refused (`ValueError`).

## Job-folder settings

- **Location.** By default the job folder is next to the images,
  `<images>/langslice/`; `job_dir=` puts it anywhere else (one per run or
  per configuration on one image folder, for example). A folder holding the
  job of another image folder is refused.
- **Output level.** `output="full"` (default) keeps everything a job writes.
  `output="lean"` keeps the results only and skips:
  - every picture (`views/`, `views.jsonl`, `views.seq`, and each section's
    `sections/<name>/views/`);
  - the undo history on disk (`history/`); undo and redo still work within
    the process that made the steps, so a reopened lean job starts without
    undo;
  - the event log (`logs/events.jsonl`);
  - the reference card for coding agents (`AGENTS.md`, `CLAUDE.md`).

  A lean job keeps `job.json`, `state.json` (the truth), `registration.json`,
  each section's maps, the deformation records and the image-model trace
  attempts (the fit reads them, and they record what was sent), and
  `exports/`. The level is a job setting (`JobSpec.output_level`, in
  `job.json`'s spec when lean), honoured by every door that opens the job.
- **Clean-up** is yours. LangSlice never deletes a job folder. Read what you
  need, then remove it: `shutil.rmtree(result.job_folder)`, or, for
  `register_section`'s default temporary folder, `shutil.rmtree(
  result.job_folder.parent)` (the section image's copy is in it).

## Outputs

`result.job_folder` holds the job's public files: `registration.json`, each
section's `coords.tif` (atlas micrometres per pixel), `labels.tif`,
`labels_fiji.tif` + `labels.csv`, `residual.tif`, `tissue.png`, `maps.json`,
and `exports/quicknii.json`, `exports/visualign.json`. Atlas coordinates are
BrainGlobe micrometres in the atlas's own axis order; image pixels are
`[row, col]` of the image file as stored. The maps are on the section's working
copy unless `full_resolution=True`. Every field and convention:
[file_formats.md](file_formats.md).

`RegistrationResult`: `job_folder`, `registration`, `exports` (by kind),
`sections` (stack order), `submitted`, `problems`, `ok`, `section(id)`.
`SectionOutput`: `id`, `folder`, `coords`, `labels`, `maps`, `residual`
(None without a deformation), `files` (every map file by kind), `trace` (the
trace record, None without one), `untested`, `problem`, and `arrays` (filled
by `arrays=True` or `read()`: `coords` `(rows, cols, 3)` float32, `labels`
`(rows, cols)` uint32, `residual` `(rows, cols, 2)` float32 when present).

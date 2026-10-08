# The Python library: scripting LangSlice

For a coding agent (or a person) who drives LangSlice from Python: open or
make a job, call the same verbs the LangSlice agent calls, look at the
pictures they return, and read the results from the job folder.

Everything here runs through the same job folder and the same operations as
every other way into LangSlice (the agent, the agent CLI, MCP, ABBA): a
method of a job handle is the agent tool of the same name, with the same
arguments. `import langslice` is light and loads no agent framework or model
client; each public name is imported on first use
(`tests/test_core_imports.py` checks it).

Code: `src/langslice/doors/library.py` (`create_job`, `open_job`, the job
handle), `src/langslice/providers/profiles.py` (`image_model`,
`default_prompt`).

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
| `langslice.open_job(folder, ...)` | an existing job (the job folder, or the image folder beside it); a job handle, every verb a method |
| `langslice.create_job(images, ...)` | the job of a folder of sections (or a `JobSpec`), from supplied placements or a registration file made elsewhere; the same handle |
| `langslice.image_model(source, ...)` | the image model profile `trace_borders` calls |
| `langslice.default_prompt(provider, plane)` | LangSlice's own trace prompt, the text to start a prompt of your own from |
| `langslice.coordinate_map(view_json)` | a saved picture's pixels in atlas micrometres |
| `langslice.load_atlas(name)` | a cached BrainGlobe atlas |

## The job handle

```python
import langslice

with langslice.open_job("/scans/brain1") as job:      # the job folder or its images
    job.verbs                                         # the verbs this job has
    rows = job.status()["rows"]
    reply = job.look(mode="overlay", sections=["s07.tif"])
    reply["pictures"], reply["artifacts"]             # numbers, captions; files
    reply.images                                      # the pictures, as PIL images
    job.position_sections(sections=[{"id": "s07.tif", "position_mm": 6.2}])
    job.elastix_affine(sections=["s07.tif"])
    job.zoom(box=[100, 80, 300, 220])                 # the newest picture, redrawn
```

Every verb the job's tasks give is a method, with the agent tools' names and
arguments: `langslice-job ops` lists them, `langslice-job schema VERB` gives
each one's arguments and description, and `job.verbs` lists this job's. The
look-before-commit gates of the agent's tools do not apply, and `look`'s
`resolution` takes any size from 128 px to the source's own pixels. A retired
verb's name raises an `AttributeError` naming the verb to use instead.

**Sections are named by filename.** Every argument that names a section takes
its image filename, or the filename without its extension when no other
section shares that stem. A number names no section: the call is refused
(`UNKNOWN_SLICE_IDS`) with the valid filenames listed.

**Replies.** Each method returns a plain dict of JSON values
(`json.dumps(reply)` works): the verb's reply as the agent CLI gives it under
`result`, in full; every status row with every field (null where a section
has none: the last placed section's `delta_to_next_mm`); and `artifacts`, the
files of its pictures as the CLI lists them (`path`, `kind`, `index`, `label`
for the `view` JPEG). A method returns once its pictures are written. The
pictures themselves, as PIL images in the order a model would receive them,
are on the reply's `images` attribute (`reply.images`), which is not a key.
A refusal is a reply too: `status` `"error"` (or `"refused"` for a gate),
`error` its code, and the facts behind it.

**Writes.** Every write is one undo step (`job.undo()`, `job.redo()`) and is
checkpointed in the job folder at once; an agent working on the same folder
picks it up, and the handle picks up the agent's writes before each call.

**Background work.** `trace_borders` returns once its image-model call has
started; the call and the fit of what it traced land in the background, and
the next reply opens with their notice. `job.close()` (or leaving the `with`
block) waits for it and for any picture still being written; a script that
exits without closing has every job it opened closed for it at exit.

**Ending.** `job.submit(summary=..., notes=[], interval_breaks=[],
left_linear=[])` checks the job's gates (every section placed, aligned and,
with Nonlinear, deformed or named in `left_linear` with its reason) and writes the maps and exports; `job.export_maps()`
writes them from the job as it stands at any time, without submitting.

The handle also has `folder`, `image_model`, `imported` (the import report of
`create_job(registration=...)`), and, for scripts that go below the verbs,
`job.job` (the `langslice.job.job.Job`), `job.state` and `job.workspace`.

## Making a job: `create_job`

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
)
```

- `images`: the folder of section images (`.png`, `.jpg`, `.jpeg`, `.tif`,
  `.tiff`), or a `JobSpec` (then only `image_model`, `atlas_loader` and
  `emit` may be given besides).
- `positions`: atlas millimetres from the anterior edge of the volume
  (BrainGlobe's orientation, the plane's axis), by filename.
- `transforms`: LangSlice's normalized in-plane affine, six numbers `[a, b,
  tx, c, d, ty]` on the oriented section: x as a fraction of its width, y of
  its height (`langslice.core.affine`; `registration.json` states it under
  `params_convention`). A transform dictionary as `registration.json` stores
  it is accepted too.
- `angles`: `{"pitch": deg, "yaw": deg}` for the whole stack, or
  `{filename: {"pitch": deg, "yaw": deg}}` to give each section its own
  plane (a section it does not name is flat).
- `orientation`: `{"flip": bool, "rotation_deg": 0|90|180|270}` per section;
  rotation first, then flip, both before the transform.
- `pixel_size_um`: the image files' pixel size; without it, read from each
  file's metadata, else estimated from the tissue width.
- `inputs=` takes any other key of `JobSpec.inputs` (`order`, `damaged`,
  `locked`, `channel_names`).
- `tasks`: the tasks the job's verbs come from (`reorder`, `position`,
  `transform`, `nonlinear`; a task that is off takes its answer from what you
  supplied). Default: `["nonlinear"]` when every section has a supplied
  transform, else `["transform", "nonlinear"]`.
- `registration`: a registration file made elsewhere, instead of the four
  placement arguments (below).
- `image_model`: below; None makes a job without one (no `trace_borders`).
- `job_dir`: where the job folder goes (below).
- `fresh=True` starts the folder's job over instead of continuing it
  (continuing with other supplied inputs is refused, naming the keys).
- Any other `JobSpec` field as a keyword (`preprocess="none"`,
  `position={"interval_um": 100}`, ...).

## Starting from an existing registration

A stack already registered linearly elsewhere (QuickNII, VisuAlign,
DeepSlice, or an earlier LangSlice job) keeps that registration:

```python
job = langslice.create_job(
    "/scans/brain1",
    atlas="allen_mouse_25um",
    registration="/scans/brain1/quicknii.json",   # or .xml, a DeepSlice .csv, a registration.json
    pixel_size_um=0.65,                        # optional (below)
    image_model=model,
)
print(job.imported["sections"], job.imported["warnings"])
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
  is. Pass `tasks=` to let the linear verbs change it.
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
  is, and a warning says so.
- An ambiguous file (two entries for one section, or one entry naming two)
  and a file that places no section are refused (`ValueError`).

## Choosing an image model

`langslice.image_model(source, *, model=None, prompt=None, prompt_file=None,
photograph_first=None, name=None)` returns the model `trace_borders` calls.
`source` is one of:

- **A provider name**: `"openai-oauth"`, `"openai-api"` or `"gemini-api"`.
  `model` names the image model; without it the provider's default
  (`gpt-image-2` on `openai-oauth`). Without a prompt this is the provider's
  **built-in profile**: LangSlice's prompt for that provider.
- **A model of your own**: a plain function, or any object with a `call`
  method (and, optionally, `provider` and `model` attributes). It is called
  once per traced section with a request whose `prompt` is the text to send,
  `slice_image` is Image 1 and `reference_images[0]` is Image 2 (both PIL
  images of the same size), and returns the edited image: a PIL image, an
  array, or a `langslice.core.nonlinear.types.GeneratedSegmentation`. Its
  reply must be the photograph with the corrected borders drawn in thin
  bright yellow, in the same frame: LangSlice extracts the yellow lines from
  it.
- **An `ImageModel`** (`langslice.providers.registry.ImageModel`), returned
  as it is when nothing else is given.

Resolving a provider name contacts nothing; the model is called only when a
section is traced.

### Profiles: a model and the prompt written for it

Different image models need different prompts, so a profile pairs a model
with its prompt. The built-in profiles use LangSlice's own prompts: the GPT
wording for `openai-oauth` and `openai-api` (the clean photograph is Image 1,
the photograph with the placed borders Image 2), the general wording for every
other provider (the placed borders first). To make your own:

```python
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
  `horizontal`).
- `photograph_first` is the attachment order your prompt describes: True,
  Image 1 is the clean photograph and Image 2 the same photograph with the
  placed borders; False, the other way round; None (default), the order of
  the provider's own prompt (photograph first for the OpenAI providers,
  borders first for every other provider and for a model of your own).
- `name` names the profile in every record (default `"custom"`).

**Untested profiles are marked.** A profile with a prompt or attachment order
of its own, or a model of your own, is not one LangSlice's prompts were
written for: every border trace it makes records `"untested": true` and
`"profile": "<name>"` in the trace record (each attempt's `request.json` and
`result.json` under `sections/<name>/image_correction/`, and the section's
`image_correction` in `state.json`), and `job.json` records the profile under
`image_model` (`provider`, `model`, `profile`, `tested`, the prompt's
SHA-256). A built-in profile adds nothing to the record.

**What is saved.** Every trace attempt keeps the prompt sent (`prompt.txt`),
the base prompt it was edited from (`base_prompt.txt`: the profile's prompt
for a custom profile) and their word diff (`prompt_diff.txt`), with the two
images sent and the model's raw reply. `trace_borders(section=...,
prompt=...)` takes a lightly edited copy of the base prompt for one section;
the diff is then against the profile's prompt. A profile's own prompt and
attachment order are part of the key a saved reply is reused under, so a
changed prompt is never answered by an old reply; the same inputs at the
same placement reuse the first reply.

A job made with an untested profile and reopened with `open_job` without
`image_model=` offers no `trace_borders`, so it is never traced again with a
different prompt by accident; pass the profile again to trace. Without
`image_model=`, a job whose provider has no key or login on this machine
offers no `trace_borders` either (the check the MCP door and the agent CLI
make: `doors.api.setup.image_model_connected`, offline, presence only).

## The job folder

- **Location.** By default the job folder is next to the images,
  `<images>/langslice/`; `job_dir=` puts it anywhere else (one per run or
  per configuration on one image folder, for example). A folder holding the
  job of another image folder is refused.
- **Contents.** `state.json` (the truth), `registration.json` (rewritten on
  every write), every picture a verb returned under `views/` and
  `sections/<name>/views/` with its index `views.jsonl`, the undo history,
  the logs, and the reference card for coding agents (`AGENTS.md`,
  `CLAUDE.md`). [file_formats.md](file_formats.md) has every file.
- **Clean-up** is yours. LangSlice never deletes a job folder.

## Outputs

`submit` and `export_maps` write each placed section's `coords.tif` (atlas
micrometres per pixel), `labels.tif`, `labels_fiji.tif` + `labels.csv`,
`residual.tif` (with a deformation), `tissue.png` and `maps.json` under
`sections/<name>/`, then `exports/quicknii.json` and
`exports/visualign.json`, and `registration.json` again. Atlas coordinates
are BrainGlobe micrometres in the atlas's own axis order; image pixels are
`[row, col]` of the image file as stored. The maps are on the section's
working copy unless `export_maps(full_resolution=True)`. Every field and
convention: [file_formats.md](file_formats.md).

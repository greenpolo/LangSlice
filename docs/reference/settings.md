# Job settings

A {term}`job` is made with settings: which atlas, which {term}`tasks <task>` the
agent does, what you already know about the sections, what the agent may not
change, and which tools it gets. They are saved in the job folder's
`job.json` (under `spec`) and hold for the whole job.

You usually do not type these settings. Tell your agent what you know ("the
sections are 40 µm thick, one in every six was kept, don't touch the
positions"), and it sets them when it creates the job with
`langslice-job FOLDER init`. Each entry below names the setting as it is
spelled in each place it can be given:

- **Flag**: the option of `langslice-job FOLDER init` and of
  `langslice linear run` (the same flags; `langslice-job schema init` lists
  them with their help).
- **Library**: the keyword of `langslice.create_job`.
- **`job.json`**: the key under `spec`.

A job already in the folder is continued with its saved settings. `init` with
other settings is refused with `SETTINGS_CHANGED`, and other supplied
`inputs` or `frozen` with `INPUTS_CHANGED`; `--fresh` starts the job over and
discards its work.

## Summary

| Setting | Default | In short |
|---|---|---|
| [`atlas`](#atlas) | `allen_mouse_25um` | BrainGlobe atlas |
| [`plane`](#plane) | `coronal` | Cutting plane |
| [`tasks`](#tasks) | `position`, `transform` | What the agent does |
| [`inputs`](#inputs) | none | What is already known per section |
| [`frozen`](#frozen) | none | What the agent may not change |
| [`disabled`](#disabled) | none | Tools and options switched off |
| [`plugins`](#plugins), `plugin_config` | none | Extra tools from tool plugins |
| [`position.*`](#position-settings) | | Section thickness and interval, spacing rule, guidance, notes |
| [`transform.*`](#transform-settings) | | Hemisphere cue, cutting angles, sections per call, notes |
| [`nonlinear.*`](#nonlinear-settings) | | Image model, base prompt, required deformation, notes |
| [`facts`](#facts) | none | Free-text facts for the agent |
| [`image_resolution`, `preprocess`, `force_view`](#pictures) | `low`, `auto`, off | The pictures the agent is shown |
| [`model`, `reasoning`, `debrief`](#built-in-agent) | | The built-in agent only |
| [`image_folder`, `job_dir`, `out`, `resume`](#folders) | | Where things are |
| [`notes`, `viewer`](#notes-and-viewer) | none, `claude` | Saved in `job.json` beside the spec |

## Atlas and plane

### `atlas`

Default `allen_mouse_25um`. Flag `--atlas`; library `atlas=`.

The name of a [BrainGlobe atlas](https://brainglobe.info/documentation/brainglobe-atlasapi/usage/atlas-details.html).
It downloads on first use. Coronal sections of the Allen mouse atlas are the
most tested case. Change it for another species or age (for example
`whs_sd_rat_39um` for rat), or for a finer voxel size of the same atlas.
`ccfv3augmented_mouse_25um` is the Allen 25 µm volume extended 0.35 mm at the
front, with a Nissl template.

### `plane`

Default `coronal`. Flag `--plane coronal|sagittal|horizontal`; library
`plane=`.

The plane the sections were cut in. Positions are read along the axis normal
to it: anterior-posterior for coronal, medial-lateral for sagittal,
dorsal-ventral for horizontal. Change it whenever your sections are not
coronal.

## What the agent does

### `tasks`

Default `position`, `transform` (Positioning and Linear). Flag
`--tasks position,transform,nonlinear`; library `tasks=`.

| Task | The agent writes | Its tools |
|---|---|---|
| `position` (Positioning) | each section's position and the stack's cutting angles | `position_sections` |
| `transform` (Linear) | each section's orientation (flip, quarter turn) and in-plane affine | `interactive_transform`, `elastix_affine` |
| `nonlinear` (Nonlinear) | each section's deformation on top of its linear placement | `fireants_syn`, `set_landmarks`, `landmark_triangulation`, `trace_borders` |

A task left out brings none of its tools and takes its answer from
[`inputs`](#inputs); its properties are frozen on every section. Nonlinear
needs a linear placement: with Positioning and Linear off, `inputs` must
supply positions and transforms, and a nonlinear fit refuses a section
without a transform (`NO_TRANSFORM_LINEAR_OFF`). Two other defaults apply: with `--registration` (an imported
registration) the default is `nonlinear` alone, and in the library with no
`tasks` it is `nonlinear`, plus `transform` unless every section has a
supplied transform.

Add `nonlinear` when you want each section bent onto the atlas after the
linear placement. Leave out `position` when the positions are known, and
`transform` when the in-plane alignment is known.

## What you already know

### `inputs`

Default empty. Library `inputs=` (and the keywords `positions=`,
`transforms=`, `angles=`, `orientation=`, `pixel_size_um=`). Each key has its
own flag:

| Key | Flag | Content |
|---|---|---|
| `positions` | `--positions JSON` | `{filename: mm}`: atlas millimetres from the anterior edge of the volume along the slicing axis. |
| `order` | `--order JSON` | `[filename, ...]`: the stack's order; sections it does not name follow. |
| `angles` | `--pitch`, `--yaw`, or `--section-angles JSON` | `{"pitch": deg, "yaw": deg}` for the whole stack, or `{filename: {"pitch": deg, "yaw": deg}}` per section (a section not named is flat). Not both forms. |
| `orientation` | `--orientation JSON` | `{filename: {"flip": true or false, "rotation_deg": 0, 90, 180 or 270}}`. |
| `transforms` | `--transforms JSON` | `{filename: transform}`: in-plane transforms. In the library, six numbers `[a, b, tx, c, d, ty]` or a transform record. |
| `damaged` | `--damaged JSON` | `{filename: note}`: notes for the agent. A note excludes no region from the fits; only the agent's `mark_damage` does. |
| `pixel_size_um` | `--pixel-size-um UM or JSON` | Micrometres per pixel: one number for every file, or `{filename: um}`. |
| `channel_names` | (hosts only) | One name per page of a multi-page image. |

Flags marked JSON take inline {term}`JSON` or the path of a JSON file. Any other key is
refused. What is supplied is a starting point: the agent may change it unless
[`frozen`](#frozen) says it may not.

`--registration FILE` (library `registration=`) fills `positions`, `angles`,
`orientation` and `transforms` from a registration made elsewhere: QuickNII or
VisuAlign JSON or XML, DeepSlice CSV, JSON or XML, or a LangSlice
`registration.json`. It cannot be combined with those keys or with
`--angles`. VisuAlign markers are not imported. See
[importing a registration](file_formats.md#importing-a-registration).

Supply `pixel_size_um` when the image files carry no pixel size in their
TIFF or OME tags. Without one, a section's scale comes from its width mapped
onto the atlas plane's width, and `registration.json` marks it
`uncalibrated`. Supply positions, angles, orientation or transforms when you
know them, so the agent starts from them.

### `frozen`

Default none. Flag `--frozen JSON`; library `frozen=`.

`{filename: [properties], "*": [properties]}`: per section, the properties the
agent may not change. `"*"` is every section; a section's own list and `"*"`
add up. The properties:

| Property | What it is | Kept when `inputs` supplies none |
|---|---|---|
| `position` | where the section sits along the slicing axis | nothing: a frozen position must be supplied in `inputs.positions` |
| `angles` | its cutting angles (pitch, yaw) | the flat plane |
| `orientation` | its flip and quarter turn, together | no flip, no turn |
| `affine` | its continuous in-plane transform | the identity |
| `deformation` | the deformation on top of its linear placement | none |

Every write that would change a frozen property is refused, by every tool and
every script, with `FROZEN`. A fit under a frozen `orientation` never mirrors
the section. The cutting angles are one plane for the stack, so
`position_sections` refuses new angles while any section has `angles` frozen.
A section with a frozen `deformation` needs no deformation at `submit`. A tool
that writes only properties frozen on every section is not offered, and a
task whose properties are all frozen on every section is off.

Common freezes:

- Positions, cutting angles and orientation known; affine and deformation
  left to the agent: `{"*": ["position", "angles", "orientation"]}` with
  those three in `inputs` and tasks `transform,nonlinear`.
- A section already aligned that must not be touched:
  `{"s_007.tif": ["orientation", "affine"]}`.
- A section left out of the deformation: `{"s_007.tif": ["deformation"]}`.

Freeze what you have checked and trust. Leave a supplied value unfrozen when
it is a starting guess the agent should improve.

### `facts`

Default none. Flag `--fact TEXT` (repeatable); library `facts=[...]`.

Free-form facts about the stack, one line each, passed to the agent
verbatim. Use them for what no other setting holds: the stain, the strain,
an injection site, a known problem with the slides.

## Which tools the agent gets

### `disabled`

Default none. Flag `--disable NAME[,NAME...]`; library `disabled=[...]`.

Tools and tool options switched off on top of what the tasks give: a tool
(`zoom`) or `tool:option` (`look:positioning`,
`interactive_transform:flip`). The valid names, what each removes and the
lists that are refused are in [Tools and options](tools.md#switching-tools-and-options-off).

Switch a tool off to keep the agent from spending its context on it, or to
take away a choice that does not apply: `interactive_transform:flip` when no
section can be mirrored, `trace_borders` when you want fits to the stain
alone. `disabled` is not a guard against a script; [`frozen`](#frozen) is.

### `plugins`

Default none. Flag `--tool-plugin MODULE:FACTORY` (repeatable;
`langslice linear run` only) and `--tool-plugin-config MODULE:FACTORY=JSON`;
library `plugins=` and `plugin_config=`.

Tool plugins add tools to the agent's toolbox: each `module:factory` is a
Python module and the function in it that returns the tools. Loading a plugin
runs its code, so plugins load only in the library and in
`langslice linear run`; the MCP server and the ABBA worker refuse a job that
names one. A plugin that cannot load is refused when the job is made.
`plugin_config` maps a plugin's reference to the options its factory is
called with. Use plugins to give the agent a registration method of your
own or a published one wrapped as a tool.

## Task settings

### `position` settings

| Key | Default | Flag | What it does |
|---|---|---|---|
| `position.thickness_um` | not supplied | `--thickness` | Section thickness, a whole number of µm. |
| `position.interval_um` | not supplied | `--interval` | Distance between consecutive sections in the stack, a whole number of µm. |
| `position.strict_interval` | off | `--strict-interval` | `submit` requires positions at most 10% from the interval; needs `interval_um`. |
| `position.playbook` | off | `--playbook` | Adds positioning guidance to the job statement: compare candidate planes and regions, write supported placements, revisit uncertainty, review the stack. |
| `position.notes` | empty | `--position-notes TEXT` | Your notes for this task, shown to the agent with it. |

Give the thickness and interval when you know them. Without them the agent
infers positions and spacing from the anatomy and records its estimates in
its notes. Use `strict_interval` only when every section of the series is
present and the interval was kept exactly.

### `transform` settings

| Key | Default | Flag | What it does |
|---|---|---|---|
| `transform.hemisphere_cue` | empty | `--hemisphere-cue TEXT` | What marks a hemisphere in these sections (a notch, an injection), shown where the agent may mirror sections. |
| `transform.angles` | off | `--angles` | Lets the agent set the stack's cutting angles when Positioning is off (`position_sections` with `cutting_angles` only). |
| `transform.max_parallel` | 4 | (library and `job.json` only) | Most sections one `interactive_transform` or `elastix_affine` call takes, 1 to 4. |
| `transform.notes` | empty | `--transform-notes TEXT` | Your notes for this task. |

Give a hemisphere cue whenever the sections carry one: without a visible
mark, a mirrored section can be told only from an oblique cutting angle.

### `nonlinear` settings

| Key | Default | Flag | What it does |
|---|---|---|---|
| `nonlinear.provider` | `openai-oauth` (library: `none`) | `--image-provider` | The {term}`image model`'s access: `openai-oauth` (ChatGPT sign-in), `openai-api`, `gemini-api`, or `none` (no image model; no `trace_borders`). |
| `nonlinear.image_model` | the provider's | `--image-model` | Which image model of that provider. |
| `nonlinear.image_prompt` | LangSlice's | `--image-prompt FILE` | The base prompt `trace_borders` sends, in place of LangSlice's own; `{plane}` in it becomes the section plane. Its traces are recorded as an untested profile. |
| `nonlinear.require_deformation` | off | `--require-deformation` | `submit` is refused until every section has a deformation at its current placement. |
| `nonlinear.notes` | empty | `--nonlinear-notes TEXT` | Your notes for this task. |

Set the provider to `none` when you have no image model, or want every
deformation fitted to the stain and landmarks alone. A custom base prompt
must describe the provider's attachment order (OpenAI providers: Image 1 the
photograph, Image 2 the placed borders; the others the other way round).
Leave `require_deformation` off unless every section must leave with a
deformation; off, the agent gives one where it improves on the linear
placement.

## Pictures

| Key | Default | Flag | What it does |
|---|---|---|---|
| `image_resolution` | `low` | `--image-resolution` | Long edge of the pictures the agent is shown: `low` 256 px opening tiles and 512 px later pictures; `medium` 384 and 768; `high` 512 and 1024; `auto` lets the agent choose each picture's size (`resolution`), from 128 px up to its model's largest image. |
| `preprocess` | `auto` | `--preprocess auto|none` | How the opening `preprocessed` image is made: `auto` blends the channels with local contrast enhancement; `none` shows the file as read, in gray. |
| `force_view` | off | `--force-view` | The tools that change the registration always return their picture: their `view` argument is not offered. |
| `host_preprocessing` | none | (hosts only) | A host's channel blend for snapshots sent one page per channel (the ABBA connector). |

Picture size and preprocessing change what the agent sees, never your image
files. `preprocessed` is also the image every fit reads by default, so
`preprocess` changes what the fits read unless the agent makes another
section image.

## Built-in agent

These apply to `langslice linear run` only. With a coding-agent app, the app
chooses the model.

| Key | Default | Flag | What it does |
|---|---|---|---|
| `model` | `openai-oauth/gpt-6.1-sol` | `--model` | The model that drives the run. |
| `reasoning` | the provider's | `--reasoning low|medium|high|xhigh|max` | Reasoning effort, for models that expose one. |
| `debrief` | on | `--no-debrief` | After `submit`, one more model call asks what tools the agent missed. |

## Folders

| Key | Default | Flag | What it does |
|---|---|---|---|
| `image_folder` | (required) | the folder argument | The folder of section images. |
| `job_dir` | `<images>/langslice` | `--job-dir PATH` | Where the job folder goes. A read-only image folder falls back to `~/.langslice/jobs/<id>/`. |
| `out` | `<job folder>/exports/linear_results.json` | `--out` | Where the built-in agent writes its results JSON. |
| `resume` | on | `--fresh` turns it off | Continue the job in the folder. Off: start over. |

## Notes and viewer

Two keys of `job.json` sit beside the spec rather than in it.

| Key | Default | Flag | What it does |
|---|---|---|---|
| `notes` | none | `--notes TEXT` (`init` only) | Your notes for the job, given to the registration agent by every door. |
| `viewer` | `claude` | `--viewer claude|codex|openai` (`init` only) | Which model reads the agent CLI's pictures; it sets their largest size (2000 px for Claude, 2048 px for Codex and OpenAI). |

Write your instructions for the agent in `notes`. Give `--viewer codex` when
a Codex agent works through the command line.

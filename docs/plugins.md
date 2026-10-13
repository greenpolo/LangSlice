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
`tests/fixtures/example_plugin.py` is a complete small one (a read tool and a
write tool on synthetic data):

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
| `name` | The tool's name: an identifier; not a native tool's, not a retired tool's, not another plugin tool's |
| `stub` | A function whose signature is the tool's arguments and whose docstring is its description, written like the stubs in `langslice/doors/declarations.py`. The description may use [needs marks](registration.md#switching-tools-and-options-off) (`[[?look:overlay text]]`) |
| `handler` | `handler(ctx, **arguments)`, returns a `ToolResult` |
| `kind` | `read` or `write` |
| `group` | `Common`, `Positioning`, `Linear` or `Nonlinear`: where a host lists it |
| `task` | `position`, `transform`, `nonlinear` or `None`: the tool exists only while that task is on |
| `writes` | The section properties it may write (`position`, `angles`, `orientation`, `affine`, `deformation`). A write outside this set is refused; a tool that writes only properties frozen on every section is not offered |
| `long` | Computes outside the job's write lock and applies under it (see [long tools](#long-tools-and-stale_input)) |
| `limits` | The most one call takes: `{"sections": 4}` refuses a longer `sections` argument (`TOO_MANY_SECTIONS`); `{"pictures": 2}` sends at most two pictures |
| `needs` | What the tool needs while it is on: groups of names, one of each on (`(("look:overlay",),)`), as for native tools |
| `version` | The plugin's own version, kept in the provenance |
| `meta` | JSON values kept with the tool in the provenance |

The checks run when the factory is called: a name that clashes, an unknown
`task`, `writes` outside the five properties, `writes` on a read tool, a bad
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

One per call; a handler must not keep it.

**The job.** `ctx.spec` (the job's spec as a read-only mapping), `ctx.job_dir`,
`ctx.work_dir()` (`<job>/plugins/<tool>/`, made on use), `ctx.task_on(task)`.

**Reading.** All in LangSlice's documented units and frames.

| Call | Gives |
|---|---|
| `ctx.sections()`, `ctx.section(id)` | `SectionView`: `id`, `index`, `file_size`, `pixel_size_um`, `pixel_size_source`, `position_mm`, `angles`, `flip`, `rotation_deg`, `affine` (`kind`, the six normalized numbers `matrix`, `knobs`, `mirrored`), `deformation`, `damaged_regions`, `damage_note`, `layers`, `frozen` |
| `ctx.section_raster(id, level=, channels=)` | `SectionRaster`: `array` (the section as its orientation sets it, RGB `uint8`, reduced to `level`: a long edge in pixels or `low` / `medium` / `high` / `auto`) with its frame: `to_section_px(xy)` and `from_section_px(xy)` convert `[x, y]` between the array's pixels and `section_px`, `mask_to_section_px(raster)` turns a raster computed on the array into the file's orientation |
| `ctx.section_image(id, ...)` | Only the `array` of `section_raster`. **Its pixels are not `section_px`** (see [frames](#frames)) |
| `ctx.atlas()` | `AtlasView`: `name`, `resolution_um`, `orientation`, `species`, `plane`, `shape` |
| `ctx.fit_frame(id)` | `FitFrame`: the section on its fit grid (`image`, `tissue`), the atlas plane at its position and angles on the native atlas grid (`atlas["template"]`, `["annotation"]`, and `["nissl"]` where the atlas has one), `atlas_to_section` (3x3), `plane` (the plane in atlas micrometres) |
| `ctx.atlas_slice(id, kind=)` | One of those atlas planes |
| `ctx.section_inputs(id)` | A digest of what a computation on the section reads (placement, appearance, damage marks); for [`expect=`](#long-tools-and-stale_input) |
| `ctx.read_registration(path)` | A registration file made elsewhere (QuickNII, VisuAlign, DeepSlice, `registration.json`) as `placements` (`id`, `position_mm`, `pitch_deg`, `yaw_deg`), nothing written |
| `ctx.layers(id)` | The layers the section shows |

**Converting.** `ctx.transform_from_matrix(id, matrix)` takes the 2x3 map from
the fit grid's pixels to native atlas pixels that registration engines return
and gives a `TransformProposal`: the transform as an Elastix fit would store it
(`params`, the knobs `interactive_transform` takes about the section's pivot,
`record`), refining the section's current placement.
`ctx.transform_from_params(id, params)` does the same for six normalized
numbers `[a, b, tx, c, d, ty]`. `proposal.row()` is a row for `set_transforms`.

**Writing.** Each write goes through the operation a native tool uses.

| Call | Writes | Needs in `writes` |
|---|---|---|
| `ctx.set_positions(rows, cutting_angles=None, expect=None)` | rows `{"id", "position_mm"}`; positions are clamped into the atlas range and reported; the stack is renumbered by position | `position` (and `angles` with `cutting_angles`) |
| `ctx.set_transforms(rows, expect=None)` | rows `{"id", "transform"}` (a `TransformProposal`'s `record`) or `{"id", "params", "note"?}` | `affine` |
| `ctx.mark_damage(rows, expect=None)` | rows `{"id", "regions", "note"?}`, as `mark_damage` | any write tool |
| `ctx.add_layer(id, layer)` | a [layer](#layers) | any write tool |

A write returns a `WriteResult`: `ok` is false when the call was refused whole
(`error`, `message`); otherwise `rows` has one row per section asked
(`status` `ok` or `error` with the code: `FROZEN`, `STALE_INPUT`,
`UNKNOWN_SLICE_IDS`, ...) and `written` the ids written. `result.reply()` is
those as reply fields. A read tool that writes, or a write outside `writes`, is
refused (`READ_TOOL_WRITES`, `WRITE_NOT_DECLARED`).

Everything a write tool's call writes is one undo step: `undo` reverts it
whole, the checkpoint and `registration.json` follow, a property the job froze
is refused for that section, and a write that changes a section's placement
clears its deformation as it does for a native tool. `with ctx.writing():`
holds the job's write lock around several writes.

**Showing.** `ctx.overlay_picture(id, proposal=None)` is the overlay `look`
draws, as it stands or with a proposal not written yet (`{"transform": ...,
"position_mm": ..., "angles": (pitch, yaw)}`); `ctx.caption(image, text)`.

### Long tools and `STALE_INPUT`

A tool declared `long=True` is not run under the job's write lock. It reads
`ctx.section_inputs(id)` for each section first, computes (an engine can take
minutes while the agent or other commands keep working), then writes with
`expect={id: digest}`. A section whose placement, appearance or damage changed
meanwhile comes back as that section's row `STALE_INPUT`, nothing written for
it; the others are written. A long call's writes are one undo step too, unless
another write landed on the job between two of them: then they stay separate
steps, so `undo` never takes back a write that was not the call's.

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

`look` takes `layers=[names]` (or `["all"]`) for modes `section` and `overlay`
once the job has plugins. Points and lines are drawn with a dark halo, label
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
or picture, and `look` has no `layers` argument.

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
| `BAD_LAYER`, `UNKNOWN_PLUGIN_LAYER` | A malformed layer; `look` naming a layer no drawn section has |
| `PLUGIN_FAILED` | The handler raised something other than `ToolError` |
| `MCP_TOOL_ERROR`, `MCP_TIMEOUT`, `MCP_UNAVAILABLE` | The adapter's server answered with an error, did not answer, or is not running |

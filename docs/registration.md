# Registration tools

LangSlice gives an agent a stack of sections, an atlas and tools for looking,
placing and fitting. The agent inspects the anatomy, chooses tools and
regions, reviews their pictures and revises the registration. Image-model
border tracing is optional; affine and ANTs fits can use the stain directly.

The tools have the same names across [CLI, MCP and agent sessions](agents.md)
and the [Python library](library.md). Use `langslice-job schema VERB --job
FOLDER` for exact arguments and limits.

## Tasks and supplied inputs

| Task | Tools | Result |
|---|---|---|
| `position` | `position_sections` | Atlas positions, section order and cutting angles |
| `transform` | `interactive_transform`, `elastix_affine` | In-plane alignment of each section |
| `nonlinear` | `ants_syn`, optionally `trace_borders` | Deformation on top of a linear placement |

Default tasks are `position,transform`. A disabled task takes its answer
from supplied inputs and removes its tools. `transform.angles` also enables
`position_sections` for cutting angles when Positioning is off. Hosts can
disable either linear method independently.

Sections are addressed by filename, or by a stem unique within the job.
Positions are millimetres along the atlas's slicing axis; in a coronal atlas
with BrainGlobe's `asr` orientation, they run from the anterior edge toward
the posterior. Ordering follows the positions. Cutting angles are pitch and
yaw in degrees. A host can supply positions, per-section or stack-wide
angles, orientation, transforms and pixel size, or import a
[registration file](file_formats.md#importing-a-registration).

`locked` sections keep their orientation and in-plane transform.
`keep_warp` protects a host's deformation, and `nonlinear_skip` excludes
sections from Nonlinear. Notes supplied through `damaged` are information
for the agent; actual damage exclusions are atlas regions marked with
`mark_damage`.

## Looking and calibration

| Tool | Use |
|---|---|
| `look` | `section`: tissue alone; `atlas`: atlas planes; `overlay`: tissue with registered atlas borders; `positioning`: sections and atlas along the slicing axis |
| `zoom` | Redraw a saved picture's pixel box, or an overlay's single atlas region |
| `grep_atlas` | Find region acronyms, names and ids |
| `grep_atlas_view` | Show the atlas with only the selected regions' borders |
| `status` | Read placements, settings, constraints and pending work |
| `list_files`, `search_files`, `read_file` | Inspect the job folder |

Every picture is numbered, captioned and saved. `zoom(picture=N,
box=[x0,y0,x1,y1])` redraws its selected area; omitting `picture` uses the
latest non-zoom picture. Section and overlay zooms reread the source image
at its native resolution; positioning zooms use the working copies.

For an overlay, `zoom(region="CA1:left")` crops to the entire selected
region with a small margin and shows only that region's border. It accepts
one acronym or numeric id, including descendants, optionally `:left` or
`:right`. It uses the saved picture's placement, deformation and display
settings. The region must occur in that plane; `region` and `box` are
mutually exclusive.

Section scale comes from supplied calibration or TIFF/OME metadata, else an
estimate from tissue width (reported as `estimated`). Positioning and
opening pictures use the same micrometres per pixel for section and atlas
panels. Tissue can therefore appear smaller than the atlas. The stored
affine contributes its isotropic scale, `sqrt(abs(det))`; each thumbnail
is framed around its anatomy without being stretched to fill the tile.

Overlay pictures show the applied deformation unless `warp="none"`.
Picture resolution affects display only, never fit geometry or stored
transforms. Source image files are never modified.

## Channels and damage

`set_channel_properties` changes raw-channel display contrast, gamma and
colormap across the stack. Automatic contrast is computed per section.
These settings affect pictures, not fits.

`set_preprocessed_channel_properties` changes the channel blend, CLAHE,
N4 correction and denoising used by the fits and image model, stack-wide or
per section. It returns before/after pictures. Inspect that channel with
`look(channels=["preprocessed"])`. Both channel tools are available in every
job; ABBA's agent-preprocessing switch controls which raw channels it exports.

`mark_damage` records atlas regions physically lost or badly displaced,
including one hemisphere when needed. Those regions are excluded from
elastix, ANTs and image-model tracing. Small tears, bubbles and folds do not
by themselves justify excluding an entire region. Hosts can disable this
tool with `agent_damage=False`.

## Positioning and linear alignment

`position_sections` writes positions and cutting angles. New jobs with
Positioning enabled give unsupplied sections provisional positions based on
the interval and atlas range. Status marks them `position_source="default"`;
the agent must write its own positions before submitting.

`interactive_transform` sets flip, quarter turn, rotation, scales, shear and
shifts for up to four sections. Values are absolute; omitted values keep
their current setting. Rotation and left-right flip orient the section
before its in-plane affine.

`elastix_affine` refines a placement against the atlas `template` or `nissl`
image. `restrict_to` selects the anatomy that steers it. Region selections
include descendants and may specify a side, such as `"CTX:left"`.
A restricted fit highlights those borders and may zoom its result picture
to them; other borders remain faint. Use region `zoom` for an isolated border.

A region too small for elastix is refused with `REGION_TOO_SMALL`: its
non-excluded support must span at least four native atlas pixels in each
in-plane dimension and contain at least sixteen pixels. The check happens
before padding or fitting. No fit is run or written for that section; use
a larger region or `interactive_transform`.

## ANTs deformation

`ants_syn` fits the atlas to the section's preprocessed stain, starting from
its current linear placement and any applied deformation. Each fit builds
on the previous one and can be undone. It accepts one to four sections,
`stiffness="soft"|"medium"|"firm"` and optional `restrict_to` regions.
The restricted atlas mask includes a 300 µm neighbourhood around them;
marked damage is excluded automatically.

Both registration engines accept `atlas_image="template"` (the default)
or `"nissl"`. The Nissl reference is read from BrainGlobe's
`ccfv3augmented_mouse_25um` atlas and is available for Allen mouse CCFv3
atlases; it downloads on first use.

The reply reports displacement, fold fraction and plausibility flags, and
shows the fitted overlay. These diagnostics report concerns rather than
rejecting fits. Judge the overlay against internal tissue anatomy.
Changing a section's position, orientation, cutting angles or linear
transform clears its deformation; the reply lists affected sections.

## Optional image-model traces

`trace_borders(section="s01.tif", restrict_to=["CA1"], prompt="")` asks an
image model to move the placed atlas borders onto the tissue, then fits an
ANTs deformation to the extracted lines. It requires a linear placement.
The model receives two images in the same frame: the preprocessed section
and that section with thin yellow atlas borders. `restrict_to` selects the
borders shown and fitted; empty uses all eligible regions. Marked damage
is excluded.

The agent can edit the base prompt for a section; blank uses it unchanged.
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
attachments, prompts, raw reply, extracted lines and fit separately.

`trace_borders` is offered only with Nonlinear enabled and an image model
connected. `provider="none"` disables it while keeping `ants_syn`.
Custom models and prompts are described in the [library guide](library.md#image-models).

## Writes, review and submission

Writes checkpoint the job and form undo steps; `undo` and `redo` persist
across sessions. Change tools return review pictures unless `view=False`;
a host's `force_view` setting removes that choice. Long fits compute outside
the write lock and recheck inputs before applying. A conflicting edit
produces `STALE_INPUT` for that section; unaffected results still apply.

`submit` checks the enabled tasks before writing maps and exports:

- Positioning requires a written position for every section. Strict spacing
  permits at most 10% deviation from the interval. Reported interval breaks
  must exceed 1.5 times the median spacing.
- Linear requires a transform for every section, including damaged ones.
- Nonlinear requires a deformation at the current placement, or an entry
  in `left_linear=[{"id":"s01.tif","reason":"..."}]`. A host setting
  `nonlinear.require_deformation=True` disallows leaving sections linear.
  Host-excluded sections are exempt. An image-model trace is never required.

Optional `position.gated` (`--gates`) requires agents to inspect a section
in overlay or positioning mode before changing its position, and review
every section in positioning mode after the last write before submitting.
These viewing gates apply to native agent and MCP tools; CLI and Python
calls still enforce the registration requirements above.

Results are [coordinate maps, labels and registration exports](file_formats.md).

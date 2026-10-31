# Tools

The registration agent works only through these tools; your agent calls them,
not you. Ask your agent to leave any tool out of a job (“don't use the image
model”), except `look`, `status`, `undo` and `submit`. Every argument is in the
[full reference](https://github.com/greenpolo/LangSlice/blob/main/docs/reference/tools.md).

## Look

In every run. These draw and read; they change nothing.

### `look`

Draws sections, atlas planes, the section with the atlas borders on it, and
the stack laid out along the cutting axis.

### `zoom`

Draws part of an earlier picture again, at the full detail of your image file.

### `grep_atlas`

Finds atlas regions by name, acronym or id, or names the region at a point of
a picture.

### `grep_atlas_view`

Draws atlas planes with chosen regions' borders.

### `status`

Lists each section's placement and what the run may change.

### `list_files`, `search_files`, `read_file`

Read the job folder.

## Prepare

In every run.

### `make_image`

Makes a version of a section to look at and fit: a blend of channels, more
contrast, less noise, inverted.

### `make_mask`

Marks the tissue a fit should compare, leaving out tears, holes and stains.

### `mark_damage`

Records atlas regions a section has lost. Every fit leaves them out.

## Place

When the agent places the sections.

### `position_sections`

Sets where each section sits along the cutting axis, and the stack's cutting
angles.

## Align

When the agent aligns the sections.

### `interactive_transform`

Mirrors, turns, rotates, scales, shears and shifts a section by hand.

### `elastix_affine`

Refines a section's alignment automatically with
[Elastix](https://elastix.dev).

## Warp

When the agent fits warps, on top of the alignment.

### `fireants_syn`

Bends the atlas onto a section with SyN, the deformable method of ANTs, run by
[FireANTs](https://github.com/rohitrango/FireANTs) on the graphics card when
there is one.

### `set_landmarks`, `landmark_triangulation`

Pairs of matching points on the atlas and the section, applied as a warp as
VisuAlign does. They move tissue that lies too far off for FireANTs.

### `trace_borders`

An image model redraws the atlas borders on the tissue; FireANTs then fits a
warp to its lines. Only when an image model is connected.

## Keep track

In every run.

### `note`

Adds a line to the run's notes.

### `undo`, `redo`

Take back the last change, or put it back.

### `restore_registration`

Puts one section back on a registration it held before.

### `submit`

Ends the run and writes the results.

### `export_maps`

Writes the region and coordinate maps. From the command line and Python only.

## Optional engines

Tool plugins, each run in an environment of its own. A job gets them when it
is made from Python or with `langslice linear run --plugins`.

### `stalign_lddmm`

Fits the atlas onto sections with
[STalign](https://github.com/JEFworks-Lab/STalign).

### `stalign_plane_search`

Estimates the cutting angles with STalign.

### `spomialign`

Registers sections with SPOmiAlign.

### `emlddmm_stack`

Fits the whole stack to the atlas at once with
[emlddmm](https://github.com/twardlab/emlddmm).

### `emlddmm_plane`

Estimates the stack's cutting angles and spacing with emlddmm.

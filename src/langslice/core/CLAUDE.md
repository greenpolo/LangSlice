# LangSlice `core/` — the picture builders

Package guide for `src/langslice/core/`. The repo-level `CLAUDE.md` holds the
project-wide rules. `AGENTS.md` here is a verbatim copy — edit one, mirror to
the other.

The core library's new home (layered refactor, phases 3b and 3c,
2026-10-03; the renders, phase 3d, 2026-10-04). It holds the pictures the
tools send and their layers, and the section renders, captions, canvas,
sheets and status table that were `linear/render.py`; the other existing
core modules (`space`, `affine`, `oblique`, `image_prep`, `atlas/`,
`deformable/`, `linear/display`, `linear/workspace`, `linear/transform`,
`linear/deformation`, ...) move in later, in one rename-only commit.

## The layer rule

- A core module takes plain inputs: the `linear.workspace.Workspace` (atlas,
  section files, render caches), the `linear.state.StackState` and its
  records, ids, numbers, the core's `linear.display.DisplayOptions`. It
  returns plain outputs: PIL images (captions burned in: tool images reach a
  model as bare attachments), numpy arrays, dicts and frozen dataclasses.
- It imports other core modules only. Never the job layer (`linear.job`),
  the operations (`ops`), a door (`linear.toolbox`, `linear.view_options`,
  `adk`, `mcp_server`), and never `google.*`, `litellm` or `openai`.
  `tests/test_core_imports.py` loads each module in a fresh interpreter and
  checks.
- No undo, no checkpoint, no look-before-commit gates, no wording for a
  model beyond the captions burned into a picture. Something that writes the
  stack is an operation (`ops/`); something that words a reply is a door.
- A picture is returned, never encoded: the doors package it (the ADK agent
  through `adk.media.packaged`, JPEG message parts; MCP as image blocks).

## Files

- `pictures.py` — the captioned section and atlas pictures the viewing tools
  send: `section_picture` (`view_slices`, `orient_slices`: the section
  tissue-framed, or mode `channels`, its raw channels side by side),
  `atlas_view_picture` (`view_atlas`), `stack_review` (`view_stack`: the
  contact sheet in written-position order and the spacing plot), and the two
  cached reference pictures, `reference_section_picture` and
  `reference_atlas_picture`, on `Workspace.picture_cache` (keyed by
  everything they draw; a section's keeps the caption of its first display,
  filenames being the stable identity).
- `placement.py` — a section on its physical canvas at a placement.
  `draw_canvas` is the one renderer of every placement picture
  (`view_placement`, `set_positions`, `fit_affine`, `adjust_transforms`):
  it draws from the render the options ask for at the call's size, the
  matrix and pivot carried onto it, an applied deformation resampling it
  first, through `core.canvas.physical_views`, and returns a `Canvas`:
  the panels and the `CanvasFrame` they were drawn in. `stored_placement`
  (the six stored numbers as a drawable matrix, identity without),
  `current_warp` (the applied record from a `deformation.RecordStore`, None
  when stale or unreadable), `placement_pictures` (one section-position pair
  in a placement mode, `PLACEMENT_MODES`: the canvas under the complete
  registration, `stacked`, or the two separate `side_by_side` references the
  door maps by index; returns `Placed`: images, row facts, the canvas),
  and the interactive transform's `stage` (the working frame, calibration,
  canvas and resolved pivot; `StageFailure` `ATLAS_RENDER_FAILED` /
  `BAD_PIVOT`), `Staged` and `staged_views`. The transform tools' pictures
  (phase 3d): `fit_picture` (`fit_affine`: a fit drawn from its `FitFrame`,
  one-sided regions with the sides the fit resolved) and `transform_views`
  (`adjust_transforms`: the staged knobs in the call's mode, or for `ab`
  the "candidate" overlay then the "stored" (or "identity") one,
  `ab_reference` giving what the B side draws: the six stored numbers, else
  the stored knobs about their pivot).
- `sections.py` — the section renders and their cache: `render_slice`
  (ROTATE first, then FLIP, then the display-only `--preprocess auto`
  enhancement; a non-default look drawn from the raw channels over the same
  frame, `_look_image`, each overlay channel dimmed by `fine_detail`),
  `render_cache_key`, `canvas_um_per_px`, `shown_section` (the larger render
  a picture is drawn from), `rescale_section_matrix`, `PREVIEW_LONG_EDGE`
  (512, the working frame every fit is computed on). Cached on the
  workspace (`render_cache`, `render_scale`): read, never mutate.
- `captions.py` — `caption` (a COPY with the text in a band above the
  picture; never caption an image a fit measures), `wrap_caption`, the
  fonts, `scale_bar_px` and the 1 mm bar `_draw_scale_bar`.
- `canvas.py` — the physical canvas: `CanvasGeometry` / `canvas_geometry`
  (the atlas section at true scale on the section's frame, anatomy centred,
  canvas grown to hold it plus `WORKING_MARGIN`), `VIEW_MODES`,
  `OUTLINE_LAYERS`, `normalize_border_style`, `line_coverage` (the one
  border rasteriser), `regions_left` / `region_polys`, `template_canvas`,
  `atlas_mask_canvas`, `zoom_box`, `pivot_on_canvas`, `placement_matrices`,
  `PanelFrame`, `physical_views` (the alignment picture in every view mode),
  `physical_overlay` and `estimate_um_per_px`.
- `sheets.py` — the stack sheets: `stack_pictures` (captioned per section,
  optionally in written-position order over its atlas match),
  `reference_slice_picture`, `stack_sheet` (one contact sheet, shrunk under
  `SHEET_MAX_LONG_EDGE`), `grid`, `beside`, `stacked`, `spacing_plot`.
- `status.py` — the status table, data for the doors rather than a
  picture: `status_rows`, `compact_rows`, `status_text`, `slice_flags`.
- `sizes.py` — the picture sizes: `PICTURE_EDGES` (opening and later long
  edge per `image_resolution`), `AUTO_RESOLUTION`, `MIN_RESOLUTION`,
  `MAX_IMAGES_PER_CALL`, `resolution_level`, `opening_edge`, `picture_edge`.
- `layers.py` — a placement picture's layers and frame record, the
  on-demand coordinate map, and the per-call picture notes (below).
- `jpeg.py` — the doors' one JPEG encoding (below).

## The frame of a picture and its layers (phase 3c)

`CanvasFrame` holds everything that fixes a physical picture's canvas: the
shown section render (before any warp), its micrometres per pixel, the
position, plane and cutting angles, the placement (`params`: the knobs about
`pivot`/`pivot_in_section`, or a 2x3 on the shown render's frame;
`spline`; `warp`), the zoom window and the panel size. `physical_views`
builds the placement with `core.canvas.placement_matrices` (the section
matrix on the section's frame, and the same map on the canvas,
`shift(offset) @ section_matrix @ shift(-offset)`) and rasterises every
atlas border through `core.canvas.line_coverage`; with `panel_frames` it
hands back one `core.canvas.PanelFrame` per picture: the picture size,
the `content_box` below the caption, the canvas `crop_box` and `factor`
(canvas px -> picture px), the `CanvasGeometry`, the `section_matrix`
(section render -> canvas, 3x3) and the lines it drew. `draw_canvas` keeps
them on `Canvas.panels` and notes each picture (below).

`layers.py` turns a `PanelFrame` into the picture's layers, on the
picture's own pixel grid (caption band included, empty there):
`labels_layer` (the atlas id under every pixel, uint32, nearest neighbour
through the mapping the borders are drawn with), `borders_layer` (the drawn
borders' coverage, uint8, `line_coverage` at full strength, so the layer
and the picture's lines cannot disagree), `picture_to_native`,
`section_to_picture` (3x3 on `[row, col, 1]`, the linear placement; a
deformation drawn on top is not in it), `frame_record` (the JSON a saved
picture's `view.json` carries under `frame`: atlas name, version,
orientation, shape and resolution; plane name, position and angles;
picture size and content box; µm per canvas and picture pixel; the canvas
crop, factor, atlas scale and offset, section offset; the section id,
render size and placement; the applied deformation's folder, relative to
the job folder; the zoom; `pixel_to_atlas_um`; `section_to_picture`; the
convention in words) and `picture_layers` (all three).

`pixel_to_atlas_um` is a 3x3 matrix taking a picture pixel `[row, col, 1]`
(pixel centres at integers, row 0 the top of the picture, caption included)
to BrainGlobe atlas micrometres in the atlas's own axis order, voxel `i`'s
centre at `i * resolution` (Nash 2026-10-03). It is exact: the picture shows
one atlas plane, so the map is affine (`oblique.plane_index_affine`, built
from the same basis `plane_index_coordinates` samples, times the
resolution). No coordinate map is stored per picture (three float32
channels per pixel would be the largest file by far): `coordinate_map(view)`
takes a `view.json` (record or path) and returns the `(rows, cols, 3)`
float32 map on demand, NaN in the caption band. The labels layer is what
that map reads in the atlas annotation, up to ties at exact half voxels.

Which pictures a call returned, and of what, is noted while it runs:
`layers.collecting()` (the job's saving hook, `Job.views.shown`, runs every
tool call inside it) and
`layers.note(image, sections=, mode=, frame=, panel=, deformation=,
extra=)`; `note_for` finds a picture's note by identity. `draw_canvas`,
`placement_pictures` (stacked, side_by_side references), `section_picture`,
`atlas_view_picture` and `stack_review` note theirs;
`ops.deformable.pictures` notes `fit_deformable`'s fits and traces, and the
tool door the pictures it labels itself (`preprocess` before/after). With
nothing collecting, a note costs nothing. The job saves the pictures
(`langslice.job.views`, `job/CLAUDE.md`).

`jpeg.py` — `encode_jpeg` (`JPEG_QUALITY` 85): the one encoding every
picture a model receives goes through; `adk.media` re-exports it, the MCP
door and the job's view store call it, so the saved bytes are the sent ones.

## `linear/render.py`

The split of `linear/render.py` into `sections`, `captions`, `canvas`,
`sheets`, `status` and `sizes` was a pure move (phase 3d, goldens
identical). `linear/render.py` stays as a re-export shim of their public
names only because the sibling repo SliceBench imports it
(`slicebench/adapters/langslice_geometry.py`: `PREVIEW_LONG_EDGE`,
`canvas_geometry`, `canvas_um_per_px`, `render_slice`); LangSlice itself
imports the core modules.

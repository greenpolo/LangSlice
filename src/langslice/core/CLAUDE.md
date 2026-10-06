# LangSlice `core/` — the core library

Package guide for `src/langslice/core/`. The repo-level `CLAUDE.md` holds the
project-wide rules. `AGENTS.md` here is a verbatim copy — edit one, mirror to
the other.

The lowest layer. The pictures the tools send and their layers, the section
renders, captions, canvas, sheets and status table, and every other module
whose imports are core-only:

- shared foundations: `space.py`, `oblique.py`, `affine.py`,
  `image_prep.py`, `thin_plate.py` (the thin-plate kernel `abba_warp.py`
  validates its landmark pairs with), and the sub-packages `atlas/`
  (`atlas/CLAUDE.md`), `deformable/` (`deformable/CLAUDE.md`) and
  `nonlinear/` (the image-model border route's prompts, request and line
  extraction, `nonlinear/CLAUDE.md`);
- the linear method's core: `state.py`, `spec.py`, `workspace.py`,
  `appearance.py`, `atlas_fetch.py`, `atlas_grep.py`, `display.py`,
  `opening.py`, `transform.py`, `deformation.py`, `discovery.py`. Their
  entries are in the linear agent environment's guide
  (`src/langslice/agent/CLAUDE.md`, "Files"), which maps the whole method;
- plain tables several layers share: `provider_names.py` and
  `media_keys.py` (the keys a tool's media travels under, read by the tool
  door, the ADK driver, MCP and the OAuth transport);
- the host-row geometry the ABBA worker emits (`doors/api/abba_worker.py`):
  `abba_affine.py` (a normalized affine in ABBA's centred world
  millimetres) and `abba_warp.py` (an applied deformation as the landmark
  pairs of a SECOND ABBA step on top of the affine row:
  `warp_world_landmarks(frame, record, params, size=, pixel_size_um=)`;
  `source` = a section point as the affine step placed it, `target` = where
  the deformation takes it, centred world mm, the frame derived from
  `normalized_to_abba_affine` and `maps.native_points`; a 9x9 to 33x33
  grid, growing until the TPS pull-back is within 5 um at off-grid probes;
  past 33x33 the pairs are sent anyway with the error measured
  (`max_error_mm`, `p99_error_mm`, `within_tolerance`), favouring speed
  over accuracy; the largest grid whose TPS does not fold is sent, and only
  a TPS folding at every grid is refused). `abba_angles.py`: ABBA's
  `ReslicedAtlas` rotations <-> pitch/yaw (`PITCH_TO_ROTATE_X_SIGN`,
  `YAW_TO_ROTATE_Y_SIGN`, both -1; `angles_to_rotate`, `rotate_to_angles`;
  SliceBench reads the constants through the `langslice.integrations.abba_linear`
  shim).

## The layer rule

- A core module takes plain inputs: the `core.workspace.Workspace` (atlas,
  section files, render caches), the `core.state.StackState` and its
  records, ids, numbers, the core's `core.display.DisplayOptions`. It
  returns plain outputs: PIL images (captions burned in: tool images reach a
  model as bare attachments), numpy arrays, dicts and frozen dataclasses.
- It imports other core modules only. Never the job layer (`job/`), the
  operations (`ops/`), a door (`doors/`), the agent driver (`agent/`), a
  host, a provider, and never `google.*`, `litellm` or `openai`.
  import-linter's contracts (`pyproject.toml`, run by
  `tests/test_import_layers.py`) check the imports;
  `tests/test_core_imports.py` loads each module in a fresh interpreter and
  checks what it loads. Provider NAMES are the core's own plain-string
  table, `provider_names.py` (`CANONICAL_PROVIDERS`,
  `canonical_provider`, and `CUSTOM_PROVIDER` `"custom"`: a job whose image
  model a script hands the library itself, accepted by the spec, never
  resolved from a name; `providers.registry` re-exports it): `spec.py`
  validates against it and `nonlinear/` picks the GPT or Gemini prompt
  wording with it. A model call is only ever the `image_call` a door
  passes in; the border routes refuse a call without one.
- No undo, no checkpoint, no look-before-commit gates, no wording for a
  model beyond the captions burned into a picture. Something that writes the
  stack is an operation (`ops/`); something that words a reply is a door.
- A picture is returned, never encoded: the doors package it (the ADK agent
  through `doors.tools.media.packaged`, JPEG message parts; MCP as image blocks).

## Cutting angles are per section

Every atlas plane drawn, fitted or mapped for a section is at that
section's own angles (`SliceState.angles`): the placement pictures
(`placement.py`, `CanvasFrame`), the side-by-side and stacked references,
`view_stack`'s atlas under each section, the opening's atlas tiles, the
fits (`transform.py`), the handoff and every deformable fit and trace
(`handoff.py`), the deformation's `linear_key`, the maps and
`registration.json` (`maps.section_frame`, memoized by the section's
angles) and so the QuickNII/VisuAlign anchorings, and `grep_atlas`'s
in-section check. A picture without a section (`view_atlas`, the
opening's atlas reference) is drawn at `StackState.view_angles` (the shared
angle, or the median of the sections' when they differ). The atlas-plane
helpers (`atlas_fetch`, `display`, `pictures.reference_atlas_picture`)
take `angles=`; None reads the stack's one angle (`state.plane_angles`),
which raises `MixedAngles` on a stack whose sections differ, so a call
that should pass a section's own fails loudly instead of drawing the
wrong plane. `captions.angles_label` words the angles in a caption (empty
for the flat plane).

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
  `BAD_PIVOT`), `Staged` and `staged_views`. The transform tools' pictures:
  `fit_picture` (`fit_affine`: a fit drawn from its `FitFrame`,
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
- `captions.py` — `caption` (a COPY with the text in a band below the
  picture, so a picture pixel is the content's own, the coordinates a
  `view.zoom` is given in; never caption an image a fit measures),
  `wrap_caption`, the fonts, `scale_bar_px` and the 1 mm bar `_draw_scale_bar`.
- Zoom: every picture tool takes `view.zoom` as `[x0, y0, x1, y1]` pixels of
  the picture the same call returns unzoomed (top-left origin, the
  convention of Claude's and OpenAI's computer-use tools; a later zoom is
  again in the unzoomed picture's pixels). Each renderer converts it with
  `display.zoom_fractions` against its own unzoomed content size
  (`framed_section`, `framed_atlas`, `placement.draw_canvas`, which draws
  the unzoomed canvas first for its size, `deformation.picture` and
  `trace_picture` through `deformation.unzoomed_size`); below that the
  renderers crop by fractions (`canvas.zoom_box`). In a picture of panels
  side by side (`channels`), the box is read on the first panel.
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
  picture: `status_rows`, `compact_rows` (a model's: null and empty fields
  left out), `uniform_rows` / `with_uniform_rows` (a script's: every
  `ROW_FIELDS` field on every row, null or its `ROW_DEFAULTS` value where
  absent, for `ROW_KEYS` `rows` and `changed`; the agent CLI and the
  library), `status_text`, `slice_flags`.
- `sizes.py` — the picture sizes: `PICTURE_EDGES` (opening and later long
  edge per `image_resolution`), `AUTO_RESOLUTION`, `MIN_RESOLUTION`,
  `MAX_IMAGES_PER_CALL`, `resolution_level`, `opening_edge`, `picture_edge`.
- `layers.py` — a placement picture's layers and frame record (and a
  `fit_deformable` picture's), the on-demand coordinate map, and the
  per-call picture notes (below).
- `maps.py` — section pixels to atlas
  micrometres for the job folder's public files (`job/formats.py`,
  `docs/file_formats.md`). `SectionFrame` / `section_frame(state,
  workspace, record)`: one placed section's linear map from its image FILE
  (`[x, y]` pixel centres; rotation and flip are part of the map,
  `orientation_matrix`, PIL's counter-clockwise quarter turns first, then
  the flip) onto the `PREVIEW_LONG_EDGE` frame the six numbers are
  normalized against, the native plane placed there by
  `handoff.linear_placement_matrix`, and `plane_index_affine` times the
  resolution to micrometres: `pixel_to_atlas_um(size)` for the file or any
  resize of it. The working copy's size comes from the file header
  (`image_prep.working_size`) and the render size from
  `image_prep.prepared_size`, so `registration.json` is rewritten on every
  write without decoding images; memoized on `Workspace.frame_cache` by
  everything it depends on (one frame per section, least recently used out
  past `FRAME_CACHE_SECTIONS`). A section with no pixel size (neither the file
  nor the host gives one) is mapped at the scale its placement pictures draw
  it at, `core.transform.calibrate` on its working frame (estimated from the
  tissue width at its CURRENT position, never the scale stored with a
  transform written elsewhere; the one decode), and `scale_problem` /
  `SCALE_UNKNOWN` says so in `registration.json`'s `problem`. No transform: the identity, as the pictures
  draw it (`IDENTITY_PARAMS`, `stored_params`); `placement_problem` says
  why a section has no map. `native_points(frame, warp, x, y)`: native
  plane `(x, y)` of file points, through an applied deformation exactly as
  its record composes it (carried onto the fit grid, displaced by its
  field, bilinear with replicated edges, then its placement undone).
  `section_maps(workspace, frame, warp, full_resolution=)`: `SectionMaps`
  on the working copy's grid (or the file's): coordinates (NaN off the
  section's footprint or off the atlas volume: `section_footprint`, the
  deformable fit's foreground rule closed over `FOOTPRINT_CLOSING_MM` and
  hole-filled, so dark tissue and tears inside the outline keep their
  coordinates; the raw rule is kept as `tissue` for `tissue.png`), atlas ids (`core.deformable.geometry.sample_native`, nearest, as the
  pictures' labels layer), and the residual `(drow, dcol)` such that
  `coords = pixel_to_atlas_um @ [p + d, 1]`; computed in `BLOCK_ROWS`
  blocks. `residual_markers(frame, warp)`: VisuAlign `[x, y, nx, ny]`
  markers on a grid of 1/36 of the long edge.
  `render_sizes(workspace, id)` (`RenderSizes`: the file, working copy and
  unturned render sizes and the file-to-render scale, header-only when it
  can) is the one place those sizes are found; `_section_frame` and the
  importer both use it.
- `import_geometry.py` — the exact inverse of `maps.section_frame`'s linear
  map, for a registration made elsewhere (`job/imports.py` reads the
  files): `placement_from_pixel_map(pixel_to_atlas_um, atlas=, plane=,
  sizes=, file_um_per_px=, orientation=)` returns `RecoveredPlacement`
  (position, pitch and yaw in `core.oblique`'s convention, quarter turn,
  flip, the six numbers on the oriented render (`render_size`) at that pixel
  size, `out_of_plane_um`, `in_plane_error_px`; `physical()` the knobs a
  stored transform carries, `core.transform.physical_params` about the
  render's centre). Nine numbers each way (normal,
  position, full in-plane affine), so anisotropic scale and shear survive;
  flip and quarter turn are a choice (unmirrored, least turn) unless given;
  a flat plane is drawn at the nearest voxel (reported); angles below
  `ANGLE_EPSILON_DEG` read as 0; a plane more than `MAX_TILT_DEG` from the
  job's plane is refused. `plane_angles`, `plane_position_mm`,
  `implied_pixel_size_um`.
- `jpeg.py` — the doors' one JPEG encoding (below).
- `handoff.py` — a written linear placement as the nonlinear work
  starts from it: `prepare_linear_registration(state, workspace, id,
  long_edge=, transform=)` (the oriented, unframed section render and the
  3x3 from native atlas-plane pixel centres onto it, calibration checked,
  `LinearRegistrationInput`; the trace's canvas and every deformable fit's
  grid; its matrix is `linear_placement_matrix`, the one path from the six
  stored numbers to the atlas, which `core.maps` uses too),
  `correction_fingerprint` (everything an image correction's inputs
  depend on; the trace's call key and a traced fit's staleness test
  read it) and `digest`. A section without a written
  transform is refused by `missing_transform_message(spec)`: with Linear
  (`transform`) off, `NO_TRANSFORM_LINEAR_OFF`, which tells the caller to
  supply the transforms (`inputs.transforms`, `--transforms`) or switch
  Linear on and run `fit_affine` first (the maps still read a missing
  transform as the identity). No provider import:
  `core/nonlinear/registration_handoff.py` re-exports the first for SliceBench.

## The frame of a picture and its layers

`CanvasFrame` holds everything that fixes a physical picture's canvas: the
shown section render (before any warp), its micrometres per pixel, the
position, plane and cutting angles, the placement (`params`: the knobs about
`pivot`/`pivot_in_section`, or a 2x3 on the shown render's frame;
`warp`), the zoom window (fractions of the canvas) and the panel size. `physical_views`
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
centre at `i * resolution`. It is exact: the picture shows
one atlas plane, so the map is affine (`oblique.plane_index_affine`, built
from the same basis `plane_index_coordinates` samples, times the
resolution). No coordinate map is stored per picture (three float32
channels per pixel would be the largest file by far): `coordinate_map(view)`
takes a `view.json` (record or path) and returns the `(rows, cols, 3)`
float32 map on demand, NaN in the caption band. The labels layer is what
that map reads in the atlas annotation, up to ties at exact half voxels.

A `fit_deformable` picture gets the same layers on its own
grid through `warp_layers(atlas, note, size)`: labels through the record's
composed map at every content pixel (no tissue rule, as a placement
picture's), borders as
`core.deformable.render.drawn_border_coverage` (exactly the lines drawn), a
frame whose `pixel_to_atlas_um` is the record's linear placement on the
picture, and, when the field was drawn, a residual layer `residual.tif`
(`RESIDUAL_LAYER`, `(2, rows, cols)` float32, `(drow, dcol)` picture
pixels) the frame names under `residual`; `coordinate_map` applies it
(beside the `view.json`, or `folder=` for a record).

Which pictures a call returned, and of what, is noted while it runs:
`layers.collecting()` (the job's saving hook, `Job.views.shown`, runs every
tool call inside it) and
`layers.note(image, sections=, mode=, frame=, panel=, deformation=,
extra=)`; `note_for` finds a picture's note by identity. `draw_canvas`,
`placement_pictures` (stacked, side_by_side references), `section_picture`,
`atlas_view_picture` and `stack_review` note theirs;
`ops.deformable.pictures` notes `fit_deformable`'s fits (through
`core.deformation.picture(note=...)`, with a `WarpNote`: the record
resampled onto the picture, the caption band, whether the field is drawn,
the border style) and traces, and the
tool door the pictures it labels itself (`preprocess` before/after). With
nothing collecting, a note costs nothing. The job saves the pictures
(`langslice.job.views`, `job/CLAUDE.md`).

`jpeg.py` — `encode_jpeg` (`JPEG_QUALITY` 85): the one encoding every
picture a model receives goes through; `doors.tools.media` re-exports it, the MCP
door and the job's view store call it, so the saved bytes are the sent ones.

## `linear/render.py`

`linear/render.py` is a re-export shim of four names, `PREVIEW_LONG_EDGE`,
`canvas_geometry`, `canvas_um_per_px` and `render_slice`, because the sibling
repo SliceBench imports them; LangSlice itself imports the core modules.

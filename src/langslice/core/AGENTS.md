# LangSlice `core/` — the picture builders

Package guide for `src/langslice/core/`. The repo-level `CLAUDE.md` holds the
project-wide rules. `AGENTS.md` here is a verbatim copy — edit one, mirror to
the other.

The core library's new home (layered refactor, phase 3b, 2026-10-03). For
now it holds the pictures the tools send; the existing core modules
(`space`, `affine`, `oblique`, `image_prep`, `atlas/`, `deformable/`,
`linear/render`, `linear/display`, `linear/workspace`, `linear/transform`,
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
  first, through `linear.render.physical_views`, and returns a `Canvas`:
  the panels and the `CanvasFrame` they were drawn in. `stored_placement`
  (the six stored numbers as a drawable matrix, identity without),
  `current_warp` (the applied record from a `deformation.RecordStore`, None
  when stale or unreadable), `placement_pictures` (one section-position pair
  in a placement mode, `PLACEMENT_MODES`: the canvas under the complete
  registration, `stacked`, or the two separate `side_by_side` references the
  door maps by index; returns `Placed`: images, row facts, the canvas),
  and the interactive transform's `stage` (the working frame, calibration,
  canvas and resolved pivot; `StageFailure` `ATLAS_RENDER_FAILED` /
  `BAD_PIVOT`), `Staged` and `staged_views`.

## The frame of a picture (for saving renders, phase 3c)

`CanvasFrame` holds everything that fixes a physical picture's canvas: the
shown section render (before any warp), its micrometres per pixel, the
position, plane and cutting angles, the placement (`params`: the knobs about
`pivot`/`pivot_in_section`, or a 2x3 on the shown render's frame;
`spline`; `warp`), the zoom window and the panel size. The canvas is
`linear.render.canvas_geometry(section.size, um_per_px, atlas, position_mm,
plane, pitch_deg, yaw_deg)`; `physical_views` places the section on it
(`section_offset`) and maps it by `matrix = shift(offset) @ section_matrix @
shift(-offset)`. The picture's layers in that one frame are then: the section
as placed (that warp of the canvas, then `deformable.warp_section_image`
when `warp` is set), the atlas labels (`geometry.annotation` through
`atlas_scale`/`atlas_offset`), the border mask (the same lines
`physical_views` draws) and the pixel-to-atlas map (canvas pixel -> plane
pixel through the same scale and offset, then the plane's 3D coordinates at
the position and angles). Phase 3c adds one function here that takes a
`CanvasFrame` and returns those layers; the doors already receive the frame
(`Canvas.frame`, `Placed.canvas`). Factor the matrix block of
`physical_views` into a helper both use, so the picture and its layers can
never disagree.

## Later

`linear/render.py` (~1,600 lines) mixes five jobs and splits by
responsibility when it moves here: the section renders and their cache
(`render_cache_key`, `render_slice`, `_look_image`, `fine_detail`,
`canvas_um_per_px`, `shown_section`, `rescale_section_matrix`); captions,
fonts and the scale bar (`caption`, `wrap_caption`, `_caption_font`,
`_font`, `scale_bar_px`, `_draw_scale_bar`); the physical canvas
(`CanvasGeometry`, `canvas_geometry`, `VIEW_MODES`, `OUTLINE_LAYERS`, the
drawing helpers, `regions_left`, `region_polys`, `pivot_on_canvas`,
`physical_views`, `physical_overlay`, `estimate_um_per_px`); the stack
sheets (`stack_pictures`, `reference_slice_picture`, `stack_sheet`, `grid`,
`beside`, `stacked`, `spacing_plot`); and the status table (`status_rows`,
`compact_rows`, `status_text`, `slice_flags`), which is data for the doors
rather than a picture. The picture sizes (`PICTURE_EDGES`,
`resolution_level`, `opening_edge`, `picture_edge`) go with the sheets or
their own small module.

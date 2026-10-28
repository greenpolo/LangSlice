# LangSlice `core/` — the core library

Package guide for `src/langslice/core/`. The repo-level `CLAUDE.md` holds the
project-wide rules. `AGENTS.md` here is a verbatim copy — edit one, mirror to
the other.

The lowest layer. The pictures the tools send and their layers, the section
renders, captions, canvas and status table, and every other module
whose imports are core-only:

- shared foundations: `space.py`, `oblique.py`, `affine.py`,
  `image_prep.py`, `thin_plate.py` (the thin-plate kernel `abba_warp.py`
  validates its landmark pairs with), and the sub-packages `atlas/`
  (`atlas/CLAUDE.md`), `deformable/` (`deformable/CLAUDE.md`) and
  `nonlinear/` (the image-model border route's prompts, request and line
  extraction, `nonlinear/CLAUDE.md`);
- `toolset.py` and `needs.py`: what a job can switch off (and, through
  `ToolDef.writes`, what its frozen properties switch off: `off_names`). `toolset.py`
  lists every tool (`TOOLS`: `ToolDef` name, kind, task group, `when`,
  scripting / image-model / hidden, `options`, `essential`, `needs`;
  `ops/registry.py` binds each to its operation), the `look` modes, the
  names a `JobSpec.disabled` takes (`valid_names`: a tool, or `tool:option`;
  `is_off`, `options_on`) and `refusal(spec)`, the one-line reason a list
  cannot work (an unknown or essential name, every `look` mode off, a task
  with every tool that writes its answer off, a tool without the pictures it
  needs), which the spec raises
  as a `ValueError` when it is made. `needs.py`: `render(text, off)`, text
  written with what it needs (`[[?NAMES text||other]]`, `[[#NAMES=item||...]]`
  lists), dropped or swapped for the disabled list; with nothing off a text
  renders to itself without its marks, byte for byte. Neither imports
  anything above the core. `toolset.tools_for(spec)` is the per-spec view
  (`TOOLS` plus the tools of `spec.plugins`; `TOOLS` itself without plugins),
  which `valid_names(spec)`, `check_names`, `refusal(spec)`, `off_names`,
  `catalog(spec)` read, so a job's `disabled` list can name a plugin tool;
- `tool_labels.py`: a short label for people per tool and `tool:option`
  (`TOOL_LABELS`, `label`), per task (`TASK_LABELS`, `TASK_ABOUT`) and per
  tool group (`GROUP_LABELS`, with the group's task), for the LangSlice
  window's prompt generator (`hosts/gui/generator.py`, its only importer).
  Never model-facing; `tests/test_gui_generator.py` checks that every tool
  and option has one and no other name does, and that none appears in the
  pinned model-facing text. A new tool or option is a row here too;
- `plugins.py` and `plugin_layers.py`: tool plugins (`docs/developers/plugins.md`), the
  core's part. `plugins.py`: the types (`ToolSpec`, `ToolResult`, `Layer`,
  `ToolError`, `PluginError`, `PLUGIN_API_VERSION`), `check_tool` (the
  one-line reasons a tool declaration is refused), `tool_def` (a plugin tool
  as a `ToolDef`: `when` is its task being on and its `atlases` / `planes` fitting
  the job; `not_offered` names the tools they exclude, which `toolset.off_names`
  counts as switched off so no text names them) and the registry of loaded
  plugins (`LoadedPlugin`, `loaded`, `of_spec`, `tool_defs`, `tool_specs`;
  `SCRIPT_CALLER` `"script"`, what a script's `job.context` writes are recorded
  as, a name no plugin tool takes;
  `set_loader` is how `doors/plugins` registers the code that imports a
  plugin: the core imports nothing by name). `JobSpec.plugins` /
  `plugin_config` are validated by loading (`toolset.tools_for`), so a bad
  plugin is refused when the spec is made. `plugin_layers.py`: the layer
  files per kind (`save_layer`, `load_layer`), and `draw_layers`, which draws
  the layers a `LookRequest.layers` names on a section or overlay picture over
  the landmark panels' geometry (outlines and halo marks, never fills;
  `section_px` through the section panel and the applied warp, `atlas_um`
  through the overlay's atlas panel or the section's linear placement). A
  section's `layers` (state: references only, absent from the serialized
  state while empty, an annotation field of the look snapshot) are read from
  `<job folder>/plugins/` through the workspace's `job_folder`;
- `frozen.py`: what the agent may not change (`JobSpec.frozen`). The five
  properties a section has (`PROPERTIES`: `position`, `angles`,
  `orientation` (flip and quarter turn together), `affine`, `deformation`),
  the `"*"` key for every section, `clean` (validates and canonicalizes a
  freeze), `for_section` / `everywhere` / `sections_of` / `groups`,
  `TASK_PROPERTIES` (what each task writes), `derived` (what a task that is
  off freezes on every section), the model-facing `statement` (sections sharing a freeze
  are one clause: every section, their filenames, or their count) and
  `message` (a refusal's sentence), and the `FROZEN` code. A frozen property
  keeps the value `inputs` supplies, else its "none": the flat plane, no flip
  and no turn, the host's identity transform, no deformation; a position has
  none. `JobSpec.has(task)` is False for a task whose properties are all
  frozen on every section, `JobSpec.frozen_for(id)` includes what a task that
  is off supplies, and `JobSpec.switched_off` is `disabled` plus the tools
  that write (`ToolDef.writes`) only properties frozen on every section and
  the mirror under a frozen orientation (`toolset.off_names`): every door and
  text renders by it;
- the linear method's core: `state.py`, `spec.py`, `workspace.py`,
  `section_images.py`, `atlas_fetch.py`, `atlas_grep.py`, `display.py`,
  `opening.py`, `transform.py`, `deformation.py`, `discovery.py`. Their
  entries are in the linear agent environment's guide
  (`src/langslice/agent/CLAUDE.md`, "Files"), which maps the whole method;
- atlas planes are cached: `oblique.py` spline-filters a
  volume once (on threads) for every cubic plane and keeps the annotation as
  dense integer indices, both in memory and on disk
  (`atlas/disk_cache.py`), samples a large plane in row bands on threads and
  keeps recent annotation planes;
  `atlas_fetch.atlas_section` keeps recent template planes on
  `Workspace.plane_cache`; `opening.opening_items` draws the columns after
  the first on threads. Each gives the same pixels as the serial path
  (`tests/test_oblique_prefilter_cache.py`); `tests/test_opening_startup_budget.py`
  holds a 100-section angled opening on an Allen-sized volume within a
  time budget;
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
  a TPS folding at every grid is refused).

Protocol metadata: `PositionSpec.thickness_um` and `interval_um`, and the
state's millimetre equivalents, are optional (`None` when not supplied).
Known values are positive integers in micrometres; strict spacing requires
an interval. Atlas reference sampling uses atlas extent when none is given.

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
- No undo, no checkpoint, no wording for a
  model beyond the captions burned into a picture. Something that writes the
  stack is an operation (`ops/`); something that words a reply is a door.
- A picture is returned, never encoded: the doors package it (the ADK agent
  through `doors.tools.media.packaged`, JPEG message parts; MCP as image blocks).

## Cutting angles are per section

Every atlas plane drawn, fitted or mapped for a section is at that
section's own angles (`SliceState.angles`): the placement pictures
(`placement.py`, `CanvasFrame`), the opening's atlas tiles, the fits
(`transform.py`), the handoff and every deformable fit and trace
(`handoff.py`), the deformation's `linear_key`, the maps and
`registration.json` (`maps.section_frame`, memoized by the section's
angles) and so the QuickNII/VisuAlign anchorings, and `grep_atlas`'s
in-section check. A picture without a section (`look`'s `atlas` and the
positioning picture's atlas thumbnails, `grep_atlas_view`, the opening's
atlas reference) is drawn at `StackState.view_angles` (the shared angle, or
the median of the sections' when they differ), and its caption says which:
`StackState.drawn_at_median(angles)`, `captions.view_angles_label` ("at the
median of the sections' cutting angles, pitch .. yaw .."). The atlas-plane
helpers (`atlas_fetch`, `display`) take `angles=`; None reads the stack's
one angle (`state.plane_angles`), which raises `MixedAngles` on a stack
whose sections differ, so a call that should pass a section's own fails
loudly instead of drawing the wrong plane. `captions.angles_label` words
the angles in a caption (empty for the flat plane).

## Files

- `placement.py` — a section on its physical canvas at a placement.
  `draw_canvas` is the one renderer of every placement picture (`look`'s
  `overlay` and so every change tool's picture of what it wrote, and
  `mark_damage`'s): it draws from the render the options ask for at the
  call's size, the matrix and pivot carried onto it, an applied deformation
  resampling it first, through `core.canvas.physical_views`, and returns a
  `Canvas`: the panels and the `CanvasFrame` they were drawn in.
  `stored_placement` (the six stored numbers as a drawable matrix, identity
  without), `current_warp` (the applied record from a
  `deformation.RecordStore`, None when stale or unreadable),
  `placement_pictures` (one section-position pair on the canvas under the
  complete registration, in the options' mode; returns `Placed`: images,
  row facts, the canvas), and the interactive transform's `stage` (the
  working frame, calibration, canvas and resolved pivot; `StageFailure`
  `ATLAS_RENDER_FAILED` / `BAD_PIVOT`) and `Staged`.
- `sections.py` — the section renders and their cache: `render_slice`
  (a look drawn unoriented at the render size, then ROTATE, then FLIP;
  `_look_image`): None is the section's `preprocessed` image, `{"image": name}`
  any section image alone in gray (a raw plane, or a made image's file, the
  opening `preprocessed` included, resized to the render), `{"overlay": [raw
  names]}` raw channels together through `channels.composite`, `{"file": True}`
  the working copy as read; `drawn_look` resolves a look on a record (a made
  image carries its file's digest), `look_token` / `render_cache_key` (the
  resolved look), `canvas_um_per_px`, `shown_section` (the larger
  render a picture is drawn from), `rescale_section_matrix`,
  `PREVIEW_LONG_EDGE` (512, the working frame every fit is computed on).
  Every look is drawn at the same size, so no image moves geometry. Cached on
  the workspace (`render_cache`, `render_scale`): read, never mutate.
- `section_images.py` — section images (`docs/reference/tools.md`,
  "`make_image`"): one named grayscale picture of a section that `look`
  draws and every fit reads. **Raw**: a plane of the file
  (`Workspace.section_channels`, `raw_plane` at the working copy's size),
  never edited. **Made**: `SliceState.images[name]` = `{"source": {"recipe":
  ...} | {"script": note} | {"opening": "adaptive" | "host" | "plain"},
  "file", "digest", "made", "weights_used"?}`, the file
  `sections/<stem>/images/<name>.png` (8-bit gray, the working copy's size,
  unoriented; `write_file` / `write_image`, never overwritten: other pixels
  under a name get `<name>.2.png`, the same pixels reuse the file;
  `_Layout.images_dir`). `validate_recipe` (weights `"auto"` or `{image:
  weight}`, `clahe_clip`, `clahe_tiles`, `n4`, `denoise`, `contrast`,
  `gamma`, `invert`; `OPENING_RECIPE` its defaults), `make` (a recipe on the
  working copy: `image_prep.custom_appearance` over the sources, then the tone
  steps; sources are raw channels or made images, so chains compose),
  `recipe_words` / `compact_recipe`, `image_names` (raw, `preprocessed`, then
  the made ones), `kind_of`, `plane_of` / `image_plane` (a raw plane or a made
  file, cached on `Workspace.image_cache`, least recently used out past
  `IMAGE_CACHE_SIZE`; `preprocessed` without an entry is
  the opening image made in memory, for a workspace with no job), `read_image` -> `ImageView`
  (`payload()`: `{name, kind, file, digest}`, a fit's `inputs.section_image`),
  `identity` (what a fit's cache key holds), `images_digest` (`{name:
  digest}`, part of `section_masks.registration_parts` and so of
  `ops.inputs.section_inputs`). The opening image: `opening_source(spec)`,
  `opening_words`, `opening_render(ws, id)` (the automatic path,
  `image_prep.adaptive_preprocess` on the working copy, in gray; a host's blend
  is the working copy itself; `preprocess="none"` the copy in gray), written
  once as the file every reader reads, like any made image; `opening_entry`, `ensure_opening` (every section without one,
  on threads, outside undo: `Job.open` and `Job.load`). `summary(state, spec)`:
  `status`'s `section_images`. `Migration(ws, layout)`: a format-3 state's fields as
  format 4 (each `appearance` recipe, a section's override honoured, made into
  a file and an entry; `preprocessed` without a recipe the opening image;
  display properties dropped; a mask's `channel` named `section_image`), one
  memo across the checkpoint and its history steps (`seed` takes a format-4
  state's entries). `Look` is the render's look type.
- `channels.py` — raw channels drawn together: `composite` (each channel
  stretched between its automatic limits, `auto_limits`, per section: on a
  dark background black at its background median and full colour at its
  `TISSUE_WHITE_PERCENTILE` (95) inside the tissue, Otsu on the mean of the
  section's channels; on a light background its 1st and 99.5th percentiles,
  `OVERLAY_STRETCH`; then added in its colour), `channel_colors` (a channel
  named after a colour keeps it), `colour_words` for captions,
  `intensity_ranges` (cached on `Workspace.intensity_cache`, from
  `image_prep.working_intensity_ranges`: the linear stretch each page's
  working plane was read with; `native.py` reads a zoom's file pixels through
  it). Nothing a fit reads comes from here.
- `captions.py` — `caption` (a COPY with the text in a band below the
  picture, so a picture pixel is the content's own, the coordinates a zoom
  is given in; `min_width` lays the band out at least that wide; never
  caption an image a fit measures), `wrap_caption`, the
  fonts, `scale_bar_px` and the 1 mm bar `_draw_scale_bar`; `angles_label`
  and `view_angles_label` (the angles of a picture without a section, said
  to be the median when the sections' differ).
- Zoom: a renderer takes its options' `zoom` as `[x0, y0, x1, y1]` pixels of
  the picture unzoomed (top-left origin, the convention of Claude's and
  OpenAI's computer-use tools). Each renderer converts it with
  `display.zoom_fractions` against its own unzoomed content size
  (`framed_section`, `framed_atlas`, `placement.draw_canvas`, which draws
  the unzoomed canvas first for its size, `deformation.picture` and
  `trace_picture` through `deformation.unzoomed_size`); below that the
  renderers crop by fractions (`canvas.zoom_box`). A zoomed placement
  picture (`canvas.physical_views` with a crop: `look`'s overlay zoomed, a
  change tool's picture zoomed to its `restrict_to` regions) is drawn at the
  call's long edge whatever the crop's size, enlarged past the canvas's
  pixels when the crop has fewer (its caption says `canvas pixels enlarged
  xN`), and its caption band is laid out at least that wide
  (`captions.caption(min_width=)`; the picture at the top left, its pixels
  the content's own). A zoomed `section` or `overlay` picture reads its
  section from the image file (`native.py`, below): `look._section_picture`
  through `native.framed_window`, `placement.draw_canvas` by handing
  `physical_views` a `detail` (`canvas.SectionDetail`) whose pixels replace
  the render's inside the section, the atlas image blended and the lines
  drawn over them as usual; the captions say `from the image file` (and
  `file pixels enlarged xN` where the file has fewer pixels than the
  picture); a file that cannot be read leaves the working copy's pixels (a
  warning logged). The `zoom` tool's redraw is `zoom.py`.
- `canvas.py` — the physical canvas: `CanvasGeometry` / `canvas_geometry`
  (the atlas section at true scale on the section's frame, anatomy centred,
  canvas grown to hold it plus `WORKING_MARGIN`), `VIEW_MODES` (`overlay`,
  `section`, `template`), the cased atlas borders (`resolve_border_color`
  picks a black or white core from the lightness under the lines,
  `CASING_PX` of the other colour on each side, `border_words` for the
  caption; `composite_cased`), `line_coverage` (the one border rasteriser), `regions_left` / `region_polys`, `zoom_box`,
  `pivot_on_canvas`, `placement_matrices`, `PanelFrame`, `physical_views`
  (the alignment picture in a view mode).
- `scale.py` — one micrometres per pixel for a section and its atlas drawn
  as separate panels (the positioning picture, the opening strips): the
  section at the scale its overlay draws it
  (`section_um_per_px`: `transform.calibrate` times `stored_scale`, the
  stored transform's `sqrt(|ad - bc|)`), the atlas plane at its voxels, the
  larger framed panel at the picture's long edge and never finer than the
  section's working copy (`pair_um_per_px`, `finest_um_per_px`).
  `pair_scale` sizes a pair at one position. `section_at` / `atlas_at` (and
  `display.framed_atlas(um_per_px=)`) draw at exactly that scale.
- `status.py` — the status table, data for the doors rather than a
  picture, a section named by its filename alone (no index):
  `status_rows` (each row's `transform` is `transform_entry`: the stored
  transform's `kind`, its knobs as `interactive_transform` takes them and
  `mirrored`, null without one; a section holding a deformation carries
  `deformation`, `deformation_entry`: `method` (the last step's,
  `step_method`), `steps` (every step's method) and `settings`
  (`step_settings`: a plugin's or script's `settings`, else the native step's
  own by the tools' names); a section with plugin layers carries `layers`: name, kind,
  frame, tool, which `uniform_rows` takes as an extra field; a section with
  masks carries `masks`: name, kind, stale, earlier_registration), `compact_rows` (a model's:
  null and empty fields left out), `uniform_rows` / `with_uniform_rows` (a script's: every
  `ROW_FIELDS` field on every row, null or its `ROW_DEFAULTS` value where
  absent, for `ROW_KEYS` `rows` and `changed`, and only for a list of
  status rows (`frozen`, `damage_by_user` and `registration`, the number of
  the registration the section holds, are set by `ops.views.status`): rows
  with a field outside `ROW_FIELDS`, such as
  `grep_atlas`'s region rows under the same `rows` key, are left as they
  are; the agent CLI and the library), `status_text`.
- `section_masks.py` — a section's masks, saved per section
  (`SliceState.masks`: kind `em` | `fit`, section image, resolved atlas image,
  params, the registration digest it was made at, file, left-out fraction;
  absent from the serialized state while empty). A mask is a float16 weight
  image `(H, W)` in [0, 1] on the fit grid, in
  `sections/<stem>/masks/<name>.npz` (`weights`, `grid` `[W, H]`, `made`),
  written by `write_mask` (never deleted: a mask made again under a name gets
  `<name>.2.npz`, so an undone state still finds its file) and read by
  `read_mask` -> `MaskView` (float32 weights, `stale` when
  `grid_digest` — the section's orientation — differs; `earlier_registration`
  when `registration_digest` — placement, made images' digests, damage marks, applied
  deformation, the content of `ops.inputs.section_inputs(...,
  deformation=True)` — differs: information, never a refusal). `em_sorting` makes the three-class EM sorting
  at the section's current registration (atlas image through the applied
  deformation; footprint = the placed outline widened by `MASK_MARGIN_MM`;
  `deformable.masks.em_weights` on the 20 µm grid, upsampled); `fit_weights`
  copies the applied record's saved sorting (`DeformableRecord.weights`).
  A mask is section-side; the damage marks are atlas-side, always applied, and
  never mask the section. A fit takes a mask as
  `prepare_fit(section_weights=)`, thresholded at 0.5.
- `engine_inputs.py` — the two blocks every fit reply and record provenance
  carry: `InputsBlock` (the section image, `{name, kind, file, digest}`, atlas image as
  asked and resolved, mask, the atlas-side damage entry, `restrict_to`, the
  engine's brightness scaling in words) and `EstimatedBlock` (the engine's own
  saved sorting and the fraction it left out). They name what was used, never
  quality.
- `engine_output.py` — `engine_output(path, label)`: file descriptor 2 pointed
  at a log file (the job's `logs/engine.log`, `Job.engine_log`, passed by the
  operations) while an engine runs, so what it and the programs it starts
  print (the GPU compiler's warnings, torch's allocator, Elastix) never reaches
  a reply; process-wide, nested and overlapping runs share the first log,
  None leaves stderr as it is. `failure_message(text, path, label)`: what a
  reply says of an engine that failed, the error's first line (at most
  `FAILURE_CHARS`), a GPU out of memory said in words, and that the log holds
  the whole error, which it appends there.
- `atlas/images.py` — `atlas_images` ends with the alias `nissl`;
  `default_image(atlas)` is what it stands for (the borrowed Nissl; the
  augmented atlas's template; an additional reference whose name contains
  `nissl`; else `template`); `resolve_image` checks a name and resolves the
  alias, and `atlas_volume` resolves it too.
- `damage.py` — damage by atlas region (`SliceState.damaged_regions`;
  `SliceState.damaged` is read-only: the section has marked regions;
  `damage_note` is words only, a host's `inputs.damaged` note first). `exclusions(record, restrict_to, exclude)`: the
  `(restrict_to, exclude)` every fit and the image model's trace run with,
  the marked regions added to *exclude* except an entry *restrict_to* names
  exactly; `normalized_entries`, `entry_key`. `damage_picture(ws, state,
  record, store=)`: the marked regions hatched (`HATCH_COLOR`, the
  deformable pictures' excluded-region ink) on the section under its
  current registration (`placement.draw_canvas`, mode `overlay`, the warp
  from *store*) and on the atlas template (mode `template`), on one canvas
  and crop, side by side (`beside`), their outlines strong over the
  faint borders; `DamagePictureError` `NO_POSITION` / `NO_REGIONS`.
- `look.py` — `look`'s four pictures through the renderers above:
  `look(ws, state, LookRequest(mode, sections=, positions_mm=, images=,
  atlas_layers=, atlas_opacity=, warp=, long_edge=, zoom=, part=,
  highlight=, region_only=, layers=, masks=, depth=, registrations=), store=)`
  returns one `LookPicture` per section (`section`:
  `display.framed_section`; `overlay`: `placement.placement_pictures` mode
  `overlay`, the applied deformation unless `warp` is `none`), per position
  (`atlas`: `display.framed_atlas` at the stack's view angles), per named
  section in `atlas` mode (at that section's own position and angles;
  `positions_mm` other than theirs `BAD_ARGS`) or per run
  of positions (`positioning`: one picture for up to `PER_PICTURE`
  sections, several for a longer stack; `part` (0-based) draws only that
  one, `UNKNOWN_PART` past the last); a picture without a section says
  when its angles are the median of differing sections'
  (`atlas_caption(median=)`). Defaults never read the job's state: every section,
  the `preprocessed` image (`images` names one section image alone, drawn in
  gray, or up to `MAX_OVERLAY_CHANNELS` raw channels together in their
  colours; `_images_for`), `ATLAS_LAYER_DEFAULTS` per
  mode. `masks=` names masks of the section outlined where their weight is
  0.5 on `section` and `overlay` pictures (`_draw_masks`: the weights from
  `section_masks.read_mask` drawn as a scalar layer by `plugin_layers`; a
  mask the section lacks is a note). Each picture has an index caption (position, angles, um/px, the
  section image drawn and what it is, `image_words`) and a JSON recipe `{"renderer": "look",
  "args", "state", "base", "shown", "um_per_px"}`: the request (a
  positioning picture's: the whole call's sections and positions and its
  `part`, so a zoom redraws that picture at the call's scale), the
  `snapshot` of the stack facts it was drawn from (`SECTION_FIELDS`, each
  section's made images included, the view angles for atlas
  and positioning), the unzoomed and shown content sizes. Both go on the
  picture's note (`layers.annotate` keeps an overlay's frame for its
  layers). `registrations` (`(section, number, held)` each) names the
  registration the state given draws a section under (the ops layer puts it
  on a copy of the state): the caption says `registration 3 (not held)`, and
  `changed_since(besides=)` leaves those sections' `REGISTRATION_FIELDS` out,
  so a zoom of the picture is not stale for drawing a registration the
  section does not hold. `redraw(recipe, window, ws, state)` draws a recipe at a window,
  from `restored(state, snapshot)` and `stale` when `changed_since`; a recipe
  another version of LangSlice saved is `UNREADABLE_RECIPE` (the zoom crops
  the saved picture). `LookError` codes: `UNKNOWN_MODE`, `UNKNOWN_SECTION`, `UNKNOWN_IMAGE`,
  `MIXED_IMAGES`, `TOO_MANY_CHANNELS`, `UNKNOWN_LAYER`, `NO_POSITIONS`,
  `NO_POSITION`, `BAD_WARP`, `UNKNOWN_PART`, `UNKNOWN_REGIONS`. An overlay's
  `highlight` regions (a restricted fit's `restrict_to`) are drawn
  `atlas.render.HIGHLIGHT_WIDTH` times as thick, the other borders faint,
  and its caption says so; `region_only` omits the context borders for a
  region zoom. `overlay_region_window` bounds the saved placement's
  side-aware selected contours with a small margin. An overlay's caption gives the stored transform's
  numbers (`placement.transform_words`). No picture number is burned into the pixels.
- `positioning.py` — the positioning picture, ABBA's layout: a millimetre
  ruler, atlas thumbnails above it, each over its own millimetre (moved
  aside only as far as its neighbours need), the sections below in
  POSITION order (`position_order`: ties in stack order, `index_original`;
  no position last, without a line), evenly spaced, each joined by a
  line to its position, so no two lines cross; each label is the
  section's filename, so a filename order that disagrees with the
  positions shows there. A stack is split, never shrunk: at most
  `PER_PICTURE` (6) sections and as many atlas thumbnails per picture
  (`split`: the fewest even consecutive runs; each atlas position goes to
  the run around it), each picture with its own ruler segment over the
  positions it holds plus a margin (`ruler_range`, at least
  `RULER_MIN_SPAN_MM`; both minimums are the adult mouse's, times
  `Workspace.brain_scale`) and a caption `part i of n: k of the N sections
  (first.tif to last.tif) ...,
  x-y mm` (on a stack whose sections' angles differ, `atlas at the median
  of the sections' cutting angles, ...`). One um/px for every thumbnail of every picture of a call
  (`scale.pair_um_per_px`, never finer than a section's working copy): the
  tile is the picture size while the fullest picture's row fits
  `POSITIONING_MAX_WIDTH` (1568 px; six tiles of about 240 px), so one
  section beside three atlas positions is drawn large. `plan` is the pure
  geometry (one `Layout` per picture, `Slot`), `paint` draws a layout at a
  magnification and crop (every length scales together),
  `positioning_pictures` does both for a stack (every picture, or one
  `part`); a zoom is drawn up to the sections' working-copy detail, then
  enlarged at most `ATLAS_UPSAMPLE` times, said in the caption.
- `zoom.py` — `redraw(recipe, box, ws, state=, store=, picture=,
  long_edge=, region="")`: a box (`[x0, y0, x1, y1]` pixels of the picture's content)
  drawn again from the source through the recipe's renderer (`RENDERERS`),
  its window composed onto the unzoomed picture's (`box_fractions`,
  `compose`), so a zoom of a zoom maps back; a section or overlay zoom
  reads its section from the image file (`native.py`), a positioning zoom
  draws its sections from their working copies, enlarged at most
  `ATLAS_UPSAMPLE` times past them. A changed stack is
  drawn as it was and flagged `stale` (a second caption band says so). A
  picture without a recipe, or whose sections are gone, is cropped from
  *picture* (the saved image) and enlarged towards the picture size,
  `redrawn` False, with a `crop` recipe (the box on that source). Returns
  `Zoomed`; `ZoomError` `BAD_BOX` / `EMPTY_BOX` / `NO_PICTURE`.
  With `region`, exactly one region replaces the box, on an overlay only
  (`BAD_REGION_PICTURE` otherwise); absent anatomy is `REGION_NOT_IN_PLANE`.
  The bounds and borders come from the saved overlay, never today's state.
- `sheet.py` — `look`'s sheet: `tile(tiles, long_edge, title=)` lays saved
  pictures (`Tile`: image, number, label) in a grid (`grid`: the columns
  closest to a square sheet), each scaled down whole under a header band
  naming its number, nothing drawn over it, the sheet captioned; `recipe`
  (renderer `sheet`: each tile's `picture`, `box` on the sheet and the
  picture's `size`), `tile_under` (the tile holding most of a box) and
  `to_tile` (a sheet box in the tile picture's pixels), which `ops.look.zoom`
  reads; `MAX_TILES` (16) per sheet.
- `zoom_grid.py` and `picture_geometry.source_frame` — the coordinate grid every
  zoom carries. `source_frame(recipe)` is where a zoom's pixels lie in the
  whole (unzoomed) picture's pixels (`SourceFrame`: `to_source`, `to_picture`,
  `covers`; from the recipe's window on `base`, or a crop's box; None for a
  whole picture), `grid_step` the round step (at most `GRID_LINES` lines along
  the longer side) and `draw_grid` the thin dashed lime-green lines (a colour no
  other drawing uses) with labels on a dark chip at every edge, over the content
  only. `zoom.redraw` draws it on every zoom; a zoom's box and a
  `set_landmarks` endpoint on a zoom are read in these numbers (`ops/`).
- `native.py` — sections at their image file's own resolution, for
  zooms. `section_pixels(ws, record, look, x, y)` -> `NativePixels` (`rgb`,
  `inside`, `enlarged`: picture pixels per file pixel): the section at
  working-copy points (before its quarter turn and flip), read around them
  only (`read_region`: a TIFF, plain, tiled, pyramidal or one page per
  channel, through tifffile's zarr store at the coarsest pyramid level
  still fine enough, only the touched tiles decoded, at most
  `MAX_READ_PIXELS` read before a stride; any other file decoded whole and
  cut), its raw channels mapped as the working planes are
  (`window_planes`, each page's `IntensityRange`), averaged down to about
  one read pixel per picture pixel. The colours: the look is drawn on the
  whole working copy as an unzoomed picture draws it (`working_look`) and
  carried to the file's channels by a local linear fit (`fit_look`, a
  colour guided filter, radius `FIT_RADIUS`, regularized by `FIT_EPS`
  towards the gain fitted over the whole region), so a plain channel
  display is carried exactly and CLAHE's local contrast at the working
  copy's scale. `framed_window(ws, record, look, window, long_edge)`: a
  window of the whole, turned picture read so. `unoriented_points`
  / `unoriented_fractions` / `unorient` undo a render's quarter turn and
  flip. `deformable.warp_source_points` carries a warped picture's points
  back to the section's.
- `sizes.py` — the picture sizes: `PICTURE_EDGES` (opening and later long
  edge per `image_resolution`), `AUTO_RESOLUTION`, `MIN_RESOLUTION`,
  `MAX_IMAGES_PER_CALL` (what the agent tools and MCP set `Job.picture_cap` to), `resolution_level`, `opening_edge`, `picture_edge`.
- `layers.py` — a placement picture's layers and frame record (and a
  deformation-fit picture's), the on-demand coordinate map, and the
  per-call picture notes (below).
- `maps.py` — section pixels to atlas
  micrometres for the job folder's public files (`job/formats.py`,
  `docs/reference/file_formats.md`). `SectionFrame` / `section_frame(state,
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
  past `FRAME_CACHE_SECTIONS`). `share_frames(state, workspace)` computes
  every placed section's frame and keeps it for the workspaces the process
  makes after it forks (`_SHARED_FRAMES`; the agent CLI's warm process,
  `doors/cli/warm.py`): a later workspace takes one only at the same image
  folder and job spec (`_scope`), key and atlas object. A section with no pixel size (neither the file
  nor the host gives one) is mapped at the scale its placement pictures draw
  it at, `core.transform.calibrate` on its working frame (uncalibrated: the
  image width mapped onto the atlas plane's width at its CURRENT position,
  never the scale stored with a transform written elsewhere; the one decode), and `scale_problem` /
  `SCALE_UNKNOWN` says so in `registration.json`'s `problem`. No transform: the identity, as the pictures
  draw it (`IDENTITY_PARAMS`, `stored_params`); `placement_problem` says
  why a section has no map. `native_points(frame, warp, x, y)`: native
  plane `(x, y)` of file points, through an applied deformation exactly as
  its record composes it (carried onto the fit grid, displaced by its
  field, bilinear with replicated edges, then its placement undone);
  `file_points(frame, warp, nx, ny)` the way back through the record's
  inverse field; `file_to_fit(frame, fit_size)` the file's pixels onto a
  record's fit grid; `point_rows` lays a list of points out as rows for a
  2-D sampler (`cv2.remap` takes at most 32767 per axis).
  `section_maps(workspace, frame, warp, full_resolution=)`: `SectionMaps`
  on the working copy's grid (or the file's): coordinates (NaN off the
  atlas volume), atlas ids (`core.deformable.geometry.sample_native`, nearest, as the
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
- `itk_io.py` — images and transforms in the ITK world (SimpleITK), for an
  engine run outside LangSlice: `write_image` (TIFF, or NIfTI / MHA / NRRD on
  a grid of spacing 1, origin 0), `mapped_points` (where a transform file, a
  displacement-field image, a SimpleITK object or a pixel-displacement array
  sends each pixel centre of a grid; ITK's fixed-to-moving convention,
  `TransformToDisplacementField`), `inverted` (`InvertDisplacementField`),
  `linear_matrix` (a transform as a 3x3 affine; `ValueError` when it is not
  affine). The library's `write_frame` / `map_from_itk` / `transform_from_itk`.
- `jpeg.py` — the doors' one JPEG encoding (below).
- `handoff.py` — a written linear placement as the nonlinear work
  starts from it: `prepare_linear_registration(state, workspace, id,
  long_edge=, transform=)` (the oriented section render, its
  `preprocessed` image, metadata `section_image` naming a remade one, and the
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
  Linear on and run `elastix_affine` on the section first (the maps still read a missing
  transform as the identity). No provider import.

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
borders' coverage, uint8, `line_coverage` at full strength, the core's
(not the casing's), so the layer and the picture's lines cannot disagree), `picture_to_native`,
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

A deformation-fit picture (`fireants_syn`, a landed trace) gets the same layers on its own
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
extra=, recipe=, caption=)`; `note_for` finds a picture's note by
identity; `annotate(image, recipe=, caption=)` sets fields on the note a
renderer already made for that image (its frame and panel kept). `draw_canvas`
and the `look` renderers note theirs; `ops.deformable.pictures` notes
the deformation fits (`fireants_syn`, a landed trace; through `core.deformation.picture(note=...)`, with
a `WarpNote`: the record resampled onto the picture, the caption band,
whether the field is drawn, the border style) and traces, and the tool door
the pictures it labels itself. With
nothing collecting, a note costs nothing. The job saves the pictures
(`langslice.job.views`, `job/CLAUDE.md`).

`jpeg.py` — `encode_jpeg` (`JPEG_QUALITY` 85): the one encoding every
picture a model receives goes through; `doors.tools.media` re-exports it, the MCP
door and the job's view store call it, so the saved bytes are the sent ones.

## Landmark correspondences

`landmarks.py` holds appearance-independent endpoint geometry, plane staleness,
and open-centre circle/diamond annotations. `picture_geometry.py` carries exact
pixel-centre affine maps alongside the raw renderer images through cropping
and resizing. `look.py` saves endpoint panels and registration context in each
recipe: section views map to original-file edge fractions; atlas views with
exactly one section use that section's own plane and map to native atlas
pixels; overlays map through the actual saved inverse warp. Generic atlas
browsing and positioning thumbnails do not supply correspondence geometry.
Annotations are drawn only on final composite pictures. Look snapshots include
the landmark set, so old zooms preserve its original markers; `ANNOTATION_FIELDS`
(`landmarks`, `damaged_regions`: only fits read the damage marks) leave
`changed_since` alone, so a pair edit or a damage mark never makes a picture
`stale` (`differs_since` still restores the old markers). `pair_states` /
`landmark_counts` give a pair's `applied` / `moved` / `pending` / `stale` state
and the status row's counts (`warp` in words), `applied_pairs` the pairs the
current landmark warp was built from and `warp_report` which pairs are in it,
not in it, or deleted but still shaping it; `draw_folds` outlines the applied warp's folded
triangles (`landmark_folds` in the deformation reference) on a warped
overlay (a vertex that is no pair of the section, such as the `~K` an
earlier version's automatic holds left in a saved warp, is skipped). An atlas `look` given a `resolution` is drawn up to `ATLAS_UPSAMPLE`
times past its voxels. Positioning index
captions include per-section landmark counts without cluttering thumbnails.

`triangulation.py` implements `PiecewiseAffine` and `visualign_mesh`: Delaunay
on section-domain pixels (preserving original aspect ratio), fixed 10% outside
corners, and inverse evaluation on the same mesh connectivity. Repeated
sources and zero-area source triangles are `DegenerateMesh` (with the vertex
indices); folds (a target triangle winding the other way) are accepted:
`folded` marks them, the forward map is the source Delaunay's, and the inverse
tries unfolded triangles before folded ones, each in simplex order.
`ops/landmarks.py` passes it the pairs alone, as VisuAlign meshes them.
`landmark_views.py` draws each edited pair as a contextual template/preprocessed
section crop (the atlas panel with the region borders), retains a 60% anatomical window and panel geometry, and labels
the selected pair. Crop zooms preserve exact panel mappings and refuse gutters.

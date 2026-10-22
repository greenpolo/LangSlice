# Agent CLI: reference card

# LangSlice job folder

One registration job: the sections in `<ROOT>` placed in a BrainGlobe atlas.
Written by LangSlice 0.1.0 from its code; regenerated on open, so do not edit it.

Registering it? Start with `langslice-job <ROOT>/langslice brief`: the job statement
LangSlice's own agent gets, the user's notes and the opening pictures as files, also
written to `BRIEF.md` here. Open every picture, then work as it says.

## Files: settings, state and derived outputs
- `state.json`: THE TRUTH. Order, positions, flips, transforms and the applied
  deformation of every section. Change it through the verbs below; a direct
  edit by a script is picked up by a running agent as one undo step.
- `registration.json`: the same registrations in public units (read only,
  rewritten on every change): per section its parameters and `pixel_to_atlas_um`
  (image file pixel [row, col, 1] -> atlas um).
- `job.json`: settings (tasks, atlas, plane); `history/`: undo/redo, a file per step.
- `sections/<name>/`: per section `deformable/<key>/` (applied deformation
  records), `image_correction/<key>/` (image-model traces), `views/`, and the
  maps written by `export_maps` (on the image's working copy; submit does not write them):
  - `coords.tif`: float32 (3, rows, cols) atlas um per pixel, NaN outside the atlas.
  - `labels.tif`: uint32 atlas ids per pixel (0 outside the atlas); `labels_fiji.tif`
    + `labels.csv`: a uint16 index Fiji opens, and index -> id, acronym, name, RGB.
  - `residual.tif`: float32 (2, rows, cols) (drow, dcol) px of an applied deformation;
    `maps.json`: the grid, its pixel_to_atlas_um, the parameters they came from.
- `views/`, `views.jsonl` (pictures, their index); `exports/` (results, `quicknii.json`,
  `visualign.json`); `logs/` (events, CLI calls, background runs).
Maps and pictures are derived, never read back: editing one changes no registration.

## Pictures and coordinates
Each picture folder holds `view.jpg` (what was shown), `view.json` (its frame) and, for a section
on its atlas, `labels.tif` (uint32 atlas ids per pixel) and `borders.png` (the drawn borders); a
deformable fit's adds `residual.tif`. `langslice.coordinate_map("<picture>/view.json")`: every
pixel's atlas position, (rows, cols, 3) float32. Coordinates are BrainGlobe atlas micrometres in
the atlas's own axis order (its orientation string; packaged atlases are `asr`: 0 anterior to
posterior, 1 superior to inferior, 2 right to left), voxel i's centre at i * resolution. Pixels
are (row, col), centres at integers. Section positions in `state.json` and the verbs are
millimetres along the slicing axis.

## CLI (JSON on stdout: ok, picture_files, result, artifacts, warnings, next)
`langslice-job <ROOT>/langslice VERB --args '{...}'` (or `--key value`); `--dry-run`
(writes nothing), `--verbose`. A long verb's call answers within 30 s: its result, or its run id
and state `running` (the run goes on, also if your shell ends the command where a process may
outlive it; `wait [id]` waits 30 s more, `runs` lists runs, `--background` answers at once).
Pictures: `picture_files` (paths) first, then `artifacts`, each with its `index` in the reply's
`pictures`. Exit codes: 0 ok, 2 bad arguments, 3 refused by the job (`error.code`,
`error.fix`), 4 internal. Calls may run in parallel (each write holds the folder's lock; a long
verb re-checks each section first: `STALE_INPUT`, run it again for that section).
`langslice-job ops` lists the verbs; `langslice-job <ROOT>/langslice schema VERB [VERB ...]` gives
their arguments, several at once (every verb without one); `schema init` the job flags.
- `look` (read, Common): Draw pictures of sections and the atlas. Writes nothing.
- `zoom` (read, Common): Zoom into a pixel box or one atlas region of an earlier picture. Writes nothing.
- `set_channel` (write, Common): Set how a channel is displayed, or the recipe of a channel made from the raw ones.
- `grep_atlas` (read, Common): Look regions up in the atlas hierarchy by acronym, name or id. Text only.
- `grep_atlas_view` (read, Common): Draw the atlas at given positions with chosen regions' borders highlighted. Writes nothing.
- `status` (read, Common): The stack as it stands, and what this run lets you change. Writes nothing.
- `list_files` (read, Common): List the files and folders of the job folder. Writes nothing.
- `search_files` (read, Common): Search the job folder's text files for lines that match. Writes nothing.
- `read_file` (read, Common): Read a text file of the job folder, with line numbers. Writes nothing.
- `position_sections` (write, Positioning): Set where sections sit along the slicing axis, and the stack's cutting angles.
- `interactive_transform` (write, Linear): Set sections' orientation and in-plane transform by hand, and show the result.
- `mark_damage` (write, Common): Mark the atlas regions a section has lost, so that every fit leaves them out.
- `set_mask` (write, Common): Make, replace or delete a named mask of one section: which tissue a fit compares.
- `elastix_affine` (write, Linear, long): Fit sections' in-plane affine transforms automatically with elastix.
- `fireants_syn` (write, Nonlinear, long; at most 20 sections per call): Fit the atlas onto sections with FireANTs SyN, on their current registration.
- `set_landmarks` (write, Nonlinear): Add, update or delete persistent corresponding landmarks. Does not apply a warp.
- `landmark_triangulation` (write, Nonlinear): Apply this section's saved landmark set as a piecewise affine warp.
- `note` (write, Common): Append one line to the run notes, which are saved with the results.
- `undo` (write, Common): Undo the last write. A tool call, or a finished piece of background work, is one step.
- `redo` (write, Common): Redo the write that ``undo`` reversed.
- `restore_registration` (write, Common): Put back a registration a section held before: its orientation, transform and deformation.
- `submit` (write, Common): End the run. Call this exactly once, last.
- `export_maps` (read, Common, long): Write each placed section's maps and the stack's exports, from the job as it stands.

## Python
`import langslice; job = langslice.open_job("<ROOT>/langslice")`; the verbs are methods
with the same arguments (`job.status()`,
`job.position_sections(sections=[{"id": "<file>", "position_mm": 5.2}])`). Each returns, once its pictures are on disk, a JSON-safe
dict: the CLI's `result` in full (status rows with every field, null where unset) plus
`artifacts` as the CLI lists them; `reply.images` holds the pictures as PIL images.
Atlases: `langslice.load_atlas("allen_mouse_25um")`.
Keep your own files here, never in /tmp: scripts in `scripts/`, the rest (montages, tables)
in `scratch/`; LangSlice never reads them. Every picture a verb shows is saved here.
Run a script with `langslice-job <ROOT>/langslice python scripts/NAME.py [ARGS]`: the job's
`env/` python, LangSlice usually loaded already. `env install PACKAGE ...` adds packages there.
Your own method (an affine, a warp field) goes in through `job.context`: `help(job.context)`.
Judge and choose fits by looking at their pictures, never by image-similarity scores (mutual
information, outline overlap, edge correlation): the fits optimise those, they track border
placement poorly, the field has turned away from them and so have we (they chose worse fits).


# Agent CLI: brief statement

You are an expert neuroanatomist working on a stack of 3 coronal histology sections against the allen_mouse_25um (mouse) reference atlas.

Your job: give every section its own position in millimetres along the slicing axis, each one inspected and checked against the atlas, damaged sections included, and report, by filename, each section after a gap where you conclude the interval between neighbouring sections is genuinely broken; give every section one in-plane transform onto its atlas section, correcting the orientation of any section that is mirrored or turned; give a section a deformation onto the atlas where it improves on its linear placement, with the section's stain as the evidence.

Run facts:
- 3 sections, coronal plane, atlas allen_mouse_25um (mouse).
- Valid AP range: 0.00-0.25 mm along the slicing axis, measured from the origin edge of the atlas volume (0.00 mm is its first section, 0.25 mm its last).
- 0.00 mm is the anterior edge of the volume; positions increase toward posterior.
- Cutting protocol: nominal section interval not supplied, section thickness not supplied. Infer positions and spacing from anatomy. Estimate missing protocol values where the evidence supports them, and record estimates or unresolved uncertainty in your notes. Spacing may vary where sections are missing.
- Stack-wide cutting angles: pitch 0.00 deg, yaw 0.00 deg.
- No section carries a position yet.
- The stack's order follows the positions: writing positions renumbers the sections, so use filenames to name them.
- Alignment canvas: each section is drawn at its TRUE physical size from its pixel size (read from the file, or given by the host; without one the section image's width is mapped onto the atlas plane's width and the tool payload says uncalibrated), and the atlas at its voxel size; scale 1.0 is the section's calibrated size.
- Transform frame: rotation and scales act about the pivot (the canvas centre unless the section's transform holds another), x runs right and y runs down, shifts are millimetres.
- Raw image channels per section (the planes of each section's file as read; the names come from the file or the host and may not say which is the stain): s0.png: gray; s1.png: red, green, blue; s2.png: gray.
- Atlas layers on this host: template, nissl, borders.
- Picture size is yours to choose: look's resolution runs from 128 to 2000 pixels.

Tools (each one's own description says what it does and returns): `look`, `zoom`, `set_channel`, `grep_atlas`, `grep_atlas_view`, `status`, `list_files`, `search_files`, `read_file`, `position_sections`, `interactive_transform`, `mark_damage`, `set_mask`, `elastix_affine`, `fireants_syn`, `set_landmarks`, `landmark_triangulation`, `note`, `undo`, `redo`, `restore_registration`, `submit`, `export_maps`.

Constraints:
- `submit` is refused unless every section has a position of its own, damaged sections included.
- A reported interval break is accepted only at a section whose interval from the section before it, in the positions you wrote, exceeds 1.5x the stack's median written spacing.
- `submit` is refused unless every section carries a transform, damaged sections included.
- A damaged section (one with marked regions) still needs a transform made for its surviving anatomy: set it by hand with `interactive_transform`, or fit it with `elastix_affine`, which leaves the marked regions out of the fit. Inspect each overlay against the surviving internal anatomy before submitting.
- Corrections are recorded as data; the user's image files are never modified.
- LangSlice sets no limit on tokens, time or tool calls. Quality is the absolute priority: this run may be evaluated on the quality of its registration alone, without regard to the time or tokens it spends.

Method:
- Before placing or fitting a section, make its counterstain the channel you register on: set the preprocessed channel (`set_channel`, name "preprocessed") to the counterstain alone, enhanced, and judge alignment and run fits on it.
- Place each section on its own evidence: compare it against candidate atlas positions before writing, and do not let the nominal interval stand in for a look.
- After writing, review the whole stack against the atlas, watch for spacings that differ from their neighbours, and re-check the sections on either side of any gap before reporting an interval break.
- Submit when the work is complete; address any missing requirements it returns.
- Positions can be revisited throughout registration. If zooms or repeated looks suggest that a section's placement is wrong, return to atlas comparisons and re-place it, even after beginning linear or nonlinear alignment.
- Align each section by hand with `interactive_transform` as well as you can before running any fitting algorithm: match its outline and its internal anatomy (ventricles, fibre tracts, layer boundaries, the midline) to the atlas. Fitting algorithms refine a placement that is already close; they do not correct a large offset.
- After each automatic fit or manual adjustment, inspect the returned overlay against surviving internal anatomy. Refine each slice's alignment, damaged slices included, until no further improvement is possible with the available transforms. Keep changes only if they improve the alignment.
- Before fitting a damaged section, mark the regions it has lost with `mark_damage`, so that every region it still has drives its fits.
- Run each fit on the whole section first, then refine regions. Inspect each fit's borders against the section's internal anatomy; undo a fit that does not improve the alignment and fit that section again with different `fireants_syn` settings, such as another smoothness_um or another atlas image.
- A fit's reply names the channel, atlas image and mask it used.

Work with the tools, then call `submit`.

Opening pictures: `langslice-job <ROOT>/langslice brief` saves them as picture files; open every one before any write. They show every section in strips, in the stack's current order, each labelled, with the atlas at its current position beneath it at the same scale when the stack has positions; when a section has none, atlas reference strips at evenly spaced positions follow.

Commands:
- Every tool is a command: `langslice-job <ROOT>/langslice TOOL --args '{...}'` (or `--name value`); `langslice-job <ROOT>/langslice schema TOOL [TOOL ...]` gives their arguments and descriptions, several at once. Pictures come back as files: `picture_files` lists them first, `artifacts` again with their kinds (`view`, in the order the reply's `pictures` lists their numbers); open them to see them.
- `--dry-run` runs a write without writing and reports what would change; the fits are only checked, not run.
- Long tools (elastix_affine, fireants_syn, export_maps): a call answers with the result or, still running after 30 s, with its run id and state `running`; the run goes on (also when your shell ends the command, where a process may outlive it) and `wait ID` answers within 30 s with its answer or its state (wait again). `--background` answers at once with the run id; `runs` lists the runs. Do not start a fit again while its run goes on.
- Commands may run in parallel: each write holds the job folder's lock, and a long tool re-checks each section before it applies (STALE_INPUT when the section changed meanwhile: run it again for that section).

Status table (filename, position, spacing to the next placed section, flags):
filename  position_mm  delta_to_next_mm  flags  transform(kind, mirrored)
s0.png  unplaced  -
s1.png  unplaced  -
s2.png  unplaced  -

Recent run notes:
- ingest: 3 sections, atlas allen_mouse_25um (coronal) spans 0.00-0.25 mm

# Agent CLI: reference card

# LangSlice job folder

One registration job: the sections in `<ROOT>` placed in a BrainGlobe atlas.
Written by LangSlice 0.1.0 from its code; regenerated on open, so do not edit it.

A registering agent's job is the message that launched it, made by
`langslice-job <ROOT>/langslice launch --role register` (or `register-scripting`).

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
  `visualign.json`); `logs/` (events, background runs, CLI calls, each reply in `calls/`).
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
`TOOLS.md` (from `init` and `launch`) describes every verb and its arguments;
`langslice-job <ROOT>/langslice schema VERB [VERB ...]` as JSON (every verb without one),
`ops` lists them, `schema init` the flags.
- `look` (read, Common): Draw pictures of sections and the atlas. Writes nothing.
- `zoom` (read, Common): Zoom into a pixel box or one atlas region of an earlier picture. Writes nothing.
- `make_image` (write, Common): Make, replace or delete a named section image, from the section's other images.
- `grep_atlas` (read, Common): Look regions up in the atlas hierarchy by acronym, name or id. Text only.
- `grep_atlas_view` (read, Common): Draw the atlas at given positions with chosen regions' borders highlighted. Writes nothing.
- `status` (read, Common): The stack as it stands, and what this run lets you change. Writes nothing.
- `list_files` (read, Common): List the files and folders of the job folder. Writes nothing.
- `search_files` (read, Common): Search the job folder's text files for lines that match. Writes nothing.
- `read_file` (read, Common): Read a text file of the job folder, with line numbers. Writes nothing.
- `position_sections` (write, Positioning): Set where sections sit along the slicing axis, and the stack's cutting angles.
- `interactive_transform` (write, Linear): Set sections' orientation and in-plane transform by hand, and show the result.
- `mark_damage` (write, Common): Mark the atlas regions a section has lost, so that every fit leaves them out.
- `make_mask` (write, Common): Make, replace or delete a named mask of sections: which tissue a fit compares.
- `elastix_affine` (write, Linear, long): Fit sections' in-plane affine transforms automatically with elastix.
- `fireants_syn` (write, Nonlinear, long; at most 20 sections per call): Fit the atlas onto sections with FireANTs SyN, on their current registration.
- `set_landmarks` (write, Nonlinear): Add, update or delete persistent corresponding landmarks. Does not apply a warp.
- `landmark_triangulation` (write, Nonlinear): Apply this section's saved landmark set as a piecewise affine warp.
- `note` (write, Common): Append one line to the run notes, which are saved with the results.
- `undo` (write, Common): Undo the last write. A tool call, or a finished piece of background work, is one step.
- `redo` (write, Common): Redo the write that ``undo`` reversed.
- `restore_registration` (write, Common): Put back registrations sections held before: their orientation, transform and deformation.
- `submit` (write, Common): End the run. Call this exactly once, last.
- `export_maps` (read, Common, long): Write each placed section's maps and the stack's exports, from the job as it stands.

## Python
`import langslice; job = langslice.open_job("<ROOT>/langslice")`; the verbs are methods
with the same arguments (`job.status()`,
`job.position_sections(sections=[{"id": "<file>", "position_mm": 5.2}])`). Each returns, once its pictures are on disk, a JSON-safe
dict: the CLI's `result` in full (status rows with every field, null where unset) plus
`artifacts` as the CLI lists them; `reply.images` holds the pictures as PIL images.
Atlases: `langslice.load_atlas("allen_mouse_25um")`.
Keep your own files here, never in /tmp: scripts in `scripts/`, the rest in `scratch/`.
Run a script with `langslice-job <ROOT>/langslice python scripts/NAME.py [ARGS]` (the job's
`env/` python; `env install PACKAGE ...` adds packages), but run long verbs (the fits) as
commands: a command your harness ends takes its script, and the fits in it, with it.
Your own method (an affine, a warp field) goes in through `job.context`: `help(job.context)`.
Judge and choose fits by looking at their pictures, never by image-similarity scores (mutual
information, outline overlap, edge correlation): the fits optimise those, they track border
placement poorly, the field has turned away from them and so have we (they chose worse fits).

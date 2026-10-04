# The agent CLI

`langslice job FOLDER VERB` runs one verb on a job folder and answers with one
JSON object on stdout. It is built for coding agents (Claude Code, Codex)
working in their own sandbox; no person is expected to use it over a GUI.
Code: `src/langslice/doors/cli/` (`job.py`, `catalog.py`, `envelope.py`,
`background.py`).

## The verbs

The verbs are the agent tools, under the same names, with the same arguments
and descriptions: one declaration per verb (`src/langslice/doors/declarations.py`)
feeds the agent tools, the MCP tools, this CLI and the library, and one
registry (`src/langslice/ops/registry.py`) lists them with their kind (read or
write) and group (Common, Positioning, Linear, Nonlinear). A job has the verbs
its tasks switch on (`status` lists them under `verbs`). A verb is never
renamed once shipped. Kebab-case is accepted (`set-positions`).

One verb is the CLI's and the library's only, never a model's tool
(`Verb.scripting` in the registry): `export_maps` (`--slices`, empty for
every section; `--full-resolution`) writes each placed section's maps
(`coords.tif`, `labels.tif`, `labels_fiji.tif` + `labels.csv`,
`tissue.png`, `residual.tif`, `maps.json`), `exports/quicknii.json`,
`exports/visualign.json` and `registration.json`, from the job as it
stands; nothing in the state changes. `submit` writes the same files.
`docs/file_formats.md` describes every file.

```bash
langslice ops                         # every verb: name, kind, group, one line
langslice schema                      # every verb's argument schema (versioned)
langslice schema fit_deformable       # one verb's
langslice schema fit_deformable --job FOLDER   # as that job declares it
```

Besides the verbs, `FOLDER` takes:

- `init`: create the job for a folder of section images, with the job flags
  of `langslice linear run` (`--tasks`, `--atlas`, `--plane`,
  `--pixel-size-um`, `--image-provider none`, ...), through the same ingest
  every host uses; no agent runs. An existing job there is continued
  (`--fresh` starts over); one made from other supplied inputs
  (`--positions`, `--transforms`, `--orientation`, `--order`,
  `--pitch`/`--yaw`, `--locked`, `--damaged`, `--pixel-size-um`, ...) is refused with `INPUTS_CHANGED` (exit 3), which
  names the inputs that differ and `--fresh`.
- `runs [ID]`: the background runs, newest first, or one run's state
  (running, finished with its answer, or lost). `status` is only the verb.
- `wait [ID]`: wait for a background run (the latest without ID);
  `--timeout SECONDS`.

`FOLDER` is the job folder or the image folder beside it.

## Arguments

- `--args '{"entries": [...]}'` or `--args @file.json`: the arguments as one
  JSON object.
- `--name value` per argument (kebab or snake case), over `--args`: a JSON
  value, plain text for a text argument; a list argument takes the flag once
  per item or a JSON list; a yes/no argument alone means true.
- `--dry-run`: a write runs on the job without writing anything (state,
  history, pictures) and reports `would_change` (per section the fields that
  would change, and the stack's). `fit_deformable` and `trace_borders` are
  checked (arguments, sections), not run. `export_maps` writes no file and
  lists under `files` what it would write.
- `--background`: answer at once with a run id; the verb runs in a detached
  process, recorded in `logs/runs/<id>.json` (its stderr in `<id>.log`).
- `--verbose`: the whole reply (whole-stack rows on writes, the
  descriptions written for a model, a picture's text lines).

Pictures take the same `view` options as the tools, `view.resolution`
included: any long edge from 128 px up to the source's own pixels (nothing is
upsampled past them). The look-before-commit gates of `position.gated` do not
apply: they are tool-only.

## The answer

```json
{"ok": true,
 "result": {"status": "ok", "written": [{"id": "s0.png", "position_mm": 0.1}], ...},
 "artifacts": [{"path": "/data/M04/langslice/sections/s0/views/000001_set_positions_overlay/view.jpg", "kind": "view"},
               {"path": ".../view.json", "kind": "view_json"},
               {"path": ".../labels.tif", "kind": "labels"},
               {"path": ".../borders.png", "kind": "borders"}],
 "warnings": [],
 "next": []}
```

When `ok` is false, `error` says why and what to do:
`{"code": "MISSING_POSITIONS", "message": "...", "fix": "Write every section's position first (set_positions)."}`;
`result` still holds the verb's reply (the ids it names, the rows it refused).

| Exit | Meaning |
| --- | --- |
| 0 | ok |
| 2 | the call is wrong: unknown verb or argument, a missing or malformed one, an unknown section, no job in FOLDER |
| 3 | the job refused it: a gate or rule (`MISSING_TRANSFORMS`, `LOCKED`, `NOTHING_TO_UNDO`, a verb this job's tasks do not have, `STALE_INPUT` for every section), another writer holding the lock past 300 s (`JOB_BUSY`), or a background run still running at `wait --timeout` |
| 4 | internal error (the traceback is on stderr); a background run that ended without an answer |

`result` is concise by default: no description written for a model, and a
write's whole-stack `rows` replaced by `n_rows` (`status` and `view_stack`
keep theirs). Pictures are never inlined: each is saved in the job folder as
every door saves it, and listed under `artifacts` by absolute path and kind:
`view` (the JPEG), `view_json` (its frame: `langslice.coordinate_map(path)`
gives each pixel's atlas micrometres), and for a section on its atlas
`labels` (uint32 atlas ids) and `borders`, plus `residual` for a
`fit_deformable` picture; `results`, `registration`, `quicknii`,
`visualign` and each section's `coords`, `labels`, `labels_fiji`,
`labels_csv`, `tissue`, `residual` and `maps` (after `submit` and `export_maps`);
`card` and `state` (after `init`). Progress and library output go to stderr.

## Live shared editing

Each call opens the job as it stands on disk (nothing is rewritten by
opening), runs the verb and closes it. A LangSlice agent run on the same
folder picks up a CLI write before its next tool call, history included, and
the next CLI call picks up the agent's; every write is one undo step either
way. Every write holds the job folder's lock (`job.lock`, across processes,
Linux, macOS and Windows): lock, reload what others saved, apply, commit,
unlock, so concurrent writers never overwrite each other. `fit_affine`,
`fit_deformable` and `trace_borders` compute outside the lock (a fit can take
minutes) and take it only to apply: a section whose inputs (position, plane,
cutting angles, orientation, transform, fit appearance, applied deformation
or trace, whatever the result was computed from) changed meanwhile is
refused as that section's row, `STALE_INPUT` (run it again), and the other
sections apply; the envelope lists it under `warnings`.

## The reference card and the library

Every job folder holds `AGENTS.md` and `CLAUDE.md` (identical, generated by
`src/langslice/doors/card.py`, rewritten when stale): what the folder holds,
that `state.json` is the truth and `registration.json`, pictures and maps
are derived (one line per file), the coordinate convention (BrainGlobe micrometres, the atlas's axis order, voxel
`i`'s centre at `i * resolution`), the verbs, and the Python entry point:

```python
import langslice

job = langslice.open_job("/data/M04")        # the verbs as methods
job.set_positions(entries=[{"id": "s0.png", "position_mm": 5.2}])["images"]  # PIL images
langslice.coordinate_map(".../view.json")    # (rows, cols, 3) float32 atlas µm
langslice.load_atlas("allen_mouse_25um")     # a BrainGlobe atlas
```

`import langslice` and `open_job` load no agent framework or model client
(`tests/test_core_imports.py`). A scripted pipeline (making jobs, image-model
profiles, `register_section` / `register_job`): [`library.md`](library.md).

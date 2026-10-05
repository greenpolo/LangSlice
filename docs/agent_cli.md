# The agent CLI

`langslice job FOLDER VERB` runs one verb on a job folder and answers with one
JSON object on stdout. It is built for coding agents (Claude Code, Codex)
working in their own shell. Code: `src/langslice/doors/cli/`.

```bash
langslice job sections/ init --tasks position,transform --pixel-size-um 0.65
langslice job sections/ brief                      # the job statement, opening pictures, status
langslice ops                                      # every verb: name, kind, group, one line, long
langslice schema fit_deformable                    # one verb's description and arguments
langslice job sections/ set_positions --entries '[{"id": "s01.tif", "position_mm": 5.2}]'
langslice job sections/ view_slices --slices s01.tif --view '{"resolution": 1200}'
langslice job sections/ fit_deformable --slices s01.tif --background   # then: wait
```

`FOLDER` is the job folder or the image folder beside it.

## The verbs

The verbs are the agent tools, with the same names, arguments and
descriptions: one declaration per verb (`src/langslice/doors/declarations.py`)
feeds the agent tools, the MCP tools, this CLI and the library, and
`src/langslice/ops/registry.py` lists them with their kind (read or write) and
group (Common, Positioning, Linear, Nonlinear). A job has the verbs its tasks
switch on (`status` lists them). Kebab-case is accepted (`set-positions`).
`export_maps` exists only here and in the library (`Verb.scripting`): it
writes each placed section's maps and the exports from the job as it stands
([file_formats.md](file_formats.md)).

`langslice schema [VERB] [--job FOLDER]` gives, per verb, its `description`,
`summary`, `kind`, `group`, `long`, the argument schema and, for a verb that
returns pictures, `picture_options`. With `--job` (or run inside a job
folder) it declares the verbs as that job's settings do. A long verb
(`fit_affine`, `fit_deformable`, `trace_borders`, `export_maps`) may take
minutes: `ops` marks it `long` and `schema` or a dry run gives the
`--background` command.

## Commands on FOLDER

- `init`: create the job with the job flags of `langslice linear run` (`--tasks`,
  `--atlas`, `--plane`, `--pixel-size-um`, `--image-provider none`,
  `--positions`, `--transforms`, ...); no agent runs. `--notes TEXT` keeps the
  user's notes in `job.json` (every door gives them to the agent);
  `--viewer claude|codex|openai` says which model reads the pictures. An
  existing job is continued (`--fresh` starts over); one made from other
  supplied inputs is refused with `INPUTS_CHANGED`, naming the differing inputs.
- `brief`: what LangSlice's own agent gets when it starts
  (`src/langslice/doors/statement.py`): the job statement, the notes and the
  status table under `result.statement`, the opening pictures saved in the job
  folder and listed as `artifacts` of kind `opening`, and all of it in
  `BRIEF.md` (with LF line endings on every platform, matching the returned
  statement). It also reports the `viewer`, the `resolution` range a call may
  ask for and the job's `image_model`. Run it again for the stack as it stands.
  The JSON envelope escapes non-ASCII characters for consoles that do not use
  UTF-8; parsing the JSON restores the original text.
- `runs [ID]`: the background runs, newest first, or one run's state.
- `wait [ID]` (`--timeout SECONDS`): wait for a background run (the latest
  without ID).

`init --registration FILE` starts from a linear registration made elsewhere
(QuickNII or VisuAlign JSON/XML, DeepSlice CSV/JSON/XML, a LangSlice
`registration.json`): each matched section gets the file's position, cutting
angles, orientation and in-plane transform as supplied inputs, and the job's
tasks are `nonlinear` only. Matching, what is read and what is refused:
[file_formats.md](file_formats.md), "Importing a registration made elsewhere".
The result's `registration` reports the placed, unmatched, missing and refused
sections; a file that cannot be read or places no section is `BAD_REGISTRATION`
(exit 2). `linear run`, `mcp` and `claude prepare` take `--registration` too.

## Arguments

- `--args '{"entries": [...]}'` or `--args @file.json`: all arguments as one JSON object.
- `--name value` per argument (kebab or snake case), overriding `--args`: a JSON
  value, or plain text for a text argument; a list argument takes the flag once
  per item or a JSON list; a yes/no argument alone means true.
- `--dry-run`: a write runs on the job without writing anything and reports
  `would_change`; the long verbs are checked, not run; `export_maps` lists the
  files it would write.
- `--background`: answer at once with a run id; the verb runs detached,
  recorded in `logs/runs/<id>.json`.
- `--verbose`: the whole reply (whole-stack rows, descriptions written for a model).

Pictures take the same `view` options as the tools, `view.resolution` included:
any long edge from 128 px up to the viewer's largest, never past the source's
own pixels. A larger request is clamped and `view.resolution_note` says so.

## Pictures and the viewer

A picture is a file the coding agent opens with its own image reader, so the
job's viewer (`init --viewer`, default `claude`) sets the opening strips' size
and the largest picture a call may ask for (`core/opening.py` `VIEWER_LIMITS`):

| Viewer | Opening strips | Largest picture |
| --- | --- | --- |
| `claude` | 1568 px, about 1.2 MP | 2000 px |
| `codex`, `openai` | 2048 px, 2,500 32-px patches | 2048 px |

The library (`langslice.open_job`) has no viewer: pictures are capped only by
their source.

## The image model

A job whose nonlinear task names an image model offers `trace_borders` only when
that model is connected here (a key or login present, checked offline by
`doors.api.setup.image_model_connected`). `status` and `brief` report it under
`image_model`; calling the verb otherwise is refused with `IMAGE_MODEL_OFF`
(exit 3).

## The answer

```json
{"ok": true,
 "result": {"status": "ok", "written": [{"id": "s0.png", "position_mm": 0.1}], "...": "..."},
 "artifacts": [{"path": "/data/sections/langslice/sections/s0/views/000001_set_positions_overlay/view.jpg", "kind": "view"}],
 "warnings": [],
 "next": []}
```

When `ok` is false, `error` has a `code`, a `message` and a `fix`; `result` still
holds the verb's reply.

| Exit | Meaning |
| --- | --- |
| 0 | ok |
| 2 | the call is wrong: unknown verb or argument, a missing or malformed one, an unknown section, no job in FOLDER |
| 3 | the job refused it: a gate or rule (`MISSING_TRANSFORMS`, `LOCKED`, `NOTHING_TO_UNDO`, a verb this job's tasks do not have, `STALE_INPUT` for every section), another writer holding the lock past 300 s (`JOB_BUSY`), a background run still running at `wait --timeout` |
| 4 | internal error (traceback on stderr); a background run that ended without an answer |

`result` is concise by default; `--verbose` gives the full text. Pictures are
never inlined: each is saved in the job folder and listed under `artifacts` by
absolute path and `kind` with its `index` (the number the reply's
`image_indexes` give): `view` (the JPEG), `view_json` (its frame:
`langslice.coordinate_map(path)` gives each pixel's atlas micrometres),
`labels` and `borders` (a section on its atlas), `residual` (a
`fit_deformable` picture); after `submit` and `export_maps`: `results`,
`registration`, `quicknii`, `visualign` and each section's `coords`, `labels`,
`labels_fiji`, `labels_csv`, `tissue`, `residual`, `maps`; after `init`:
`card`, `state`, `brief`.

## Call log, traces, shared editing

Every call is logged in `logs/calls.jsonl` (verb, arguments, `ok`, exit code,
error code, artifacts, seconds, background run id; a lean job logs nothing).
With `LANGSLICE_TRACE_DIR` set, each call is also traced like an MCP tool call
(`src/langslice/doors/trace.py`) into `cli_<images>_<digest>.jsonl`.

Each call opens the job as it stands on disk, runs the verb and closes it. A
`langslice linear run` on the same folder picks up a CLI write before its next
tool call and the reverse; every write is one undo step. Writes hold the job
folder's lock (across processes): lock, reload, apply, commit, unlock.
`fit_affine`, `fit_deformable` and `trace_borders` compute outside the lock and
take it only to apply: a section whose inputs changed meanwhile is refused as
that section's row, `STALE_INPUT` (run it again), and the others apply.

## The reference card and the library

Every job folder holds `AGENTS.md` and `CLAUDE.md` (identical, generated by
`src/langslice/doors/card.py`): run `brief` first; what the folder holds
(`state.json` is the truth; `registration.json`, pictures and maps are derived);
the coordinate convention; that calls may run in parallel; the verbs; and the
Python entry point:

```python
import langslice

job = langslice.open_job("/data/sections")   # the verbs as methods
job.set_positions(entries=[{"id": "s0.png", "position_mm": 5.2}])["images"]   # PIL images
langslice.coordinate_map(".../view.json")    # (rows, cols, 3) float32 atlas micrometres
```

`import langslice` loads no agent framework or model client. A scripted
pipeline (`create_job`, image-model profiles, `register_section`,
`register_job`): [library.md](library.md). Ready-made Claude Code and Codex
setups: [`connectors/claude-code/`](https://github.com/greenpolo/LangSlice/blob/main/connectors/claude-code/README.md),
[`connectors/codex/`](https://github.com/greenpolo/LangSlice/blob/main/connectors/codex/README.md).

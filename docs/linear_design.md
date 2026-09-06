# LangSlice linear — unified environment design (2026-09-05)

`langslice linear` places a stack of histology sections on a BrainGlobe atlas:
order, position along the slicing axis, and one in-plane transform per section.
It is ONE agent environment — one state, one toolbox, one job statement — that
a host switches on task by task. Running everything is the default.

## Why one environment

Looking at the whole stack once yields the information for every task: which
sections are mirrored, which are out of order, where sections are missing, which
are damaged, where each one sits. Splitting those into separate agent sessions
throws that shared reading away and re-pays for it. The one exception is the
per-section interactive alignment, which is local to a section and runs as a
bounded sub-session the main agent spawns and reviews.

## The job spec

A host (CLI, ABBA plugin, engine service) fills one spec. Every checkbox a user
sees maps to a field here; nothing else is user-facing.

```
JobSpec
  image_folder, atlas, plane, model, out, preprocess (auto|none)
  tasks: subset of {reorder, position, transform}     # default: all three
  reorder:
    flip: bool = True             # may flip sections across the midline
    hemisphere_cue: str = ""      # user text: what marks a hemisphere (notch, injection...)
  position:
    thickness_um, interval_um     # cutting protocol; passed as facts
    strict_interval: bool = False # sections must sit exactly one interval apart
    deepslice: bool = False       # run_deepslice tool available (coronal mouse/rat only)
    bayesian: bool = False        # fit_position tool available (oblique.py fitter)
  transform:
    angles: bool = False          # may set the stack-wide cutting angles
    elastix: bool = False         # fit_affine may use Elastix intensity affine
    subagents: bool = True        # align_slice runs as a tool the main agent calls;
                                  # False = engine runs the sessions after submit
  facts: free-form user facts, one line each, passed verbatim
  inputs: order/positions/angles supplied by the host for tasks that are OFF
```

Task OFF means its outputs are inputs: reorder off → discovery order is fixed;
position off → positions come from the host and are facts; transform off →
no transform tools, no alignment.

## State

`StackState` is the checkpoint, the result, and the thing every tool writes.
Same JSON shape for all three.

```
StackState
  image_folder, atlas, plane, interval_mm, thickness_mm
  spec: the JobSpec as run
  cutting_angles_deg: {pitch, yaw}      # stack-wide; 0/0 = flat
  interval_breaks: [corrected indices]  # section AFTER a gap the agent concluded is real
  notes: [str]                          # agent notes, run log
  slices: [SliceState]
  submitted: bool
SliceState
  id, index_original, index_corrected
  flip: bool, rotation_deg: 0|90|180|270   # rotate first, then flip left-right
  damaged: bool, damage_note: str          # agent-internal: excludes from DeepSlice
                                           # and automatic affine; never a user option
  position_mm | null, confidence
  transform | null: {kind: silhouette|elastix|interactive, params|matrix, iou?, note?}
  caveats: [str]
```

Order and position are separate fields that must agree at submit: reordering a
section that already has a position CLEARS that position (it was assigned under
the wrong neighbours), and `submit` refuses positions that are not monotone
along the corrected order. Standalone reorder therefore outputs a permutation;
unified runs output positions and the order comes for free.

The user's image files are never modified. Everything is a proposal as data.

## Toolbox

Conventions, applied to every tool: sections are addressed by filename or
corrected index; every write returns the same status rows; every write is
undoable; every write checkpoints; fits have a preview form that computes
without writing. Tools report data. No advice, no interpretation, no strategy
in any payload or prompt (see `lean-harness` history in `linear/CLAUDE.md`).

| tool | task gate | does |
| --- | --- | --- |
| `status` | always | one row per section in corrected order: index, id, position_mm, spacing_to_next_mm, flip, rotation_deg, damaged(+note), transform kind, confidence, caveats; plus cutting angles and interval breaks. The `ls` of the environment. |
| `view_slices(ids)` | always | up to 8 sections at higher resolution, rendered as corrected. |
| `fetch_atlas(positions_mm)` | always | up to 8 atlas sections, rendered at the current cutting angles. |
| `note(text)` | always | append to the run notes. |
| `undo()` / `redo()` | always | snapshot stack; a batch call undoes as one. |
| `orient_slices([{id, flip?, rotate_deg?}])` | reorder.flip (flip) / reorder (rotate) | toggle flip, add rotation. |
| `reorder_slices(new_order)` | reorder | full permutation; clears positions of sections whose index changed. |
| `move_slice(id, after)` | reorder | incremental move; same clearing rule. |
| `mark_damaged([{id, note}])` / unmark | always | agent-internal classification. |
| `set_positions([{id, position_mm, confidence?}])` | position | batch write, clamped to the atlas range. |
| `distribute_spacing(fixed=[{id, position_mm}], keep=[ids])` | position | linear interpolation between fixed points, extrapolating at the implied interval; `keep` sections are not moved; returns rows, writes nothing (`apply=True` writes). |
| `run_deepslice(ids?, allow_angle_change, keep=[ids])` | position.deepslice | positions (+ angles) for undamaged sections; UNAVAILABLE unless installed and plane/atlas supported. |
| `fit_position(id, window_mm, angles?)` | position.bayesian | `oblique.fit_oblique` at the section's current position: best position (and angles) with score; writes nothing. |
| `set_cutting_angles(pitch_deg, yaw_deg)` | transform.angles | stack-wide; subsequent atlas fetches and fits use them. |
| `fit_affine(ids, method=silhouette\|elastix, apply=True)` | transform | per-section in-plane affine against its atlas section; returns iou and an overlay panel per section (≤8); refuses damaged sections. |
| `align_slice(id, notes)` | transform (subagents) | runs the bounded interactive sub-session for ONE section; returns params + final overlay for the main agent to accept or re-call. |
| `copy_transform(from_id, to_ids)` | transform | copy one section's transform to others. |
| `submit(summary, notes, interval_breaks)` | always | ends the run; gated (below). |

`fetch_atlas` and `view_slices` frame tissue the same way so apparent scale is
not a cue. Seed message: every section as its own labelled image in corrected
order plus the status table.

## Submit gates (constraints, not coaching)

- position on: every section has a position; positions monotone along the
  corrected order (`ORDER_POSITION_MISMATCH`, naming the pairs).
- strict_interval: every consecutive spacing within 10% of the interval;
  `interval_breaks` must be empty.
- not strict: each reported break index must sit where the written interval
  exceeds 1.5x the stack's median written spacing (`INTERVAL_BREAKS_UNSUPPORTED`).
- transform on, subagents off: nothing to gate; the engine runs the sessions
  after submit (silhouette affine for intact, interactive for damaged).

Refusals state the numbers and stop.

## Session

One ADK `LlmAgent`, tools built from the spec, the job statement built from the
spec and state, the seed message from the state. The loop nudges when the model
answers in prose and ends at submit or the turn budget. Every write tool
checkpoints; a run that dies mid-way resumes from the checkpoint with the
state it had (the agent is re-seeded, not replayed). `LANGSLICE_TRACE_DIR`
records the full trajectory (`trace.py`, unchanged).

`align_slice` is an `AgentTool`-style sub-session: its own small toolbox
(`preview_transform`, `submit_transform`), its own turn cap, seeded with the
section, its atlas section, and the notes the main agent passed.

## CLI

```
langslice linear run FOLDER [--tasks reorder,position,transform]
    [--atlas ..] [--plane ..] [--model ..] [--preprocess auto|none]
    [--no-flip] [--hemisphere-cue TEXT]
    [--thickness UM] [--interval UM] [--strict-interval] [--deepslice] [--bayesian]
    [--angles] [--elastix] [--no-subagents]
    [--fact TEXT ...] [--positions JSON] [--order JSON]
    [--out PATH] [--fresh] [--trace-dir PATH]
langslice linear quick-affine ...   (unchanged)
```

`estimate`, `estimate-brain`, `--stop-after`, `--rerun-from` and
`collect-traces` are removed. A single section is a stack of one.

## What leaves the repo

- The single-slice agent (`runner.py`, `single_slice.py`, `prompts.py`,
  `tools.py`, `validators.py`, `session.py`, `_types.py`) and its tests.
- The node graph (`whole_brain/nodes.py`, `engine.py` routing, `survey.py`,
  `position.py`, `review.py` as separate agents) — replaced by the one toolbox.
- `trace_collection.py` and `collect-traces` → `../LangSlice-Training`.
- The engine service's `estimate.run` (hosts use the plugin or `linear run`).

## Open / deferred

- DeepSlice integration behind `run_deepslice` (optional extra; stub reports
  UNAVAILABLE).
- Elastix affine method (behind `--elastix`), first version may land after the
  silhouette method.
- Host adapters (ABBA hand-back of order/positions/transforms).

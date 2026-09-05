# LangSlice `linear/ — slice-position estimation`

Package guide for `src/langslice/linear/`. The repo-level `CLAUDE.md` holds the
project-wide rules; this file holds what is specific to this package. `AGENTS.md`
here is a verbatim copy — edit one, mirror to the other.

- `linear/` — slice-position estimation. Single-slice ADK agent, prompts,
  tools, validators, session/runner plumbing, trace collection, and
  `linear/whole_brain/` — the whole-brain engine: a node graph
  (`ingest → survey → fix → seed → position → transforms → review → emit`)
  over one `StackState`, with bounded loop-back edges, a JSON checkpoint
  after every node, and results in the same shape as the checkpoint.
  Flips and reorders are recorded as data on the state; user image files are
  never modified. Every agent step is seeded with the whole stack as a
  labelled sequence of per-section images (`_step_common.stack_image_parts`),
  not a thumbnail grid; the contact sheet is still written next to the
  checkpoint but is a human-facing artifact only. `survey.py` is the
  stack-triage agent (damage, hemisphere flips, order, interval breaks in one
  pass) and `fix` rebuilds the sheet and routes back for a re-check;
  `seed` runs an automatic seeder if one is installed
  (only `deepslice.py`, which is not) and otherwise passes the stack through
  unplaced; `position.py` is the agent pass that places the stack, and it is
  deliberately LEAN — no per-slice estimation worker, data-only tool payloads,
  and a prompt that carries the job, the run's facts, one factual line per
  tool and the hard constraints, nothing else. No strategies, no rules of
  thumb, no failure-mode warnings anywhere in the step: full-trace forensics
  showed the per-slice worker's estimates carried ~no signal on real data
  while eating 82% of the wall-clock, and every major benchmark failure traced
  back to advice text the harness injected. Tools report data; the agent
  reasons. Tools: `view_slices`, `fetch_atlas`, `stack_positions`
  (index, id, `position_mm`, `spacing_to_next_mm` — no nominal comparison, no
  legend), `interpolate_between` (computes, writes nothing), `set_positions`
  (writes, returns the same rows) and `submit_positions`. The gates stay,
  because a constraint stating a fact is not coaching: `submit_positions`
  refuses `interval_breaks` the agent's own written positions do not show (the
  interval there must exceed 1.5x the stack's median written spacing) and a
  submission whose positions run the wrong way along the stack's known axis
  direction (`DIRECTION_REVERSED`, a plain code check). Refusal messages state
  the numbers that caused them and stop. There used to be a second gate here —
  `atlas_structures_at`/`structure_range` landmark tools plus a
  `submit_positions` end-anchor requirement — but a benchmarked ablation (both
  test brains) found it made placement worse, not better, so it was deleted
  rather than kept behind a flag. Interpolation beyond the
  outermost fixed points steps at the interval those points imply, never at
  the nominal one; `transforms.py` proposes one in-plane
  alignment per section — the shared silhouette affine (plain code) for intact
  sections, a per-section interactive agent loop (preview → look → adjust →
  submit) for damaged ones; `review.py` is the final whole-stack consistency
  agent, gets the same lean prompt and data-only seed, and can route back to
  `position` once. `_step_common.py` holds what the
  agent steps share (`render_slice`, `view_slices`, manifest, ADK session
  loop); `render_slice` also applies the display-only fluorescence
  preprocessing (`--preprocess auto|none`, `BrainConfig.preprocess`) and, for
  the paths that SHOW a section to a model (`frame=True`), the tissue crop that
  matches the framing of fetched atlas sections — never for the affine-fitting
  path, whose coordinates the crop would move.
  `--stop-after NODE` runs one step and checkpoints; `--rerun-from
  {position,transforms,review}` rewinds an existing checkpoint's node and
  everything downstream of it (`engine.rewind_state`), notes those steps wrote
  included, then resumes — for re-benchmarking one step without re-paying for
  the agent steps ahead of it, and without seeding the fresh pass with the
  rejected one's numbers.

# `doors/agent/`: LangSlice's own agent

The ADK driver of `langslice linear run FOLDER`. The tools, their descriptions
and the job statement are in `tools/`; the job is in `job/`. This note holds
the rules the driver and its environment keep.

## One environment, not a pipeline

One run is one `Stack`, one toolbox, one job statement and one ADK session that
ends at `submit`. A single section is a stack of one. There is no node graph, no
per-step agent, no single-section agent and no sub-session: looking at the whole
stack once yields the information for every task, and splitting it re-pays for
that reading.

## Lean harness

Tools return data, no interpretation in any payload. The job statement carries
the job, the facts, the tools by name, the constraints and a short Method per
task that is on, holding facts and first principles only, never an order of
steps. Approved by Nash 2026-10-22, with his principles in his words: "more
visible structure = good, more visual similarity = good". The Method says:

- place each section on its own evidence and compare before writing; review the
  whole stack and re-check both sides of a gap before reporting a break;
- fitting algorithms refine a placement that is already close and do not
  correct a large offset;
- inspect each fit against surviving internal anatomy and keep only changes that
  improve it; undo a deformable fit that does not;
- every fit leaves the marked damage regions out;
- a fit compares one section image with one atlas image and does best when both
  show as much internal structure as possible and look alike; no fixed recipe.

With more than one deformable method in the job, none is the default, each
suits different sections, each builds on the section's current registration so
methods can be chained, and one line each says what they are
(`tools.descriptions.statement.DEFORMABLE_METHODS`, the methods' own papers'
claims; Nash approved the wording 2026-10-21). Which to try, in what order, is
the model's choice. Still out of the statement: rules of thumb, failure-mode
warnings and region names (the same text runs against every atlas, species and
plane). Each tool's description is the only place it is described.
`position.playbook` (`--position-playbook`) adds a step-by-step positioning
method; it is guidance, not enforcement.

## Not negotiable

- Gates are constraints, not coaching: a refusal states the numbers that caused
  it; a failed `submit` neither writes nor ends the run.
- One transform representation: every stored transform and fit payload carries
  `physical` (the knobs `interactive_transform` takes) next to the six
  normalized numbers, so a fit's numbers hand straight to the manual controls.
  A value left out keeps the current one.
- Corrections are data: positions, flips, rotations, angles and transforms are
  proposals on the state; the user's image files are never modified.
- The order follows the positions: `position_sections` numbers the stack by
  position after every call, so there is no ordering tool and no order gate.
- Every write is one undo step; a batch undoes as one; a landed piece of
  background work is a step of its own.
- Spend is the operator's: warnings go to the log and the trace, NEVER into a
  message or a tool reply (`spend.py`). No budget stops a run.

## Pictures and context

The OAuth lane resends the whole history on every call. A cached input token
costs about 0.13 of an uncached one, so an unchanged prefix is nearly free and
new images are most of a run's cost. Therefore:

- Images stay (`adk_plugins.WorkingSetImages`): every tool image is kept until
  500 are live, then the oldest media-bearing calls are cut in one batch to 250.
  The agent chooses later steps from earlier pictures, and a cut breaks the
  cache at the cut point, so a normal run never reaches it. A dropped picture
  can be drawn again: `zoom` takes its number.
- A tool's images ride inside its `function_call_output`; a separate user
  message would open a new turn.
- Several tool calls per model turn: one call's history cost instead of one per
  tool; the toolbox serializes them. One `prompt_cache_key` per session.
- Rows are compact (`core.stack.status.compact_rows`): null and empty fields are
  absent from every payload.
- Prefix caching is not content addressing: a byte-identical picture appended
  later is new input. Never budget a re-sent picture as a cache hit.

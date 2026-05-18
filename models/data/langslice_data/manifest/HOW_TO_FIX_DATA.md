# How to fix training data (without clobbering another agent)

This is a task-oriented walkthrough for any agent that needs to correct,
drop, or override training-data rows. **Read the "Multi-agent rules"
section first.** It's the difference between your fix sticking and
another agent's parallel session silently undoing it.

For the architecture and file layout, see
[`SHARDS.md`](./SHARDS.md). For the QC app's data contracts, see
[`../qc_app/CONTRACTS.md`](../qc_app/CONTRACTS.md).

---

## Multi-agent rules (read before doing anything)

The user runs **multiple Claude sessions in parallel.** One agent does
synthetic data, another fixes coronal positions, another QC's sagittal
brains — at the same time. Each shard is owned by whoever is editing
it; never assume your session is the only one writing.

**Hard rules.** Break any of these and you will silently destroy another
agent's work:

1. **Prefer overrides over direct shard edits.** A shard is a derived
   artifact: `sources + overrides → adapter → shard`. Each plane-agent
   owns their shards, so a direct edit works immediately and no other
   agent will silently rebuild over it — but the next time *you* rebuild
   that shard (e.g. to apply a new override, propagate an upstream
   metadata fix, or test a config change), your direct edit is gone.
   Direct edits also leave no audit trail in `data/datasets/` or the
   override file. One-off triage you're certain you'll never rebuild is
   fine; for anything you want to persist, use the override file or
   upstream source.
2. **Never write a script that touches more than one shard.** Each fix
   is scoped to one `(plane, dataset)` pair. Multi-shard scripts cannot
   coexist with parallel sessions.
3. **Never run a "rebuild everything" loop.** There isn't one anymore —
   `rebuild_shard.py` requires a `<plane>/<dataset>` argument. If you
   find yourself wanting to iterate over all shards, stop and ask the
   user.
4. **Never bypass the diff gate.** `rebuild_shard` defaults to dry-run.
   To commit, you must pass `--accept-diff N` matching the exact number
   of changed rows the dry-run reported. If `N` is bigger than you
   expected, **stop** — something else changed under you.
5. **Never resurrect an archived script.** `_local/eval/legacy/` exists
   so agents can read what the old patchers used to do. Running them
   re-introduces the multi-agent footgun this whole architecture was
   built to prevent.
6. **Snapshot before any destructive operation.** `rebuild_shard`
   already writes `.bak.YYYYMMDD_HHMMSS` automatically. If you're about
   to edit upstream sources for many subjects, copy them first.
7. **If you're an allocation agent: never run `rebuild_shard.py`.** Your
   job is split assembly, not GT curation. Write only via `allocate.py`.
8. **If you're a GT-fix agent: never run `allocate.py`.** Your job is
   position/label correctness. Allocation is a separate concern.

**What's safe to do in parallel?** Anything scoped to one shard. Two
agents working on `coronal/allen_dev_coronal` and
`sagittal/allen_ish_sagittal` can never collide — they touch different
files, different shards, different overrides.

---

## Allocation agents (eval / RLVR / SFT data assembly)

If your job is "pick brains for the eval split" or "build the RLVR
training pool" or similar, you are an **allocation agent**, not a
GT-curation agent. The two roles operate on disjoint files and never
mix.

**Your only write surface:** `data/manifest/allocations/<plane>/<split>.jsonl`.
Append-only with tombstones.

**Your tool:** `_local/eval/allocate.py`.

```
python _local/eval/allocate.py add <plane>/<split> <section_id> [...] --dataset <dataset> --added-by <your-id>
python _local/eval/allocate.py remove <plane>/<split> <section_id> [...] --removed-by <your-id>
python _local/eval/allocate.py list <plane>/<split>
```

`<plane>` is `coronal | sagittal | horizontal`. `<split>` is
`eval | rlvr | sft`. The CLI validates that each `section_id` exists in
the corresponding inventory shard and enforces two cross-split rules:

- **Section-level:** a section_id can be in at most one split for a given
  plane. Always enforced.
- **Brain-level:** a brain (identified by `(dataset, subject_id)`) can be
  in at most one split across all planes. Enforced by default; pass
  `--allow-subject-in-other-splits` only if you intend to deliberately
  split a single brain across splits.

Adds/removes are append-only; tombstones suppress prior adds on next read.

The brain-level rule means: if any section of brain X is already in
eval, you cannot add any other section of brain X to rlvr or sft. This
guarantees that a single brain's images never bleed across the eval
boundary, which was the original motivation for the layer.

**Hard rules for allocation agents:**

- **Never run `rebuild_shard.py`.** That's the GT-curation tool. Your
  allocation changes do not require a shard rebuild.
- **Never edit `data/manifest/shards/...` or `data/manifest/overrides/...`.**
- **Never edit `data/datasets/...`.** Source data is owned by plane-agents.
- **GT fixes propagate automatically.** When a coronal-agent corrects a
  position in a shard, your training data view of the same row reflects
  the correction the next time the QC app or trainer reads it. No copies,
  no drift, no allocation update needed.

**Example workflow.** The eval agent is told to pick 22 brains for the
eval split. For each section_id in those brains:

```
python _local/eval/allocate.py add coronal/eval m287:763 m287:764 m287:765 \
    --dataset deepslice_gt --added-by eval-agent-2026-05-07
```

The QC app's Eval module sees those rows in the eval split on next
reload (mtime-triggered).

**To remove a row from a split,** run `allocate.py remove`. Append a
tombstone; the row stays in the inventory (its GT data is unchanged) but
no longer appears in that split.

---

## Decision tree

```
I want to ...
├── drop a whole dataset                      → set `disabled: true` in that shard's override file
├── drop specific subjects                    → append to `excluded_subjects` in override file
├── drop specific sections                    → append to `excluded_sections` in override file
├── flip ALL of a subject's positions on
│   one axis (e.g., brain stored backward)    → append to `subject_axis_flips` in override file
├── replace ONE section's position with a
│   specific value                            → append to `section_position_overrides` in override file
├── correct an atlas label for a subject      → append to `atlas_overrides` in override file
├── fix many positions at once via upstream   → edit data/datasets/<name>/{metadata.json,
│   metadata (preferred for systemic bugs)      section_positions.json} → rebuild that shard
├── add a new dataset                         → write adapter, register shard, run partition
├── add a section to an eval/RLVR/SFT split   → python _local/eval/allocate.py add <plane>/<split> <section_id>
├── remove a section from a split             → python _local/eval/allocate.py remove <plane>/<split> <section_id>
└── check whether things still work           → python _local/eval/validate_manifest.py
```

For every fix below, the workflow is:

1. **Identify the shard.** Find which `(plane, dataset)` pair contains
   the rows you want to fix. Plane comes from `slice_axis`:
   `ap → coronal`, `ml → sagittal`, `dv → horizontal`.
2. **Make the smallest possible edit** — to the override file or to
   upstream sources, never to the shard itself.
3. **Dry-run** `python _local/eval/rebuild_shard.py <plane>/<dataset>`
   and read the diff. The number of changed rows must match what you
   intended.
4. **Commit** with `--accept-diff N` where `N` is exactly that number.
5. **Validate** cross-shard integrity:
   `python _local/eval/validate_manifest.py`.

---

## Scenario 1: Drop specific subjects

**When:** A subject's registration is unreliable, or its imaging quality
is too poor for training. You want every row for that subject removed
from the manifest.

**Parallel safety:** Touches only the one override file. Safe to run
while other agents work on any other shard.

**Steps.**

1. Identify the shard. Example: dropping `silva_ieg/cFos_Arc/343HC`.
   Silva is coronal, so the shard is `coronal/silva_ieg`.

2. Open `data/manifest/overrides/coronal/silva_ieg.json`. Append to
   `excluded_subjects`:

   ```json
   {
     "disabled": false,
     "excluded_subjects": [
       {"subject_id": "cFos_Arc/343HC", "reason": "ABBA registration noise"}
     ],
     "excluded_sections": [],
     "subject_axis_flips": [],
     "section_position_overrides": [],
     "atlas_overrides": []
   }
   ```

3. Dry-run:

   ```powershell
   python _local/eval/rebuild_shard.py coronal/silva_ieg
   ```

   Read the diff. It should report N rows removed (where N is the row
   count for that subject) and 0 changed/added.

4. Commit:

   ```powershell
   python _local/eval/rebuild_shard.py coronal/silva_ieg --accept-diff N
   ```

5. Validate:

   ```powershell
   python _local/eval/validate_manifest.py
   ```

   Confirm row totals decrease by exactly N.

---

## Scenario 2: Drop specific sections

**When:** A handful of bad sections (motion artifact, edge of slab,
anomalous ROI) need removal but the rest of the subject is fine.

**Parallel safety:** Same as above — one shard, one file.

**Steps.**

1. Identify the shard. Example: deepslice_gt has section `m287:761_..._s123`
   that's broken. deepslice_gt is coronal.

2. Edit `data/manifest/overrides/coronal/deepslice_gt.json`:

   ```json
   {
     "excluded_sections": [
       {
         "section_id": "m287:761_3165_4806_tg2576_m287_4G8_s123",
         "subject_id": "m287",
         "reason": "motion artifact, ROI off-slab"
       }
     ]
   }
   ```

3. Dry-run, commit, validate as above.

`subject_id` is informational — matching is by `section_id`.

---

## Scenario 3: Flip a subject's axis polarity

**When:** One subject was stored in inverted convention on one axis.
For example, an operator flipped the DV polarity for ebrains_tta_atlas
subject `NOP_2877`, so its `dv` positions are stored as
`(extent − correct_dv)` instead of the correct value.

This is the most common "real fix" — the row data is right, just the
sign is wrong.

**Parallel safety:** One shard, one file.

**Steps.**

1. Identify the shard. ebrains_tta_atlas spans coronal + horizontal.
   The DV-flip subjects are in horizontal (since DV is the perpendicular
   axis only for horizontal slices). So:
   `data/manifest/overrides/horizontal/ebrains_tta_atlas.json`.

2. Append to `subject_axis_flips`:

   ```json
   {
     "subject_axis_flips": [
       {
         "subject_id": "NOP_2877",
         "axis": "dv",
         "reason": "operator-flipped DV polarity"
       }
     ]
   }
   ```

3. Dry-run. The rebuild applies
   `position_mm = extent − position_mm` (where `extent` comes from the
   atlas's known voxel extent) for every row matching
   `(subject_id, axis)` AND whose `slice_axis == axis`. Verify the
   reported diff is the row count for that subject.

4. Commit and validate.

**Caveat:** the flip targets rows where `slice_axis` matches the flip
`axis`. If the subject has rows in multiple shards (different planes),
you may need to repeat in each — but only if the flip applies to that
plane's slice axis. Usually only one plane's axis is wrong.

---

## Scenario 4: Override ONE section's position to a specific value

**When:** A single section needs a corrected absolute position. Use
this when the fix doesn't fit a flip pattern — e.g., a one-off bregma
correction (BAP M634 had a specific section whose anchored position
was wrong by ~0.5mm).

**Parallel safety:** One shard, one file.

**Important:** the override value is the **final BG-ASR coordinate**,
not a delta from the current value. The override REPLACES whatever the
adapter + flips produced. This means if a `subject_axis_flip` is also
applied to this row, the section override wins (it runs after).

**Steps.**

1. Identify the shard. M634 is in `bap_horizontal`. So
   `data/manifest/overrides/horizontal/bap_horizontal.json`.

2. Append to `section_position_overrides`:

   ```json
   {
     "section_position_overrides": [
       {
         "section_id": "M634:s042",
         "axis": "ap",
         "position_mm": 5.43,
         "reason": "bregma voxel correction (2026-05-06)"
       }
     ]
   }
   ```

   The `axis` field must equal the row's `slice_axis`. Mismatch is an
   error.

3. Dry-run, commit, validate.

---

## Scenario 5: Correct an atlas label

**When:** A row's `atlas` field is wrong. Example: a P14 dev mouse
section was tagged `admba_3d_p10_mouse` instead of `admba_3d_p14_mouse`.

**Parallel safety:** One shard, one file.

**Steps.**

1. Identify the shard. nissl_p9p120 is coronal:
   `data/manifest/overrides/coronal/nissl_p9p120_mouse.json`.

2. Append to `atlas_overrides`:

   ```json
   {
     "atlas_overrides": [
       {
         "subject_id": "P14_M03",
         "atlas": "admba_3d_p14_mouse",
         "reason": "atlas label corrected from p10 to p14"
       }
     ]
   }
   ```

   `subject_id` is required; if you ever want to retag a whole shard,
   omit it (the registry treats no subject_id as a wildcard for that
   shard).

3. Dry-run, commit, validate.

`atlas_overrides` runs last in the application order, so any prior
flips/section-overrides have already been applied to `position_mm`. If
the new atlas has a different extent and your position_mm is now
implausible, the validator will catch it.

---

## Scenario 6: Fix many sections via upstream metadata edits

**When:** A systemic bug was found in the source data — e.g., the Allen
API anchor pixel coordinate was wrong, so every section in 60+ subjects
needs re-fitting. This is too big for `section_position_overrides`.

**Parallel safety:** Touches files under
`data/datasets/<dataset_name>/`. Two agents both fixing the same
dataset's metadata WILL conflict. Coordinate before doing this.

**Steps.**

1. Edit `data/datasets/<dataset_name>/{metadata.json,
   section_positions.json}` directly. Use the source-stable fixers in
   `_local/eval/` as a starting point if one exists for the dataset
   (e.g., `fix_allen_dev_positions.py`, `fix_silva_ieg_positions.py`).
   Write your fix as a script that reads the source, computes the
   correction, and writes it back.

2. Dry-run the rebuild for every shard fed by that dataset. Most
   datasets are single-shard; multi-plane datasets (Boccara,
   bap_head_3plane, deepslice_gt, timm_nissl, ebrains_tta_atlas, etc.)
   need one rebuild per plane:

   ```powershell
   python _local/eval/rebuild_shard.py coronal/<dataset>
   python _local/eval/rebuild_shard.py sagittal/<dataset>
   python _local/eval/rebuild_shard.py horizontal/<dataset>
   ```

   The dry-run for each will show the per-shard diff. Validate the diff
   row counts match what you intended.

3. Commit each shard with its own `--accept-diff N`.

4. Validate cross-shard:

   ```powershell
   python _local/eval/validate_manifest.py
   ```

**Anti-pattern:** Don't write a script that loops over all shards and
runs `rebuild_shard` for each. The diff gate is per-shard so you can
catch unexpected cross-dataset contamination. Bypassing it for
"convenience" is exactly the foot-gun this architecture replaced.

---

## Scenario 7: Add a new dataset

**When:** Bringing in a new dataset that doesn't have a shard yet.

**Parallel safety:** Adds new shard files; doesn't touch existing
shards. Safe in parallel as long as another agent isn't simultaneously
defining the same dataset name.

**Steps.**

1. Write an adapter function in `_local/eval/build_manifest.py` (the
   adapter modules still live there even though `main()` is disabled).
   The adapter yields dict records with the canonical row schema.

2. Register the new adapter in `ADAPTERS` (in build_manifest.py) AND
   add the `(plane, dataset)` mapping to `EXPLICIT_SHARDS` in
   `_local/eval/rebuild_shard.py`. If the adapter only emits one plane,
   the suffix-based fallback in `_infer_default_shards` may pick it up
   automatically; verify by reading the resulting registry.

3. Initial shard creation goes through `rebuild_shard` with the
   `--accept-diff N` matching the expected row count. There's no
   existing shard to diff against, so the diff is "N rows added,
   0 changed, 0 removed."

4. Validate.

---

## Verify your changes

Three things to check after any fix:

1. **Per-shard rebuild produces an empty diff on a re-run.** After
   `rebuild_shard ... --accept-diff N`, running it again with no flag
   should show diff == 0. If not, something in your override or
   upstream edit is non-deterministic.

2. **Cross-shard validation is clean (or no worse than before).**

   ```powershell
   python _local/eval/validate_manifest.py
   ```

   Validator reports duplicate IDs, plane/axis mismatches, missing
   critical fields, plausibility violations. If your fix introduces
   a new error, fix the override or revert.

3. **The QC app sees your change.** The QC app's `InventoryApp` reloads
   on shard mtime change. Refresh the page; the affected dataset's row
   count should match the diff you committed.

---

## Anti-patterns (don't)

- **Editing `data/manifest/shards/<plane>/<dataset>.jsonl` with a text
  editor for anything you want to persist.** It will be overwritten by
  the next rebuild. The override system exists so you don't have to.
  (One-off triage is fine if you understand the tradeoff — see Hard
  Rule 1.)

- **Writing a `fix_<thing>.py` that reads a shard, modifies rows, and
  writes back.** This was the old pattern. It is the reason the May
  2026 incidents happened. The override file is the durable
  representation.

- **Running `python _local/eval/rebuild_shard.py <plane>/<dataset>
  --accept-diff 99999` "to be safe."** The diff gate exists to make you
  read the actual number. Pass the exact value the dry-run reported.

- **Looping `for plane in coronal sagittal horizontal: rebuild_shard
  $plane/...`.** If you genuinely need multiple shards rebuilt, run
  them sequentially with separate `--accept-diff` values, eyeballing
  each diff. The cost (typing three commands) is the safety mechanism.

- **Editing `data/manifest.jsonl` directly.** It is no longer canonical.
  The shards are. If a downstream tool you maintain still reads
  `data/manifest.jsonl`, point it at `data/manifest/shards/*/*.jsonl`
  instead.

- **Resurrecting a script from `_local/eval/legacy/`.** The stub at the
  original path will tell you exactly that. If you genuinely need the
  logic, rewrite it as an upstream metadata edit + `rebuild_shard`, or
  as a per-shard override entry.

- **Bypassing the validator's findings.** If validate_manifest reports
  duplicate section_ids or atlas mismatches, those are real bugs.
  Don't tell the QC app or trainer to ignore them; fix the data.

- **Setting a `split` field on any row.** That field doesn't exist
  anymore. `rebuild_shard.py` rejects rows that have it. Splits are
  computed from the allocation layer at read time.

---

## Quick reference: where each fix lives

| Action                               | File to edit                                              | Override key                  |
|--------------------------------------|-----------------------------------------------------------|-------------------------------|
| Disable a whole dataset              | `data/manifest/overrides/<plane>/<dataset>.json`          | `disabled: true`              |
| Drop subjects                        | `data/manifest/overrides/<plane>/<dataset>.json`          | `excluded_subjects`           |
| Drop sections                        | `data/manifest/overrides/<plane>/<dataset>.json`          | `excluded_sections`           |
| Flip subject axis polarity           | `data/manifest/overrides/<plane>/<dataset>.json`          | `subject_axis_flips`          |
| Replace one section's position       | `data/manifest/overrides/<plane>/<dataset>.json`          | `section_position_overrides`  |
| Retag atlas for a subject            | `data/manifest/overrides/<plane>/<dataset>.json`          | `atlas_overrides`             |
| Fix many positions systemically      | `data/datasets/<dataset>/{metadata.json, ...}`            | (none — upstream)             |
| Add a new dataset                    | `_local/eval/build_manifest.py` + `rebuild_shard.py`       | (none — code)                 |
| Add section to eval/RLVR/SFT split   | `data/manifest/allocations/<plane>/<split>.jsonl`         | (use `allocate.py add`)       |
| Remove section from a split          | `data/manifest/allocations/<plane>/<split>.jsonl`         | (use `allocate.py remove`)    |

After every override edit:
`python _local/eval/rebuild_shard.py <plane>/<dataset>` (dry-run, then
commit with `--accept-diff N`).

Then: `python _local/eval/validate_manifest.py`.

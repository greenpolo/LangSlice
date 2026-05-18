# Manifest shards + allocations

The training-data manifest is **two architecturally disjoint layers**:

- **Shards** — GT data, one shard per (plane, dataset) pair.
- **Allocations** — split membership (eval / rlvr / sft), one file per
  (plane, split) pair.

Splits are computed at read time from the allocations layer; shard rows
**never** carry a `split` field. The two layers are written by different
agents (GT-fix vs allocation) and never mixed in one session. See
[`HOW_TO_FIX_DATA.md`](./HOW_TO_FIX_DATA.md) for the role separation and
hard rules.

## Layout

```
data/manifest/
  shards/
    coronal/<dataset>.jsonl
    sagittal/<dataset>.jsonl
    horizontal/<dataset>.jsonl
  overrides/
    coronal/<dataset>.json
    sagittal/<dataset>.json
    horizontal/<dataset>.json
  allocations/
    coronal/{eval,rlvr,sft}.jsonl
    sagittal/{eval,rlvr,sft}.jsonl
    horizontal/{eval,rlvr,sft}.jsonl
  manifest_summary.md           ← written only by validate_manifest.py --report
```

`<plane>` is one of `coronal`, `sagittal`, `horizontal`. The plane is derived
from the row's `slice_axis` field: `ap → coronal`, `ml → sagittal`,
`dv → horizontal`.

## Why per-shard

Two recent incidents lost ~17k rows of curation when one agent's
"rebuild everything" run clobbered another agent's parallel work. Sharding
scopes the blast radius of any rebuild to exactly one (plane, dataset)
file. Two agents working on different shards in parallel sessions cannot
clobber each other.

Plane is the top-level grouping because the QC app navigates by plane
(coronal/sagittal/horizontal pills) and most agent fixes are
plane-specific.

## Why a separate allocations layer

GT correctness and split assembly are different concerns operating on
different cadences:

- GT-fix agents change `position_mm`, drop bad subjects, flip axes.
  Their writes go through `rebuild_shard.py`.
- Allocation agents pick which sections become eval / rlvr / sft. Their
  writes go through `allocate.py` and never touch shards.

Storing `split` on the row coupled the two — a GT-fix rebuild could
silently lose split assignments, and a split rebuild could silently
revert GT fixes. Splitting them removes that coupling entirely:
`rebuild_shard.py` actively refuses to write rows with a `split` key
(defensive guard), and `allocate.py` never opens a shard.

## How to fix GT data (GT-fix agent)

1. **Position fix on a real section** (e.g., correcting an Allen
   anchoring): edit the upstream source under
   `data/datasets/<name>/{metadata.json, section_positions.json}`. Then
   rebuild that one shard.
2. **Drop a subject or section**: append to
   `data/manifest/overrides/<plane>/<dataset>.json` under
   `excluded_subjects` or `excluded_sections`. Then rebuild.
3. **Flip a subject's polarity on one axis**: append to
   `subject_axis_flips`. Then rebuild.
4. **Override one section's position to a specific value**: append to
   `section_position_overrides`. The value is the FINAL BG-ASR
   coordinate, not a delta. Then rebuild.
5. **Correct an atlas label**: append to `atlas_overrides`. Then rebuild.

**Prefer overrides over direct shard edits.** A shard is a derived
artifact (`sources + overrides → adapter → shard`). Each plane-agent
owns their shard files, so a direct edit works immediately — but the
next time anyone rebuilds that shard (including you, to apply a future
override or upstream fix), the direct edit is gone. Direct edits also
leave no audit trail. One-off triage with no future rebuild is fine; for
anything you want to persist, edit the override file or upstream source.

**GT-fix agents never run `allocate.py`.** Split assembly is a separate
concern. See `HOW_TO_FIX_DATA.md` Hard Rule 8.

## How to assemble splits (allocation agent)

Allocation agents assemble `eval` / `rlvr` / `sft` membership without
touching GT. The only legal write surface is
`data/manifest/allocations/<plane>/<split>.jsonl` and the only legal
write tool is `_local/eval/allocate.py`.

```
python _local/eval/allocate.py add <plane>/<split> <section_id> --dataset <name> --added-by <agent_id>
python _local/eval/allocate.py remove <plane>/<split> <section_id> --removed-by <agent_id>
python _local/eval/allocate.py list <plane>/<split>
```

The CLI validates that each `section_id` exists in the corresponding
inventory shard and that it isn't already in another split for the same
plane (a section can be in at most one split per plane). Adds and
removes are append-only; tombstones suppress prior adds on next read.

**Allocation agents never run `rebuild_shard.py`, never edit shards,
never edit overrides.** GT correctness is a separate concern. See
`HOW_TO_FIX_DATA.md` Hard Rule 7.

## Per-shard overrides schema

`data/manifest/overrides/<plane>/<dataset>.json`:

```json
{
  "disabled": false,
  "excluded_subjects": [
    {"subject_id": "...", "reason": "..."}
  ],
  "excluded_sections": [
    {"section_id": "...", "subject_id": "...", "reason": "..."}
  ],
  "subject_axis_flips": [
    {"subject_id": "...", "axis": "ap|ml|dv", "reason": "..."}
  ],
  "section_position_overrides": [
    {"section_id": "...", "axis": "ap|ml|dv", "position_mm": 5.43, "reason": "..."}
  ],
  "atlas_overrides": [
    {"subject_id": "...", "atlas": "...", "reason": "..."}
  ]
}
```

All keys optional. Application order is fixed:
`excluded_subjects → excluded_sections → subject_axis_flips →
section_position_overrides → atlas_overrides`.

`section_position_overrides[*].axis` must equal the row's `slice_axis`;
mismatch is an error. The override value REPLACES whatever the adapter +
flips produced, so it always wins.

## Allocations schema

`data/manifest/allocations/<plane>/<split>.jsonl` is per-line JSONL.
Append-only. Each line is either an add or a tombstone:

```json
{
  "section_id": "<id matching some row in shards/<plane>/<dataset>.jsonl>",
  "dataset": "<dataset name>",
  "added_by": "<allocation agent id>",
  "added_at": "<ISO-8601 timestamp>"
}
```

```json
{
  "section_id": "<id>",
  "tombstone": true,
  "removed_by": "<id>",
  "removed_at": "<ISO-8601>"
}
```

Read via `_local/eval/allocations.py:load_allocation()`, which applies
tombstones in order. Invariant: a `section_id` appears in **at most
one** split for a given plane. The validator enforces this and rejects
allocations referencing rows absent from any shard for that plane.

## Commands

### Rebuild one shard

```
python _local/eval/rebuild_shard.py <plane>/<dataset>
```

Default: dry-run. Prints diff vs the current shard, exits 1 if non-empty.

To commit:

```
python _local/eval/rebuild_shard.py <plane>/<dataset> --accept-diff N
```

Where `N` is the exact number of changed/added/removed rows reported by
the dry-run. The script refuses to write unless N matches.

`--accept-diff-file PATH` accepts a file listing the expected
`(section_id, position_mm)` tuples for line-by-line confirmation when
the diff is non-trivial.

`--out PATH` writes to a non-canonical location for what-if analysis;
PATH must NOT be inside `data/manifest/shards/` or
`data/manifest/overrides/`.

### Validate everything

Read-only. Cannot write any shard.

```
python _local/eval/validate_manifest.py            # exit 0 on clean, 1 on errors
python _local/eval/validate_manifest.py --report   # also writes data/manifest/manifest_summary.md
python _local/eval/validate_manifest.py --check-images  # also verifies image files exist (slow)
```

Checks: duplicate section_id / image_path, plane-axis consistency,
dataset/filename match, atlas resolution, position plausibility,
non-empty critical fields.

### One-shot migration (legacy → shards)

```
python _local/eval/partition_manifest.py
```

Reads `data/manifest.jsonl` and `data/manifest_overrides.json`, writes
the sharded layout. Refuses if `data/manifest/shards/` already exists.
Use `--dry-run` to preview.

## What changed for old `build_manifest.py` users

The old script ran every adapter and overwrote the whole manifest in one
go. There is no "rebuild everything" command in the new shape. To
refresh ten shards, run the per-shard command ten times. This is by
design — the absence of a global rebuild eliminates the foot-gun that
caused the May 2026 incidents.

`build_manifest.py` is left in place as an archived stub for one
release; it errors out with a pointer to `rebuild_shard.py`.

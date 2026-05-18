# QC App File Contracts

The QC app at `_local/qc_app/` reads four module-specific data sources. Other Claude
sessions assembling SFT, RLVR, Eval, or synthetic datasets should write into the
paths described below — no app code change required, manifests are reloaded on
mtime change.

| Module    | Reads from                                                        | Refresh trigger              |
|-----------|-------------------------------------------------------------------|------------------------------|
| Inventory | `data/manifest/shards/<plane>/<dataset>.jsonl` (falls back to `data/manifest.jsonl` if sharded dir is missing) | mtime change                 |
| SFT       | `_local/trace_collection/*_trace_manifest_qc.jsonl` + `runs*/**/results.jsonl` | mtime change on either       |
| RLVR      | `data/manifest/shards/<plane>/<dataset>.jsonl` rows with `split == "rlvr"` + `_local/rlvr_progress/` | mtime; live poll for progress |
| Eval      | `data/manifest/shards/<plane>/<dataset>.jsonl` rows with `split == "eval"` | mtime change                 |
| Synthetic | `_local/synth_data/manifest.jsonl` + `images/{seed:08d}.png`      | mtime change                 |

## Inventory contract — `data/manifest/`

The canonical inventory is a folder of plane-first shards plus a separate
allocation layer. Each shard is one `(plane, dataset)` pair; allocations
are per-`(plane, split)`:

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
```

The QC app reads BOTH layers and computes a `_split` field on each row at
load time via `compute_split_for(plane, section_id)` from
`_local/eval/allocations.py`. mtime watches both `shards/` and
`allocations/` so any change in either layer triggers reload.

Shard rows are per-line JSON with at minimum:

```json
{
  "dataset": "<source dataset name>",
  "subject_id": "<brain id>",
  "section_id": "<section id>",
  "image_path": "<repo-relative path to histology png>",
  "atlas": "<brainglobe atlas name>",
  "orientation": "coronal|sagittal|horizontal",
  "position_mm": 0.0,
  "species": "<species>",
  "imaging": "<modality>",
  "staining": "<staining>",
  "registration_source": "<provenance>",
  "is_hemisphere": false,
  "exclude_from_training": false
}
```

Rows do **not** carry a `split` field anymore. Whether a row belongs to
the eval / rlvr / sft split is determined by allocation membership, looked
up at read time. `rebuild_shard.py` actively refuses to write rows with a
`split` key (defensive guard against regressions).

### How to fix data

Prefer overrides over direct shard edits. Each plane-agent owns their shard
files, so a direct edit works immediately — but the next time anyone rebuilds
that shard, the edit is gone, and there's no audit trail. One-off triage you'll
never rebuild is fine; anything you want to persist should go through the
override file or upstream source.

**Never write a script that touches more than one shard.**

Make persistent fixes through upstream data or per-shard overrides, then
rebuild exactly one shard:

1. To drop subjects or sections: append to
   `data/manifest/overrides/<plane>/<dataset>.json` under
   `excluded_subjects` or `excluded_sections`.
2. To flip a subject's polarity on one axis: append to `subject_axis_flips`.
3. To replace one section's position: append to
   `section_position_overrides`. The value is the final BG-ASR coordinate, not a
   delta.
4. To correct atlas labels: append to `atlas_overrides`.
5. Rebuild that shard. The bare command runs as a dry-run (exits 1 on
   any diff); pass `--accept-diff N` matching the dry-run's row count to
   commit:

```
python _local/eval/rebuild_shard.py <plane>/<dataset>                  # dry-run
python _local/eval/rebuild_shard.py <plane>/<dataset> --accept-diff N  # commit
```

See `_local/eval/SHARDS.md` for the full architecture, override schema, rebuild
flags, validation commands, and migration notes.

## Allocation layer contract — `data/manifest/allocations/`

Per-line JSONL at `data/manifest/allocations/<plane>/<split>.jsonl`. One
file per `(plane, split)` pair (3 planes × 3 splits = 9 files). Each line:

```json
{
  "section_id": "<id matching some row in shards/<plane>/<dataset>.jsonl>",
  "dataset": "<dataset name>",
  "added_by": "<allocation agent id>",
  "added_at": "<ISO-8601 timestamp>"
}
```

Tombstone shape (removes the section from the split on next load):

```json
{
  "section_id": "<id>",
  "tombstone": true,
  "removed_by": "<id>",
  "removed_at": "<ISO-8601>"
}
```

Append-only. Read via `_local/eval/allocations.py:load_allocation()` which
applies tombstones. Invariant: a section_id may appear in **at most one**
split for a given plane. The validator enforces this and also rejects
allocations that reference section_ids absent from any shard for that plane.

Allocation agents (the eval / RLVR / SFT data-assembly agents) write to
this layer **only** via `_local/eval/allocate.py`:

```
python _local/eval/allocate.py add coronal/eval <section_id> --dataset <dataset> --added-by <agent>
python _local/eval/allocate.py remove coronal/eval <section_id> --removed-by <agent>
python _local/eval/allocate.py list coronal/eval
```

Allocation agents must NEVER touch shard files, override files, or source
data. GT-fix agents must NEVER touch allocation files. The two layers are
disjoint by design — GT-position fixes propagate to whatever splits a row
belongs to automatically because `_split` is computed at read time, not
copied.

## SFT contract — `_local/trace_collection/`

Two file types are joined by `id`:

### Trace manifest — `_local/trace_collection/*_trace_manifest_qc.jsonl`

Per-line JSON. Single-slice example:

```json
{
  "id": "<unique trace id>",
  "kind": "single",
  "image": "<repo-relative path>",
  "position_mm": 0.0,
  "atlas": "allen_mouse_25um",
  "plane": "coronal",
  "metadata": {
    "source_dataset": "<dataset>",
    "subject_id": "<brain id>",
    "section_id": "<section id>"
  }
}
```

Group example (multi-slice):

```json
{
  "id": "<unique trace id>",
  "kind": "group",
  "images": ["<path1>", "<path2>"],
  "positions_mm": [0.5, 1.2],
  "atlas": "allen_mouse_25um",
  "plane": "coronal",
  "metadata": {
    "source_dataset": "<dataset>",
    "subject_id": "<brain id>",
    "section_ids": ["<sec1>", "<sec2>"]
  }
}
```

The QC app discovers any file matching the glob
`_local/trace_collection/*_trace_manifest_qc.jsonl`. Multiple parallel
collection campaigns can drop their own `<campaign>_trace_manifest_qc.jsonl`
without coordination.

### Results — `_local/trace_collection/runs*/**/results.jsonl`

Per-line JSON, joined to manifest by `id`:

```json
{
  "id": "<matches manifest id>",
  "submitted_positions_mm": [0.5],
  "truth_positions_mm": [0.4],
  "category": {
    "label": "accepted|rejected|...",
    "accepted": true,
    "max_error_mm": 0.1
  },
  "sft_quality": {"accepted_for_sft": true},
  "estimated_cost_usd": 0.05
}
```

Legacy field names accepted: `estimated_positions_mm`, `estimated_position_mm`.

The QC app auto-discovers every `results.jsonl` under directories matching
`_local/trace_collection/runs*/`. Drop a new run, refresh the page — it shows up.

### What the SFT sub-tab renders

Each trace card shows three thumbnails:
1. The real histology slice (`image` / `images[0]`).
2. The GT atlas slice at `position_mm` / `positions_mm[0]`.
3. The Gemini-estimated atlas slice at `submitted_positions_mm[0]` (omitted if
   no results.jsonl entry exists yet — manifest-only traces still render with
   just the first two thumbnails).

Cards are organized by `metadata.source_dataset`.

## RLVR contract — `data/manifest/` + `_local/rlvr_progress/`

### Inputs
Rows in the sharded inventory whose `section_id` appears in
`data/manifest/allocations/<plane>/rlvr.jsonl`. The QC app stamps `_split`
at load time via lookup; same effect as the old `split == "rlvr"` filter.
GT data (positions, atlas, etc.) comes from the shard rows. The
allocation files only carry membership references.

### Live trainer progress
The trainer writes three files into `_local/rlvr_progress/` (configurable via
`--rlvr-progress-dir`):

- `consumed.jsonl` (append-only): one JSON per training step.
  `{"step": int, "ts": float, "ids": ["<row id>", ...], "kind": "<phase>", "phase": "<phase>"}`
- `queue.json` (atomic rewrites): the in-flight queue snapshot.
  `{"step": int, "ts": float, "queued": [{"ids": [...], "kind": "..."}, ...]}`
- `run.json` (optional): run metadata.
  `{"config": "...", "started_at": float, "target_steps": int, "target_samples": int, "model": "...", "notes": "..."}`

Row ids in `consumed.jsonl[].ids` and `queue.json.queued[*].ids` must match the
inventory row ids the QC app uses: `f"{dataset}/{section_id}"`.

The QC app polls these files every 2 s while the RLVR sub-tab is visible.

## Eval contract — `data/manifest/`

Rows in the sharded inventory whose `section_id` appears in
`data/manifest/allocations/<plane>/eval.jsonl`. Same lookup pattern as RLVR;
GT data lives in the shard, membership lives in the allocation file.
Expected size: ~22 brains. The Eval sub-tab is a flat brain list (no Sources
rollup) since the dataset is small.

## Synthetic contract — `_local/synth_data/`

Default output of `python -m synth_dataset write --out _local/synth_data --n N`.

### `_local/synth_data/manifest.jsonl`
Per-line JSON, generated by `models/langslice-gemma-4/data/synth_dataset.py`:

```json
{
  "atlas_name": "allen_mouse_25um",
  "atlas_version": "...",
  "plane": "coronal",
  "position_mm": 0.0,
  "yaw_deg": 0.0,
  "pitch_deg": 0.0,
  "roll_deg": 0.0,
  "modality": "dapi|nissl|brightfield|fluorescence|ish",
  "mode": "<modality-specific variant>",
  "counterstain": "hematoxylin|auto|null",
  "damage_intensity": "light|medium|heavy",
  "apply_damage": true,
  "seed": 12345678,
  "shape": [H, W, C],
  "generator_version": "..."
}
```

### Images
`_local/synth_data/images/{seed:08d}.png` — uint8 RGB, paired by `seed`.

### What the Synthetic sub-tab renders
Source cards group by `(modality, atlas_name, plane)`. Drilling in shows a grid
of seed-keyed thumbnails paired with their atlas slices.

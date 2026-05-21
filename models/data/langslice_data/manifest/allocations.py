"""Allocation layer for the LangSlice evaluation manifest.

The allocation layer separates *which rows belong to which split* (eval/rlvr/sft)
from the inventory shards themselves.  Inventory shards contain ground-truth
positions; allocation files record which section_ids have been assigned to which
training/eval split.

Physical layout::

    data/manifest/allocations/
        coronal/eval.jsonl
        coronal/rlvr.jsonl
        coronal/sft.jsonl
        sagittal/eval.jsonl
        sagittal/rlvr.jsonl
        sagittal/sft.jsonl
        horizontal/eval.jsonl
        horizontal/rlvr.jsonl
        horizontal/sft.jsonl

Each file is JSONL; each line is one of:

    Active assignment::
        {"section_id": str, "dataset": str, "added_by": str, "added_at": ISO-8601}

    Tombstone (logical delete)::
        {"section_id": str, "tombstone": true, "removed_by": str, "removed_at": ISO-8601}

Files are append-only.  Tombstones shadow earlier active assignments.
``load_allocation`` applies tombstones at read time; callers receive only the
active (non-tombstoned) assignments.

Allocation agents (eval/rlvr/sft data assembly) write here exclusively.
GT-curation agents (rebuild_shard.py) write to inventory shards exclusively.
Neither crosses into the other's layer.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[4]
from .paths import resolve_manifest_root  # noqa: E402

ALLOCATIONS_ROOT: Path = resolve_manifest_root(_REPO_ROOT) / "allocations"

PLANES: tuple[str, ...] = ("coronal", "sagittal", "horizontal")
SPLITS: tuple[str, ...] = ("eval", "rlvr", "sft")


def allocation_path(plane: str, split: str) -> Path:
    """Return the canonical path for a (plane, split) allocation file.

    Raises ``ValueError`` if *plane* is not in ``PLANES`` or *split* is not in
    ``SPLITS``.
    """
    if plane not in PLANES:
        raise ValueError(f"bad plane {plane!r}; expected one of {', '.join(PLANES)}")
    if split not in SPLITS:
        raise ValueError(f"bad split {split!r}; expected one of {', '.join(SPLITS)}")
    return ALLOCATIONS_ROOT / plane / f"{split}.jsonl"


def load_allocation(plane: str, split: str) -> dict[str, dict]:
    """Load all active (non-tombstoned) entries for *(plane, split)*.

    Returns a dict mapping ``section_id -> entry dict``.  Lines are processed
    in file order; a tombstone entry removes the section_id from the result.
    Missing allocation file returns an empty dict (not an error).
    """
    path = allocation_path(plane, split)
    if not path.exists():
        return {}
    result: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(entry, dict):
                raise ValueError(
                    f"{path}:{lineno}: expected JSON object, got {type(entry).__name__}"
                )
            section_id = entry.get("section_id")
            if not section_id or not isinstance(section_id, str):
                raise ValueError(f"{path}:{lineno}: missing or invalid section_id")
            if entry.get("tombstone") is True:
                result.pop(section_id, None)
            else:
                result[section_id] = entry
    return result


def add_to_allocation(
    plane: str,
    split: str,
    section_id: str,
    dataset: str,
    added_by: str,
) -> None:
    """Append an active assignment entry to the allocation file.

    The allocation file is created (including parent directories) if it does
    not exist.  The append is a single JSON line; on POSIX and Windows, appends
    of small records (< 4 KB) to an ``O_APPEND``-opened file are atomic.
    """
    path = allocation_path(plane, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "section_id": section_id,
        "dataset": dataset,
        "added_by": added_by,
        "added_at": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def remove_from_allocation(
    plane: str,
    split: str,
    section_id: str,
    removed_by: str,
) -> None:
    """Append a tombstone entry to the allocation file.

    No deletion occurs; a later ``load_allocation`` call will exclude the
    tombstoned section_id.
    """
    path = allocation_path(plane, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "section_id": section_id,
        "tombstone": True,
        "removed_by": removed_by,
        "removed_at": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def compute_split_for(plane: str, section_id: str) -> str | None:
    """Return the split name for *section_id* in *plane*, or ``None``.

    Scans all three split files for the given plane.  If the section_id
    appears as active in more than one split, raises ``ValueError`` (this
    signals a data-integrity bug; ``validate_manifest.py`` should also catch
    it).
    """
    found: list[str] = []
    for split in SPLITS:
        entries = load_allocation(plane, split)
        if section_id in entries:
            found.append(split)
    if len(found) > 1:
        raise ValueError(
            f"section_id {section_id!r} is active in multiple splits for plane "
            f"{plane!r}: {', '.join(found)}"
        )
    return found[0] if found else None


def iter_all_allocations() -> Iterator[tuple[str, str, str, dict]]:
    """Yield ``(plane, split, section_id, entry)`` for every active allocation.

    Iterates all 9 (plane, split) pairs in a deterministic order.  Tombstoned
    entries are excluded.
    """
    for plane in PLANES:
        for split in SPLITS:
            for section_id, entry in load_allocation(plane, split).items():
                yield plane, split, section_id, entry


def _main() -> int:  # pragma: no cover
    """Minimal smoke-test: list active allocations for a given plane/split."""
    import argparse

    parser = argparse.ArgumentParser(description="Dump active allocations for a plane/split pair.")
    parser.add_argument("plane_split", help="<plane>/<split>")
    args = parser.parse_args()
    parts = args.plane_split.split("/")
    if len(parts) != 2:
        print(f"bad argument {args.plane_split!r}; expected <plane>/<split>", file=sys.stderr)
        return 1
    plane, split = parts
    try:
        entries = load_allocation(plane, split)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for section_id, entry in sorted(entries.items()):
        print(json.dumps({section_id: entry}, ensure_ascii=False))
    print(f"total: {len(entries)} active entries", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

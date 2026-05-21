"""CLI for allocation agents: add, remove, or list section_ids in a split.

An allocation agent assigns rows to a training/eval split without ever touching
inventory shards.  This script is the only tool an allocation agent should need.

Usage::

    # Assign one or more section_ids to a split
    python _local/eval/allocate.py add <plane>/<split> <section_id> [<section_id> ...] \\
        --dataset <dataset> --added-by <agent>

    # Remove section_ids (appends tombstones)
    python _local/eval/allocate.py remove <plane>/<split> <section_id> [<section_id> ...] \\
        --removed-by <agent>

    # List active assignments
    python _local/eval/allocate.py list <plane>/<split>

The ``add`` command validates that every section_id exists in the inventory
shard ``data/manifest/shards/<plane>/<dataset>.jsonl`` before appending.  It
also enforces two cross-split rules:

  * *Section-level:* a section_id can be in at most one split for a given
    plane.  Always enforced.
  * *Brain-level:* a brain (identified by ``(dataset, subject_id)``) can be in
    at most one split across all planes.  No section of an eval brain may
    bleed into rlvr, and vice versa.  Enforced by default; pass
    ``--allow-subject-in-other-splits`` only when you genuinely intend to
    split a brain across splits.

Writes are restricted to ``data/manifest/allocations/<plane>/<split>.jsonl``.
Any other write target is refused at startup.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[4]

from . import allocations as alloc_lib  # noqa: E402
from .paths import resolve_manifest_root  # noqa: E402

SHARDS_ROOT = resolve_manifest_root(_REPO_ROOT) / "shards"
ALLOCATIONS_CANONICAL_ROOT = alloc_lib.ALLOCATIONS_ROOT.resolve()


def _resolve(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (_REPO_ROOT / path).resolve()


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False


def _assert_write_allowed(plane: str, split: str) -> None:
    """Hard-fail if the computed write path is outside the allocations tree."""
    target = alloc_lib.allocation_path(plane, split).resolve()
    if not _is_inside(target, ALLOCATIONS_CANONICAL_ROOT):
        raise SystemExit(
            f"FATAL: computed write target {target} is outside the canonical "
            f"allocations root {ALLOCATIONS_CANONICAL_ROOT}.  Aborting."
        )


def _load_shard_section_ids(plane: str, dataset: str) -> set[str]:
    """Return all section_ids present in a shard (or raise if shard is missing)."""
    ids: set[str] = set()
    for sid, _subj in _iter_shard_rows(plane, dataset):
        ids.add(sid)
    return ids


def _load_shard_subject_lookup(plane: str, dataset: str) -> dict[str, str]:
    """Return ``{section_id: subject_id}`` for a shard, omitting rows that
    lack a subject_id (some test fixtures).  Raise if shard missing."""
    out: dict[str, str] = {}
    for sid, subj in _iter_shard_rows(plane, dataset):
        if subj is not None:
            out[sid] = subj
    return out


def _iter_shard_rows(plane: str, dataset: str):
    """Yield ``(section_id, subject_id_or_None)`` for each row in a shard."""
    path = SHARDS_ROOT / plane / f"{dataset}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"shard not found: {path}\n  (looked up plane={plane!r}, dataset={dataset!r})"
        )
    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            sid = row.get("section_id")
            if sid is None:
                continue
            subj = row.get("subject_id")
            yield str(sid), (str(subj) if subj is not None else None)


def _find_subject_conflicts(
    target_plane: str,
    target_split: str,
    target_dataset: str,
    new_section_ids: list[str],
    target_subject_lookup: dict[str, str],
) -> list[tuple[str, str, str, str, str]]:
    """Return brain-level conflicts that would be created by this add.

    A conflict is a tuple ``(new_section_id, subject_id, conflicting_plane,
    conflicting_split, existing_section_id)`` indicating that a brain
    ``(target_dataset, subject_id)`` already has at least one section
    assigned to a different ``(plane, split)``.  Cross-plane is included
    because some allocation pipelines may key on subject_id alone, and
    coronal-eval bleeding into sagittal-rlvr would still violate the rule.
    """
    new_subjects: dict[str, str] = {}
    for sid in new_section_ids:
        subj = target_subject_lookup.get(sid)
        if subj is None:
            continue
        new_subjects[sid] = subj
    if not new_subjects:
        return []
    new_brains = {(target_dataset, subj) for subj in new_subjects.values()}

    other_brains: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    shard_cache: dict[tuple[str, str], dict[str, str]] = {}

    for plane in alloc_lib.PLANES:
        for split in alloc_lib.SPLITS:
            if plane == target_plane and split == target_split:
                continue
            try:
                entries = alloc_lib.load_allocation(plane, split)
            except (FileNotFoundError, ValueError):
                continue
            for ex_sid, entry in entries.items():
                ds = entry.get("dataset")
                if not ds:
                    continue
                key = (plane, ds)
                if key not in shard_cache:
                    try:
                        shard_cache[key] = _load_shard_subject_lookup(plane, ds)
                    except (FileNotFoundError, ValueError):
                        shard_cache[key] = {}
                ex_subj = shard_cache[key].get(str(ex_sid))
                if ex_subj is None:
                    continue
                brain_key = (ds, ex_subj)
                if brain_key in new_brains:
                    other_brains.setdefault(brain_key, []).append((plane, split, str(ex_sid)))

    conflicts: list[tuple[str, str, str, str, str]] = []
    for new_sid, new_subj in new_subjects.items():
        brain_key = (target_dataset, new_subj)
        for ex_plane, ex_split, ex_sid in other_brains.get(brain_key, []):
            conflicts.append((new_sid, new_subj, ex_plane, ex_split, ex_sid))
    return conflicts


def _parse_plane_split(value: str) -> tuple[str, str]:
    parts = value.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError(f"expected <plane>/<split>, got {value!r}")
    plane, split = parts
    if plane not in alloc_lib.PLANES:
        raise argparse.ArgumentTypeError(
            f"bad plane {plane!r}; expected one of {', '.join(alloc_lib.PLANES)}"
        )
    if split not in alloc_lib.SPLITS:
        raise argparse.ArgumentTypeError(
            f"bad split {split!r}; expected one of {', '.join(alloc_lib.SPLITS)}"
        )
    return plane, split


def cmd_add(args: argparse.Namespace) -> int:
    plane, split = args.plane_split
    _assert_write_allowed(plane, split)

    # Walk the shard once to build BOTH the existence set (for section_id
    # validation) and the {section_id: subject_id} lookup (for the
    # brain-level cross-split check below).  Some test fixtures omit
    # subject_id; those rows still validate existence but skip the brain
    # check.
    shard_ids: set[str] = set()
    subject_lookup: dict[str, str] = {}
    try:
        for sid, subj in _iter_shard_rows(plane, args.dataset):
            shard_ids.add(sid)
            if subj is not None:
                subject_lookup[sid] = subj
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    missing = [sid for sid in args.section_ids if sid not in shard_ids]
    if missing:
        print(
            f"ERROR: the following section_ids do not exist in "
            f"data/manifest/shards/{plane}/{args.dataset}.jsonl:",
            file=sys.stderr,
        )
        for sid in missing:
            print(f"  {sid}", file=sys.stderr)
        return 1

    # Section-level cross-split check: a section_id can be in at most one
    # split per plane.  Always enforced.
    conflicts: list[tuple[str, str]] = []
    for sid in args.section_ids:
        existing_split = alloc_lib.compute_split_for(plane, sid)
        if existing_split is not None and existing_split != split:
            conflicts.append((sid, existing_split))

    if conflicts:
        print(
            f"ERROR: the following section_ids are already assigned to another "
            f"split for plane {plane!r}:",
            file=sys.stderr,
        )
        for sid, existing in conflicts:
            print(f"  {sid} -> {existing}", file=sys.stderr)
        print(
            "Remove them from the other split first using the 'remove' command.",
            file=sys.stderr,
        )
        return 1

    # Brain-level cross-split check: a (dataset, subject_id) brain can be
    # in at most one split across all planes.  Enforced by default; the
    # --allow-subject-in-other-splits flag bypasses it when an operator
    # genuinely intends to split a brain across splits.
    if not args.allow_subject_in_other_splits:
        brain_conflicts = _find_subject_conflicts(
            plane, split, args.dataset, args.section_ids, subject_lookup
        )
        if brain_conflicts:
            seen: set[tuple[str, str, str, str]] = set()
            print(
                "ERROR: the following sections belong to brains already "
                "allocated to another split:",
                file=sys.stderr,
            )
            for new_sid, subj, ex_plane, ex_split, ex_sid in brain_conflicts:
                key = (subj, ex_plane, ex_split, args.dataset)
                if key in seen:
                    continue
                seen.add(key)
                print(
                    f"  {new_sid}  ({args.dataset}/{subj})  -> brain already in "
                    f"{ex_plane}/{ex_split} via {ex_sid}",
                    file=sys.stderr,
                )
            print(
                "Pass --allow-subject-in-other-splits if splitting this brain "
                "across splits is intentional, otherwise filter the input list.",
                file=sys.stderr,
            )
            return 1

    # Append allocation entries
    for sid in args.section_ids:
        alloc_lib.add_to_allocation(plane, split, sid, dataset=args.dataset, added_by=args.added_by)
        print(f"added: {plane}/{split} <- {sid}")

    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    plane, split = args.plane_split
    _assert_write_allowed(plane, split)

    # Warn about section_ids that are not currently active in this split
    current = alloc_lib.load_allocation(plane, split)
    not_active = [sid for sid in args.section_ids if sid not in current]
    if not_active:
        print(
            f"WARNING: the following section_ids are not currently active in "
            f"{plane}/{split} (tombstone will still be appended):",
            file=sys.stderr,
        )
        for sid in not_active:
            print(f"  {sid}", file=sys.stderr)

    for sid in args.section_ids:
        alloc_lib.remove_from_allocation(plane, split, sid, removed_by=args.removed_by)
        print(f"removed: {plane}/{split} -x {sid}")

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    plane, split = args.plane_split
    try:
        entries = alloc_lib.load_allocation(plane, split)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not entries:
        print(f"{plane}/{split}: (empty)")
        return 0

    print(f"{plane}/{split}: {len(entries)} active allocations")
    for section_id in sorted(entries):
        entry = entries[section_id]
        dataset = entry.get("dataset", "?")
        added_by = entry.get("added_by", "?")
        print(f"  {section_id}  dataset={dataset}  added_by={added_by}")
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # add
    add_parser = subparsers.add_parser("add", help="Add section_ids to a split allocation.")
    add_parser.add_argument("plane_split", type=_parse_plane_split, metavar="<plane>/<split>")
    add_parser.add_argument("section_ids", nargs="+", metavar="section_id")
    add_parser.add_argument(
        "--dataset", required=True, help="Dataset name (must match a shard stem)."
    )
    add_parser.add_argument("--added-by", required=True, dest="added_by", help="Agent identifier.")
    add_parser.add_argument(
        "--allow-subject-in-other-splits",
        action="store_true",
        dest="allow_subject_in_other_splits",
        help=(
            "Bypass the brain-level cross-split check.  Use only when you "
            "intentionally want to split a brain across eval/rlvr/sft."
        ),
    )

    # remove
    remove_parser = subparsers.add_parser(
        "remove", help="Remove section_ids from a split (tombstone)."
    )
    remove_parser.add_argument("plane_split", type=_parse_plane_split, metavar="<plane>/<split>")
    remove_parser.add_argument("section_ids", nargs="+", metavar="section_id")
    remove_parser.add_argument(
        "--removed-by", required=True, dest="removed_by", help="Agent identifier."
    )

    # list
    list_parser = subparsers.add_parser("list", help="List active allocations for a plane/split.")
    list_parser.add_argument("plane_split", type=_parse_plane_split, metavar="<plane>/<split>")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.command == "add":
            return cmd_add(args)
        elif args.command == "remove":
            return cmd_remove(args)
        elif args.command == "list":
            return cmd_list(args)
        else:
            print(f"unknown command: {args.command}", file=sys.stderr)
            return 1
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

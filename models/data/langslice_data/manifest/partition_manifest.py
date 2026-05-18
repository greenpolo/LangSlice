"""Partition the flat evaluation manifest into plane-first manifest shards.

This is a one-shot migration helper. It copies the existing flat manifest and
global override file into the new sharded layout without rebuilding rows from
source adapters or importing build_manifest.py.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PLANE_BY_AXIS = {
    "ap": "coronal",
    "ml": "sagittal",
    "dv": "horizontal",
}
PLANES = ("coronal", "sagittal", "horizontal")
DATASET_RE = re.compile(r"[A-Za-z0-9_]+")
SHARD_FILENAME_RE = re.compile(r"[A-Za-z0-9_]+\.jsonl")
AXES = frozenset(PLANE_BY_AXIS)


class PartitionError(RuntimeError):
    """Raised for user-correctable migration input or output errors."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Partition data/manifest.jsonl into data/manifest shards.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest.jsonl").resolve(),
        help="Input flat JSONL manifest. Defaults to data/manifest.jsonl.",
    )
    parser.add_argument(
        "--overrides",
        type=Path,
        default=Path("data/manifest_overrides.json").resolve(),
        help="Input global override JSON. Defaults to data/manifest_overrides.json.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/manifest").resolve(),
        help="Output manifest directory. Defaults to data/manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print planned writes without creating files.",
    )
    return parser.parse_args(argv)


def resolved_path(path: Path) -> Path:
    return path.expanduser().resolve()


def display_path(path: Path, *, trailing_slash: bool = False) -> str:
    try:
        rendered = path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        rendered = path.resolve().as_posix()
    if trailing_slash and not rendered.endswith("/"):
        rendered += "/"
    return rendered


def validate_dataset(dataset: Any, *, line_number: int, section_id: Any) -> str:
    if not isinstance(dataset, str) or not dataset:
        raise PartitionError(
            "manifest row has missing/empty dataset "
            f"(line={line_number}, section_id={section_id!r})"
        )
    if not DATASET_RE.fullmatch(dataset):
        raise PartitionError(
            "manifest row has invalid dataset "
            f"{dataset!r}; expected [A-Za-z0-9_]+ "
            f"(line={line_number}, section_id={section_id!r})"
        )
    filename = f"{dataset}.jsonl"
    if not SHARD_FILENAME_RE.fullmatch(filename):
        raise PartitionError(
            f"invalid shard filename {filename!r}; expected [A-Za-z0-9_]+.jsonl"
        )
    return dataset


def derive_plane(row: dict[str, Any], *, line_number: int) -> str:
    axis = row.get("slice_axis")
    plane = PLANE_BY_AXIS.get(axis)
    if plane is None:
        raise PartitionError(
            "manifest row has missing/unrecognized slice_axis "
            f"{axis!r} (line={line_number}, "
            f"section_id={row.get('section_id')!r}, dataset={row.get('dataset')!r})"
        )
    return plane


def read_manifest(
    manifest_path: Path,
) -> tuple[dict[tuple[str, str], list[str]], Counter[tuple[str, str]], int]:
    if not manifest_path.is_file():
        raise PartitionError(f"input manifest not found: {display_path(manifest_path)}")

    shards: dict[tuple[str, str], list[str]] = defaultdict(list)
    counts: Counter[tuple[str, str]] = Counter()
    input_rows = 0

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            input_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PartitionError(
                    f"invalid JSON in manifest line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise PartitionError(
                    f"manifest line {line_number} is not a JSON object"
                )

            plane = derive_plane(row, line_number=line_number)
            dataset = validate_dataset(
                row.get("dataset"),
                line_number=line_number,
                section_id=row.get("section_id"),
            )
            key = (plane, dataset)
            shards[key].append(line)
            counts[key] += 1

    return dict(shards), counts, input_rows


def load_overrides(overrides_path: Path) -> dict[str, Any]:
    if not overrides_path.exists():
        return {}
    if not overrides_path.is_file():
        raise PartitionError(f"overrides path is not a file: {display_path(overrides_path)}")
    try:
        with overrides_path.open("r", encoding="utf-8") as handle:
            overrides = json.load(handle)
    except json.JSONDecodeError as exc:
        raise PartitionError(f"invalid JSON in overrides file: {exc}") from exc
    if not isinstance(overrides, dict):
        raise PartitionError("overrides file must contain a JSON object")
    return overrides


def empty_shard_overrides() -> dict[str, Any]:
    return {
        "disabled": False,
        "excluded_subjects": [],
        "excluded_sections": [],
        "subject_axis_flips": [],
        "section_position_overrides": [],
        "atlas_overrides": [],
    }


def dataset_planes_from_shards(
    shards: dict[tuple[str, str], list[str]],
) -> dict[str, list[str]]:
    planes_by_dataset: dict[str, set[str]] = defaultdict(set)
    for plane, dataset in shards:
        planes_by_dataset[dataset].add(plane)
    return {
        dataset: [plane for plane in PLANES if plane in planes]
        for dataset, planes in planes_by_dataset.items()
    }


def require_override_list(overrides: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = overrides.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise PartitionError(f"overrides key {key!r} must be a list")
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise PartitionError(f"overrides {key}[{index}] must be an object")
    return value


def require_string(entry: dict[str, Any], key: str, *, context: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value:
        raise PartitionError(f"{context} has missing/empty {key!r}: {entry!r}")
    return value


def find_planes_for_override(
    dataset: str,
    planes_by_dataset: dict[str, list[str]],
    *,
    override_kind: str,
    entry: dict[str, Any],
    unassigned: dict[str, list[dict[str, Any]]],
) -> list[str]:
    planes = planes_by_dataset.get(dataset, [])
    if planes:
        return planes
    unassigned[override_kind].append(dict(entry))
    return []


def copy_entry(
    entry: dict[str, Any],
    fields: tuple[str, ...],
    *,
    duplicated: bool,
) -> dict[str, Any]:
    copied = {field: entry[field] for field in fields if field in entry}
    if duplicated:
        copied["_duplicated_from_global"] = True
    return copied


def partition_overrides(
    overrides: dict[str, Any],
    shards: dict[tuple[str, str], list[str]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    shard_overrides = {key: empty_shard_overrides() for key in shards}
    planes_by_dataset = dataset_planes_from_shards(shards)
    unassigned: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for entry in require_override_list(overrides, "disabled_datasets"):
        dataset = require_string(entry, "name", context="disabled_datasets entry")
        validate_dataset(dataset, line_number=0, section_id="<disabled_datasets>")
        planes = find_planes_for_override(
            dataset,
            planes_by_dataset,
            override_kind="disabled_datasets",
            entry=entry,
            unassigned=unassigned,
        )
        for plane in planes:
            shard_overrides[(plane, dataset)]["disabled"] = True

    for entry in require_override_list(overrides, "excluded_subjects"):
        dataset = require_string(entry, "dataset", context="excluded_subjects entry")
        planes = find_planes_for_override(
            dataset,
            planes_by_dataset,
            override_kind="excluded_subjects",
            entry=entry,
            unassigned=unassigned,
        )
        duplicated = len(planes) > 1
        for plane in planes:
            shard_overrides[(plane, dataset)]["excluded_subjects"].append(
                copy_entry(entry, ("subject_id", "reason"), duplicated=duplicated)
            )

    for entry in require_override_list(overrides, "excluded_sections"):
        dataset = require_string(entry, "dataset", context="excluded_sections entry")
        planes = find_planes_for_override(
            dataset,
            planes_by_dataset,
            override_kind="excluded_sections",
            entry=entry,
            unassigned=unassigned,
        )
        duplicated = len(planes) > 1
        for plane in planes:
            shard_overrides[(plane, dataset)]["excluded_sections"].append(
                copy_entry(
                    entry,
                    ("section_id", "subject_id", "reason"),
                    duplicated=duplicated,
                )
            )

    for entry in require_override_list(overrides, "subject_axis_flips"):
        dataset = require_string(entry, "dataset", context="subject_axis_flips entry")
        axis = require_string(entry, "axis", context="subject_axis_flips entry")
        if axis not in AXES:
            raise PartitionError(
                f"subject_axis_flips entry has invalid axis {axis!r}: {entry!r}"
            )
        planes = find_planes_for_override(
            dataset,
            planes_by_dataset,
            override_kind="subject_axis_flips",
            entry=entry,
            unassigned=unassigned,
        )
        duplicated = len(planes) > 1
        for plane in planes:
            shard_overrides[(plane, dataset)]["subject_axis_flips"].append(
                copy_entry(
                    entry,
                    ("subject_id", "axis", "reason"),
                    duplicated=duplicated,
                )
            )

    log: dict[str, Any] = {
        "_doc": overrides.get("_doc"),
        "_partial_dataset_summary": overrides.get("_partial_dataset_summary"),
        "unassigned_overrides": dict(unassigned),
    }
    unknown_keys = sorted(
        set(overrides)
        - {
            "_doc",
            "_partial_dataset_summary",
            "disabled_datasets",
            "excluded_subjects",
            "excluded_sections",
            "subject_axis_flips",
        }
    )
    if unknown_keys:
        log["unknown_override_keys"] = {
            key: overrides[key]
            for key in unknown_keys
        }
    return shard_overrides, log


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise PartitionError(
            f"{display_path(partial)} already exists; remove it before re-running"
        )
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)


def assert_outputs_absent(out_dir: Path) -> None:
    for path in (out_dir / "shards", out_dir / "overrides"):
        if path.exists():
            raise PartitionError(
                f"{display_path(path, trailing_slash=True)} already exists; "
                "remove it before re-running"
            )
    log_path = out_dir / "_partition_log.json"
    if log_path.exists():
        raise PartitionError(
            f"{display_path(log_path)} already exists; remove it before re-running"
        )


def build_output_texts(
    out_dir: Path,
    shards: dict[tuple[str, str], list[str]],
    shard_overrides: dict[tuple[str, str], dict[str, Any]],
    partition_log: dict[str, Any],
    *,
    manifest_path: Path,
    overrides_path: Path,
    input_rows: int,
) -> dict[Path, str]:
    outputs: dict[Path, str] = {}
    for (plane, dataset), rows in sorted(shards.items()):
        shard_path = out_dir / "shards" / plane / f"{dataset}.jsonl"
        override_path = out_dir / "overrides" / plane / f"{dataset}.json"
        outputs[shard_path] = "\n".join(rows) + "\n"
        outputs[override_path] = (
            json.dumps(shard_overrides[(plane, dataset)], indent=2, ensure_ascii=False)
            + "\n"
        )

    log = dict(partition_log)
    log["migration"] = {
        "input_manifest": display_path(manifest_path),
        "input_overrides": display_path(overrides_path),
        "input_rows": input_rows,
        "shard_count": len(shards),
    }
    outputs[out_dir / "_partition_log.json"] = (
        json.dumps(log, indent=2, ensure_ascii=False) + "\n"
    )
    return outputs


def print_summary(
    counts: Counter[tuple[str, str]],
    *,
    input_rows: int,
) -> None:
    plane_totals = {
        plane: sum(count for (row_plane, _dataset), count in counts.items() if row_plane == plane)
        for plane in PLANES
    }
    plane_dataset_counts = {
        plane: sum(1 for row_plane, _dataset in counts if row_plane == plane)
        for plane in PLANES
    }
    total = sum(plane_totals.values())

    print("Partition complete:")
    for plane in PLANES:
        print(
            f"  {plane}: {plane_totals[plane]} rows "
            f"across {plane_dataset_counts[plane]} datasets"
        )
    print(f"  total: {total} rows")
    print(f"  input manifest: {input_rows} rows")
    print(f"  match: {total == input_rows}")
    if total != input_rows:
        raise PartitionError(
            f"partitioned total {total} does not match input manifest {input_rows}"
        )

    print("Top datasets by plane:")
    for plane in PLANES:
        top = Counter(
            {
                dataset: count
                for (row_plane, dataset), count in counts.items()
                if row_plane == plane
            }
        ).most_common(10)
        print(f"  {plane}:")
        if not top:
            print("    <none>")
            continue
        for dataset, count in top:
            print(f"    {dataset}: {count}")


def print_write_plan(outputs: dict[Path, str]) -> None:
    print("Dry run: no files written.")
    print(f"Would write {len(outputs)} files:")
    for path in sorted(outputs):
        text = outputs[path]
        if path.suffix == ".jsonl":
            row_count = text.count("\n")
            print(f"  {display_path(path)} ({row_count} rows)")
        else:
            print(f"  {display_path(path)}")


def run(
    *,
    manifest_path: Path,
    overrides_path: Path,
    out_dir: Path,
    dry_run: bool,
) -> None:
    manifest_path = resolved_path(manifest_path)
    overrides_path = resolved_path(overrides_path)
    out_dir = resolved_path(out_dir)

    shards, counts, input_rows = read_manifest(manifest_path)
    overrides = load_overrides(overrides_path)
    shard_overrides, partition_log = partition_overrides(overrides, shards)
    outputs = build_output_texts(
        out_dir,
        shards,
        shard_overrides,
        partition_log,
        manifest_path=manifest_path,
        overrides_path=overrides_path,
        input_rows=input_rows,
    )

    print_summary(counts, input_rows=input_rows)
    if dry_run:
        print_write_plan(outputs)
        return

    assert_outputs_absent(out_dir)
    for path, text in sorted(outputs.items()):
        write_text_atomic(path, text)
    print(f"Wrote {len(outputs)} files under {display_path(out_dir, trailing_slash=True)}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(
            manifest_path=args.manifest,
            overrides_path=args.overrides,
            out_dir=args.out_dir,
            dry_run=args.dry_run,
        )
    except PartitionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

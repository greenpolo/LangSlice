"""Rebuild one manifest shard at a time.

This entry point is the per-shard counterpart to ``build_manifest.py``.  A
shard is identified by ``<plane>/<dataset>`` and is written under
``data/manifest/shards/<plane>/<dataset>.jsonl`` after comparing the rebuilt
rows with the current shard.  Per-shard curation lives beside it under
``data/manifest/overrides/<plane>/<dataset>.json`` and is applied after the
adapter has emitted raw rows.

CLI modes:
    python _local/eval/rebuild_shard.py <plane>/<dataset>
    python _local/eval/rebuild_shard.py <plane>/<dataset> --accept-diff N
    python _local/eval/rebuild_shard.py <plane>/<dataset> --accept-diff-file PATH
    python _local/eval/rebuild_shard.py <plane>/<dataset> --out PATH

Dry-run mode exits 0 when the rebuilt shard is identical and 1 when any row is
added, removed, or changed.  Accepted writes are atomic via ``.partial`` plus
``Path.replace()``; existing canonical shards also receive a timestamped backup
written through the same partial-file path.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from build_manifest import (  # type: ignore[import-not-found]
    _BG_ASR_EXTENTS_MM,
    ADAPTERS,
    _normpath,
    _resolve_dev_atlas,
    load_allocation,
    make_record,
    split_for,
)

log = logging.getLogger("rebuild_shard")

PLANES = {"coronal", "sagittal", "horizontal"}
DATASET_RE = re.compile(r"^[A-Za-z0-9_]+$")
PLANE_FOR_AXIS = {"ap": "coronal", "ml": "sagittal", "dv": "horizontal"}


@dataclass(frozen=True)
class ShardEntry:
    adapter_name: str
    adapter_fn: Callable[[Path, dict], Iterator[dict]]
    output_dataset_name: str


@dataclass(frozen=True)
class DiffResult:
    changed: list[tuple[str, dict, dict]]
    added: list[tuple[str, dict]]
    removed: list[tuple[str, dict]]

    @property
    def total(self) -> int:
        return len(self.changed) + len(self.added) + len(self.removed)


def _canonical_shards_root() -> Path:
    quoted = Path(chr(34) + "data/manifest/shards" + chr(34))
    return Path(str(quoted).strip(chr(34)))


SHARDS_ROOT = _canonical_shards_root()
OVERRIDES_ROOT = Path("data/manifest/overrides")
ALLOCATION_PATH = Path("data/manifest_allocation.json")


def _allen_adapter_no_connectivity_override(
    data_root: Path,
    allocation: dict,
    *,
    dataset_name: str,
    rel_path: str,
    imaging: str,
    staining: str,
    orientation: str,
    require_anchored_ap_mm: bool,
) -> Iterator[dict]:
    """Copy of build_manifest._allen_adapter without connectivity formula override."""
    src = data_root / rel_path
    if not src.exists():
        log.warning("%s: %s missing, skipping", dataset_name, src)
        return
    records = json.loads(src.read_text(encoding="utf-8"))
    for r in records:
        # May 2026 incident: the section-number formula override clobbered
        # connectivity fixers' metadata-stored AP values for WT/Cre shards.
        if require_anchored_ap_mm and r.get("ap_mm_anchored") is not True:
            raise ValueError(f"{dataset_name}: metadata row lacks ap_mm_anchored == True")
        ap_mm = r.get("ap_mm")
        if require_anchored_ap_mm and ap_mm is None:
            raise ValueError(f"{dataset_name}: row {r.get('section_image_id')} has null ap_mm")
        img_path = r.get("image_path")
        if ap_mm is None or img_path is None:
            continue
        img_norm = Path(_normpath(img_path))
        if not img_norm.is_absolute() and not img_norm.exists():
            img_norm = Path(_normpath(str(img_norm)))
        if not img_norm.exists():
            continue
        subject_id = str(r.get("dataset_id", ""))
        split = split_for(allocation, dataset_name, subject_id)
        yield make_record(
            image_path=img_norm,
            dataset=dataset_name,
            subject_id=subject_id,
            section_id=f"{subject_id}:{r['section_image_id']}",
            split=split,
            species="mouse",
            atlas="allen_mouse_25um",
            orientation=orientation,
            position_mm=ap_mm,
            imaging=imaging,
            staining=staining,
            is_hemisphere=False,
            registration_source="allen_api",
            source_file=src.relative_to(data_root.parent),
            local_key=r["section_image_id"],
        )


def _adapter_allen_connectivity_wt_no_override(data_root: Path, allocation: dict) -> Iterator[dict]:
    yield from _allen_adapter_no_connectivity_override(
        data_root,
        allocation,
        dataset_name="allen_connectivity_wt",
        rel_path="datasets/allen_connectivity_wt/metadata.json",
        imaging="fluorescence",
        staining="AAV-EGFP (WT cortico-subcortical)",
        orientation="coronal",
        require_anchored_ap_mm=True,
    )


def _adapter_allen_connectivity_cre_no_override(
    data_root: Path, allocation: dict
) -> Iterator[dict]:
    yield from _allen_adapter_no_connectivity_override(
        data_root,
        allocation,
        dataset_name="allen_connectivity_cre",
        rel_path="datasets/allen_connectivity_cre/metadata.json",
        imaging="fluorescence",
        staining="AAV viral tracers (Cre-line transgenics)",
        orientation="coronal",
        require_anchored_ap_mm=True,
    )


ADAPTER_FN_OVERRIDES: dict[str, Callable[[Path, dict], Iterator[dict]]] = {
    "allen_connectivity_wt": _adapter_allen_connectivity_wt_no_override,
    "allen_connectivity_cre": _adapter_allen_connectivity_cre_no_override,
}

EXPLICIT_SHARDS: dict[str, list[tuple[str, str]]] = {
    "boccara_horizontal": [
        ("coronal", "boccara_coronal"),
        ("sagittal", "boccara_sagittal"),
        ("horizontal", "boccara_horizontal"),
    ],
    "m01_m09": [
        ("coronal", "m01_m03"),
        ("coronal", "m04_m09"),
    ],
    "silva_ieg": [("coronal", "silva_ieg")],
    "zenodo_pnnpv": [("coronal", "zenodo_pnnpv")],
    "nr2f1_couptf1": [("sagittal", "nr2f1_couptf1")],
    "nr2f1_yfp": [("sagittal", "nr2f1_yfp")],
    "allen_connectivity": [("coronal", "allen_connectivity")],
    "allen_connectivity_wt": [("coronal", "allen_connectivity_wt")],
    "allen_connectivity_cre": [("coronal", "allen_connectivity_cre")],
    "rat_tract_eval": [("coronal", "rat_tract_eval")],
    "rat_tract_rlvr": [("coronal", "rat_tract_rlvr")],
    "rat_tract_ic": [("coronal", "rat_tract_ic")],
    "timm_nissl": [
        ("coronal", "timm_nissl"),
        ("sagittal", "timm_nissl"),
        ("horizontal", "timm_nissl"),
    ],
    "bjaalie_misc": [
        ("sagittal", "bjaalie_misc"),
        ("horizontal", "bjaalie_misc"),
    ],
    "leergaard_cytomyelo": [
        ("coronal", "leergaard_cytomyelo"),
        ("sagittal", "leergaard_cytomyelo"),
    ],
    "brun2024_p14rat": [("sagittal", "brun2024_p14rat")],
    "bap_head_3plane": [
        ("coronal", "bap_head_3plane"),
        ("sagittal", "bap_head_3plane"),
        ("horizontal", "bap_head_3plane"),
    ],
    "deepslice_gt": [
        ("coronal", "deepslice_gt"),
        ("sagittal", "deepslice_gt"),
        ("horizontal", "deepslice_gt"),
    ],
    "ebrains_tta_atlas": [
        ("coronal", "ebrains_tta_atlas"),
        ("horizontal", "ebrains_tta_atlas"),
    ],
    "ebrains_receptor_rat": [
        ("coronal", "ebrains_receptor_rat"),
        ("sagittal", "ebrains_receptor_rat"),
        ("horizontal", "ebrains_receptor_rat"),
    ],
    "dopamap": [("coronal", "dopamap")],
    "ad_bxd": [("coronal", "ad_bxd")],
    "bjerke_calbindin_dev_mouse": [("coronal", "bjerke_calbindin_dev_mouse")],
    "bjerke_parvalbumin_dev_mouse": [("coronal", "bjerke_parvalbumin_dev_mouse")],
}


def _infer_default_shards(adapter_name: str) -> list[tuple[str, str]]:
    if adapter_name.endswith("_coronal") or "_coronal_" in adapter_name:
        return [("coronal", adapter_name)]
    if adapter_name.endswith("_sagittal") or "_sagittal_" in adapter_name:
        return [("sagittal", adapter_name)]
    if adapter_name.endswith("_horizontal") or "_horizontal_" in adapter_name:
        return [("horizontal", adapter_name)]
    log.warning("adapter %s has no plane hint; registering all planes", adapter_name)
    return [(plane, adapter_name) for plane in sorted(PLANES)]


def build_shard_registry() -> dict[tuple[str, str], ShardEntry]:
    registry: dict[tuple[str, str], ShardEntry] = {}
    reachable: set[str] = set()
    for adapter_name, adapter_fn in ADAPTERS.items():
        fn = ADAPTER_FN_OVERRIDES.get(adapter_name, adapter_fn)
        shard_specs = EXPLICIT_SHARDS.get(adapter_name) or _infer_default_shards(adapter_name)
        for plane, output_dataset_name in shard_specs:
            key = (plane, output_dataset_name)
            if key in registry:
                other = registry[key].adapter_name
                raise RuntimeError(
                    f"shard registry collision for {plane}/{output_dataset_name}: "
                    f"{other} and {adapter_name}"
                )
            registry[key] = ShardEntry(
                adapter_name=adapter_name,
                adapter_fn=fn,
                output_dataset_name=output_dataset_name,
            )
            reachable.add(adapter_name)
    for orphan in sorted(set(ADAPTERS) - reachable):
        log.warning("adapter %s is not reachable from shard registry", orphan)
    return registry


REGISTRY = build_shard_registry()


def parse_shard(value: str) -> tuple[str, str]:
    parts = value.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("shard must be <plane>/<dataset>")
    plane, dataset = parts
    if plane not in PLANES:
        raise ValueError(f"bad plane {plane!r}; expected one of {', '.join(sorted(PLANES))}")
    if DATASET_RE.fullmatch(dataset) is None:
        raise ValueError(f"bad dataset {dataset!r}; expected [A-Za-z0-9_]+")
    return plane, dataset


def shard_path(plane: str, dataset: str) -> Path:
    return SHARDS_ROOT / plane / f"{dataset}.jsonl"


def overrides_path(plane: str, dataset: str) -> Path:
    return OVERRIDES_ROOT / plane / f"{dataset}.json"


def derived_plane(rec: dict) -> str | None:
    axis = rec.get("slice_axis")
    if isinstance(axis, str) and axis in PLANE_FOR_AXIS:
        return PLANE_FOR_AXIS[axis]
    orientation = rec.get("orientation")
    if isinstance(orientation, str) and orientation in PLANES:
        return orientation
    return None


def load_shard_overrides(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _string_set(values: Any, key: str) -> set[str]:
    if values is None:
        return set()
    if not isinstance(values, list):
        raise ValueError(f"{key} must be a list")
    result: set[str] = set()
    for item in values:
        if isinstance(item, str):
            result.add(item)
        elif isinstance(item, dict) and key[:-1] in item:
            result.add(str(item[key[:-1]]))
        elif isinstance(item, dict) and key == "excluded_subjects" and "subject_id" in item:
            result.add(str(item["subject_id"]))
        elif isinstance(item, dict) and key == "excluded_sections" and "section_id" in item:
            result.add(str(item["section_id"]))
        else:
            raise ValueError(f"bad {key} entry: {item!r}")
    return result


def _subject_axis_flips(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(subject_id): str(axis) for subject_id, axis in value.items()}
    if not isinstance(value, list):
        raise ValueError("subject_axis_flips must be a list or object")
    result: dict[str, str] = {}
    for item in value:
        if not isinstance(item, dict) or "subject_id" not in item or "axis" not in item:
            raise ValueError(f"bad subject_axis_flips entry: {item!r}")
        result[str(item["subject_id"])] = str(item["axis"])
    return result


def _section_position_overrides(value: Any) -> dict[str, tuple[float, str | None]]:
    if value is None:
        return {}
    result: dict[str, tuple[float, str | None]] = {}
    if isinstance(value, dict):
        for section_id, entry in value.items():
            if isinstance(entry, dict):
                pos = entry.get("position_mm")
                axis = entry.get("axis")
            else:
                pos = entry
                axis = None
            if pos is None:
                raise ValueError(f"position override for {section_id!r} has null position_mm")
            result[str(section_id)] = (
                round(float(pos), 4),
                str(axis) if axis is not None else None,
            )
        return result
    if not isinstance(value, list):
        raise ValueError("section_position_overrides must be a list or object")
    for item in value:
        if not isinstance(item, dict) or "section_id" not in item or "position_mm" not in item:
            raise ValueError(f"bad section_position_overrides entry: {item!r}")
        axis = item.get("axis")
        result[str(item["section_id"])] = (
            round(float(item["position_mm"]), 4),
            str(axis) if axis is not None else None,
        )
    return result


def _atlas_overrides(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, dict):
        result: dict[str, str] = {}
        for section_id, entry in value.items():
            atlas = entry.get("atlas") if isinstance(entry, dict) else entry
            if atlas is None:
                raise ValueError(f"atlas override for {section_id!r} has null atlas")
            result[str(section_id)] = str(atlas)
        return result
    if not isinstance(value, list):
        raise ValueError("atlas_overrides must be a list or object")
    result = {}
    for item in value:
        if not isinstance(item, dict) or "section_id" not in item:
            raise ValueError(f"bad atlas_overrides entry: {item!r}")
        atlas = item.get("atlas") or item.get("new_atlas")
        if atlas is None:
            raise ValueError(f"bad atlas_overrides entry: {item!r}")
        result[str(item["section_id"])] = str(atlas)
    return result


def apply_overrides(records: list[dict], overrides: dict[str, Any]) -> list[dict]:
    if overrides.get("disabled") is True:
        raise ValueError("shard override has disabled == true")

    # Phase C.2: the `split` field is now managed by the allocation layer
    # (data/manifest/allocations/<plane>/<split>.jsonl).  Inventory shards must
    # never contain a `split` key.  If any row still carries one, the migration
    # script has not been run, or a bug re-introduced the field.
    rows_with_split = [str(rec.get("section_id", "<unknown>")) for rec in records if "split" in rec]
    if rows_with_split:
        raise ValueError(
            f"inventory shard contains {len(rows_with_split)} row(s) with a 'split' field "
            f"(first: {rows_with_split[0]!r}).  "
            "The 'split' field must not appear in inventory shards -- run "
            "migrate_split_to_allocations.py to lift splits into the allocation layer."
        )

    excluded_subjects = _string_set(overrides.get("excluded_subjects"), "excluded_subjects")
    excluded_sections = _string_set(overrides.get("excluded_sections"), "excluded_sections")
    axis_flips = _subject_axis_flips(overrides.get("subject_axis_flips"))
    position_overrides = _section_position_overrides(overrides.get("section_position_overrides"))
    atlas_overrides = _atlas_overrides(overrides.get("atlas_overrides"))

    kept: list[dict] = []
    for rec in records:
        subject_id = str(rec.get("subject_id", ""))
        section_id = str(rec.get("section_id", ""))
        if subject_id in excluded_subjects:
            continue
        if section_id in excluded_sections:
            continue
        kept.append(dict(rec))

    for rec in kept:
        subject_id = str(rec.get("subject_id", ""))
        axis = axis_flips.get(subject_id)
        if not axis:
            continue
        if axis != rec.get("slice_axis"):
            continue
        extent = _BG_ASR_EXTENTS_MM.get((rec.get("atlas"), axis))
        if extent is None:
            raise ValueError(
                f"cannot flip {rec.get('section_id')}: no extent for {rec.get('atlas')} {axis}"
            )
        rec["position_mm"] = round(extent - float(rec["position_mm"]), 4)
        rec["axis_flip_applied"] = axis

    by_section = _index_by_section(kept, "rebuilt shard")
    for section_id, (position_mm, axis) in position_overrides.items():
        rec = by_section.get(section_id)
        if rec is None:
            raise ValueError(
                f"section_position_overrides references missing section {section_id!r}"
            )
        if axis is not None and axis != rec.get("slice_axis"):
            raise ValueError(
                f"section_position_overrides axis {axis!r} does not match "
                f"{section_id} slice_axis {rec.get('slice_axis')!r}"
            )
        rec["position_mm"] = position_mm
        rec["position_override_applied"] = True

    for section_id, new_atlas in atlas_overrides.items():
        rec = by_section.get(section_id)
        if rec is None:
            raise ValueError(f"atlas_overrides references missing section {section_id!r}")
        old_atlas = rec.get("atlas")
        axis = rec.get("slice_axis")
        old_extent = _BG_ASR_EXTENTS_MM.get((old_atlas, axis))
        new_extent = _BG_ASR_EXTENTS_MM.get((new_atlas, axis))
        if old_extent is not None and new_extent is not None and old_extent != new_extent:
            log.warning(
                "%s: atlas override changes %s %s extent from %.4f to %.4f",
                section_id,
                old_atlas,
                axis,
                old_extent,
                new_extent,
            )
        rec["atlas"] = new_atlas
        rec["atlas_override_applied"] = True

    return kept


def _resolve_legacy_atlases(records: list[dict]) -> None:
    if _resolve_dev_atlas is None:
        return
    for rec in records:
        if rec.get("atlas") != "admba":
            continue
        new_atlas, reason = _resolve_dev_atlas(rec.get("dataset", ""), rec.get("age"))
        if new_atlas:
            rec["atlas"] = new_atlas
            rec["atlas_legacy_tag"] = "admba"
            rec["atlas_resolution_reason"] = reason
        else:
            rec["atlas_unmappable"] = True
            rec["atlas_resolution_reason"] = reason


def _iter_records(entry: ShardEntry, data_root: Path, allocation: dict) -> Iterator[dict]:
    yield from entry.adapter_fn(data_root, allocation)


def collect(plane: str, dataset: str, entry: ShardEntry, data_root: Path) -> list[dict]:
    allocation = load_allocation(ALLOCATION_PATH)
    records = [
        dict(rec)
        for rec in _iter_records(entry, data_root, allocation)
        if rec.get("dataset") == entry.output_dataset_name and derived_plane(rec) == plane
    ]
    # Adapters still call split_for(...) and stamp `split` onto records for
    # back-compat with the legacy build_manifest.py main loop. Split now
    # lives in the allocation layer; strip it here before apply_overrides
    # runs its defensive `no split field` check.
    for rec in records:
        rec.pop("split", None)
    _resolve_legacy_atlases(records)
    overrides = load_shard_overrides(overrides_path(plane, dataset))
    return apply_overrides(records, overrides)


def _index_by_section(records: Iterable[dict], label: str) -> dict[str, dict]:
    by_section: dict[str, dict] = {}
    for rec in records:
        section_id = str(rec.get("section_id", ""))
        if not section_id:
            raise ValueError(f"{label} contains a row without section_id")
        if section_id in by_section:
            raise ValueError(f"{label} contains duplicate section_id {section_id!r}")
        by_section[section_id] = rec
    return by_section


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
    return rows


def diff_records(current: list[dict], rebuilt: list[dict]) -> DiffResult:
    old = _index_by_section(current, "current shard")
    new = _index_by_section(rebuilt, "rebuilt shard")
    changed = [
        (sid, old[sid], new[sid]) for sid in sorted(old.keys() & new.keys()) if old[sid] != new[sid]
    ]
    added = [(sid, new[sid]) for sid in sorted(new.keys() - old.keys())]
    removed = [(sid, old[sid]) for sid in sorted(old.keys() - new.keys())]
    return DiffResult(changed=changed, added=added, removed=removed)


def print_diff_summary(diff: DiffResult) -> None:
    print(
        f"diff: {len(diff.changed)} changed, {len(diff.added)} added, "
        f"{len(diff.removed)} removed ({diff.total} total)"
    )
    for label, rows in (
        ("changed", diff.changed),
        ("added", diff.added),
        ("removed", diff.removed),
    ):
        for row in rows[:20]:
            if label == "changed":
                section_id, old, new = row
                print(
                    f"  {label}: {section_id} {old.get('position_mm')} -> {new.get('position_mm')}"
                )
            else:
                section_id, rec = row
                print(f"  {label}: {section_id} position_mm={rec.get('position_mm')}")
        if len(rows) > 20:
            print(f"  {label}: ... {len(rows) - 20} more")


def serialize_jsonl(records: Iterable[dict]) -> str:
    return "".join(json.dumps(rec, ensure_ascii=False) + "\n" for rec in records)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(text, encoding="utf-8", newline="\n")
    partial.replace(path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_bytes(data)
    partial.replace(path)


def backup_existing(path: Path) -> Path | None:
    if not path.exists():
        return None
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak.{ts}")
    atomic_write_bytes(backup, path.read_bytes())
    return backup


def _norm_position(value: Any) -> str:
    return f"{float(value):.4f}"


def diff_tuple_set(diff: DiffResult) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for section_id, _old, new in diff.changed:
        result.add((section_id, _norm_position(new.get("position_mm"))))
    for section_id, rec in diff.added:
        result.add((section_id, _norm_position(rec.get("position_mm"))))
    for section_id, rec in diff.removed:
        result.add((section_id, _norm_position(rec.get("position_mm"))))
    return result


def _tuple_from_item(item: Any) -> tuple[str, str]:
    if isinstance(item, dict):
        section_id = item.get("section_id")
        position_mm = item.get("position_mm")
    elif isinstance(item, (list, tuple)) and len(item) >= 2:
        section_id, position_mm = item[0], item[1]
    else:
        raise ValueError(f"bad diff tuple item: {item!r}")
    if section_id is None or position_mm is None:
        raise ValueError(f"bad diff tuple item: {item!r}")
    return str(section_id), _norm_position(position_mm)


def load_accept_diff_file(path: Path) -> set[tuple[str, str]]:
    raw = path.read_text(encoding="utf-8")
    stripped = raw.strip()
    if not stripped:
        return set()
    if stripped[0] in "[{":
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            items = payload.get("changed") or payload.get("changes") or payload.get("rows")
            if items is None:
                raise ValueError(f"{path} JSON object must contain changed, changes, or rows")
        else:
            items = payload
        if not isinstance(items, list):
            raise ValueError(f"{path} must contain a list of changed tuples")
        return {_tuple_from_item(item) for item in items}

    result: set[tuple[str, str]] = set()
    for lineno, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if lineno == 1 and line.lower().replace(" ", "") in {
            "section_id,position_mm",
            "section_id position_mm",
        }:
            continue
        if "," in line:
            section_id, position_mm = line.rsplit(",", 1)
        else:
            parts = line.rsplit(None, 1)
            if len(parts) != 2:
                raise ValueError(f"{path}:{lineno}: expected section_id and position_mm")
            section_id, position_mm = parts
        result.add((section_id.strip(), _norm_position(position_mm.strip())))
    return result


def _resolve_for_comparison(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def _is_inside(path: Path, root: Path) -> bool:
    resolved_path = _resolve_for_comparison(path)
    resolved_root = _resolve_for_comparison(root)
    try:
        resolved_path.relative_to(resolved_root)
        return True
    except ValueError:
        return False


def validate_out_path(path: Path) -> None:
    if _is_inside(path, SHARDS_ROOT) or _is_inside(path, OVERRIDES_ROOT):
        raise ValueError(
            "--out PATH must not be inside data/manifest/shards/ or data/manifest/overrides/"
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", help="<plane>/<dataset>")
    parser.add_argument("--accept-diff", type=int, metavar="N")
    parser.add_argument("--accept-diff-file", type=Path, metavar="PATH")
    parser.add_argument("--out", type=Path, metavar="PATH")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.accept_diff is not None and args.accept_diff_file is not None:
        parser.error("--accept-diff and --accept-diff-file are mutually exclusive")
    if args.out is not None and (args.accept_diff is not None or args.accept_diff_file is not None):
        parser.error("--out is a what-if mode and cannot be combined with accept-diff gates")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not SHARDS_ROOT.exists():
        log.error("%s does not exist; run partition_manifest.py first", SHARDS_ROOT)
        return 2

    try:
        plane, dataset = parse_shard(args.shard)
    except ValueError as e:
        log.error("%s", e)
        return 2

    entry = REGISTRY.get((plane, dataset))
    if entry is None:
        log.error("no adapter registered for %s/%s", plane, dataset)
        return 2

    try:
        if args.out is not None:
            validate_out_path(args.out)
        canonical = shard_path(plane, dataset)
        data_root = args.data_root.resolve()
        rebuilt = collect(plane, dataset, entry, data_root)

        if args.out is not None:
            atomic_write_text(args.out, serialize_jsonl(rebuilt))
            log.info("wrote %s (%d rows)", args.out, len(rebuilt))
            return 0

        current = load_jsonl(canonical)
        diff = diff_records(current, rebuilt)
        print_diff_summary(diff)

        if args.accept_diff is None and args.accept_diff_file is None:
            return 1 if diff.total else 0

        if args.accept_diff is not None and diff.total != args.accept_diff:
            log.error(
                "diff gate rejected: expected %d changes, saw %d", args.accept_diff, diff.total
            )
            return 1

        if args.accept_diff_file is not None:
            expected = load_accept_diff_file(args.accept_diff_file)
            actual = diff_tuple_set(diff)
            if actual != expected:
                log.error(
                    "diff-file gate rejected: expected %d tuples, saw %d",
                    len(expected),
                    len(actual),
                )
                missing = sorted(expected - actual)[:20]
                extra = sorted(actual - expected)[:20]
                for item in missing:
                    log.error("  missing actual tuple: %s", item)
                for item in extra:
                    log.error("  unexpected actual tuple: %s", item)
                return 1

        canonical.parent.mkdir(parents=True, exist_ok=True)
        backup = backup_existing(canonical)
        if backup is not None:
            log.info("backed up %s -> %s", canonical, backup)
        atomic_write_text(canonical, serialize_jsonl(rebuilt))
        log.info("wrote %s (%d rows)", canonical, len(rebuilt))
        return 0
    except Exception as e:  # noqa: BLE001
        log.error("%s", e)
        if args.verbose:
            log.exception("failed")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Unified LangSlice QC app.

Run from repo root:

    python _local/qc_app/app.py
    # then open http://localhost:8765

Three modules in one server:
    Inventory   — data/manifest.jsonl (all rows; sources -> drill-in -> lightbox)
    Training    — SFT (trace_collection), RLVR (manifest split=='rlvr' + live progress),
                  Eval (manifest split=='eval')
    Synthetic   — _local/synth_data/manifest.jsonl + images/{seed:08d}.png

No verdict workflows. See `_local/qc_app/CONTRACTS.md` for the file contracts
each module reads.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import socket
import sys
import threading
import time
import webbrowser
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from langslice_data.manifest.allocations import (  # noqa: E402
    ALLOCATIONS_ROOT as _DEFAULT_ALLOCATIONS_ROOT,
)

from langslice_harness.atlas.core import (  # noqa: E402
    get_composite_slice,
    get_reference_slice,
    load_atlas,
)

logger = logging.getLogger("qc_app")

DEFAULT_INVENTORY_SHARDS = REPO_ROOT / "data/manifest/shards"
DEFAULT_INVENTORY_MANIFEST = REPO_ROOT / "data/manifest.jsonl"  # legacy fallback
DEFAULT_SYNTH_DIR = REPO_ROOT / "_local/synth_data"
DEFAULT_SFT_DIR = REPO_ROOT / "_local/trace_collection"
DEFAULT_RLVR_PROGRESS_DIR = REPO_ROOT / "_local/rlvr_progress"
INVENTORY_THUMB_CACHE = REPO_ROOT / "_local/qc_app/thumb_cache"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Imported at module level so tests can monkeypatch the singleton; see
# training_progress.py for the on-disk contract this consumes.
import sys as _sys  # noqa: E402

_sys.path.insert(0, str(Path(__file__).resolve().parent))
from training_progress import TrainingProgress, status_per_id  # noqa: E402

# ----------------------------------------------------------------------
# Atlas rendering with caching


_atlas_lock = threading.Lock()
_atlas_status: dict[str, str] = {}  # atlas_name -> "ok" | error message
_render_cache: OrderedDict[tuple[str, str, int, str], bytes] = OrderedDict()
# Bumped from 256 -> 1024 because aggressive prefetch (5+ slices ahead, plus
# composite + reference variants per slice) used to evict tiles seconds after
# loading them. 1024 PNGs is ~25MB at typical sizes; trivial.
_RENDER_CACHE_MAX = 1024
ATLAS_DISK_CACHE = REPO_ROOT / "_local" / "qc_app" / "atlas_cache"


def _quantize_position(position_mm: float) -> int:
    """Quantize to ~1 micron so cache hits across float jitter."""
    return int(round(position_mm * 1000.0))


def _disk_cache_path(atlas_name: str, plane: str, pos_microns: int, mode: str) -> Path:
    safe_atlas = atlas_name.replace("/", "_").replace("\\", "_")
    safe_plane = plane.replace("/", "_").replace("\\", "_")
    sign = "n" if pos_microns < 0 else "p"
    return ATLAS_DISK_CACHE / safe_atlas / safe_plane / f"{mode}_{sign}{abs(pos_microns):07d}.png"


def _get_or_load_atlas(atlas_name: str):
    status = _atlas_status.get(atlas_name)
    if status and status != "ok":
        return None
    try:
        atlas = load_atlas(atlas_name)
        _atlas_status[atlas_name] = "ok"
        return atlas
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        _atlas_status[atlas_name] = msg
        logger.warning("atlas %s unavailable: %s", atlas_name, msg)
        return None


def _placeholder_png(text: str) -> bytes:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (512, 512), color=(40, 40, 48))
    draw = ImageDraw.Draw(img)
    lines = [text[i : i + 40] for i in range(0, len(text), 40)] or [""]
    y = 240 - 12 * len(lines)
    for line in lines:
        draw.text((20, y), line, fill=(220, 220, 220))
        y += 20
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_atlas_png(atlas_name: str, plane: str, position_mm: float, mode: str) -> bytes:
    pos_microns = _quantize_position(position_mm)
    key = (atlas_name, plane, pos_microns, mode)
    with _atlas_lock:
        cached = _render_cache.get(key)
        if cached is not None:
            _render_cache.move_to_end(key)
            return cached

    disk_path = _disk_cache_path(atlas_name, plane, pos_microns, mode)
    if disk_path.exists():
        png = disk_path.read_bytes()
        with _atlas_lock:
            _render_cache[key] = png
            while len(_render_cache) > _RENDER_CACHE_MAX:
                _render_cache.popitem(last=False)
        return png

    atlas = _get_or_load_atlas(atlas_name)
    if atlas is None:
        png = _placeholder_png(
            f"Atlas '{atlas_name}' unavailable\n{_atlas_status.get(atlas_name, '')}"
        )
        persist_to_disk = False
    else:
        try:
            if mode == "reference":
                img = get_reference_slice(atlas, position_mm, plane=plane).convert("RGB")
            else:
                img = get_composite_slice(atlas, position_mm, plane=plane)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png = buf.getvalue()
            persist_to_disk = True
        except Exception as exc:
            png = _placeholder_png(
                f"Render failed\n{atlas_name} / {plane} / {position_mm:.3f}mm\n{exc}"
            )
            persist_to_disk = False

    if persist_to_disk:
        try:
            disk_path.parent.mkdir(parents=True, exist_ok=True)
            disk_path.write_bytes(png)
        except OSError as exc:
            logger.warning("disk-cache write failed for %s: %s", disk_path, exc)

    with _atlas_lock:
        _render_cache[key] = png
        while len(_render_cache) > _RENDER_CACHE_MAX:
            _render_cache.popitem(last=False)
    return png


# ----------------------------------------------------------------------
# SFT trace manifest + estimates loading
#
# Reused by the SFT sub-tab to render real / GT / estimated triple-image cards.


def load_manifest(path: Path) -> list[dict]:
    entries: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s:
            continue
        entries.append(json.loads(s))
    return entries


def _iter_result_files(path: Path) -> list[Path]:
    """Resolve --results into a list of jsonl files.

    Accepts a single .jsonl file, or a directory that contains
    `results.jsonl` files at any depth (e.g. `runs_verified/`).
    """
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.rglob("results.jsonl"))
    return []


def load_estimates(path: Path) -> dict[str, dict]:
    """Map trace id -> result payload.

    Each entry has keys: positions_mm (always list[float]|None),
    truth_positions_mm, label, accepted, max_error_mm,
    trajectory_excursion_mm, cost_usd, sft_accepted, sft_notes,
    source_file (which results.jsonl this came from).

    Reads `submitted_positions_mm` (collect-traces output) as well as the
    legacy `estimated_position[s]_mm` field names, so older estimate files
    still load.
    """
    out: dict[str, dict] = {}
    files = _iter_result_files(path)
    if not files:
        return out

    for fpath in files:
        for raw in fpath.read_text(encoding="utf-8").splitlines():
            s = raw.strip()
            if not s:
                continue
            d = json.loads(s)
            tid = d.get("id")
            if not tid:
                continue

            positions: list[float] | None = d.get("submitted_positions_mm") or d.get(
                "estimated_positions_mm"
            )
            if positions is None and d.get("estimated_position_mm") is not None:
                positions = [float(d["estimated_position_mm"])]

            cat = d.get("category") or {}
            sft = d.get("sft_quality") or {}
            sft_notes_parts: list[str] = []
            for key in ("missing_rationale_before_tool", "overlong_rationale"):
                v = sft.get(key)
                if v:
                    sft_notes_parts.append(f"{key}={v}")
            tool_turns = sft.get("assistant_tool_turns")
            if tool_turns is not None:
                sft_notes_parts.append(f"turns={tool_turns}")

            out[tid] = {
                "positions_mm": positions,
                "truth_positions_mm": d.get("truth_positions_mm"),
                "label": cat.get("label"),
                "accepted": cat.get("accepted"),
                "max_error_mm": cat.get("max_error_mm"),
                "trajectory_excursion_mm": cat.get("trajectory_excursion_mm"),
                "cost_usd": d.get("estimated_cost_usd"),
                "sft_accepted": sft.get("accepted_for_sft"),
                "sft_notes": ", ".join(sft_notes_parts) if sft_notes_parts else None,
                "source_file": str(fpath.relative_to(REPO_ROOT))
                if REPO_ROOT in fpath.parents
                else str(fpath),
            }
    return out


PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "": 4, None: 4}


def _entry_brain_key(e: dict) -> tuple[str, str]:
    meta = e.get("metadata", {})
    return meta.get("source_dataset", "?"), meta.get("subject_id", "?")


def _entry_is_unmappable(e: dict) -> bool:
    return bool(e.get("metadata", {}).get("atlas_unmappable"))


def _entry_to_payload(e: dict, plane: str, estimates: dict[str, dict]) -> dict:
    positions = (
        list(e.get("positions_mm", []))
        if e.get("kind") == "group"
        else [float(e.get("position_mm", 0.0))]
    )
    images = list(e.get("images", [])) if e.get("kind") == "group" else [e.get("image", "")]
    est = estimates.get(e["id"], {}) if estimates else {}
    return {
        "id": e["id"],
        "kind": e.get("kind", "single"),
        "section_ids": (
            e.get("metadata", {}).get("section_ids")
            or [e.get("metadata", {}).get("section_id", "")]
        ),
        "positions_mm": positions,
        "images": images,
        "atlas": e.get("atlas", ""),
        "plane": plane,
        "estimated_positions_mm": est.get("positions_mm"),
        "result_label": est.get("label"),
        "result_accepted": est.get("accepted"),
        "result_max_error_mm": est.get("max_error_mm"),
        "result_trajectory_excursion_mm": est.get("trajectory_excursion_mm"),
        "result_cost_usd": est.get("cost_usd"),
        "result_sft_accepted": est.get("sft_accepted"),
        "result_sft_notes": est.get("sft_notes"),
        "result_source_file": est.get("source_file"),
        "atlas_unmappable": _entry_is_unmappable(e),
        "atlas_resolution_reason": e.get("metadata", {}).get("atlas_resolution_reason"),
    }


# ----------------------------------------------------------------------
# Inventory mode: scans the full data/manifest.jsonl for triage.
#
# Inventory rows are flat manifest records: image_path, dataset, subject_id,
# position_mm, atlas, etc. No verdict workflow — all rows surface as-is and
# Training/RLVR + Training/Eval modules filter by `split`.


# Stable per-row id, never collides across datasets even when subject_ids
# alone might.
def inventory_row_id(row: dict) -> str:
    ds = row.get("dataset", "?")
    sid = row.get("section_id") or ""
    if not sid:
        # Fall back to the image_path filename so we still produce something
        # stable; manifest rows shipped without section_id are extremely rare.
        sid = Path(row.get("image_path", "")).stem or "?"
    return f"{ds}/{sid}"


def _iter_shard_files(shards_dir: Path) -> list[Path]:
    """Return shard files under <shards_dir>/<plane>/<dataset>.jsonl in
    deterministic order (planes coronal -> sagittal -> horizontal,
    dataset alphabetical within each plane).
    """
    out: list[Path] = []
    for plane in ("coronal", "sagittal", "horizontal"):
        plane_dir = shards_dir / plane
        if not plane_dir.is_dir():
            continue
        out.extend(sorted(plane_dir.glob("*.jsonl")))
    return out


def _plane_for_axis(slice_axis: str | None) -> str | None:
    """Map a row's slice_axis field to its canonical plane name.

    Returns one of 'coronal', 'sagittal', 'horizontal', or None if the
    axis is unrecognised or absent.
    """
    if not slice_axis:
        return None
    mapping = {
        "ap": "coronal",
        "ml": "sagittal",
        "dv": "horizontal",
    }
    return mapping.get(slice_axis.lower())


def _iter_allocation_files(allocations_root: Path) -> list[Path]:
    """Return all allocation JSONL files under <allocations_root>/<plane>/<split>.jsonl."""
    out: list[Path] = []
    for plane in ("coronal", "sagittal", "horizontal"):
        plane_dir = allocations_root / plane
        if not plane_dir.is_dir():
            continue
        out.extend(sorted(plane_dir.glob("*.jsonl")))
    return out


def load_inventory_manifest(
    source: Path,
    allocations_root: Path | None = None,
) -> list[dict]:
    """Load the inventory manifest into a list of plain dicts.

    `source` is either a sharded directory (`data/manifest/shards/`
    containing `<plane>/<dataset>.jsonl`) or a legacy single-file
    `data/manifest.jsonl`. Sharded layout takes precedence when both exist.

    Stamps an `_id` on each row up front (computed via `inventory_row_id`)
    so other modules can join on it without recomputing per-request.

    Stamps `_split` (derived from the allocation layer) onto each row.
    For rows where the plane cannot be determined, `_split` is set to "".
    The `split` field (if present on legacy rows) is used as a fallback only
    when the allocations layer has no entry for that section_id.
    """
    # Build a fast lookup: (plane, section_id) -> split from allocation files.
    alloc_root = allocations_root if allocations_root is not None else _DEFAULT_ALLOCATIONS_ROOT
    alloc_map: dict[tuple[str, str], str] = {}
    for plane in ("coronal", "sagittal", "horizontal"):
        for split in ("eval", "rlvr", "sft"):
            # Read directly from the (possibly custom) allocations root rather
            # than calling load_allocation() which uses the module-level constant.
            alloc_path = alloc_root / plane / f"{split}.jsonl"
            if not alloc_path.exists():
                continue
            try:
                with alloc_path.open("r", encoding="utf-8") as _fh:
                    for _lineno, _raw in enumerate(_fh, start=1):
                        _stripped = _raw.strip()
                        if not _stripped:
                            continue
                        try:
                            _entry = json.loads(_stripped)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(_entry, dict):
                            continue
                        _sid = _entry.get("section_id")
                        if not _sid:
                            continue
                        if _entry.get("tombstone") is True:
                            alloc_map.pop((plane, _sid), None)
                        else:
                            alloc_map[(plane, _sid)] = split
            except OSError:
                continue

    rows: list[dict] = []
    files: list[Path]
    if source.is_dir():
        files = _iter_shard_files(source)
    else:
        files = [source]
    for path in files:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                s = raw.strip()
                if not s:
                    continue
                try:
                    row = json.loads(s)
                except json.JSONDecodeError:
                    continue
                row["_id"] = inventory_row_id(row)
                plane = _plane_for_axis(row.get("slice_axis"))
                if plane:
                    section_id = str(row.get("section_id", ""))
                    row["_split"] = alloc_map.get((plane, section_id), "")
                else:
                    # Fallback: use legacy split field if plane can't be derived.
                    row["_split"] = row.get("split") or ""
                rows.append(row)
    return rows


def _matches_inventory_filter(row: dict, q: dict) -> bool:
    plane = q.get("plane")
    if plane and plane != "all" and row.get("orientation") != plane:
        return False

    datasets = q.get("datasets")  # list[str] or None
    if datasets and row.get("dataset") not in datasets:
        return False

    subject = q.get("subject")
    if subject and row.get("subject_id") != subject:
        return False

    atlas = q.get("atlas")
    if atlas and row.get("atlas") != atlas:
        return False

    species = q.get("species")
    if species and row.get("species") != species:
        return False

    reg = q.get("registration_source")
    if reg and (row.get("registration_source") or "") != reg:
        return False

    exclude = q.get("exclude")  # "true" | "false" | "all"
    is_excluded = bool(row.get("exclude_from_training", False))
    if exclude == "true" and not is_excluded:
        return False
    if exclude == "false" and is_excluded:
        return False

    splits = q.get("splits")  # list[str] or None  (eval/sft/rlvr)
    if splits:
        if (row.get("_split") or "") not in splits:
            return False

    # rlvr_status filter is computed against the live training progress
    # snapshot the caller passed in via q["_rlvr_status_map"]. Done this way
    # so we don't redundantly hit TrainingProgress per row inside the loop.
    rlvr_filter = q.get("rlvr_status")  # "consumed" | "queued" | "unseen" | None
    if rlvr_filter:
        rmap = q.get("_rlvr_status_map") or {}
        st = rmap.get(row["_id"], "unseen")
        if st != rlvr_filter:
            return False

    qstr = (q.get("q") or "").strip().lower()
    if qstr:
        haystack = (
            f"{row.get('dataset', '')} {row.get('subject_id', '')} "
            f"{row.get('section_id', '')} {row.get('image_path', '')}"
        ).lower()
        if qstr not in haystack:
            return False

    return True


def _sort_inventory(rows: list[dict], sort_order: str) -> list[dict]:
    if sort_order == "random":
        import random as _rand

        rng = _rand.Random(0xCAFE)
        rows = list(rows)
        rng.shuffle(rows)
        return rows
    if sort_order == "filename":
        return sorted(rows, key=lambda r: r.get("image_path", ""))
    if sort_order == "dataset":
        return sorted(
            rows,
            key=lambda r: (
                r.get("dataset", ""),
                r.get("subject_id", ""),
                float(r.get("position_mm", 0.0) or 0.0),
            ),
        )
    # default: position
    return sorted(
        rows,
        key=lambda r: (
            r.get("dataset", ""),
            r.get("subject_id", ""),
            float(r.get("position_mm", 0.0) or 0.0),
        ),
    )


def _row_to_payload(
    row: dict,
    *,
    rlvr_status: dict[str, dict] | None = None,
) -> dict:
    """Project a manifest row down to what the UI actually needs.

    The full row carries per-dataset provenance metadata that isn't useful
    in the grid view; trimming makes pages noticeably smaller over the wire.

    ``rlvr_status``: optional id -> {status, step, ts, kind, phase, queued_again}
    map (from ``training_progress.status_per_id``). When provided, each rlvr
    row gets its consumed/queued/unseen markers attached.
    """
    payload = {
        "id": row["_id"],
        "dataset": row.get("dataset"),
        "subject_id": row.get("subject_id"),
        "section_id": row.get("section_id"),
        "image_path": row.get("image_path"),
        "atlas": row.get("atlas"),
        "orientation": row.get("orientation"),
        "slice_axis": row.get("slice_axis"),
        "position_mm": row.get("position_mm"),
        "species": row.get("species"),
        "imaging": row.get("imaging"),
        "staining": row.get("staining"),
        "registration_source": row.get("registration_source"),
        "is_hemisphere": row.get("is_hemisphere"),
        "exclude_from_training": bool(row.get("exclude_from_training", False)),
        "split": row.get("_split") or "",
    }
    if rlvr_status is not None and (row.get("_split") or "") == "rlvr":
        st = rlvr_status.get(row["_id"])
        if st is not None:
            payload["rlvr_status"] = st["status"]
            payload["rlvr_step"] = st["step"]
            payload["rlvr_ts"] = st["ts"]
            payload["rlvr_kind"] = st["kind"]
            payload["rlvr_queued_again"] = st["queued_again"]
        else:
            payload["rlvr_status"] = "unseen"
    return payload


def compute_inventory_facets(rows: list[dict]) -> dict:
    """Per-filter-value counts for the inventory UI."""
    datasets: dict[str, dict] = {}
    atlases: dict[str, dict] = {}
    planes: dict[str, dict] = {}
    species: dict[str, dict] = {}
    regs: dict[str, dict] = {}
    n_excluded = 0

    def _bump(d, key):
        if key not in d:
            d[key] = {"n_total": 0}
        d[key]["n_total"] += 1

    for row in rows:
        if row.get("exclude_from_training"):
            n_excluded += 1
        _bump(datasets, row.get("dataset", "?"))
        _bump(atlases, row.get("atlas") or "(none)")
        _bump(planes, row.get("orientation") or "(none)")
        _bump(species, row.get("species") or "(none)")
        _bump(regs, row.get("registration_source") or "(none)")

    def _shape(d: dict) -> list[dict]:
        return sorted(
            ({"value": k, "n_total": v["n_total"]} for k, v in d.items()),
            key=lambda r: (-r["n_total"], r["value"]),
        )

    return {
        "datasets": _shape(datasets),
        "atlases": _shape(atlases),
        "planes": _shape(planes),
        "species": _shape(species),
        "registration_sources": _shape(regs),
        "n_total": len(rows),
        "n_excluded": n_excluded,
    }


def compute_inventory_dataset_rollup(rows: list[dict]) -> list[dict]:
    """One row per dataset with everything Sources view needs in a single pass.

    Includes a representative `samples` list (anterior / middle / posterior)
    so the Sources view can render thumbnails without firing N more requests
    per dataset.
    """
    by_ds: dict[str, dict] = {}
    for row in rows:
        ds = row.get("dataset", "?")
        bucket = by_ds.setdefault(
            ds,
            {
                "dataset": ds,
                "n_total": 0,
                "n_excluded": 0,
                "planes": {},
                "atlases": set(),
                "species": set(),
                "registration_sources": set(),
                "subjects": set(),
                "_rows_for_samples": [],  # stripped before serializing
            },
        )
        bucket["n_total"] += 1
        if row.get("exclude_from_training"):
            bucket["n_excluded"] += 1
        plane = row.get("orientation") or "?"
        bucket["planes"][plane] = bucket["planes"].get(plane, 0) + 1
        if row.get("atlas"):
            bucket["atlases"].add(row["atlas"])
        if row.get("species"):
            bucket["species"].add(row["species"])
        if row.get("registration_source"):
            bucket["registration_sources"].add(row["registration_source"])
        if row.get("subject_id"):
            bucket["subjects"].add(row["subject_id"])
        bucket["_rows_for_samples"].append(row)

    out: list[dict] = []
    for _ds, b in by_ds.items():
        rs = b.pop("_rows_for_samples")
        # Pick anterior + middle + posterior representatives by position_mm.
        rs_with_pos = [r for r in rs if isinstance(r.get("position_mm"), (int, float))]
        rs_with_pos.sort(key=lambda r: float(r["position_mm"]))
        samples: list[dict] = []
        if rs_with_pos:
            picks = []
            picks.append(rs_with_pos[0])
            mid = rs_with_pos[len(rs_with_pos) // 2]
            if mid is not picks[0]:
                picks.append(mid)
            last = rs_with_pos[-1]
            if last is not picks[0] and last is not picks[-1]:
                picks.append(last)
            for r in picks:
                samples.append(
                    {
                        "id": r["_id"],
                        "image_path": r.get("image_path"),
                        "atlas": r.get("atlas"),
                        "orientation": r.get("orientation"),
                        "position_mm": r.get("position_mm"),
                        "subject_id": r.get("subject_id"),
                        "section_id": r.get("section_id"),
                    }
                )
        b["samples"] = samples
        b["atlases"] = sorted(b["atlases"])
        b["species"] = sorted(b["species"])
        b["registration_sources"] = sorted(b["registration_sources"])
        b["n_subjects"] = len(b["subjects"])
        b.pop("subjects", None)
        out.append(b)

    out.sort(key=lambda d: (-d["n_total"], d["dataset"]))
    return out


def compute_training_summary(
    rows: list[dict],
    progress: TrainingProgress | None,
) -> dict:
    """Per-split rollup for the Training tab's stats strip.

    Returns counts for eval/sft/rlvr (and any other splits present), plus an
    rlvr-specific block with consumed / queued / unseen tallies. ``progress``
    may be ``None`` if no progress dir is configured -- in that case the
    rlvr block reports zero consumed/queued and everything is "unseen".
    """
    by_split: dict[str, dict] = {}
    for r in rows:
        s = r.get("_split") or "(none)"
        bucket = by_split.setdefault(
            s,
            {
                "split": s,
                "n_total": 0,
                "datasets": {},  # ds -> count
                "planes": {},
                "atlases": {},
            },
        )
        bucket["n_total"] += 1
        ds = r.get("dataset") or "?"
        bucket["datasets"][ds] = bucket["datasets"].get(ds, 0) + 1
        plane = r.get("orientation") or "?"
        bucket["planes"][plane] = bucket["planes"].get(plane, 0) + 1
        atlas = r.get("atlas") or "?"
        bucket["atlases"][atlas] = bucket["atlases"].get(atlas, 0) + 1

    # rlvr-specific consumed / queued / unseen.
    rlvr = by_split.get("rlvr")
    rlvr_block: dict
    if rlvr is None:
        rlvr_block = {
            "n_total": 0,
            "n_consumed": 0,
            "n_queued": 0,
            "n_consumed_and_queued": 0,
            "n_unseen": 0,
        }
    else:
        rlvr_ids = {r["_id"] for r in rows if (r.get("_split") or "") == "rlvr"}
        if progress is not None:
            progress.refresh()
            consumed_map = progress.consumed_by_id()
            queued_set = progress.queued_ids()
        else:
            consumed_map = {}
            queued_set = set()
        consumed_in_split = rlvr_ids & set(consumed_map.keys())
        queued_in_split = rlvr_ids & queued_set
        rlvr_block = {
            "n_total": len(rlvr_ids),
            "n_consumed": len(consumed_in_split),
            "n_queued": len(queued_in_split),
            "n_consumed_and_queued": len(consumed_in_split & queued_in_split),
            "n_unseen": len(rlvr_ids - consumed_in_split - queued_in_split),
        }

    return {
        "splits": list(by_split.values()),
        "rlvr": rlvr_block,
    }


# ----------------------------------------------------------------------
# Histology / synth thumbnailing (server-side, cached on disk + memory).
#
# Most images are several MB. The grid view renders 30+ tiles per page;
# sending originals would saturate localhost loopback. Resize once, cache
# by (resolved-path, w, h, mtime), serve as JPEG.


_thumb_lock = threading.Lock()
_thumb_mem_cache: OrderedDict[tuple[str, int, int, float], bytes] = OrderedDict()
_THUMB_MEM_MAX = 512


def _thumb_disk_path(cache_key: str, width: int, height: int, mtime: float) -> Path:
    safe = cache_key.replace("\\", "/").replace("/", "_").replace(":", "_")
    stamp = int(mtime)
    return INVENTORY_THUMB_CACHE / f"{width}x{height}" / f"{safe}.{stamp}.jpg"


def render_thumbnail(
    path: str,
    width: int,
    height: int,
    *,
    base_dir: Path | None = None,
) -> tuple[bytes | None, str]:
    """Return (jpeg_bytes, content_type) or (None, error_message).

    Resizes preserving aspect ratio so the longest side fits within the
    requested box. Returns JPEG (small + fast). Cache key is the resolved
    absolute path string + size + mtime so edits on disk invalidate stale
    thumbs without manual cache clears.

    If ``base_dir`` is None, ``path`` is resolved against ``REPO_ROOT`` and
    the result must remain inside ``REPO_ROOT``. Otherwise it is resolved
    against ``base_dir.resolve()`` and confined inside that base.
    """
    if base_dir is None:
        root = REPO_ROOT.resolve()
    else:
        root = base_dir.resolve()
    candidate = (root / path).resolve()
    if not str(candidate).startswith(str(root)):
        return None, "forbidden"
    if not candidate.is_file():
        return None, "not_found"

    try:
        mtime = candidate.stat().st_mtime
    except OSError:
        return None, "stat_failed"

    cache_key = str(candidate)
    key = (cache_key, width, height, mtime)

    with _thumb_lock:
        cached = _thumb_mem_cache.get(key)
        if cached is not None:
            _thumb_mem_cache.move_to_end(key)
            return cached, "image/jpeg"

    disk_path = _thumb_disk_path(cache_key, width, height, mtime)
    if disk_path.exists():
        try:
            data = disk_path.read_bytes()
            with _thumb_lock:
                _thumb_mem_cache[key] = data
                while len(_thumb_mem_cache) > _THUMB_MEM_MAX:
                    _thumb_mem_cache.popitem(last=False)
            return data, "image/jpeg"
        except OSError:
            pass

    try:
        from PIL import Image

        with Image.open(candidate) as im:
            im = im.convert("RGB")
            im.thumbnail((width, height), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=82, optimize=True)
            data = buf.getvalue()
    except Exception as exc:
        return None, f"resize_failed: {exc}"

    try:
        disk_path.parent.mkdir(parents=True, exist_ok=True)
        disk_path.write_bytes(data)
    except OSError as exc:
        logger.warning("thumb-cache write failed for %s: %s", disk_path, exc)

    with _thumb_lock:
        _thumb_mem_cache[key] = data
        while len(_thumb_mem_cache) > _THUMB_MEM_MAX:
            _thumb_mem_cache.popitem(last=False)
    return data, "image/jpeg"


# ----------------------------------------------------------------------
# Inventory App


class InventoryApp:
    """Holds the inventory manifest; reloads on mtime change."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        training_progress: TrainingProgress | None = None,
        allocations_root: Path | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.training_progress = training_progress
        self.allocations_root: Path = (
            allocations_root if allocations_root is not None else _DEFAULT_ALLOCATIONS_ROOT
        )
        self._manifest_mtime: float = 0.0
        self.rows: list[dict] = []
        self._reload_if_stale()

    def rlvr_split_ids(self) -> list[str]:
        """All row ids whose _split == 'rlvr'. Used for unseen-count math."""
        self._reload_if_stale()
        return [r["_id"] for r in self.rows if (r.get("_split") or "") == "rlvr"]

    def split_counts(self) -> dict[str, int]:
        self._reload_if_stale()
        out: dict[str, int] = {}
        for r in self.rows:
            s = r.get("_split") or "(none)"
            out[s] = out.get(s, 0) + 1
        return out

    def _current_mtime(self) -> float | None:
        """Max mtime across the manifest source and allocation files.

        For a sharded directory this is the max mtime over all shard files,
        so any single shard rebuild triggers a reload. For a legacy single
        file, just that file's mtime.  Allocation files are also watched so
        that adding rows to an allocation file triggers a re-stamp of _split.
        """
        path = self.manifest_path
        mtimes: list[float] = []
        try:
            if path.is_dir():
                files = _iter_shard_files(path)
                if not files:
                    return None
                mtimes.extend(f.stat().st_mtime for f in files)
            else:
                mtimes.append(path.stat().st_mtime)
        except FileNotFoundError:
            return None
        # Also watch allocation files so splits refresh when they change.
        for alloc_file in _iter_allocation_files(self.allocations_root):
            try:
                mtimes.append(alloc_file.stat().st_mtime)
            except OSError:
                pass
        return max(mtimes) if mtimes else None

    def _reload_if_stale(self) -> bool:
        mt = self._current_mtime()
        if mt is None:
            return False
        if mt != self._manifest_mtime:
            self.rows = load_inventory_manifest(
                self.manifest_path, allocations_root=self.allocations_root
            )
            self._manifest_mtime = mt
            logger.info("(re)loaded inventory manifest: %d rows", len(self.rows))
            return True
        return False

    def filter_rows(self, q: dict) -> list[dict]:
        self._reload_if_stale()
        # If the query asks for rlvr_status filtering, snapshot the live
        # progress once into the query dict so _matches_inventory_filter
        # doesn't have to look it up per row.
        if q.get("rlvr_status") and self.training_progress is not None:
            self.training_progress.refresh()
            consumed = self.training_progress.consumed_by_id()
            queued = self.training_progress.queued_ids()
            rlvr_map: dict[str, str] = {}
            for r in self.rows:
                rid = r["_id"]
                if rid in consumed:
                    rlvr_map[rid] = "consumed"
                elif rid in queued:
                    rlvr_map[rid] = "queued"
                else:
                    rlvr_map[rid] = "unseen"
            q = dict(q)
            q["_rlvr_status_map"] = rlvr_map
        return [r for r in self.rows if _matches_inventory_filter(r, q)]


# ----------------------------------------------------------------------
# Synth data layer


def load_synth_manifest(path: Path) -> list[dict]:
    """Load `_local/synth_data/manifest.jsonl`. Empty if file missing."""
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            s = raw.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError:
                continue
            seed = row.get("seed")
            try:
                seed_int = int(seed) if seed is not None else 0
            except (TypeError, ValueError):
                seed_int = 0
            row["_id"] = f"synth/{seed_int:08d}"
            rows.append(row)
    return rows


def _matches_synth_filter(row: dict, q: dict) -> bool:
    plane = q.get("plane")
    if plane and plane != "all" and row.get("plane") != plane:
        return False

    modality = q.get("modality")
    if modality and row.get("modality") != modality:
        return False

    atlas = q.get("atlas")
    if atlas and row.get("atlas_name") != atlas:
        return False

    mode = q.get("mode")
    if mode and row.get("mode") != mode:
        return False

    apply_damage = q.get("apply_damage")  # "true" | "false" | "all"
    if apply_damage and apply_damage != "all":
        is_damaged = bool(row.get("apply_damage"))
        if apply_damage == "true" and not is_damaged:
            return False
        if apply_damage == "false" and is_damaged:
            return False

    damage_intensity = q.get("damage_intensity")
    if damage_intensity and (row.get("damage_intensity") or "") != damage_intensity:
        return False

    qstr = (q.get("q") or "").strip().lower()
    if qstr:
        haystack = (
            f"{row.get('modality', '')} {row.get('atlas_name', '')} "
            f"{row.get('mode', '')} {row.get('seed', '')}"
        ).lower()
        if qstr not in haystack:
            return False

    return True


def _sort_synth(rows: list[dict], sort_order: str) -> list[dict]:
    if sort_order == "modality":
        return sorted(
            rows,
            key=lambda r: (
                r.get("modality", ""),
                r.get("atlas_name", ""),
                r.get("plane", ""),
                float(r.get("position_mm", 0.0) or 0.0),
            ),
        )
    if sort_order == "seed":
        return sorted(rows, key=lambda r: int(r.get("seed", 0) or 0))
    if sort_order == "random":
        import random as _rand

        rng = _rand.Random(0xCAFE)
        rows = list(rows)
        rng.shuffle(rows)
        return rows
    # default: position
    return sorted(
        rows,
        key=lambda r: (
            r.get("atlas_name", ""),
            r.get("plane", ""),
            float(r.get("position_mm", 0.0) or 0.0),
        ),
    )


def _synth_row_to_payload(row: dict, *, synth_dir: Path | None = None) -> dict:
    seed = row.get("seed", 0)
    try:
        seed_int = int(seed)
    except (TypeError, ValueError):
        seed_int = 0
    # Surface the underlying PNG's mtime so the frontend can build a
    # versioned URL (?v=<mtime>) that defeats the browser's in-document
    # memory cache after a regen. Cache-Control: no-store on its own is
    # not enough because the browser keeps the same-URL bytes alive in
    # the document's image cache until the document tears down.
    image_mtime = 0.0
    if synth_dir is not None:
        try:
            image_mtime = (synth_dir / "images" / f"{seed_int:08d}.png").stat().st_mtime
        except OSError:
            pass
    return {
        "id": row["_id"],
        "seed": seed_int,
        "modality": row.get("modality"),
        "mode": row.get("mode"),
        "atlas_name": row.get("atlas_name"),
        "plane": row.get("plane"),
        "position_mm": row.get("position_mm"),
        "apply_damage": bool(row.get("apply_damage", False)),
        "damage_intensity": row.get("damage_intensity"),
        "image_path": f"images/{seed_int:08d}.png",
        "image_mtime": image_mtime,
    }


def compute_synth_facets(rows: list[dict]) -> dict:
    """Counts by modality, atlas_name, plane, mode, damage_intensity, apply_damage."""
    modality: dict[str, int] = {}
    atlas: dict[str, int] = {}
    planes: dict[str, int] = {}
    modes: dict[str, int] = {}
    damage_intensities: dict[str, int] = {}
    apply_damage = {"true": 0, "false": 0}

    for r in rows:
        m = r.get("modality") or "(none)"
        modality[m] = modality.get(m, 0) + 1
        a = r.get("atlas_name") or "(none)"
        atlas[a] = atlas.get(a, 0) + 1
        p = r.get("plane") or "(none)"
        planes[p] = planes.get(p, 0) + 1
        mo = r.get("mode") or "(none)"
        modes[mo] = modes.get(mo, 0) + 1
        di = r.get("damage_intensity") or "(none)"
        damage_intensities[di] = damage_intensities.get(di, 0) + 1
        if bool(r.get("apply_damage", False)):
            apply_damage["true"] += 1
        else:
            apply_damage["false"] += 1

    def _shape(d: dict) -> list[dict]:
        return sorted(
            ({"value": k, "n_total": v} for k, v in d.items()),
            key=lambda r: (-r["n_total"], r["value"]),
        )

    return {
        "modality": _shape(modality),
        "atlases": _shape(atlas),
        "planes": _shape(planes),
        "modes": _shape(modes),
        "damage_intensities": _shape(damage_intensities),
        "apply_damage": apply_damage,
        "n_total": len(rows),
    }


def compute_synth_rollup(rows: list[dict], *, synth_dir: Path | None = None) -> list[dict]:
    """Group by (modality, atlas_name, plane). Each bucket has a small samples list."""
    by_key: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        key = (
            row.get("modality") or "?",
            row.get("atlas_name") or "?",
            row.get("plane") or "?",
        )
        bucket = by_key.setdefault(
            key,
            {
                "dataset": f"{key[0]}/{key[1]}/{key[2]}",
                "modality": key[0],
                "atlas_name": key[1],
                "plane": key[2],
                "n_total": 0,
                "modes": set(),
                "damage_intensities": set(),
                "_rows_for_samples": [],
            },
        )
        bucket["n_total"] += 1
        if row.get("mode"):
            bucket["modes"].add(row["mode"])
        if row.get("damage_intensity"):
            bucket["damage_intensities"].add(row["damage_intensity"])
        bucket["_rows_for_samples"].append(row)

    out: list[dict] = []
    for _key, b in by_key.items():
        rs = b.pop("_rows_for_samples")
        rs_with_pos = [r for r in rs if isinstance(r.get("position_mm"), (int, float))]
        rs_with_pos.sort(key=lambda r: float(r["position_mm"]))
        samples: list[dict] = []
        if rs_with_pos:
            picks = []
            picks.append(rs_with_pos[0])
            mid = rs_with_pos[len(rs_with_pos) // 2]
            if mid is not picks[0]:
                picks.append(mid)
            last = rs_with_pos[-1]
            if last is not picks[0] and last is not picks[-1]:
                picks.append(last)
            for r in picks:
                seed = r.get("seed", 0)
                try:
                    seed_int = int(seed)
                except (TypeError, ValueError):
                    seed_int = 0
                image_mtime = 0.0
                if synth_dir is not None:
                    try:
                        image_mtime = (synth_dir / "images" / f"{seed_int:08d}.png").stat().st_mtime
                    except OSError:
                        pass
                samples.append(
                    {
                        "seed": seed_int,
                        "position_mm": r.get("position_mm"),
                        "image_mtime": image_mtime,
                    }
                )
        b["samples"] = samples
        b["modes"] = sorted(b["modes"])
        b["damage_intensities"] = sorted(b["damage_intensities"])
        out.append(b)

    out.sort(key=lambda d: (-d["n_total"], d["dataset"]))
    return out


class SynthApp:
    """Holds the synth manifest; reloads on mtime change."""

    def __init__(self, synth_dir: Path) -> None:
        self.synth_dir = synth_dir
        self.manifest_path = synth_dir / "manifest.jsonl"
        self._manifest_mtime: float = 0.0
        self.rows: list[dict] = []
        self._reload_if_stale()

    def _reload_if_stale(self) -> bool:
        try:
            mt = self.manifest_path.stat().st_mtime
        except FileNotFoundError:
            if self.rows:
                self.rows = []
                self._manifest_mtime = 0.0
                return True
            return False
        if mt != self._manifest_mtime:
            self.rows = load_synth_manifest(self.manifest_path)
            self._manifest_mtime = mt
            logger.info("(re)loaded synth manifest: %d rows", len(self.rows))
            return True
        return False

    def filter_rows(self, q: dict) -> list[dict]:
        self._reload_if_stale()
        return [r for r in self.rows if _matches_synth_filter(r, q)]


# ----------------------------------------------------------------------
# SFT trace data layer


def discover_sft_manifests(sft_dir: Path) -> list[Path]:
    """Discover SFT trace manifests in priority order.

    Picks up both `*_trace_manifest_qc.jsonl` (the pre-explode QC manifest
    used for verdict collection) and `*_trace_manifest_verified.jsonl` (the
    post-explode trace-collection input — what `langslice collect-traces`
    actually ran against). Returned sorted with `_verified` files LAST so
    they overwrite QC entries on duplicate ids during merge — the verified
    manifest's exploded singles have IDs that match results.jsonl.
    """
    if not sft_dir.is_dir():
        return []
    qc = sorted(sft_dir.glob("*_trace_manifest_qc.jsonl"))
    verified = sorted(sft_dir.glob("*_trace_manifest_verified.jsonl"))
    return qc + verified


def discover_sft_results(sft_dir: Path) -> list[Path]:
    """rglob `results.jsonl` under any `sft_dir/runs*` directory."""
    if not sft_dir.is_dir():
        return []
    out: list[Path] = []
    for runs_dir in sorted(sft_dir.glob("runs*")):
        if runs_dir.is_dir():
            out.extend(sorted(runs_dir.rglob("results.jsonl")))
    return out


def discover_sft_rescues(sft_dir: Path) -> list[Path]:
    """rglob `rescued.jsonl` under any `sft_dir/runs*` directory."""
    if not sft_dir.is_dir():
        return []
    out: list[Path] = []
    for runs_dir in sorted(sft_dir.glob("runs*")):
        if runs_dir.is_dir():
            out.extend(sorted(runs_dir.rglob("rescued.jsonl")))
    return out


def discover_sft_raw_traces(sft_dir: Path) -> list[Path]:
    """Glob `raw_trace.json` under `sft_dir/runs*/batch_*/runs/*/`."""
    if not sft_dir.is_dir():
        return []
    out: list[Path] = []
    for runs_dir in sorted(sft_dir.glob("runs*")):
        if runs_dir.is_dir():
            out.extend(sorted(runs_dir.glob("batch_*/runs/*/raw_trace.json")))
    return out


def load_sft_rescues(sft_dir: Path) -> dict[str, dict]:
    """Map trace id -> rescued payload (max_error_mm, truth_mm, submitted_mm)."""
    out: dict[str, dict] = {}
    for path in discover_sft_rescues(sft_dir):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                tid = rec.get("id")
                if tid:
                    out[tid] = rec
        except Exception as exc:
            logger.warning("SFT rescue load failed for %s: %s", path, exc)
    return out


def load_sft_estimates_best_per_id(sft_dir: Path) -> dict[str, dict]:
    """Walk every raw_trace.json under sft_dir/runs*/ and return the lowest-error
    trace per id, scored against the *current* canonical + plane-relative metric.

    Re-rolling produces multiple traces per slice across `runs_reroll_*` dirs.
    The bare `update`-based merge of per-file results.jsonl is last-wins, which
    can pick a re-roll that *worsened* a slice. This function re-scores every
    raw trace with the live tolerance helpers so the SFT view always reflects
    the best trace we have for each id.
    """
    try:
        from langslice_harness.harness.estimation.trace_collection import (
            canonicalize_positions,
            plane_rescue_threshold_mm,
            plane_tolerance_mm,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("could not import scoring helpers: %s", exc)
        return {}

    best: dict[str, dict] = {}
    attempts: dict[str, int] = {}

    for runs_dir in sorted(sft_dir.glob("runs*")):
        if not runs_dir.is_dir():
            continue
        for raw in runs_dir.glob("batch_*/runs/*/raw_trace.json"):
            try:
                d = json.loads(raw.read_text(encoding="utf-8"))
            except Exception:
                continue
            m = d.get("manifest") or {}
            sid = m.get("id")
            if not sid or m.get("kind") != "single":
                continue
            plane = m.get("plane")
            atlas = m.get("atlas")
            sub = d.get("submitted_positions_mm") or []
            tru = m.get("truth_positions_mm") or []
            if not plane or not atlas or not sub or not tru:
                continue

            attempts[sid] = attempts.get(sid, 0) + 1

            sub_c = canonicalize_positions(sub, atlas, plane) or sub
            tru_c = canonicalize_positions(tru, atlas, plane) or tru
            err = abs(float(sub_c[0]) - float(tru_c[0]))
            tol = plane_tolerance_mm(atlas, plane)
            rescue = plane_rescue_threshold_mm(atlas, plane)

            cat = d.get("category") or {}
            sft = d.get("sft_quality") or {}
            sft_notes_parts: list[str] = []
            for key in ("missing_rationale_before_tool", "overlong_rationale"):
                v = sft.get(key)
                if v:
                    sft_notes_parts.append(f"{key}={v}")
            tool_turns = sft.get("assistant_tool_turns")
            if tool_turns is not None:
                sft_notes_parts.append(f"turns={tool_turns}")

            scored = {
                "positions_mm": [float(s) for s in sub],
                "submitted_canonical_mm": [float(c) for c in sub_c],
                "truth_positions_mm": [float(t) for t in tru],
                "truth_canonical_mm": [float(c) for c in tru_c],
                "max_error_mm": err,
                "tolerance_mm": tol,
                "rescue_threshold_mm": rescue,
                "accepted": err <= tol,
                "rescuable": err <= rescue,
                "label": (
                    "clean_success"
                    if err <= tol
                    else "rescuable"
                    if err <= rescue
                    else cat.get("label") or "out_of_tolerance"
                ),
                "cost_usd": d.get("estimated_cost_usd"),
                "model": d.get("model"),
                "sft_accepted": sft.get("accepted_for_sft"),
                "sft_notes": ", ".join(sft_notes_parts) if sft_notes_parts else None,
                "source_file": str(raw.relative_to(REPO_ROOT))
                if REPO_ROOT in raw.parents
                else str(raw),
            }

            existing = best.get(sid)
            if existing is None or err < existing["max_error_mm"]:
                best[sid] = scored

    for sid, info in best.items():
        info["attempts"] = attempts.get(sid, 1)
    return best


def load_sft_traces(
    sft_dir: Path,
) -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    """Returns (manifest_entries, estimates_by_id, rescues_by_id).

    The SFT section is for inspecting traces we actually collected, so this
    function returns only entries that have a corresponding trace in
    results.jsonl. Manifest entries with `kind=group` are also dropped —
    every group was exploded into per-slice singles before trace collection,
    so the group row itself is never a trace target. Drop it to avoid
    showing duplicate-feeling rows that have no estimate.

    Manifests are still merged by id (verified wins over qc) so the right
    metadata reaches each surviving entry.

    Estimates are picked best-per-id (lowest canonical error) across every
    `runs*` directory, so re-rolls that improved a slice win and re-rolls
    that worsened it are ignored.
    """
    entries_by_id: dict[str, dict] = {}
    for mp in discover_sft_manifests(sft_dir):
        try:
            for entry in load_manifest(mp):
                eid = entry.get("id")
                if eid:
                    entries_by_id[eid] = entry
        except Exception as exc:
            logger.warning("SFT manifest load failed for %s: %s", mp, exc)

    estimates = load_sft_estimates_best_per_id(sft_dir)

    rescues = load_sft_rescues(sft_dir)

    kept = [
        e for e in entries_by_id.values() if e.get("kind") != "group" and e.get("id") in estimates
    ]
    return kept, estimates, rescues


def _sft_entry_to_payload(entry: dict, est: dict | None, rescue: dict | None = None) -> dict:
    kind = entry.get("kind", "single")
    if kind == "group":
        positions = list(entry.get("positions_mm", []))
        images = list(entry.get("images", []))
    else:
        positions = [float(entry.get("position_mm", 0.0))]
        images = [entry.get("image", "")]
    meta = entry.get("metadata", {}) or {}
    image = images[0] if images else ""
    position = positions[0] if positions else 0.0
    truth_positions = None
    submitted = None
    max_error = None
    accepted = None
    label = None
    cost_usd = None
    sft_accepted = None
    sft_notes = None
    source_file = None
    if est is not None:
        truth_positions = est.get("truth_positions_mm")
        submitted = est.get("positions_mm")
        max_error = est.get("max_error_mm")
        accepted = est.get("accepted")
        label = est.get("label")
        cost_usd = est.get("cost_usd")
        sft_accepted = est.get("sft_accepted")
        sft_notes = est.get("sft_notes")
        source_file = est.get("source_file")
    return {
        "id": entry.get("id", ""),
        "source_dataset": meta.get("source_dataset", "?"),
        "subject_id": meta.get("subject_id", "?"),
        "kind": kind,
        "image": image,
        "images": images,
        "position_mm": position,
        "positions_mm": positions,
        "atlas": entry.get("atlas", ""),
        "plane": entry.get("plane", ""),
        "truth_positions_mm": truth_positions,
        "submitted_positions_mm": submitted,
        "max_error_mm": max_error,
        "accepted": accepted,
        "label": label,
        "cost_usd": cost_usd,
        "sft_accepted": sft_accepted,
        "sft_notes": sft_notes,
        "source_file": source_file,
        "rescued": rescue is not None,
        "rescue_original_mm": (rescue.get("submitted_mm") if rescue else None),
    }


def compute_sft_rollup(
    entries: list[dict],
    estimates: dict[str, dict],
    rescues: dict[str, dict] | None = None,
) -> list[dict]:
    """Group by source_dataset. Returns one bucket per dataset."""
    rescues = rescues or {}
    by_ds: dict[str, dict] = {}
    for e in entries:
        meta = e.get("metadata", {}) or {}
        ds = meta.get("source_dataset", "?")
        bucket = by_ds.setdefault(
            ds,
            {
                "dataset": ds,
                "n_total": 0,
                "n_with_estimate": 0,
                "n_accepted": 0,
                "n_rescued": 0,
                "planes": {},
                "atlases": set(),
                "subjects": set(),
                "_entries_for_samples": [],
            },
        )
        bucket["n_total"] += 1
        eid = e.get("id")
        if eid in estimates:
            bucket["n_with_estimate"] += 1
            if estimates[eid].get("accepted"):
                bucket["n_accepted"] += 1
        if eid in rescues:
            bucket["n_rescued"] += 1
        plane = e.get("plane") or "?"
        bucket["planes"][plane] = bucket["planes"].get(plane, 0) + 1
        if e.get("atlas"):
            bucket["atlases"].add(e["atlas"])
        if meta.get("subject_id"):
            bucket["subjects"].add(meta["subject_id"])
        bucket["_entries_for_samples"].append(e)

    out: list[dict] = []
    for _ds, b in by_ds.items():
        es = b.pop("_entries_for_samples")

        def _first_pos(en: dict) -> float:
            if en.get("kind") == "group":
                positions = en.get("positions_mm") or [0.0]
                return float(positions[0])
            return float(en.get("position_mm", 0.0) or 0.0)

        es_sorted = sorted(es, key=_first_pos)
        samples: list[str] = []
        if es_sorted:
            picks = []
            picks.append(es_sorted[0])
            mid = es_sorted[len(es_sorted) // 2]
            if mid is not picks[0]:
                picks.append(mid)
            last = es_sorted[-1]
            if last is not picks[0] and last is not picks[-1]:
                picks.append(last)
            samples = [p.get("id", "") for p in picks if p.get("id")]
        b["samples"] = samples
        b["atlases"] = sorted(b["atlases"])
        b["n_subjects"] = len(b["subjects"])
        b.pop("subjects", None)
        out.append(b)

    out.sort(key=lambda d: (-d["n_total"], d["dataset"]))
    return out


class SftApp:
    """Holds SFT trace manifests + estimates; reloads on mtime change.

    Tracks the mtime of every discovered manifest + results file; if any
    has changed (or the set has changed), reloads from scratch.
    """

    def __init__(self, sft_dir: Path) -> None:
        self.sft_dir = sft_dir
        self._mtimes: dict[Path, float] = {}
        self.entries: list[dict] = []
        self.estimates: dict[str, dict] = {}
        self.rescues: dict[str, dict] = {}
        self._reload_if_stale()

    def _current_mtimes(self) -> dict[Path, float]:
        out: dict[Path, float] = {}
        for p in discover_sft_manifests(self.sft_dir):
            try:
                out[p] = p.stat().st_mtime
            except OSError:
                continue
        for p in discover_sft_results(self.sft_dir):
            try:
                out[p] = p.stat().st_mtime
            except OSError:
                continue
        for p in discover_sft_rescues(self.sft_dir):
            try:
                out[p] = p.stat().st_mtime
            except OSError:
                continue
        for p in discover_sft_raw_traces(self.sft_dir):
            try:
                out[p] = p.stat().st_mtime
            except OSError:
                continue
        return out

    def _reload_if_stale(self) -> bool:
        current = self._current_mtimes()
        if current == self._mtimes:
            return False
        self.entries, self.estimates, self.rescues = load_sft_traces(self.sft_dir)
        self._mtimes = current
        logger.info(
            "(re)loaded SFT traces: %d entries, %d estimates, %d rescues",
            len(self.entries),
            len(self.estimates),
            len(self.rescues),
        )
        return True

    def filter_entries(self, q: dict) -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
        self._reload_if_stale()
        source_dataset = q.get("source_dataset")
        plane = q.get("plane")
        atlas = q.get("atlas")
        qstr = (q.get("q") or "").strip().lower()

        kept: list[dict] = []
        for e in self.entries:
            meta = e.get("metadata", {}) or {}
            if source_dataset and meta.get("source_dataset") != source_dataset:
                continue
            if plane and plane != "all" and e.get("plane") != plane:
                continue
            if atlas and e.get("atlas") != atlas:
                continue
            if qstr:
                hay = (
                    f"{meta.get('source_dataset', '')} {meta.get('subject_id', '')} "
                    f"{e.get('id', '')} {e.get('atlas', '')} {e.get('plane', '')}"
                ).lower()
                if qstr not in hay:
                    continue
            kept.append(e)
        return kept, self.estimates, self.rescues


# ----------------------------------------------------------------------
# HTTP server


class Handler(BaseHTTPRequestHandler):
    server_version = "LangSliceQC/0.2"

    def log_message(self, fmt: str, *args) -> None:  # quiet default access log
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("%s - " + fmt, self.address_string(), *args)

    # -- helpers --------------------------------------------------------
    def _send(
        self,
        status: int,
        body: bytes,
        *,
        content_type: str = "application/octet-stream",
        cache: bool = False,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", "public, max-age=86400")
        else:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, content_type="application/json")

    # -- routes ---------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - http.server interface
        url = urlparse(self.path)
        try:
            if url.path == "/" or url.path == "/index.html":
                self._serve_static("app.html")
                return
            if url.path.startswith("/static/"):
                self._serve_static(url.path[len("/static/") :])
                return

            # Inventory
            if url.path == "/api/inventory/manifest":
                self._handle_inventory_manifest(parse_qs(url.query))
                return
            if url.path == "/api/inventory/facets":
                self._handle_inventory_facets(parse_qs(url.query))
                return
            if url.path == "/api/inventory/datasets":
                self._handle_inventory_datasets(parse_qs(url.query))
                return
            if url.path == "/api/inventory/thumbnail":
                self._handle_inventory_thumbnail(parse_qs(url.query))
                return

            # Training: shared atlas + image, RLVR live progress, SFT traces, Eval brains
            if url.path == "/api/training/status":
                self._handle_training_status(parse_qs(url.query))
                return
            if url.path == "/api/training/recent":
                self._handle_training_recent(parse_qs(url.query))
                return
            if url.path == "/api/training/summary":
                self._handle_training_summary(parse_qs(url.query))
                return
            if url.path == "/api/training/eval":
                self._handle_training_eval(parse_qs(url.query))
                return

            # RLVR
            if url.path == "/api/rlvr/sources":
                self._handle_rlvr_sources(parse_qs(url.query))
                return
            if url.path == "/api/rlvr/manifest":
                self._handle_rlvr_manifest(parse_qs(url.query))
                return

            # SFT
            if url.path == "/api/sft/sources":
                self._handle_sft_sources(parse_qs(url.query))
                return
            if url.path == "/api/sft/manifest":
                self._handle_sft_manifest(parse_qs(url.query))
                return

            # Synth
            if url.path == "/api/synth/manifest":
                self._handle_synth_manifest(parse_qs(url.query))
                return
            if url.path == "/api/synth/datasets":
                self._handle_synth_datasets(parse_qs(url.query))
                return
            if url.path == "/api/synth/facets":
                self._handle_synth_facets(parse_qs(url.query))
                return
            if url.path == "/api/synth/image":
                self._handle_synth_image(parse_qs(url.query))
                return
            if url.path == "/api/synth/thumbnail":
                self._handle_synth_thumbnail(parse_qs(url.query))
                return

            # Shared
            if url.path == "/api/atlas":
                self._handle_atlas(parse_qs(url.query))
                return
            if url.path == "/api/image":
                self._handle_image(parse_qs(url.query))
                return

            self._send(HTTPStatus.NOT_FOUND, b"not found", content_type="text/plain")
        except Exception as exc:
            logger.exception("GET %s failed", self.path)
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        # Verdict workflows are gone; no POST endpoints remain. Read body to
        # keep the connection clean (some clients send JSON with content-length
        # even on read-only ops) and 404.
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length:
                self.rfile.read(length)
        except Exception:
            pass
        self._send(HTTPStatus.NOT_FOUND, b"not found", content_type="text/plain")

    # -- handlers -------------------------------------------------------
    def _serve_static(self, name: str) -> None:
        target = (STATIC_DIR / name).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
            self._send(HTTPStatus.NOT_FOUND, b"not found", content_type="text/plain")
            return
        ext = target.suffix.lower()
        ct = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".svg": "image/svg+xml",
        }.get(ext, "application/octet-stream")
        self._send(HTTPStatus.OK, target.read_bytes(), content_type=ct, cache=False)

    def _handle_atlas(self, qs: dict) -> None:
        try:
            atlas = qs["atlas"][0]
            plane = qs["plane"][0]
            pos = float(qs["pos"][0])
        except (KeyError, IndexError, ValueError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"bad query: {exc}"})
            return
        mode = qs.get("mode", ["composite"])[0]
        png = render_atlas_png(atlas, plane, pos, mode)
        self._send(HTTPStatus.OK, png, content_type="image/png", cache=True)

    def _handle_image(self, qs: dict) -> None:
        try:
            rel = qs["path"][0]
        except (KeyError, IndexError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"bad query: {exc}"})
            return
        candidate = (REPO_ROOT / rel).resolve()
        if not str(candidate).startswith(str(REPO_ROOT.resolve())):
            self._send(HTTPStatus.FORBIDDEN, b"forbidden", content_type="text/plain")
            return
        if not candidate.is_file():
            self._send(HTTPStatus.NOT_FOUND, b"not found", content_type="text/plain")
            return
        ct = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".tif": "image/tiff",
            ".tiff": "image/tiff",
        }.get(candidate.suffix.lower(), "application/octet-stream")
        self._send(HTTPStatus.OK, candidate.read_bytes(), content_type=ct, cache=True)

    # -- inventory handlers ---------------------------------------------
    def _inv_app(self) -> InventoryApp | None:
        return getattr(self.server, "inventory_app", None)

    def _synth_app(self) -> SynthApp | None:
        return getattr(self.server, "synth_app", None)

    def _sft_app(self) -> SftApp | None:
        return getattr(self.server, "sft_app", None)

    def _parse_inventory_query(self, qs: dict) -> dict:
        def _one(key: str) -> str | None:
            v = qs.get(key, [""])[0]
            return v if v else None

        def _multi(key: str) -> list[str] | None:
            # Accept ?dataset=a&dataset=b OR ?dataset=a,b
            raw = qs.get(key, [])
            vals: list[str] = []
            for item in raw:
                vals.extend(p for p in (item or "").split(",") if p)
            return vals or None

        return {
            "plane": _one("plane"),
            "datasets": _multi("dataset"),
            "subject": _one("subject"),
            "atlas": _one("atlas"),
            "species": _one("species"),
            "registration_source": _one("registration_source"),
            "exclude": (_one("exclude") or "false").lower(),
            "splits": _multi("split"),
            "rlvr_status": _one("rlvr_status"),
            "q": _one("q"),
        }

    def _handle_inventory_manifest(self, qs: dict) -> None:
        inv = self._inv_app()
        if inv is None:
            self._send_json(
                HTTPStatus.OK,
                {
                    "items": [],
                    "offset": 0,
                    "limit": 0,
                    "total": 0,
                    "sort": "position",
                },
            )
            return
        q = self._parse_inventory_query(qs)
        sort_order = (qs.get("sort", ["position"])[0] or "position").lower()
        try:
            limit = max(1, min(200, int(qs.get("limit", ["48"])[0] or 48)))
        except ValueError:
            limit = 48
        try:
            offset = max(0, int(qs.get("offset", ["0"])[0] or 0))
        except ValueError:
            offset = 0

        rows = inv.filter_rows(q)
        rows = _sort_inventory(rows, sort_order)
        total = len(rows)
        page = rows[offset : offset + limit]
        # Thread training-progress info into payloads when the result set
        # involves rlvr rows. Cheap (a hash lookup per row) and the UI uses
        # these to render the consumed/queued badges.
        rlvr_status_map: dict[str, dict] | None = None
        prog = getattr(inv, "training_progress", None)
        page_has_rlvr = any((r.get("_split") or "") == "rlvr" for r in page)
        if prog is not None and page_has_rlvr:
            ids_on_page = [r["_id"] for r in page if (r.get("_split") or "") == "rlvr"]
            rlvr_status_map = status_per_id(prog, ids_on_page)
        items = [_row_to_payload(r, rlvr_status=rlvr_status_map) for r in page]
        self._send_json(
            HTTPStatus.OK,
            {
                "items": items,
                "offset": offset,
                "limit": limit,
                "total": total,
                "sort": sort_order,
            },
        )

    def _handle_inventory_facets(self, qs: dict) -> None:
        inv = self._inv_app()
        if inv is None:
            self._send_json(
                HTTPStatus.OK,
                {
                    "datasets": [],
                    "atlases": [],
                    "planes": [],
                    "species": [],
                    "registration_sources": [],
                    "n_total": 0,
                    "n_excluded": 0,
                },
            )
            return
        q = self._parse_inventory_query(qs)
        rows = inv.filter_rows(q)
        facets = compute_inventory_facets(rows)
        self._send_json(HTTPStatus.OK, facets)

    def _handle_inventory_datasets(self, qs: dict) -> None:
        inv = self._inv_app()
        if inv is None:
            self._send_json(HTTPStatus.OK, {"datasets": [], "n_total_rows": 0})
            return
        q = self._parse_inventory_query(qs)
        rows = inv.filter_rows(q)
        rollup = compute_inventory_dataset_rollup(rows)
        self._send_json(HTTPStatus.OK, {"datasets": rollup, "n_total_rows": len(rows)})

    def _handle_inventory_thumbnail(self, qs: dict) -> None:
        try:
            rel = qs["path"][0]
        except (KeyError, IndexError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "path required"})
            return
        try:
            w = max(32, min(1024, int(qs.get("w", ["256"])[0] or 256)))
            h = max(32, min(1024, int(qs.get("h", ["256"])[0] or 256)))
        except ValueError:
            w, h = 256, 256
        data, ct = render_thumbnail(rel, w, h, base_dir=None)
        if data is None:
            status = HTTPStatus.NOT_FOUND if ct == "not_found" else HTTPStatus.BAD_REQUEST
            self._send(status, ct.encode("utf-8"), content_type="text/plain")
            return
        self._send(HTTPStatus.OK, data, content_type=ct, cache=True)

    # -- training-progress handlers -------------------------------------
    def _handle_training_status(self, qs: dict) -> None:
        """Live-poll endpoint. Returns counts + recent activity tail.

        Empty when no progress files exist yet (training not started). The
        ``rlvr`` block crosses the manifest's rlvr split with the consumed
        / queued sets so the UI can render unseen counts.
        """
        inv = self._inv_app()
        prog = getattr(inv, "training_progress", None) if inv else None
        try:
            recent_limit = max(1, min(200, int(qs.get("recent", ["20"])[0] or 20)))
        except ValueError:
            recent_limit = 20

        if prog is None:
            self._send_json(
                HTTPStatus.OK,
                {
                    "available": False,
                    "current_step": -1,
                    "n_consumed_ids": 0,
                    "n_queued_ids": 0,
                    "queue": {"step": -1, "ts": 0.0, "queued": []},
                    "recent": [],
                    "run": None,
                    "last_consumed_ts": None,
                    "polled_at": time.time(),
                    "rlvr": {
                        "n_total": 0,
                        "n_consumed": 0,
                        "n_queued": 0,
                        "n_consumed_and_queued": 0,
                        "n_unseen": 0,
                    },
                },
            )
            return

        payload = prog.status_payload(recent_limit=recent_limit)

        # Cross with the rlvr split for the unseen tally.
        rlvr_block = {
            "n_total": 0,
            "n_consumed": 0,
            "n_queued": 0,
            "n_consumed_and_queued": 0,
            "n_unseen": 0,
        }
        if inv is not None:
            rlvr_id_set = set(inv.rlvr_split_ids())
            consumed_set = set(prog.consumed_by_id().keys()) & rlvr_id_set
            queued_set = prog.queued_ids() & rlvr_id_set
            rlvr_block = {
                "n_total": len(rlvr_id_set),
                "n_consumed": len(consumed_set),
                "n_queued": len(queued_set),
                "n_consumed_and_queued": len(consumed_set & queued_set),
                "n_unseen": len(rlvr_id_set - consumed_set - queued_set),
            }
        payload["rlvr"] = rlvr_block
        self._send_json(HTTPStatus.OK, payload)

    def _handle_training_recent(self, qs: dict) -> None:
        inv = self._inv_app()
        prog = getattr(inv, "training_progress", None) if inv else None
        try:
            limit = max(1, min(500, int(qs.get("limit", ["50"])[0] or 50)))
        except ValueError:
            limit = 50
        if prog is None:
            self._send_json(HTTPStatus.OK, {"recent": []})
            return
        self._send_json(HTTPStatus.OK, {"recent": prog.recent_payload(limit=limit)})

    def _handle_training_summary(self, qs: dict) -> None:
        """Per-split rollup for the Training tab header.

        Honors the same query filters the manifest endpoint accepts (so the
        user can scope the rollup to e.g. a single dataset) -- but ignores
        the rlvr_status filter (which only applies to row pages).
        """
        inv = self._inv_app()
        if inv is None:
            self._send_json(
                HTTPStatus.OK,
                {
                    "splits": [],
                    "rlvr": {
                        "n_total": 0,
                        "n_consumed": 0,
                        "n_queued": 0,
                        "n_consumed_and_queued": 0,
                        "n_unseen": 0,
                    },
                },
            )
            return
        q = self._parse_inventory_query(qs)
        # Drop rlvr_status filter -- summary is over the full split membership.
        q = dict(q)
        q.pop("rlvr_status", None)
        rows = inv.filter_rows(q)
        prog = getattr(inv, "training_progress", None)
        self._send_json(HTTPStatus.OK, compute_training_summary(rows, prog))

    def _handle_training_eval(self, qs: dict) -> None:
        """Flat brain list for the Eval sub-tab.

        Returns one entry per (dataset, subject_id) for rows with
        split == 'eval', with three sample sections (anterior / middle /
        posterior).
        """
        inv = self._inv_app()
        if inv is None:
            self._send_json(HTTPStatus.OK, [])
            return
        rows = inv.filter_rows({"splits": ["eval"]})
        # Group by (dataset, subject_id).
        by_brain: dict[tuple[str, str], dict] = {}
        for r in rows:
            key = (r.get("dataset") or "?", r.get("subject_id") or "?")
            bucket = by_brain.setdefault(
                key,
                {
                    "dataset": key[0],
                    "subject_id": key[1],
                    "n_sections": 0,
                    "atlas": r.get("atlas"),
                    "plane": r.get("orientation"),
                    "_rows": [],
                },
            )
            bucket["n_sections"] += 1
            bucket["_rows"].append(r)

        out: list[dict] = []
        for _key, b in by_brain.items():
            rs = b.pop("_rows")
            rs_with_pos = [r for r in rs if isinstance(r.get("position_mm"), (int, float))]
            rs_with_pos.sort(key=lambda r: float(r["position_mm"]))
            samples: list[dict] = []
            if rs_with_pos:
                picks = []
                picks.append(rs_with_pos[0])
                mid = rs_with_pos[len(rs_with_pos) // 2]
                if mid is not picks[0]:
                    picks.append(mid)
                last = rs_with_pos[-1]
                if last is not picks[0] and last is not picks[-1]:
                    picks.append(last)
                for r in picks:
                    samples.append(
                        {
                            "image_path": r.get("image_path"),
                            "section_id": r.get("section_id"),
                            "position_mm": r.get("position_mm"),
                        }
                    )
            b["samples"] = samples
            out.append(b)
        out.sort(key=lambda d: (d["dataset"], d["subject_id"]))
        self._send_json(HTTPStatus.OK, out)

    # -- RLVR handlers --------------------------------------------------
    def _handle_rlvr_sources(self, qs: dict) -> None:
        inv = self._inv_app()
        if inv is None:
            self._send_json(HTTPStatus.OK, {"datasets": []})
            return
        q = self._parse_inventory_query(qs)
        # Always force rlvr filter, regardless of query input.
        q = dict(q)
        q["splits"] = ["rlvr"]
        rows = inv.filter_rows(q)
        rollup = compute_inventory_dataset_rollup(rows)
        # Augment with consumed/queued/unseen per dataset bucket if
        # training_progress is wired up.
        prog = getattr(inv, "training_progress", None)
        if prog is not None and rollup:
            prog.refresh()
            consumed = set(prog.consumed_by_id().keys())
            queued = prog.queued_ids()
            ids_by_ds: dict[str, set[str]] = {}
            for r in rows:
                ds = r.get("dataset", "?")
                ids_by_ds.setdefault(ds, set()).add(r["_id"])
            for bucket in rollup:
                ds_ids = ids_by_ds.get(bucket["dataset"], set())
                cons = ds_ids & consumed
                que = ds_ids & queued
                bucket["n_consumed"] = len(cons)
                bucket["n_queued"] = len(que)
                bucket["n_unseen"] = len(ds_ids - cons - que)
        self._send_json(HTTPStatus.OK, {"datasets": rollup})

    def _handle_rlvr_manifest(self, qs: dict) -> None:
        inv = self._inv_app()
        if inv is None:
            self._send_json(
                HTTPStatus.OK,
                {
                    "items": [],
                    "offset": 0,
                    "limit": 0,
                    "total": 0,
                    "sort": "position",
                },
            )
            return
        q = self._parse_inventory_query(qs)
        # Always force rlvr filter regardless of query input.
        q = dict(q)
        q["splits"] = ["rlvr"]
        sort_order = (qs.get("sort", ["position"])[0] or "position").lower()
        try:
            limit = max(1, min(200, int(qs.get("limit", ["48"])[0] or 48)))
        except ValueError:
            limit = 48
        try:
            offset = max(0, int(qs.get("offset", ["0"])[0] or 0))
        except ValueError:
            offset = 0

        rows = inv.filter_rows(q)
        rows = _sort_inventory(rows, sort_order)
        total = len(rows)
        page = rows[offset : offset + limit]
        rlvr_status_map: dict[str, dict] | None = None
        prog = getattr(inv, "training_progress", None)
        if prog is not None:
            ids_on_page = [r["_id"] for r in page]
            rlvr_status_map = status_per_id(prog, ids_on_page)
        items = [_row_to_payload(r, rlvr_status=rlvr_status_map) for r in page]
        self._send_json(
            HTTPStatus.OK,
            {
                "items": items,
                "offset": offset,
                "limit": limit,
                "total": total,
                "sort": sort_order,
            },
        )

    # -- SFT handlers ---------------------------------------------------
    def _parse_sft_query(self, qs: dict) -> dict:
        def _one(key: str) -> str | None:
            v = qs.get(key, [""])[0]
            return v if v else None

        return {
            "source_dataset": _one("source_dataset"),
            "plane": _one("plane"),
            "atlas": _one("atlas"),
            "q": _one("q"),
        }

    def _handle_sft_sources(self, qs: dict) -> None:
        sft = self._sft_app()
        if sft is None:
            self._send_json(HTTPStatus.OK, {"datasets": []})
            return
        entries, estimates, rescues = sft.filter_entries(self._parse_sft_query(qs))
        self._send_json(
            HTTPStatus.OK,
            {"datasets": compute_sft_rollup(entries, estimates, rescues)},
        )

    def _handle_sft_manifest(self, qs: dict) -> None:
        sft = self._sft_app()
        if sft is None:
            self._send_json(
                HTTPStatus.OK,
                {
                    "items": [],
                    "offset": 0,
                    "limit": 0,
                    "total": 0,
                },
            )
            return
        try:
            limit = max(1, min(200, int(qs.get("limit", ["48"])[0] or 48)))
        except ValueError:
            limit = 48
        try:
            offset = max(0, int(qs.get("offset", ["0"])[0] or 0))
        except ValueError:
            offset = 0
        entries, estimates, rescues = sft.filter_entries(self._parse_sft_query(qs))

        def _first_pos(en: dict) -> float:
            if en.get("kind") == "group":
                positions = en.get("positions_mm") or [0.0]
                return float(positions[0])
            return float(en.get("position_mm", 0.0) or 0.0)

        entries = sorted(entries, key=_first_pos)
        total = len(entries)
        page = entries[offset : offset + limit]
        items = [
            _sft_entry_to_payload(
                e,
                estimates.get(e.get("id", "")),
                rescues.get(e.get("id", "")),
            )
            for e in page
        ]
        self._send_json(
            HTTPStatus.OK,
            {
                "items": items,
                "offset": offset,
                "limit": limit,
                "total": total,
            },
        )

    # -- Synth handlers -------------------------------------------------
    def _parse_synth_query(self, qs: dict) -> dict:
        def _one(key: str) -> str | None:
            v = qs.get(key, [""])[0]
            return v if v else None

        return {
            "plane": _one("plane"),
            "modality": _one("modality"),
            "atlas": _one("atlas"),
            "mode": _one("mode"),
            "apply_damage": (_one("apply_damage") or "all").lower(),
            "damage_intensity": _one("damage_intensity"),
            "q": _one("q"),
        }

    def _handle_synth_manifest(self, qs: dict) -> None:
        synth = self._synth_app()
        if synth is None:
            self._send_json(
                HTTPStatus.OK,
                {
                    "items": [],
                    "offset": 0,
                    "limit": 0,
                    "total": 0,
                    "sort": "position",
                },
            )
            return
        sort_order = (qs.get("sort", ["position"])[0] or "position").lower()
        try:
            limit = max(1, min(200, int(qs.get("limit", ["48"])[0] or 48)))
        except ValueError:
            limit = 48
        try:
            offset = max(0, int(qs.get("offset", ["0"])[0] or 0))
        except ValueError:
            offset = 0
        rows = synth.filter_rows(self._parse_synth_query(qs))
        rows = _sort_synth(rows, sort_order)
        total = len(rows)
        page = rows[offset : offset + limit]
        items = [_synth_row_to_payload(r, synth_dir=synth.synth_dir) for r in page]
        self._send_json(
            HTTPStatus.OK,
            {
                "items": items,
                "offset": offset,
                "limit": limit,
                "total": total,
                "sort": sort_order,
            },
        )

    def _handle_synth_datasets(self, qs: dict) -> None:
        synth = self._synth_app()
        if synth is None:
            self._send_json(HTTPStatus.OK, {"datasets": []})
            return
        rows = synth.filter_rows(self._parse_synth_query(qs))
        self._send_json(
            HTTPStatus.OK,
            {"datasets": compute_synth_rollup(rows, synth_dir=synth.synth_dir)},
        )

    def _handle_synth_facets(self, qs: dict) -> None:
        synth = self._synth_app()
        if synth is None:
            self._send_json(
                HTTPStatus.OK,
                {
                    "modality": [],
                    "atlases": [],
                    "planes": [],
                    "modes": [],
                    "damage_intensities": [],
                    "apply_damage": {"true": 0, "false": 0},
                    "n_total": 0,
                },
            )
            return
        rows = synth.filter_rows(self._parse_synth_query(qs))
        self._send_json(HTTPStatus.OK, compute_synth_facets(rows))

    def _handle_synth_image(self, qs: dict) -> None:
        synth = self._synth_app()
        if synth is None:
            self._send(HTTPStatus.NOT_FOUND, b"not found", content_type="text/plain")
            return
        try:
            seed = int(qs["seed"][0])
        except (KeyError, IndexError, ValueError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "seed required (int)"})
            return
        base = synth.synth_dir.resolve()
        candidate = (base / "images" / f"{seed:08d}.png").resolve()
        if not str(candidate).startswith(str(base)):
            self._send(HTTPStatus.FORBIDDEN, b"forbidden", content_type="text/plain")
            return
        if not candidate.is_file():
            self._send(HTTPStatus.NOT_FOUND, b"not found", content_type="text/plain")
            return
        # cache=False so the browser revalidates each request: synth PNGs
        # have deterministic filenames keyed by seed, but regenerating with
        # the same seed produces different bytes at the same URL, so a
        # cached copy is misleading. Local-only tool; bandwidth cost is
        # negligible.
        self._send(
            HTTPStatus.OK,
            candidate.read_bytes(),
            content_type="image/png",
            cache=False,
        )

    def _handle_synth_thumbnail(self, qs: dict) -> None:
        synth = self._synth_app()
        if synth is None:
            self._send(HTTPStatus.NOT_FOUND, b"not found", content_type="text/plain")
            return
        try:
            seed = int(qs["seed"][0])
        except (KeyError, IndexError, ValueError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "seed required (int)"})
            return
        try:
            w = max(32, min(1024, int(qs.get("w", ["256"])[0] or 256)))
            h = max(32, min(1024, int(qs.get("h", ["256"])[0] or 256)))
        except ValueError:
            w, h = 256, 256
        data, ct = render_thumbnail(f"images/{seed:08d}.png", w, h, base_dir=synth.synth_dir)
        if data is None:
            status = HTTPStatus.NOT_FOUND if ct == "not_found" else HTTPStatus.BAD_REQUEST
            self._send(status, ct.encode("utf-8"), content_type="text/plain")
            return
        # cache=False — same rationale as _handle_synth_image. Regen with
        # the same seed changes the bytes but not the URL.
        self._send(HTTPStatus.OK, data, content_type=ct, cache=False)


class _ExclusiveHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that refuses to share a port with another process.

    Windows TCP allows two processes to bind to the same port by default,
    silently splitting incoming connections. SO_EXCLUSIVEADDRUSE makes the
    second bind attempt fail with EADDRINUSE so our fallback logic can pick
    a different port. No-op on POSIX.
    """

    allow_reuse_address = False

    def server_bind(self) -> None:  # pragma: no cover - exercised at runtime
        if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_EXCLUSIVEADDRUSE,
                1,  # type: ignore[attr-defined]
            )
        super().server_bind()


def bind_with_port_fallback(
    factory,
    host: str,
    requested_port: int,
    *,
    max_attempts: int = 20,
):
    """Try `requested_port`, then port+1, ..., until a free one is found.

    Lets multiple QC sessions launch concurrently without each needing an
    explicit `--port`. Returns (server, actual_port).
    """
    last_err: OSError | None = None
    for offset in range(max_attempts):
        try_port = requested_port + offset
        try:
            return factory(host, try_port), try_port
        except OSError as exc:
            last_err = exc
            continue
    if last_err is None:
        raise RuntimeError("port bind failed without an OS error")
    raise last_err


def prewarm_atlases(rows: list[dict]) -> threading.Thread:
    """Background-load every atlas referenced by the rows in usage order.

    Cold disk-load of a BrainGlobe NIFTI volume can be 30-60s per atlas; with
    ~30 unique atlases in a session that's a half-hour of UI stalls if each
    one loads on first navigation. This thread populates the
    `langslice_harness.atlas.core.load_atlas` lru_cache up front, ordered by
    how many rows reference each atlas (most-used first).
    """
    counts: dict[str, int] = {}
    for r in rows:
        a = r.get("atlas")
        if a:
            counts[a] = counts.get(a, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])

    def _worker() -> None:
        total = len(ordered)
        for idx, (name, n) in enumerate(ordered, 1):
            t0 = time.monotonic()
            try:
                load_atlas(name)
                logger.info(
                    "prewarm %d/%d %s (%d entries) in %.1fs",
                    idx,
                    total,
                    name,
                    n,
                    time.monotonic() - t0,
                )
            except Exception as exc:
                logger.warning("prewarm %d/%d %s failed: %s", idx, total, name, exc)
        logger.info("atlas prewarm complete: %d atlases", total)

    t = threading.Thread(target=_worker, name="atlas-prewarm", daemon=True)
    t.start()
    return t


def make_app_server(
    *,
    inventory_app: InventoryApp,
    synth_app: SynthApp,
    sft_app: SftApp,
    host: str,
    port: int,
    training_progress: TrainingProgress | None = None,
) -> ThreadingHTTPServer:
    """Build a unified ThreadingHTTPServer that hosts all three modules."""
    handler_cls = type("BoundAppHandler", (Handler,), {})
    server = _ExclusiveHTTPServer((host, port), handler_cls)
    server.mode = "app"
    if training_progress is not None and inventory_app.training_progress is None:
        inventory_app.training_progress = training_progress
    server.inventory_app = inventory_app
    server.synth_app = synth_app
    server.sft_app = sft_app
    return server


# ----------------------------------------------------------------------
# Test helper


class _StubResponse:
    def __init__(self, status_code: int, headers: dict, body: bytes) -> None:
        self.status_code = status_code
        self.headers = headers
        self.body = body


class _StubServer:
    """Minimal stand-in for ThreadingHTTPServer used by make_test_handler."""

    def __init__(
        self,
        *,
        mode: str = "app",
        inventory_app: InventoryApp | None = None,
        synth_app: SynthApp | None = None,
        sft_app: SftApp | None = None,
    ) -> None:
        self.mode = mode
        self.inventory_app = inventory_app
        self.synth_app = synth_app
        self.sft_app = sft_app


class _StubHandler(Handler):
    """A Handler that does no socket I/O; instead captures responses in-memory."""

    # We bypass BaseHTTPRequestHandler.__init__ entirely (it expects a real
    # socket).  The test driver populates fields manually before each call.
    def __init__(self, server: _StubServer) -> None:  # noqa: D401 - intentional
        self.server = server
        self._status: int | None = None
        self._headers: dict[str, str] = {}
        self._body = bytearray()
        self._wrote_body = False
        self.command = "GET"
        self.path = "/"
        self.headers = _StubHeaders()
        self.rfile = io.BytesIO()
        self.wfile = _CaptureWriter(self._body)

    # ---- override BaseHTTPRequestHandler write methods ----
    def send_response(self, code: int, message: str | None = None) -> None:  # noqa: D401
        self._status = int(code)

    def send_header(self, key: str, value: str) -> None:
        self._headers[key] = value

    def end_headers(self) -> None:  # nothing to flush
        return

    def log_message(self, fmt: str, *args) -> None:
        return

    # ---- entry points used by tests ----
    def get(self, path: str) -> _StubResponse:
        self.command = "GET"
        self.path = path
        self._status = None
        self._headers = {}
        self._body = bytearray()
        self.wfile = _CaptureWriter(self._body)
        self.do_GET()
        assert self._status is not None, "do_GET did not set a status"
        return _StubResponse(self._status, dict(self._headers), bytes(self._body))

    def post(self, path: str, body: bytes | dict | None = None) -> _StubResponse:
        if isinstance(body, dict):
            raw = json.dumps(body).encode("utf-8")
        elif isinstance(body, (bytes, bytearray)):
            raw = bytes(body)
        elif body is None:
            raw = b""
        else:
            raise TypeError(f"unsupported body type: {type(body)!r}")
        self.command = "POST"
        self.path = path
        self._status = None
        self._headers = {}
        self._body = bytearray()
        self.wfile = _CaptureWriter(self._body)
        self.headers = _StubHeaders({"Content-Length": str(len(raw))})
        self.rfile = io.BytesIO(raw)
        self.do_POST()
        assert self._status is not None, "do_POST did not set a status"
        return _StubResponse(self._status, dict(self._headers), bytes(self._body))


class _StubHeaders:
    def __init__(self, mapping: dict[str, str] | None = None) -> None:
        self._mapping = {k.lower(): v for k, v in (mapping or {}).items()}

    def get(self, key: str, default: str = "") -> str:
        return self._mapping.get(key.lower(), default)


class _CaptureWriter:
    def __init__(self, sink: bytearray) -> None:
        self._sink = sink

    def write(self, data: bytes) -> int:
        self._sink.extend(data)
        return len(data)

    def flush(self) -> None:
        return


def make_test_handler(
    *,
    mode: str = "app",
    inventory_app: InventoryApp | None = None,
    synth_app: SynthApp | None = None,
    sft_app: SftApp | None = None,
) -> _StubHandler:
    """Test-only helper: returns a handler-like object with `.get(path)` and
    `.post(path, body)` that simulates dispatch without binding a socket.
    """
    server = _StubServer(
        mode=mode,
        inventory_app=inventory_app,
        synth_app=synth_app,
        sft_app=sft_app,
    )
    return _StubHandler(server)


# ----------------------------------------------------------------------
# CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory-manifest",
        type=Path,
        default=None,
        help=(
            "inventory source. Either a sharded directory "
            f"({DEFAULT_INVENTORY_SHARDS.relative_to(REPO_ROOT)}/<plane>/<dataset>.jsonl) "
            f"or a legacy single-file manifest "
            f"({DEFAULT_INVENTORY_MANIFEST.relative_to(REPO_ROOT)}). "
            "Default: shards dir if it exists, else the legacy file."
        ),
    )
    parser.add_argument(
        "--synth-dir",
        type=Path,
        default=DEFAULT_SYNTH_DIR,
        help=(
            "directory containing synth manifest.jsonl + images/ (default: "
            f"{DEFAULT_SYNTH_DIR.relative_to(REPO_ROOT)})"
        ),
    )
    parser.add_argument(
        "--sft-dir",
        type=Path,
        default=DEFAULT_SFT_DIR,
        help=(
            "directory containing *_trace_manifest_qc.jsonl + runs*/ results "
            f"(default: {DEFAULT_SFT_DIR.relative_to(REPO_ROOT)})"
        ),
    )
    parser.add_argument(
        "--rlvr-progress-dir",
        type=Path,
        default=DEFAULT_RLVR_PROGRESS_DIR,
        help=(
            "directory the RLVR trainer writes consumed.jsonl/queue.json/run.json "
            f"into; the Training tab live-polls this. default: "
            f"{DEFAULT_RLVR_PROGRESS_DIR.relative_to(REPO_ROOT)}"
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="don't auto-open the browser on startup",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    inventory_source = args.inventory_manifest
    if inventory_source is None:
        if DEFAULT_INVENTORY_SHARDS.is_dir():
            inventory_source = DEFAULT_INVENTORY_SHARDS
        else:
            inventory_source = DEFAULT_INVENTORY_MANIFEST
    if not inventory_source.exists():
        logger.error("inventory source not found: %s", inventory_source)
        return 2

    # The training-progress dir need not exist yet; the reader is defensive
    # about missing files. Trainer will create the dir + files on first write.
    progress = TrainingProgress(args.rlvr_progress_dir)
    inventory_app = InventoryApp(
        manifest_path=inventory_source,
        training_progress=progress,
    )
    synth_app = SynthApp(synth_dir=args.synth_dir)
    sft_app = SftApp(sft_dir=args.sft_dir)

    server, actual_port = bind_with_port_fallback(
        lambda h, p: make_app_server(
            inventory_app=inventory_app,
            synth_app=synth_app,
            sft_app=sft_app,
            host=h,
            port=p,
            training_progress=progress,
        ),
        args.host,
        args.port,
    )
    if actual_port != args.port:
        logger.info("port %d busy; using %d (override with --port)", args.port, actual_port)
    url = f"http://{args.host}:{actual_port}/"
    logger.info(
        "QC app serving inventory=%d rows, synth=%d rows, sft=%d entries at %s",
        len(inventory_app.rows),
        len(synth_app.rows),
        len(sft_app.entries),
        url,
    )
    prewarm_atlases(inventory_app.rows)
    if not args.no_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url, new=2)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

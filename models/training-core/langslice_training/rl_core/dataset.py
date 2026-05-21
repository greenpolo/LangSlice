"""HF ``Dataset`` assembly for the LangSlice RLVR pipeline.

Each row carries everything the env's ``reset(**kwargs)`` and the model's
chat-template need: a chat-format ``prompt``, the tissue ``image``/``images``,
and the metadata that the env consumes (``atlas_name``, ``plane``,
``valid_range_mm``, ``ground_truth_positions_mm``, ``kind``, ``subject_id``).

System prompts come verbatim from
``langslice_harness.harness.estimation.prompts`` so any prompt change in
production is automatically picked up here.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image

DEFAULT_CLAHE_FRACTION = 0.25

Plane = Literal["coronal", "sagittal", "horizontal"]


def canonicalize_atlas_name(atlas_name: str) -> str:
    try:
        from langslice_harness.atlas.core import canonicalize_atlas_name as _canonicalize
    except ModuleNotFoundError:
        return atlas_name

    return _canonicalize(atlas_name)


def get_position_range_mm(atlas, *, plane: Plane):  # noqa: ANN001
    from langslice_harness.atlas.core import get_position_range_mm as _range_mm

    return _range_mm(atlas, plane=plane)


def load_atlas(atlas_name: str):
    from langslice_harness.atlas.core import load_atlas as _load_atlas

    return _load_atlas(atlas_name)


def species_from_atlas_name(atlas_name: str) -> str:
    from langslice_harness.atlas.core import species_from_atlas_name as _species

    return _species(atlas_name)


def get_in_plane_long_edge(atlas, *, plane: Plane) -> int:  # noqa: ANN001
    from langslice_harness.atlas.core import get_in_plane_long_edge as _long_edge

    return _long_edge(atlas, plane=plane)


def _normalize_image(image: Image.Image) -> Image.Image:
    from langslice_harness.image_prep import normalize_image as _normalize

    return _normalize(image)


def _prepare_image_for_vlm(image: Image.Image, *, max_long_edge: int) -> Image.Image:
    from langslice_harness.image_prep import prepare_image_for_vlm as _prepare

    return _prepare(image, max_long_edge=max_long_edge).image


def _adaptive_preprocess(image: Image.Image) -> Image.Image:
    from langslice_harness.image_prep import adaptive_preprocess as _adapt

    return _adapt(image)


def build_group_prompt(**kwargs: Any) -> str:
    from langslice_harness.harness.estimation.prompts import build_group_prompt as _build

    return _build(**kwargs)


def build_single_slice_prompt(**kwargs: Any) -> str:
    from langslice_harness.harness.estimation.prompts import build_single_slice_prompt as _build

    return _build(**kwargs)


@dataclass(frozen=True)
class SingleSliceExample:
    image_path: Path
    atlas_name: str
    plane: Plane
    ap_mm: float
    subject_id: str
    section_id: str


@dataclass(frozen=True)
class GroupExample:
    image_paths: tuple[Path, ...]
    atlas_name: str
    plane: Plane
    ap_mm: tuple[float, ...]
    subject_id: str
    interval_mm: float
    thickness_um: int
    section_ids: tuple[str, ...]


class RowDataset:
    """Tiny map-style dataset that lazy-decodes images on access.

    Hugging Face ``Dataset.from_list`` serializes PIL images into Arrow. That is
    useful for portable datasets, but it is painfully slow during RLVR startup
    and gives us no benefit because these rows are ephemeral rollout inputs.
    TRL's GRPO dataloader uses an identity collator, so a simple ``__len__`` /
    ``__getitem__`` dataset is enough.

    Each row stores image paths + per-image CLAHE flags + atlas long edge
    under underscore-prefixed keys. ``__getitem__`` runs the SFT-matched
    preprocessing pipeline on demand and returns a dict with the public
    keys plus a decoded ``image`` (single) or ``images`` (group) PIL value.
    """

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key.startswith("_"):
                    continue
                if key not in keys:
                    keys.append(key)
        # Trainer code expects an "image" or "images" column; expose both
        # so neither single-only nor mixed runs trip an attribute check.
        for k in ("image", "images"):
            if k not in keys:
                keys.append(k)
        self.column_names = keys

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._rows[index]
        out: dict[str, Any] = {k: v for k, v in row.items() if not k.startswith("_")}
        paths: tuple[str, ...] = row["_image_paths"]
        flags: tuple[bool, ...] = row["_apply_clahes"]
        long_edge: int = row["_atlas_long_edge"]
        decoded = [
            preprocess_query_image(p, atlas_long_edge=long_edge, apply_clahe=c)
            for p, c in zip(paths, flags, strict=True)
        ]
        if row["kind"] == "single":
            out["image"] = decoded[0]
        else:
            out["images"] = decoded
        return out


def _load_test_images_subject(
    subject_dir: Path, *, default_plane: Plane = "coronal"
) -> list[SingleSliceExample]:
    gt_path = subject_dir / "ground_truth.json"
    if not gt_path.exists():
        return []
    raw = json.loads(gt_path.read_text())
    out: list[SingleSliceExample] = []
    for fname, meta in raw.items():
        # Field name is historical (``ap_mm``); the value is the
        # slice-normal-axis position in mm regardless of plane.
        if "ap_mm" not in meta or "atlas" not in meta:
            continue
        image_path = subject_dir / fname
        if not image_path.exists():
            continue
        plane: Plane = meta.get("plane", default_plane)
        out.append(
            SingleSliceExample(
                image_path=image_path,
                atlas_name=str(meta["atlas"]),
                plane=plane,
                ap_mm=float(meta["ap_mm"]),
                subject_id=subject_dir.name,
                section_id=f"{subject_dir.name}/{fname}",
            )
        )
    return out


def load_test_images(
    test_images_root: Path, *, subject_glob: str = "M*"
) -> list[SingleSliceExample]:
    """Walk ``references/TestImages/M*`` and load all single-slice ground-truths."""
    out: list[SingleSliceExample] = []
    for subject_dir in sorted(test_images_root.glob(subject_glob)):
        if not subject_dir.is_dir():
            continue
        out.extend(_load_test_images_subject(subject_dir))
    return out


_DEFAULT_RLVR_PLANES: tuple[Plane, ...] = ("coronal", "sagittal", "horizontal")


def _resolve_active_allocation(plane: str, manifest_root: Path) -> set[str]:
    """Return active section_ids for (plane, ``rlvr``), honoring tombstones."""
    path = manifest_root / "allocations" / plane / "rlvr.jsonl"
    if not path.is_file():
        return set()
    state: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                continue
            entry = json.loads(stripped)
            sid = entry["section_id"]
            if entry.get("action") == "remove":
                state.pop(sid, None)
            else:
                state[sid] = entry
    return set(state)


def _index_shard_by_section(shard_path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with shard_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            sid = row.get("section_id")
            if sid:
                out[sid] = row
    return out


def load_rlvr_allocation(
    manifest_root: Path,
    *,
    repo_root: Path,
    planes: Iterable[Plane] = _DEFAULT_RLVR_PLANES,
) -> list[SingleSliceExample]:
    """Load every section in ``allocations/<plane>/rlvr.jsonl`` as a SingleSliceExample.

    Joins:
      - ``manifest_root/allocations/<plane>/rlvr.jsonl`` (active allocations,
        tombstones honored)
      - ``manifest_root/shards/<plane>/<dataset>.jsonl`` (ground-truth rows)

    Image paths in shards are repo-relative; we resolve them to absolute paths
    using ``repo_root`` so the rest of the pipeline can hydrate them directly.
    Allocation rows whose section_id is missing from its dataset's shard, or
    whose shard file is missing, are dropped (with stderr warning) so manifest
    drift never hard-fails a training run.
    """
    import sys

    out: list[SingleSliceExample] = []
    for plane in planes:
        active = _resolve_active_allocation(plane, manifest_root)
        if not active:
            continue
        # Group allocations by dataset so we open each shard at most once
        alloc_path = manifest_root / "allocations" / plane / "rlvr.jsonl"
        by_dataset: dict[str, set[str]] = {}
        with alloc_path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                stripped = raw.strip()
                if not stripped:
                    continue
                entry = json.loads(stripped)
                sid = entry["section_id"]
                if sid not in active:
                    continue
                ds = entry.get("dataset")
                if not ds:
                    continue
                by_dataset.setdefault(ds, set()).add(sid)

        for dataset, section_ids in by_dataset.items():
            shard_path = manifest_root / "shards" / plane / f"{dataset}.jsonl"
            if not shard_path.is_file():
                print(
                    f"WARN: rlvr allocation references missing shard {shard_path}; "
                    f"dropping {len(section_ids)} sections",
                    file=sys.stderr,
                )
                continue
            shard_index = _index_shard_by_section(shard_path)
            missing = section_ids - shard_index.keys()
            if missing:
                print(
                    f"WARN: {len(missing)} rlvr allocation rows have no matching shard "
                    f"row in {shard_path}; dropping",
                    file=sys.stderr,
                )
            for sid in section_ids - missing:
                row = shard_index[sid]
                image_path = (repo_root / row["image_path"]).resolve()
                if not image_path.is_file():
                    print(
                        f"WARN: rlvr allocation row {sid} image not on disk: {image_path}",
                        file=sys.stderr,
                    )
                    continue
                out.append(
                    SingleSliceExample(
                        image_path=image_path,
                        atlas_name=str(row["atlas"]),
                        plane=plane,
                        ap_mm=float(row["position_mm"]),
                        subject_id=str(row["subject_id"]),
                        section_id=sid,
                    )
                )
    print(
        f"[load_rlvr_allocation] loaded {len(out)} examples across "
        f"{len({(e.atlas_name, e.plane) for e in out})} (atlas, plane) pairs, "
        f"{len({e.subject_id for e in out})} subjects",
        file=sys.stderr,
        flush=True,
    )
    return out


def _interval_mm(positions: Sequence[float]) -> float:
    """Median spacing between adjacent positions, fallback 0.0 if <2 slices."""
    if len(positions) < 2:
        return 0.0
    diffs = sorted(abs(positions[i + 1] - positions[i]) for i in range(len(positions) - 1))
    mid = len(diffs) // 2
    return float(diffs[mid] if len(diffs) % 2 else 0.5 * (diffs[mid - 1] + diffs[mid]))


def make_group_examples(
    examples: list[SingleSliceExample],
    *,
    group_size: int,
    rng: random.Random,
    thickness_um: int = 30,
) -> list[GroupExample]:
    """Bin single-slice examples by subject + atlas + plane and slice into groups.

    Slices within a subject are sorted by position; we then take consecutive
    windows of ``group_size`` so the group's ordering and interval are real
    rather than synthesized.
    """
    if group_size < 2:
        raise ValueError(f"group_size must be >= 2, got {group_size}")

    by_subject: dict[tuple[str, str, Plane], list[SingleSliceExample]] = {}
    for ex in examples:
        key = (ex.subject_id, canonicalize_atlas_name(ex.atlas_name), ex.plane)
        by_subject.setdefault(key, []).append(ex)

    out: list[GroupExample] = []
    for (subject_id, atlas_name, plane), bucket in by_subject.items():
        if len(bucket) < group_size:
            continue
        ordered = sorted(bucket, key=lambda e: e.ap_mm)
        # Sliding window — gives more groups per subject than chunked windows
        # and naturally produces overlapping training examples.
        for start in range(0, len(ordered) - group_size + 1):
            window = ordered[start : start + group_size]
            positions = tuple(e.ap_mm for e in window)
            out.append(
                GroupExample(
                    image_paths=tuple(e.image_path for e in window),
                    atlas_name=atlas_name,
                    plane=plane,
                    ap_mm=positions,
                    subject_id=subject_id,
                    interval_mm=_interval_mm(positions),
                    thickness_um=thickness_um,
                    section_ids=tuple(e.section_id for e in window),
                )
            )
    rng.shuffle(out)
    return out


_SYSTEM_PROMPT_SINGLE_CACHE: dict[tuple[str, Plane], tuple[str, float, float]] = {}


def _system_prompt_single(atlas_name: str, plane: Plane) -> tuple[str, float, float]:
    """Cached per (atlas_name, plane). Loading a BrainGlobe atlas is multi-second
    and there are at most ~20 unique pairs even in the 14k allocation, so this
    cache turns a quadratic dataset-assembly cost into a near-linear one.
    """
    key = (atlas_name, plane)
    cached = _SYSTEM_PROMPT_SINGLE_CACHE.get(key)
    if cached is not None:
        return cached
    atlas = load_atlas(atlas_name)
    pos_lo, pos_hi = get_position_range_mm(atlas, plane=plane)
    species = species_from_atlas_name(atlas_name)
    prompt = build_single_slice_prompt(
        atlas_name=atlas_name,
        plane=plane,
        pos_lo=pos_lo,
        pos_hi=pos_hi,
        species=species,
    )
    out = (prompt, pos_lo, pos_hi)
    _SYSTEM_PROMPT_SINGLE_CACHE[key] = out
    return out


def _system_prompt_group(
    atlas_name: str, plane: Plane, n_slices: int, interval_mm: float, thickness_um: int
) -> tuple[str, float, float]:
    atlas = load_atlas(atlas_name)
    pos_lo, pos_hi = get_position_range_mm(atlas, plane=plane)
    species = species_from_atlas_name(atlas_name)
    prompt = build_group_prompt(
        atlas_name=atlas_name,
        plane=plane,
        pos_lo=pos_lo,
        pos_hi=pos_hi,
        species=species,
        n_slices=n_slices,
        interval_mm=interval_mm,
        thickness_um=thickness_um,
    )
    return prompt, pos_lo, pos_hi


_ATLAS_LONG_EDGE_CACHE: dict[tuple[str, Plane], int] = {}


def _atlas_in_plane_long_edge(atlas_name: str, plane: Plane) -> int:
    """Cached lookup of the atlas in-plane long edge in pixels.

    Mirrors ``_atlas_long_edge_for`` in build_sft_corpus.py so the RLVR
    pipeline downsamples query images to the same target resolution that
    the SFT corpus + production inference both use.
    """
    key = (atlas_name, plane)
    cached = _ATLAS_LONG_EDGE_CACHE.get(key)
    if cached is not None:
        return cached
    atlas = load_atlas(atlas_name)
    long_edge = int(get_in_plane_long_edge(atlas, plane=plane))
    _ATLAS_LONG_EDGE_CACHE[key] = long_edge
    return long_edge


def _should_apply_clahe(section_id: str, clahe_fraction: float) -> bool:
    """Deterministic per-section CLAHE assignment matching SFT corpus exactly.

    Same hash-and-threshold rule as ``build_sft_corpus._stage_query_image``:
    sections shared between SFT and RLVR get the SAME CLAHE state, so the
    visual content the model sees during RL is bit-identical to what it
    saw at SFT time.
    """
    if clahe_fraction <= 0.0:
        return False
    if clahe_fraction >= 1.0:
        return True
    digest = hashlib.sha1(section_id.encode("utf-8")).hexdigest()
    rand01 = (int(digest, 16) % 10_000_000) / 10_000_000
    return rand01 < clahe_fraction


def preprocess_query_image(
    image_path: str | Path,
    *,
    atlas_long_edge: int,
    apply_clahe: bool,
) -> Image.Image:
    """SFT-matched preprocessing: normalize → atlas-aware downsample → CLAHE?"""
    with Image.open(str(image_path)) as raw:
        loaded = raw.copy()
    normalized = _normalize_image(loaded)
    prepped = _prepare_image_for_vlm(normalized, max_long_edge=atlas_long_edge)
    if apply_clahe:
        prepped = _adaptive_preprocess(prepped)
    return prepped.convert("RGB")


def _row_from_single(ex: SingleSliceExample, *, clahe_fraction: float) -> dict[str, Any]:
    system, pos_lo, pos_hi = _system_prompt_single(ex.atlas_name, ex.plane)
    user_text = "Estimate the position (mm) of this slice."
    return {
        # Chat messages — image goes IN the user content list before its text,
        # matching Gemma 4's image-before-text rule.
        "prompt": [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            },
        ],
        # Lazy-decode plumbing — RowDataset.__getitem__ reads these underscore
        # keys, runs preprocess_query_image, and exposes "image" to the trainer.
        "_image_paths": (str(ex.image_path),),
        "_apply_clahes": (_should_apply_clahe(ex.section_id, clahe_fraction),),
        "_atlas_long_edge": _atlas_in_plane_long_edge(ex.atlas_name, ex.plane),
        "atlas_name": canonicalize_atlas_name(ex.atlas_name),
        "plane": ex.plane,
        "valid_range_mm": (pos_lo, pos_hi),
        "ground_truth_positions_mm": (ex.ap_mm,),
        "kind": "single",
        "subject_id": ex.subject_id,
        # ``section_id`` lets the curriculum sampler / per-rollout logger
        # identify each row. ``env.reset(**row)`` accepts and ignores it.
        "section_id": ex.section_id,
    }


def _row_from_group(ex: GroupExample, *, clahe_fraction: float) -> dict[str, Any]:
    n = len(ex.image_paths)
    system, pos_lo, pos_hi = _system_prompt_group(
        ex.atlas_name,
        ex.plane,
        n_slices=n,
        interval_mm=ex.interval_mm,
        thickness_um=ex.thickness_um,
    )
    user_blocks: list[dict[str, Any]] = []
    for i in range(1, n + 1):
        user_blocks.append({"type": "image"})
        user_blocks.append({"type": "text", "text": f"Slice {i} of {n}."})
    user_blocks.append(
        {"type": "text", "text": f"Estimate the position (mm) of all {n} slices, in order."}
    )
    return {
        "prompt": [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": user_blocks},
        ],
        "_image_paths": tuple(str(p) for p in ex.image_paths),
        "_apply_clahes": tuple(_should_apply_clahe(sid, clahe_fraction) for sid in ex.section_ids),
        "_atlas_long_edge": _atlas_in_plane_long_edge(ex.atlas_name, ex.plane),
        "atlas_name": canonicalize_atlas_name(ex.atlas_name),
        "plane": ex.plane,
        "valid_range_mm": (pos_lo, pos_hi),
        "ground_truth_positions_mm": ex.ap_mm,
        "kind": "group",
        "subject_id": ex.subject_id,
        # Group rollouts have multiple sections — synthesize a stable
        # composite id so curriculum-style logs can still bucket them.
        # The first section_id alone would be ambiguous across overlapping
        # windows, so include the position range too.
        "section_id": (
            f"group:{ex.section_ids[0]}..{ex.section_ids[-1]}"
            if ex.section_ids
            else f"group:{ex.subject_id}"
        ),
    }


def build_rlvr_rows(
    *,
    single_examples: Iterable[SingleSliceExample],
    group_examples: Iterable[GroupExample],
    single_fraction: float = 0.7,
    seed: int = 0,
    clahe_fraction: float = DEFAULT_CLAHE_FRACTION,
) -> list[dict[str, Any]]:
    """Mix single-slice and group rows at the requested ratio.

    ``single_fraction`` accepts the closed interval [0.0, 1.0]:
        - ``1.0`` → singles only (Phase A);
        - ``0.0`` → groups only;
        - any value in (0, 1) → mixed, capped by whichever bucket runs out
          first so no row is silently oversampled into duplicates.
    """
    if not 0.0 <= single_fraction <= 1.0:
        raise ValueError(f"single_fraction must be in [0, 1], got {single_fraction}")

    rng = random.Random(seed)
    singles = list(single_examples)
    groups = list(group_examples)
    rng.shuffle(singles)
    rng.shuffle(groups)

    if not singles and not groups:
        return []

    if single_fraction == 1.0:
        n_single, n_group = len(singles), 0
    elif single_fraction == 0.0:
        n_single, n_group = 0, len(groups)
    elif singles and groups:
        # Cap each bucket so the overall mix lands at single_fraction without
        # repeats. Whichever side is the binding constraint determines total.
        max_total_by_singles = int(len(singles) / single_fraction)
        max_total_by_groups = int(len(groups) / (1.0 - single_fraction))
        total = min(max_total_by_singles, max_total_by_groups)
        n_single = int(round(total * single_fraction))
        n_group = total - n_single
    elif singles:
        n_single = len(singles)
        n_group = 0
    else:
        n_single = 0
        n_group = len(groups)

    import sys
    import time

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for i, s in enumerate(singles[:n_single]):
        rows.append(_row_from_single(s, clahe_fraction=clahe_fraction))
        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-9)
            eta = (n_single - (i + 1)) / max(rate, 1e-9)
            cache_n = len(_SYSTEM_PROMPT_SINGLE_CACHE)
            print(
                f"[build_rlvr_rows] singles {i + 1}/{n_single} "
                f"({rate:.1f}/s, eta {eta:.0f}s, atlas_cache={cache_n})",
                file=sys.stderr,
                flush=True,
            )
    for i, g in enumerate(groups[:n_group]):
        rows.append(_row_from_group(g, clahe_fraction=clahe_fraction))
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(
                f"[build_rlvr_rows] groups {i + 1}/{n_group} (elapsed {elapsed:.0f}s)",
                file=sys.stderr,
                flush=True,
            )
    print(
        f"[build_rlvr_rows] done: {len(rows)} rows in {time.time() - t0:.1f}s",
        file=sys.stderr,
        flush=True,
    )
    rng.shuffle(rows)
    return rows


def split_subjects_for_holdout(
    examples: Iterable[SingleSliceExample],
    *,
    eval_holdout_every: int = 5,
) -> tuple[set[str], set[str]]:
    """Deterministic subject-level train/eval split.

    Subject ids are sorted and every ``eval_holdout_every``-th one (1-indexed,
    so the FIRST eligible position is the Nth-1 element) is assigned to eval.
    A non-positive ``eval_holdout_every`` disables the holdout (everything to
    train, eval set empty).

    Returns ``(train_subject_ids, eval_subject_ids)``. The two sets are
    guaranteed to be disjoint — every subject appears in exactly one.
    """
    sorted_ids = sorted({ex.subject_id for ex in examples})
    if eval_holdout_every <= 0:
        return set(sorted_ids), set()
    eval_ids = {sid for i, sid in enumerate(sorted_ids) if (i + 1) % eval_holdout_every == 0}
    train_ids = set(sorted_ids) - eval_ids
    return train_ids, eval_ids


def to_hf_dataset(rows: list[dict[str, Any]]) -> Any:
    """Return a lightweight map-style dataset for TRL's GRPO dataloader."""
    return RowDataset(rows)

"""Path rewrite helpers for unified SFT corpora."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import shutil
from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_QUERY_PREFIX = "queries/"
_ATLAS_PREFIX = "atlas/"
_DATA_DATASETS_SEGMENT = "data/datasets/"
_ATLAS_SNAP_TOLERANCE_MM = 0.10
_ATLAS_POS_PATH_RE = re.compile(
    r"(?:^|/)atlas/(?P<atlas>[^/]+)/(?P<plane>[^/]+)/(?P<pos>-?\d+\.\d+)mm\.jpg$"
)


def _resolve_existing(
    rel: str,
    parents: Iterable[Path],
    *,
    snap_atlas_to_grid: bool = False,
) -> Path | None:
    parents_list = list(parents)
    candidates: list[Path] = []
    rel_norm = rel.replace("\\", "/")
    if os.path.isabs(rel_norm):
        candidates.append(Path(rel_norm))
    for parent in parents_list:
        candidates.append(Path(parent) / rel_norm)
    for c in candidates:
        try:
            if c.is_file():
                return c.resolve()
        except OSError:
            continue
    if snap_atlas_to_grid:
        return _snap_to_nearest_atlas(rel_norm, parents_list)
    return None


_JPEG_SOI = b"\xff\xd8\xff"
_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _is_valid_image_header(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            head = fh.read(8)
    except OSError:
        return False
    if len(head) < 4:
        return False
    return head[:3] == _JPEG_SOI or head[:8] == _PNG_SIG


@lru_cache(maxsize=256)
def _list_atlas_positions(plane_dir: Path) -> tuple[tuple[float, Path], ...]:
    if not plane_dir.is_dir():
        return ()
    out: list[tuple[float, Path]] = []
    for p in plane_dir.glob("*.jpg"):
        try:
            pos = float(p.stem.replace("mm", ""))
        except ValueError:
            continue
        out.append((pos, p))
    out.sort(key=lambda kv: kv[0])
    return tuple(out)


def _snap_to_nearest_atlas(rel_norm: str, parents: list[Path]) -> Path | None:
    m = _ATLAS_POS_PATH_RE.search(rel_norm)
    if m is None:
        return None
    target_pos = float(m.group("pos"))
    rel_plane_dir = Path(rel_norm).parent
    for parent in parents:
        try:
            plane_dir = (Path(parent) / rel_plane_dir).resolve()
        except OSError:
            continue
        positions = _list_atlas_positions(plane_dir)
        if not positions:
            continue
        nearest_pos, nearest_path = min(positions, key=lambda kv: abs(kv[0] - target_pos))
        if abs(nearest_pos - target_pos) <= _ATLAS_SNAP_TOLERANCE_MM:
            return nearest_path
    return None


def _safe_link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try:
            if dst.resolve() == src.resolve():
                return "exists"
        except OSError:
            pass
        return "skip"
    try:
        os.symlink(src, dst)
        return "symlink"
    except (OSError, NotImplementedError):
        try:
            shutil.copy2(src, dst)
            return "copy"
        except OSError as exc:
            logger.warning("Failed to stage %s -> %s (%s)", src, dst, exc)
            return "skip"


def _find_repo_root_for_rewriter(start: Path) -> Path | None:
    start = Path(start).resolve()
    for cand in (start, *start.parents):
        if (cand / "pyproject.toml").is_file():
            return cand
    return None


def _bucket_for(path_str: str) -> str:
    norm = path_str.replace("\\", "/")
    if "/queries/" in norm or norm.startswith("queries/"):
        return _QUERY_PREFIX
    if "/atlas/" in norm or norm.startswith("atlas/"):
        return _ATLAS_PREFIX
    return _ATLAS_PREFIX


def _rewritten_subpath(rel: str, *, bucket: str) -> str:
    rel_norm = rel.replace("\\", "/")
    if rel_norm.startswith(bucket):
        rel_norm = rel_norm[len(bucket) :]
    if "/" not in rel_norm:
        return bucket + rel_norm
    digest = hashlib.sha1(rel_norm.encode("utf-8")).hexdigest()[:8]
    name = Path(rel_norm).name
    return f"{bucket}{digest}_{name}"


def build_unified_corpus(
    *,
    base_corpus: Path,
    iterative_corpus_dir: Path,
    iterative_jsonls: list[Path] | None = None,
    output_dir: Path,
    output_jsonl: Path,
    extra_image_roots: Iterable[Path] = (),
    base_sample_n: int | None = None,
    base_sample_seed: int = 0,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_jsonl = Path(output_jsonl)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if iterative_jsonls is None:
        iterative_jsonls = sorted(Path(iterative_corpus_dir).glob("round_*.jsonl"))

    sources: list[tuple[Path, Path, list[str] | None]] = []
    if base_sample_n == 0:
        logger.info("base_sample_n=0: excluding base corpus")
    elif base_corpus.is_file():
        if base_sample_n is not None and base_sample_n > 0:
            all_lines = base_corpus.read_text(encoding="utf-8").splitlines()
            nonempty = [line for line in all_lines if line.strip()]
            rng = random.Random(base_sample_seed)
            n = min(base_sample_n, len(nonempty))
            sampled = rng.sample(nonempty, n)
            logger.info(
                "Sampled %d / %d rows from base corpus (seed=%d)",
                n,
                len(nonempty),
                base_sample_seed,
            )
            sources.append((base_corpus, base_corpus.parent, sampled))
        else:
            sources.append((base_corpus, base_corpus.parent, None))
    for it in iterative_jsonls:
        if Path(it).is_file():
            sources.append((Path(it), Path(it).parent, None))

    stats = {
        "rows_input": 0,
        "rows_kept": 0,
        "rows_dropped_missing_images": 0,
        "images_symlinked": 0,
        "images_copied": 0,
        "images_skipped": 0,
    }
    stage_cache: dict[Path, str] = {}
    repo_root = _find_repo_root_for_rewriter(output_dir)

    def _stage(src: Path, bucket: str, rel: str) -> str | None:
        try:
            src_resolved = src.resolve()
        except OSError:
            return None
        cached = stage_cache.get(src_resolved)
        if cached is not None:
            return cached
        if bucket == _QUERY_PREFIX and repo_root is not None:
            try:
                rel_to_repo = src_resolved.relative_to(repo_root).as_posix()
            except ValueError:
                rel_to_repo = None
            if rel_to_repo is not None and rel_to_repo.startswith(_DATA_DATASETS_SEGMENT):
                stage_cache[src_resolved] = rel_to_repo
                return rel_to_repo
        target_rel = _rewritten_subpath(rel, bucket=bucket)
        target_abs = output_dir / target_rel
        strategy = _safe_link_or_copy(src_resolved, target_abs)
        if strategy == "symlink":
            stats["images_symlinked"] += 1
        elif strategy == "copy":
            stats["images_copied"] += 1
        elif strategy == "skip":
            stats["images_skipped"] += 1
        if not target_abs.exists() and not target_abs.is_symlink():
            return None
        stage_cache[src_resolved] = target_rel
        return target_rel

    with output_jsonl.open("w", encoding="utf-8") as out_fh:
        for src_jsonl, src_root, preloaded in sources:
            parents = [src_root, *extra_image_roots]
            if preloaded is not None:
                line_source: Iterator[str] = iter(preloaded)
                fh_ctx = nullcontext(line_source)
            else:
                fh_ctx = src_jsonl.open("r", encoding="utf-8")
            with fh_ctx as in_fh:
                for raw in in_fh:
                    stripped = raw.strip()
                    if not stripped:
                        continue
                    stats["rows_input"] += 1
                    try:
                        row = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        logger.warning("Skipping malformed row in %s: %s", src_jsonl, exc)
                        stats["rows_dropped_missing_images"] += 1
                        continue

                    new_qips: list[str] = []
                    row_ok = True
                    for q in row.get("query_image_paths", []) or []:
                        resolved = _resolve_existing(q, parents)
                        if resolved is None or not _is_valid_image_header(resolved):
                            row_ok = False
                            break
                        target_rel = _stage(resolved, _QUERY_PREFIX, q)
                        if target_rel is None:
                            row_ok = False
                            break
                        new_qips.append(target_rel)
                    if not row_ok:
                        stats["rows_dropped_missing_images"] += 1
                        continue
                    row["query_image_paths"] = new_qips

                    trace = row.get("trace") or []
                    for step in trace:
                        if "tool_result" not in step:
                            continue
                        tr = step["tool_result"] or {}
                        new_paths: list[str] = []
                        for p in tr.get("image_paths") or []:
                            bucket = _bucket_for(p)
                            resolved = _resolve_existing(p, parents, snap_atlas_to_grid=True)
                            if resolved is None:
                                row_ok = False
                                break
                            target_rel = _stage(resolved, bucket, p)
                            if target_rel is None:
                                row_ok = False
                                break
                            new_paths.append(target_rel)
                        if not row_ok:
                            break
                        tr["image_paths"] = new_paths
                        step["tool_result"] = tr
                    if not row_ok:
                        stats["rows_dropped_missing_images"] += 1
                        continue
                    row["trace"] = trace

                    out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stats["rows_kept"] += 1

    logger.info(
        "Unified corpus built: rows=%d kept=%d dropped=%d (symlinks=%d copies=%d skipped=%d) -> %s",
        stats["rows_input"],
        stats["rows_kept"],
        stats["rows_dropped_missing_images"],
        stats["images_symlinked"],
        stats["images_copied"],
        stats["images_skipped"],
        output_jsonl,
    )
    return stats

__all__ = ["build_unified_corpus"]

"""Unified-image-tree builder for expert-iteration unioned corpora.

Wave 1 problem: ``_validate_row`` resolves image paths relative to the
unioned JSONL's parent directory. A naive concatenation of
``base_corpus`` + ``round_*.jsonl`` produces rows whose paths point at
two different roots (the base-corpus parent and the iterative-corpus
parent), so half of them fail the existence check.

Solution: stage every referenced image into a single tree under
``<output_dir>/queries/`` and ``<output_dir>/atlas/``, then rewrite each
row's ``query_image_paths`` and trace ``tool_result.image_paths`` to
point at the unified tree before validation. Symlinks are preferred for
speed; on platforms or filesystems that don't allow symlinks (notably
Windows without Developer Mode + Docker bind mounts that drop symlink
support), we fall back to file copies.

Layout written under ``out_dir``::

    out_dir/
      queries/<original-relative-name>
      atlas/<original-relative-name>
      round_<k>_corpus.jsonl  (rewritten rows)

We preserve original filenames inside each subtree so two corpora
referring to the same physical image collide on the same staged target
(idempotent, deduplicated, restartable).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import shutil
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator

logger = logging.getLogger(__name__)


_QUERY_PREFIX = "queries/"
_ATLAS_PREFIX = "atlas/"

# When a query (slice) image's resolved source lives under
# ``<repo_root>/data/datasets/<plane>/<dataset>/...``, we skip staging and
# emit the canonical repo-relative path directly. Two wins:
#
#   1. No NTFS copy. The langslice-data-fast volume already hosts data/datasets
#      on ext4, so the JPEG decode at training time reads from fast storage
#      instead of an out/.../queries/ copy on slow NTFS-via-WSL2.
#   2. Query embedding cache actually hits. The cache is keyed on the
#      canonical `data/datasets/<plane>/<dataset>/...` path (per
#      `embeddings.precompute_query`); the prior `queries/<hash>_<name>.png`
#      shape was missing the cache 100% of the time.
#
# Atlas images don't follow this path — the snap-to-grid + path-shape
# requirements of the atlas cache (keyed on `atlas/<atlas>/<plane>/<basename>`)
# still demand staging into the unified tree.
_DATA_DATASETS_SEGMENT = "data/datasets/"

# Tolerance for snapping a synth-requested atlas position to the nearest
# rendered atlas image on disk.  0.10 mm comfortably covers the 0.02-0.05 mm
# off-grid drift that comes from synth_corpus drawing positions from empirical
# distributions while the rendered atlas grid sits at coarser (~0.10 mm) steps.
# Larger gaps almost always mean "asked for a position outside this atlas's
# range" and should still drop, not snap.
_ATLAS_SNAP_TOLERANCE_MM = 0.10

# Regex matching paths like "atlas/<atlas_name>/<plane>/<X.XX>mm.jpg" anywhere
# in the path string. Used by the snap fallback to extract the requested
# position and locate the directory of rendered images.
_ATLAS_POS_PATH_RE = re.compile(
    r"(?:^|/)atlas/(?P<atlas>[^/]+)/(?P<plane>[^/]+)/(?P<pos>-?\d+\.\d+)mm\.jpg$"
)


# ──────────────────────────────────────────────────────────────────────────
# Path resolution & symlink/copy
# ──────────────────────────────────────────────────────────────────────────

def _resolve_existing(
    rel: str,
    parents: Iterable[Path],
    *,
    snap_atlas_to_grid: bool = False,
) -> Path | None:
    """Try each candidate parent in order; return the first hit, or None.

    ``rel`` may be an absolute path (in which case parents are ignored if it
    exists), a path relative to one of the parents, or a path relative to the
    repo root. We also try replacing backslashes with forward slashes so
    Windows-style paths from a saved JSONL work on POSIX agents.

    snap_atlas_to_grid:
        When True and the exact path doesn't resolve, try snapping to the
        nearest rendered atlas image in the same atlas/plane directory
        (tolerance ``_ATLAS_SNAP_TOLERANCE_MM``).  Used for tool_result atlas
        images, which synth picks from a continuous distribution while the
        on-disk render grid is discrete.  Off for query images — those are
        real source slices and must resolve exactly.
    """
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
    """Fast check that ``path`` begins with a JPEG or PNG magic header.

    Catches the failure mode we hit on round 0: source-slice downloads that
    saved an HTML error response (starts with ``Content-Type:…``) under a
    ``.jpg`` extension.  PIL crashes on those during training; dropping them
    at staging is far cheaper than the alternative.
    """
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
    """Enumerate rendered (position_mm, abs_path) tuples in a plane directory.

    Cached per directory because every dropped row in a batch hits the same
    handful of (atlas, plane) dirs; re-listing thousands of times wastes
    time without changing the answer.  Sorted by position ascending so
    nearest-neighbor lookups can short-circuit if needed.
    """
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
    """Snap a missing atlas image path to the nearest rendered position on disk.

    Parses the path's ``<X.XX>mm.jpg`` token, scans the parent atlas/plane
    directory for available positions, and returns the nearest one within
    ``_ATLAS_SNAP_TOLERANCE_MM``.  Returns None if the path doesn't match the
    canonical atlas pattern, no directory is found, or the nearest position
    is beyond tolerance — the caller then drops the row, same as the
    pre-snap behavior.

    The 0.02-0.05 mm drift we typically observe is well below the model's
    own positional resolution (~0.17 mm MAE), so the snap is invisible to
    training quality while reclaiming the 40% drop rate we saw on round 0.
    """
    m = _ATLAS_POS_PATH_RE.search(rel_norm)
    if m is None:
        return None
    target_pos = float(m.group("pos"))
    rel_plane_dir = Path(rel_norm).parent  # preserves the full "…/atlas/<n>/<plane>" prefix

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
    """Symlink ``src`` to ``dst`` if possible; otherwise copy.

    Returns the strategy used: "exists" if dst already pointed at the same
    inode, "symlink", "copy", or "skip" if dst already exists with different
    contents (we leave it alone — first writer wins).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try:
            if dst.resolve() == src.resolve():
                return "exists"
        except OSError:
            pass
        # Different file already at dst: don't clobber.
        return "skip"
    try:
        os.symlink(src, dst)
        return "symlink"
    except (OSError, NotImplementedError):
        # Windows + Docker bind mounts often refuse symlinks; fall back.
        try:
            shutil.copy2(src, dst)
            return "copy"
        except OSError as exc:
            logger.warning("Failed to stage %s -> %s (%s)", src, dst, exc)
            return "skip"


def _find_repo_root_for_rewriter(start: Path) -> Path | None:
    """Walk up from *start* looking for pyproject.toml. Used for the
    data/datasets shortcut in build_unified_corpus. Mirror of
    sft/dataset.py::_find_repo_root so the rewriter and the validator agree
    on what "repo root" means.
    """
    start = Path(start).resolve()
    for cand in (start, *start.parents):
        if (cand / "pyproject.toml").is_file():
            return cand
    return None


def _bucket_for(path_str: str) -> str:
    """Decide whether a path goes into the queries/ or atlas/ subtree.

    The langslice-native trace schema stores query images in
    ``query_image_paths`` and atlas-tool-result images under
    ``trace[*].tool_result.image_paths``. Caller passes ``which`` directly,
    so this is just a sanity check that infers from path tokens when
    debugging an out-of-band rewrite.
    """
    norm = path_str.replace("\\", "/")
    if "/queries/" in norm or norm.startswith("queries/"):
        return _QUERY_PREFIX
    if "/atlas/" in norm or norm.startswith("atlas/"):
        return _ATLAS_PREFIX
    # Default to "atlas" since unknown items are most likely atlas
    # tool-result artifacts in our pipeline.
    return _ATLAS_PREFIX


def _rewritten_subpath(rel: str, *, bucket: str) -> str:
    """Return the unified-tree subpath for a referenced image.

    We strip the bucket prefix if present (so ``queries/foo.jpg`` and
    ``../queries/foo.jpg`` both land under ``queries/foo.jpg``) and
    flatten the remaining path so two distinct images with the same
    basename in different folders don't collide.

    The flattening uses an 8-char hash of the source path joined with the
    basename, balancing readability with collision avoidance.
    """
    rel_norm = rel.replace("\\", "/")
    if rel_norm.startswith(bucket):
        rel_norm = rel_norm[len(bucket):]
    # Try to keep clean basenames when the path is already simple.
    if "/" not in rel_norm:
        return bucket + rel_norm
    digest = hashlib.sha1(rel_norm.encode("utf-8")).hexdigest()[:8]
    name = Path(rel_norm).name
    return f"{bucket}{digest}_{name}"


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────

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
    """Concatenate base + iterative round JSONLs and stage all images.

    For each row in each input JSONL, every referenced image (query +
    every ``tool_result.image_paths`` entry) is resolved against:
        1. The JSONL's parent directory (the original "root").
        2. Any extra_image_roots (e.g. the repo root for absolute paths).
        3. The path itself if absolute.

    The resolved image is then symlinked (or copied) into
    ``output_dir/queries/`` or ``output_dir/atlas/``, and the row's path
    field is rewritten to the unified-tree relative path. Rows that
    reference images we cannot find are dropped with a warning so the
    final corpus passes ``_validate_row``.

    Returns a stats dict::

        {
          "rows_input": int,
          "rows_kept": int,
          "rows_dropped_missing_images": int,
          "images_symlinked": int,
          "images_copied": int,
          "images_skipped": int,
        }
    """
    output_dir = Path(output_dir)
    output_jsonl = Path(output_jsonl)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if iterative_jsonls is None:
        iterative_jsonls = sorted(Path(iterative_corpus_dir).glob("round_*.jsonl"))

    # Each source: (jsonl path, parent_root, preloaded_lines | None).
    # When preloaded_lines is set we iterate that list instead of the file —
    # used by the base_sample_n knob to mix in a random sample of the
    # distilled corpus rather than re-walking all 1011 rows.
    #
    # base_sample_n semantics:
    #   None  -> include the full base corpus
    #   N > 0 -> include a random sample of N rows
    #   0     -> exclude the base corpus entirely (pure-iterative training,
    #            used by fully-synth iSFT rounds where the base SFT corpus has
    #            already shaped the anchor checkpoint)
    sources: list[tuple[Path, Path, list[str] | None]] = []
    if base_sample_n == 0:
        logger.info(
            "base_sample_n=0 — excluding base corpus entirely; "
            "training on iterative rounds only",
        )
    elif base_corpus.is_file():
        if base_sample_n is not None and base_sample_n > 0:
            all_lines = base_corpus.read_text(encoding="utf-8").splitlines()
            nonempty = [l for l in all_lines if l.strip()]
            rng = random.Random(base_sample_seed)
            n = min(base_sample_n, len(nonempty))
            sampled = rng.sample(nonempty, n)
            logger.info(
                "Sampled %d / %d rows from distilled base corpus (seed=%d)",
                n, len(nonempty), base_sample_seed,
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

    # Cache (resolved-source) -> unified-tree-relative target so multiple rows
    # that reference the same physical image only pay the staging cost once.
    stage_cache: dict[Path, str] = {}

    # Pre-resolve the repo root so the data/datasets shortcut can emit
    # canonical repo-relative paths. We walk up from output_dir looking for
    # pyproject.toml; if not found (test harness layouts, etc.) the shortcut
    # is disabled and every query image is staged the old-fashioned way.
    repo_root = _find_repo_root_for_rewriter(output_dir)

    def _stage(src: Path, bucket: str, rel: str) -> str | None:
        try:
            src_resolved = src.resolve()
        except OSError:
            return None
        cached = stage_cache.get(src_resolved)
        if cached is not None:
            return cached
        # Query-image shortcut: if the source already lives under
        # <repo_root>/data/datasets/, emit its repo-relative path directly
        # and bypass staging. The dataset validator falls back to repo-root
        # resolution for paths that don't resolve relative to the JSONL
        # parent dir (sft/dataset.py::_find_repo_root).
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
        # ``exists`` is silent — already-correct staging.
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

                    # Rewrite query image paths.
                    new_qips: list[str] = []
                    row_ok = True
                    for q in row.get("query_image_paths", []) or []:
                        resolved = _resolve_existing(q, parents)
                        if resolved is None:
                            logger.warning(
                                "Drop row (subject_id=%s): missing query image %s "
                                "(roots=%s)", row.get("subject_id"), q, parents,
                            )
                            row_ok = False
                            break
                        if not _is_valid_image_header(resolved):
                            logger.warning(
                                "Drop row (subject_id=%s): query image is not "
                                "a valid JPEG/PNG (likely a saved error response) %s",
                                row.get("subject_id"), resolved,
                            )
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

                    # Rewrite tool_result.image_paths in every non-submit step.
                    # Atlas images snap to the nearest on-grid render so that
                    # synth-generated off-grid positions (typical drift 0.02 mm)
                    # don't drop entire rows; tolerance is bounded by
                    # ``_ATLAS_SNAP_TOLERANCE_MM`` so out-of-range requests
                    # still fail.
                    trace = row.get("trace") or []
                    for step in trace:
                        if "tool_result" not in step:
                            continue
                        tr = step["tool_result"] or {}
                        new_paths: list[str] = []
                        for p in tr.get("image_paths") or []:
                            bucket = _bucket_for(p)
                            resolved = _resolve_existing(
                                p, parents, snap_atlas_to_grid=True,
                            )
                            if resolved is None:
                                logger.warning(
                                    "Drop row (subject_id=%s): missing trace image %s",
                                    row.get("subject_id"), p,
                                )
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
        "Unified corpus built: rows=%d kept=%d dropped=%d "
        "(symlinks=%d copies=%d skipped=%d) -> %s",
        stats["rows_input"], stats["rows_kept"], stats["rows_dropped_missing_images"],
        stats["images_symlinked"], stats["images_copied"], stats["images_skipped"],
        output_jsonl,
    )
    return stats

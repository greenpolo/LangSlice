"""Snap atlas tool_result paths in a corpus jsonl + validate query images.

build_overnight_corpus.py emits exact float positions from the empirical
distribution; the on-disk atlas grid is discretized at varying steps. Rows
where any tool_result atlas image can't snap within
``_ATLAS_SNAP_TOLERANCE_MM`` (0.10mm) are dropped (matches path_rewriter
behavior). Additionally, every query_image_path is fs.stat'd here — rows
whose query JPG is missing on disk are dropped.

After this script, the output jsonl is **fully image-validated**: every
remaining row has all paths existing on disk. Pass
``--skip-image-validation`` to train_sft to take advantage of this and
skip the ~30 min of redundant NTFS-via-Docker p9 RPC overhead during
``load_examples`` for 65k+ row corpora.

Usage:
  python models/langslice-gemma-4/training/tools/snap_corpus_atlas_paths.py \
      --in  out/iSFT/<corpus>/combined_corpus.jsonl \
      --out out/iSFT/<corpus>/combined_corpus_snapped.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from functools import lru_cache
from pathlib import Path


def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    msg = "Could not locate LangSlice repo root from snap-corpus tool path."
    raise RuntimeError(msg)


HERE = Path(__file__).resolve().parent
REPO_ROOT = _find_repo_root()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("snap_corpus")

_ATLAS_SNAP_TOLERANCE_MM = 0.10  # mirror path_rewriter
_ATLAS_POS_PATH_RE = re.compile(
    r"(?:^|/)atlas/(?P<atlas>[^/]+)/(?P<plane>[^/]+)/(?P<pos>-?\d+\.\d+)mm\.jpg$"
)


@lru_cache(maxsize=256)
def _list_atlas_positions(plane_dir_str: str) -> tuple[tuple[float, str], ...]:
    plane_dir = Path(plane_dir_str)
    if not plane_dir.is_dir():
        return ()
    out: list[tuple[float, str]] = []
    for p in plane_dir.glob("*.jpg"):
        try:
            pos = float(p.stem.replace("mm", ""))
        except ValueError:
            continue
        out.append((pos, p.name))
    out.sort(key=lambda kv: kv[0])
    return tuple(out)


def _snap_atlas_path(rel: str) -> str | None:
    """Snap a `models/.../atlas/<a>/<p>/<X.XX>mm.jpg` path to nearest on-grid file.

    Returns the snapped relative path, or None if no grid entry is within
    tolerance (caller should drop the row).
    """
    abs_path = REPO_ROOT / rel
    if abs_path.is_file():
        return rel
    m = _ATLAS_POS_PATH_RE.search(rel.replace("\\", "/"))
    if not m:
        return None
    target_pos = float(m.group("pos"))
    plane_dir = (REPO_ROOT / rel).parent
    positions = _list_atlas_positions(str(plane_dir))
    if not positions:
        return None
    nearest_pos, nearest_name = min(positions, key=lambda kv: abs(kv[0] - target_pos))
    if abs(nearest_pos - target_pos) > _ATLAS_SNAP_TOLERANCE_MM:
        return None
    rel_parts = rel.replace("\\", "/").rsplit("/", 1)
    return f"{rel_parts[0]}/{nearest_name}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--out", dest="out_path", type=Path, required=True)
    args = ap.parse_args()

    n_in = 0
    n_kept = 0
    n_dropped_atlas = 0
    n_dropped_query = 0
    n_snapped_paths = 0
    n_exact_paths = 0

    with args.in_path.open("r", encoding="utf-8") as fin, \
         args.out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            stripped = line.strip()
            if not stripped:
                continue
            n_in += 1
            row = json.loads(stripped)
            # Validate query images first — cheaper to drop here than walk the
            # whole trace if the query doesn't exist.
            query_ok = True
            for q in row.get("query_image_paths", []) or []:
                if not (REPO_ROOT / q).is_file():
                    query_ok = False
                    break
            if not query_ok:
                n_dropped_query += 1
                continue
            # Snap atlas paths in tool_result; drop on out-of-tolerance miss.
            row_ok = True
            for step in row.get("trace", []):
                tr = step.get("tool_result")
                if not isinstance(tr, dict):
                    continue
                new_paths: list[str] = []
                for p in tr.get("image_paths") or []:
                    snapped = _snap_atlas_path(p)
                    if snapped is None:
                        row_ok = False
                        break
                    if snapped != p:
                        n_snapped_paths += 1
                    else:
                        n_exact_paths += 1
                    new_paths.append(snapped)
                if not row_ok:
                    break
                tr["image_paths"] = new_paths
            if not row_ok:
                n_dropped_atlas += 1
                continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_kept += 1
            if n_in % 10000 == 0:
                logger.info(
                    "processed %d rows (kept %d, dropped_atlas %d, dropped_query %d, "
                    "snapped %d, exact %d)",
                    n_in, n_kept, n_dropped_atlas, n_dropped_query,
                    n_snapped_paths, n_exact_paths,
                )

    logger.info(
        "DONE: in=%d kept=%d dropped_atlas=%d dropped_query=%d "
        "snapped_paths=%d exact_paths=%d",
        n_in, n_kept, n_dropped_atlas, n_dropped_query,
        n_snapped_paths, n_exact_paths,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

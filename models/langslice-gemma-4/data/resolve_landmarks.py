"""One-shot helper to draft `landmark_atlas_map.json` via fuzzy match against
BrainGlobe structure trees.

Run once after editing `landmarks.json`; review the output, hand-resolve any
entries flagged as ambiguous (`status: "ambiguous"`) or missing
(`status: "no_match"`), then save as `landmark_atlas_map.json`.

Usage:
    python models/langslice-gemma-4/data/resolve_landmarks.py \
        --atlases allen_mouse_25um whs_sd_rat_39um admba \
        --out models/langslice-gemma-4/data/landmark_atlas_map.draft.json
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from langslice_harness.atlas.core import load_atlas  # noqa: E402

_LANDMARKS_PATH = Path(__file__).resolve().parent / "landmarks.json"


def _flatten_landmarks(landmarks: dict) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for entries in landmarks["landmarks_by_orientation"].values():
        for entry in entries:
            name = entry["name"]
            if name not in seen:
                out.append(name)
                seen.add(name)
    return out


def _fuzzy_match(name: str, atlas) -> tuple[str | None, str]:
    """Return (acronym, status). Status: 'ok' | 'ambiguous' | 'no_match'."""
    candidates = []
    name_lower = name.lower()
    for s in atlas.structures.values():
        # Match against full name and acronym; both are useful signals.
        full = s.get("name", "").lower()
        ac = s.get("acronym", "")
        score = max(
            difflib.SequenceMatcher(None, name_lower, full).ratio(),
            difflib.SequenceMatcher(None, name_lower, ac.lower()).ratio(),
        )
        if score >= 0.55:
            candidates.append((score, ac))
    candidates.sort(key=lambda x: x[0], reverse=True)
    if not candidates:
        return (None, "no_match")
    if len(candidates) >= 2 and candidates[0][0] - candidates[1][0] < 0.05:
        return (candidates[0][1], "ambiguous")
    return (candidates[0][1], "ok")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--atlases", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    landmarks = json.loads(_LANDMARKS_PATH.read_text(encoding="utf-8"))
    names = _flatten_landmarks(landmarks)

    out: dict[str, dict] = {}
    for name in names:
        out[name] = {}
        for atlas_name in args.atlases:
            atlas = load_atlas(atlas_name)
            acronym, status = _fuzzy_match(name, atlas)
            out[name][atlas_name] = {
                "acronym": acronym,
                "include_descendants": False,  # curator decides
                "status": status,
            }

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote draft map for {len(names)} landmarks to {args.out}")
    print("Review entries with status='ambiguous' or 'no_match' before "
          "saving as landmark_atlas_map.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Standalone weight-update CLI.

Used by the expert-iteration multi-round driver as the curriculum hook
between rounds:

    python -m curriculum.cli \
        --slicebench-summary out/expert_iteration/run_001/round_0/slicebench_tiny/summary.json \
        --section-bins out/expert_iteration/run_001/section_bins.json \
        --prev-weights out/expert_iteration/run_001/round_0_weights.json \
        --output out/expert_iteration/run_001/round_1_weights.json \
        --alpha 1.0 \
        --max-weight-change 3.0

The ``--section-bins`` file is the JSON-serialized output of
:func:`bins.compute_section_bins`, with values shaped as either:

    {"<section_id>": <bin_idx_int>}                 # legacy int form
    {"<section_id>": ["<plane>", "<bin_label>"]}    # plane+label tuple form

Both shapes are accepted; the tuple form gives plane-aware weighting
while the int form averages MAE across planes per ``q_idx``.

If ``--prev-weights`` is omitted, smoothing + the cap are skipped and
the output is the raw computed weights. This is the right default for
round 0 → round 1.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .weights import (
    compute_weights,
    read_per_bin_mae,
    read_weights_json,
    write_weights_json,
)

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m curriculum.cli",
        description="Compute curriculum weights from a slicebench summary.",
    )
    p.add_argument(
        "--slicebench-summary", type=Path, required=True,
        help="Path to slicebench summary.json with per_coord_bin block.",
    )
    p.add_argument(
        "--section-bins", type=Path, required=True,
        help="JSON file mapping section_id → bin_idx (int) or [plane, bin_label].",
    )
    p.add_argument(
        "--output", type=Path, required=True,
        help="Where to write the new weights JSON.",
    )
    p.add_argument(
        "--prev-weights", type=Path, default=None,
        help="Optional previous-round weights for EMA smoothing + ratio cap.",
    )
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--max-weight-change", type=float, default=3.0)
    p.add_argument("--floor-fraction", type=float, default=0.1)
    p.add_argument("--smoothing", type=float, default=0.5)
    p.add_argument(
        "--baseline-mae", type=float, default=None,
        help="Optional baseline_mae override; defaults to mean of per_bin_mae.",
    )
    p.add_argument(
        "--round-idx", type=int, default=None,
        help="Optional round index recorded in output _metadata.",
    )
    return p.parse_args(argv)


def _load_section_bins(path: Path) -> dict[str, tuple[str, str] | int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, tuple[str, str] | int] = {}
    for sid, value in raw.items():
        if isinstance(value, list) and len(value) == 2:
            out[sid] = (str(value[0]), str(value[1]))
        elif isinstance(value, int):
            out[sid] = value
        else:
            raise ValueError(
                f"section_bins[{sid}] must be int or [plane, bin_label] list, got {value!r}"
            )
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)

    per_bin_mae = read_per_bin_mae(args.slicebench_summary)
    if not per_bin_mae:
        logger.warning(
            "%s has no non-empty per_coord_bin entries; emitting uniform weights.",
            args.slicebench_summary,
        )

    section_bins = _load_section_bins(args.section_bins)
    if not section_bins:
        logger.error("section_bins is empty; nothing to weight.")
        return 2

    prev_weights = (
        read_weights_json(args.prev_weights)
        if args.prev_weights and args.prev_weights.is_file()
        else None
    )

    weights = compute_weights(
        per_bin_mae,
        section_bins,
        baseline_mae=args.baseline_mae,
        alpha=args.alpha,
        max_weight_change=args.max_weight_change,
        prev_weights=prev_weights,
        floor_fraction=args.floor_fraction,
        smoothing=args.smoothing,
    )

    metadata = {
        "alpha": args.alpha,
        "max_weight_change": args.max_weight_change,
        "floor_fraction": args.floor_fraction,
        "smoothing": args.smoothing,
        "baseline_mae": args.baseline_mae,
        "n_per_bin_mae": len(per_bin_mae),
        "n_sections": len(section_bins),
        "had_prev_weights": prev_weights is not None,
        "slicebench_summary": str(args.slicebench_summary),
    }
    if args.round_idx is not None:
        metadata["round"] = int(args.round_idx)

    write_weights_json(args.output, weights, metadata=metadata)
    logger.info(
        "Wrote %d weights -> %s (per_bin_mae=%d, baseline=%s, prev=%s)",
        len(weights),
        args.output,
        len(per_bin_mae),
        args.baseline_mae if args.baseline_mae is not None else "auto",
        bool(prev_weights),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

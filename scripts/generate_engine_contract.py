"""Generate engine schema + TypeScript contract files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _build_parser() -> argparse.ArgumentParser:
    from langslice_harness.api.typegen import ENGINE_SCHEMA_PATH

    parser = argparse.ArgumentParser(
        description="Generate LangSlice engine schema and TypeScript contract artifacts.",
    )
    parser.add_argument(
        "--schema-out",
        type=Path,
        default=ENGINE_SCHEMA_PATH,
        help="Path (relative to project root) to write JSON schema bundle.",
    )
    return parser


def main() -> int:
    from langslice_harness.api.typegen import write_contract_outputs

    parser = _build_parser()
    args = parser.parse_args()

    schema_path, ts_paths = write_contract_outputs(
        project_root=PROJECT_ROOT,
        schema_path=args.schema_out,
    )

    print(f"Wrote schema bundle: {schema_path}")
    for ts_path in ts_paths:
        print(f"Wrote TypeScript contract: {ts_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Downsample all TIF files in data/ to max 2K PNG and delete originals.

Usage:
    python eval/downsample_tifs.py                    # process all data/
    python eval/downsample_tifs.py --dir data/rlvr    # process specific dir
    python eval/downsample_tifs.py --dry-run           # show what would be done
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # These are legitimate histology scans

MAX_LONG_EDGE = 2048


def downsample_tif(tif_path: Path, delete_original: bool = True) -> Path | None:
    """Downsample a TIF to max 2K PNG. Returns the PNG path or None on failure."""
    png_path = tif_path.with_suffix(".png")
    if png_path.exists():
        if delete_original and tif_path.exists():
            tif_path.unlink()
        return png_path

    try:
        img = Image.open(tif_path)
        original_size = img.size
        img.thumbnail((MAX_LONG_EDGE, MAX_LONG_EDGE), Image.LANCZOS)
        img.save(png_path, "PNG")

        tif_mb = tif_path.stat().st_size / 1e6
        png_mb = png_path.stat().st_size / 1e6

        if delete_original:
            tif_path.unlink()

        print(f"  {tif_path.name}: {original_size[0]}x{original_size[1]} ({tif_mb:.0f}MB) "
              f"-> {img.size[0]}x{img.size[1]} ({png_mb:.1f}MB)")
        return png_path
    except Exception as e:
        print(f"  ERROR: {tif_path.name}: {e}")
        return None


def process_directory(root: Path, dry_run: bool = False) -> None:
    tifs = sorted(root.rglob("*.tif"))
    if not tifs:
        print(f"No TIF files found in {root}")
        return

    total_tif_gb = sum(t.stat().st_size for t in tifs) / 1e9
    print(f"Found {len(tifs)} TIF files ({total_tif_gb:.1f} GB) in {root}")

    if dry_run:
        for t in tifs[:10]:
            print(f"  Would process: {t} ({t.stat().st_size / 1e6:.0f} MB)")
        if len(tifs) > 10:
            print(f"  ... and {len(tifs) - 10} more")
        return

    processed = 0
    for t in tifs:
        result = downsample_tif(t, delete_original=True)
        if result:
            processed += 1

    remaining_tifs = list(root.rglob("*.tif"))
    pngs = list(root.rglob("*.png"))
    total_png_gb = sum(p.stat().st_size for p in pngs) / 1e9
    print(f"\nDone. {processed}/{len(tifs)} converted. "
          f"PNGs: {len(pngs)} files ({total_png_gb:.1f} GB). "
          f"Remaining TIFs: {len(remaining_tifs)}")


def main():
    parser = argparse.ArgumentParser(description="Downsample TIFs to 2K PNG")
    parser.add_argument("--dir", type=str, default="data", help="Directory to process")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    process_directory(Path(args.dir), dry_run=args.dry_run)


if __name__ == "__main__":
    main()

"""Extract real brain-nucleus SHAPES from OpenOrganelle via microsim.

Source: jrc_mus-hippocampus-1 `nucleus_seg` (mouse hippocampus, FIB-SEM).
The layer is stored mode-binned (instance identity preserved across scales);
s0 = 128 nm, so s1 = 256 nm -- a near-perfect source for our 0.377 um grid.

Pipeline:
  read a block of nucleus_seg (bin_mode='standard' -> stored level, no re-bin)
  -> connected-components to separate individual nuclei
  -> drop border-touching (clipped) and out-of-range nuclei
  -> anisotropic downsample each to the lab grid (0.5 z, 0.377 xy um)
  -> save soft-occupancy shape templates (+ a montage to look at).

These are SHAPE only (the seg is the nuclear envelope). Internal DAPI/chromatin
texture is synthesized separately (user's choice) when stamping.

    python cosem_shapes.py                  # default block + level 1 (256nm)
    python cosem_shapes.py --level 1 --block 256 384 384
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

try:  # moved out of scipy.spatial.qhull in newer scipy
    from scipy.spatial import QhullError
except ImportError:  # pragma: no cover
    from scipy.spatial.qhull import QhullError

from .lab_optics import LAB_DZ_UM, LAB_PX_UM

OUT = Path(__file__).parent / "out" / "cosem"
OUT.mkdir(parents=True, exist_ok=True)

DATASET = "jrc_mus-hippocampus-1"
LAYER = "nucleus_seg"
# accept nuclei with equivalent diameter in this range (um); rejects debris + merges
MIN_DIAM_UM, MAX_DIAM_UM = 3.0, 16.0
# shape-quality gates (computed on the source-res binary mask):
#   solidity = voxels / convex-hull-volume  -> low for crescents / concave merges
#   aspect   = sqrt(lambda_max / lambda_min) of the inertia tensor -> high for slivers
MIN_SOLIDITY = 0.80
MAX_ASPECT = 3.5


def _shape_quality(mask: np.ndarray) -> tuple[float, float] | None:
    """(solidity, aspect) for a 3D binary mask, or None if degenerate (a sliver)."""
    coords = np.argwhere(mask)
    if coords.shape[0] < 16:
        return None
    try:
        hull_vol = ConvexHull(coords.astype(np.float64)).volume
    except (QhullError, ValueError):
        return None  # coplanar / degenerate -> reject as sliver
    if hull_vol <= 0:
        return None
    solidity = float(coords.shape[0] / hull_vol)
    c = coords - coords.mean(0)
    cov = (c.T @ c) / coords.shape[0]
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 1e-6, None)
    aspect = float(np.sqrt(ev[-1] / ev[0]))
    return solidity, aspect


def open_seg(level: int):
    """Open the seg tensorstore at `level`; return (ts, shape, vox_um)."""
    from microsim.cosem import CosemDataset

    img = CosemDataset.fetch(DATASET).image(name=LAYER)
    vox_um = abs(img.grid_scale[0]) / 1000.0 * (2 ** level)  # s0 scale * 2^level
    ts = img.read(level=level, bin_mode="standard", transpose=("z", "y", "x"))
    shp = tuple(int(s) for s in ts.shape)
    return ts, shp, vox_um


def read_subblock(ts, shp, start, block):
    """Read one (z,y,x) sub-block clipped to the volume; raise on a bad chunk."""
    stops = [min(s, st + b) for s, st, b in zip(shp, start, block, strict=False)]
    sl = tuple(slice(max(a, 0), b) for a, b in zip(start, stops, strict=False))
    return np.asarray(ts[sl].read().result())


def block_starts(shp, block, step):
    """Non-overlapping tile starts covering the WHOLE volume (z, y, x).

    Uses range(0, n, step) so a final partial tile at each axis end is included
    (read_subblock clips it). Sampling across all sub-regions averages out the
    local tissue co-alignment that biases any single contiguous block.
    """
    zs = list(range(0, shp[0], step)) or [0]
    ys = list(range(0, shp[1], step)) or [0]
    xs = list(range(0, shp[2], step)) or [0]
    return [(z, y, x) for z in zs for y in ys for x in xs]


def extract_nuclei(seg: np.ndarray, vox_um: float) -> list[np.ndarray]:
    """Connected-components -> per-nucleus soft-occupancy mask at the lab grid."""
    labels, n = ndimage.label(seg > 0)
    print(f"  {n} connected components in block {seg.shape}")
    objs = ndimage.find_objects(labels)
    vox_vol = vox_um ** 3
    zoom = (vox_um / LAB_DZ_UM, vox_um / LAB_PX_UM, vox_um / LAB_PX_UM)

    templates: list[np.ndarray] = []
    diams: list[float] = []
    drop = {"border": 0, "diam": 0, "shape": 0, "empty": 0}
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        # reject nuclei touching the block border (clipped, not whole)
        touches = any(s.start == 0 or s.stop == dim
                      for s, dim in zip(sl, seg.shape, strict=False))
        if touches:
            drop["border"] += 1
            continue
        mask = (labels[sl] == i)
        vol_um3 = mask.sum() * vox_vol
        diam = (6.0 * vol_um3 / np.pi) ** (1 / 3)
        if not (MIN_DIAM_UM <= diam <= MAX_DIAM_UM):
            drop["diam"] += 1
            continue
        # shape-quality gate on the source-res mask (drops crescents / slivers)
        q = _shape_quality(mask)
        if q is None or q[0] < MIN_SOLIDITY or q[1] > MAX_ASPECT:
            drop["shape"] += 1
            continue
        # anisotropic downsample the binary mask -> soft occupancy on the lab grid
        soft = ndimage.zoom(mask.astype(np.float32), zoom, order=1)
        if soft.max() < 0.3:
            drop["empty"] += 1
            continue
        templates.append(np.clip(soft, 0, 1).astype(np.float32))
        diams.append(diam)

    print(f"  rejected: {drop['border']} border, {drop['diam']} out-of-size, "
          f"{drop['shape']} bad-shape, {drop['empty']} empty")
    if diams:
        d = np.array(diams)
        print(f"  kept {len(templates)} whole nuclei | "
              f"equiv-diam um: min {d.min():.1f} median {np.median(d):.1f} "
              f"max {d.max():.1f}")
    return templates


def montage(templates: list[np.ndarray], path: Path, ncol: int = 8) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = templates[: min(len(templates), ncol * 2)]
    nrow = (len(sel) + ncol - 1) // ncol
    fig, ax = plt.subplots(nrow, ncol, figsize=(ncol * 1.5, nrow * 1.5),
                           facecolor="black")
    for a, t in zip(np.atleast_1d(ax).ravel(), sel, strict=False):
        a.imshow(t[t.shape[0] // 2], cmap="bone", interpolation="bilinear")
        a.axis("off")
    for a in np.atleast_1d(ax).ravel()[len(sel):]:
        a.axis("off")
    fig.suptitle("real hippocampus nucleus shapes (mid-z), downsampled to lab grid",
                 color="white", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor="black")
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=1)        # s1 = 256 nm
    ap.add_argument("--block", type=int, nargs=3, default=(256, 256, 256))
    ap.add_argument("--target", type=int, default=320, help="stop after this many nuclei")
    ap.add_argument("--max-blocks", type=int, default=30)
    args = ap.parse_args()
    block = tuple(args.block)

    print(f"opening {DATASET}::{LAYER} level s{args.level} ...")
    ts, shp, vox_um = open_seg(args.level)
    print(f"  volume {shp} vox={vox_um * 1000:.0f}nm "
          f"(~{shp[0] * vox_um:.0f}x{shp[1] * vox_um:.0f}x{shp[2] * vox_um:.0f}um)")

    starts = block_starts(shp, block, step=block[1])
    print(f"  tiling volume into {len(starts)} sub-blocks of {block} (whole-volume sample)")
    templates: list[np.ndarray] = []
    read_ok = 0
    for k, start in enumerate(starts[: args.max_blocks], start=1):
        try:
            seg = read_subblock(ts, shp, start, block)
        except Exception as exc:  # bad/corrupt blosc chunk -> skip this tile
            print(f"  [block {k}] start {start} SKIPPED ({type(exc).__name__})")
            continue
        read_ok += 1
        print(f"  [block {k}] start {start} -> {seg.shape} "
              f"({np.unique(seg).size} ids)")
        templates.extend(extract_nuclei(seg, vox_um))
        if len(templates) >= args.target:
            break

    print(f"read {read_ok} blocks; {len(templates)} total nuclei")
    if not templates:
        print("no whole nuclei found -- try a different block or level")
        return

    np.save(Path(__file__).parent / "assets" / "hippocampus_shapes.npy",
            np.array(templates, dtype=object), allow_pickle=True)
    print(f"  saved {len(templates)} templates -> {OUT / 'hippocampus_shapes.npy'}")
    montage(templates, OUT / "hippo_nuclei_montage.png")


if __name__ == "__main__":
    main()

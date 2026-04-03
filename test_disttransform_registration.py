"""
Distance-transform registration experiment.

Extracts edges from two brain images, computes inverted distance transforms
(smooth, texture-invariant fields), then registers them with DIPY SyN (SSD).
The resulting displacement field warps atlas borders onto the original slice.
"""

import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import distance_transform_edt
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric
import time

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SLICE_PATH = "test_runs/M01_002_007_clahe_model.png"
WARP_SLICE_PATH = "test_runs/overlay_prototype_v4_warp_slice.png"
BORDERS_PATH = "test_runs/atlas_borders_5.93mm_matched.png"
OUT_PATH = "test_runs/v4_disttransform_borders.jpg"

LONG_EDGE = 1024

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_gray(path):
    """Load image as float64 grayscale numpy array."""
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float64)


def resize_to(img, target_shape):
    """Resize numpy array (HxW) to target_shape (H, W) using PIL."""
    pil = Image.fromarray(img.astype(np.uint8) if img.max() > 1.0 else (img * 255).astype(np.uint8))
    pil = pil.resize((target_shape[1], target_shape[0]), Image.LANCZOS)
    return np.array(pil, dtype=np.float64)


def compute_work_size(orig_h, orig_w, long_edge):
    """Return (H, W) scaled so long edge = long_edge, preserving aspect."""
    ratio = long_edge / max(orig_h, orig_w)
    return int(round(orig_h * ratio)), int(round(orig_w * ratio))


def extract_edges_canny(img_f64, low=20, high=80):
    """Canny edge detection on a float64 [0,255] image. Returns binary uint8."""
    img_u8 = np.clip(img_f64, 0, 255).astype(np.uint8)
    # Light Gaussian blur to reduce noise while keeping edges
    blurred = cv2.GaussianBlur(img_u8, (5, 5), sigmaX=1.0)
    edges = cv2.Canny(blurred, low, high)
    return edges  # uint8, 0 or 255


def inverted_distance_transform(edge_map):
    """
    Distance transform on the BACKGROUND (non-edge pixels), then invert
    so that edges have HIGH values and flat regions have LOW values.
    This gives gradient-based optimizers a nice smooth landscape that
    pulls toward structural edges.
    """
    # edge_map: 255 at edges, 0 elsewhere
    binary = (edge_map == 0).astype(np.float64)  # 1 where NO edge
    dt = distance_transform_edt(binary)
    # Invert: max - dt so edges are at the peak
    dt_inv = dt.max() - dt
    # Normalize to [0, 1]
    if dt_inv.max() > 0:
        dt_inv = dt_inv / dt_inv.max()
    return dt_inv


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    # 1. Load images
    print("[1/7] Loading images...")
    slice_img = load_gray(SLICE_PATH)
    warp_img = load_gray(WARP_SLICE_PATH)
    borders_img = load_gray(BORDERS_PATH)
    print(f"  Slice:   {slice_img.shape}")
    print(f"  Warped:  {warp_img.shape}")
    print(f"  Borders: {borders_img.shape}")

    # 2. Compute working size from the original slice
    work_h, work_w = compute_work_size(slice_img.shape[0], slice_img.shape[1], LONG_EDGE)
    print(f"  Work size: {work_h} x {work_w}")

    # Resize all to work size
    slice_work = resize_to(slice_img, (work_h, work_w))
    warp_work = resize_to(warp_img, (work_h, work_w))
    borders_work = resize_to(borders_img, (work_h, work_w))

    # 3. Edge extraction
    print("[2/7] Extracting edges (Canny 20-80)...")
    edges_slice = extract_edges_canny(slice_work, low=20, high=80)
    edges_warp = extract_edges_canny(warp_work, low=20, high=80)
    slice_edge_count = np.count_nonzero(edges_slice)
    warp_edge_count = np.count_nonzero(edges_warp)
    print(f"  Slice edges: {slice_edge_count} px ({100*slice_edge_count/(work_h*work_w):.1f}%)")
    print(f"  Warp edges:  {warp_edge_count} px ({100*warp_edge_count/(work_h*work_w):.1f}%)")

    # 4. Distance transforms (inverted)
    print("[3/7] Computing inverted distance transforms...")
    dt_slice = inverted_distance_transform(edges_slice)
    dt_warp = inverted_distance_transform(edges_warp)
    print(f"  DT slice range: [{dt_slice.min():.3f}, {dt_slice.max():.3f}]")
    print(f"  DT warp range:  [{dt_warp.min():.3f}, {dt_warp.max():.3f}]")

    # 5. SyN Registration
    print("[4/7] Running DIPY SyN registration (SSD metric)...")
    print("  This may take a minute...")
    t_reg = time.time()

    dim = 2
    metric = SSDMetric(dim)
    # Multi-resolution: coarse to fine
    level_iters = [200, 100, 50, 25]
    sdr = SymmetricDiffeomorphicRegistration(
        metric,
        level_iters=level_iters,
        step_length=0.25,
        opt_tol=1e-5,
    )

    # Static = slice (target), Moving = warp (to be aligned to target)
    mapping = sdr.optimize(dt_slice, dt_warp)
    reg_time = time.time() - t_reg
    print(f"  Registration completed in {reg_time:.1f}s")

    # 6. Extract forward displacement field
    print("[5/7] Extracting forward displacement field...")
    fwd_field = mapping.get_forward_field()
    print(f"  Field shape: {fwd_field.shape}")
    print(f"  Field magnitude range: [{np.linalg.norm(fwd_field, axis=-1).min():.2f}, "
          f"{np.linalg.norm(fwd_field, axis=-1).max():.2f}] px")

    # 7. Warp the atlas borders using the displacement field
    print("[6/7] Warping atlas borders with displacement field...")
    # The mapping transforms from warp-space -> slice-space
    # We want to warp the borders (which are in warp-space) to slice-space
    warped_borders = mapping.transform(borders_work, interpolation='linear')
    print(f"  Warped borders range: [{warped_borders.min():.1f}, {warped_borders.max():.1f}]")

    # 8. Composite: green overlay on original slice
    print("[7/7] Compositing green overlay and saving...")

    # Normalize slice to uint8
    slice_u8 = np.clip(slice_work, 0, 255).astype(np.uint8)

    # Create RGB canvas from grayscale slice
    canvas = np.stack([slice_u8, slice_u8, slice_u8], axis=-1)

    # Threshold warped borders to get a mask
    border_mask = warped_borders > 30  # borders are bright lines on dark bg

    # Green overlay: set border pixels to green
    canvas[border_mask, 0] = 0     # R
    canvas[border_mask, 1] = 255   # G
    canvas[border_mask, 2] = 0     # B

    # Save
    out_pil = Image.fromarray(canvas)
    out_pil.save(OUT_PATH, quality=95)

    total = time.time() - t0
    print(f"\nDone! Saved to {OUT_PATH}")
    print(f"Total time: {total:.1f}s")


if __name__ == "__main__":
    main()

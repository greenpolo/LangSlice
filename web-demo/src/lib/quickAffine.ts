/**
 * Silhouette-based affine preview registration for the 3D viewer.
 *
 * Browser port of `harness/registration/quick_affine.py`. Extracts a filled
 * tissue silhouette via Otsu + corner-polarity + hole-fill + largest-CC, then
 * aligns it to the atlas root silhouette (alpha>0 of the baked section PNG)
 * via a closed-form image-moments affine. The 4-way sign ambiguity on the
 * eigenvectors is resolved by trying all four and keeping the best silhouette
 * IoU. Output is an RGBA ImageBitmap, alpha-clipped to the atlas silhouette,
 * suitable for direct upload as a Three.js texture.
 */

import { getCoronalSliceLayers } from "./browserCommands";

const TARGET_LONG_EDGE = 512;
const MIN_AREA_FRAC = 0.05;
const MAX_AREA_FRAC = 0.9;
const SIGN_PATTERNS: ReadonlyArray<readonly [number, number]> = [
  [1, 1],
  [-1, 1],
  [1, -1],
  [-1, -1],
];

export interface QuickAffineParams {
  /** Input slice. Already loaded; we won't fetch it. */
  slice: ImageBitmap | HTMLImageElement;
  /** AP position in mm. We snap to the nearest bundled atlas section. */
  atlasApMm: number;
  /** Long-edge target for silhouette ops + warp output. Default 512. */
  targetLongEdge?: number;
}

export interface QuickAffineResult {
  /** RGBA bitmap, slice warped to atlas, alpha clipped to atlas root silhouette.
   *  Suitable for direct upload as a Three.js texture. */
  warped: ImageBitmap;
  /** Silhouette IoU of the winning sign pattern (0..1). */
  iou: number;
  /** Wall-clock elapsed ms. */
  elapsedMs: number;
  /** Which of the 4 sign patterns won — [1,1] / [-1,1] / [1,-1] / [-1,-1]. */
  signPattern: [number, number];
}

interface SizedRgba {
  rgba: Uint8ClampedArray;
  w: number;
  h: number;
}

interface Pose {
  cx: number;
  cy: number;
  /** [λ1, λ2] descending. */
  eigvals: [number, number];
  /** Column-major 2x2: [[v1x, v2x], [v1y, v2y]]. */
  V: [[number, number], [number, number]];
}

function makeCanvas(w: number, h: number): {
  canvas: OffscreenCanvas | HTMLCanvasElement;
  ctx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;
} {
  if (typeof OffscreenCanvas !== "undefined") {
    const c = new OffscreenCanvas(w, h);
    const ctx = c.getContext("2d") as OffscreenCanvasRenderingContext2D | null;
    if (!ctx) throw new Error("Failed to acquire OffscreenCanvas 2D context.");
    return { canvas: c, ctx };
  }
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  if (!ctx) throw new Error("Failed to acquire HTMLCanvasElement 2D context.");
  return { canvas: c, ctx };
}

async function canvasToBitmap(
  canvas: OffscreenCanvas | HTMLCanvasElement,
): Promise<ImageBitmap> {
  return await createImageBitmap(canvas as unknown as CanvasImageSource);
}

/** Resize input so its long edge equals `longEdge`, preserving aspect. */
function resizeLongEdgeToRgba(
  src: ImageBitmap | HTMLImageElement,
  longEdge: number,
): SizedRgba {
  const sw = (src as ImageBitmap).width ?? (src as HTMLImageElement).naturalWidth;
  const sh = (src as ImageBitmap).height ?? (src as HTMLImageElement).naturalHeight;
  const current = Math.max(sw, sh);
  const scale = current === longEdge ? 1 : longEdge / current;
  const w = Math.max(1, Math.round(sw * scale));
  const h = Math.max(1, Math.round(sh * scale));
  const { canvas, ctx } = makeCanvas(w, h);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(src as CanvasImageSource, 0, 0, w, h);
  const img = ctx.getImageData(0, 0, w, h);
  void canvas;
  return { rgba: img.data, w, h };
}

function rgbaToLuma(rgba: Uint8ClampedArray, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j++) {
    out[j] = (0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2]) | 0;
  }
  return out;
}

/** Otsu threshold. Returns 0/255 binary mask. */
function otsu(gray: Uint8Array): Uint8Array {
  const n = gray.length;
  const hist = new Int32Array(256);
  for (let i = 0; i < n; i++) hist[gray[i]]++;
  let sumAll = 0;
  for (let t = 0; t < 256; t++) sumAll += t * hist[t];
  let wB = 0;
  let sumB = 0;
  let bestT = 0;
  let bestVar = -1;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = n - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sumAll - sumB) / wF;
    const diff = mB - mF;
    const between = wB * wF * diff * diff;
    if (between > bestVar) {
      bestVar = between;
      bestT = t;
    }
  }
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = gray[i] > bestT ? 255 : 0;
  return out;
}

/** Sample four corner patches; if their mean > 127, invert (background was bright). */
function polarizeByCorners(bin: Uint8Array, w: number, h: number): void {
  const patch = Math.max(4, Math.min(h, w) >> 5);
  let sum = 0;
  let count = 0;
  for (let y = 0; y < patch; y++) {
    for (let x = 0; x < patch; x++) {
      sum += bin[y * w + x];
      sum += bin[y * w + (w - 1 - x)];
      sum += bin[(h - 1 - y) * w + x];
      sum += bin[(h - 1 - y) * w + (w - 1 - x)];
      count += 4;
    }
  }
  if (sum / count > 127) {
    for (let i = 0; i < bin.length; i++) bin[i] = 255 - bin[i];
  }
}

/** Scanline flood-fill the BACKGROUND (0-pixels) from every border 0-pixel.
 *  Any 0-pixel not reached is an interior hole; flip those to 255. */
function fillHoles(bin: Uint8Array, w: number, h: number): Uint8Array {
  const n = w * h;
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = bin[i];
  // visited tracks BG pixels reachable from the border
  const visited = new Uint8Array(n);
  const stack = new Int32Array(n);
  let sp = 0;
  const push = (x: number, y: number): void => {
    const idx = y * w + x;
    if (visited[idx] || out[idx] !== 0) return;
    visited[idx] = 1;
    stack[sp++] = idx;
  };
  for (let x = 0; x < w; x++) {
    push(x, 0);
    push(x, h - 1);
  }
  for (let y = 0; y < h; y++) {
    push(0, y);
    push(w - 1, y);
  }
  while (sp > 0) {
    const idx = stack[--sp];
    const y = (idx / w) | 0;
    // walk left
    let xl = idx - y * w;
    while (xl > 0 && out[y * w + xl - 1] === 0 && !visited[y * w + xl - 1]) {
      xl--;
      visited[y * w + xl] = 1;
    }
    // walk right
    let xr = idx - y * w;
    while (xr < w - 1 && out[y * w + xr + 1] === 0 && !visited[y * w + xr + 1]) {
      xr++;
      visited[y * w + xr] = 1;
    }
    // probe rows above/below across [xl..xr]
    for (let x = xl; x <= xr; x++) {
      if (y > 0) {
        const up = (y - 1) * w + x;
        if (out[up] === 0 && !visited[up]) {
          visited[up] = 1;
          stack[sp++] = up;
        }
      }
      if (y < h - 1) {
        const dn = (y + 1) * w + x;
        if (out[dn] === 0 && !visited[dn]) {
          visited[dn] = 1;
          stack[sp++] = dn;
        }
      }
    }
  }
  // Any 0-pixel not visited is an interior hole; fill it.
  for (let i = 0; i < n; i++) {
    if (out[i] === 0 && !visited[i]) out[i] = 255;
  }
  return out;
}

/** 8-connected two-pass union-find labeling; keep only the largest non-bg CC. */
function largestComponent(bin: Uint8Array, w: number, h: number): Uint8Array {
  const n = w * h;
  const labels = new Int32Array(n);
  const parent: number[] = [0]; // label 0 = background
  const find = (x: number): number => {
    let r = x;
    while (parent[r] !== r) r = parent[r];
    while (parent[x] !== r) {
      const nx = parent[x];
      parent[x] = r;
      x = nx;
    }
    return r;
  };
  const union = (a: number, b: number): void => {
    const ra = find(a);
    const rb = find(b);
    if (ra === rb) return;
    if (ra < rb) parent[rb] = ra;
    else parent[ra] = rb;
  };

  let nextLabel = 1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      if (bin[idx] === 0) continue;
      // 4 already-visited 8-neighbors: NW, N, NE, W
      const nbrs: number[] = [];
      if (y > 0) {
        if (x > 0) {
          const l = labels[idx - w - 1];
          if (l) nbrs.push(l);
        }
        const l = labels[idx - w];
        if (l) nbrs.push(l);
        if (x < w - 1) {
          const l2 = labels[idx - w + 1];
          if (l2) nbrs.push(l2);
        }
      }
      if (x > 0) {
        const l = labels[idx - 1];
        if (l) nbrs.push(l);
      }
      if (nbrs.length === 0) {
        labels[idx] = nextLabel;
        parent.push(nextLabel);
        nextLabel++;
      } else {
        let minL = nbrs[0];
        for (let k = 1; k < nbrs.length; k++) if (nbrs[k] < minL) minL = nbrs[k];
        labels[idx] = minL;
        for (let k = 0; k < nbrs.length; k++) union(minL, nbrs[k]);
      }
    }
  }

  // Second pass: resolve to roots and count areas.
  const areas: number[] = [];
  for (let i = 0; i < n; i++) {
    const l = labels[i];
    if (l === 0) continue;
    const r = find(l);
    labels[i] = r;
    while (areas.length <= r) areas.push(0);
    areas[r]++;
  }
  let bestLabel = 0;
  let bestArea = 0;
  for (let r = 1; r < areas.length; r++) {
    if (areas[r] > bestArea) {
      bestArea = areas[r];
      bestLabel = r;
    }
  }
  const out = new Uint8Array(n);
  if (bestLabel === 0) return out;
  for (let i = 0; i < n; i++) out[i] = labels[i] === bestLabel ? 255 : 0;
  return out;
}

function extractSliceSilhouette(gray: Uint8Array, w: number, h: number): Uint8Array {
  const bin = otsu(gray);
  polarizeByCorners(bin, w, h);
  const filled = fillHoles(bin, w, h);
  return largestComponent(filled, w, h);
}

/** Compute (centroid, eigvalsDesc, V_cols) for a 0/255 mask. */
function momentsPose(mask: Uint8Array, w: number, h: number): Pose {
  let m00 = 0;
  let m10 = 0;
  let m01 = 0;
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      if (mask[row + x] > 0) {
        m00++;
        m10 += x;
        m01 += y;
      }
    }
  }
  if (m00 < 1e-6) throw new Error("Empty mask — cannot compute moments.");
  const cx = m10 / m00;
  const cy = m01 / m00;
  let s20 = 0;
  let s02 = 0;
  let s11 = 0;
  for (let y = 0; y < h; y++) {
    const dy = y - cy;
    const row = y * w;
    for (let x = 0; x < w; x++) {
      if (mask[row + x] > 0) {
        const dx = x - cx;
        s20 += dx * dx;
        s02 += dy * dy;
        s11 += dx * dy;
      }
    }
  }
  const a = s20 / m00;
  const d = s02 / m00;
  const b = s11 / m00;
  const trace = a + d;
  const halfDiff = (a - d) / 2;
  const disc = Math.sqrt(Math.max(0, halfDiff * halfDiff + b * b));
  const l1 = trace / 2 + disc;
  const l2 = trace / 2 - disc;
  let v1x: number;
  let v1y: number;
  if (Math.abs(b) > 1e-12) {
    v1x = l1 - d;
    v1y = b;
  } else if (a >= d) {
    v1x = 1;
    v1y = 0;
  } else {
    v1x = 0;
    v1y = 1;
  }
  const norm = Math.hypot(v1x, v1y) || 1;
  v1x /= norm;
  v1y /= norm;
  // Perpendicular for v2.
  const v2x = -v1y;
  const v2y = v1x;
  return {
    cx,
    cy,
    eigvals: [l1, l2],
    V: [
      [v1x, v2x],
      [v1y, v2y],
    ],
  };
}

/** Build a 2x3 affine `[R | t]` mapping src silhouette → dst silhouette,
 *  with the given sign pattern flipping columns of V_src. */
function affineFromPose(
  src: Pose,
  dst: Pose,
  sign: readonly [number, number],
): { R: [[number, number], [number, number]]; t: [number, number] } {
  // V_src signed: flip column signs.
  const s0 = sign[0];
  const s1 = sign[1];
  const Vs: [[number, number], [number, number]] = [
    [src.V[0][0] * s0, src.V[0][1] * s1],
    [src.V[1][0] * s0, src.V[1][1] * s1],
  ];
  const sx = Math.sqrt(
    Math.max(dst.eigvals[0], 1e-9) / Math.max(src.eigvals[0], 1e-9),
  );
  const sy = Math.sqrt(
    Math.max(dst.eigvals[1], 1e-9) / Math.max(src.eigvals[1], 1e-9),
  );
  // M = V_dst · diag(sx, sy)  (scales V_dst columns)
  const M00 = dst.V[0][0] * sx;
  const M01 = dst.V[0][1] * sy;
  const M10 = dst.V[1][0] * sx;
  const M11 = dst.V[1][1] * sy;
  // R = M · Vs^T
  const R00 = M00 * Vs[0][0] + M01 * Vs[0][1];
  const R01 = M00 * Vs[1][0] + M01 * Vs[1][1];
  const R10 = M10 * Vs[0][0] + M11 * Vs[0][1];
  const R11 = M10 * Vs[1][0] + M11 * Vs[1][1];
  const tx = dst.cx - (R00 * src.cx + R01 * src.cy);
  const ty = dst.cy - (R10 * src.cx + R11 * src.cy);
  return { R: [[R00, R01], [R10, R11]], t: [tx, ty] };
}

/** Make an RGBA ImageBitmap from a 0/255 mask, white where mask>0. */
async function maskToBitmap(
  mask: Uint8Array,
  w: number,
  h: number,
): Promise<ImageBitmap> {
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let i = 0, j = 0; i < mask.length; i++, j += 4) {
    const v = mask[i] > 0 ? 255 : 0;
    rgba[j] = v;
    rgba[j + 1] = v;
    rgba[j + 2] = v;
    rgba[j + 3] = v;
  }
  return await createImageBitmap(new ImageData(rgba, w, h));
}

/** Build an RGBA ImageBitmap from a sized RGBA buffer. */
async function rgbaToBitmap(src: SizedRgba): Promise<ImageBitmap> {
  // Copy into a fresh Uint8ClampedArray so TS 6 sees a plain ArrayBuffer
  // backing store (getImageData returns Uint8ClampedArray<ArrayBufferLike>,
  // which the ImageData constructor's strict typing rejects).
  const copy = new Uint8ClampedArray(src.rgba.length);
  copy.set(src.rgba);
  return await createImageBitmap(new ImageData(copy, src.w, src.h));
}

/** Apply a 2x3 affine `[R|t]` to `bitmap` onto a fresh w×h canvas; return raw RGBA. */
function warpToRgba(
  bitmap: ImageBitmap,
  R: [[number, number], [number, number]],
  t: [number, number],
  w: number,
  h: number,
  smooth: boolean,
): Uint8ClampedArray {
  const { canvas, ctx } = makeCanvas(w, h);
  ctx.imageSmoothingEnabled = smooth;
  ctx.imageSmoothingQuality = smooth ? "high" : "low";
  // Canvas2D setTransform args: (a, b, c, d, e, f) where the mapping is
  //   x' = a*x + c*y + e
  //   y' = b*x + d*y + f
  // Our affine is [R | t] with x' = R00*x + R01*y + tx, so a=R00, b=R10, c=R01, d=R11.
  ctx.setTransform(R[0][0], R[1][0], R[0][1], R[1][1], t[0], t[1]);
  ctx.drawImage(bitmap, 0, 0);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  const img = ctx.getImageData(0, 0, w, h);
  void canvas;
  return img.data;
}

function rgbaAlphaToMask(rgba: Uint8ClampedArray): Uint8Array {
  const out = new Uint8Array(rgba.length >> 2);
  for (let i = 0, j = 3; i < out.length; i++, j += 4) {
    out[i] = rgba[j] > 0 ? 255 : 0;
  }
  return out;
}

function maskIoU(a: Uint8Array, b: Uint8Array): number {
  let inter = 0;
  let union = 0;
  for (let i = 0; i < a.length; i++) {
    const av = a[i] > 0 ? 1 : 0;
    const bv = b[i] > 0 ? 1 : 0;
    if (av | bv) {
      union++;
      if (av & bv) inter++;
    }
  }
  return union > 0 ? inter / union : 0;
}

async function fetchAtlasMask(
  apMm: number,
  w: number,
  h: number,
): Promise<Uint8Array> {
  const layers = await getCoronalSliceLayers(apMm);
  const blob = await (await fetch(layers.sectionUrl)).blob();
  const bmp = await createImageBitmap(blob);
  const { canvas, ctx } = makeCanvas(w, h);
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(bmp, 0, 0, w, h);
  bmp.close();
  const img = ctx.getImageData(0, 0, w, h);
  void canvas;
  const out = new Uint8Array(w * h);
  for (let i = 0, j = 3; i < out.length; i++, j += 4) {
    out[i] = img.data[j] > 0 ? 255 : 0;
  }
  return out;
}

export async function quickAffineRegister(
  p: QuickAffineParams,
): Promise<QuickAffineResult> {
  const started = performance.now();
  const longEdge = p.targetLongEdge ?? TARGET_LONG_EDGE;

  const sliceSized = resizeLongEdgeToRgba(p.slice, longEdge);
  const { w, h } = sliceSized;
  const gray = rgbaToLuma(sliceSized.rgba, w, h);
  const sliceMask = extractSliceSilhouette(gray, w, h);

  let on = 0;
  for (let i = 0; i < sliceMask.length; i++) if (sliceMask[i] > 0) on++;
  const areaFrac = on / sliceMask.length;
  if (areaFrac < MIN_AREA_FRAC || areaFrac > MAX_AREA_FRAC) {
    const pct = (areaFrac * 100).toFixed(2);
    const lo = (MIN_AREA_FRAC * 100).toFixed(0);
    const hi = (MAX_AREA_FRAC * 100).toFixed(0);
    throw new Error(
      `Slice silhouette area fraction ${pct}% out of range [${lo}%, ${hi}%] — Otsu likely failed.`,
    );
  }

  const atlasMask = await fetchAtlasMask(p.atlasApMm, w, h);

  const srcPose = momentsPose(sliceMask, w, h);
  const dstPose = momentsPose(atlasMask, w, h);

  const sliceMaskBitmap = await maskToBitmap(sliceMask, w, h);

  let bestIoU = -1;
  let bestR: [[number, number], [number, number]] | null = null;
  let bestT: [number, number] | null = null;
  let bestSign: [number, number] = [1, 1];
  for (const sign of SIGN_PATTERNS) {
    const { R, t } = affineFromPose(srcPose, dstPose, sign);
    const warpedRgba = warpToRgba(sliceMaskBitmap, R, t, w, h, false);
    const warpedMask = rgbaAlphaToMask(warpedRgba);
    const iou = maskIoU(warpedMask, atlasMask);
    if (iou > bestIoU) {
      bestIoU = iou;
      bestR = R;
      bestT = t;
      bestSign = [sign[0], sign[1]];
    }
  }
  sliceMaskBitmap.close();
  if (!bestR || !bestT) throw new Error("Affine search produced no candidate.");

  // Apply winning affine to slice RGB, smooth.
  const sliceRgbBitmap = await rgbaToBitmap(sliceSized);
  const { canvas: finalCanvas, ctx: finalCtx } = makeCanvas(w, h);
  finalCtx.imageSmoothingEnabled = true;
  finalCtx.imageSmoothingQuality = "high";
  finalCtx.setTransform(
    bestR[0][0],
    bestR[1][0],
    bestR[0][1],
    bestR[1][1],
    bestT[0],
    bestT[1],
  );
  finalCtx.drawImage(sliceRgbBitmap, 0, 0);
  finalCtx.setTransform(1, 0, 0, 1, 0, 0);
  sliceRgbBitmap.close();

  // Clip to atlas root via destination-in. Use the atlas mask as the alpha
  // layer (RGB = white, alpha = mask). Anything outside the mask becomes
  // fully transparent.
  const atlasAlphaBitmap = await maskToBitmap(atlasMask, w, h);
  finalCtx.globalCompositeOperation = "destination-in";
  finalCtx.drawImage(atlasAlphaBitmap, 0, 0);
  finalCtx.globalCompositeOperation = "source-over";
  atlasAlphaBitmap.close();

  const warped = await canvasToBitmap(finalCanvas);
  const elapsedMs = performance.now() - started;
  return { warped, iou: bestIoU, elapsedMs, signPattern: bestSign };
}

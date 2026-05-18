/**
 * Browser port of `harness/registration/image_gen_registration.py`.
 *
 * Sends three images (atlas colored regions, atlas reference, slice) plus a
 * fixed prompt to Gemini's image-generation model, which returns the colored
 * regions warped to match the slice. itk-wasm/elastix then computes the dense
 * affine + B-spline transform between the atlas colored regions (moving) and
 * the model output (fixed). The transform is reapplied per-channel via
 * transformix to the atlas reference + atlas borders, alpha-clipped to the
 * atlas root silhouette, and returned as ImageBitmaps ready for texture
 * upload. Mirrors `_register_colored_images` + `_warp_atlas_rgb` exactly.
 */

import { GoogleGenAI } from "@google/genai";
import {
  defaultParameterMap,
  elastix,
  transformix,
} from "@itk-wasm/elastix";
import { getCoronalSliceLayers } from "./browserCommands";

const MAX_LONG_EDGE = 2048;
const DEFAULT_MODEL = "gemini-3.1-flash-image-preview";

const SEGMENTATION_PROMPT =
  "Deform the colored region map (Image 1) to match the shape of the " +
  "real brain tissue in Image 3.\n" +
  "\n" +
  "IMAGE 1: A colored brain atlas region map. Each anatomical region " +
  "is a unique solid color.\n" +
  "IMAGE 2: A grayscale atlas reference showing the same brain anatomy.\n" +
  "IMAGE 3: A real histology photograph of a mouse brain coronal section.\n" +
  "\n" +
  "Warp each colored region in Image 1 so it aligns with the " +
  "corresponding anatomy visible in Image 3. The atlas is symmetric " +
  "and idealized; the real tissue is asymmetric, stretched, and may " +
  "have tears or damage. Use Image 2 to identify how atlas structures " +
  "correspond to features in the histology.\n" +
  "\n" +
  "Preserve the exact colors from Image 1. Where tissue is damaged " +
  "or missing in Image 3, fill in the expected atlas regions. Reflect " +
  "the natural left-right asymmetry of this individual brain section.\n" +
  "\n" +
  "Output only the deformed colored regions on a black background.";

export type ImageGenStage = "gemini" | "elastix" | "warp" | "done";

export type RegistrationThinkingPreset = "off" | "low" | "medium" | "high";
export type RegistrationMediaResolutionPreset = "low" | "medium" | "high";

export interface ImageGenRegistrationParams {
  /** Already loaded. */
  slice: ImageBitmap | HTMLImageElement;
  /** Snap to nearest bundled atlas section. */
  atlasApMm: number;
  /** Gemini Developer API key. The judge pastes this. */
  geminiApiKey: string;
  /** Optional override; defaults to "gemini-3.1-flash-image-preview". */
  imageModelId?: string;
  /** Optional AbortSignal — Gemini call respects it; elastix doesn't natively. */
  signal?: AbortSignal;
  /** Optional progress callback. */
  onProgress?: (stage: ImageGenStage, note?: string) => void;
  /** Optional Gemini knobs — only honored when the resolved model id matches
   *  /gemini/. Defaults: thinking medium, media resolution medium. */
  thinkingPreset?: RegistrationThinkingPreset;
  mediaResolution?: RegistrationMediaResolutionPreset;
}

export interface ImageGenRegistrationResult {
  /** Atlas reference warped to match the slice's shape, alpha-clipped to atlas root. */
  warpedAtlas: ImageBitmap;
  /** Atlas borders warped through the same Elastix transform, also alpha-clipped. */
  warpedBorders: ImageBitmap;
  /** Gemini's raw output (warped colored regions) — useful for the QA overlay. */
  generatedColoredRegions: ImageBitmap;
  /** Wall-clock total elapsed ms. */
  elapsedMs: number;
}

interface SizedRgba {
  rgba: Uint8ClampedArray;
  w: number;
  h: number;
}

interface ItkImage {
  imageType: {
    dimension: 2;
    componentType: "float32";
    pixelType: "Scalar";
    components: 1;
  };
  origin: [number, number];
  spacing: [number, number];
  direction: Float64Array;
  size: [number, number];
  data: Float32Array;
  metadata?: Map<string, unknown>;
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

function imgWidth(src: ImageBitmap | HTMLImageElement): number {
  return (src as ImageBitmap).width ?? (src as HTMLImageElement).naturalWidth;
}

function imgHeight(src: ImageBitmap | HTMLImageElement): number {
  return (src as ImageBitmap).height ?? (src as HTMLImageElement).naturalHeight;
}

/** Mirrors python `_target_size_for_slice`: cap long edge at 2048. */
function targetSizeFor(src: ImageBitmap | HTMLImageElement): [number, number] {
  const w = imgWidth(src);
  const h = imgHeight(src);
  if (w <= 0 || h <= 0) throw new Error("Slice has zero dimension.");
  const longEdge = Math.max(w, h);
  if (longEdge <= MAX_LONG_EDGE) return [w, h];
  const scale = MAX_LONG_EDGE / longEdge;
  return [Math.max(1, Math.round(w * scale)), Math.max(1, Math.round(h * scale))];
}

function drawToRgba(
  src: ImageBitmap | HTMLImageElement,
  w: number,
  h: number,
  smooth: boolean,
): SizedRgba {
  const { canvas, ctx } = makeCanvas(w, h);
  ctx.imageSmoothingEnabled = smooth;
  ctx.imageSmoothingQuality = smooth ? "high" : "low";
  ctx.drawImage(src as CanvasImageSource, 0, 0, w, h);
  const img = ctx.getImageData(0, 0, w, h);
  void canvas;
  return { rgba: img.data, w, h };
}

async function fetchBitmap(url: string): Promise<ImageBitmap> {
  const r = await fetch(url, { cache: "force-cache" });
  if (!r.ok) throw new Error(`Fetch ${url} failed: HTTP ${r.status}`);
  return await createImageBitmap(await r.blob());
}

/** Build an ImageData from a Uint8ClampedArray with strict ArrayBuffer typing.
 *  TS 6's ImageData typings reject Uint8ClampedArray<ArrayBufferLike> (the
 *  default ArrayBufferLike from ctx.getImageData), so copy into a fresh
 *  Uint8ClampedArray which is backed by a plain ArrayBuffer. */
function makeImageData(rgba: Uint8ClampedArray, w: number, h: number): ImageData {
  const copy = new Uint8ClampedArray(rgba.length);
  copy.set(rgba);
  return new ImageData(copy, w, h);
}

function rgbaToPng(sized: SizedRgba): Promise<Blob> {
  const { canvas, ctx } = makeCanvas(sized.w, sized.h);
  ctx.putImageData(makeImageData(sized.rgba, sized.w, sized.h), 0, 0);
  if (typeof OffscreenCanvas !== "undefined" && canvas instanceof OffscreenCanvas) {
    return canvas.convertToBlob({ type: "image/png" });
  }
  return new Promise<Blob>((res, rej) => {
    (canvas as HTMLCanvasElement).toBlob(
      (b) => (b ? res(b) : rej(new Error("toBlob failed."))),
      "image/png",
    );
  });
}

async function rgbaToBase64Png(sized: SizedRgba): Promise<string> {
  const blob = await rgbaToPng(sized);
  const buf = await blob.arrayBuffer();
  const bytes = new Uint8Array(buf);
  let bin = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    bin += String.fromCharCode.apply(
      null,
      Array.from(bytes.subarray(i, Math.min(i + CHUNK, bytes.length))),
    );
  }
  return btoa(bin);
}

function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function throwIfAborted(signal: AbortSignal | undefined, where: string): void {
  if (signal?.aborted) {
    const e = new Error(`Aborted before ${where}.`);
    e.name = "AbortError";
    throw e;
  }
}

/** RGB → luma, identical weights to `cv2.cvtColor(..., RGB2GRAY)`. */
function rgbaToFloat32Luma(sized: SizedRgba): Float32Array {
  const { rgba, w, h } = sized;
  const out = new Float32Array(w * h);
  for (let i = 0, j = 0; j < out.length; i += 4, j++) {
    out[j] = 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
  }
  return out;
}

/** Single channel (R/G/B) of an RGBA buffer as a Float32 plane. */
function rgbaChannelToFloat32(sized: SizedRgba, channel: 0 | 1 | 2): Float32Array {
  const { rgba, w, h } = sized;
  const out = new Float32Array(w * h);
  for (let i = channel, j = 0; j < out.length; i += 4, j++) {
    out[j] = rgba[i];
  }
  return out;
}

/** Wrap a Float32 plane in itk-wasm's Image shape (2-D scalar). */
function makeItkImage(data: Float32Array, w: number, h: number): ItkImage {
  return {
    imageType: {
      dimension: 2,
      componentType: "float32",
      pixelType: "Scalar",
      components: 1,
    },
    origin: [0, 0],
    spacing: [1, 1],
    direction: new Float64Array([1, 0, 0, 1]),
    size: [w, h],
    data,
    metadata: new Map(),
  };
}

/** Atlas root silhouette: alpha>0 of the section PNG → binary 0/255. */
function rgbaAlphaToMask(sized: SizedRgba): Uint8Array {
  const out = new Uint8Array(sized.w * sized.h);
  for (let i = 0, j = 3; i < out.length; i++, j += 4) {
    out[i] = sized.rgba[j] > 0 ? 255 : 0;
  }
  return out;
}

/** Mirrors `_build_elastix_parameter_object`: affine then bspline, with the
 *  exact overrides used in the python pipeline. */
async function buildParameterObject(): Promise<{
  webWorker: Worker;
  parameterObject: Record<string, string[]>[];
}> {
  const affineRes = await defaultParameterMap("affine", {
    numberOfResolutions: 4,
  });
  const affineMap: Record<string, string[]> = affineRes.parameterMap as Record<
    string,
    string[]
  >;
  affineMap["Metric"] = ["AdvancedNormalizedCorrelation"];
  affineMap["MaximumNumberOfIterations"] = ["512"];
  affineMap["NumberOfResolutions"] = ["4"];
  affineMap["AutomaticTransformInitialization"] = ["true"];
  affineMap["AutomaticTransformInitializationMethod"] = ["CenterOfGravity"];

  const bsplineRes = await defaultParameterMap("bspline", {
    webWorker: affineRes.webWorker,
    numberOfResolutions: 4,
    finalGridSpacing: 32,
  });
  const bsplineMap: Record<string, string[]> = bsplineRes.parameterMap as Record<
    string,
    string[]
  >;
  bsplineMap["FinalGridSpacingInPhysicalUnits"] = ["32"];
  bsplineMap["Metric"] = [
    "AdvancedNormalizedCorrelation",
    "TransformBendingEnergyPenalty",
  ];
  bsplineMap["Metric0Weight"] = ["1.0"];
  bsplineMap["Metric1Weight"] = ["3.0"];
  bsplineMap["MaximumNumberOfIterations"] = ["1024"];
  bsplineMap["NumberOfResolutions"] = ["4"];

  return {
    webWorker: bsplineRes.webWorker as Worker,
    parameterObject: [affineMap, bsplineMap],
  };
}

function thinkingBudgetFromPreset(preset: RegistrationThinkingPreset): number {
  switch (preset) {
    case "off":
      return 0;
    case "low":
      return 2048;
    case "medium":
      return 8192;
    case "high":
      return 24576;
  }
}
function mediaResolutionFromPreset(
  preset: RegistrationMediaResolutionPreset,
): "MEDIA_RESOLUTION_LOW" | "MEDIA_RESOLUTION_MEDIUM" | "MEDIA_RESOLUTION_HIGH" {
  switch (preset) {
    case "low":
      return "MEDIA_RESOLUTION_LOW";
    case "medium":
      return "MEDIA_RESOLUTION_MEDIUM";
    case "high":
      return "MEDIA_RESOLUTION_HIGH";
  }
}

async function callGeminiForWarpedRegions(
  apiKey: string,
  modelId: string,
  coloredB64: string,
  referenceB64: string,
  sliceB64: string,
  signal: AbortSignal | undefined,
  thinkingPreset: RegistrationThinkingPreset | undefined,
  mediaResolution: RegistrationMediaResolutionPreset | undefined,
): Promise<ImageBitmap> {
  const ai = new GoogleGenAI({ apiKey });
  throwIfAborted(signal, "Gemini request");

  const extras: Record<string, unknown> = {};
  if (/gemini/i.test(modelId)) {
    if (thinkingPreset) {
      extras.thinkingConfig = {
        thinkingBudget: thinkingBudgetFromPreset(thinkingPreset),
      };
    }
    if (mediaResolution) {
      extras.mediaResolution = mediaResolutionFromPreset(mediaResolution);
    }
  }

  let response: Awaited<ReturnType<typeof ai.models.generateContent>>;
  try {
    response = await ai.models.generateContent({
      model: modelId,
      contents: [
        { inlineData: { mimeType: "image/png", data: coloredB64 } },
        { inlineData: { mimeType: "image/png", data: referenceB64 } },
        { inlineData: { mimeType: "image/png", data: sliceB64 } },
        { text: SEGMENTATION_PROMPT },
      ],
      config: { responseModalities: ["TEXT", "IMAGE"], ...extras },
    });
  } catch (err) {
    const e = err as { status?: number; message?: string; name?: string };
    const status = e?.status != null ? ` (HTTP ${e.status})` : "";
    const name = e?.name ? `${e.name}: ` : "";
    throw new Error(
      `${name}Gemini image generation request failed${status}: ${e?.message ?? err}`,
    );
  }
  throwIfAborted(signal, "Gemini response handling");

  const block = (response as { promptFeedback?: { blockReason?: string } })
    ?.promptFeedback?.blockReason;
  if (block) {
    throw new Error(`Gemini blocked the image generation request: ${block}`);
  }

  const candidates =
    (response as { candidates?: Array<{ content?: { parts?: Array<unknown> } }> })
      .candidates ?? [];
  for (const cand of candidates) {
    const parts = cand?.content?.parts ?? [];
    for (const part of parts) {
      const inline = (part as { inlineData?: { data?: string; mimeType?: string } })
        .inlineData;
      if (inline?.data && inline.data.length > 0) {
        const bytes = base64ToBytes(inline.data);
        // Copy into a fresh Uint8Array<ArrayBuffer> — TS 6 strict typing rejects
        // Uint8Array<ArrayBufferLike> as a BlobPart.
        const copy = new Uint8Array(bytes.byteLength);
        copy.set(bytes);
        const blob = new Blob([copy.buffer], { type: inline.mimeType ?? "image/png" });
        return await createImageBitmap(blob);
      }
    }
  }
  throw new Error("Gemini image generation returned no image part.");
}

/** Run transformix on one Float32 plane; return uint8 channel of same WxH. */
async function warpChannel(
  webWorker: Worker,
  transformParameterObject: unknown,
  channel: Float32Array,
  w: number,
  h: number,
): Promise<Uint8ClampedArray> {
  const img = makeItkImage(channel, w, h);
  const out = await transformix(
    img as unknown as Parameters<typeof transformix>[0],
    transformParameterObject as Parameters<typeof transformix>[1],
    { webWorker },
  );
  // itk-wasm transformix returns { result: Image } in the browser package.
  const warped = (out as unknown as { result?: ItkImage; output?: ItkImage }).result
    ?? (out as unknown as { result?: ItkImage; output?: ItkImage }).output;
  if (!warped?.data) {
    throw new Error("transformix returned no image data.");
  }
  const src = warped.data as unknown as Float32Array;
  const dst = new Uint8ClampedArray(w * h);
  for (let i = 0; i < dst.length; i++) {
    let v = src[i];
    if (!Number.isFinite(v)) v = 0;
    if (v < 0) v = 0;
    else if (v > 255) v = 255;
    dst[i] = v;
  }
  return dst;
}

async function packWarpedRgba(
  r: Uint8ClampedArray,
  g: Uint8ClampedArray,
  b: Uint8ClampedArray,
  alpha: Uint8Array,
  w: number,
  h: number,
): Promise<ImageBitmap> {
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let i = 0, j = 0; i < r.length; i++, j += 4) {
    rgba[j] = r[i];
    rgba[j + 1] = g[i];
    rgba[j + 2] = b[i];
    rgba[j + 3] = alpha[i];
  }
  return await createImageBitmap(new ImageData(rgba, w, h));
}

export async function imageGenRegister(
  p: ImageGenRegistrationParams,
): Promise<ImageGenRegistrationResult> {
  if (!p.geminiApiKey || p.geminiApiKey.trim().length === 0) {
    throw new Error("Gemini API key is required for image-gen registration.");
  }
  if (!p.slice) throw new Error("Slice input is required.");
  throwIfAborted(p.signal, "registration start");

  const started = performance.now();
  const modelId = p.imageModelId ?? DEFAULT_MODEL;
  const [w, h] = targetSizeFor(p.slice);

  // --- Step 1: fetch atlas layers (snapped to nearest bundled section) ---
  const layers = await getCoronalSliceLayers(p.atlasApMm);
  const [atlasColoredBmp, atlasReferenceBmp, atlasBordersBmp] = await Promise.all([
    fetchBitmap(layers.coloredUrl),
    fetchBitmap(layers.sectionUrl),
    fetchBitmap(layers.bordersUrl),
  ]);
  if (atlasColoredBmp.width === 0 || atlasReferenceBmp.width === 0) {
    throw new Error("Atlas layer fetch produced zero-sized image.");
  }

  // --- Step 2: resize every input to (w, h) ---
  const coloredAtTarget = drawToRgba(atlasColoredBmp, w, h, true);
  const referenceAtTarget = drawToRgba(atlasReferenceBmp, w, h, true);
  const bordersAtTarget = drawToRgba(atlasBordersBmp, w, h, false);
  const sliceAtTarget = drawToRgba(p.slice, w, h, true);
  atlasColoredBmp.close?.();
  atlasReferenceBmp.close?.();
  atlasBordersBmp.close?.();

  // --- Step 3: Gemini call for warped colored regions ---
  p.onProgress?.("gemini", "Generating warped atlas via Nano Banana 2...");
  const [coloredB64, referenceB64, sliceB64] = await Promise.all([
    rgbaToBase64Png(coloredAtTarget),
    rgbaToBase64Png(referenceAtTarget),
    rgbaToBase64Png(sliceAtTarget),
  ]);
  const generatedColoredBmp = await callGeminiForWarpedRegions(
    p.geminiApiKey,
    modelId,
    coloredB64,
    referenceB64,
    sliceB64,
    p.signal,
    p.thinkingPreset,
    p.mediaResolution,
  );
  throwIfAborted(p.signal, "elastix start");

  // Resize the model output to (w, h) so all elastix planes line up.
  const modelOutputAtTarget = drawToRgba(generatedColoredBmp, w, h, true);

  // --- Step 4: Elastix register (atlas colored → Gemini output) ---
  p.onProgress?.("elastix", "Running affine + B-spline registration...");
  const { webWorker, parameterObject } = await buildParameterObject();
  const fixedGray = rgbaToFloat32Luma(modelOutputAtTarget);
  const movingGray = rgbaToFloat32Luma(coloredAtTarget);
  const fixedImage = makeItkImage(fixedGray, w, h);
  const movingImage = makeItkImage(movingGray, w, h);

  let elastixOut: Awaited<ReturnType<typeof elastix>>;
  try {
    elastixOut = await elastix(
      parameterObject as unknown as Parameters<typeof elastix>[0],
      {
        webWorker,
        fixed: fixedImage as unknown as NonNullable<Parameters<typeof elastix>[1]>["fixed"],
        moving: movingImage as unknown as NonNullable<Parameters<typeof elastix>[1]>["moving"],
      },
    );
  } catch (err) {
    throw new Error(
      `Elastix registration failed: ${(err as Error)?.message ?? err}`,
    );
  }
  const transformParameterObject = (elastixOut as unknown as {
    transformParameterObject: unknown;
  }).transformParameterObject;
  if (!transformParameterObject) {
    throw new Error("Elastix returned no transformParameterObject.");
  }

  // --- Step 5: per-channel warp via transformix ---
  p.onProgress?.("warp", "Warping atlas + borders through dense transform...");
  const atlasRoot = rgbaAlphaToMask(referenceAtTarget);
  const bordersAlphaMask = rgbaAlphaToMask(bordersAtTarget);

  const refR = rgbaChannelToFloat32(referenceAtTarget, 0);
  const refG = rgbaChannelToFloat32(referenceAtTarget, 1);
  const refB = rgbaChannelToFloat32(referenceAtTarget, 2);
  const bordR = rgbaChannelToFloat32(bordersAtTarget, 0);
  const bordG = rgbaChannelToFloat32(bordersAtTarget, 1);
  const bordB = rgbaChannelToFloat32(bordersAtTarget, 2);
  // Borders alpha as a Float32 plane so we get a warped alpha mask too.
  const bordA = new Float32Array(bordersAlphaMask.length);
  for (let i = 0; i < bordA.length; i++) bordA[i] = bordersAlphaMask[i];

  const [
    warpedRefR,
    warpedRefG,
    warpedRefB,
    warpedBordR,
    warpedBordG,
    warpedBordB,
    warpedBordA,
  ] = await Promise.all([
    warpChannel(webWorker, transformParameterObject, refR, w, h),
    warpChannel(webWorker, transformParameterObject, refG, w, h),
    warpChannel(webWorker, transformParameterObject, refB, w, h),
    warpChannel(webWorker, transformParameterObject, bordR, w, h),
    warpChannel(webWorker, transformParameterObject, bordG, w, h),
    warpChannel(webWorker, transformParameterObject, bordB, w, h),
    warpChannel(webWorker, transformParameterObject, bordA, w, h),
  ]);

  const warpedAtlasBmp = await packWarpedRgba(
    warpedRefR,
    warpedRefG,
    warpedRefB,
    atlasRoot,
    w,
    h,
  );
  // Borders alpha: use the warped alpha (so feathered edges stay subtle), but
  // gate by the atlas root so stray ink outside the brain silhouette gets
  // dropped to fully transparent.
  const warpedBordersAlpha = new Uint8Array(w * h);
  for (let i = 0; i < warpedBordersAlpha.length; i++) {
    warpedBordersAlpha[i] = atlasRoot[i] > 0 ? warpedBordA[i] : 0;
  }
  const warpedBordersBmp = await packWarpedRgba(
    warpedBordR,
    warpedBordG,
    warpedBordB,
    warpedBordersAlpha,
    w,
    h,
  );

  p.onProgress?.("done");
  return {
    warpedAtlas: warpedAtlasBmp,
    warpedBorders: warpedBordersBmp,
    generatedColoredRegions: generatedColoredBmp,
    elapsedMs: performance.now() - started,
  };
}

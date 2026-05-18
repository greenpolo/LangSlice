/** 3D slice-plane rendering. Browser variant consumes ImageBitmap objects
 *  produced by quickAffine + imageGenRegistration; no filesystem paths. */

import { useMemo, useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { useAppStore } from "../stores/appStore";
import type { AtlasInfo, Plane, SliceInfo } from "../lib/types";
import { getCoronalSliceLayers } from "../lib/browserCommands";
import { getCachedBitmap } from "../lib/bitmapCache";

const UM_TO_MM = 1 / 1000;

interface SlicePose {
  size: [number, number];
  rotation: [number, number, number];
  position: [number, number, number];
}

interface MeshExtents {
  ap: number;
  dv: number;
  ml: number;
}

function computeSlicePose(
  plane: Plane,
  apMm: number,
  atlasInfo: AtlasInfo,
  meshExtents: MeshExtents,
): SlicePose {
  const dvMm = (atlasInfo.shape[1] * atlasInfo.resolution[1]) / 1000;
  const mlMm = (atlasInfo.shape[2] * atlasInfo.resolution[2]) / 1000;
  const apTotalMm = atlasInfo.ap_max_mm - atlasInfo.ap_min_mm;
  const apFraction = (apMm - atlasInfo.ap_min_mm) / apTotalMm;

  switch (plane) {
    case "sagittal":
      return {
        size: [apTotalMm, dvMm],
        rotation: [0, 0, 0],
        position: [0, 0, 0],
      };
    case "horizontal":
      return {
        size: [apTotalMm, mlMm],
        rotation: [-Math.PI / 2, 0, 0],
        position: [0, 0, 0],
      };
    case "coronal":
    default: {
      const apPosition = -meshExtents.ap / 2 + apFraction * meshExtents.ap;
      return {
        size: [mlMm, dvMm],
        rotation: [0, Math.PI / 2, 0],
        position: [apPosition, 0, 0],
      };
    }
  }
}

/** Full registration > quick-affine preview. Returns the best bitmap or null. */
function pickWarpedBitmap(slice: SliceInfo): ImageBitmap | null {
  const full = slice.registrationResult?.warpedAtlas ?? null;
  if (full) return full;
  return slice.quickAffineWarpedBitmap ?? null;
}

/** Bake puts brainglobe `asr` row 0 = SUPERIOR at the top of the image and
 *  `BrainMesh.tsx:23` negates mesh Y so world +Y = SUPERIOR. With the coronal
 *  plane's rotation [0, π/2, 0], local +Y stays world +Y, so we need image
 *  row 0 (SUPERIOR) to land at texture V=1 (= local +Y).
 *
 *  Chrome silently ignores `UNPACK_FLIP_Y_WEBGL` for `ImageBitmap` sources
 *  (https://crbug.com/1080891), so setting `tex.flipY = true` on a
 *  `CanvasTexture(bmp)` is a no-op and image row 0 ends up at V=0 — upside
 *  down. We pre-flip the pixels into an `HTMLCanvasElement` instead, where
 *  the flip is part of the source data and there's no upload-time flag to
 *  ignore. (Tauri's `DataTexture` path uses raw bytes, which is why it's
 *  immune to this quirk.) */
function bitmapToTexture(bmp: ImageBitmap): THREE.Texture {
  const canvas = document.createElement("canvas");
  canvas.width = bmp.width;
  canvas.height = bmp.height;
  const ctx = canvas.getContext("2d")!;
  ctx.translate(0, canvas.height);
  ctx.scale(1, -1);
  ctx.drawImage(bmp, 0, 0);
  const tex = new THREE.CanvasTexture(canvas);
  tex.flipY = false;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = false;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

function WarpedSlicePlaneMesh({
  slice,
  plane,
  atlasInfo,
  meshExtents,
}: {
  slice: SliceInfo;
  plane: Plane;
  atlasInfo: AtlasInfo;
  meshExtents: MeshExtents;
}) {
  const bitmap = pickWarpedBitmap(slice);
  const [texture, setTexture] = useState<THREE.Texture | null>(null);

  useEffect(() => {
    if (bitmap === null) {
      setTexture(null);
      return;
    }
    const tex = bitmapToTexture(bitmap);
    setTexture(tex);
    return () => {
      tex.dispose();
    };
  }, [bitmap, plane]);

  if (texture === null || slice.apMm === undefined) return null;

  const { size, rotation, position } = computeSlicePose(
    plane,
    slice.apMm,
    atlasInfo,
    meshExtents,
  );

  return (
    <mesh position={position} rotation={rotation}>
      <planeGeometry args={size} />
      <meshBasicMaterial
        map={texture}
        side={THREE.DoubleSide}
        transparent
        depthWrite={false}
      />
    </mesh>
  );
}

/** Live preview at the current slider AP. Fetches the bundled borders PNG
 *  for the snapped section via the URL-keyed bitmap LRU and uploads it as a
 *  texture.
 *
 *  Perf contract: keep the previous texture visible until the new one
 *  arrives (no blank frames), and drop stale async results if the slider
 *  has moved on. Bitmaps are owned by the LRU, so we only dispose the
 *  Three.js texture wrapper on cleanup. */
function PreviewBorderPlane({
  apMm,
  plane,
  atlasInfo,
  meshExtents,
}: {
  apMm: number;
  plane: Plane;
  atlasInfo: AtlasInfo;
  meshExtents: MeshExtents;
}) {
  const [texture, setTexture] = useState<THREE.Texture | null>(null);
  // Track which apMm the currently-displayed texture corresponds to so the
  // mesh's plane position stays in sync with the texture (not the slider).
  const [textureApMm, setTextureApMm] = useState<number | null>(null);
  const apMmRef = useRef(apMm);
  apMmRef.current = apMm;

  useEffect(() => {
    let cancelled = false;
    let createdTex: THREE.Texture | null = null;
    (async () => {
      try {
        const layers = await getCoronalSliceLayers(apMm);
        // Stale-result guard: if the slider moved past this AP while we
        // were fetching, drop the result instead of swapping it in.
        if (cancelled || apMmRef.current !== apMm) return;
        const bmp = await getCachedBitmap(layers.bordersUrl);
        if (cancelled || apMmRef.current !== apMm) return;
        const tex = bitmapToTexture(bmp);
        createdTex = tex;
        // Replace the previous texture in-place. The previous tex is
        // disposed by its own effect-cleanup when its `apMm` dep changes,
        // so we don't need to dispose it here. We keep the prior tex
        // visible during the swap (no setTexture(null) blanking).
        setTexture(tex);
        setTextureApMm(apMm);
      } catch {
        // Best-effort: leave the existing texture in place on failure
        // rather than flashing to black.
      }
    })();
    return () => {
      cancelled = true;
      // Dispose ONLY the Three.js texture wrapper we created in this
      // effect; the underlying ImageBitmap is owned by the LRU and will
      // be closed on cache eviction.
      if (createdTex) createdTex.dispose();
    };
  }, [apMm, plane]);

  if (texture === null || textureApMm === null) return null;

  // Pose at the texture's snapped AP, not the live slider AP, so the cyan
  // wireframe doesn't visibly lag behind its true Z position when there's
  // any async delay.
  const { size, rotation, position } = computeSlicePose(
    plane,
    textureApMm,
    atlasInfo,
    meshExtents,
  );
  return (
    <mesh position={position} rotation={rotation}>
      <planeGeometry args={size} />
      <meshBasicMaterial
        map={texture}
        side={THREE.DoubleSide}
        transparent
        depthWrite={false}
      />
    </mesh>
  );
}

export function AtlasSlicePlane() {
  const atlasInfo = useAppStore((s) => s.atlasInfo);
  const currentApMm = useAppStore((s) => s.currentApMm);
  const brainMesh = useAppStore((s) => s.brainMesh);
  const selectedSliceIndex = useAppStore((s) => s.selectedSliceIndex);
  const selectedBrain = useAppStore(
    (s) => s.brains.find((b) => b.id === s.selectedBrainId) ?? null,
  );

  const selectedSlice =
    selectedBrain && selectedSliceIndex !== null
      ? selectedBrain.slices[selectedSliceIndex] ?? null
      : null;
  const plane = selectedBrain?.plane ?? "coronal";

  const meshExtents = useMemo<MeshExtents | null>(() => {
    if (!brainMesh || brainMesh.positions.length === 0) return null;
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;
    for (let i = 0; i < brainMesh.positions.length; i += 3) {
      const x = brainMesh.positions[i] * UM_TO_MM;
      const y = -brainMesh.positions[i + 1] * UM_TO_MM;
      const z = brainMesh.positions[i + 2] * UM_TO_MM;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (z < minZ) minZ = z;
      if (z > maxZ) maxZ = z;
    }
    return { ap: maxX - minX, dv: maxY - minY, ml: maxZ - minZ };
  }, [brainMesh]);

  if (!atlasInfo || !meshExtents) return null;

  const visibleSlices: { index: number; slice: SliceInfo }[] =
    selectedBrain?.slices
      .map((slice, index) => ({ index, slice }))
      .filter(({ slice }) => {
        if (!slice.apLocked) return false;
        if (slice.visibleIn3D === false) return false;
        if (slice.quickAffineRunning) return false;
        return pickWarpedBitmap(slice) !== null;
      }) ?? [];

  const selectedHasVisiblePlane =
    selectedSlice !== null &&
    visibleSlices.some(({ slice }) => slice === selectedSlice);
  const showPreviewPlane = !selectedHasVisiblePlane;

  return (
    <>
      {showPreviewPlane && (
        <PreviewBorderPlane
          apMm={currentApMm}
          plane={plane}
          atlasInfo={atlasInfo}
          meshExtents={meshExtents}
        />
      )}
      {visibleSlices.map(({ index, slice }) => (
        <WarpedSlicePlaneMesh
          key={index}
          slice={slice}
          plane={plane}
          atlasInfo={atlasInfo}
          meshExtents={meshExtents}
        />
      ))}
    </>
  );
}

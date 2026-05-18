import { useMemo, useEffect, useState } from "react";
import * as THREE from "three";
import { convertFileSrc } from "@tauri-apps/api/core";
import { useAppStore } from "../stores/appStore";
import type { AtlasInfo, Plane, SliceInfo } from "../lib/types";

const UM_TO_MM = 1 / 1000;

/**
 * 3D slice-plane rendering.
 *
 * Two concerns, two components:
 *
 *   - `PreviewBorderPlane` renders the teal atlas-borders DataTexture at the
 *     current AP slider position. Shown when nothing better is available
 *     for the selected slice (no warp, or slice is hidden/unlocked). Acts
 *     as live feedback while the user drags the slider before locking.
 *
 *   - `WarpedSlicePlaneMesh` renders one warped histology slice as a sheet
 *     positioned at *that slice's* AP. Multiple of these are rendered
 *     simultaneously — every locked slice with `visibleIn3D !== false` and
 *     a usable warp path produces one. Lets users see the 3D registration
 *     fan out across all their slices.
 *
 * Plane orientation conventions (verified against `BrainMesh.tsx`,
 * `Scene3D.tsx`, and `AGENTS.md`):
 *   - World X = AP, World Y = DV (up), World Z = ML.
 *   - Three.js `<planeGeometry args={[w, h]}>` builds in the XY plane with
 *     normal along +Z. We rotate to point the normal along the desired
 *     world axis.
 */

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

/** Compute a slice plane's size + rotation + AP-axis position given the
 * brain mesh extents and atlas info. Coronal is the verified path;
 * sagittal/horizontal are best-effort (ML/DV-varying positions aren't
 * plumbed through per-slice yet). */
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

/** Pick the best warped-slice path for a slice. Full-registration's inverse
 * warp wins over the quick-affine preview; null if neither is usable
 * (legacy run with failed inverse, or no preview run yet). */
function pickWarpedPath(slice: SliceInfo): string | null {
  const inverseStatus = slice.registrationResult?.inverseWarpStatus ?? null;
  const fullPath = slice.registrationResult?.sliceWarpedToAtlasPath ?? null;
  const fullUsable =
    fullPath !== null &&
    fullPath !== "" &&
    !(typeof inverseStatus === "string" && inverseStatus.startsWith("failed"));
  if (fullUsable) return fullPath;
  const quickPath = slice.quickAffineWarpedPath ?? null;
  if (quickPath !== null && quickPath !== "") return quickPath;
  return null;
}

/** Single mesh that loads a slice's warped PNG and renders it at the slice's
 * own AP position. Texture is loaded on mount, disposed on unmount or path
 * change so we don't leak GPU memory when slices toggle. */
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
  const warpedPath = pickWarpedPath(slice);
  const [texture, setTexture] = useState<THREE.Texture | null>(null);

  useEffect(() => {
    if (warpedPath === null) {
      setTexture(null);
      return;
    }
    const loader = new THREE.TextureLoader();
    let cancelled = false;
    let loaded: THREE.Texture | null = null;

    loader.load(
      convertFileSrc(warpedPath),
      (tex) => {
        if (cancelled) {
          tex.dispose();
          return;
        }
        tex.flipY = true;
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.needsUpdate = true;
        loaded = tex;
        setTexture(tex);
      },
      undefined,
      (err) => {
        console.error("Failed to load warped-slice texture:", err);
        if (!cancelled) setTexture(null);
      },
    );

    return () => {
      cancelled = true;
      if (loaded) loaded.dispose();
      setTexture(null);
    };
  }, [warpedPath]);

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

/** Border-texture preview plane at the current slider AP. Shown only when
 * nothing better is available for the selected slice (no warp, or the
 * selected slice has been hidden in 3D, or no slice is selected). */
function PreviewBorderPlane({
  texture,
  apMm,
  plane,
  atlasInfo,
  meshExtents,
}: {
  texture: THREE.Texture;
  apMm: number;
  plane: Plane;
  atlasInfo: AtlasInfo;
  meshExtents: MeshExtents;
}) {
  const { size, rotation, position } = computeSlicePose(
    plane,
    apMm,
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
  const borderPixels = useAppStore((s) => s.currentBorderPixels);
  const borderWidth = useAppStore((s) => s.borderWidth);
  const borderHeight = useAppStore((s) => s.borderHeight);
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

  // Cheap DataTexture made directly from raw border pixels — no decode
  // step, no GPU upload of a PNG. Used by the preview plane.
  const borderTexture = useMemo(() => {
    if (!borderPixels || borderWidth === 0) return null;
    const rgba = new Uint8Array(borderWidth * borderHeight * 4);
    for (let i = 0; i < borderPixels.length; i++) {
      if (borderPixels[i] > 128) {
        const j = i * 4;
        rgba[j] = 45;
        rgba[j + 1] = 212;
        rgba[j + 2] = 191;
        rgba[j + 3] = 200;
      }
    }
    const tex = new THREE.DataTexture(rgba, borderWidth, borderHeight, THREE.RGBAFormat);
    tex.flipY = true;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.generateMipmaps = false;
    tex.anisotropy = 4;
    tex.needsUpdate = true;
    return tex;
  }, [borderPixels, borderWidth, borderHeight]);

  // Mesh extents drive AP-axis positioning so planes sit inside the brain
  // bounding box rather than at world origin.
  const meshExtents = useMemo<MeshExtents | null>(() => {
    if (!brainMesh || brainMesh.positions.length === 0) return null;
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;
    for (let i = 0; i < brainMesh.positions.length; i += 3) {
      const x = brainMesh.positions[i] * UM_TO_MM;
      // BrainMesh.tsx flips Y for display; mirror that here.
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

  // Visibility predicate for warped-slice meshes. The selected slice is
  // also drawn through this loop when it qualifies — only the unselected
  // hidden/unlocked case falls back to the preview plane.
  const visibleSlices: { index: number; slice: SliceInfo }[] =
    selectedBrain?.slices
      .map((slice, index) => ({ index, slice }))
      .filter(({ slice }) => {
        if (!slice.apLocked) return false;
        if (slice.visibleIn3D === false) return false;
        // Hide the plane during an in-flight re-run so we don't render a
        // stale path at the wrong AP for a couple seconds.
        if (slice.quickAffineRunning) return false;
        return pickWarpedPath(slice) !== null;
      }) ?? [];

  // Show the live slider-position preview only when the SELECTED slice
  // doesn't already have a warped plane in the volume. That keeps the
  // border texture out of the way once the user has locked something in.
  const selectedHasVisiblePlane =
    selectedSlice !== null &&
    visibleSlices.some(({ slice }) => slice === selectedSlice);
  const showPreviewPlane =
    !selectedHasVisiblePlane && borderTexture !== null;

  return (
    <>
      {showPreviewPlane && borderTexture && (
        <PreviewBorderPlane
          texture={borderTexture}
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

import { useMemo, useRef, useEffect } from "react";
import * as THREE from "three";
import { useAppStore } from "../stores/appStore";

const UM_TO_MM = 1 / 1000;

export function AtlasSlicePlane() {
  const borderPixels = useAppStore((s) => s.currentBorderPixels);
  const borderWidth = useAppStore((s) => s.borderWidth);
  const borderHeight = useAppStore((s) => s.borderHeight);
  const atlasInfo = useAppStore((s) => s.atlasInfo);
  const currentApMm = useAppStore((s) => s.currentApMm);
  const brainMesh = useAppStore((s) => s.brainMesh);
  const materialRef = useRef<THREE.MeshBasicMaterial>(null);

  // Build DataTexture from raw border pixels — pure JS, no decode step
  const texture = useMemo(() => {
    if (!borderPixels || borderWidth === 0) return null;

    const rgba = new Uint8Array(borderWidth * borderHeight * 4);
    for (let i = 0; i < borderPixels.length; i++) {
      if (borderPixels[i] > 128) {
        const j = i * 4;
        rgba[j] = 45;       // R (teal)
        rgba[j + 1] = 212;  // G
        rgba[j + 2] = 191;  // B
        rgba[j + 3] = 200;  // A
      }
    }

    const tex = new THREE.DataTexture(rgba, borderWidth, borderHeight, THREE.RGBAFormat);
    tex.flipY = true;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.NearestFilter;
    tex.needsUpdate = true;
    return tex;
  }, [borderPixels, borderWidth, borderHeight]);

  useEffect(() => {
    if (materialRef.current && texture) {
      materialRef.current.map = texture;
      materialRef.current.needsUpdate = true;
    }
  }, [texture]);

  // Compute mesh AP extent for positioning
  const meshApExtent = useMemo(() => {
    if (!brainMesh || brainMesh.positions.length === 0) return null;
    let minX = Infinity, maxX = -Infinity;
    for (let i = 0; i < brainMesh.positions.length; i += 3) {
      const x = brainMesh.positions[i] * UM_TO_MM;
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
    }
    return maxX - minX;
  }, [brainMesh]);

  if (!texture || !atlasInfo || !meshApExtent) return null;

  const dvMm = (atlasInfo.shape[1] * atlasInfo.resolution[1]) / 1000;
  const mlMm = (atlasInfo.shape[2] * atlasInfo.resolution[2]) / 1000;
  const apTotalMm = atlasInfo.ap_max_mm - atlasInfo.ap_min_mm;
  const apFraction = (currentApMm - atlasInfo.ap_min_mm) / apTotalMm;
  const apPosition = -meshApExtent / 2 + apFraction * meshApExtent;

  return (
    <mesh position={[apPosition, 0, 0]} rotation={[0, Math.PI / 2, 0]}>
      <planeGeometry args={[mlMm, dvMm]} />
      <meshBasicMaterial
        ref={materialRef}
        map={texture}
        side={THREE.DoubleSide}
        transparent
        depthWrite={false}
      />
    </mesh>
  );
}

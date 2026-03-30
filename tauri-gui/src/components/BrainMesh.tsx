import { useMemo } from "react";
import * as THREE from "three";
import { useAppStore } from "../stores/appStore";

/**
 * BrainGlobe OBJ meshes are in micrometers.
 * We scale to mm for a reasonable Three.js coordinate space.
 */
const UM_TO_MM = 1 / 1000;

function useBrainGeometry() {
  const brainMesh = useAppStore((s) => s.brainMesh);

  return useMemo(() => {
    if (!brainMesh || brainMesh.positions.length === 0) return null;

    const geo = new THREE.BufferGeometry();

    // Scale from um to mm, flip Y axis (BrainGlobe DV is inverted)
    const positions = new Float32Array(brainMesh.positions.length);
    for (let i = 0; i < brainMesh.positions.length; i += 3) {
      positions[i] = brainMesh.positions[i] * UM_TO_MM;
      positions[i + 1] = -brainMesh.positions[i + 1] * UM_TO_MM;
      positions[i + 2] = brainMesh.positions[i + 2] * UM_TO_MM;
    }
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));

    // Indices
    const indices = new Uint32Array(brainMesh.indices);
    geo.setIndex(new THREE.BufferAttribute(indices, 1));

    // Compute normals for proper lighting
    geo.computeVertexNormals();

    // Center at origin
    geo.computeBoundingBox();
    geo.center();

    return geo;
  }, [brainMesh]);
}

/** Semi-transparent brain surface — brainrender style */
export function BrainMesh() {
  const geometry = useBrainGeometry();
  if (!geometry) return null;

  return (
    <mesh geometry={geometry}>
      <meshPhongMaterial
        color="#c8c8d0"
        transparent
        opacity={0.12}
        side={THREE.DoubleSide}
        depthWrite={false}
        shininess={30}
      />
    </mesh>
  );
}

/** Edge wireframe overlay for the cartoon/scientific look */
export function BrainMeshWireframe() {
  const geometry = useBrainGeometry();
  if (!geometry) return null;

  return (
    <lineSegments>
      <edgesGeometry args={[geometry, 15]} />
      <lineBasicMaterial color="#2dd4bf" transparent opacity={0.06} />
    </lineSegments>
  );
}

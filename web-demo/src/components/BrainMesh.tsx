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

/**
 * Edge wireframe overlay for the cartoon/scientific look.
 *
 * `edgesGeometry` finds dihedral edges between adjacent triangles whose
 * normals differ by more than the threshold angle. Edge count scales with
 * triangulation density even on smooth surfaces: the Allen 25um root mesh
 * (~49k vertices) produces a sparse cartoon outline, but the dense ADMBA
 * developmental atlases (300k+ vertices) cover the brain in a fuzzy mesh
 * of micro-edges that looks noisy rather than stylized.
 *
 * MeshView (BrainGlobe's reference WebGL viewer) renders these meshes with
 * no wireframe at all — just flat colored fills. We split the difference:
 * keep the wireframe for sparse meshes, drop it for dense ones, and bump
 * the threshold angle so even the kept cases render a tighter silhouette.
 */
const WIREFRAME_VERTEX_LIMIT = 100_000;
const WIREFRAME_ANGLE_THRESHOLD_DEG = 25;

export function BrainMeshWireframe() {
  const brainMesh = useAppStore((s) => s.brainMesh);
  const geometry = useBrainGeometry();
  if (!geometry || !brainMesh) return null;

  // Dense meshes look noisy with edgesGeometry. Fall back to a clean fill.
  if (brainMesh.positions.length / 3 > WIREFRAME_VERTEX_LIMIT) return null;

  return (
    <lineSegments>
      <edgesGeometry args={[geometry, WIREFRAME_ANGLE_THRESHOLD_DEG]} />
      <lineBasicMaterial color="#2dd4bf" transparent opacity={0.06} />
    </lineSegments>
  );
}

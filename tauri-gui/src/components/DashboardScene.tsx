import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useAppStore } from "../stores/appStore";

const UM_TO_MM = 1 / 1000;

let cachedGeo: THREE.BufferGeometry | null = null;
let cachedId: number | null = null;

function useSharedGeo() {
  const brainMesh = useAppStore((s) => s.brainMesh);
  return useMemo(() => {
    if (!brainMesh || brainMesh.positions.length === 0) return null;
    const id = brainMesh.positions.length;
    if (cachedGeo && cachedId === id) return cachedGeo;

    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(brainMesh.positions.length);
    for (let i = 0; i < brainMesh.positions.length; i += 3) {
      pos[i] = brainMesh.positions[i] * UM_TO_MM;
      pos[i + 1] = -brainMesh.positions[i + 1] * UM_TO_MM;
      pos[i + 2] = brainMesh.positions[i + 2] * UM_TO_MM;
    }
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setIndex(new THREE.BufferAttribute(new Uint32Array(brainMesh.indices), 1));
    geo.computeVertexNormals();
    geo.computeBoundingBox();
    geo.center();
    cachedGeo = geo;
    cachedId = id;
    return geo;
  }, [brainMesh]);
}

function RotatingBrain() {
  const geo = useSharedGeo();
  const groupRef = useRef<THREE.Group>(null);
  const hoveredBrainId = useAppStore((s) => s.hoveredBrainId);
  const timeRef = useRef(0);

  useFrame((_, delta) => {
    if (!groupRef.current) return;
    if (!hoveredBrainId) {
      timeRef.current += delta;
    }
    const baseAngle = 0.7;
    const swing = 0.26;
    const speed = 0.3;
    groupRef.current.rotation.y = baseAngle + Math.sin(timeRef.current * speed) * swing;
  });

  if (!geo) return null;

  return (
    <group ref={groupRef}>
      <mesh geometry={geo}>
        <meshPhongMaterial
          color="#8890a0"
          transparent
          opacity={hoveredBrainId ? 0.08 : 0.04}
          side={THREE.DoubleSide}
          depthWrite={false}
          shininess={10}
        />
      </mesh>

      <lineSegments>
        <edgesGeometry args={[geo, 15]} />
        <lineBasicMaterial
          color="#2dd4bf"
          transparent
          opacity={hoveredBrainId ? 0.05 : 0.02}
        />
      </lineSegments>

    </group>
  );
}

function SceneContent() {
  return (
    <>
      <ambientLight intensity={0.3} />
      <directionalLight position={[8, 5, 6]} intensity={0.4} />
      <RotatingBrain />
    </>
  );
}

export function DashboardScene() {
  const brainMesh = useAppStore((s) => s.brainMesh);
  if (!brainMesh) return null;

  return (
    <div className="dashboard-scene">
      <Canvas
        camera={{ position: [0, 2, 16], fov: 35, near: 0.1, far: 200 }}
        gl={{ antialias: true, alpha: true }}
        style={{ background: "transparent" }}
      >
        <SceneContent />
      </Canvas>
    </div>
  );
}

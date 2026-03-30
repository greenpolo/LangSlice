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

function HoveredSliceLines() {
  const hoveredBrainId = useAppStore((s) => s.hoveredBrainId);
  const brains = useAppStore((s) => s.brains);
  const atlasInfo = useAppStore((s) => s.atlasInfo);
  const geo = useSharedGeo();

  const lines = useMemo(() => {
    if (!hoveredBrainId || !atlasInfo || !geo) return [];
    const brain = brains.find((b) => b.id === hoveredBrainId);
    if (!brain || brain.slices.length === 0) return [];

    const bb = geo.boundingBox!;
    const apExtent = bb.max.x - bb.min.x;
    const halfAp = apExtent / 2;
    const apRange = atlasInfo.ap_max_mm - atlasInfo.ap_min_mm;

    return brain.slices.map((slice, i) => {
      let fraction: number;
      if (slice.apMm !== undefined) {
        fraction = (slice.apMm - atlasInfo.ap_min_mm) / apRange;
      } else {
        fraction = brain.slices.length > 1 ? i / (brain.slices.length - 1) : 0.5;
      }
      const x = -halfAp + fraction * apExtent;
      const color =
        slice.status === "done" ? "#2dd4bf" :
        slice.status === "running" ? "#f97316" : "#4b5563";
      const opacity =
        slice.status === "done" ? 0.5 :
        slice.status === "running" ? 0.6 : 0.2;
      return { x, color, opacity };
    });
  }, [hoveredBrainId, brains, atlasInfo, geo]);

  if (lines.length === 0 || !geo) return null;

  const bb = geo.boundingBox!;
  const height = (bb.max.y - bb.min.y) * 0.8;

  return (
    <>
      {lines.map((line, i) => (
        <mesh key={i} position={[line.x, 0, 0]} rotation={[0, Math.PI / 2, 0]}>
          <planeGeometry args={[0.08, height]} />
          <meshBasicMaterial
            color={line.color}
            transparent
            opacity={line.opacity}
            side={THREE.DoubleSide}
            depthWrite={false}
          />
        </mesh>
      ))}
    </>
  );
}

function RotatingBrain() {
  const geo = useSharedGeo();
  const groupRef = useRef<THREE.Group>(null);
  const hoveredBrainId = useAppStore((s) => s.hoveredBrainId);

  const timeRef = useRef(0);

  useFrame((_, delta) => {
    if (!groupRef.current) return;
    // Gentle rocking around a three-quarter view — AP axis always visible
    if (!hoveredBrainId) {
      timeRef.current += delta;
    }
    // Base angle: ~40 degrees (three-quarter view), oscillate +/- 15 degrees
    const baseAngle = 0.7; // ~40 deg
    const swing = 0.26;    // ~15 deg
    const speed = 0.3;
    groupRef.current.rotation.y = baseAngle + Math.sin(timeRef.current * speed) * swing;
  });

  if (!geo) return null;

  return (
    <group ref={groupRef}>
      {/* Ghost mesh */}
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

      {/* Wireframe */}
      <lineSegments>
        <edgesGeometry args={[geo, 15]} />
        <lineBasicMaterial
          color="#2dd4bf"
          transparent
          opacity={hoveredBrainId ? 0.05 : 0.02}
        />
      </lineSegments>

      {/* Slice lines for hovered brain */}
      <HoveredSliceLines />
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

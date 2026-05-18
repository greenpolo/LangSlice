import { Canvas } from "@react-three/fiber";
import { OrbitControls, GizmoHelper, GizmoViewport } from "@react-three/drei";
import { Suspense } from "react";
import { BrainMesh, BrainMeshWireframe } from "./BrainMesh";
import { AtlasSlicePlane } from "./AtlasSlicePlane";
import { useAppStore } from "../stores/appStore";

function SceneContent() {
  return (
    <>
      {/* Lighting — soft ambient + directional for depth */}
      <ambientLight intensity={0.5} />
      <directionalLight position={[8, 6, 10]} intensity={0.7} color="#e8eaed" />
      <directionalLight position={[-6, -4, -8]} intensity={0.2} color="#94a3b8" />
      <hemisphereLight
        color="#e0f2fe"
        groundColor="#1e293b"
        intensity={0.3}
      />

      {/* Brain visualization */}
      <BrainMesh />
      <BrainMeshWireframe />
      <AtlasSlicePlane />

      {/* Orbit controls — three-quarters default view */}
      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.08}
        rotateSpeed={0.5}
        zoomSpeed={0.8}
        minDistance={3}
        maxDistance={50}
      />

      {/* Navigation gizmo */}
      <GizmoHelper alignment="bottom-right" margin={[60, 60]}>
        <GizmoViewport
          axisColors={["#ef4444", "#22c55e", "#3b82f6"]}
          labelColor="#fff"
        />
      </GizmoHelper>
    </>
  );
}

export function Scene3D() {
  const atlasInfo = useAppStore((s) => s.atlasInfo);
  const selectedSliceIndex = useAppStore((s) => s.selectedSliceIndex);
  const selectedBrain = useAppStore(
    (s) => s.brains.find((b) => b.id === s.selectedBrainId) ?? null,
  );
  const selectedSlice =
    selectedBrain && selectedSliceIndex !== null
      ? selectedBrain.slices[selectedSliceIndex] ?? null
      : null;
  // Hide the hint once the slice is visible in the volume — that's true
  // whether it came from a full registration or just the quick-affine
  // preview triggered by Lock Position.
  const showPreRegNote =
    selectedSlice !== null &&
    !selectedSlice.registrationResult &&
    !selectedSlice.quickAffineWarpedBitmap;

  if (!atlasInfo) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">&#9671;</div>
        <div className="empty-state-title">No atlas loaded</div>
        <div className="empty-state-desc">
          Select an atlas from the panel and click Load Atlas to visualize
        </div>
      </div>
    );
  }

  return (
    <div className="canvas-container">
      <Canvas
        camera={{
          position: [-12, 8, 10],
          fov: 40,
          near: 0.1,
          far: 500,
        }}
        gl={{ antialias: true, alpha: false }}
        style={{ background: "#06080c" }}
      >
        <Suspense fallback={null}>
          <SceneContent />
        </Suspense>
      </Canvas>
      {showPreRegNote && (
        <div className="scene3d-prereg-note">
          Lock an AP position manually or run Estimate to see the slice appear in the brain volume.
        </div>
      )}
    </div>
  );
}

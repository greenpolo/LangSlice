/** Dashboard grid. Browser variant: one bundled atlas (allen_mouse_25um),
 *  Load Demo Brain + AddBrainCard for arbitrary user uploads, and a
 *  non-blocking note pointing at Local Models when the sidecar isn't live. */

import { useAppStore } from "../stores/appStore";
import { BrainCard, AddBrainCard } from "./BrainCard";
import { DashboardScene } from "./DashboardScene";
import { useSidecarStatus } from "../lib/sidecarProbe";

const ATLAS_NAME = "allen_mouse_25um";

export function Dashboard() {
  const brains = useAppStore((s) => s.brains);
  const atlasInfo = useAppStore((s) => s.atlasInfo);
  const atlasLoading = useAppStore((s) => s.atlasLoading);
  const loadAtlas = useAppStore((s) => s.loadAtlas);
  const loadDemoBrain = useAppStore((s) => s.loadDemoBrain);
  const openLocalModels = useAppStore((s) => s.openLocalModels);
  const sidecar = useSidecarStatus();

  return (
    <div className="dashboard">
      {/* Background 3D scene */}
      <DashboardScene />

      {/* Foreground content */}
      <div className="dashboard-content">
        {/* Atlas info bar — only one atlas in the demo bundle */}
        <div className="dashboard-atlas-bar">
          <div className="dashboard-atlas-left">
            <span className="section-label" style={{ margin: 0 }}>Atlas</span>
            <span
              style={{
                fontFamily: "var(--font-mono, monospace)",
                padding: "0 8px",
              }}
            >
              {ATLAS_NAME}
            </span>
            <button
              className="btn-primary"
              style={{ width: "auto", padding: "8px 20px" }}
              onClick={() => loadAtlas(ATLAS_NAME)}
              disabled={atlasLoading}
            >
              {atlasLoading ? "Loading..." : "Reload"}
            </button>
            <button
              className="btn-primary"
              style={{ width: "auto", padding: "8px 20px" }}
              onClick={() => void loadDemoBrain()}
              title="Walsh Lab demo brain (coronal, ABBA-registered)"
            >
              Load Demo Brain
            </button>
          </div>
          {atlasInfo && (
            <div className="dashboard-atlas-info">
              <span>{atlasInfo.shape.join(" x ")} voxels</span>
              <span className="dashboard-atlas-sep">/</span>
              <span>{atlasInfo.ap_count} slices</span>
              <span className="dashboard-atlas-sep">/</span>
              <span>{atlasInfo.resolution.map((r) => r.toFixed(0)).join("x")} um</span>
            </div>
          )}
        </div>

        {/* Sidecar note — only shown when sidecar isn't healthy. */}
        {sidecar.state !== "ok" && (
          <div
            className="dashboard-sidecar-note"
            style={{
              margin: "12px 0",
              padding: 12,
              borderRadius: 6,
              border: "1px solid #444",
              background: "#1a1a1a",
              color: "#ccc",
              display: "flex",
              alignItems: "center",
              gap: 8,
              flexWrap: "wrap",
            }}
          >
            <strong>Position estimation requires a local LiteRT-LM sidecar.</strong>
            <span>
              Quick-affine preview and image-gen registration work without it
              &mdash; but the agent loop needs Gemma 4 running locally on{" "}
              <code>http://127.0.0.1:8765</code>.
            </span>
            <button
              type="button"
              className="btn-secondary"
              style={{ marginLeft: "auto", width: "auto", padding: "4px 10px" }}
              onClick={openLocalModels}
            >
              Open Local Models &rarr;
            </button>
          </div>
        )}

        {/* Header */}
        <div className="dashboard-header">
          <h2 className="dashboard-title">Brains</h2>
          <span className="dashboard-count">{brains.length} loaded</span>
        </div>

        {/* Grid */}
        <div className="dashboard-grid">
          {brains.map((brain) => (
            <BrainCard key={brain.id} brain={brain} />
          ))}
          <AddBrainCard />
        </div>
      </div>
    </div>
  );
}

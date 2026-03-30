import { useEffect } from "react";
import { useAppStore } from "../stores/appStore";
import { BrainCard, AddBrainCard } from "./BrainCard";
import { DashboardScene } from "./DashboardScene";

export function Dashboard() {
  const brains = useAppStore((s) => s.brains);
  const availableAtlases = useAppStore((s) => s.availableAtlases);
  const atlasName = useAppStore((s) => s.atlasName);
  const atlasInfo = useAppStore((s) => s.atlasInfo);
  const atlasLoading = useAppStore((s) => s.atlasLoading);
  const fetchAtlasList = useAppStore((s) => s.fetchAtlasList);
  const loadAtlas = useAppStore((s) => s.loadAtlas);

  useEffect(() => {
    fetchAtlasList();
  }, [fetchAtlasList]);

  return (
    <div className="dashboard">
      {/* Background 3D scene */}
      <DashboardScene />

      {/* Foreground content */}
      <div className="dashboard-content">
        {/* Atlas selector bar */}
        <div className="dashboard-atlas-bar">
          <div className="dashboard-atlas-left">
            <span className="section-label" style={{ margin: 0 }}>Atlas</span>
            <select
              className="control-select"
              style={{ width: 220 }}
              value={atlasName}
              onChange={(e) => loadAtlas(e.target.value)}
              disabled={atlasLoading}
            >
              {availableAtlases.map((name) => (
                <option key={name} value={name}>{name}</option>
              ))}
              {availableAtlases.length === 0 && (
                <option value={atlasName}>{atlasName}</option>
              )}
            </select>
            <button
              className="btn-primary"
              style={{ width: "auto", padding: "8px 20px" }}
              onClick={() => loadAtlas(atlasName)}
              disabled={atlasLoading}
            >
              {atlasLoading ? "Loading..." : "Load"}
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

import { useEffect } from "react";
import { useAppStore } from "../stores/appStore";
import { BrainCard, AddBrainCard } from "./BrainCard";
import { DashboardScene } from "./DashboardScene";
import { AtlasManagerModal } from "./AtlasManagerModal";

const DEMO_BRAIN_PATH =
  "C:\\LabSoftware\\LangSlice\\tauri-gui\\demo\\demo_brain";
// DeMBA has per-region rgb_triplets (835/836 structures), which the image-gen
// registration pipeline needs for colored-region warping. The older AMBA
// developmental atlas (admba_3d_p14_mouse) lacks those colors so registration
// won't work against it.
const DEMO_ATLAS_NAME = "demba_allen_seg_dev_mouse_p14_25um";

export function Dashboard() {
  const brains = useAppStore((s) => s.brains);
  const availableAtlases = useAppStore((s) => s.availableAtlases);
  const atlasName = useAppStore((s) => s.atlasName);
  const atlasInfo = useAppStore((s) => s.atlasInfo);
  const atlasLoading = useAppStore((s) => s.atlasLoading);
  const fetchAtlasList = useAppStore((s) => s.fetchAtlasList);
  const loadAtlas = useAppStore((s) => s.loadAtlas);
  const addBrainFromFolder = useAppStore((s) => s.addBrainFromFolder);
  const openAtlasManager = useAppStore((s) => s.openAtlasManager);

  useEffect(() => {
    fetchAtlasList();
  }, [fetchAtlasList]);

  const handleLoadDemo = async () => {
    // If the demo atlas isn't downloaded, route the user through the
    // atlas manager (focus = demo atlas) instead of failing the load.
    // Reviewers cloning the repo won't have it on disk.
    if (!availableAtlases.includes(DEMO_ATLAS_NAME)) {
      openAtlasManager(DEMO_ATLAS_NAME);
      return;
    }
    const atlasTask =
      atlasName === DEMO_ATLAS_NAME && !atlasLoading
        ? Promise.resolve()
        : loadAtlas(DEMO_ATLAS_NAME);
    await Promise.all([atlasTask, addBrainFromFolder(DEMO_BRAIN_PATH)]);
  };

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
            <button
              className="btn-primary"
              style={{ width: "auto", padding: "8px 20px" }}
              onClick={handleLoadDemo}
              disabled={atlasLoading}
              title="Demo brain (coronal, 10 sections spanning AP 3.45-6.91 mm) + demba_allen_seg_dev_mouse_p14_25um atlas"
            >
              Load Demo
            </button>
            <button
              className="btn-secondary"
              style={{ width: "auto", padding: "8px 14px" }}
              onClick={() => openAtlasManager(null)}
              title="Browse and download BrainGlobe atlases"
            >
              Manage atlases…
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

      <AtlasManagerModal />
    </div>
  );
}

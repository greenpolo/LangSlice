import type { Brain, Plane } from "../lib/types";
import { useAppStore } from "../stores/appStore";
import { SliceAxis } from "./SliceAxis";

const statusColors = {
  queued: "var(--text-dim)",
  running: "#f97316",
  complete: "var(--accent)",
} as const;

const statusLabels = {
  queued: "Queued",
  running: "Running",
  complete: "Complete",
} as const;

const planeOptions: { key: Plane; label: string }[] = [
  { key: "coronal", label: "Coronal" },
  { key: "sagittal", label: "Sagittal" },
  { key: "horizontal", label: "Horizontal" },
];

export function BrainCard({ brain }: { brain: Brain }) {
  const navigateToBrain = useAppStore((s) => s.navigateToBrain);
  const removeBrain = useAppStore((s) => s.removeBrain);
  const loadBrainImages = useAppStore((s) => s.loadBrainImages);
  const setHoveredBrain = useAppStore((s) => s.setHoveredBrain);
  const setBrainPlane = useAppStore((s) => s.setBrainPlane);
  const atlasInfo = useAppStore((s) => s.atlasInfo);

  const hasImages = brain.slices.length > 0;
  const progress = hasImages ? brain.completedCount / brain.slices.length : 0;

  const handleLoadImages = async (e: React.MouseEvent) => {
    e.stopPropagation();
    const { open } = await import("@tauri-apps/plugin-dialog");
    const folder = await open({ directory: true, title: "Select slice folder" });
    if (folder) {
      await loadBrainImages(brain.id, folder as string);
    }
  };

  return (
    <div
      className="brain-card-v2"
      onClick={() => navigateToBrain(brain.id)}
      onMouseEnter={() => setHoveredBrain(brain.id)}
      onMouseLeave={() => setHoveredBrain(null)}
    >
      {/* Remove button */}
      <button
        className="brain-card-v2-remove"
        onClick={(e) => { e.stopPropagation(); removeBrain(brain.id); }}
        title="Remove"
      >
        &times;
      </button>

      {/* Data plate */}
      <div className="brain-plate">
        <div className="brain-plate-header">
          <span className="brain-plate-name">{brain.name}</span>
          {hasImages && (
            <span className="brain-plate-count">{brain.slices.length}</span>
          )}
        </div>

        <div className="brain-plate-sub">
          {hasImages
            ? `${brain.slices.length} section${brain.slices.length !== 1 ? "s" : ""}`
            : "No sections loaded"
          }
        </div>

        <div
          className="brain-plate-plane"
          onClick={(e) => e.stopPropagation()}
          role="group"
          aria-label={`Slicing plane for brain ${brain.name}`}
        >
          {planeOptions.map(({ key, label }) => (
            <button
              key={key}
              type="button"
              className={`brain-plate-plane-btn ${brain.plane === key ? "active" : ""}`}
              onClick={(e) => {
                e.stopPropagation();
                setBrainPlane(brain.id, key);
              }}
              aria-pressed={brain.plane === key}
              title={`Set slicing plane to ${label}`}
            >
              {label}
            </button>
          ))}
        </div>

        {hasImages && atlasInfo && (
          <SliceAxis slices={brain.slices} />
        )}

        {!hasImages && (
          <div className="brain-plate-empty">
            <span className="brain-plate-empty-hint">—</span>
          </div>
        )}
      </div>

      {/* Info bar */}
      <div className="brain-card-v2-info">
        <div className="brain-card-v2-row">
          <div className="brain-card-v2-status">
            <span className="brain-card-v2-dot" style={{ background: statusColors[brain.status] }} />
            <span>{statusLabels[brain.status]}</span>
          </div>
          <span className="brain-card-v2-progress-text">
            {brain.completedCount}/{brain.slices.length}
          </span>
        </div>
        {hasImages && (
          <div className="brain-card-v2-progress-track">
            <div className="brain-card-v2-progress-fill" style={{ width: `${progress * 100}%` }} />
          </div>
        )}

        {/* Load images — overlays only the info bar */}
        <div className="brain-card-v2-info-overlay" onClick={handleLoadImages}>
          <span>Load Images</span>
        </div>
      </div>
    </div>
  );
}

export function AddBrainCard() {
  const addEmptyBrains = useAppStore((s) => s.addEmptyBrains);
  const addBrainFromFolder = useAppStore((s) => s.addBrainFromFolder);

  const handleDropFolder = async () => {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const folder = await open({ directory: true, title: "Select brain slice folder" });
    if (folder) {
      await addBrainFromFolder(folder as string);
    }
  };

  return (
    <div className="brain-card-v2 brain-card-v2-add">
      <div className="brain-card-v2-add-default">
        <div className="brain-card-v2-add-icon">+</div>
        <div className="brain-card-v2-add-label">Add Brain</div>
      </div>

      <div className="brain-card-v2-add-options">
        <button className="brain-card-v2-add-btn" onClick={() => addEmptyBrains(1)}>+ 1</button>
        <button className="brain-card-v2-add-btn" onClick={() => addEmptyBrains(5)}>+ 5</button>
        <button className="brain-card-v2-add-btn" onClick={() => addEmptyBrains(10)}>+ 10</button>
        <div className="brain-card-v2-add-sep" />
        <button className="brain-card-v2-add-btn brain-card-v2-add-btn-alt" onClick={handleDropFolder}>
          From Folder
        </button>
      </div>
    </div>
  );
}

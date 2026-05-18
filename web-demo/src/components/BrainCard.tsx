/** Dashboard brain card. Browser variant: AddBrainCard opens a browser file
 *  picker (no Tauri dialog) and creates blob: URL slices. Demo brain is loaded
 *  from the atlas bar; this card just renders existing brains. */

import { useRef } from "react";
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

const DEMO_BRAIN_ROOT = `${import.meta.env.BASE_URL}demo_brain`;

/** Bundled demo-brain slices store bare filenames; uploaded brains store
 *  `blob:` / `http(s):` / `data:` URLs that we use as-is. */
function previewUrl(path: string): string {
  if (
    path.startsWith("blob:") ||
    path.startsWith("http://") ||
    path.startsWith("https://") ||
    path.startsWith("data:")
  ) {
    return path;
  }
  return `${DEMO_BRAIN_ROOT}/${path}`;
}

export function BrainCard({ brain }: { brain: Brain }) {
  const navigateToBrain = useAppStore((s) => s.navigateToBrain);
  const removeBrain = useAppStore((s) => s.removeBrain);
  const setHoveredBrain = useAppStore((s) => s.setHoveredBrain);
  const setBrainPlane = useAppStore((s) => s.setBrainPlane);
  const atlasInfo = useAppStore((s) => s.atlasInfo);

  const hasImages = brain.slices.length > 0;
  const progress = hasImages ? brain.completedCount / brain.slices.length : 0;

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

        {/* Hover preview of the first slice — works for both bundled demo
            brains (bare filenames) and uploaded brains (blob: URLs). */}
        {hasImages && (
          <div className="brain-card-v2-info-overlay">
            <img
              src={previewUrl(brain.slices[0].path)}
              alt={brain.slices[0].name}
              style={{
                width: "100%", height: "100%", objectFit: "cover",
                opacity: 0.85,
              }}
              draggable={false}
            />
          </div>
        )}
      </div>
    </div>
  );
}

/** Add-brain tile: opens a browser file picker (multi-select images). */
export function AddBrainCard() {
  const addBrainFromFiles = useAppStore((s) => s.addBrainFromFiles);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const onPick = () => inputRef.current?.click();

  const onChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    const arr = Array.from(files);
    // Reset the input so re-picking the same selection still fires onChange.
    e.target.value = "";
    await addBrainFromFiles(arr);
  };

  return (
    <div className="brain-card-v2 brain-card-v2-add" onClick={onPick}>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".png,.jpg,.jpeg,.tif,.tiff,.webp,image/png,image/jpeg,image/tiff,image/webp"
        style={{ display: "none" }}
        onChange={onChange}
      />
      <div className="brain-card-v2-add-default">
        <div className="brain-card-v2-add-icon">+</div>
        <div className="brain-card-v2-add-label">Upload slices</div>
      </div>

      <div className="brain-card-v2-add-options">
        <button
          type="button"
          className="brain-card-v2-add-btn brain-card-v2-add-btn-alt"
          onClick={(e) => {
            e.stopPropagation();
            onPick();
          }}
        >
          Pick image files
        </button>
      </div>
    </div>
  );
}

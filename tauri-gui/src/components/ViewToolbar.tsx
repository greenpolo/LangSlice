import { useAppStore } from "../stores/appStore";
import type { ViewMode } from "../lib/types";

const modes: { key: ViewMode; label: string }[] = [
  { key: "3d", label: "3D" },
  { key: "split", label: "Split" },
  { key: "overlay", label: "Overlay" },
];

export function ViewToolbar() {
  const { viewMode, setViewMode } = useAppStore();

  return (
    <div className="view-toolbar">
      {modes.map(({ key, label }) => (
        <button
          key={key}
          className={`view-btn ${viewMode === key ? "active" : ""}`}
          onClick={() => setViewMode(key)}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

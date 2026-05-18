import { useAppStore } from "../stores/appStore";
import type { ViewMode } from "../lib/types";

const modes: { key: ViewMode; label: string }[] = [
  { key: "3d", label: "3D" },
  { key: "split", label: "Split" },
];

export function ViewToolbar() {
  const viewMode = useAppStore((s) => s.viewMode);
  const setViewMode = useAppStore((s) => s.setViewMode);
  const clahePreview = useAppStore((s) => s.clahePreview);
  const setClahePreview = useAppStore((s) => s.setClahePreview);

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
      <label
        className="view-toolbar-clahe"
        title="Show the CLAHE-equalized variant in the slice viewers (where available)"
      >
        <input
          type="checkbox"
          checked={clahePreview}
          onChange={(e) => setClahePreview(e.target.checked)}
        />
        <span>CLAHE</span>
      </label>
    </div>
  );
}

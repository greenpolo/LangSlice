import { useAppStore } from "../stores/appStore";

/** Full-viewport curtain rendered whenever an atlas is being loaded.
 *
 * Gated on `atlasLoadProgress !== null`, which covers both the Rust-side
 * phases (Reading TIFFs, Decoding meshes, etc.) and the frontend-side
 * post-load work (volume decode, mesh fetch — the silent 95–100% tail
 * that previously let users start clicking through stale state).
 *
 * Lives at the app shell so it overlays both the Dashboard and the
 * brain-detail views. Captures clicks so no foreground UI is reachable
 * until the load finishes. */
export function AtlasLoadingOverlay() {
  const progress = useAppStore((s) => s.atlasLoadProgress);

  if (!progress) return null;

  return (
    <div
      className="atlas-loading-overlay"
      role="dialog"
      aria-busy="true"
      aria-live="polite"
    >
      <div className="atlas-loading-card">
        <div className="atlas-loading-title">Loading atlas</div>
        <div className="atlas-loading-phase">{progress.phase}</div>
        <div className="atlas-loading-bar">
          <div
            className="atlas-loading-bar-fill"
            style={{ width: `${progress.percent}%` }}
          />
        </div>
        <div className="atlas-loading-percent">{progress.percent}%</div>
      </div>
    </div>
  );
}

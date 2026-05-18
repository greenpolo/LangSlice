import { useEffect, useMemo, useRef, useState } from "react";
import { useAppStore } from "../stores/appStore";
import type { AvailableAtlas } from "../lib/types";

/** Atlas-manager modal — lists every BrainGlobe atlas with installed/
 * downloadable state and a per-row download button. When `focus` is set on
 * the store (Load-Demo missing-atlas prompt), that row is auto-scrolled
 * into view and its CTA is highlighted. */
export function AtlasManagerModal() {
  const open = useAppStore((s) => s.atlasManagerOpen);
  const focus = useAppStore((s) => s.atlasManagerFocus);
  const close = useAppStore((s) => s.closeAtlasManager);
  const atlases = useAppStore((s) => s.availableForDownload);
  const downloadProgress = useAppStore((s) => s.atlasDownloadProgress);
  const downloadAtlas = useAppStore((s) => s.downloadAtlas);

  const [filter, setFilter] = useState("");
  const focusRowRef = useRef<HTMLDivElement | null>(null);

  // Reset filter when the modal opens. Pre-fill from focus so the user
  // sees the missing atlas immediately.
  useEffect(() => {
    if (!open) return;
    setFilter(focus ?? "");
  }, [open, focus]);

  // Scroll the focus row into view after the list renders.
  useEffect(() => {
    if (!open || !focus) return;
    const t = setTimeout(() => {
      focusRowRef.current?.scrollIntoView({ block: "center" });
    }, 50);
    return () => clearTimeout(t);
  }, [open, focus, atlases]);

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return atlases;
    return atlases.filter((a) => a.name.toLowerCase().includes(q));
  }, [filter, atlases]);

  if (!open) return null;

  return (
    <div
      className="dialog-overlay"
      onClick={(e) => {
        if (e.target === e.currentTarget) close();
      }}
    >
      <div className="dialog atlas-manager-dialog">
        <div className="dialog-header">
          <span className="dialog-title">Manage atlases</span>
          <button
            type="button"
            className="dialog-close"
            onClick={close}
            aria-label="Close"
          >
            {"×"}
          </button>
        </div>

        <div className="atlas-manager-search">
          <input
            type="text"
            className="atlas-manager-filter"
            placeholder="Filter atlases by name..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            autoFocus
          />
          <span className="atlas-manager-count">
            {filtered.length} / {atlases.length}
          </span>
        </div>

        <div className="atlas-manager-body">
          {atlases.length === 0 ? (
            <div className="atlas-manager-empty">Loading atlas catalog...</div>
          ) : filtered.length === 0 ? (
            <div className="atlas-manager-empty">No matching atlases.</div>
          ) : (
            filtered.map((atlas) => (
              <AtlasRow
                key={atlas.name}
                atlas={atlas}
                isFocus={atlas.name === focus}
                refProp={atlas.name === focus ? focusRowRef : null}
                progress={downloadProgress[atlas.name] ?? null}
                onDownload={() => downloadAtlas(atlas.name)}
              />
            ))
          )}
        </div>

        <div className="dialog-footer">
          <button type="button" className="btn-secondary" onClick={close}>
            Done
          </button>
        </div>
      </div>
    </div>
  );
}

interface RowProps {
  atlas: AvailableAtlas;
  isFocus: boolean;
  refProp: React.RefObject<HTMLDivElement | null> | null;
  progress: import("../lib/types").AtlasDownloadProgress | null;
  onDownload: () => void;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function AtlasRow({ atlas, isFocus, refProp, progress, onDownload }: RowProps) {
  const isDownloading =
    progress !== null && progress.phase !== "complete";
  const installed = atlas.downloaded || progress?.phase === "complete";
  const percent =
    progress?.phase === "downloading" &&
    progress.completed !== undefined &&
    progress.total !== undefined &&
    progress.total > 0
      ? (progress.completed / progress.total) * 100
      : null;

  const statusLabel = (() => {
    if (!isDownloading && installed) return `Installed v${atlas.version}`;
    if (!isDownloading) return `Available v${atlas.latest_version}`;
    if (progress?.phase === "resolving") return "Resolving...";
    if (progress?.phase === "downloading") {
      if (percent !== null && progress.total !== undefined) {
        return `${percent.toFixed(0)}% (${formatBytes(progress.completed ?? 0)} / ${formatBytes(progress.total)})`;
      }
      return "Downloading...";
    }
    if (progress?.phase === "extracting") return "Extracting...";
    return "Working...";
  })();

  return (
    <div
      ref={refProp}
      className={`atlas-row ${isFocus ? "atlas-row-focus" : ""}`}
    >
      <div className="atlas-row-main">
        <span className="atlas-row-name">{atlas.name}</span>
        <span className="atlas-row-status">{statusLabel}</span>
      </div>
      <div className="atlas-row-action">
        {installed ? (
          <span className="atlas-row-installed">{"✓"} Installed</span>
        ) : isDownloading ? (
          <div
            className="atlas-row-progress"
            role="progressbar"
            aria-valuenow={percent ?? undefined}
            aria-valuemin={0}
            aria-valuemax={100}
          >
            <div
              className={`atlas-row-progress-fill ${percent === null ? "indeterminate" : ""}`}
              style={percent !== null ? { width: `${percent}%` } : undefined}
            />
          </div>
        ) : (
          <button
            type="button"
            className={`btn-primary atlas-row-download ${isFocus ? "highlight" : ""}`}
            onClick={onDownload}
          >
            Download
          </button>
        )}
      </div>
    </div>
  );
}

import { useEffect } from "react";
import { useAppStore } from "../stores/appStore";

export function SplitView() {
  const compositeDataUrl = useAppStore((s) => s.currentCompositeDataUrl);
  const volumesLoaded = useAppStore((s) => s.volumesLoaded);
  const currentApMm = useAppStore((s) => s.currentApMm);
  const selectedSliceImage = useAppStore((s) => s.selectedSliceImage);
  const selectedSliceIndex = useAppStore((s) => s.selectedSliceIndex);
  const sliceImageLoading = useAppStore((s) => s.sliceImageLoading);
  const selectedBrainId = useAppStore((s) => s.selectedBrainId);
  const brains = useAppStore((s) => s.brains);
  const showBorders = useAppStore((s) => s.showBorders);
  const activeChannel = useAppStore((s) => s.activeChannel);
  const additionalVolumeNames = useAppStore((s) => s.additionalVolumeNames);
  const setShowBorders = useAppStore((s) => s.setShowBorders);
  const setActiveChannel = useAppStore((s) => s.setActiveChannel);
  const buildComposite = useAppStore((s) => s.buildComposite);

  const brain = brains.find((b) => b.id === selectedBrainId);
  const selectedSliceName = brain && selectedSliceIndex !== null
    ? brain.slices[selectedSliceIndex]?.name
    : null;

  // Build composite on mount if volumes are ready
  useEffect(() => {
    if (volumesLoaded && !compositeDataUrl) {
      buildComposite();
    }
  }, [volumesLoaded, compositeDataUrl, buildComposite]);

  const channels = ["reference", ...additionalVolumeNames];

  return (
    <div className="split-view">
      {/* Left: histology slice */}
      <div className="split-panel">
        <div className="split-panel-label">
          Histology
          {selectedSliceName && (
            <span className="split-panel-ap">{selectedSliceName}</span>
          )}
        </div>
        {sliceImageLoading ? (
          <div className="split-panel-empty">
            <span className="split-panel-empty-text">Loading...</span>
          </div>
        ) : selectedSliceImage ? (
          <div className="split-panel-image">
            <img
              src={`data:image/png;base64,${selectedSliceImage}`}
              alt="Histology slice"
              className="split-image"
            />
          </div>
        ) : (
          <div className="split-panel-empty">
            <span className="split-panel-empty-text">
              {brain && brain.slices.length > 0
                ? "Select a slice from the list"
                : "Load images first"
              }
            </span>
          </div>
        )}
      </div>

      {/* Divider */}
      <div className="split-divider" />

      {/* Right: atlas composite */}
      <div className="split-panel">
        <div className="split-panel-label">
          Atlas
          <span className="split-panel-ap">{currentApMm.toFixed(2)} mm</span>
        </div>

        {/* Channel controls */}
        <div className="split-panel-controls">
          <label className="split-toggle">
            <input
              type="checkbox"
              checked={showBorders}
              onChange={(e) => setShowBorders(e.target.checked)}
            />
            Borders
          </label>
          {channels.length > 1 && (
            <select
              className="control-select split-channel-select"
              value={activeChannel}
              onChange={(e) => setActiveChannel(e.target.value)}
            >
              {channels.map((ch) => (
                <option key={ch} value={ch}>
                  {ch.charAt(0).toUpperCase() + ch.slice(1)}
                </option>
              ))}
            </select>
          )}
        </div>

        {compositeDataUrl ? (
          <div className="split-panel-image">
            <img
              src={compositeDataUrl}
              alt="Atlas composite"
              className="split-image"
            />
          </div>
        ) : (
          <div className="split-panel-empty">
            <span className="split-panel-empty-text">
              {volumesLoaded ? "Building composite..." : "Load atlas first"}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

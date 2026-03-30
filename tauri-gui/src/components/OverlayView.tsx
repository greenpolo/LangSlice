import { useEffect } from "react";
import { useAppStore } from "../stores/appStore";

export function OverlayView() {
  const compositeDataUrl = useAppStore((s) => s.currentCompositeDataUrl);
  const volumesLoaded = useAppStore((s) => s.volumesLoaded);
  const currentApMm = useAppStore((s) => s.currentApMm);
  const atlasOpacity = useAppStore((s) => s.atlasOpacity);
  const setAtlasOpacity = useAppStore((s) => s.setAtlasOpacity);
  const selectedSliceImage = useAppStore((s) => s.selectedSliceImage);
  const selectedSliceIndex = useAppStore((s) => s.selectedSliceIndex);
  const buildComposite = useAppStore((s) => s.buildComposite);

  const hasSlice = selectedSliceImage !== null && selectedSliceIndex !== null;

  useEffect(() => {
    if (volumesLoaded && !compositeDataUrl) {
      buildComposite();
    }
  }, [volumesLoaded, compositeDataUrl, buildComposite]);

  return (
    <div className="overlay-view">
      <div className="overlay-canvas">
        {hasSlice ? (
          <div className="overlay-layers">
            <img
              src={`data:image/jpeg;base64,${selectedSliceImage}`}
              alt="Histology"
              className="overlay-base-image"
            />
            {compositeDataUrl && (
              <img
                src={compositeDataUrl}
                alt="Atlas overlay"
                className="overlay-atlas-image"
                style={{ opacity: atlasOpacity }}
              />
            )}
            <div className="overlay-badge">{currentApMm.toFixed(2)} mm</div>
          </div>
        ) : compositeDataUrl ? (
          <div className="overlay-atlas-only">
            <img
              src={compositeDataUrl}
              alt="Atlas composite"
              className="overlay-image"
            />
            <div className="overlay-badge">{currentApMm.toFixed(2)} mm</div>
          </div>
        ) : (
          <div className="overlay-empty">
            <span>
              {volumesLoaded ? "Building composite..." : "Load atlas first"}
            </span>
          </div>
        )}
      </div>

      <div className="overlay-controls">
        <span className="overlay-control-label">Atlas Opacity</span>
        <input
          type="range"
          className="ap-slider"
          min={0}
          max={1}
          step={0.05}
          value={atlasOpacity}
          onChange={(e) => setAtlasOpacity(parseFloat(e.target.value))}
        />
        <span className="overlay-control-value">{Math.round(atlasOpacity * 100)}%</span>
      </div>
    </div>
  );
}

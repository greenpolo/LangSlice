import { useAppStore } from "../stores/appStore";

export function SettingsPanel() {
  const atlasInfo = useAppStore((s) => s.atlasInfo);
  const currentApMm = useAppStore((s) => s.currentApMm);
  const setApPosition = useAppStore((s) => s.setApPosition);
  const selectedBrainId = useAppStore((s) => s.selectedBrainId);
  const brains = useAppStore((s) => s.brains);

  const selectedBrain = brains.find((b) => b.id === selectedBrainId);

  return (
    <div className="panel-left">
      {/* Brain slice list */}
      {selectedBrain && (
        <div className="section-block">
          <div className="section-label">Slices</div>
          {selectedBrain.slices.length === 0 ? (
            <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-dim)" }}>
              No images loaded
            </div>
          ) : (
            <div className="slice-list">
              {selectedBrain.slices.map((slice, i) => (
                <div key={i} className="slice-list-item">
                  <span className="slice-list-status">
                    {slice.status === "done" ? "\u2713" :
                     slice.status === "running" ? "\u27F3" : "\u25CB"}
                  </span>
                  <span className="slice-list-name">{slice.name}</span>
                  {slice.apMm !== undefined && (
                    <span className="slice-list-ap">{slice.apMm.toFixed(2)}</span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Atlas Info */}
      {atlasInfo && (
        <div className="section-block">
          <div className="section-label">Atlas</div>
          <div className="info-grid">
            <span className="info-key">Name</span>
            <span className="info-val">{atlasInfo.name}</span>
            <span className="info-key">Shape</span>
            <span className="info-val">{atlasInfo.shape.join(" x ")}</span>
            <span className="info-key">AP</span>
            <span className="info-val">{atlasInfo.ap_min_mm.toFixed(1)} - {atlasInfo.ap_max_mm.toFixed(1)} mm</span>
          </div>
        </div>
      )}

      {/* AP Slider */}
      {atlasInfo && (
        <div className="section-block">
          <div className="section-label">AP Position</div>
          <div className="ap-slider-container">
            <div className="ap-value">
              {currentApMm.toFixed(2)}
              <span className="ap-unit">mm</span>
            </div>
            <input
              type="range"
              className="ap-slider"
              min={atlasInfo.ap_min_mm}
              max={atlasInfo.ap_max_mm}
              step={atlasInfo.resolution[0] / 1000}
              value={currentApMm}
              onChange={(e) => setApPosition(parseFloat(e.target.value))}
            />
          </div>
        </div>
      )}

      <div style={{ flex: 1 }} />
    </div>
  );
}

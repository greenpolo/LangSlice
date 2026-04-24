import { useAppStore } from "../stores/appStore";

export function SettingsPanel() {
  const atlasInfo = useAppStore((s) => s.atlasInfo);
  const currentApMm = useAppStore((s) => s.currentApMm);
  const setApPosition = useAppStore((s) => s.setApPosition);
  const selectedBrainId = useAppStore((s) => s.selectedBrainId);
  const selectedSliceIndex = useAppStore((s) => s.selectedSliceIndex);
  const sliceImageLoading = useAppStore((s) => s.sliceImageLoading);
  const brains = useAppStore((s) => s.brains);
  const selectSlice = useAppStore((s) => s.selectSlice);
  const pipelineRunning = useAppStore((s) => s.pipelineRunning);
  const runPipeline = useAppStore((s) => s.runPipeline);
  const exportResults = useAppStore((s) => s.exportResults);

  // Agent settings
  const agentModel = useAppStore((s) => s.agentModel);
  const agentWorkflow = useAppStore((s) => s.agentWorkflow);
  const agentThinking = useAppStore((s) => s.agentThinking);
  const agentTemperature = useAppStore((s) => s.agentTemperature);
  const agentVlmResolution = useAppStore((s) => s.agentVlmResolution);
  const agentMaxIterations = useAppStore((s) => s.agentMaxIterations);
  const setSetting = useAppStore((s) => s.setAgentSetting);

  const selectedBrain = brains.find((b) => b.id === selectedBrainId);
  const selectedSlice = selectedBrain && selectedSliceIndex !== null
    ? selectedBrain.slices[selectedSliceIndex] : null;
  const canRun = selectedSlice && !pipelineRunning;
  const canExport = selectedSlice && selectedSlice.apMm !== undefined;

  return (
    <div className="panel-left">
      {/* Slice list */}
      {selectedBrain && (
        <div className="section-block" style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
          <div className="section-label">
            Slices
            {selectedBrain.slices.length > 0 && (
              <span style={{ float: "right", textTransform: "none", letterSpacing: 0 }}>
                {selectedBrain.slices.length}
              </span>
            )}
          </div>
          {selectedBrain.slices.length === 0 ? (
            <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-dim)" }}>
              No images loaded
            </div>
          ) : (
            <div className="slice-list">
              {selectedBrain.slices.map((slice, i) => (
                <div
                  key={i}
                  className={`slice-list-item ${selectedSliceIndex === i ? "slice-list-item--selected" : ""}`}
                  onClick={() => selectSlice(i)}
                >
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
          {sliceImageLoading && (
            <div style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--text-dim)", marginTop: 6 }}>
              Loading image...
            </div>
          )}
        </div>
      )}

      {/* Run / Export buttons */}
      <div className="section-block">
        <button
          className="btn-primary"
          onClick={runPipeline}
          disabled={!canRun}
          style={{ marginBottom: 6 }}
        >
          {pipelineRunning ? "Running..." : "Run Agent"}
        </button>
        <button
          className="btn-primary"
          onClick={exportResults}
          disabled={!canExport}
          style={{ background: canExport ? "var(--bg-surface)" : undefined, color: canExport ? "var(--accent)" : undefined, border: "1px solid var(--border)" }}
        >
          Export
        </button>
      </div>

      {/* Agent Parameters */}
      <div className="section-block">
        <div className="section-label">Agent</div>

        <div className="settings-row">
          <label className="settings-label">Model</label>
          <select className="control-select" value={agentModel} onChange={(e) => setSetting("agentModel", e.target.value)}>
            <option value="gemini-3-flash-preview">gemini-3-flash</option>
            <option value="gemini-3.1-pro-preview">gemini-3.1-pro</option>
            <option value="gemini-3-pro-image-preview">gemini-3-pro-image</option>
            <option value="gemini-3.1-flash-image-preview">gemini-3.1-flash-image</option>
          </select>
        </div>

        <div className="settings-row">
          <label className="settings-label">Workflow</label>
          <select className="control-select" value={agentWorkflow} onChange={(e) => setSetting("agentWorkflow", e.target.value)}>
            <option value="auto">Auto</option>
            <option value="tool_use">Tool Use</option>
            <option value="image_gen">Image Gen</option>
          </select>
        </div>

        <div className="settings-row">
          <label className="settings-label">Thinking</label>
          <select className="control-select" value={agentThinking} onChange={(e) => setSetting("agentThinking", e.target.value)}>
            <option value="MINIMAL">Minimal</option>
            <option value="LOW">Low</option>
            <option value="MEDIUM">Medium</option>
            <option value="HIGH">High</option>
          </select>
        </div>

        <div className="settings-row">
          <label className="settings-label">Temp</label>
          <input
            type="number"
            className="control-select"
            style={{ width: 70 }}
            min={0} max={2} step={0.05}
            value={agentTemperature}
            onChange={(e) => setSetting("agentTemperature", parseFloat(e.target.value))}
          />
        </div>

        <div className="settings-row">
          <label className="settings-label">VLM Res</label>
          <select className="control-select" value={agentVlmResolution} onChange={(e) => setSetting("agentVlmResolution", parseInt(e.target.value))}>
            <option value={512}>512</option>
            <option value={1024}>1K</option>
            <option value={2048}>2K</option>
            <option value={4096}>4K</option>
          </select>
        </div>

        <div className="settings-row">
          <label className="settings-label">AP Turns</label>
          <input
            type="number"
            className="control-select"
            style={{ width: 60 }}
            min={1} max={50}
            value={agentMaxIterations}
            onChange={(e) => setSetting("agentMaxIterations", parseInt(e.target.value))}
          />
        </div>
      </div>

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
    </div>
  );
}

/** Left-side per-stage settings panel. Browser variant: a single local
 *  sidecar for estimation, a single Gemini image model for registration. */

import { useAppStore } from "../stores/appStore";

const REGISTER_IMAGE_MODELS: { value: string; label: string }[] = [
  { value: "gemini-3.1-flash-image-preview", label: "Nano Banana 2 (Flash Image)" },
  { value: "gemini-3-pro-image-preview", label: "Nano Banana Pro (Pro Image)" },
];

const DEMO_BRAIN_ROOT = `${import.meta.env.BASE_URL}demo_brain`;

export function SettingsPanel() {
  const atlasInfo = useAppStore((s) => s.atlasInfo);
  const currentApMm = useAppStore((s) => s.currentApMm);
  const setApPosition = useAppStore((s) => s.setApPosition);
  const selectedBrainId = useAppStore((s) => s.selectedBrainId);
  const selectedSliceIndex = useAppStore((s) => s.selectedSliceIndex);
  const brains = useAppStore((s) => s.brains);
  const selectSlice = useAppStore((s) => s.selectSlice);
  const toggleSliceVisibleIn3D = useAppStore((s) => s.toggleSliceVisibleIn3D);
  const pipelineRunning = useAppStore((s) => s.pipelineRunning);
  const pipelineStatus = useAppStore((s) => s.pipelineStatus);
  const runEstimatePosition = useAppStore((s) => s.runEstimatePosition);
  const runRegistration = useAppStore((s) => s.runRegistration);
  const setApLocked = useAppStore((s) => s.setApLocked);
  const demoBrainGroundTruthMm = useAppStore((s) => s.demoBrainGroundTruthMm);

  const estimateModel = useAppStore((s) => s.estimateModel);
  const estimateTemperature = useAppStore((s) => s.estimateTemperature);
  const estimateMaxIterations = useAppStore((s) => s.estimateMaxIterations);
  const estimateUseClahe = useAppStore((s) => s.estimateUseClahe);
  const estimateThinkingPreset = useAppStore((s) => s.estimateThinkingPreset);
  const estimateMediaResolution = useAppStore((s) => s.estimateMediaResolution);

  const registerImageModel = useAppStore((s) => s.registerImageModel);
  const registerTemperature = useAppStore((s) => s.registerTemperature);
  const registerUseClahe = useAppStore((s) => s.registerUseClahe);
  const registerThinkingPreset = useAppStore((s) => s.registerThinkingPreset);
  const registerMediaResolution = useAppStore((s) => s.registerMediaResolution);

  const setStageSetting = useAppStore((s) => s.setStageSetting);

  const selectedBrain = brains.find((b) => b.id === selectedBrainId);
  const selectedSlice = selectedBrain && selectedSliceIndex !== null
    ? selectedBrain.slices[selectedSliceIndex] : null;
  const canEstimate = Boolean(selectedSlice) && !pipelineRunning;

  const estimateIsGemini = /gemini/i.test(estimateModel);
  // Demo brain ships pre-baked CLAHE variants. Uploaded brains don't have
  // them, so the per-stage CLAHE toggle has nothing to send — disable it.
  const selectedSliceHasClahe = Boolean(selectedSlice?.clahePath);
  const hasApPosition =
    Boolean(selectedSlice) && selectedSlice?.apMm !== undefined;
  const apLocked = Boolean(selectedSlice?.apLocked);
  const canLockPosition = hasApPosition && !pipelineRunning;
  const canRegister = hasApPosition && apLocked && !pipelineRunning;

  const selectedGroundTruthMm = selectedSlice
    ? demoBrainGroundTruthMm[selectedSlice.path]
    : undefined;

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
              {selectedBrain.slices.map((slice, i) => {
                const hiddenIn3D = slice.visibleIn3D === false;
                return (
                  <div
                    key={i}
                    className={`slice-list-item ${selectedSliceIndex === i ? "slice-list-item--selected" : ""}`}
                    onClick={() => selectSlice(i)}
                  >
                    <span className="slice-list-status">
                      {slice.status === "done" ? "✓" :
                       slice.status === "running" ? "⟳" : "○"}
                    </span>
                    <img
                      src={`${DEMO_BRAIN_ROOT}/${slice.path}`}
                      alt=""
                      className="slice-list-thumb"
                      loading="lazy"
                      draggable={false}
                    />
                    <span className="slice-list-name">{slice.name}</span>
                    {slice.apMm !== undefined && (
                      <span className="slice-list-ap">{slice.apMm.toFixed(2)}</span>
                    )}
                    {slice.apLocked && (
                      <button
                        type="button"
                        className={`slice-visibility-btn ${hiddenIn3D ? "slice-visibility-btn--hidden" : ""}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleSliceVisibleIn3D(i);
                        }}
                        title={hiddenIn3D ? "Show in 3D volume" : "Hide from 3D volume"}
                        aria-label={hiddenIn3D ? "Show slice in 3D" : "Hide slice from 3D"}
                        aria-pressed={!hiddenIn3D}
                      >
                        {hiddenIn3D ? "◌" : "●"}
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* === Estimate Position === */}
      <div className="section-block">
        <div className="section-label">Estimate Position</div>

        <div className="settings-row">
          <label className="settings-label">Model</label>
          <input
            type="text"
            className="control-select"
            value={estimateModel}
            onChange={(e) => setStageSetting("estimate", "model", e.target.value)}
            title="Model id served by the local litert-lm sidecar"
          />
        </div>

        <details className="advanced-disclosure">
          <summary>Advanced</summary>
          <div className="advanced-body">
            <div className="settings-row">
              <label
                className="settings-label"
                title="Use the pre-baked CLAHE variant of this slice"
              >
                CLAHE
              </label>
              <input
                type="checkbox"
                checked={estimateUseClahe}
                disabled={!selectedSliceHasClahe}
                onChange={(e) => setStageSetting("estimate", "useClahe", e.target.checked)}
                title={
                  selectedSliceHasClahe
                    ? "Send the pre-baked CLAHE version of the slice to the model"
                    : "This slice has no pre-baked CLAHE variant (uploads aren't CLAHE'd in-browser)"
                }
              />
            </div>

            {estimateIsGemini && (
              <>
                <div className="settings-row">
                  <label className="settings-label" title="Gemini thinking budget">
                    Thinking
                  </label>
                  <select
                    className="control-select"
                    value={estimateThinkingPreset}
                    onChange={(e) => setStageSetting("estimate", "thinkingPreset", e.target.value)}
                  >
                    <option value="off">Off</option>
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                  </select>
                </div>
                <div className="settings-row">
                  <label className="settings-label" title="Gemini media resolution (token budget per image)">
                    Media res.
                  </label>
                  <select
                    className="control-select"
                    value={estimateMediaResolution}
                    onChange={(e) => setStageSetting("estimate", "mediaResolution", e.target.value)}
                  >
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                  </select>
                </div>
              </>
            )}

            <div className="settings-row">
              <label className="settings-label">Temperature</label>
              <input
                type="number"
                className="control-select"
                style={{ width: 70 }}
                min={0} max={2} step={0.05}
                value={estimateTemperature}
                onChange={(e) => setStageSetting("estimate", "temperature", parseFloat(e.target.value))}
              />
            </div>

            <div className="settings-row">
              <label className="settings-label" title="Max tool-loop iterations">
                Max Iterations
              </label>
              <input
                type="number"
                className="control-select"
                style={{ width: 70 }}
                min={1} max={20}
                value={estimateMaxIterations}
                onChange={(e) => setStageSetting("estimate", "maxIterations", parseInt(e.target.value))}
              />
            </div>
          </div>
        </details>

        <button
          className="btn-primary"
          onClick={runEstimatePosition}
          disabled={!canEstimate}
          style={{ marginTop: 8 }}
        >
          {pipelineRunning && pipelineStatus === "estimating"
            ? "Estimating..."
            : "Estimate Position"}
        </button>
      </div>

      {/* === Run Registration === */}
      <div className="section-block">
        <div className="section-label">Run Registration</div>

        <div className="settings-row">
          <label className="settings-label" title="Image-generation model (Nano Banana family)">
            Model
          </label>
          <select
            className="control-select"
            value={registerImageModel}
            onChange={(e) => setStageSetting("register", "imageModel", e.target.value)}
          >
            {REGISTER_IMAGE_MODELS.map((m) => (
              <option key={m.value} value={m.value}>{m.label}</option>
            ))}
          </select>
        </div>

        <details className="advanced-disclosure">
          <summary>Advanced</summary>
          <div className="advanced-body">
            <div className="settings-row">
              <label
                className="settings-label"
                title="Use the pre-baked CLAHE variant of this slice"
              >
                CLAHE
              </label>
              <input
                type="checkbox"
                checked={registerUseClahe}
                disabled={!selectedSliceHasClahe}
                onChange={(e) => setStageSetting("register", "useClahe", e.target.checked)}
                title={
                  selectedSliceHasClahe
                    ? "Send the pre-baked CLAHE version of the slice to Gemini"
                    : "This slice has no pre-baked CLAHE variant (uploads aren't CLAHE'd in-browser)"
                }
              />
            </div>

            <div className="settings-row">
              <label className="settings-label" title="Gemini thinking budget">
                Thinking
              </label>
              <select
                className="control-select"
                value={registerThinkingPreset}
                onChange={(e) => setStageSetting("register", "thinkingPreset", e.target.value)}
              >
                <option value="off">Off</option>
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </div>

            <div className="settings-row">
              <label className="settings-label" title="Gemini media resolution (token budget per image)">
                Media res.
              </label>
              <select
                className="control-select"
                value={registerMediaResolution}
                onChange={(e) => setStageSetting("register", "mediaResolution", e.target.value)}
              >
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </div>

            <div className="settings-row">
              <label className="settings-label">Temperature</label>
              <input
                type="number"
                className="control-select"
                style={{ width: 70 }}
                min={0} max={2} step={0.05}
                value={registerTemperature}
                onChange={(e) => setStageSetting("register", "temperature", parseFloat(e.target.value))}
              />
            </div>
          </div>
        </details>

        <button
          className="btn-primary"
          onClick={runRegistration}
          disabled={!canRegister}
          title={
            canRegister
              ? "Run image-gen registration at the slice's locked AP position"
              : !hasApPosition
                ? "Set an AP position via Estimate or the slider first"
                : "Lock the AP position to enable registration"
          }
          style={{ marginTop: 8 }}
        >
          {pipelineRunning && pipelineStatus === "registering"
            ? "Registering..."
            : "Run Registration"}
        </button>
      </div>

      {/* AP Slider */}
      {atlasInfo && (
        <div className="section-block">
          <div className="section-label">
            AP Position
            {selectedSlice && (
              <span
                className={apLocked ? "ap-lock-pill ap-lock-pill--locked" : "ap-lock-pill"}
                style={{ float: "right", textTransform: "none", letterSpacing: 0 }}
              >
                {apLocked ? "Locked" : "Unlocked"}
              </span>
            )}
          </div>
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
          {selectedGroundTruthMm !== undefined && selectedSlice?.apMm !== undefined && (
            <div
              style={{
                fontFamily: "var(--font-mono)", fontSize: 11,
                color: "var(--text-dim)", marginTop: 6,
              }}
              title="Ground-truth AP from the bundled demo brain manifest"
            >
              estimate {selectedSlice.apMm.toFixed(2)} mm · ground truth{" "}
              {selectedGroundTruthMm.toFixed(2)} mm · Δ{" "}
              {Math.abs(selectedSlice.apMm - selectedGroundTruthMm).toFixed(2)} mm
            </div>
          )}
          <button
            className={apLocked ? "btn-primary ap-lock-btn ap-lock-btn--locked" : "btn-primary ap-lock-btn"}
            onClick={() => setApLocked(!apLocked)}
            disabled={!canLockPosition}
            title={
              !hasApPosition
                ? "Pick an AP position with the slider or run Estimate first"
                : apLocked
                  ? "Click to unlock and re-adjust the AP position"
                  : "Click to lock this AP position and enable Run Registration"
            }
            style={{ marginTop: 8 }}
          >
            {apLocked ? "Position Locked — click to unlock" : "Lock Position"}
          </button>
          {selectedSlice?.quickAffineRunning && (
            <div className="quick-affine-status quick-affine-status--running">
              <span className="quick-affine-spinner" aria-hidden>◌</span>
              Aligning preview...
            </div>
          )}
          {!selectedSlice?.quickAffineRunning &&
            selectedSlice?.quickAffineWarpedBitmap &&
            !selectedSlice?.registrationResult && (
              <div className="quick-affine-status quick-affine-status--ready">
                <span aria-hidden>◆</span> Preview ready in 3D viewer
              </div>
            )}
          {selectedSlice?.quickAffineError && !selectedSlice?.quickAffineRunning && (
            <div
              className="quick-affine-status quick-affine-status--error"
              title={selectedSlice.quickAffineError}
            >
              Preview alignment failed (see logs)
            </div>
          )}
        </div>
      )}
    </div>
  );
}

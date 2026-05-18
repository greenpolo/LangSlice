import { convertFileSrc } from "@tauri-apps/api/core";
import { useAppStore } from "../stores/appStore";
import { useMemo } from "react";

// Estimation is a tool-use loop over Gemini text/vision models; registration
// uses the image-generation "Nano Banana" family for the colored-region warp
// and a Gemini text model for the agentic review pass.
const CLOUD_GEMINI_ESTIMATE_MODELS: { value: string; label: string }[] = [
  { value: "gemini-3-flash-preview", label: "Gemini 3 Flash" },
  { value: "gemini-3.1-pro-preview", label: "Gemini 3.1 Pro" },
];

// Gemma 4 cloud models routed through Google AI Studio (same google-genai
// SDK as Gemini). Tool-calling support on Gemma is more variable than
// Gemini's — Flash/Pro are still the reliable picks for estimation, but
// these are available for users who want to compare.
const CLOUD_GEMMA_ESTIMATE_MODELS: { value: string; label: string }[] = [
  { value: "gemma-4-31b-it", label: "Gemma 4 31B" },
  { value: "gemma-4-26b-a4b-it", label: "Gemma 4 26B A4B" },
];

const REGISTER_IMAGE_MODELS: { value: string; label: string }[] = [
  { value: "gemini-3.1-flash-image-preview", label: "Nano Banana 2 (Flash Image)" },
  { value: "gemini-3-pro-image-preview", label: "Nano Banana Pro (Pro Image)" },
];

const THINKING_LEVELS = ["MINIMAL", "LOW", "MEDIUM", "HIGH"] as const;
const MEDIA_RESOLUTIONS = ["low", "medium", "high", "ultra_high"] as const;

// Encode a local-engine model as a single string the <select> can round-trip.
// Format: local::<endpoint>::<modelId>. Endpoint and modelId may contain ':'
// (e.g. "http://...:1234/v1", "gemma4:e4b") so we use "::" as a separator.
const LOCAL_PREFIX = "local::";
function encodeLocalChoice(endpoint: string, modelId: string): string {
  return `${LOCAL_PREFIX}${endpoint}::${modelId}`;
}
function decodeLocalChoice(value: string): { endpoint: string; modelId: string } | null {
  if (!value.startsWith(LOCAL_PREFIX)) return null;
  const rest = value.slice(LOCAL_PREFIX.length);
  const sep = rest.indexOf("::");
  if (sep < 0) return null;
  return { endpoint: rest.slice(0, sep), modelId: rest.slice(sep + 2) };
}

export function SettingsPanel() {
  const atlasInfo = useAppStore((s) => s.atlasInfo);
  const currentApMm = useAppStore((s) => s.currentApMm);
  const setApPosition = useAppStore((s) => s.setApPosition);
  const selectedBrainId = useAppStore((s) => s.selectedBrainId);
  const selectedSliceIndex = useAppStore((s) => s.selectedSliceIndex);
  const sliceImageLoading = useAppStore((s) => s.sliceImageLoading);
  const brains = useAppStore((s) => s.brains);
  const selectSlice = useAppStore((s) => s.selectSlice);
  const toggleSliceVisibleIn3D = useAppStore((s) => s.toggleSliceVisibleIn3D);
  const pipelineRunning = useAppStore((s) => s.pipelineRunning);
  const pipelineStatus = useAppStore((s) => s.pipelineStatus);
  const runEstimatePosition = useAppStore((s) => s.runEstimatePosition);
  const runRegistration = useAppStore((s) => s.runRegistration);
  const exportResults = useAppStore((s) => s.exportResults);
  const setApLocked = useAppStore((s) => s.setApLocked);

  // Estimate stage
  const estimateModel = useAppStore((s) => s.estimateModel);
  const estimateEndpoint = useAppStore((s) => s.estimateEndpoint);
  const estimateThinking = useAppStore((s) => s.estimateThinking);
  const estimateTemperature = useAppStore((s) => s.estimateTemperature);
  const estimateMediaResolution = useAppStore((s) => s.estimateMediaResolution);
  const estimateMaxIterations = useAppStore((s) => s.estimateMaxIterations);
  const estimatePreprocess = useAppStore((s) => s.estimatePreprocess);
  const setEstimateModelChoice = useAppStore((s) => s.setEstimateModelChoice);
  const localEngines = useAppStore((s) => s.localEngines);

  // Register stage
  const registerImageModel = useAppStore((s) => s.registerImageModel);
  const registerThinking = useAppStore((s) => s.registerThinking);
  const registerTemperature = useAppStore((s) => s.registerTemperature);

  const setStageSetting = useAppStore((s) => s.setStageSetting);

  const selectedBrain = brains.find((b) => b.id === selectedBrainId);
  const selectedSlice = selectedBrain && selectedSliceIndex !== null
    ? selectedBrain.slices[selectedSliceIndex] : null;
  const canEstimate = Boolean(selectedSlice) && !pipelineRunning;
  const hasApPosition =
    Boolean(selectedSlice) && selectedSlice?.apMm !== undefined;
  const apLocked = Boolean(selectedSlice?.apLocked);
  const canLockPosition = hasApPosition && !pipelineRunning;
  const canRegister = hasApPosition && apLocked && !pipelineRunning;
  const canExport = selectedSlice && selectedSlice.apMm !== undefined;

  // Flatten reachable engines into one list of <option>s, grouped by engine.
  const localModelGroups = useMemo(
    () =>
      localEngines
        .filter((e) => e.reachable && e.models.length > 0)
        .map((engine) => ({
          label: engine.name,
          options: engine.models.map((m) => ({
            value: encodeLocalChoice(m.endpoint, m.id),
            label: m.id,
          })),
        })),
    [localEngines],
  );

  const estimateSelectValue =
    estimateEndpoint !== null
      ? encodeLocalChoice(estimateEndpoint, estimateModel)
      : estimateModel;

  const handleEstimateModelChange = (value: string) => {
    const decoded = decodeLocalChoice(value);
    if (decoded) {
      setEstimateModelChoice(decoded.modelId, decoded.endpoint);
    } else {
      setEstimateModelChoice(value, null);
    }
  };

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
                      src={convertFileSrc(slice.path)}
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
          {sliceImageLoading && (
            <div style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--text-dim)", marginTop: 6 }}>
              Loading image...
            </div>
          )}
        </div>
      )}

      {/* === Estimate Position === */}
      <div className="section-block">
        <div className="section-label">Estimate Position</div>

        <div className="settings-row">
          <label className="settings-label">Model</label>
          <select
            className="control-select"
            value={estimateSelectValue}
            onChange={(e) => handleEstimateModelChange(e.target.value)}
          >
            <optgroup label="Cloud · Gemini">
              {CLOUD_GEMINI_ESTIMATE_MODELS.map((m) => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </optgroup>
            <optgroup label="Cloud · Gemma">
              {CLOUD_GEMMA_ESTIMATE_MODELS.map((m) => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </optgroup>
            {localModelGroups.map((g) => (
              <optgroup key={g.label} label={`Local · ${g.label}`}>
                {g.options.map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </optgroup>
            ))}
          </select>
        </div>

        <details className="advanced-disclosure">
          <summary>Advanced</summary>
          <div className="advanced-body">
            <div className="settings-row">
              <label className="settings-label">Thinking</label>
              <select
                className="control-select"
                value={estimateThinking}
                onChange={(e) => setStageSetting("estimate", "thinking", e.target.value)}
              >
                {THINKING_LEVELS.map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </div>

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
              <label className="settings-label" title="Gemini API media_resolution">
                Media Resolution
              </label>
              <select
                className="control-select"
                value={estimateMediaResolution}
                onChange={(e) => setStageSetting("estimate", "mediaResolution", e.target.value)}
              >
                {MEDIA_RESOLUTIONS.map((r) => (
                  <option key={r} value={r}>{r.replace("_", " ")}</option>
                ))}
              </select>
            </div>

            <div className="settings-row">
              <label className="settings-label" title="Max tool-loop iterations">
                Max Iterations
              </label>
              <input
                type="number"
                className="control-select"
                style={{ width: 70 }}
                min={1} max={50}
                value={estimateMaxIterations}
                onChange={(e) => setStageSetting("estimate", "maxIterations", parseInt(e.target.value))}
              />
            </div>

            <div className="settings-row">
              <label className="settings-label" title="Auto applies adaptive CLAHE + brightness normalization">
                Preprocess
              </label>
              <select
                className="control-select"
                value={estimatePreprocess}
                onChange={(e) => setStageSetting("estimate", "preprocess", e.target.value)}
              >
                <option value="auto">Auto (CLAHE)</option>
                <option value="none">None</option>
              </select>
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
              <label className="settings-label">Thinking</label>
              <select
                className="control-select"
                value={registerThinking}
                onChange={(e) => setStageSetting("register", "thinking", e.target.value)}
              >
                {THINKING_LEVELS.map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
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

      {/* Export */}
      <div className="section-block">
        <button
          className="btn-primary"
          onClick={exportResults}
          disabled={!canExport}
          style={{ background: canExport ? "var(--bg-surface)" : undefined, color: canExport ? "var(--accent)" : undefined, border: "1px solid var(--border)" }}
        >
          Export
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
          {/* Quick-affine preview status — only shown when relevant so we
            * don't clutter the panel in the common case. */}
          {selectedSlice?.quickAffineRunning && (
            <div className="quick-affine-status quick-affine-status--running">
              <span className="quick-affine-spinner" aria-hidden>◌</span>
              Aligning preview...
            </div>
          )}
          {!selectedSlice?.quickAffineRunning &&
            selectedSlice?.quickAffineWarpedPath &&
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

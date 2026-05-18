import { useEffect } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
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
  const registrationViewMode = useAppStore((s) => s.registrationViewMode);
  const setRegistrationViewMode = useAppStore((s) => s.setRegistrationViewMode);
  const borderSource = useAppStore((s) => s.borderSource);
  const setBorderSource = useAppStore((s) => s.setBorderSource);

  const brain = brains.find((b) => b.id === selectedBrainId);
  const selectedSlice = brain && selectedSliceIndex !== null
    ? brain.slices[selectedSliceIndex] ?? null
    : null;
  const selectedSliceName = selectedSlice?.name ?? null;
  const registrationResult = selectedSlice?.registrationResult ?? null;

  // Inverse-warp availability: status field is null/"ok" or "failed: ...".
  // The path can also be null/empty if the inverse step never produced output.
  const inverseStatus = registrationResult?.inverseWarpStatus ?? null;
  const inverseFailed =
    typeof inverseStatus === "string" && inverseStatus.startsWith("failed");
  const inverseOverlayPath = registrationResult?.sliceAtlasBorderOverlayPath ?? null;
  const hasInverseOverlay =
    !inverseFailed &&
    typeof inverseOverlayPath === "string" &&
    inverseOverlayPath.length > 0;

  // Forward overlay always exists when registrationResult is non-null
  // (Task 2/4 contract).
  const elastixForwardPath = registrationResult?.warpedBorderOverlayPath ?? null;
  const generatedForwardPath =
    registrationResult?.generatedBorderOverlayPath ?? null;
  const hasGeneratedForward =
    typeof generatedForwardPath === "string" && generatedForwardPath.length > 0;
  // Effective forward overlay obeys the border-source toggle, with a clean
  // fallback to elastix when generated isn't available (e.g. older runs that
  // predate the generated_border_overlay artifact).
  const forwardOverlayPath =
    borderSource === "generated" && hasGeneratedForward
      ? generatedForwardPath
      : elastixForwardPath;
  const hasForwardOverlay =
    typeof forwardOverlayPath === "string" && forwardOverlayPath.length > 0;

  // Decide which path drives the histology-panel image when registration is
  // available. If the user picked slice-to-atlas but the inverse is missing,
  // we fall back to the original slice (not to forward, so the toggle state
  // is visually distinguishable from "Atlas → Slice").
  let registrationImageSrc: string | null = null;
  if (registrationResult) {
    if (registrationViewMode === "atlas-to-slice" && hasForwardOverlay) {
      registrationImageSrc = convertFileSrc(forwardOverlayPath!);
    } else if (registrationViewMode === "slice-to-atlas" && hasInverseOverlay) {
      registrationImageSrc = convertFileSrc(inverseOverlayPath!);
    }
  }

  // Build composite on mount if volumes are ready
  useEffect(() => {
    if (volumesLoaded && !compositeDataUrl) {
      buildComposite();
    }
  }, [volumesLoaded, compositeDataUrl, buildComposite]);

  const channels = ["reference", ...additionalVolumeNames];

  // Show the failure-state pill the moment registration completes, not only
  // when the user toggles to slice-to-atlas — otherwise the disabled button
  // is a dead-end with no explanation.
  const inverseUnavailable =
    registrationResult !== null && !hasInverseOverlay;
  const inverseReason = inverseFailed && inverseStatus
    ? inverseStatus.replace(/^failed:\s*/, "")
    : null;

  return (
    <div className="split-view">
      {/* Left: histology slice (or registration overlay when available) */}
      <div className="split-panel">
        <div className="split-panel-label">
          {registrationResult
            ? registrationViewMode === "atlas-to-slice"
              ? "Atlas → Slice"
              : "Slice → Atlas"
            : "Histology"}
          {selectedSliceName && (
            <span className="split-panel-ap">{selectedSliceName}</span>
          )}
        </div>

        {/* Registration view-mode toggle — only shown post-registration. */}
        {registrationResult && (
          <div className="split-panel-controls">
            <div
              className="registration-view-toggle"
              role="group"
              aria-label="Registration view direction"
            >
              <button
                type="button"
                className={`registration-view-toggle-btn ${
                  registrationViewMode === "atlas-to-slice" ? "active" : ""
                }`}
                onClick={() => setRegistrationViewMode("atlas-to-slice")}
                aria-pressed={registrationViewMode === "atlas-to-slice"}
                title="Show atlas borders warped onto the original slice"
              >
                Atlas {"→"} Slice
              </button>
              <button
                type="button"
                className={`registration-view-toggle-btn ${
                  registrationViewMode === "slice-to-atlas" ? "active" : ""
                }`}
                onClick={() => setRegistrationViewMode("slice-to-atlas")}
                aria-pressed={registrationViewMode === "slice-to-atlas"}
                title={
                  hasInverseOverlay
                    ? "Show slice warped to atlas space, with atlas borders"
                    : inverseReason
                      ? `Inverse warp failed: ${inverseReason}`
                      : "Inverse warp was not produced for this registration"
                }
                disabled={!hasInverseOverlay}
              >
                Slice {"→"} Atlas
              </button>
            </div>

            {/* Border-source toggle. Only meaningful for atlas→slice direction;
             * slice→atlas always uses atlas-space borders (no concept of a
             * "generated" prediction warped backwards). */}
            {registrationViewMode === "atlas-to-slice" && (
              <div
                className="registration-view-toggle"
                role="group"
                aria-label="Border source"
              >
                <button
                  type="button"
                  className={`registration-view-toggle-btn ${
                    borderSource === "elastix" ? "active" : ""
                  }`}
                  onClick={() => setBorderSource("elastix")}
                  aria-pressed={borderSource === "elastix"}
                  title="Atlas borders warped onto the slice via Elastix"
                >
                  Elastix
                </button>
                <button
                  type="button"
                  className={`registration-view-toggle-btn ${
                    borderSource === "generated" ? "active" : ""
                  }`}
                  onClick={() => setBorderSource("generated")}
                  aria-pressed={borderSource === "generated"}
                  title={
                    hasGeneratedForward
                      ? "Borders extracted directly from the model's image-gen output (no Elastix step)"
                      : "Generated borders weren't produced for this registration (older run)"
                  }
                  disabled={!hasGeneratedForward}
                >
                  Generated
                </button>
              </div>
            )}

            {inverseUnavailable && (
              <span
                className="registration-view-toggle-warn"
                title={inverseReason ?? undefined}
              >
                {inverseFailed ? "Inverse warp failed" : "Inverse warp unavailable"}
              </span>
            )}
          </div>
        )}

        {sliceImageLoading ? (
          <div className="split-panel-empty">
            <span className="split-panel-empty-text">Loading...</span>
          </div>
        ) : registrationImageSrc ? (
          <div className="split-panel-image">
            <img
              src={registrationImageSrc}
              alt={
                registrationViewMode === "atlas-to-slice"
                  ? "Atlas borders warped to slice"
                  : "Slice warped to atlas with borders"
              }
              className="split-image"
            />
          </div>
        ) : selectedSliceImage ? (
          <div className="split-panel-image">
            <img
              src={`data:image/jpeg;base64,${selectedSliceImage}`}
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

/** Side-by-side 2D viewer. Browser variant pulls atlas layers via
 *  getCoronalSliceLayers and renders registration ImageBitmaps when present. */

import { useEffect, useRef, useState } from "react";
import { useAppStore } from "../stores/appStore";
import { getCoronalSliceLayers } from "../lib/browserCommands";

const DEMO_BRAIN_ROOT = `${import.meta.env.BASE_URL}demo_brain`;

function useBitmapUrl(bitmap: ImageBitmap | undefined | null): string | null {
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!bitmap) {
      setUrl(null);
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      setUrl(null);
      return;
    }
    ctx.drawImage(bitmap, 0, 0);
    let dataUrl: string | null = null;
    try {
      dataUrl = canvas.toDataURL("image/png");
    } catch {
      dataUrl = null;
    }
    setUrl(dataUrl);
    return () => setUrl(null);
  }, [bitmap]);

  return url;
}

export function SplitView() {
  const currentApMm = useAppStore((s) => s.currentApMm);
  const selectedSliceIndex = useAppStore((s) => s.selectedSliceIndex);
  const selectedBrainId = useAppStore((s) => s.selectedBrainId);
  const brains = useAppStore((s) => s.brains);
  const registrationViewMode = useAppStore((s) => s.registrationViewMode);
  const setRegistrationViewMode = useAppStore((s) => s.setRegistrationViewMode);
  const borderSource = useAppStore((s) => s.borderSource);
  const setBorderSource = useAppStore((s) => s.setBorderSource);
  const clahePreview = useAppStore((s) => s.clahePreview);

  const brain = brains.find((b) => b.id === selectedBrainId);
  const selectedSlice = brain && selectedSliceIndex !== null
    ? brain.slices[selectedSliceIndex] ?? null
    : null;
  const selectedSliceName = selectedSlice?.name ?? null;
  const registrationResult = selectedSlice?.registrationResult ?? null;

  const warpedAtlasUrl = useBitmapUrl(registrationResult?.warpedAtlas);
  const warpedBordersUrl = useBitmapUrl(registrationResult?.warpedBorders);
  const generatedRegionsUrl = useBitmapUrl(
    registrationResult?.generatedColoredRegions,
  );

  // Fetch atlas layers for the current AP so the right pane and the
  // border overlay are aligned with the slider.
  const [layers, setLayers] = useState<{
    sectionUrl: string;
    coloredUrl: string;
    bordersUrl: string;
  } | null>(null);
  const apRef = useRef(currentApMm);
  apRef.current = currentApMm;
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const l = await getCoronalSliceLayers(currentApMm);
        if (cancelled) return;
        setLayers({
          sectionUrl: l.sectionUrl,
          coloredUrl: l.coloredUrl,
          bordersUrl: l.bordersUrl,
        });
      } catch {
        if (!cancelled) setLayers(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [currentApMm]);

  // Show the CLAHE-equalized variant when the preview toggle is on AND the
  // slice has one (bundled demo brain does, uploads don't).
  const activeSlicePath = selectedSlice
    ? clahePreview && selectedSlice.clahePath
      ? selectedSlice.clahePath
      : selectedSlice.path
    : null;
  const sliceImageUrl = activeSlicePath
    ? `${DEMO_BRAIN_ROOT}/${activeSlicePath}`
    : null;

  // Pick which overlay drives the left pane.
  let registrationImageSrc: string | null = null;
  if (registrationResult) {
    if (registrationViewMode === "atlas-to-slice") {
      registrationImageSrc =
        borderSource === "generated" ? generatedRegionsUrl : warpedBordersUrl;
    } else {
      // Browser pipeline does not produce an inverse warp; show the warped
      // atlas reference at the slice's position.
      registrationImageSrc = warpedAtlasUrl;
    }
  }

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
              >
                Slice {"→"} Atlas
              </button>
            </div>

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
                  disabled={!generatedRegionsUrl}
                >
                  Generated
                </button>
              </div>
            )}
          </div>
        )}

        {registrationImageSrc ? (
          <div className="split-panel-image">
            <img
              src={registrationImageSrc}
              alt={
                registrationViewMode === "atlas-to-slice"
                  ? "Atlas warped to slice"
                  : "Atlas reference at slice position"
              }
              className="split-image"
            />
          </div>
        ) : sliceImageUrl ? (
          <div className="split-panel-image">
            <img
              src={sliceImageUrl}
              alt="Histology slice"
              className="split-image"
            />
          </div>
        ) : (
          <div className="split-panel-empty">
            <span className="split-panel-empty-text">
              {brain && brain.slices.length > 0
                ? "Select a slice from the list"
                : "Load the demo brain first"
              }
            </span>
          </div>
        )}
      </div>

      {/* Divider */}
      <div className="split-divider" />

      {/* Right: atlas section at current AP */}
      <div className="split-panel">
        <div className="split-panel-label">
          Atlas
          <span className="split-panel-ap">{currentApMm.toFixed(2)} mm</span>
        </div>

        {layers ? (
          <div className="split-panel-image">
            {/* Inner wrapper sizes itself to the letterboxed section image so
                the absolutely-positioned borders overlay lines up exactly.
                Stacking with inset:0 on the .split-panel-image directly
                stretches the borders to the full panel, breaking alignment. */}
            <div
              style={{
                position: "relative",
                display: "inline-block",
                maxWidth: "100%",
                maxHeight: "100%",
              }}
            >
              <img
                src={layers.sectionUrl}
                alt="Atlas reference"
                className="split-image"
                style={{ display: "block" }}
              />
              <img
                src={layers.bordersUrl}
                alt="Atlas borders"
                className="split-image"
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  height: "100%",
                  pointerEvents: "none",
                  mixBlendMode: "screen",
                }}
              />
            </div>
          </div>
        ) : (
          <div className="split-panel-empty">
            <span className="split-panel-empty-text">Loading atlas...</span>
          </div>
        )}
      </div>
    </div>
  );
}

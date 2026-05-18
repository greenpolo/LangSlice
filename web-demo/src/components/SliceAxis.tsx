import { useMemo } from "react";
import type { SliceInfo } from "../lib/types";
import { useAppStore } from "../stores/appStore";

export function SliceAxis({ slices }: { slices: SliceInfo[] }) {
  const atlasInfo = useAppStore((s) => s.atlasInfo);

  const ticks = useMemo(() => {
    if (slices.length === 0 || !atlasInfo) return [];
    const apRange = atlasInfo.ap_max_mm - atlasInfo.ap_min_mm;

    return slices.map((slice, i) => {
      let fraction: number;
      if (slice.apMm !== undefined) {
        fraction = (slice.apMm - atlasInfo.ap_min_mm) / apRange;
      } else {
        fraction = slices.length > 1 ? i / (slices.length - 1) : 0.5;
      }
      return { fraction, status: slice.status };
    });
  }, [slices, atlasInfo]);

  return (
    <div className="slice-axis">
      {/* Axis line */}
      <div className="slice-axis-track">
        {/* Tick marks */}
        {ticks.map((tick, i) => (
          <div
            key={i}
            className={`slice-axis-tick slice-axis-tick--${tick.status}`}
            style={{ left: `${tick.fraction * 100}%` }}
          />
        ))}
      </div>
      {/* Labels */}
      <div className="slice-axis-labels">
        <span>A</span>
        <span>P</span>
      </div>
    </div>
  );
}

import { useEffect, useRef } from "react";
import { listen } from "@tauri-apps/api/event";
import { useAppStore } from "../stores/appStore";

export function AgentPanel() {
  const { logs, pipelineStatus, pipelineError, pipelineRunning, addLog } = useAppStore();
  const logEndRef = useRef<HTMLDivElement>(null);
  // Tracks whether we're inside a `json.dumps(..., indent=2)` block that the
  // CLI prints at the end of estimate/register --json. Rust still captures
  // these lines for extract_json(), but the agent panel should not echo
  // hundreds of marker coordinates back at the user.
  const jsonDepthRef = useRef(0);

  // Listen for pipeline-log events from Rust sidecar
  useEffect(() => {
    const unlisten = listen<string>("pipeline-log", (event) => {
      const line = event.payload;
      const trimmed = line.trim();
      if (!trimmed) return;

      // Track {/} pairs only when they appear at column 0 (json.dumps with
      // indent=2 always closes the top-level object that way). Increment on
      // `{`, skip the line entirely; decrement on `}`, also skip; while we're
      // inside the block, swallow everything.
      if (trimmed === "{" && !line.startsWith(" ")) {
        jsonDepthRef.current += 1;
        return;
      }
      if (trimmed === "}" && !line.startsWith(" ")) {
        if (jsonDepthRef.current > 0) {
          jsonDepthRef.current -= 1;
          return;
        }
      }
      if (jsonDepthRef.current > 0) return;

      // Belt-and-braces: a single insanely long line is almost certainly an
      // Elastix coordinate dump or similar noise that slipped past the depth
      // tracker (e.g. a flat-printed JSON without indent).
      if (line.length > 500) return;

      addLog(line);
    });
    return () => { unlisten.then((fn) => fn()); };
  }, [addLog]);

  // Auto-scroll log area
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const statusLabel =
    pipelineStatus === "idle" ? "Ready" :
    pipelineStatus === "loading_atlas" ? "Loading atlas..." :
    pipelineStatus === "estimating" ? "Estimating AP..." :
    pipelineStatus === "registering" ? "Registering..." :
    pipelineStatus === "complete" ? "Complete" :
    pipelineStatus === "error" ? (pipelineError || "Error") :
    pipelineStatus;

  const dotClass =
    pipelineRunning ? "loading" :
    pipelineStatus === "complete" ? "success" :
    pipelineStatus === "error" ? "error" :
    "idle";

  return (
    <div className="panel-right">
      <div className="section-block">
        <div className="section-label">Status</div>
        <div className="status-indicator">
          <div className={`status-dot ${dotClass}`} />
          <span style={{ color: pipelineStatus === "error" ? "var(--error)" : "var(--text-secondary)" }}>
            {statusLabel}
          </span>
        </div>
      </div>

      <div className="section-block" style={{ flex: 1, display: "flex", flexDirection: "column" }}>
        <div className="section-label">Log</div>
        <div className="log-area">
          {logs.length === 0 && (
            <div className="log-entry">
              <span className="log-msg">Awaiting commands...</span>
            </div>
          )}
          {logs.map((msg, i) => (
            <div key={i} className="log-entry">
              <span className="log-msg">{msg}</span>
            </div>
          ))}
          <div ref={logEndRef} />
        </div>
      </div>
    </div>
  );
}

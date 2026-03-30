import { useEffect, useRef } from "react";
import { listen } from "@tauri-apps/api/event";
import { useAppStore } from "../stores/appStore";

export function AgentPanel() {
  const { logs, pipelineStatus, pipelineError, pipelineRunning, addLog } = useAppStore();
  const logEndRef = useRef<HTMLDivElement>(null);

  // Listen for pipeline-log events from Rust sidecar
  useEffect(() => {
    const unlisten = listen<string>("pipeline-log", (event) => {
      if (event.payload.trim()) {
        addLog(event.payload);
      }
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

import { useAppStore } from "../stores/appStore";

export function AgentPanel() {
  const { logs, pipelineStatus, pipelineError } = useAppStore();

  const statusLabel =
    pipelineStatus === "idle" ? "Ready" :
    pipelineStatus === "loading_atlas" ? "Loading atlas..." :
    pipelineStatus === "error" ? (pipelineError || "Error") :
    pipelineStatus;

  const dotClass =
    pipelineStatus === "idle" ? "idle" :
    pipelineStatus === "loading_atlas" ? "loading" :
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
        </div>
      </div>
    </div>
  );
}

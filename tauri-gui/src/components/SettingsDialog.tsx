import { useState, useEffect, useCallback, useRef } from "react";
import { listen } from "@tauri-apps/api/event";
import * as commands from "../lib/commands";
import type { CuratedModel } from "../lib/commands";

interface Props {
  open: boolean;
  onClose: () => void;
}

type Provider = "ai_studio" | "vertex_adc" | "vertex_api_key" | "ollama";
type SettingsTab = "google" | "vertex" | "local";
type OllamaStatusState = "ready" | "starting" | "not_installed" | "error";

interface PullProgress {
  status: string;
  completed_bytes?: number;
  total_bytes?: number;
  percent?: number;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

export function SettingsDialog({ open, onClose }: Props) {
  const [provider, setProvider] = useState<Provider>("ai_studio");
  const [aiStudioKey, setAiStudioKey] = useState("");
  const [vertexProject, setVertexProject] = useState("");
  const [vertexLocation, setVertexLocation] = useState("us-central1");
  const [vertexKey, setVertexKey] = useState("");
  const [envPath, setEnvPath] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  // Tab state
  const [activeTab, setActiveTab] = useState<SettingsTab>("google");

  // Ollama state
  const [ollamaState, setOllamaState] = useState<OllamaStatusState>("not_installed");
  const [ollamaVersion, setOllamaVersion] = useState<string | undefined>();
  const [ollamaPort, setOllamaPort] = useState<number | undefined>();
  const [ollamaError, setOllamaError] = useState<string | undefined>();
  const [curatedModels, setCuratedModels] = useState<CuratedModel[]>([]);
  const [pullingModels, setPullingModels] = useState<Record<string, PullProgress>>({});
  const [selectedOllamaModel, setSelectedOllamaModel] = useState<string>("");
  const unlistenRef = useRef<(() => void) | null>(null);

  // Load env on open
  useEffect(() => {
    if (!open) return;
    commands.readEnvFile().then((result) => {
      setEnvPath(result.path);
      const v = result.vars;
      const backend = (v.LANGSLICE_GENAI_BACKEND || "ai_studio").toLowerCase();

      if (backend === "ollama" || v.OPENAI_BASE_URL?.includes("localhost")) {
        setProvider("ollama");
        setActiveTab("local");
        setSelectedOllamaModel(v.OPENAI_MODEL || "");
      } else if (backend === "vertex_adc") {
        setProvider("vertex_adc");
        setActiveTab("vertex");
      } else if (backend === "vertex_api_key") {
        setProvider("vertex_api_key");
        setActiveTab("vertex");
      } else {
        setProvider("ai_studio");
        setActiveTab("google");
      }

      setAiStudioKey(v.GEMINI_API_KEY || "");
      setVertexProject(v.GOOGLE_CLOUD_PROJECT || "");
      setVertexLocation(v.GOOGLE_CLOUD_LOCATION || "us-central1");
      setVertexKey(v.GOOGLE_CLOUD_API_KEY || "");
      setSaved(false);
    });
  }, [open]);

  // Poll Ollama status when local tab is active
  const refreshOllama = useCallback(async () => {
    try {
      const status = await commands.ollamaStatus();
      setOllamaState(status.status);
      setOllamaVersion(status.version);
      setOllamaPort(status.port);
      setOllamaError(status.error);

      if (status.status === "ready") {
        const models = await commands.ollamaAvailableModels();
        setCuratedModels(models);
        // Auto-select the first recommended installed model if none selected
        if (!selectedOllamaModel) {
          const recommended = models.find((m) => m.recommended && m.installed);
          const firstInstalled = models.find((m) => m.installed);
          if (recommended) setSelectedOllamaModel(recommended.name);
          else if (firstInstalled) setSelectedOllamaModel(firstInstalled.name);
        }
      }
    } catch {
      setOllamaState("error");
      setOllamaError("Failed to query Ollama status");
    }
  }, [selectedOllamaModel]);

  useEffect(() => {
    if (!open || activeTab !== "local") return;
    refreshOllama();
  }, [open, activeTab, refreshOllama]);

  // Listen for pull progress events
  useEffect(() => {
    if (!open) return;

    let cancelled = false;

    listen<{
      model: string;
      status: string;
      completed_bytes?: number;
      total_bytes?: number;
      percent?: number;
    }>("ollama-pull-progress", (event) => {
      if (cancelled) return;
      const { model, status, completed_bytes, total_bytes, percent } = event.payload;

      if (status === "success") {
        setPullingModels((prev) => {
          const next = { ...prev };
          delete next[model];
          return next;
        });
        // Refresh the model list
        refreshOllama();
      } else if (status === "error") {
        setPullingModels((prev) => {
          const next = { ...prev };
          delete next[model];
          return next;
        });
      } else {
        setPullingModels((prev) => ({
          ...prev,
          [model]: { status, completed_bytes, total_bytes, percent },
        }));
      }
    }).then((unlisten) => {
      if (cancelled) {
        unlisten();
      } else {
        unlistenRef.current = unlisten;
      }
    });

    return () => {
      cancelled = true;
      if (unlistenRef.current) {
        unlistenRef.current();
        unlistenRef.current = null;
      }
    };
  }, [open, refreshOllama]);

  const handleStartOllama = async () => {
    setOllamaState("starting");
    try {
      await commands.ollamaStart();
      // Poll until ready
      const poll = setInterval(async () => {
        const status = await commands.ollamaStatus();
        if (status.status === "ready") {
          clearInterval(poll);
          refreshOllama();
        } else if (status.status === "error") {
          clearInterval(poll);
          setOllamaState("error");
          setOllamaError(status.error);
        }
      }, 1000);
    } catch (e) {
      setOllamaState("error");
      setOllamaError(e instanceof Error ? e.message : String(e));
    }
  };

  const handlePullModel = async (name: string) => {
    setPullingModels((prev) => ({
      ...prev,
      [name]: { status: "starting", percent: 0 },
    }));
    try {
      await commands.ollamaPullModel(name);
    } catch (e) {
      setPullingModels((prev) => {
        const next = { ...prev };
        delete next[name];
        return next;
      });
      console.error("Pull failed:", e);
    }
  };

  const handleDeleteModel = async (name: string) => {
    try {
      await commands.ollamaDeleteModel(name);
      refreshOllama();
    } catch (e) {
      console.error("Delete failed:", e);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    const vars: Record<string, string> = {};

    if (activeTab === "local" || provider === "ollama") {
      vars.LANGSLICE_GENAI_BACKEND = "ollama";
      vars.OPENAI_BASE_URL = `http://localhost:${ollamaPort || 11434}/v1`;
      vars.OPENAI_API_KEY = "ollama";
      vars.OPENAI_MODEL = selectedOllamaModel;
    } else {
      vars.LANGSLICE_GENAI_BACKEND = provider;
      if (provider === "ai_studio") {
        vars.GEMINI_API_KEY = aiStudioKey;
      } else if (provider === "vertex_adc") {
        vars.GOOGLE_CLOUD_PROJECT = vertexProject;
        vars.GOOGLE_CLOUD_LOCATION = vertexLocation;
      } else if (provider === "vertex_api_key") {
        vars.GOOGLE_CLOUD_PROJECT = vertexProject;
        vars.GOOGLE_CLOUD_LOCATION = vertexLocation;
        vars.GOOGLE_CLOUD_API_KEY = vertexKey;
      }
    }

    try {
      await commands.writeEnvFile(vars);
      setSaved(true);
      setTimeout(() => onClose(), 600);
    } catch (e) {
      console.error("Failed to save settings:", e);
    } finally {
      setSaving(false);
    }
  };

  if (!open) return null;

  // --- Styles ---

  const tabBarStyle: React.CSSProperties = {
    display: "flex",
    gap: "2px",
    padding: "3px",
    borderRadius: "var(--radius)",
    background: "var(--bg-deep)",
    border: "1px solid var(--border)",
    marginBottom: "12px",
  };

  const tabBtnBase: React.CSSProperties = {
    flex: 1,
    padding: "7px 12px",
    fontFamily: "var(--font-mono)",
    fontSize: "11px",
    fontWeight: 500,
    letterSpacing: "0.3px",
    border: "none",
    borderRadius: "4px",
    cursor: "pointer",
    transition: "all 0.15s",
  };

  const tabBtnInactive: React.CSSProperties = {
    ...tabBtnBase,
    color: "var(--text-dim)",
    background: "transparent",
  };

  const tabBtnActive: React.CSSProperties = {
    ...tabBtnBase,
    color: "var(--accent)",
    background: "var(--accent-dim)",
  };

  const statusBarStyle: React.CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    padding: "10px 12px",
    borderRadius: "var(--radius)",
    background: "rgba(255, 255, 255, 0.02)",
    border: "1px solid var(--border)",
    fontFamily: "var(--font-mono)",
    fontSize: "11px",
  };

  const dotStyle = (
    color: string,
    animate = false,
  ): React.CSSProperties => ({
    width: "6px",
    height: "6px",
    borderRadius: "50%",
    background: color,
    flexShrink: 0,
    animation: animate ? "pulse-dot 1.5s infinite" : "none",
  });

  const modelCardStyle: React.CSSProperties = {
    padding: "12px",
    borderRadius: "var(--radius)",
    background: "var(--bg-surface)",
    border: "1px solid var(--border)",
    display: "flex",
    flexDirection: "column",
    gap: "8px",
  };

  const badgeStyle = (color: string): React.CSSProperties => ({
    display: "inline-block",
    padding: "2px 6px",
    borderRadius: "3px",
    fontSize: "9px",
    fontFamily: "var(--font-mono)",
    fontWeight: 500,
    letterSpacing: "0.3px",
    background: color === "accent" ? "var(--accent-dim)" : "rgba(255, 255, 255, 0.04)",
    color: color === "accent" ? "var(--accent)" : "var(--text-dim)",
    border:
      color === "accent"
        ? "1px solid var(--accent-border)"
        : "1px solid var(--border)",
  });

  const progressBarTrack: React.CSSProperties = {
    height: "4px",
    background: "var(--bg-deep)",
    borderRadius: "2px",
    overflow: "hidden",
    width: "100%",
  };

  const smallBtnStyle = (
    variant: "primary" | "ghost" | "danger",
  ): React.CSSProperties => {
    const base: React.CSSProperties = {
      padding: "5px 12px",
      fontFamily: "var(--font-mono)",
      fontSize: "10px",
      fontWeight: 500,
      letterSpacing: "0.3px",
      border: "none",
      borderRadius: "var(--radius)",
      cursor: "pointer",
      transition: "all 0.15s",
      textTransform: "uppercase",
    };
    if (variant === "primary") {
      return {
        ...base,
        color: "var(--bg-deep)",
        background: "var(--accent)",
      };
    }
    if (variant === "danger") {
      return {
        ...base,
        color: "var(--error)",
        background: "transparent",
        border: "1px solid rgba(248, 113, 113, 0.2)",
      };
    }
    // ghost
    return {
      ...base,
      color: "var(--text-secondary)",
      background: "var(--bg-surface)",
      border: "1px solid var(--border)",
    };
  };

  // --- Render helpers ---

  const renderOllamaStatusBar = () => {
    if (ollamaState === "not_installed") {
      return (
        <div style={statusBarStyle}>
          <div style={dotStyle("var(--warning)")} />
          <span style={{ color: "var(--warning)", flex: 1 }}>Ollama not running</span>
          <button style={smallBtnStyle("primary")} onClick={handleStartOllama}>
            Start
          </button>
        </div>
      );
    }
    if (ollamaState === "starting") {
      return (
        <div style={statusBarStyle}>
          <div style={dotStyle("var(--accent)", true)} />
          <span style={{ color: "var(--text-secondary)", flex: 1 }}>
            Starting Ollama...
          </span>
        </div>
      );
    }
    if (ollamaState === "ready") {
      return (
        <div style={statusBarStyle}>
          <div style={dotStyle("var(--success)")} />
          <span style={{ color: "var(--success)", flex: 1 }}>
            Ollama running
            {ollamaPort && (
              <span style={{ color: "var(--text-dim)", marginLeft: "8px" }}>
                :{ollamaPort}
              </span>
            )}
            {ollamaVersion && (
              <span style={{ color: "var(--text-dim)", marginLeft: "8px" }}>
                v{ollamaVersion}
              </span>
            )}
          </span>
        </div>
      );
    }
    // error
    return (
      <div style={statusBarStyle}>
        <div style={dotStyle("var(--error)")} />
        <span
          style={{
            color: "var(--error)",
            flex: 1,
            fontSize: "10px",
            lineHeight: "1.4",
          }}
        >
          {ollamaError || "Ollama error"}
        </span>
        <button style={smallBtnStyle("ghost")} onClick={handleStartOllama}>
          Retry
        </button>
      </div>
    );
  };

  const renderModelCard = (model: CuratedModel) => {
    const pulling = pullingModels[model.name];
    const isSelected = selectedOllamaModel === model.name;

    return (
      <div
        key={model.name}
        style={{
          ...modelCardStyle,
          borderColor: isSelected
            ? "var(--accent-border)"
            : "var(--border)",
          background: isSelected
            ? "rgba(45, 212, 191, 0.04)"
            : "var(--bg-surface)",
          cursor: model.installed ? "pointer" : "default",
        }}
        onClick={() => {
          if (model.installed) setSelectedOllamaModel(model.name);
        }}
      >
        {/* Header row */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span
              style={{
                fontFamily: "var(--font-display)",
                fontSize: "13px",
                fontWeight: 600,
                color: "var(--text-primary)",
              }}
            >
              {model.display_name}
            </span>
            {model.recommended && (
              <span style={badgeStyle("accent")}>recommended</span>
            )}
            {model.installed && isSelected && (
              <span style={badgeStyle("accent")}>selected</span>
            )}
          </div>
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "10px",
              color: "var(--text-dim)",
            }}
          >
            {model.installed && model.local_size
              ? formatBytes(model.local_size)
              : `${model.size_gb} GB`}
          </span>
        </div>

        {/* Description */}
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "10px",
            color: "var(--text-dim)",
            lineHeight: "1.5",
          }}
        >
          {model.description}
        </div>

        {/* Badges row */}
        <div style={{ display: "flex", gap: "4px", flexWrap: "wrap", alignItems: "center" }}>
          {model.capabilities.map((cap) => (
            <span key={cap} style={badgeStyle("dim")}>
              {cap}
            </span>
          ))}
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "9px",
              color: "var(--text-dim)",
              marginLeft: "auto",
            }}
          >
            {model.min_vram_gb} GB VRAM min
          </span>
        </div>

        {/* Action row */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "flex-end",
            gap: "8px",
            marginTop: "2px",
          }}
        >
          {pulling ? (
            // Downloading state
            <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "4px" }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontFamily: "var(--font-mono)",
                  fontSize: "9px",
                  color: "var(--text-dim)",
                }}
              >
                <span>{pulling.status}</span>
                <span>
                  {pulling.percent != null
                    ? `${pulling.percent.toFixed(0)}%`
                    : ""}
                  {pulling.completed_bytes != null && pulling.total_bytes != null && (
                    <span style={{ marginLeft: "6px" }}>
                      {formatBytes(pulling.completed_bytes)} / {formatBytes(pulling.total_bytes)}
                    </span>
                  )}
                </span>
              </div>
              <div style={progressBarTrack}>
                <div
                  style={{
                    height: "100%",
                    width: `${pulling.percent ?? 0}%`,
                    background: "var(--accent)",
                    borderRadius: "2px",
                    transition: "width 0.3s ease",
                  }}
                />
              </div>
            </div>
          ) : model.installed ? (
            // Installed state
            <>
              <span
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "10px",
                  color: "var(--success)",
                  display: "flex",
                  alignItems: "center",
                  gap: "4px",
                  flex: 1,
                }}
              >
                <svg
                  width="12"
                  height="12"
                  viewBox="0 0 12 12"
                  fill="none"
                  style={{ flexShrink: 0 }}
                >
                  <path
                    d="M2.5 6L5 8.5L9.5 3.5"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
                Installed
              </span>
              <button
                style={smallBtnStyle("danger")}
                onClick={(e) => {
                  e.stopPropagation();
                  handleDeleteModel(model.name);
                }}
              >
                Delete
              </button>
            </>
          ) : (
            // Not installed state
            <button
              style={smallBtnStyle("primary")}
              onClick={(e) => {
                e.stopPropagation();
                handlePullModel(model.name);
              }}
            >
              Install
            </button>
          )}
        </div>
      </div>
    );
  };

  const renderGoogleTab = () => (
    <>
      {/* AI Studio key */}
      <div className="settings-row">
        <label className="settings-label">API Key</label>
        <input
          type="password"
          className="control-select"
          placeholder="AIzaSy..."
          value={aiStudioKey}
          onChange={(e) => setAiStudioKey(e.target.value)}
        />
      </div>
      <div className="dialog-info">
        Uses Google AI Studio free tier. Requires a Gemini API key.
      </div>
    </>
  );

  const renderVertexTab = () => (
    <>
      {/* Sub-selector: ADC vs API Key */}
      <div className="settings-row">
        <label className="settings-label">Auth</label>
        <select
          className="control-select"
          value={provider === "vertex_api_key" ? "vertex_api_key" : "vertex_adc"}
          onChange={(e) => setProvider(e.target.value as Provider)}
        >
          <option value="vertex_adc">Application Default Credentials</option>
          <option value="vertex_api_key">API Key</option>
        </select>
      </div>

      <div className="settings-row">
        <label className="settings-label">Project</label>
        <input
          type="text"
          className="control-select"
          placeholder="my-gcp-project"
          value={vertexProject}
          onChange={(e) => setVertexProject(e.target.value)}
        />
      </div>
      <div className="settings-row">
        <label className="settings-label">Location</label>
        <input
          type="text"
          className="control-select"
          placeholder="us-central1"
          value={vertexLocation}
          onChange={(e) => setVertexLocation(e.target.value)}
        />
      </div>

      {provider === "vertex_api_key" && (
        <div className="settings-row">
          <label className="settings-label">API Key</label>
          <input
            type="password"
            className="control-select"
            placeholder="AIzaSy..."
            value={vertexKey}
            onChange={(e) => setVertexKey(e.target.value)}
          />
        </div>
      )}

      <div className="dialog-info">
        {provider === "vertex_adc"
          ? "Uses Vertex AI with Application Default Credentials. Run `gcloud auth application-default login` first."
          : "Uses Vertex AI with a specific API key tied to a GCP project."}
      </div>
    </>
  );

  const renderLocalTab = () => (
    <>
      {renderOllamaStatusBar()}

      {ollamaState === "ready" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "8px", marginTop: "4px" }}>
          {curatedModels.length === 0 ? (
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "11px",
                color: "var(--text-dim)",
                textAlign: "center",
                padding: "20px 0",
              }}
            >
              Loading models...
            </div>
          ) : (
            curatedModels.map(renderModelCard)
          )}
        </div>
      )}

      {ollamaState === "not_installed" && (
        <div className="dialog-info">
          Ollama enables running models locally. Start Ollama to manage local
          models.
        </div>
      )}
    </>
  );

  return (
    <div className="dialog-overlay" onClick={onClose}>
      <div
        className="dialog"
        style={{ width: "480px" }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="dialog-header">
          <span className="dialog-title">Settings</span>
          <button className="dialog-close" onClick={onClose}>
            &times;
          </button>
        </div>

        <div className="dialog-body">
          {/* Tab bar */}
          <div style={tabBarStyle}>
            <button
              style={activeTab === "google" ? tabBtnActive : tabBtnInactive}
              onClick={() => {
                setActiveTab("google");
                setProvider("ai_studio");
              }}
            >
              Google AI Studio
            </button>
            <button
              style={activeTab === "vertex" ? tabBtnActive : tabBtnInactive}
              onClick={() => {
                setActiveTab("vertex");
                if (provider !== "vertex_adc" && provider !== "vertex_api_key") {
                  setProvider("vertex_adc");
                }
              }}
            >
              Vertex AI
            </button>
            <button
              style={activeTab === "local" ? tabBtnActive : tabBtnInactive}
              onClick={() => {
                setActiveTab("local");
                setProvider("ollama");
              }}
            >
              Local Models
            </button>
          </div>

          {/* Tab content */}
          {activeTab === "google" && renderGoogleTab()}
          {activeTab === "vertex" && renderVertexTab()}
          {activeTab === "local" && renderLocalTab()}

          <div className="dialog-env-path">{envPath}</div>
        </div>

        <div className="dialog-footer">
          <button
            className="btn-primary"
            style={{
              width: "auto",
              padding: "8px 24px",
              background: "var(--bg-surface)",
              color: "var(--text-secondary)",
              border: "1px solid var(--border)",
            }}
            onClick={onClose}
          >
            Cancel
          </button>
          <button
            className="btn-primary"
            style={{ width: "auto", padding: "8px 24px" }}
            onClick={handleSave}
            disabled={saving || (activeTab === "local" && !selectedOllamaModel)}
          >
            {saved ? "Saved" : saving ? "Saving..." : "Save"}
          </button>
        </div>
      </div>
    </div>
  );
}

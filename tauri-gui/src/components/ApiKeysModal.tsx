import { useEffect, useState } from "react";
import { useAppStore } from "../stores/appStore";
import * as commands from "../lib/commands";

type Provider = "ai_studio" | "vertex_adc" | "vertex_api_key" | "openrouter";
type Tab = "google" | "vertex" | "openrouter";

/** Cloud-API-key settings — the Local Models tab has been split out into its
 * own modal. This component owns the read/write of the `.env` file for the
 * cloud-only backends. */
export function ApiKeysModal() {
  const open = useAppStore((s) => s.apiKeysOpen);
  const close = useAppStore((s) => s.closeApiKeys);

  const [provider, setProvider] = useState<Provider>("ai_studio");
  const [activeTab, setActiveTab] = useState<Tab>("google");

  const [aiStudioKey, setAiStudioKey] = useState("");
  const [vertexProject, setVertexProject] = useState("");
  const [vertexLocation, setVertexLocation] = useState("us-central1");
  const [vertexKey, setVertexKey] = useState("");
  const [openRouterKey, setOpenRouterKey] = useState("");
  const [openRouterModel, setOpenRouterModel] = useState("google/gemma-4-31b-it");

  const [envPath, setEnvPath] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (!open) return;
    commands.readEnvFile().then((result) => {
      setEnvPath(result.path);
      const v = result.vars;
      const backend = (v.LANGSLICE_GENAI_BACKEND || "ai_studio").toLowerCase();

      if (backend === "openrouter" || v.OPENAI_BASE_URL?.includes("openrouter.ai")) {
        setProvider("openrouter");
        setActiveTab("openrouter");
        setOpenRouterKey(v.OPENAI_API_KEY || "");
        setOpenRouterModel(v.OPENAI_MODEL || "google/gemma-4-31b-it");
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

  const handleSave = async () => {
    setSaving(true);
    const vars: Record<string, string> = {};

    if (activeTab === "openrouter" || provider === "openrouter") {
      vars.LANGSLICE_GENAI_BACKEND = "openrouter";
      vars.OPENAI_BASE_URL = "https://openrouter.ai/api/v1";
      vars.OPENAI_API_KEY = openRouterKey;
      vars.OPENAI_MODEL = openRouterModel;
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
      setTimeout(() => close(), 600);
    } catch (e) {
      console.error("Failed to save API keys:", e);
    } finally {
      setSaving(false);
    }
  };

  if (!open) return null;

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

  return (
    <div
      className="dialog-overlay"
      onClick={(e) => {
        if (e.target === e.currentTarget) close();
      }}
    >
      <div className="dialog" style={{ width: "480px" }}>
        <div className="dialog-header">
          <span className="dialog-title">Cloud API keys</span>
          <button
            type="button"
            className="dialog-close"
            onClick={close}
            aria-label="Close"
          >
            {"×"}
          </button>
        </div>

        <div className="dialog-body">
          <div style={tabBarStyle}>
            <button
              type="button"
              style={activeTab === "google" ? tabBtnActive : tabBtnInactive}
              onClick={() => {
                setActiveTab("google");
                setProvider("ai_studio");
              }}
            >
              Google AI Studio
            </button>
            <button
              type="button"
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
              type="button"
              style={activeTab === "openrouter" ? tabBtnActive : tabBtnInactive}
              onClick={() => {
                setActiveTab("openrouter");
                setProvider("openrouter");
              }}
            >
              OpenRouter
            </button>
          </div>

          {activeTab === "google" && (
            <>
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
          )}

          {activeTab === "vertex" && (
            <>
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
          )}

          {activeTab === "openrouter" && (
            <>
              <div className="settings-row">
                <label className="settings-label">API Key</label>
                <input
                  type="password"
                  className="control-select"
                  placeholder="sk-or-..."
                  value={openRouterKey}
                  onChange={(e) => setOpenRouterKey(e.target.value)}
                />
              </div>
              <div className="settings-row">
                <label className="settings-label">Model</label>
                <select
                  className="control-select"
                  value={openRouterModel}
                  onChange={(e) => setOpenRouterModel(e.target.value)}
                >
                  <option value="google/gemma-4-31b-it">Gemma 4 31B ($0.13/M)</option>
                  <option value="google/gemma-4-26b-a4b-it">Gemma 4 26B MoE (free)</option>
                  <option value="openai/gpt-4.1">GPT-4.1</option>
                  <option value="openai/gpt-4.1-mini">GPT-4.1 Mini</option>
                  <option value="openai/gpt-4.1-nano">GPT-4.1 Nano</option>
                </select>
              </div>
              <div className="dialog-info">
                Routes to many model providers via{" "}
                <span style={{ color: "var(--accent)" }}>openrouter.ai</span>.
                Uses the OpenAI-compatible Responses API. Get a key at
                openrouter.ai/keys.
              </div>
            </>
          )}

          <div className="dialog-env-path">{envPath}</div>
        </div>

        <div className="dialog-footer">
          <button type="button" className="btn-secondary" onClick={close}>
            Cancel
          </button>
          <button
            type="button"
            className="btn-primary"
            style={{ width: "auto", padding: "8px 24px" }}
            onClick={handleSave}
            disabled={saving || (activeTab === "openrouter" && !openRouterKey)}
          >
            {saved ? "Saved" : saving ? "Saving..." : "Save"}
          </button>
        </div>
      </div>
    </div>
  );
}

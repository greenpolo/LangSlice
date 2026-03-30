import { useState, useEffect } from "react";
import * as commands from "../lib/commands";

interface Props {
  open: boolean;
  onClose: () => void;
}

type Provider = "ai_studio" | "vertex_adc" | "vertex_api_key";

export function SettingsDialog({ open, onClose }: Props) {
  const [provider, setProvider] = useState<Provider>("ai_studio");
  const [aiStudioKey, setAiStudioKey] = useState("");
  const [vertexProject, setVertexProject] = useState("");
  const [vertexLocation, setVertexLocation] = useState("us-central1");
  const [vertexKey, setVertexKey] = useState("");
  const [envPath, setEnvPath] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (!open) return;
    commands.readEnvFile().then((result) => {
      setEnvPath(result.path);
      const v = result.vars;
      const backend = (v.LANGSLICE_GENAI_BACKEND || "ai_studio").toLowerCase();
      if (backend === "vertex_adc") setProvider("vertex_adc");
      else if (backend === "vertex_api_key") setProvider("vertex_api_key");
      else setProvider("ai_studio");

      setAiStudioKey(v.GEMINI_API_KEY || "");
      setVertexProject(v.GOOGLE_CLOUD_PROJECT || "");
      setVertexLocation(v.GOOGLE_CLOUD_LOCATION || "us-central1");
      setVertexKey(v.GOOGLE_CLOUD_API_KEY || "");
      setSaved(false);
    });
  }, [open]);

  const handleSave = async () => {
    setSaving(true);
    const vars: Record<string, string> = {
      LANGSLICE_GENAI_BACKEND: provider,
    };

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

  return (
    <div className="dialog-overlay" onClick={onClose}>
      <div className="dialog" onClick={(e) => e.stopPropagation()}>
        <div className="dialog-header">
          <span className="dialog-title">Settings</span>
          <button className="dialog-close" onClick={onClose}>&times;</button>
        </div>

        <div className="dialog-body">
          {/* Provider */}
          <div className="settings-row">
            <label className="settings-label">Provider</label>
            <select
              className="control-select"
              value={provider}
              onChange={(e) => setProvider(e.target.value as Provider)}
            >
              <option value="ai_studio">Google AI Studio</option>
              <option value="vertex_adc">Vertex AI (ADC)</option>
              <option value="vertex_api_key">Vertex AI (API Key)</option>
            </select>
          </div>

          {/* AI Studio key */}
          {provider === "ai_studio" && (
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
          )}

          {/* Vertex Project + Location */}
          {(provider === "vertex_adc" || provider === "vertex_api_key") && (
            <>
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
            </>
          )}

          {/* Vertex API Key */}
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

          {/* Info text */}
          <div className="dialog-info">
            {provider === "ai_studio" && "Uses Google AI Studio free tier. Requires a Gemini API key."}
            {provider === "vertex_adc" && "Uses Vertex AI with Application Default Credentials. Run `gcloud auth application-default login` first."}
            {provider === "vertex_api_key" && "Uses Vertex AI with a specific API key tied to a GCP project."}
          </div>

          <div className="dialog-env-path">
            {envPath}
          </div>
        </div>

        <div className="dialog-footer">
          <button
            className="btn-primary"
            style={{ width: "auto", padding: "8px 24px", background: "var(--bg-surface)", color: "var(--text-secondary)", border: "1px solid var(--border)" }}
            onClick={onClose}
          >
            Cancel
          </button>
          <button
            className="btn-primary"
            style={{ width: "auto", padding: "8px 24px" }}
            onClick={handleSave}
            disabled={saving}
          >
            {saved ? "Saved" : saving ? "Saving..." : "Save"}
          </button>
        </div>
      </div>
    </div>
  );
}

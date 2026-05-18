/** API Keys modal. Browser demo: only GEMINI_API_KEY is needed (image-gen
 *  registration). Persists via browserCommands -> localStorage. */

import { useEffect, useState } from "react";
import { useAppStore } from "../stores/appStore";
import * as commands from "../lib/browserCommands";

export function ApiKeysModal() {
  const open = useAppStore((s) => s.apiKeysOpen);
  const close = useAppStore((s) => s.closeApiKeys);

  const [aiStudioKey, setAiStudioKey] = useState("");
  const [envPath, setEnvPath] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (!open) return;
    commands.readEnvFile().then((result) => {
      setEnvPath(result.path);
      setAiStudioKey(result.vars.GEMINI_API_KEY || "");
      setSaved(false);
    });
  }, [open]);

  const handleSave = async () => {
    setSaving(true);
    try {
      const existing = await commands.readEnvFile();
      const vars: Record<string, string> = { ...existing.vars };
      vars.GEMINI_API_KEY = aiStudioKey;
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

  return (
    <div
      className="dialog-overlay"
      onClick={(e) => {
        if (e.target === e.currentTarget) close();
      }}
    >
      <div className="dialog" style={{ width: "480px" }}>
        <div className="dialog-header">
          <span className="dialog-title">Gemini API key</span>
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
            Used for the image-generation registration step (Nano Banana).
            Estimation runs on the local <code>litert-lm</code> sidecar and
            does not need an API key.
          </div>

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
            disabled={saving}
          >
            {saved ? "Saved" : saving ? "Saving..." : "Save"}
          </button>
        </div>
      </div>
    </div>
  );
}

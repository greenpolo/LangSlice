import { useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { useAppStore } from "../stores/appStore";
import * as commands from "../lib/commands";
import type {
  CustomEndpoint,
  LocalEngineStatus,
  LocalModel,
} from "../lib/types";
import type { CuratedModel } from "../lib/commands";

// Placeholder Hugging Face URL — replace once the langslice-gemma E4B
// weights are uploaded.
const LANGSLICE_GEMMA_HF = "https://huggingface.co/langslice/langslice-gemma-4-e4b";

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

/** Top-bar Local Models modal.
 *
 * Layout:
 *   - Pinned LangSlice-Gemma row (HF link).
 *   - One section per auto-probed engine (Ollama, LM Studio, llama-server,
 *     vLLM, Jan), then any user-added custom endpoints.
 *   - "Add custom endpoint" inline form.
 *   - Refresh + Done in the footer.
 *
 * Engines that are reachable expand to show their model list with a
 * "Use for Estimate" button per row. Ollama gets extra affordances
 * (Start server, curated install) since its lifecycle is managed by us. */
export function LocalModelsModal() {
  const open = useAppStore((s) => s.localModelsOpen);
  const close = useAppStore((s) => s.closeLocalModels);
  const engines = useAppStore((s) => s.localEngines);
  const probing = useAppStore((s) => s.localEnginesProbing);
  const probe = useAppStore((s) => s.probeLocalEngines);
  const customEndpoints = useAppStore((s) => s.customEndpoints);
  const addCustomEndpoint = useAppStore((s) => s.addCustomEndpoint);
  const removeCustomEndpoint = useAppStore((s) => s.removeCustomEndpoint);
  const estimateModel = useAppStore((s) => s.estimateModel);
  const estimateEndpoint = useAppStore((s) => s.estimateEndpoint);
  const setEstimateModelChoice = useAppStore((s) => s.setEstimateModelChoice);

  if (!open) return null;

  return (
    <div
      className="dialog-overlay"
      onClick={(e) => {
        if (e.target === e.currentTarget) close();
      }}
    >
      <div className="dialog atlas-manager-dialog" style={{ width: "720px" }}>
        <div className="dialog-header">
          <span className="dialog-title">Local models</span>
          <button
            type="button"
            className="dialog-close"
            onClick={close}
            aria-label="Close"
          >
            {"×"}
          </button>
        </div>

        <div className="atlas-manager-body" style={{ maxHeight: "70vh" }}>
          <LangSliceGemmaRow />

          {probing && engines.length === 0 ? (
            <div className="atlas-manager-empty">Probing engines…</div>
          ) : (
            engines.map((engine) => (
              <EngineSection
                key={`${engine.name}-${engine.port}-${engine.endpoint}`}
                engine={engine}
                selectedModelId={estimateModel}
                selectedEndpoint={estimateEndpoint}
                onUseModel={(model) =>
                  setEstimateModelChoice(model.id, model.endpoint)
                }
                onRemoveCustom={
                  customEndpoints.some((ep) => ep.url === engine.endpoint)
                    ? () => removeCustomEndpoint(engine.endpoint)
                    : null
                }
              />
            ))
          )}

          <CustomEndpointForm
            existingUrls={customEndpoints.map((e) => e.url)}
            onAdd={addCustomEndpoint}
          />
        </div>

        <div className="dialog-footer">
          <button
            type="button"
            className="btn-secondary"
            onClick={() => probe()}
            disabled={probing}
          >
            {probing ? "Refreshing…" : "Refresh"}
          </button>
          <button type="button" className="btn-primary" style={{ width: "auto", padding: "8px 24px" }} onClick={close}>
            Done
          </button>
        </div>
      </div>
    </div>
  );
}

/** Pinned "we are building this model" row at the top of the list. Until
 * the weights are uploaded, the HF link points to a placeholder URL. */
function LangSliceGemmaRow() {
  return (
    <div className="atlas-row atlas-row-focus">
      <div className="atlas-row-main">
        <span className="atlas-row-name">LangSlice-Gemma E4B</span>
        <span className="atlas-row-status">
          Custom fine-tune of Gemma 4 E4B for AP estimation — coming soon
        </span>
      </div>
      <div className="atlas-row-action">
        <a
          className="btn-secondary atlas-row-download"
          href={LANGSLICE_GEMMA_HF}
          target="_blank"
          rel="noopener noreferrer"
          title="Open the LangSlice-Gemma model card on Hugging Face"
          style={{ textDecoration: "none", display: "inline-block", textAlign: "center" }}
        >
          Hugging Face →
        </a>
      </div>
    </div>
  );
}

interface EngineSectionProps {
  engine: LocalEngineStatus;
  selectedModelId: string;
  selectedEndpoint: string | null;
  onUseModel: (model: LocalModel) => void;
  onRemoveCustom: (() => void) | null;
}

/** One engine row + collapsible model list. Ollama gets extra affordances
 * (start server when not running, curated install panel) because its
 * lifecycle is managed by this app. */
function EngineSection({
  engine,
  selectedModelId,
  selectedEndpoint,
  onUseModel,
  onRemoveCustom,
}: EngineSectionProps) {
  const [expanded, setExpanded] = useState(engine.reachable && engine.models.length > 0);
  const isOllama = engine.name === "Ollama";

  const statusText = engine.reachable
    ? `Detected · ${engine.models.length} model${engine.models.length === 1 ? "" : "s"}`
    : engine.error
      ? `Not running (${engine.error})`
      : "Not running";

  return (
    <div className="local-engine-section">
      <div
        className="local-engine-row"
        role="button"
        tabIndex={0}
        onClick={() => setExpanded((v) => !v)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            setExpanded((v) => !v);
          }
        }}
      >
        <div className="local-engine-row-main">
          <span className="local-engine-row-name">{engine.name}</span>
          <span
            className={`local-engine-row-status ${engine.reachable ? "ok" : "off"}`}
          >
            {statusText}
          </span>
          <span className="local-engine-row-endpoint">{engine.endpoint}</span>
        </div>
        <div className="local-engine-row-action">
          {onRemoveCustom && (
            <button
              type="button"
              className="btn-secondary"
              style={{ width: "auto", padding: "4px 10px" }}
              onClick={(e) => {
                e.stopPropagation();
                onRemoveCustom();
              }}
            >
              Remove
            </button>
          )}
          <span className="local-engine-row-chevron">
            {expanded ? "▾" : "▸"}
          </span>
        </div>
      </div>

      {expanded && (
        <div className="local-engine-body">
          {isOllama && !engine.reachable && <OllamaSpawnBar />}

          {engine.reachable && engine.models.length === 0 && (
            <div className="local-engine-hint">
              {engine.name === "LM Studio"
                ? "LM Studio only lists currently-loaded models. Load a model in LM Studio's UI to see it here."
                : "Engine is reachable but reports zero models."}
            </div>
          )}

          {engine.models.map((m) => {
            const isSelected =
              selectedModelId === m.id && selectedEndpoint === m.endpoint;
            return (
              <div key={m.id} className="local-model-row">
                <span className="local-model-id">{m.id}</span>
                <button
                  type="button"
                  className={isSelected ? "btn-primary" : "btn-secondary"}
                  style={{ width: "auto", padding: "4px 12px" }}
                  onClick={() => onUseModel(m)}
                  disabled={isSelected}
                >
                  {isSelected ? "Selected" : "Use for Estimate"}
                </button>
              </div>
            );
          })}

          {isOllama && engine.reachable && <OllamaCuratedInstall />}
        </div>
      )}
    </div>
  );
}

/** Spawn-Ollama affordance shown when the Ollama engine row is expanded
 * but the daemon isn't responding to the probe. */
function OllamaSpawnBar() {
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const probe = useAppStore((s) => s.probeLocalEngines);

  const handleStart = async () => {
    setError(null);
    setStarting(true);
    try {
      await commands.ollamaStart();
      // Poll briefly until the probe reflects ready state.
      for (let i = 0; i < 20; i++) {
        await new Promise((r) => setTimeout(r, 500));
        const status = await commands.ollamaStatus();
        if (status.status === "ready") break;
        if (status.status === "error") {
          setError(status.error ?? "Ollama failed to start");
          break;
        }
      }
      await probe();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setStarting(false);
    }
  };

  return (
    <div className="local-engine-hint" style={{ display: "flex", gap: "12px", alignItems: "center" }}>
      <span style={{ flex: 1 }}>
        {error ? `Failed to start: ${error}` : "Start the Ollama daemon to detect installed models."}
      </span>
      <button
        type="button"
        className="btn-primary"
        style={{ width: "auto", padding: "5px 14px" }}
        onClick={handleStart}
        disabled={starting}
      >
        {starting ? "Starting…" : "Start Ollama"}
      </button>
    </div>
  );
}

/** Curated Ollama install panel — lets the reviewer pull a langslice
 * favorite without leaving the GUI. Streams `ollama-pull-progress`. */
function OllamaCuratedInstall() {
  const [models, setModels] = useState<CuratedModel[]>([]);
  const [pulling, setPulling] = useState<Record<string, PullProgress>>({});
  const probe = useAppStore((s) => s.probeLocalEngines);
  const unlistenRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    commands.ollamaAvailableModels().then(setModels).catch(() => setModels([]));
  }, []);

  useEffect(() => {
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
        setPulling((prev) => {
          const next = { ...prev };
          delete next[model];
          return next;
        });
        commands.ollamaAvailableModels().then(setModels);
        probe();
      } else if (status === "error") {
        setPulling((prev) => {
          const next = { ...prev };
          delete next[model];
          return next;
        });
      } else {
        setPulling((prev) => ({
          ...prev,
          [model]: { status, completed_bytes, total_bytes, percent },
        }));
      }
    }).then((unlisten) => {
      if (cancelled) unlisten();
      else unlistenRef.current = unlisten;
    });

    return () => {
      cancelled = true;
      if (unlistenRef.current) {
        unlistenRef.current();
        unlistenRef.current = null;
      }
    };
  }, [probe]);

  const handlePull = async (name: string) => {
    setPulling((prev) => ({ ...prev, [name]: { status: "starting", percent: 0 } }));
    try {
      await commands.ollamaPullModel(name);
    } catch (e) {
      setPulling((prev) => {
        const next = { ...prev };
        delete next[name];
        return next;
      });
      console.error("Pull failed:", e);
    }
  };

  if (models.length === 0) return null;

  return (
    <details className="local-engine-quickinstall">
      <summary>Quick install (curated Ollama models)</summary>
      <div style={{ display: "flex", flexDirection: "column", gap: "6px", marginTop: "8px" }}>
        {models.map((m) => {
          const p = pulling[m.name];
          return (
            <div key={m.name} className="local-model-row">
              <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
                <span className="local-model-id">{m.display_name}</span>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--text-dim)" }}>
                  {m.description} · {m.size_gb} GB · {m.min_vram_gb} GB VRAM min
                </span>
              </div>
              {m.installed ? (
                <span style={{ fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--accent)" }}>
                  ✓ Installed
                </span>
              ) : p ? (
                <div style={{ width: 160 }}>
                  <div className="atlas-row-progress">
                    <div
                      className="atlas-row-progress-fill"
                      style={{ width: `${p.percent ?? 0}%` }}
                    />
                  </div>
                  <span style={{ fontSize: "9px", color: "var(--text-dim)" }}>
                    {p.percent != null && `${p.percent.toFixed(0)}%`}
                    {p.completed_bytes != null && p.total_bytes != null &&
                      ` (${formatBytes(p.completed_bytes)} / ${formatBytes(p.total_bytes)})`}
                  </span>
                </div>
              ) : (
                <button
                  type="button"
                  className="btn-secondary"
                  style={{ width: "auto", padding: "4px 12px" }}
                  onClick={() => handlePull(m.name)}
                >
                  Install
                </button>
              )}
            </div>
          );
        })}
      </div>
    </details>
  );
}

interface CustomEndpointFormProps {
  existingUrls: string[];
  onAdd: (ep: CustomEndpoint) => Promise<void>;
}

function CustomEndpointForm({ existingUrls, onAdd }: CustomEndpointFormProps) {
  const [label, setLabel] = useState("");
  const [url, setUrl] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    setError(null);
    const trimmedUrl = url.trim().replace(/\/$/, "");
    if (!trimmedUrl) {
      setError("URL is required.");
      return;
    }
    if (existingUrls.includes(trimmedUrl)) {
      setError("That URL is already in the list.");
      return;
    }
    setSubmitting(true);
    try {
      await onAdd({
        label: label.trim() || trimmedUrl,
        url: trimmedUrl,
        apiKey: apiKey.trim() || undefined,
      });
      setLabel("");
      setUrl("");
      setApiKey("");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="local-engine-section custom-endpoint-form">
      <div className="local-engine-row" style={{ cursor: "default" }}>
        <div className="local-engine-row-main">
          <span className="local-engine-row-name">Add custom endpoint</span>
          <span className="local-engine-row-status off">
            Point at a non-default port or remote OpenAI-compatible server.
          </span>
        </div>
      </div>
      <div className="local-engine-body">
        <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
          <input
            type="text"
            className="control-select"
            placeholder="Label (e.g. My llama-server)"
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            style={{ flex: "1 1 180px" }}
          />
          <input
            type="text"
            className="control-select"
            placeholder="http://localhost:9000/v1"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            style={{ flex: "1 1 260px" }}
          />
          <input
            type="password"
            className="control-select"
            placeholder="API key (optional)"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            style={{ flex: "1 1 160px" }}
          />
          <button
            type="button"
            className="btn-primary"
            style={{ width: "auto", padding: "6px 16px" }}
            onClick={handleSubmit}
            disabled={submitting}
          >
            {submitting ? "Probing…" : "Add"}
          </button>
        </div>
        {error && (
          <div className="local-engine-hint" style={{ color: "var(--error)" }}>
            {error}
          </div>
        )}
      </div>
    </div>
  );
}

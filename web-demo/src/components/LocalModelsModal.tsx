/** Local Models modal — browser variant.
 *
 * Ported from `tauri-gui/src/components/LocalModelsModal.tsx`. Probes every
 * registered OpenAI-compatible local engine (Ollama, LM Studio, llama-server,
 * vLLM, Jan, LiteRT-LM) plus any user-added custom endpoints, and exposes
 * a per-model "Use for Estimate" picker that drives `runEstimate`'s baseUrl.
 *
 * Cuts from the Tauri version that don't make sense in a browser:
 *   - OllamaSpawnBar (no process spawning)
 *   - OllamaCuratedInstall (no `ollama pull` orchestration / progress stream)
 *
 * The LiteRT setup instructions from the prior browser-only modal are
 * preserved as a collapsible <details> block below the engine table so the
 * "how do I run the sidecar" docs stay reachable.
 */

import { useState, type ReactNode } from "react";
import { useAppStore } from "../stores/appStore";
import { useSidecarStatus, type SidecarStatus } from "../lib/sidecarProbe";
import type {
  CustomEndpoint,
  LocalEngineStatus,
  LocalModel,
} from "../lib/types";

const LANGSLICE_GEMMA_HF =
  "https://huggingface.co/greenpolo/langslice-gemma-4-E4B#known-limitations";

const LITERT_RELEASE_URL =
  "https://github.com/google-ai-edge/LiteRT-LM/releases";
const GEMMA_WEIGHTS_URL =
  "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm";
const RUN_CMD =
  "litert-lm serve --port 8765 --backend gpu --model /path/to/gemma-4-E4B-it.litertlm";
const COPY_LABEL_MS = 1500;

/** Top-bar Local Models modal.
 *
 * Layout:
 *   - Pinned LangSlice-Gemma row (HF link).
 *   - One section per auto-probed engine (Ollama, LM Studio, llama-server,
 *     vLLM, Jan, LiteRT-LM), then any user-added custom endpoints.
 *   - "Add custom endpoint" inline form.
 *   - LiteRT-LM sidecar setup instructions in a <details> footer.
 *   - Refresh + Done in the footer.
 *
 * Engines that are reachable expand to show their model list with a
 * "Use for Estimate" button per row. */
export function LocalModelsModal(): ReactNode {
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

  // Keep the live sidecar pill visible — gives at-a-glance "is the default
  // litert-lm endpoint up" feedback even when the user is browsing engines.
  const sidecar = useSidecarStatus({ enabled: open });

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
          <SidecarStatusPill status={sidecar} />

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

          <LiteRTSetupFooter />
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
          <button
            type="button"
            className="btn-primary"
            style={{ width: "auto", padding: "8px 24px" }}
            onClick={close}
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}

/** Pinned LangSlice-Gemma row at the top of the list, linking to the
 *  Hugging Face model card. */
function LangSliceGemmaRow(): ReactNode {
  return (
    <div className="atlas-row atlas-row-focus">
      <div className="atlas-row-main">
        <span className="atlas-row-name">LangSlice-Gemma E4B</span>
        <span className="atlas-row-status">
          Custom fine-tune of Gemma 4 E4B for AP estimation
        </span>
      </div>
      <div className="atlas-row-action">
        <a
          className="btn-secondary atlas-row-download"
          href={LANGSLICE_GEMMA_HF}
          target="_blank"
          rel="noopener noreferrer"
          title="Open the LangSlice-Gemma model card on Hugging Face"
          style={{
            textDecoration: "none",
            display: "inline-block",
            textAlign: "center",
          }}
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

/** One engine row + collapsible model list. No Ollama-specific affordances
 *  in the browser — daemon lifecycle isn't ours to manage from a tab. */
function EngineSection({
  engine,
  selectedModelId,
  selectedEndpoint,
  onUseModel,
  onRemoveCustom,
}: EngineSectionProps): ReactNode {
  const [expanded, setExpanded] = useState(
    engine.reachable && engine.models.length > 0,
  );

  // Drop the parenthetical when the error is the generic "not running"
  // sentinel — otherwise we render "Not running (not running)".
  const errorSuffix =
    engine.error && engine.error !== "not running" ? ` (${engine.error})` : "";
  const statusText = engine.reachable
    ? `Detected · ${engine.models.length} model${engine.models.length === 1 ? "" : "s"}`
    : `Not running${errorSuffix}`;

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
        </div>
      )}
    </div>
  );
}

interface CustomEndpointFormProps {
  existingUrls: string[];
  onAdd: (ep: CustomEndpoint) => Promise<void>;
}

/** Inline label / url / apiKey form for adding a non-default-port or remote
 *  OpenAI-compatible server. URL is normalized (trailing slash stripped)
 *  before the duplicate check so users adding the same URL with/without
 *  a slash get a clean rejection. */
function CustomEndpointForm({
  existingUrls,
  onAdd,
}: CustomEndpointFormProps): ReactNode {
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

/** LiteRT-LM sidecar setup instructions, kept around as a collapsible
 *  footer so judges/users can still find the run command. Mirrors the
 *  copy + steps from the previous browser-only modal. */
function LiteRTSetupFooter(): ReactNode {
  const [copied, setCopied] = useState(false);
  const copyCmd = async () => {
    try {
      await navigator.clipboard.writeText(RUN_CMD);
      setCopied(true);
      setTimeout(() => setCopied(false), COPY_LABEL_MS);
    } catch {
      // Clipboard API can be gated by permissions; silent fallback is fine —
      // the command is right above the button for hand-copy.
    }
  };

  return (
    <details
      className="local-engine-quickinstall"
      style={{ marginTop: 12, padding: "8px 12px" }}
    >
      <summary>Bring your own LiteRT sidecar (setup steps)</summary>
      <div style={{ marginTop: 8 }}>
        <p style={{ marginTop: 0 }}>
          The position-estimation agent ships pointed at a local{" "}
          <code>litert-lm serve</code> sidecar on{" "}
          <code>http://127.0.0.1:8765</code>. Pick a different engine above to
          override; otherwise follow these steps to bring up the default
          target.
        </p>
        <ol style={{ paddingLeft: 20, lineHeight: 1.6 }}>
          <li>
            Download <code>litert-lm</code>:{" "}
            <a
              href={LITERT_RELEASE_URL}
              target="_blank"
              rel="noreferrer noopener"
            >
              Google&rsquo;s official release ↑
            </a>
          </li>
          <li>
            Download the Gemma 4 E4B weights:{" "}
            <a
              href={GEMMA_WEIGHTS_URL}
              target="_blank"
              rel="noreferrer noopener"
            >
              gemma-4-E4B-it.litertlm (3.66 GB) ↑
            </a>
          </li>
          <li>
            Run it locally:
            <pre
              style={{
                background: "#111",
                padding: 10,
                borderRadius: 4,
                overflowX: "auto",
                marginTop: 8,
                marginBottom: 8,
              }}
            >
              <code>{RUN_CMD}</code>
            </pre>
            <button
              type="button"
              className="btn-secondary"
              style={{ width: "auto", padding: "6px 14px" }}
              onClick={copyCmd}
            >
              {copied ? "Copied!" : "Copy command"}
            </button>
          </li>
        </ol>
        <p
          style={{
            fontStyle: "italic",
            color: "#888",
            marginBottom: 0,
          }}
        >
          First request to <code>127.0.0.1</code> will trigger a Chrome 142+
          Local Network Access permission prompt &mdash; grant it.
        </p>
      </div>
    </details>
  );
}

/** Live status pill for the bundled-default litert-lm sidecar. Kept from
 *  the prior browser-only modal so users see at-a-glance whether the
 *  default Estimate target is up; doesn't replace the engine table. */
function SidecarStatusPill({ status }: { status: SidecarStatus }): ReactNode {
  let label: string;
  let bg: string;
  let color: string;
  switch (status.state) {
    case "unknown":
      label = "Probing LiteRT sidecar…";
      bg = "#222";
      color = "#ccc";
      break;
    case "ok": {
      const n = status.models.length;
      label = `LiteRT sidecar live (${n} model${n === 1 ? "" : "s"})`;
      bg = "#1f6b1f";
      color = "#dffadf";
      break;
    }
    case "not-running":
      label =
        "LiteRT sidecar not running — pick another engine above, or expand the setup footer to start it.";
      bg = "#3a2a10";
      color = "#ffd28a";
      break;
    case "cors-blocked":
      label = `LiteRT sidecar reachable but blocking the page. ${status.hint}`;
      bg = "#3a2a10";
      color = "#ffd28a";
      break;
  }
  return (
    <div
      className={`setup-status setup-status-${status.state}`}
      style={{
        marginBottom: 12,
        padding: 10,
        borderRadius: 6,
        background: bg,
        color,
      }}
    >
      {label}
    </div>
  );
}

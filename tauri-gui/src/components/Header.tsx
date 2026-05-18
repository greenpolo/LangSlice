import { useAppStore } from "../stores/appStore";
import { ApiKeysModal } from "./ApiKeysModal";
import { LocalModelsModal } from "./LocalModelsModal";
import logoSvg from "../assets/logo.svg";

export function Header() {
  const currentView = useAppStore((s) => s.currentView);
  const selectedBrainId = useAppStore((s) => s.selectedBrainId);
  const brains = useAppStore((s) => s.brains);
  const navigateToDashboard = useAppStore((s) => s.navigateToDashboard);
  const openApiKeys = useAppStore((s) => s.openApiKeys);
  const openLocalModels = useAppStore((s) => s.openLocalModels);
  const openChat = useAppStore((s) => s.openChat);
  const chatOpen = useAppStore((s) => s.chatOpen);
  const chatStreaming = useAppStore((s) => s.chatStreaming);

  const selectedBrain = brains.find((b) => b.id === selectedBrainId);

  return (
    <>
      <header className="app-header">
        <div className="header-brand">
          <img className="header-logo" src={logoSvg} alt="LangSlice" />
        </div>

        <div className="header-divider" />

        <button
          type="button"
          className="header-tab-btn"
          onClick={openApiKeys}
          title="Configure cloud API keys (Google, Vertex, OpenRouter)"
        >
          API Keys
        </button>
        <button
          type="button"
          className="header-tab-btn"
          onClick={openLocalModels}
          title="Browse and use local models from Ollama, LM Studio, llama-server, vLLM, Jan, or a custom endpoint"
        >
          Local Models
        </button>
        <button
          type="button"
          className={`header-tab-btn header-chat-btn ${chatOpen ? "active" : ""}`}
          onClick={openChat}
          title="Chat with any reachable local model"
        >
          <span className="header-chat-glyph" aria-hidden>
            &gt;_
          </span>
          Chat
          {chatStreaming && <span className="header-chat-pulse" aria-hidden />}
        </button>

        <div className="header-divider" />

        {currentView === "dashboard" ? (
          <span className="header-tag">atlas registration</span>
        ) : (
          <>
            <button className="header-back" onClick={navigateToDashboard}>
              &#8592; All Brains
            </button>
            {selectedBrain && (
              <>
                <div className="header-divider" />
                <span className="header-brain-name">{selectedBrain.name}</span>
              </>
            )}
          </>
        )}

        <div className="header-spacer" />

        <span className="header-tag">v0.1.0</span>
      </header>

      <ApiKeysModal />
      <LocalModelsModal />
    </>
  );
}

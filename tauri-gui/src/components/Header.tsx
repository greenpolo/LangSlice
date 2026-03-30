import { useAppStore } from "../stores/appStore";

export function Header() {
  const currentView = useAppStore((s) => s.currentView);
  const selectedBrainId = useAppStore((s) => s.selectedBrainId);
  const brains = useAppStore((s) => s.brains);
  const navigateToDashboard = useAppStore((s) => s.navigateToDashboard);

  const selectedBrain = brains.find((b) => b.id === selectedBrainId);

  return (
    <header className="app-header">
      <div className="header-brand">
        <div className="header-logo">LS</div>
        <span className="header-title">LangSlice</span>
      </div>

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
  );
}

import { useState } from "react";
import { useAppStore } from "../stores/appStore";
import { SettingsDialog } from "./SettingsDialog";
import logoSvg from "../assets/logo.svg";

export function Header() {
  const currentView = useAppStore((s) => s.currentView);
  const selectedBrainId = useAppStore((s) => s.selectedBrainId);
  const brains = useAppStore((s) => s.brains);
  const navigateToDashboard = useAppStore((s) => s.navigateToDashboard);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const selectedBrain = brains.find((b) => b.id === selectedBrainId);

  return (
    <>
      <header className="app-header">
        <div className="header-brand">
          <img className="header-logo" src={logoSvg} alt="LangSlice" />
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

        <button
          className="header-settings-btn"
          onClick={() => setSettingsOpen(true)}
          title="Settings"
        >
          &#9881;
        </button>

        <span className="header-tag">v0.1.0</span>
      </header>

      <SettingsDialog open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </>
  );
}

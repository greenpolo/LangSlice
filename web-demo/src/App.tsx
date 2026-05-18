/** Root SPA shell — always renders the dashboard. Sidecar status is surfaced
 *  by the Dashboard's note + the Local Models modal, never as a curtain. */

import { useEffect } from "react";
import { Header } from "./components/Header";
import { Dashboard } from "./components/Dashboard";
import { SettingsPanel } from "./components/SettingsPanel";
import { AgentPanel } from "./components/AgentPanel";
import { ViewToolbar } from "./components/ViewToolbar";
import { Scene3D } from "./components/Scene3D";
import { SplitView } from "./components/SplitView";
import { AtlasLoadingOverlay } from "./components/AtlasLoadingOverlay";
import { useAppStore } from "./stores/appStore";

function MainView() {
  const viewMode = useAppStore((s) => s.viewMode);

  switch (viewMode) {
    case "3d":
      return <Scene3D />;
    case "split":
      return <SplitView />;
  }
}

function BrainDetailView() {
  return (
    <div className="app-body">
      <SettingsPanel />
      <div className="main-area">
        <ViewToolbar />
        <MainView />
      </div>
      <AgentPanel />
    </div>
  );
}

function App() {
  const currentView = useAppStore((s) => s.currentView);
  const atlasInfo = useAppStore((s) => s.atlasInfo);
  const atlasLoading = useAppStore((s) => s.atlasLoading);
  const loadAtlas = useAppStore((s) => s.loadAtlas);
  const atlasName = useAppStore((s) => s.atlasName);

  // Auto-load the bundled atlas once on mount.
  useEffect(() => {
    if (atlasInfo === null && !atlasLoading) {
      void loadAtlas(atlasName);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="app-root">
      <Header />
      {currentView === "dashboard" ? <Dashboard /> : <BrainDetailView />}
      <AtlasLoadingOverlay />
    </div>
  );
}

export default App;

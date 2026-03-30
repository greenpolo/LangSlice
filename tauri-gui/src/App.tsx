import { Header } from "./components/Header";
import { Dashboard } from "./components/Dashboard";
import { SettingsPanel } from "./components/SettingsPanel";
import { AgentPanel } from "./components/AgentPanel";
import { ViewToolbar } from "./components/ViewToolbar";
import { Scene3D } from "./components/Scene3D";
import { SplitView } from "./components/SplitView";
import { OverlayView } from "./components/OverlayView";
import { useAppStore } from "./stores/appStore";

function MainView() {
  const viewMode = useAppStore((s) => s.viewMode);

  switch (viewMode) {
    case "3d":
      return <Scene3D />;
    case "split":
      return <SplitView />;
    case "overlay":
      return <OverlayView />;
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

  return (
    <div className="app-root">
      <Header />
      {currentView === "dashboard" ? (
        <Dashboard />
      ) : (
        <BrainDetailView />
      )}
    </div>
  );
}

export default App;

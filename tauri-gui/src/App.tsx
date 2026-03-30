import { Header } from "./components/Header";
import { Dashboard } from "./components/Dashboard";
import { SettingsPanel } from "./components/SettingsPanel";
import { AgentPanel } from "./components/AgentPanel";
import { ViewToolbar } from "./components/ViewToolbar";
import { Scene3D } from "./components/Scene3D";
import { useAppStore } from "./stores/appStore";

function BrainDetailView() {
  return (
    <div className="app-body">
      <SettingsPanel />
      <div className="main-area">
        <ViewToolbar />
        <Scene3D />
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

import { useEffect, useState } from "react";
import TopBar from "./components/TopBar";
import Sidebar from "./components/Sidebar";
import LogTerminal from "./components/LogTerminal";
import WelcomeTab from "./components/WelcomeTab";
import ConfigTab from "./components/ConfigTab";
import RunTab from "./components/RunTab";
import ToolsTab from "./components/ToolsTab";
import { api, type ProjectListing, type FileEntry } from "./api";

// Application shell: top bar + left sidebar + center tabbed workspace + bottom
// log terminal. G7 shipped Welcome; G8 adds Config. Later prompts register
// Data, Tune, Run, Canvas, and Results tabs into the same workspace.

// Shared context each tab's render receives, so a tab can read the open project,
// attach the bottom terminal to a log file it cares about, and (Config tab) edit
// the project's config file.
interface TabContext {
  listing: ProjectListing | null;
  setLogFile: (file: string | null) => void;
  configPath: string | null;
}

interface Tab {
  id: string;
  label: string;
  render: (ctx: TabContext) => JSX.Element;
}

const TABS: Tab[] = [
  { id: "welcome", label: "Welcome", render: () => <WelcomeTab /> },
  { id: "config", label: "Config", render: (ctx) => <ConfigTab configPath={ctx.configPath} /> },
  {
    id: "run",
    label: "Run",
    render: (ctx) => <RunTab listing={ctx.listing} setLogFile={ctx.setLogFile} />,
  },
  {
    id: "tools",
    label: "Tools",
    render: (ctx) => <ToolsTab listing={ctx.listing} setLogFile={ctx.setLogFile} />,
  },
];

export default function App() {
  const [listing, setListing] = useState<ProjectListing | null>(null);
  const [projectError, setProjectError] = useState<string | null>(null);
  const [logFile, setLogFile] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<string>(TABS[0].id);

  const openProject = async (path: string) => {
    try {
      setProjectError(null);
      const result = await api.openProject(path);
      setListing(result);
    } catch (e) {
      setProjectError(String(e instanceof Error ? e.message : e));
      setListing(null);
    }
  };

  // On load, auto-open the project the server was launched with, if any.
  useEffect(() => {
    api.config().then((cfg) => {
      if (cfg.initial_project) openProject(cfg.initial_project);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onSelectFile = (entry: FileEntry) => setLogFile(entry.path);

  const projectName = listing ? listing.dir.split("/").pop() || listing.dir : null;
  const current = TABS.find((t) => t.id === activeTab) || TABS[0];
  // The Config tab edits the first config file the project exposes.
  const configPath = listing && listing.configs.length ? listing.configs[0].path : null;

  return (
    <div className="app">
      <TopBar projectName={projectName} />
      <div className="app-body">
        <Sidebar
          listing={listing}
          onOpen={openProject}
          onSelectFile={onSelectFile}
          error={projectError}
        />
        <main className="workspace">
          <nav className="tabbar">
            {TABS.map((t) => (
              <button
                key={t.id}
                className={`tab ${t.id === activeTab ? "active" : ""}`}
                onClick={() => setActiveTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </nav>
          <div className="tab-content">
            {current.render({ configPath, listing, setLogFile })}
          </div>
        </main>
      </div>
      <LogTerminal file={logFile} />
    </div>
  );
}
